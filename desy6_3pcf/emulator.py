"""Neural-network emulators of ``<Map^3>`` on thin redshift slices.

Each emulator maps a handful of cosmological (and, for the TATT terms,
intrinsic-alignment) parameters to ``<Map^3>`` at every filter and redshift
slice, *before* the line-of-sight integration.  The redshift distributions,
their uncertainties and the lensing/NLA kernels are applied analytically
afterwards (:func:`integrate_slices`), so the emulators are independent of
the source sample.

An emulator on disk is a pair of files sharing a prefix:

* ``<prefix>.pkl``  -- the CosmoPower network;
* ``<prefix>.json`` -- parameter names and ranges, redshift slices, filters,
  the feature transform, and the settings used to make the training set.
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

TRANSFORMS = ("log10", "asinh")
_trapezoid = getattr(np, "trapezoid", None) or np.trapz



def import_cosmopower():
    """Import ``cosmopower_NN`` with the workarounds cosmopower 0.1.x needs
    on current numpy (no ``np.infty``) and TensorFlow (Keras 2 API)."""
    import os

    if not hasattr(np, "infty"):
        np.infty = np.inf
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    from cosmopower import cosmopower_NN

    return cosmopower_NN


class FeatureScaler:
    """Invertible map from ``<Map^3>`` slices to network targets in ``[0, 1]``.

    ``log10`` suits strictly positive features (the matter-bispectrum
    emulators).  ``asinh`` (of the feature in units of a per-feature scale)
    behaves like a signed logarithm and is used for the TATT components,
    which change sign across parameter space.
    """

    def __init__(self, transform, minimum, maximum, scale=None):
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
        self.transform = transform
        self.minimum = np.asarray(minimum, dtype=float)
        self.maximum = np.asarray(maximum, dtype=float)
        self.scale = None if scale is None else np.asarray(scale, dtype=float)

    @classmethod
    def fit(cls, features, transform):
        features = np.asarray(features, dtype=float)
        scale = None
        if transform == "log10":
            if np.any(features <= 0):
                raise ValueError("the log10 transform needs strictly positive features; use 'asinh'")
        elif transform == "asinh":
            scale = np.median(np.abs(features), axis=0)
            scale[scale == 0] = 1.0
        raw = cls(transform, 0.0, 1.0, scale)._forward(features)
        lo, hi = raw.min(axis=0), raw.max(axis=0)
        hi = np.where(hi > lo, hi, lo + 1.0)
        return cls(transform, lo, hi, scale)

    def _forward(self, features):
        if self.transform == "log10":
            return np.log10(features)
        return np.arcsinh(features / self.scale)

    def encode(self, features):
        return (self._forward(np.asarray(features, dtype=float)) - self.minimum) / (self.maximum - self.minimum)

    def decode(self, targets):
        raw = np.asarray(targets) * (self.maximum - self.minimum) + self.minimum
        if self.transform == "log10":
            return 10.0**raw
        return np.sinh(raw) * self.scale

    def to_dict(self):
        return {
            "transform": self.transform,
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["transform"], d["minimum"], d["maximum"], d.get("scale"))


class Map3SliceEmulator:
    """A trained network together with everything needed to use it.

    Parameters
    ----------
    network : object
        Anything with ``predictions_np(dict) -> array (nsample, nfeature)``,
        in practice a ``cosmopower_NN``.
    parameters : list of str
        Inputs in CosmoSIS notation, e.g. ``cosmological_parameters--omega_m``.
    param_min, param_max : array
        Extent of the training set, used to normalise the inputs.
    z_slices : array
    filters : array, shape (3, nfilter)
        Aperture radii in arcmin.  Features are ordered slice-major:
        ``feature[iz * nfilter + ifilter]``.
    scaler : FeatureScaler
    info : dict
        Free-form provenance (binning, fastnc settings, component, ...).
    """

    def __init__(self, network, parameters, param_min, param_max, z_slices, filters, scaler, info=None):
        self.network = network
        self.parameters = [p.lower() for p in parameters]
        self.param_min = np.asarray(param_min, dtype=float)
        self.param_max = np.asarray(param_max, dtype=float)
        self.z_slices = np.asarray(z_slices, dtype=float)
        self.filters = np.atleast_2d(np.asarray(filters, dtype=float))
        self.scaler = scaler
        self.info = dict(info or {})

    @property
    def nfeature(self):
        return self.z_slices.size * self.filters.shape[1]

    def metadata(self):
        return {
            "parameters": self.parameters,
            "param_min": self.param_min.tolist(),
            "param_max": self.param_max.tolist(),
            "z_slices": self.z_slices.tolist(),
            "filters": self.filters.tolist(),
            "scaler": self.scaler.to_dict(),
            "info": self.info,
        }

    def save_metadata(self, prefix):
        Path(f"{prefix}.json").write_text(json.dumps(self.metadata(), indent=1))

    @classmethod
    def load(cls, prefix):
        """Load ``<prefix>.json`` and the CosmoPower network ``<prefix>.pkl``."""
        prefix = str(prefix)
        for suffix in (".json", ".pkl"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
        meta = json.loads(Path(f"{prefix}.json").read_text())
        network = import_cosmopower()(restore=True, restore_filename=prefix)
        return cls(
            network,
            meta["parameters"],
            meta["param_min"],
            meta["param_max"],
            meta["z_slices"],
            meta["filters"],
            FeatureScaler.from_dict(meta["scaler"]),
            meta.get("info"),
        )

    def normalise(self, params):
        """Scale parameters, shape ``(nsample, nparam)``, to the unit cube."""
        return (np.atleast_2d(params) - self.param_min) / (self.param_max - self.param_min)

    def in_range(self, params):
        u = self.normalise(params)
        return np.all((u >= 0.0) & (u <= 1.0), axis=1)

    def predict_features(self, params):
        """Features for parameters of shape ``(nsample, nparam)``."""
        u = self.normalise(params)
        inputs = {name: u[:, i] for i, name in enumerate(self.parameters)}
        return self.scaler.decode(self.network.predictions_np(inputs))

    def predict(self, params):
        """``<Map^3>`` slices of shape ``(nz, nfilter)`` for one parameter set."""
        return self.predict_features(params)[0].reshape(self.z_slices.size, self.filters.shape[1])

    def filter_indices(self, filters):
        """Columns of :meth:`predict` matching aperture triplets ``(3, n)``."""
        filters = np.atleast_2d(filters)
        out = []
        for f in filters.T:
            hit = np.flatnonzero(np.all(np.isclose(self.filters, f[:, None]), axis=0))
            if hit.size != 1:
                raise KeyError(f"filters {tuple(f)} arcmin were not emulated")
            out.append(hit[0])
        return np.array(out)


def upsample_redshifts(z, n):
    """``z`` merged with ``n`` equally spaced points over its range."""
    if n <= z.size:
        return z
    return np.union1d(z, np.linspace(z[0], z[-1], n))


def integrate_slices(z, slices, chi_of_z, weight_of_z, n_upsample=0):
    """Line-of-sight integral ``int dchi weight(z) slices(z)``.

    The slices, of shape ``(nz, ...)``, vary smoothly with redshift and are
    spline-interpolated onto a finer grid on which the kernels (which can
    be much narrower) are evaluated exactly.
    """
    z = np.asarray(z, dtype=float)
    zf = upsample_redshifts(z, n_upsample)
    if zf.size != z.size:
        slices = CubicSpline(z, slices, axis=0)(zf)
    weight = np.asarray(weight_of_z(zf))
    weight = weight.reshape(weight.shape + (1,) * (np.ndim(slices) - 1))
    return _trapezoid(slices * weight, chi_of_z(zf), axis=0)
