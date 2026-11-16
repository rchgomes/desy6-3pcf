"""Binning conventions of the DES Y6 third-order shear analysis.

Everything that defines *where* the data vector lives is collected here so
that measurement, theory, emulator and likelihood agree by construction:

* the shear 3PCF is measured in TreeCorr ``LogSAS`` binning, i.e. two side
  lengths ``(d2, d3)`` in logarithmic bins and the opening angle ``phi`` in
  linear bins on ``[0, pi]``;
* the 3PCF is compressed into aperture-mass skewness
  ``<Map^3>(theta1, theta2, theta3)`` at a set of filter radii;
* for emulation, ``<Map^3>`` is tabulated on thin redshift slices before the
  line-of-sight integration.
"""
from itertools import combinations_with_replacement

import numpy as np

# --- 3PCF measurement binning (arcmin) -------------------------------------
MIN_SEP = 1.0
MAX_SEP = 150.0
NBINS = 25
NPHI_BINS = 100

# --- aperture-mass filter radii (arcmin) -----------------------------------
FILTERS = (3.0, 5.0, 10.0, 20.0, 40.0)

# --- redshift slices on which the emulators are trained --------------------
Z_SLICES = tuple(
    np.round(
        np.concatenate(
            [
                [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07],
                np.arange(0.1, 1.0, 0.025),
                np.arange(1.0, 2.0001, 0.05),
            ]
        ),
        4,
    ).tolist()
)

ARCMIN = np.pi / (180.0 * 60.0)

# --- TATT bispectrum components: d = matter density, E = E-mode shape, one
#     letter per vertex of the triangle ---------------------------------------
TATT_COMPONENTS = ("ddE", "dEd", "Edd", "dEE", "EdE", "EEd", "EEE")


def log_bin_width(min_sep=MIN_SEP, max_sep=MAX_SEP, nbins=NBINS):
    """Width in ``ln(theta)`` of the logarithmic side-length bins."""
    return np.log(max_sep / min_sep) / nbins


def theta_lower_edges(min_sep=MIN_SEP, max_sep=MAX_SEP, nbins=NBINS):
    """Lower edges of the side-length bins, in the units of ``min_sep``."""
    return min_sep * np.exp(log_bin_width(min_sep, max_sep, nbins) * np.arange(nbins))


def mean_separation_factor(dlnt, offset=0.0):
    """Ratio of the mean separation in a logarithmic bin to its lower edge.

    For pairs distributed uniformly on the sky the mean separation in the
    bin ``[t, t e^dlnt]`` is ``t * 2/3 (e^{3 dlnt} - 1) / (e^{2 dlnt} - 1)``.
    ``offset`` is an additive correction to that factor, for calibrating
    against the ``meand2``/``meand3`` reported by TreeCorr on real catalogues.
    """
    return 2.0 / 3.0 * np.expm1(3.0 * dlnt) / np.expm1(2.0 * dlnt) + offset


def phi_centres(nphi=NPHI_BINS):
    """Centres of ``nphi`` linear opening-angle bins on ``[0, pi]``."""
    return (np.arange(nphi) + 0.5) * np.pi / nphi


def filter_triplets(filters=FILTERS, mode="equal"):
    """Aperture-radius triplets as an array of shape ``(3, n)``.

    ``mode='equal'`` gives ``(R, R, R)`` for every radius; ``mode='all'``
    gives every unordered triplet ``R1 <= R2 <= R3``.
    """
    filters = np.sort(np.asarray(filters, dtype=float))
    if mode == "equal":
        return np.array([filters, filters, filters])
    if mode == "all":
        return np.array(list(combinations_with_replacement(filters, 3))).T
    raise ValueError(f"filter mode must be 'equal' or 'all', got {mode!r}")


def tomographic_triplets(nbin, which="all"):
    """Unordered redshift-bin triplets ``i <= j <= k`` (1-based).

    ``which`` is ``'all'``, ``'auto'`` (``i == j == k``) or ``'cross'``.
    """
    triplets = list(combinations_with_replacement(range(1, nbin + 1), 3))
    if which == "auto":
        triplets = [t for t in triplets if t[0] == t[2]]
    elif which == "cross":
        triplets = [t for t in triplets if t[0] != t[2]]
    elif which != "all":
        raise ValueError(f"which must be 'all', 'auto' or 'cross', got {which!r}")
    return np.array(triplets, dtype=int)


def parse_tomographic_triplets(spec, nbin):
    """Parse a selection such as ``'all'``, ``'auto'`` or ``'1,1,2 2,3,4'``."""
    triplets = []
    for item in (spec.split() or ["all"]):
        if "," in item:
            triplets.append(sorted(int(i) for i in item.split(",")))
        else:
            triplets.extend(tomographic_triplets(nbin, item).tolist())
    return np.unique(np.array(triplets, dtype=int), axis=0)


def combination_name(scomb):
    """Label of a tomographic triplet (``'1_2_3'``) or a redshift slice."""
    if np.isscalar(scomb):
        return str(scomb)
    return "_".join(str(s) for s in scomb)
