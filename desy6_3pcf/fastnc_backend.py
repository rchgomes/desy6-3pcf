"""The only module of this package that talks to ``fastnc``.

The analysis currently runs on the legacy ``tatt`` branch of fastnc
(v1.1.x), which provides the TATT intrinsic-alignment bispectra and the
baryonic-feedback models.  Everything fastnc-specific (class names, call
order, conventions, workarounds) is confined to this file; porting to the
fastnc v2 API means re-implementing the two public classes below,

* :class:`NaturalComponents` -- shear 3PCF natural components on the
  measurement bins, for tomographic triplets or thin redshift slices;
* :class:`LensingKernels` -- line-of-sight kernels used to integrate the
  emulated redshift slices,

without touching the CosmoSIS modules or the emulator code.
"""
from dataclasses import dataclass, field

import numpy as np

# fastnc 1.x calls ``np.trapz``, which numpy 2 renamed.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import fastnc  # noqa: E402
from astropy.cosmology import wCDM  # noqa: E402
from fastnc import baryon as _baryon  # noqa: E402

from .binning import TATT_COMPONENTS  # noqa: E402

BISPECTRUM_MODELS = ("bihalofit", "gilmarin", "tatt")
BARYON_MODELS = ("none", "takahashi", "baccoemu")
BACCO_PARAMETERS = ("M_c", "eta", "beta", "M1_z0_cen", "theta_inn")

# Line-of-sight sampling of the bispectrum projection (DES Y3 settings).
_LOS_CONFIG = {"zmin": 1e-4, "zmid": 0.1, "nzbin_log": 70, "nzbin_lin": 100}


@dataclass
class TheoryInputs:
    """Everything fastnc needs from the upstream pipeline for one cosmology."""

    h0: float
    omega_m: float
    omega_b: float
    sigma_8: float
    n_s: float
    w: float = -1.0
    k_lin: np.ndarray = None  # h/Mpc
    p_lin: np.ndarray = None  # linear P(k, z=0), (Mpc/h)^3
    k_nl: np.ndarray = None
    p_nl: np.ndarray = None  # non-linear P(k, z=0); TATT only
    z_growth: np.ndarray = None
    d_growth: np.ndarray = None  # linear growth factor D(z)
    nz_z: np.ndarray = None
    nz: list = None  # one n(z) per tomographic bin; None for redshift slices
    ia: dict = field(default_factory=dict)  # a1, alpha1, a2, alpha2, bias_ta, z_piv
    baryon: dict = field(default_factory=dict)  # fb, or BACCO parameters

    def astropy_cosmology(self):
        return wCDM(
            H0=100.0 * self.h0,
            Om0=self.omega_m,
            Ode0=1.0 - self.omega_m,
            Ob0=self.omega_b,
            w0=self.w,
            meta={"sigma8": self.sigma_8, "n": self.n_s},
        )


def _healpix_window(nside):
    """Product of HEALPix pixel window functions for the three multipoles."""
    import healpy as hp
    from scipy.interpolate import interp1d

    w = hp.sphtfunc.pixwin(nside)
    fnc = interp1d(np.arange(w.size), w, kind="linear", bounds_error=False, fill_value=(1, 0))
    return lambda l1, l2, l3: fnc(l1) * fnc(l2) * fnc(l3)


def _set_nla_flag(bispectrum, flag):
    # fastnc keeps ``config_IA`` as a class attribute shared by *all*
    # bispectrum instances; give each instance its own copy so that two
    # instances with different IA settings cannot overwrite each other.
    bispectrum.config_IA = {"NLA": bool(flag)}


def _set_sources(bispectrum, inputs):
    if inputs.nz is None:
        # Redshift slices do not use the source distribution, but fastnc
        # still wants one to build its kernels.
        z = np.linspace(0.01, 2.0, 50)
        bispectrum.set_source_distribution([z], [np.exp(-0.5 * ((z - 1.0) / 0.3) ** 2)], [1])
    else:
        nbin = len(inputs.nz)
        bispectrum.set_source_distribution([inputs.nz_z] * nbin, list(inputs.nz), list(range(1, nbin + 1)))


def _set_nla(bispectrum, ia):
    bispectrum.set_NLA_param(
        {"AIA": ia.get("a1", 0.0), "alphaIA": ia.get("alpha1", 0.0), "z0": ia.get("z_piv", 0.62)}
    )


def _fastnc_key(scomb):
    """Sample-combination key in the form fastnc expects."""
    if np.isscalar(scomb):
        return float(scomb)
    return tuple(int(s) for s in scomb)


class NaturalComponents:
    """Shear 3PCF natural components from fastnc.

    Parameters
    ----------
    theta_lower : array
        Lower edges of the logarithmic side-length bins, in radians.
    dlnt : float
        Logarithmic bin width.
    phi : array
        Opening-angle bin centres in radians.
    bispectrum_model : {'bihalofit', 'gilmarin', 'tatt'}
        ``'tatt'`` computes intrinsic-alignment terms only.
    nla : bool
        Add the NLA term to the lensing kernels (matter-bispectrum models,
        tomographic triplets only).
    tatt_component : str or None
        Restrict the TATT bispectrum to one of :data:`TATT_COMPONENTS`.
    remove_alignment : bool
        Subtract from each TATT component the tree-level term that the NLA
        kernel of the matter-bispectrum model already accounts for.
    pixwin_nside : int or None
        Multiply the convergence bispectrum by the HEALPix pixel window.
    """

    def __init__(
        self,
        theta_lower,
        dlnt,
        phi,
        *,
        bispectrum_model="bihalofit",
        nla=True,
        tatt_component=None,
        remove_alignment=True,
        ct=0.0,
        lmax=50,
        mmax=100,
        nfft=150,
        nrbin=35,
        projection="cent",
        multipole_type="legendre",
        mu=(0, 1, 2, 3),
        use_cache=True,
        pixwin_nside=None,
        baryon_model="none",
    ):
        if bispectrum_model not in BISPECTRUM_MODELS:
            raise ValueError(f"bispectrum_model must be one of {BISPECTRUM_MODELS}")
        if baryon_model not in BARYON_MODELS:
            raise ValueError(f"baryon_model must be one of {BARYON_MODELS}")
        if tatt_component is not None and tatt_component not in TATT_COMPONENTS:
            raise ValueError(f"tatt_component must be one of {TATT_COMPONENTS}")
        self.model = bispectrum_model
        self.is_tatt = bispectrum_model == "tatt"
        if self.is_tatt and baryon_model != "none":
            raise ValueError("baryonic feedback is only implemented for the matter bispectrum models")
        self.tatt_component = tatt_component
        self.remove_alignment = remove_alignment
        self.ct = ct
        self.baryon_model = baryon_model
        self._bacco = None

        config = dict(_LOS_CONFIG, Lmax=lmax, multipole_type=multipole_type, nrbin=nrbin)
        if self.is_tatt:
            self.bispectrum = fastnc.bispectrum.BispectrumTATT(dict(config, puv_grid=True))
        elif bispectrum_model == "gilmarin":
            self.bispectrum = fastnc.bispectrum.BispectrumGilMarin(config)
        else:
            self.bispectrum = fastnc.bispectrum.BispectrumHalofit(config)
        _set_nla_flag(self.bispectrum, nla and not self.is_tatt)
        if pixwin_nside:
            self.bispectrum.set_window_function(_healpix_window(pixwin_nside))

        self.theta_lower = np.asarray(theta_lower, dtype=float)
        self.dlnt = float(dlnt)
        self.phi = np.asarray(phi, dtype=float)
        self.calculator = fastnc.fastnc.FastNaturalComponents(
            {
                "Lmax": lmax,
                "Mmax": mmax,
                "projection": projection,
                "nfft": nfft,
                "t1": self.theta_lower,
                "t2": self.theta_lower,
                "phi": self.phi,
                "dlnt": self.dlnt,
                "mu": list(mu),
                "multipole_type": multipole_type,
                "cache": use_cache,
            }
        )

    def update(self, inputs):
        """Load cosmology, power spectra, growth, n(z), IA and baryons."""
        bs = self.bispectrum
        bs.set_cosmology(inputs.astropy_cosmology())
        _set_nla(bs, inputs.ia)
        if self.is_tatt:
            ia = inputs.ia
            bs.set_IA_param(
                {
                    "a1": ia["a1"],
                    "alphaIA": ia["alpha1"],
                    "a2": ia["a2"],
                    "alphaIA_2": ia["alpha2"],
                    "bias_ta": ia["bias_ta"],
                    "z0": ia["z_piv"],
                }
            )
            # fastnc evaluates both spectra on the k-grid of the linear one.
            p_nl = np.exp(np.interp(np.log(inputs.k_lin), np.log(inputs.k_nl), np.log(inputs.p_nl)))
            bs.set_pknl(inputs.k_lin, p_nl)
        _set_sources(bs, inputs)
        bs.set_pklin(inputs.k_lin, inputs.p_lin)
        bs.set_lgr(inputs.z_growth, inputs.d_growth)
        self._set_baryons(inputs.baryon)
        bs.compute_kernel()

    def _set_baryons(self, params):
        bs = self.bispectrum
        if self.baryon_model == "takahashi":
            bs.set_baryon_model(_baryon.BaryonTakahashiTNGfit(fb=params.get("fb", 1.0)))
        elif self.baryon_model == "baccoemu":
            bacco = {k: params[k] for k in BACCO_PARAMETERS if k in params}
            if self._bacco is None:  # loading the emulator is slow; do it once
                self._bacco = _baryon.BaryonBACCOemu(bs, **bacco)
            else:
                self._bacco.set_baryon_params(**bacco)
            bs.set_baryon_model(self._bacco)

    def compute(self, sample_combinations):
        """Natural components for tomographic triplets or redshift slices.

        Returns a dict mapping each sample combination (in the order given)
        to a complex array of shape ``(4, ntheta, ntheta, nphi)`` holding
        ``Gamma_0..Gamma_3`` in the TreeCorr convention.

        For a redshift slice ``z`` the result is the 3PCF of the bispectrum
        ``B(l1/chi, l2/chi, l3/chi; z)`` with *no* line-of-sight weight, so
        that the tomographic signal is

        ``int dchi  q_1 q_2 q_3 (1+z)^3 / chi  Gamma(z)``

        with ``q_i`` the kernels of :class:`LensingKernels`.
        """
        bs, nc = self.bispectrum, self.calculator
        keys = [_fastnc_key(s) for s in sample_combinations]
        bs.interpolate(
            scombs=keys,
            select_tatt_component=self.tatt_component,
            Ct=self.ct,
            remove_alignment=self.remove_alignment,
            puv_grid=self.is_tatt,
        )
        bs.decompose(scombs=keys, puv_grid=self.is_tatt)

        out = {}
        for key in keys:
            nc.set_bispectrum(bs)
            nc.compute(scomb=key)
            # fastnc follows the vertex labelling of Porth et al.; swapping
            # Gamma_2 and Gamma_3 and flipping the sign gives TreeCorr's.
            gamma = -np.array([nc.Gamma0, nc.Gamma1, nc.Gamma3, nc.Gamma2])
            if np.isscalar(key) and not self.is_tatt:
                # fastnc leaves the factor (1+z)^3/chi of the projection in
                # the slices of the matter-bispectrum models (but not of the
                # TATT ones); take it out so both share one convention.
                chi = float(bs.z2chi(key))
                gamma = gamma * chi / (1.0 + key) ** 3
            out[key] = gamma
        return out


class LensingKernels:
    """Line-of-sight kernels in fastnc's conventions.

    ``lensing`` is the lensing efficiency (optionally including the NLA
    term) and ``source`` the normalised source distribution in comoving
    distance; both carry the factors that make

    ``int dchi  q_1(chi) q_2(chi) q_3(chi) (1+z)^3 / chi  B(l/chi; z)``

    the projected bispectrum for any mix of the two kernel types.
    """

    def __init__(self):
        config = dict(_LOS_CONFIG, Lmax=10)
        self._bispectrum = {}
        for nla in (False, True):
            self._bispectrum[nla] = fastnc.bispectrum.BispectrumHalofit(config)
            _set_nla_flag(self._bispectrum[nla], nla)

    def update(self, inputs):
        for bs in self._bispectrum.values():
            bs.set_cosmology(inputs.astropy_cosmology())
            _set_nla(bs, inputs.ia)
            _set_sources(bs, inputs)
            bs.set_lgr(inputs.z_growth, inputs.d_growth)
            bs.compute_kernel()

    def chi(self, z):
        """Comoving distance in Mpc/h."""
        return self._bispectrum[False].z2chi(z)

    def lensing(self, sample, chi, nla=False):
        return self._bispectrum[bool(nla)].chi2g_dict[int(sample)](chi)

    def source(self, sample, chi):
        return self._bispectrum[False].chi2W_dict[int(sample)](chi)
