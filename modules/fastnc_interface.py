"""Shear 3PCF natural components from fastnc, on the measurement bins.

Reads the cosmology, the linear (and for TATT the non-linear) matter power
spectrum, the growth factor and the source redshift distributions, and
writes ``Gamma_0..Gamma_3`` for every sample combination requested by
``map3_binning`` to the ``natural_components`` section.

The signal is the sum of a matter-bispectrum part and, for
``ia_model = tatt``, the TATT intrinsic-alignment bispectra:

=========  ===========================================================
ia_model   what is computed
=========  ===========================================================
none       matter bispectrum with lensing kernels
nla        matter bispectrum with lensing + NLA kernels
tatt       as ``nla`` plus the TATT components with their tree-level
           NLA-like term removed (``remove_alignment = T``), or
           lensing-only matter part plus the full TATT components
           (``remove_alignment = F``)
=========  ===========================================================

Setting ``matter_model = none`` and/or ``tatt_component`` isolates single
terms, which is how the emulator training sets are built.

Options (defaults are the DES Y6 measurement settings)
-----------------------------------------------------
min_sep, max_sep : float, arcmin (1, 150)
nbins : int (25)
nphi_bins : int (100)
    Linear opening-angle bins on [0, pi].
mean_separation_offset : float (0)
    Additive correction to the analytic mean-separation factor of the
    side-length bins (see ``desy6_3pcf.binning.mean_separation_factor``).
matter_model : ``bihalofit`` (default), ``gilmarin`` or ``none``
ia_model : ``none``, ``nla`` (default) or ``tatt``
tatt_component : one of ddE dEd Edd dEE EdE EEd EEE (default: their sum)
remove_alignment : bool (T)
ct : float (0)
baryon_model : ``none`` (default), ``takahashi`` or ``baccoemu``
    Applied to the matter bispectrum; parameters are read from the
    ``baryon_parameters`` section (``fb``, or ``bacco_M_c`` etc.).
use_pixwin : bool (F), nside : int
Lmax (50), Mmax (100), nfft (150), nrbin (35), nrbin_tatt (70),
projection (cent), multipole_type (legendre), use_cache (T)
"""
import numpy as np
from cosmosis.datablock import option_section

from desy6_3pcf import binning
from desy6_3pcf.block import NC, read_theory_inputs, sample_combinations
from desy6_3pcf.fastnc_backend import NaturalComponents


def setup(options):
    def get(kind, name, default):
        return getattr(options, f"get_{kind}")(option_section, name, default)

    min_sep = get("double", "min_sep", binning.MIN_SEP)
    max_sep = get("double", "max_sep", binning.MAX_SEP)
    nbins = get("int", "nbins", binning.NBINS)
    nphi = get("int", "nphi_bins", binning.NPHI_BINS)
    dlnt = binning.log_bin_width(min_sep, max_sep, nbins)
    theta_lower = binning.theta_lower_edges(min_sep, max_sep, nbins)

    matter_model = get("string", "matter_model", "bihalofit")
    ia_model = get("string", "ia_model", "nla")
    if ia_model not in ("none", "nla", "tatt"):
        raise ValueError("ia_model must be none, nla or tatt")
    remove_alignment = get("bool", "remove_alignment", True)
    if matter_model == "none" and ia_model != "tatt":
        raise ValueError("nothing to compute: matter_model = none needs ia_model = tatt")

    common = dict(
        lmax=get("int", "Lmax", 50),
        mmax=get("int", "Mmax", 100),
        nfft=get("int", "nfft", 150),
        projection=get("string", "projection", "cent"),
        multipole_type=get("string", "multipole_type", "legendre"),
        use_cache=get("bool", "use_cache", True),
        pixwin_nside=get("int", "nside", 0) if get("bool", "use_pixwin", False) else None,
    )
    bins = (theta_lower * binning.ARCMIN, dlnt, binning.phi_centres(nphi))

    calculators = []
    if matter_model != "none":
        calculators.append(
            NaturalComponents(
                *bins,
                bispectrum_model=matter_model,
                nla=ia_model == "nla" or (ia_model == "tatt" and remove_alignment),
                nrbin=get("int", "nrbin", 35),
                baryon_model=get("string", "baryon_model", "none"),
                **common,
            )
        )
    if ia_model == "tatt":
        calculators.append(
            NaturalComponents(
                *bins,
                bispectrum_model="tatt",
                tatt_component=get("string", "tatt_component", "") or None,
                remove_alignment=remove_alignment,
                ct=get("double", "ct", 0.0),
                nrbin=get("int", "nrbin_tatt", 70),
                **common,
            )
        )

    factor = binning.mean_separation_factor(dlnt, get("double", "mean_separation_offset", 0.0))
    return {
        "calculators": calculators,
        "theta": theta_lower * factor,  # arcmin, representative value of each bin
        "phi": bins[2],
        "dlnt": dlnt,
        "dphi": np.pi / nphi,
    }


def execute(block, config):
    scombs = sample_combinations(block, NC)
    slices = np.isscalar(scombs[0])

    total = None
    for calculator in config["calculators"]:
        calculator.update(read_theory_inputs(block, need_nz=not slices, need_nonlinear=calculator.is_tatt))
        gammas = list(calculator.compute(scombs).values())
        total = gammas if total is None else [a + b for a, b in zip(total, gammas)]

    for scomb, gamma in zip(scombs, total):
        name = binning.combination_name(scomb)
        block[NC, f"real-bin_{name}"] = gamma.real
        block[NC, f"imag-bin_{name}"] = gamma.imag

    block[NC, "theta"] = config["theta"]
    block[NC, "phi"] = config["phi"]
    block[NC, "dlnt"] = config["dlnt"]
    block[NC, "dphi"] = config["dphi"]
    return 0


def cleanup(config):
    pass
