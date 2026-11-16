"""Helpers shared by the CosmoSIS modules (no CosmoSIS import needed)."""
import numpy as np

from .binning import combination_name

COSMO = "cosmological_parameters"
IA = "intrinsic_alignment_parameters"
BARYON = "baryon_parameters"
NC = "natural_components"
MAP3 = "map3"
DATA_VECTOR = "data_vector"
LIKELIHOODS = "likelihoods"


def get_array(options, section, name, default=None, dtype=float):
    """Read a whitespace-separated list of numbers from an ini option."""
    if not options.has_value(section, name):
        return default
    value = options[section, name]  # CosmoSIS parses numeric lists itself
    if isinstance(value, str):
        value = value.split()
    return np.atleast_1d(np.asarray(value, dtype=dtype))


def read_ia(block):
    """Intrinsic-alignment parameters present in the block, lower-cased."""
    out = {}
    for key in ("a1", "alpha1", "a2", "alpha2", "bias_ta", "z_piv"):
        if block.has_value(IA, key):
            out[key] = block[IA, key]
    return out


def read_baryon(block):
    """Baryonic-feedback parameters (``fb``, or ``bacco_<name>``)."""
    out = {}
    if block.has_value(BARYON, "fb"):
        out["fb"] = block[BARYON, "fb"]
    for key in ("M_c", "eta", "beta", "M1_z0_cen", "theta_inn"):
        if block.has_value(BARYON, f"bacco_{key}"):
            out[key] = block[BARYON, f"bacco_{key}"]
    return out


def read_theory_inputs(block, need_nz=True, need_power=True, need_nonlinear=False):
    """Collect the inputs of the fastnc backend from a CosmoSIS data block."""
    from .fastnc_backend import TheoryInputs

    kwargs = dict(
        h0=block[COSMO, "h0"],
        omega_m=block[COSMO, "omega_m"],
        omega_b=block[COSMO, "omega_b"],
        sigma_8=block[COSMO, "sigma_8"],
        n_s=block[COSMO, "n_s"],
        w=block.get_double(COSMO, "w", -1.0),
        z_growth=block["growth_parameters", "z"],
        d_growth=block["growth_parameters", "d_z"],
        ia=read_ia(block),
        baryon=read_baryon(block),
    )
    if need_power:
        kwargs["k_lin"] = block["matter_power_lin", "k_h"]
        kwargs["p_lin"] = _at_redshift_zero(block, "matter_power_lin")
    if need_nonlinear:
        kwargs["k_nl"] = block["matter_power_nl", "k_h"]
        kwargs["p_nl"] = _at_redshift_zero(block, "matter_power_nl")
    if need_nz:
        nbin = block["nz_source", "nbin"]
        kwargs["nz_z"] = block["nz_source", "z"]
        kwargs["nz"] = [block["nz_source", f"bin_{i + 1}"] for i in range(nbin)]
    return TheoryInputs(**kwargs)


def _at_redshift_zero(block, section):
    z = np.atleast_1d(block[section, "z"])
    i = int(np.argmin(np.abs(z)))
    if abs(z[i]) > 1e-6:
        raise ValueError(f"{section} does not contain z = 0")
    return block[section, "p_k"][i, :]


def sample_combinations(block, section=MAP3):
    """Tomographic triplets (int rows) or redshift slices (floats)."""
    scombs = block[section, "sample_combinations"]
    if np.ndim(scombs) == 2:
        return [tuple(int(s) for s in row) for row in scombs]
    return [float(z) for z in np.atleast_1d(scombs)]


def gamma_from_block(block, scomb):
    name = combination_name(scomb)
    return block[NC, f"real-bin_{name}"] + 1j * block[NC, f"imag-bin_{name}"]
