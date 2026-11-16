"""Joint likelihood of correlated data vectors (2pt + <Map^3>).

Each individual likelihood upstream leaves ``<name>_data``,
``<name>_theory`` and ``<name>_mask`` in the ``data_vector`` section; this
module stacks them and evaluates the likelihood with the joint covariance,
including the cross-covariance between the probes.  The individual
likelihoods must not be listed in ``[pipeline] likelihoods``.

The covariance file has an image extension (default ``COVMAT``) whose header
gives the start index of every data set, ``NAME_i`` / ``STRT_i``, in the
order and with the lengths of the *uncut* data files.

Options
-------
data_file : str
covmat_name : str (COVMAT)
like_names : str, e.g. ``2pt map3``
<like_name>_data_sets : str, e.g. ``2pt_data_sets = xip xim``
like_name : str (joint)
exclude_cross_cov : bool (F)
covariance_realizations, free_parameters, percival : as in ``map3_likelihood``
"""
import numpy as np
from astropy.io import fits
from cosmosis.datablock import option_section

from desy6_3pcf.block import DATA_VECTOR, LIKELIHOODS
from desy6_3pcf.likelihood import log_likelihood


def read_covariance(filename, covmat_name="COVMAT"):
    """Return the covariance and the index range of every data set in it."""
    with fits.open(filename) as hdul:
        cov = np.array(hdul[covmat_name].data, dtype=float)
        header = hdul[covmat_name].header
        names, starts = [], []
        while f"NAME_{len(names)}" in header:
            starts.append(int(header[f"STRT_{len(names)}"]))
            names.append(header[f"NAME_{len(names)}"].lower())
    ends = starts[1:] + [cov.shape[0]]
    return cov, {name: np.arange(a, b) for name, a, b in zip(names, starts, ends)}


def setup(options):
    like_names = options.get_string(option_section, "like_names").split()
    cov, indices = read_covariance(
        options.get_string(option_section, "data_file"),
        options.get_string(option_section, "covmat_name", "COVMAT"),
    )
    like_indices = {}
    for name in like_names:
        data_sets = options.get_string(option_section, f"{name}_data_sets").lower().split()
        like_indices[name] = np.concatenate([indices[d] for d in data_sets])
    return {
        "name": options.get_string(option_section, "like_name", "joint"),
        "like_names": like_names,
        "cov": cov,
        "indices": like_indices,
        "exclude_cross_cov": options.get_bool(option_section, "exclude_cross_cov", False),
        "nsim": options.get_int(option_section, "covariance_realizations", -1),
        "npar": options.get_int(option_section, "free_parameters", -1),
        "percival": options.get_bool(option_section, "percival", False),
        "cache": {},
    }


def _inverse_covariance(config, masks):
    key = b"".join(m.tobytes() for m in masks)
    if key not in config["cache"]:
        used = [config["indices"][n][m] for n, m in zip(config["like_names"], masks)]
        cov = config["cov"][np.ix_(np.concatenate(used), np.concatenate(used))].copy()
        if config["exclude_cross_cov"]:
            edges = np.cumsum([0] + [u.size for u in used])
            block_id = np.repeat(np.arange(len(used)), np.diff(edges))
            cov[block_id[:, None] != block_id[None, :]] = 0.0
        config["cache"] = {key: np.linalg.inv(cov)}
    return config["cache"][key]


def execute(block, config):
    names = config["like_names"]
    masks = [np.asarray(block[DATA_VECTOR, f"{n}_mask"]).astype(bool) for n in names]
    data = np.concatenate([block[DATA_VECTOR, f"{n}_data"] for n in names])
    theory = np.concatenate([block[DATA_VECTOR, f"{n}_theory"] for n in names])
    for n, m in zip(names, masks):
        if m.size != config["indices"][n].size:
            raise ValueError(f"{n}: mask has {m.size} elements but the covariance has {config['indices'][n].size}")

    like, chi2, inv_cov = log_likelihood(
        data - theory, _inverse_covariance(config, masks), config["nsim"], config["npar"], config["percival"]
    )
    name = config["name"]
    block[LIKELIHOODS, f"{name}_like"] = like
    block[DATA_VECTOR, f"{name}_chi2"] = chi2
    block[DATA_VECTOR, f"{name}_n"] = data.size
    block[DATA_VECTOR, f"{name}_data"] = data
    block[DATA_VECTOR, f"{name}_theory"] = theory
    block[DATA_VECTOR, f"{name}_inverse_covariance"] = inv_cov
    return 0


def cleanup(config):
    pass
