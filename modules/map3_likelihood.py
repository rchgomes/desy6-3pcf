"""Likelihood of the <Map^3> data vector.

Only the data points predicted upstream (i.e. selected in ``map3_binning``)
enter; their positions in the data file are written to
``data_vector/map3_mask`` for the joint likelihood.

Options
-------
data_file : str
    <Map^3> FITS file with a ``COVMAT`` extension.
covariance_realizations : int (-1)
    Number of simulations behind the covariance (Hartlap correction).
free_parameters : int (-1)
    Number of sampled parameters (Dodelson-Schneider / Percival).
percival : bool (F)
    Use the Percival et al. (2022) likelihood.
"""
import numpy as np
from cosmosis.datablock import option_section

from desy6_3pcf.binning import combination_name
from desy6_3pcf.block import DATA_VECTOR, LIKELIHOODS, MAP3, sample_combinations
from desy6_3pcf.datavector import Map3Data
from desy6_3pcf.likelihood import log_likelihood


def setup(options):
    return {
        "data": Map3Data.from_fits(options.get_string(option_section, "data_file")),
        "nsim": options.get_int(option_section, "covariance_realizations", -1),
        "npar": options.get_int(option_section, "free_parameters", -1),
        "percival": options.get_bool(option_section, "percival", False),
        "cache": {},
    }


def execute(block, config):
    data = config["data"]
    theory = np.zeros(data.size)
    used = np.zeros(data.size, dtype=bool)
    for scomb in sample_combinations(block):
        name = combination_name(scomb)
        where = data.indices(scomb, block[MAP3, f"filters_{name}"])
        theory[where] = block[MAP3, f"map3-bin_{name}"]
        used[where] = True

    key = used.tobytes()
    if key not in config["cache"]:
        config["cache"] = {key: np.linalg.inv(data.cov[np.ix_(used, used)])}
    diff = data.value[used] - theory[used]
    like, chi2, inv_cov = log_likelihood(diff, config["cache"][key], config["nsim"], config["npar"], config["percival"])

    block[LIKELIHOODS, "map3_like"] = like
    block[DATA_VECTOR, "map3_chi2"] = chi2
    block[DATA_VECTOR, "map3_n"] = diff.size
    block[DATA_VECTOR, "map3_data"] = data.value[used]
    block[DATA_VECTOR, "map3_theory"] = theory[used]
    block[DATA_VECTOR, "map3_inverse_covariance"] = inv_cov
    block[DATA_VECTOR, "map3_mask"] = used.astype(int)
    return 0


def cleanup(config):
    pass
