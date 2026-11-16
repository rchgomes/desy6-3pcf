"""Record which elements of a 2pt data file a likelihood actually used.

The standard 2pt likelihood applies its scale cuts internally and only
exposes the surviving data vector.  The joint likelihood needs to know where
those elements sit in the full file in order to pick the matching rows of
the joint covariance; this module recovers that mask by matching values, in
file order.

Options
-------
data_file : str
data_sets : str, e.g. ``xip xim``
like_name : str, e.g. ``2pt``
"""
import numpy as np
from astropy.io import fits
from cosmosis.datablock import option_section

from desy6_3pcf.block import DATA_VECTOR


def match_mask(full, used):
    """Boolean mask selecting the ordered subsequence ``used`` of ``full``."""
    mask = np.zeros(full.size, dtype=bool)
    i = 0
    for j, value in enumerate(full):
        if i < used.size and value == used[i]:
            mask[j] = True
            i += 1
    if i != used.size:
        raise ValueError("the used data vector is not an ordered subset of the data file")
    return mask


def setup(options):
    data_sets = options.get_string(option_section, "data_sets").split()
    with fits.open(options.get_string(option_section, "data_file")) as hdul:
        full = np.concatenate([np.array(hdul[name].data["VALUE"], dtype=float) for name in data_sets])
    return {"full": full, "like_name": options.get_string(option_section, "like_name")}


def execute(block, config):
    name = config["like_name"]
    mask = match_mask(config["full"], block[DATA_VECTOR, f"{name}_data"])
    block[DATA_VECTOR, f"{name}_mask"] = mask.astype(int)
    return 0


def cleanup(config):
    pass
