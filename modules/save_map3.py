"""Write the predicted <Map^3> to a FITS file (synthetic data vectors).

Options
-------
filename : str
    Output file.
copy_covariance : str, optional
    Existing <Map^3> file whose layout and covariance are copied; without
    it the file contains exactly the predicted points and no covariance.
"""
import numpy as np
from cosmosis.datablock import option_section

from desy6_3pcf.binning import combination_name
from desy6_3pcf.block import MAP3, sample_combinations
from desy6_3pcf.datavector import Map3Data


def setup(options):
    return {
        "filename": options.get_string(option_section, "filename"),
        "template": options.get_string(option_section, "copy_covariance", ""),
    }


def execute(block, config):
    scombs = sample_combinations(block)
    filters = [block[MAP3, f"filters_{combination_name(s)}"] for s in scombs]
    if config["template"]:
        data = Map3Data.from_fits(config["template"])
    else:
        z = np.concatenate([np.repeat(np.array(s)[:, None], f.shape[1], axis=1) for s, f in zip(scombs, filters)], axis=1)
        data = Map3Data(z, np.concatenate(filters, axis=1), np.zeros(z.shape[1]))
    for scomb, f in zip(scombs, filters):
        data.value[data.indices(scomb, f)] = block[MAP3, f"map3-bin_{combination_name(scomb)}"]
    print(f"[save_map3] writing {config['filename']}")
    data.to_fits(config["filename"])
    return 0


def cleanup(config):
    pass
