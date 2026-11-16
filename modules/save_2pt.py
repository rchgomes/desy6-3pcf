"""Write the 2pt theory vector into a copy of the 2pt data file.

The output has the layout, redshift distributions and covariance of the
input file, with the values of the data sets replaced by the prediction
that the 2pt likelihood compared to the data (so binning and bin averaging
are identical by construction).  Elements removed by scale cuts keep their
original values.  Run after the 2pt likelihood and its ``data_mask``.

Options
-------
data_file : str
filename : str
data_sets : str, e.g. ``xip xim``
like_name : str (2pt)
"""
import numpy as np
from astropy.io import fits
from cosmosis.datablock import option_section

from desy6_3pcf.block import DATA_VECTOR


def setup(options):
    return {
        "data_file": options.get_string(option_section, "data_file"),
        "filename": options.get_string(option_section, "filename"),
        "data_sets": options.get_string(option_section, "data_sets").split(),
        "like_name": options.get_string(option_section, "like_name", "2pt"),
    }


def execute(block, config):
    name = config["like_name"]
    theory = block[DATA_VECTOR, f"{name}_theory"]
    mask = np.asarray(block[DATA_VECTOR, f"{name}_mask"]).astype(bool)
    with fits.open(config["data_file"]) as hdul:
        full = np.concatenate([np.array(hdul[d].data["VALUE"], dtype=float) for d in config["data_sets"]])
        full[mask] = theory
        start = 0
        for d in config["data_sets"]:
            n = len(hdul[d].data)
            hdul[d].data["VALUE"] = full[start:start + n]
            start += n
        print(f"[save_2pt] writing {config['filename']}")
        hdul.writeto(config["filename"], overwrite=True)
    return 0


def cleanup(config):
    pass
