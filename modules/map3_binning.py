"""Define which <Map^3> data points the pipeline predicts.

Third-order statistics are expensive, so the combinations of redshift bins
and aperture radii are fixed once, up front, and shared by every downstream
module through the data block.

Three ways of specifying them:

1. ``data_file``: take the points of a <Map^3> FITS file, optionally reduced
   by ``sample_combinations`` and ``theta_range`` (scale cuts);
2. ``nbin`` + ``filters``: build them from scratch (no data file needed,
   e.g. to generate a synthetic data vector);
3. ``z_slices`` + ``filters``: thin redshift slices instead of tomographic
   bins, used to generate the emulator training sets.

Options
-------
data_file : str
sample_combinations : str
    ``all`` (default), ``auto``, ``cross`` and/or explicit triplets ``1,1,2``.
theta_range : two floats, arcmin
    Keep points whose three aperture radii all lie in this range.
nbin : int
filters : floats, arcmin (default: the Y6 filters)
filter_mode : ``equal`` (default) or ``all``
z_slices : floats (default with ``slices = T``: the Y6 emulator slices)
slices : bool
"""
import numpy as np
from cosmosis.datablock import option_section

from desy6_3pcf import binning
from desy6_3pcf.block import MAP3, NC, get_array
from desy6_3pcf.datavector import Map3Data


def setup(options):
    filters = get_array(options, option_section, "filters", default=np.array(binning.FILTERS))
    filter_mode = options.get_string(option_section, "filter_mode", "equal")
    selection = options.get_string(option_section, "sample_combinations", "all")
    use_slices = options.get_bool(option_section, "slices", False) or options.has_value(option_section, "z_slices")

    if use_slices:
        scombs = get_array(options, option_section, "z_slices", default=np.array(binning.Z_SLICES))
        triplets = binning.filter_triplets(filters, filter_mode)
        return {"sample_combinations": scombs, "filters": {binning.combination_name(z): triplets for z in scombs}}

    if options.has_value(option_section, "data_file"):
        data = Map3Data.from_fits(options.get_string(option_section, "data_file"))
        nbin = int(data.z.max())
    else:
        nbin = options.get_int(option_section, "nbin")
        data = Map3Data.from_bins(
            binning.tomographic_triplets(nbin), binning.filter_triplets(filters, filter_mode)
        )

    keep = np.zeros(data.size, dtype=bool)
    for triplet in binning.parse_tomographic_triplets(selection, nbin):
        keep |= data.triplet_mask(triplet)
    theta_range = get_array(options, option_section, "theta_range")
    if theta_range is not None:
        keep &= data.theta_mask(*theta_range)
    if not keep.any():
        raise ValueError("no <Map^3> data points survive the selection")
    data.select(keep)
    print(f"[map3_binning] using {data.size} <Map^3> data points")

    scombs = data.tomographic_triplets()
    return {
        "sample_combinations": scombs,
        "filters": {binning.combination_name(s): data.filters(s) for s in scombs},
    }


def execute(block, config):
    block[MAP3, "sample_combinations"] = config["sample_combinations"]
    block[NC, "sample_combinations"] = config["sample_combinations"]
    for name, filters in config["filters"].items():
        block[MAP3, f"filters_{name}"] = filters
    return 0


def cleanup(config):
    pass
