"""Convert natural components of the shear 3PCF into <Map^3>.

The binned theory 3PCF is integrated with the same kernels and the same
discretisation as TreeCorr applies to the measurement.  Besides one entry
per sample combination, the concatenation over all combinations is written
to ``map3/vector`` so that a whole training sample fits in one
``extra_output`` of the CosmoSIS ``list`` sampler.
"""
import numpy as np

from desy6_3pcf.binning import combination_name
from desy6_3pcf.block import MAP3, NC, gamma_from_block, sample_combinations
from desy6_3pcf.map3 import Map3Transform


def setup(options):
    return {"transforms": {}}


def _transform(block, config, filters):
    # The kernels depend only on the bins and the filters: build them once.
    key = filters.tobytes()
    if key not in config["transforms"]:
        theta = block[NC, "theta"]
        config["transforms"][key] = Map3Transform(
            theta, theta, block[NC, "phi"], block[NC, "dlnt"], block[NC, "dphi"], filters
        )
    return config["transforms"][key]


def execute(block, config):
    vector = []
    for scomb in sample_combinations(block):
        name = combination_name(scomb)
        filters = block[MAP3, f"filters_{name}"]
        map3 = _transform(block, config, filters)(gamma_from_block(block, scomb))
        block[MAP3, f"map3-bin_{name}"] = map3
        vector.append(map3)
    block[MAP3, "vector"] = np.concatenate(vector)
    return 0


def cleanup(config):
    pass
