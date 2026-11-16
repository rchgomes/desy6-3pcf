"""Multiplicative shear calibration of <Map^3>: a factor (1 + m_i) per bin."""
import numpy as np

from desy6_3pcf.binning import combination_name
from desy6_3pcf.block import MAP3, sample_combinations


def setup(options):
    return {}


def execute(block, config):
    for scomb in sample_combinations(block):
        m = [block["shear_calibration_parameters", f"m{s}"] for s in scomb]
        block[MAP3, f"map3-bin_{combination_name(scomb)}"] *= np.prod(1.0 + np.array(m))
    return 0


def cleanup(config):
    pass
