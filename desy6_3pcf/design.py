"""Experimental designs (Sobol sequences) for the emulator training sets."""
import numpy as np
import yaml
from scipy.stats import qmc


def load_design(filename, name):
    """Read parameter ranges of design ``name`` from a YAML file.

    Returns a dict ``{'section--parameter': (low, high)}``.
    """
    with open(filename) as f:
        designs = yaml.safe_load(f)
    if name not in designs:
        raise KeyError(f"design {name!r} not found in {filename}; available: {sorted(designs)}")
    return {key.lower(): tuple(map(float, rng)) for key, rng in designs[name].items()}


def sobol_samples(ranges, n, seed=0, skip=0):
    """``n`` scrambled-Sobol points in the box ``ranges``, skipping the first
    ``skip`` points of the sequence so that a training set can be extended."""
    sampler = qmc.Sobol(d=len(ranges), scramble=True, seed=seed)
    if skip:
        sampler.fast_forward(skip)
    low, high = np.array(list(ranges.values())).T
    return qmc.scale(sampler.random(n), low, high)


def write_cosmosis_list(filename, names, samples):
    """Write samples in the format read by the CosmoSIS ``list`` sampler."""
    np.savetxt(filename, samples, header=" ".join(names))
