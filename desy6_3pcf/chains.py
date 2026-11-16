"""Reading CosmoSIS text outputs."""
import numpy as np


def read_columns(filename):
    """Column names (lower case) of a CosmoSIS text output."""
    with open(filename) as f:
        return f.readline().lstrip("#").split()


def read_chain(filename):
    """Return ``(names, samples)``; names are lower case, ``section--name``."""
    names = [n.lower() for n in read_columns(filename)]
    samples = np.atleast_2d(np.loadtxt(filename))
    if samples.shape[1] != len(names):
        raise ValueError(f"{filename}: header has {len(names)} columns but the data have {samples.shape[1]}")
    return names, samples


def vector_columns(names, prefix):
    """Indices of the columns ``<prefix>_0, <prefix>_1, ...`` in order."""
    prefix = prefix.lower().replace("/", "--") + "_"
    found = {}
    for i, name in enumerate(names):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            found[int(name[len(prefix):])] = i
    if not found:
        raise KeyError(f"no columns named {prefix}<i>")
    if sorted(found) != list(range(len(found))):
        raise ValueError(f"columns {prefix}<i> are not contiguous")
    return np.array([found[i] for i in range(len(found))])
