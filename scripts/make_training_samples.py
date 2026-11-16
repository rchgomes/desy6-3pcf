#!/usr/bin/env python
"""Draw the Sobol design of an emulator training set.

Writes ``<outdir>/<design>_<chunk>.txt`` in the format of the CosmoSIS
``list`` sampler.  The sequence is deterministic, so a training set can be
extended later with ``--skip`` (number of points already generated) and
``--first-chunk``.

    python scripts/make_training_samples.py reference_lcdm --n 2000 --chunks 4
"""
import argparse
import os
from pathlib import Path

import numpy as np

from desy6_3pcf.design import load_design, sobol_samples, write_cosmosis_list

ROOT = Path(os.environ.get("DESY6_3PCF_DIR", Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("design", help="name of a design in the designs file, e.g. reference_lcdm, tatt")
    parser.add_argument("--n", type=int, required=True, help="number of samples to draw")
    parser.add_argument("--chunks", type=int, default=1, help="split the samples into this many files")
    parser.add_argument("--skip", type=int, default=0, help="number of points of the sequence to skip")
    parser.add_argument("--first-chunk", type=int, default=0, help="index of the first output file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--designs", default=ROOT / "config/emulator/designs.yaml")
    parser.add_argument("--outdir", default=ROOT / "data/training/samples")
    args = parser.parse_args()

    ranges = load_design(args.designs, args.design)
    samples = sobol_samples(ranges, args.n, seed=args.seed, skip=args.skip)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(np.array_split(samples, args.chunks)):
        filename = outdir / f"{args.design}_{args.first_chunk + i}.txt"
        write_cosmosis_list(filename, list(ranges), chunk)
        print(f"{filename}: {len(chunk)} samples")


if __name__ == "__main__":
    main()
