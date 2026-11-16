#!/usr/bin/env python
"""Compare two <Map^3> FITS files point by point (e.g. emulator vs fastnc).

    python scripts/compare_map3.py reference.fits other.fits

Differences are quoted relative to the first file and, if it carries a
covariance, in units of its errors together with the total chi2.
"""
import argparse

import numpy as np

from desy6_3pcf.datavector import Map3Data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("other")
    args = parser.parse_args()

    a, b = Map3Data.from_fits(args.reference), Map3Data.from_fits(args.other)
    if a.size != b.size or not (np.array_equal(a.z, b.z) and np.allclose(a.theta, b.theta)):
        raise SystemExit("the two files do not contain the same data points")
    diff = b.value - a.value
    sigma = None if a.cov is None else np.sqrt(np.diag(a.cov))

    print(" z1 z2 z3   theta1 theta2 theta3    reference        other   diff[%]" + ("  diff/sigma" if sigma is not None else ""))
    for i in range(a.size):
        line = "{:3d}{:3d}{:3d}  {:7.1f}{:7.1f}{:7.1f} {:12.4e} {:12.4e} {:9.3f}".format(
            *a.z[:, i], *a.theta[:, i], a.value[i], b.value[i], 100 * diff[i] / a.value[i])
        if sigma is not None:
            line += f" {diff[i] / sigma[i]:11.3f}"
        print(line)
    print(f"\nmax |diff| = {100 * np.max(np.abs(diff / a.value)):.3f} %")
    if sigma is not None:
        print(f"max |diff|/sigma = {np.max(np.abs(diff / sigma)):.3f},  chi2 = {diff @ np.linalg.solve(a.cov, diff):.4f}")


if __name__ == "__main__":
    main()
