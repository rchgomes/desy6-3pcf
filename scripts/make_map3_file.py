#!/usr/bin/env python
"""Pack a <Map^3> measurement and its covariance into the pipeline's formats.

    python scripts/make_map3_file.py --map3 map3.npy --cov cov_map3.npy --nsim 1000 \\
        -o data/dv/map3.fits

    python scripts/make_map3_file.py --joint-cov cov_joint.npy --nsim 1000 \\
        --lengths xip=200 xim=200 map3=100 -o data/dv/cov_xip_xim_map3.fits

The <Map^3> vector is ordered by tomographic triplet (1,1,1), (1,1,2), ...,
(4,4,4) and, within each, by filter: (3,3,3)', (5,5,5)', ... for the default
equal-aperture filters.  Without ``--map3`` the values are zero, which is
enough as a layout/covariance template for synthetic data vectors.

Sign convention: the pipeline predicts <Map^3> positive for the lensing
signal; make sure the measurement follows the same convention.
"""
import argparse

import numpy as np
from astropy.io import fits

from desy6_3pcf import binning
from desy6_3pcf.datavector import Map3Data


def load(filename):
    return np.load(filename) if str(filename).endswith(".npy") else np.loadtxt(filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--map3", help="data vector (.npy or text)")
    parser.add_argument("--cov", help="<Map^3> covariance (.npy or text)")
    parser.add_argument("--nsim", type=int, default=0, help="number of simulations behind the covariance (0: analytic)")
    parser.add_argument("--nbin", type=int, default=4, help="number of tomographic bins")
    parser.add_argument("--filters", type=float, nargs="+", default=list(binning.FILTERS))
    parser.add_argument("--filter-mode", choices=["equal", "all"], default="equal")
    parser.add_argument("--joint-cov", help="joint covariance of all probes (.npy or text); writes a covariance file")
    parser.add_argument("--lengths", nargs="+", metavar="NAME=N",
                        help="data sets of the joint covariance, in order, with their lengths")
    args = parser.parse_args()

    if args.joint_cov:
        cov = load(args.joint_cov)
        header = fits.Header()
        header["EXTNAME"] = "COVMAT"
        header["NSIM"] = args.nsim
        start = 0
        for i, item in enumerate(args.lengths):
            name, n = item.split("=")
            header[f"NAME_{i}"] = name
            header[f"STRT_{i}"] = start
            start += int(n)
        if start != cov.shape[0]:
            raise SystemExit(f"--lengths add up to {start} but the covariance is {cov.shape[0]}x{cov.shape[0]}")
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cov, header=header)]).writeto(args.output, overwrite=True)
    else:
        data = Map3Data.from_bins(
            binning.tomographic_triplets(args.nbin),
            binning.filter_triplets(args.filters, args.filter_mode),
            cov=None if args.cov is None else load(args.cov),
            nsim=args.nsim,
        )
        if args.map3:
            data.value[:] = load(args.map3)
        data.to_fits(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
