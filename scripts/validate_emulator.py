#!/usr/bin/env python
"""Accuracy of a trained emulator on its held-out test samples.

    python scripts/validate_emulator.py data/emulators/reference_lcdm

Prints percentiles of the fractional error per redshift slice and writes
``<prefix>_validation.png``.  This tests the network before the
line-of-sight integration; for the end-to-end test against brute-force
fastnc see config/validation/predict.ini.
"""
import argparse

import numpy as np

from desy6_3pcf.emulator import Map3SliceEmulator


def fractional_error(emulator, params, truth):
    """(prediction - truth) / truth, with |truth| floored at the feature scale
    for emulators of signed quantities (which cross zero)."""
    prediction = emulator.predict_features(params)
    denominator = np.abs(truth)
    if emulator.scaler.scale is not None:
        denominator = np.maximum(denominator, emulator.scaler.scale)
    return (prediction - truth) / denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", help="emulator prefix (as passed to train_emulator.py -o)")
    parser.add_argument("--test", help="test set (default: <prefix>_test.npz)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    emulator = Map3SliceEmulator.load(args.prefix)
    test = np.load(args.test or f"{args.prefix}_test.npz")
    params, truth = test["params"], test["features"]
    inside = emulator.in_range(params)
    print(f"{len(params)} test samples, {np.sum(~inside)} outside the training range (ignored)")
    error = 100.0 * fractional_error(emulator, params[inside], truth[inside])

    nz, nf = emulator.z_slices.size, emulator.filters.shape[1]
    abs_error = np.abs(error).reshape(-1, nz, nf)
    print("\n|error| in per cent, all features: " + ", ".join(
        f"{q}% < {np.percentile(abs_error, q):.3f}" for q in (68, 95, 99, 100)))
    print("\n  z      68%      95%      max")
    for i, z in enumerate(emulator.z_slices):
        e = abs_error[:, i]
        print(f"{z:6.3f} {np.percentile(e, 68):8.3f} {np.percentile(e, 95):8.3f} {e.max():8.3f}")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for j in range(nf):
            label = "(" + ", ".join(f"{t:g}" for t in emulator.filters[:, j]) + ")'"
            axes[0].plot(emulator.z_slices, np.percentile(abs_error[:, :, j], 95, axis=0), label=label)
        axes[0].set(xlabel="redshift slice", ylabel="95th percentile of |error| [%]", yscale="log")
        axes[0].legend(title="filters", fontsize=8)
        axes[1].hist(error.ravel(), bins=200, range=np.percentile(error, [0.5, 99.5]), color="0.3")
        axes[1].set(xlabel="(emulated - true) / true [%]", ylabel="features x test samples")
        fig.suptitle(f"{args.prefix}: {emulator.info.get('component', '')}")
        fig.tight_layout()
        fig.savefig(f"{args.prefix}_validation.png", dpi=150)
        print(f"\nwrote {args.prefix}_validation.png")


if __name__ == "__main__":
    main()
