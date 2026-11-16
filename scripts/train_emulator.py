#!/usr/bin/env python
"""Train a <Map^3> redshift-slice emulator from CosmoSIS training-set outputs.

    python scripts/train_emulator.py reference_lcdm \\
        data/training/output/reference_lcdm_*.txt -o data/emulators/reference_lcdm

    python scripts/train_emulator.py tatt --component ddE \\
        data/training/output/tatt_ddE_*.txt -o data/emulators/tatt_ddE

Outputs ``<prefix>.pkl`` (CosmoPower network), ``<prefix>.json`` (metadata)
and ``<prefix>_test.npz`` (held-out samples for validate_emulator.py).
"""
import argparse
import os
from pathlib import Path

import numpy as np

from desy6_3pcf import binning
from desy6_3pcf.chains import read_chain, vector_columns
from desy6_3pcf.design import load_design
from desy6_3pcf.emulator import FeatureScaler, Map3SliceEmulator, import_cosmopower

ROOT = Path(os.environ.get("DESY6_3PCF_DIR", Path(__file__).resolve().parents[1]))


def load_training_set(filenames, parameters):
    """Stack chains; return (params, features) with failed samples removed."""
    params, features = [], []
    for filename in filenames:
        names, samples = read_chain(filename)
        params.append(samples[:, [names.index(p) for p in parameters]])
        features.append(samples[:, vector_columns(names, "map3--vector")])
    params, features = np.concatenate(params), np.concatenate(features)
    good = np.all(np.isfinite(features), axis=1) & np.all(np.isfinite(params), axis=1)
    print(f"{len(good)} samples read, {np.sum(~good)} with non-finite values dropped")
    return params[good], features[good]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("design", help="design the samples were drawn from (defines the input parameters)")
    parser.add_argument("chains", nargs="+", help="CosmoSIS list-sampler outputs")
    parser.add_argument("-o", "--output", required=True, help="output prefix")
    parser.add_argument("--component", choices=binning.TATT_COMPONENTS, help="TATT component (omit for the reference emulator)")
    parser.add_argument("--no-remove-alignment", action="store_true",
                        help="the TATT training set was made with remove_alignment = F")
    parser.add_argument("--transform", choices=["auto", "log10", "asinh"], default="auto")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 256, 1024, 1024, 384, 192])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[60, 30, 20, 10, 5])
    parser.add_argument("--patience", type=int, nargs="+", default=[200, 200, 150, 100, 100])
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--validation-split", type=float, default=0.25)
    parser.add_argument("--z-slices", type=float, nargs="+", default=list(binning.Z_SLICES))
    parser.add_argument("--filters", type=float, nargs="+", default=list(binning.FILTERS))
    parser.add_argument("--filter-mode", choices=["equal", "all"], default="equal")
    parser.add_argument("--designs", default=ROOT / "config/emulator/designs.yaml")
    args = parser.parse_args()

    design = load_design(args.designs, args.design)
    parameters = list(design)
    box_min, box_max = np.array(list(design.values())).T
    filters = binning.filter_triplets(args.filters, args.filter_mode)
    params, features = load_training_set(args.chains, parameters)
    nfeature = len(args.z_slices) * filters.shape[1]
    if features.shape[1] != nfeature:
        raise SystemExit(
            f"the chains hold {features.shape[1]} features but {len(args.z_slices)} slices x "
            f"{filters.shape[1]} filters = {nfeature}; pass the --z-slices/--filters of the training runs"
        )

    transform = args.transform
    if transform == "auto":
        transform = "log10" if np.all(features > 0) else "asinh"
    print(f"{len(parameters)} parameters -> {nfeature} features, '{transform}' transform")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(params))
    ntest = int(round(args.test_fraction * len(params)))
    test, train = order[:ntest], order[ntest:]

    scaler = FeatureScaler.fit(features[train], transform)
    emulator = Map3SliceEmulator(
        None, parameters, box_min, box_max,  # valid over the whole design box
        args.z_slices, filters, scaler,
        info={
            "design": args.design,
            "component": args.component or "reference",
            "remove_alignment": not args.no_remove_alignment,
            "n_train": int(len(train)),
            "chains": [os.path.basename(c) for c in args.chains],
            "measurement": {"min_sep": binning.MIN_SEP, "max_sep": binning.MAX_SEP,
                            "nbins": binning.NBINS, "nphi_bins": binning.NPHI_BINS},
        },
    )

    cosmopower_NN = import_cosmopower()
    import tensorflow as tf

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    network = cosmopower_NN(parameters=parameters, modes=np.arange(nfeature), n_hidden=args.hidden, verbose=True)
    unit = emulator.normalise(params[train])
    nstep = len(args.learning_rates)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    network.train(
        training_parameters={name: unit[:, i] for i, name in enumerate(parameters)},
        training_features=scaler.encode(features[train]),
        filename_saved_model=str(args.output),
        validation_split=args.validation_split,
        learning_rates=args.learning_rates,
        batch_sizes=args.batch_sizes,
        gradient_accumulation_steps=[1] * nstep,
        patience_values=args.patience,
        max_epochs=[args.max_epochs] * nstep,
    )

    emulator.save_metadata(args.output)
    np.savez(f"{args.output}_test.npz", params=params[test], features=features[test])
    print(f"saved {args.output}.pkl, {args.output}.json and {len(test)} test samples in {args.output}_test.npz")


if __name__ == "__main__":
    main()
