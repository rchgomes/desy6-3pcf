#!/usr/bin/env python
"""Summarise Fisher matrices from the CosmoSIS ``fisher`` sampler.

    python scripts/fisher_summary.py output/fisher/fisher-joint-fastnc-lcdm-nla.txt
    python scripts/fisher_summary.py A.txt B.txt --labels 2pt 2pt+Map3 --plot fisher.png

Prints marginalised and conditional 1-sigma errors and the figure of merit
of a parameter pair (default Omega_m - S_8); with ``--plot`` draws the 68%
and 95% ellipses.  The Gaussian priors of the run's priors file are already
in the matrix (the fisher sampler adds them); further ones can be added
with ``--add-prior`` and parameters can be held fixed with ``--fix``.
"""
import argparse

from desy6_3pcf import fisher as F


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--add-prior", nargs="*", default=[], metavar="NAME=SIGMA",
                        help="extra Gaussian priors, e.g. cosmological_parameters--h0=0.01")
    parser.add_argument("--fix", nargs="*", default=[], help="parameters to hold fixed (section--name)")
    parser.add_argument("--pair", nargs=2, default=["cosmological_parameters--omega_m", "cosmological_parameters--s_8"])
    parser.add_argument("--plot", help="output image with the ellipses of --pair")
    args = parser.parse_args()
    labels = args.labels or args.files
    priors = {item.split("=")[0].lower(): float(item.split("=")[1]) for item in args.add_prior}

    results = []
    for filename, label in zip(args.files, labels):
        names, fisher = F.read_fisher(filename)
        fisher = F.add_gaussian_priors(names, fisher, priors)
        names, fisher = F.fix_parameters(names, fisher, args.fix)
        marginal, conditional = F.marginal_errors(names, fisher), F.conditional_errors(names, fisher)
        print(f"\n== {label}")
        print(f"{'parameter':48s} {'marginalised':>13s} {'conditional':>13s}")
        for name in names:
            print(f"{name:48s} {marginal[name]:13.5g} {conditional[name]:13.5g}")
        if all(p.lower() in names for p in args.pair):
            print(f"FoM({args.pair[0].split('--')[1]}, {args.pair[1].split('--')[1]}) = "
                  f"{F.figure_of_merit(names, fisher, *args.pair):.1f}")
        results.append((label, names, fisher))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4.5))
        for k, (label, names, fisher) in enumerate(results):
            for nsigma, alpha in ((2.0, 0.25), (1.0, 0.5)):
                x, y = F.ellipse(names, fisher, *args.pair, nsigma=nsigma)
                ax.fill(x, y, color=f"C{k}", alpha=alpha, label=label if nsigma == 1.0 else None)
        ax.set(xlabel=r"$\Delta$ " + args.pair[0].split("--")[1], ylabel=r"$\Delta$ " + args.pair[1].split("--")[1])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
