"""Forecasts from Fisher matrices written by the CosmoSIS ``fisher`` sampler."""
import numpy as np

from .chains import read_columns


def read_fisher(filename):
    """Return ``(parameter_names, fisher_matrix)``."""
    names = [n.lower() for n in read_columns(filename)]
    matrix = np.atleast_2d(np.loadtxt(filename))
    if matrix.shape != (len(names), len(names)):
        raise ValueError(f"{filename} does not contain a {len(names)}x{len(names)} Fisher matrix")
    return names, matrix


def add_gaussian_priors(names, fisher, sigmas):
    """Add independent Gaussian priors ``{name: sigma}`` to a Fisher matrix."""
    fisher = fisher.copy()
    for name, sigma in sigmas.items():
        i = names.index(name.lower())
        fisher[i, i] += 1.0 / sigma**2
    return fisher


def fix_parameters(names, fisher, fixed):
    """Remove (i.e. condition on) the parameters in ``fixed``."""
    fixed = {f.lower() for f in fixed}
    keep = [i for i, n in enumerate(names) if n not in fixed]
    return [names[i] for i in keep], fisher[np.ix_(keep, keep)]


def covariance(fisher):
    return np.linalg.inv(fisher)


def marginal_errors(names, fisher):
    return dict(zip(names, np.sqrt(np.diag(covariance(fisher)))))


def conditional_errors(names, fisher):
    return dict(zip(names, 1.0 / np.sqrt(np.diag(fisher))))


def figure_of_merit(names, fisher, p1, p2):
    """``1 / sqrt(det C)`` of the marginal covariance of two parameters."""
    idx = [names.index(p1.lower()), names.index(p2.lower())]
    return 1.0 / np.sqrt(np.linalg.det(covariance(fisher)[np.ix_(idx, idx)]))


def ellipse(names, fisher, p1, p2, centre=(0.0, 0.0), nsigma=1.0, npoints=200):
    """Points on the ``nsigma`` marginal contour of two parameters.

    ``nsigma = 1`` and ``2`` map to the 68.3% and 95.4% contours.
    """
    idx = [names.index(p1.lower()), names.index(p2.lower())]
    cov = covariance(fisher)[np.ix_(idx, idx)]
    delta_chi2 = {1.0: 2.30, 2.0: 6.18, 3.0: 11.83}.get(float(nsigma), nsigma**2)
    t = np.linspace(0.0, 2.0 * np.pi, npoints)
    circle = np.array([np.cos(t), np.sin(t)])
    return np.asarray(centre)[:, None] + np.sqrt(delta_chi2) * np.linalg.cholesky(cov) @ circle
