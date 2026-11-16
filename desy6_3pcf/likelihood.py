"""Gaussian-type likelihoods for covariances estimated from simulations."""
import numpy as np


def hartlap_factor(nsim, ndata):
    """Anderson-Hartlap debiasing factor of the inverse covariance."""
    return (nsim - ndata - 2.0) / (nsim - 1.0)


def dodelson_schneider_factor(nsim, ndata, npar):
    """Dodelson & Schneider (2013) inflation of the parameter covariance,
    expressed as a factor on the inverse data covariance."""
    B = (nsim - ndata - 2.0) / ((nsim - ndata - 1.0) * (nsim - ndata - 4.0))
    return 1.0 / (1.0 + B * (ndata - npar))


def log_likelihood(diff, inv_cov, nsim=-1, npar=-1, percival=False):
    """Log-likelihood of a residual vector.

    Parameters
    ----------
    diff : array
        Data minus theory.
    inv_cov : array
        Inverse of the (unscaled) covariance matrix.
    nsim : int
        Number of simulations behind the covariance; ``<= 0`` means the
        covariance is treated as exactly known.
    npar : int
        Number of free parameters, used by the Dodelson-Schneider and
        Percival corrections; ``<= 0`` disables the former.
    percival : bool
        Use the Percival et al. (2022) t-distribution-like likelihood
        instead of a Gaussian with rescaled inverse covariance.

    Returns
    -------
    loglike, chi2, inv_cov
        ``chi2`` and ``inv_cov`` include the Hartlap / Dodelson-Schneider
        factors in the Gaussian case and are unscaled in the Percival case.
    """
    ndata = diff.size
    if percival:
        if nsim <= 0 or npar <= 0:
            raise ValueError("the Percival likelihood needs covariance_realizations and free_parameters")
        B = (nsim - ndata - 2.0) / ((nsim - ndata - 1.0) * (nsim - ndata - 4.0))
        factor = (ndata - npar) * B
        m = npar + 2.0 + (nsim - 1.0 + factor) / (1.0 + factor)
        chi2 = float(diff @ inv_cov @ diff)
        return -0.5 * m * np.log1p(chi2 / (nsim - 1.0)), chi2, inv_cov

    if nsim > 0:
        if nsim <= ndata + 2:
            raise ValueError(f"{nsim} realizations cannot constrain {ndata} data points")
        inv_cov = inv_cov * hartlap_factor(nsim, ndata)
        if npar > 0:
            inv_cov = inv_cov * dodelson_schneider_factor(nsim, ndata, npar)
    chi2 = float(diff @ inv_cov @ diff)
    return -0.5 * chi2, chi2, inv_cov
