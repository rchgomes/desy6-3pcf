import numpy as np
import pytest

from desy6_3pcf.likelihood import dodelson_schneider_factor, hartlap_factor, log_likelihood


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 30))
    return rng.normal(size=30), np.linalg.inv(a @ a.T + 30 * np.eye(30))


def test_gaussian(problem):
    diff, icov = problem
    like, chi2, out = log_likelihood(diff, icov)
    assert np.isclose(chi2, diff @ icov @ diff) and np.isclose(like, -0.5 * chi2) and out is icov


def test_hartlap_and_dodelson_schneider(problem):
    diff, icov = problem
    chi2 = diff @ icov @ diff
    _, hartlap, _ = log_likelihood(diff, icov, nsim=400)
    assert np.isclose(hartlap, chi2 * (400 - 30 - 2) / 399) and hartlap < chi2
    _, both, scaled = log_likelihood(diff, icov, nsim=400, npar=10)
    factor = hartlap_factor(400, 30) * dodelson_schneider_factor(400, 30, 10)
    assert np.isclose(both, chi2 * factor) and np.allclose(scaled, icov * factor)
    assert both < hartlap


def test_percival_approaches_gaussian_for_many_simulations(problem):
    diff, icov = problem
    gaussian, _, _ = log_likelihood(diff, icov)
    percival, chi2, _ = log_likelihood(diff, icov, nsim=10**7, npar=10, percival=True)
    assert np.isclose(percival, gaussian, rtol=1e-4) and np.isclose(chi2, diff @ icov @ diff)


def test_invalid_settings(problem):
    diff, icov = problem
    with pytest.raises(ValueError):
        log_likelihood(diff, icov, nsim=31)
    with pytest.raises(ValueError):
        log_likelihood(diff, icov, percival=True)
