import numpy as np
import pytest

from desy6_3pcf import binning
from desy6_3pcf.datavector import Map3Data


@pytest.fixture
def data():
    d = Map3Data.from_bins(binning.tomographic_triplets(4), binning.filter_triplets())
    d.value[:] = np.arange(d.size, dtype=float)
    d.cov = np.diag(1.0 + np.arange(d.size))
    d.nsim = 500
    return d


def test_y6_layout(data):
    assert data.size == 100
    assert len(data.tomographic_triplets()) == 20
    assert np.all(data.filters((1, 1, 1))[0] == [3, 5, 10, 20, 40])


def test_fits_roundtrip(data, tmp_path):
    filename = tmp_path / "map3.fits"
    data.to_fits(filename)
    back = Map3Data.from_fits(filename)
    assert np.array_equal(back.z, data.z) and np.allclose(back.theta, data.theta)
    assert np.allclose(back.value, data.value) and np.allclose(back.cov, data.cov)
    assert back.nsim == 500 and back.name == "map3"


def test_indices_ignore_redshift_order(data):
    filters = np.array([[10.0, 40.0]] * 3)
    where = data.indices((3, 1, 2), filters)
    assert np.all(data.z[:, where] == [[1], [2], [3]])
    assert np.all(data.theta[0, where] == [10.0, 40.0])
    with pytest.raises(KeyError):
        data.indices((1, 1, 1), [[7.0], [7.0], [7.0]])


def test_select_applies_scale_cut_to_covariance(data):
    keep = data.theta_mask(5.0, 20.0)
    cov = data.cov[np.ix_(keep, keep)].copy()
    data.select(keep)
    assert data.size == 60 and np.allclose(data.cov, cov)
    assert data.theta.min() == 5.0 and data.theta.max() == 20.0
