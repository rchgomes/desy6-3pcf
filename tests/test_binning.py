import numpy as np

from desy6_3pcf import binning


def test_y6_measurement_binning():
    edges = binning.theta_lower_edges()
    dlnt = binning.log_bin_width()
    assert edges.size == 25 and np.isclose(edges[0], 1.0)
    assert np.isclose(edges[-1] * np.exp(dlnt), 150.0)
    phi = binning.phi_centres()
    assert phi.size == 100 and np.isclose(phi[0] + phi[-1], np.pi)


def test_mean_separation_factor_lies_inside_the_bin():
    dlnt = binning.log_bin_width()
    factor = binning.mean_separation_factor(dlnt)
    assert np.exp(dlnt / 2) < factor < np.exp(dlnt)  # area weighting favours large separations
    assert np.isclose(binning.mean_separation_factor(1e-6), 1.0, atol=1e-5)


def test_filter_triplets():
    equal = binning.filter_triplets()
    assert equal.shape == (3, 5) and np.all(equal[0] == [3, 5, 10, 20, 40])
    every = binning.filter_triplets(mode="all")
    assert every.shape == (3, 35) and np.all(np.diff(every, axis=0) >= 0)


def test_tomographic_triplets():
    assert len(binning.tomographic_triplets(4)) == 20
    assert len(binning.tomographic_triplets(4, "auto")) == 4
    assert len(binning.tomographic_triplets(4, "cross")) == 16
    parsed = binning.parse_tomographic_triplets("auto 2,1,1", 4)
    assert [1, 1, 2] in parsed.tolist() and len(parsed) == 5


def test_z_slices_are_increasing_and_named_reproducibly():
    z = np.array(binning.Z_SLICES)
    assert np.all(np.diff(z) > 0) and z[-1] == 2.0
    assert binning.combination_name(z[0]) == "0.005"
    assert binning.combination_name((1, 2, 3)) == "1_2_3"
