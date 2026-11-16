import numpy as np
import pytest

from desy6_3pcf import binning
from desy6_3pcf.map3 import Map3Transform


def _bins(nphi=20):
    dlnt = binning.log_bin_width()
    theta = binning.theta_lower_edges() * binning.mean_separation_factor(dlnt)
    return theta, binning.phi_centres(nphi), dlnt, np.pi / nphi


def _fake_gamma(theta, phi, seed=1):
    rng = np.random.default_rng(seed)
    shape = (4, theta.size, theta.size, phi.size)
    smooth = 1e-6 / (theta[:, None, None] * theta[None, :, None])
    return smooth * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def test_linear_and_real():
    theta, phi, dlnt, dphi = _bins()
    transform = Map3Transform(theta, theta, phi, dlnt, dphi, binning.filter_triplets())
    a, b = _fake_gamma(theta, phi, 1), _fake_gamma(theta, phi, 2)
    out = transform(2 * a + 3 * b)
    assert out.shape == (5,) and np.isrealobj(out)
    assert np.allclose(out, 2 * transform(a) + 3 * transform(b))


def test_symmetric_in_second_and_third_aperture():
    theta, phi, dlnt, dphi = _bins()
    gamma = _fake_gamma(theta, phi)
    a = Map3Transform(theta, theta, phi, dlnt, dphi, [[5.0], [10.0], [20.0]])(gamma)
    b = Map3Transform(theta, theta, phi, dlnt, dphi, [[5.0], [20.0], [10.0]])(gamma)
    assert np.allclose(a, b)


def test_shape_is_checked():
    theta, phi, dlnt, dphi = _bins()
    transform = Map3Transform(theta, theta, phi, dlnt, dphi, binning.filter_triplets())
    with pytest.raises(ValueError):
        transform(np.zeros((4, 3, 3, 3)))


@pytest.mark.parametrize("R, k2, k3", [(5.0, 1.0, 1.0), (3.0, 2.0, 4.0), (10.0, 0.5, 4.0)])
def test_matches_treecorr(R, k2, k3):
    """Same integral as TreeCorr applies to a measured 3PCF."""
    treecorr = pytest.importorskip("treecorr")
    nphi = 20
    theta, phi, dlnt, dphi = _bins(nphi)
    gamma = _fake_gamma(theta, phi)
    ggg = treecorr.GGGCorrelation(
        min_sep=binning.MIN_SEP, max_sep=binning.MAX_SEP, nbins=binning.NBINS,
        nphi_bins=nphi, sep_units="arcmin", bin_type="LogSAS",
    )
    for i in range(4):
        getattr(ggg, f"gam{i}r")[...] = gamma[i].real
        getattr(ggg, f"gam{i}i")[...] = gamma[i].imag
        getattr(ggg, f"vargam{i}")[...] = 1.0
    d2, d3, p = np.meshgrid(theta, theta, phi, indexing="ij")
    ggg.meand2[...], ggg.meand3[...], ggg.meanphi[...] = d2, d3, p
    expected = ggg.calculateMap3(R=np.array([R]), k2=k2, k3=k3)[0][0]
    ours = Map3Transform(theta, theta, phi, dlnt, dphi, [[R], [k2 * R], [k3 * R]])(gamma)[0]
    assert np.isclose(ours, expected, rtol=1e-10)
