"""Checks of the fastnc backend that do not need a bispectrum evaluation."""
import numpy as np
import pytest

pytest.importorskip("fastnc")

from desy6_3pcf.fastnc_backend import LensingKernels, NaturalComponents, TheoryInputs  # noqa: E402


def inputs(a1):
    z = np.linspace(0.0, 2.5, 251)
    zg = np.linspace(0.0, 4.0, 401)
    return TheoryInputs(
        h0=0.67, omega_m=0.3, omega_b=0.049, sigma_8=0.8, n_s=0.96,
        z_growth=zg, d_growth=1.0 / (1.0 + zg),
        nz_z=z, nz=[z**2 * np.exp(-((z / z0) ** 1.5)) for z0 in (0.35, 0.7)],
        ia={"a1": a1, "alpha1": 0.0, "z_piv": 0.3},
    )


def test_nla_only_enters_the_kernels_that_ask_for_it():
    # Regression test: fastnc shares its IA switch between instances unless
    # the backend gives each one its own.
    kernels = LensingKernels()
    kernels.update(inputs(a1=1.0))
    chi = kernels.chi(np.linspace(0.2, 1.0, 9))
    pure, with_nla = kernels.lensing(1, chi), kernels.lensing(1, chi, nla=True)
    assert np.all(pure > 0)
    assert np.all(with_nla < pure)  # positive a1 suppresses the signal

    kernels.update(inputs(a1=0.0))
    assert np.allclose(kernels.lensing(1, chi, nla=True), kernels.lensing(1, chi))
    assert np.allclose(kernels.lensing(1, chi), pure)


def test_source_kernel_is_normalised():
    kernels = LensingKernels()
    kernels.update(inputs(a1=0.0))
    z = np.linspace(0.01, 2.49, 2000)
    chi = kernels.chi(z)
    # W = p(chi) / chi / (1 + z) with int p(chi) dchi = 1
    for sample in (1, 2):
        assert np.isclose(np.trapz(kernels.source(sample, chi) * chi * (1 + z), chi), 1.0, rtol=2e-3)


def test_invalid_settings_are_rejected():
    bins = (np.geomspace(1e-3, 1e-2, 3), 0.2, np.linspace(0.1, 3.0, 4))
    with pytest.raises(ValueError):
        NaturalComponents(*bins, bispectrum_model="unknown")
    with pytest.raises(ValueError):
        NaturalComponents(*bins, bispectrum_model="tatt", tatt_component="EdX")
    with pytest.raises(ValueError):
        NaturalComponents(*bins, bispectrum_model="tatt", baryon_model="takahashi")
