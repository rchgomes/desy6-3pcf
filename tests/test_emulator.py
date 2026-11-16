import json

import numpy as np
import pytest

from desy6_3pcf import binning
from desy6_3pcf.emulator import FeatureScaler, Map3SliceEmulator, integrate_slices


@pytest.mark.parametrize("transform", ["log10", "asinh"])
def test_scaler_roundtrip(transform):
    rng = np.random.default_rng(0)
    features = 10.0 ** rng.uniform(-12, -6, size=(50, 8))
    if transform == "asinh":
        features *= rng.choice([-1.0, 1.0], size=features.shape)
    scaler = FeatureScaler.fit(features, transform)
    encoded = scaler.encode(features)
    assert encoded.min() >= 0.0 and encoded.max() <= 1.0
    assert np.allclose(scaler.decode(encoded), features, rtol=1e-10)
    again = FeatureScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
    assert np.allclose(again.decode(encoded), features, rtol=1e-10)


def test_log_transform_rejects_signed_features():
    with pytest.raises(ValueError):
        FeatureScaler.fit(np.array([[1.0, -1.0], [2.0, 3.0]]), "log10")


class LinearNetwork:
    """Stand-in for a trained network: targets linear in the inputs."""

    def __init__(self, names, nfeature):
        self.names, self.nfeature = names, nfeature

    def predictions_np(self, inputs):
        total = sum(inputs[name] for name in self.names)
        return np.outer(total, np.linspace(0.1, 0.3, self.nfeature))


def make_emulator():
    names = ["cosmological_parameters--omega_m", "cosmological_parameters--s_8"]
    z = np.array([0.1, 0.5, 1.0])
    filters = binning.filter_triplets()
    scaler = FeatureScaler("log10", np.full(15, -10.0), np.full(15, -6.0))
    return Map3SliceEmulator(LinearNetwork(names, 15), names, [0.1, 0.6], [0.5, 1.0], z, filters, scaler)


def test_predict_shapes_and_ordering():
    emulator = make_emulator()
    slices = emulator.predict([0.3, 0.8])
    assert slices.shape == (3, 5)
    features = emulator.predict_features([[0.3, 0.8]])[0]
    assert features[1 * 5 + 2] == slices[1, 2]  # slice-major ordering
    assert np.isclose(features[0], 10.0 ** (-10.0 + 4.0 * 0.1))


def test_range_and_filter_lookup():
    emulator = make_emulator()
    assert emulator.in_range([[0.3, 0.8], [0.6, 0.8]]).tolist() == [True, False]
    assert emulator.filter_indices([[40.0, 3.0]] * 3).tolist() == [4, 0]
    with pytest.raises(KeyError):
        emulator.filter_indices([[3.0], [5.0], [10.0]])


def test_integrate_slices():
    z = np.linspace(0.0, 2.0, 21)
    slices = np.stack([z**2, 1 + z], axis=1)
    chi = lambda x: 3000.0 * x  # noqa: E731
    out = integrate_slices(z, slices, chi, lambda x: np.ones_like(x), n_upsample=0)
    assert np.allclose(out, 3000.0 * np.array([8.0 / 3.0, 4.0]), rtol=2e-3)
    # a kernel much narrower than the slice spacing needs the upsampling
    kernel = lambda x: np.exp(-0.5 * ((x - 1.03) / 0.02) ** 2)  # noqa: E731
    exact = 3000.0 * 0.02 * np.sqrt(2 * np.pi) * np.array([1.03**2 + 0.02**2, 2.03])
    fine = integrate_slices(z, slices, chi, kernel, n_upsample=2000)
    assert np.allclose(fine, exact, rtol=1e-3)
