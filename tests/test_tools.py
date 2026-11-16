import numpy as np
import pytest

from desy6_3pcf import fisher
from desy6_3pcf.chains import read_chain, vector_columns
from desy6_3pcf.design import load_design, sobol_samples, write_cosmosis_list


def test_design_to_cosmosis_list(tmp_path):
    designs = tmp_path / "designs.yaml"
    designs.write_text(
        "small: &small\n  cosmological_parameters--omega_m: [0.1, 0.5]\n"
        "big:\n  <<: *small\n  cosmological_parameters--S_8: [0.6, 0.9]\n"
    )
    ranges = load_design(designs, "big")
    assert list(ranges) == ["cosmological_parameters--omega_m", "cosmological_parameters--s_8"]
    samples = sobol_samples(ranges, 64)
    assert samples.shape == (64, 2)
    assert samples[:, 0].min() >= 0.1 and samples[:, 1].max() <= 0.9
    # extending a design continues the same sequence
    assert np.allclose(sobol_samples(ranges, 32, skip=32), samples[32:])
    with pytest.raises(KeyError):
        load_design(designs, "missing")

    filename = tmp_path / "list.txt"
    write_cosmosis_list(filename, list(ranges), samples)
    names, back = read_chain(filename)
    assert names == list(ranges) and np.allclose(back, samples)


def test_vector_columns(tmp_path):
    filename = tmp_path / "chain.txt"
    filename.write_text("#cosmological_parameters--omega_m\tMAP3--VECTOR_1\tMAP3--VECTOR_0\tpost\n0.3 2.0 1.0 0.0\n")
    names, samples = read_chain(filename)
    assert samples[0, vector_columns(names, "map3--vector")].tolist() == [1.0, 2.0]
    with pytest.raises(KeyError):
        vector_columns(names, "map3--nothing")


def test_fisher_forecasts(tmp_path):
    names = ["cosmological_parameters--omega_m", "cosmological_parameters--s_8", "shear_calibration_parameters--m1"]
    cov = np.array([[4e-4, -1e-4, 0.0], [-1e-4, 1e-4, 2e-4], [0.0, 2e-4, 1e-2]])
    filename = tmp_path / "fisher.txt"
    np.savetxt(filename, np.linalg.inv(cov), header="\t".join(names))
    read_names, matrix = fisher.read_fisher(filename)
    assert read_names == names

    errors = fisher.marginal_errors(names, matrix)
    assert np.isclose(errors[names[0]], 0.02) and np.isclose(errors[names[2]], 0.1)
    assert all(fisher.conditional_errors(names, matrix)[n] <= errors[n] for n in names)
    assert np.isclose(fisher.figure_of_merit(names, matrix, names[0], names[1]),
                      1 / np.sqrt(np.linalg.det(cov[:2, :2])))

    tighter = fisher.add_gaussian_priors(names, matrix, {names[2]: 0.01})
    assert fisher.marginal_errors(names, tighter)[names[1]] < errors[names[1]]
    fixed_names, fixed = fisher.fix_parameters(names, matrix, [names[2]])
    assert fixed_names == names[:2] and fixed.shape == (2, 2)

    x, y = fisher.ellipse(names, matrix, names[0], names[1], nsigma=1.0)
    delta = np.array([x, y])
    chi2 = np.einsum("in,ij,jn->n", delta, np.linalg.inv(cov[:2, :2]), delta)
    assert np.allclose(chi2, 2.30)
