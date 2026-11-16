"""Emulated <Map^3>: a drop-in replacement of ``fastnc_interface`` + ``nc_to_map3``.

The networks predict <Map^3> on thin redshift slices; this module weights
the slices with the line-of-sight kernels of each tomographic triplet and
integrates.  Source redshift distributions (and their nuisance parameters)
and NLA therefore never enter the networks.

Model
-----
``ia_model = none``: lensing only, from the reference (matter-bispectrum)
emulator.

``ia_model = nla``: as above with the NLA term added to the lensing kernels.

``ia_model = tatt``: additionally the seven TATT components, each weighted
by lensing kernels at the ``d`` positions and source distributions at the
``E`` positions.  If the TATT emulators were trained with
``remove_alignment = T`` the NLA kernel term is kept in the reference part.

Options
-------
reference : str
    Prefix of the reference emulator (``<prefix>.json`` / ``<prefix>.pkl``).
    Use the LCDM or the wCDM network according to the model being sampled.
ia_model : ``none``, ``nla`` (default) or ``tatt``
tatt_template : str
    Prefix of the TATT emulators with ``{component}`` in place of the name,
    e.g. ``.../tatt_{component}_wcdm``.
nz_upsampling : int (300)
    Number of redshifts on which the slices are interpolated before the
    line-of-sight integration.
out_of_range : ``reject`` (default), ``warn`` or ``ignore``
    What to do outside the training range of a network; ``reject`` makes
    the pipeline fail for that point, i.e. zero posterior.
"""
import numpy as np
from cosmosis.datablock import option_section

from desy6_3pcf.binning import combination_name
from desy6_3pcf.block import MAP3, read_theory_inputs, sample_combinations
from desy6_3pcf.emulator import Map3SliceEmulator, integrate_slices
from desy6_3pcf.fastnc_backend import TATT_COMPONENTS, LensingKernels


def setup(options):
    ia_model = options.get_string(option_section, "ia_model", "nla")
    if ia_model not in ("none", "nla", "tatt"):
        raise ValueError("ia_model must be none, nla or tatt")
    config = {
        "ia_model": ia_model,
        "reference": Map3SliceEmulator.load(options.get_string(option_section, "reference")),
        "tatt": {},
        "kernels": LensingKernels(),
        "nz_upsampling": options.get_int(option_section, "nz_upsampling", 300),
        "out_of_range": options.get_string(option_section, "out_of_range", "reject"),
    }
    if ia_model == "tatt":
        template = options.get_string(option_section, "tatt_template")
        for component in TATT_COMPONENTS:
            config["tatt"][component] = Map3SliceEmulator.load(template.format(component=component))
        removed = {bool(e.info.get("remove_alignment", True)) for e in config["tatt"].values()}
        if len(removed) != 1:
            raise ValueError("the TATT emulators were trained with inconsistent remove_alignment settings")
        config["nla_in_reference"] = removed.pop()
    else:
        config["nla_in_reference"] = ia_model == "nla"
    return config


def _predict(block, emulator, label, policy):
    params = np.array([block[tuple(p.split("--"))] for p in emulator.parameters])
    if policy != "ignore" and not emulator.in_range(params)[0]:
        outside = [
            f"{p}={v:.4g}"
            for p, v, lo, hi in zip(emulator.parameters, params, emulator.param_min, emulator.param_max)
            if not lo <= v <= hi
        ]
        message = f"[map3_emulator] {label} emulator outside its training range: {', '.join(outside)}"
        if policy == "reject":
            raise ValueError(message)
        print(message)
    return emulator.predict(params)


def execute(block, config):
    policy = config["out_of_range"]
    try:
        reference = _predict(block, config["reference"], "reference", policy)
        tatt = {c: _predict(block, e, c, policy) for c, e in config["tatt"].items()}
    except ValueError as error:
        print(error)
        return 1

    kernels = config["kernels"]
    kernels.update(read_theory_inputs(block, need_power=False))
    chi = kernels.chi
    nup = config["nz_upsampling"]

    def los_integral(emulator, slices, kernel_functions):
        def weight(z):
            c = chi(z)
            return np.prod([k(c) for k in kernel_functions], axis=0) * (1.0 + z) ** 3 / c

        return integrate_slices(emulator.z_slices, slices, chi, weight, nup)

    for scomb in sample_combinations(block):
        name = combination_name(scomb)
        filters = block[MAP3, f"filters_{name}"]

        emulator = config["reference"]
        lensing = [lambda c, s=s: kernels.lensing(s, c, nla=config["nla_in_reference"]) for s in scomb]
        map3 = los_integral(emulator, reference[:, emulator.filter_indices(filters)], lensing)

        for component, emulator in config["tatt"].items():
            mixed = [
                (lambda c, s=s: kernels.source(s, c)) if letter == "E" else (lambda c, s=s: kernels.lensing(s, c))
                for s, letter in zip(scomb, component)
            ]
            map3 = map3 + los_integral(emulator, tatt[component][:, emulator.filter_indices(filters)], mixed)

        block[MAP3, f"map3-bin_{name}"] = map3
    return 0


def cleanup(config):
    pass
