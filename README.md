# desy6-3pcf

Third-order cosmic shear with DES Y6: modelling, emulation and inference for
the aperture-mass skewness ⟨M_ap³⟩ derived from the shear three-point
correlation function (3PCF), on its own and jointly with the shear two-point
functions.

The theory is computed with [fastnc](https://github.com/git-sunao/fastnc)
(Sugiyama et al. 2024, [arXiv:2407.01798](https://arxiv.org/abs/2407.01798)) and the pipelines run in
[CosmoSIS](https://cosmosis.readthedocs.io). 

## Analysis choices

| | DES Y6 |
|---|---|
| 3PCF binning (TreeCorr `LogSAS`) | 1′–150′, 25 log bins, 100 φ bins on [0, π] |
| Aperture radii | 3, 5, 10, 20, 40 arcmin |
| Filter combinations | equal apertures (configurable: all triplets) |
| Data vector | 20 tomographic triplets × 5 filters = 100 points |
| Scale cuts | none yet |
| 2pt compression | none yet|
| Models | ΛCDM, wCDM × NLA, TATT; optional baryonic feedback |

These numbers live in one place, [`desy6_3pcf/binning.py`](desy6_3pcf/binning.py),
and are the defaults of every module and script.

## Layout

```
desy6_3pcf/        Python package: all the logic, no CosmoSIS dependency
  binning.py         measurement bins, filters, emulator redshift slices
  map3.py            3PCF natural components -> <Map^3> (TreeCorr's integral)
  fastnc_backend.py  the only code that calls fastnc
  emulator.py        redshift-slice emulators and their line-of-sight integration
  datavector.py      <Map^3> FITS format
  likelihood.py      Gaussian / Hartlap / Dodelson-Schneider / Percival
  design.py, chains.py, fisher.py
modules/           thin CosmoSIS modules built on the package
config/
  common/            shared module settings (cosmology, cosmic shear, <Map^3>, samplers)
  values/, priors/   sampled parameters for {lcdm, wcdm} x {nla, tatt}
  emulator/          training-set designs and runs
  validation/        single-point predictions, synthetic data vectors
  fisher/            Fisher forecasts (brute-force fastnc or emulators)
  inference/         chains: joint 2pt + <Map^3>, <Map^3> only, 2pt only
scripts/           command-line tools (training, validation, Fisher summaries, file packing)
tests/             unit tests (pytest)
data/              your data products; not tracked, see data/README.md
```

## Installation

```bash
# CosmoSIS and its standard library: https://cosmosis.readthedocs.io
pip install cosmosis
git clone https://github.com/joezuntz/cosmosis-standard-library

# fastnc: the legacy "tatt" branch (v1.1.x), which has the TATT bispectra and baryons
pip install "fastnc @ git+https://github.com/git-sunao/fastnc@7b2336ce37a909bed51f07f32a2a4f096e71b6ce"

# this package
git clone https://github.com/rchgomes/desy6-3pcf
cd desy6-3pcf
pip install -e ".[emulator,plots,test]"     # emulator = cosmopower + tensorflow
```

Optional extras: `baryons` (baccoemu), `pixwin` (healpy).

Before running anything:

```bash
export COSMOSIS_STD_LIB=/path/to/cosmosis-standard-library
source setup-env.sh          # sets DESY6_3PCF_DIR
python -m pytest             # ~5 s
```

fastnc caches its mode-coupling functions in `~/.fastnc` (override with
`FASTNC_CACHE_DIR`); the first run takes a few extra minutes to fill it.

> **fastnc version.** fastnc `main` is now the v2 rewrite, which does not yet
> include intrinsic alignments or baryons. Every fastnc call is confined to
> `desy6_3pcf/fastnc_backend.py`, so moving to v2 means re-implementing the two
> classes in that file and nothing else.

## How the model is built

```
                     ┌──────────── brute force ────────────┐
cosmology ─ camb ─┬─ fastnc_interface ─ nc_to_map3 ─┐
                  │   Γ0..Γ3 on the 3PCF bins   <Map^3>   ├─ map3_m_bias ─ map3_like ─┐
                  └─ map3_emulator ───────────────────────┘                        ├─ joint_like
                      └──────────── emulated ─────────────┘        2pt_like ─ 2pt_mask ─┘
```

* **Brute force.** `fastnc_interface` predicts the natural components of the
  3PCF on the measurement bins; `nc_to_map3` integrates them with the same
  kernels and discretisation TreeCorr applies to the data, so the estimator's
  binning is part of the model. Seconds to minutes per evaluation.
* **Emulated.** Networks predict ⟨M_ap³⟩ on 64 thin redshift slices (0.005 ≤ z ≤ 2)
  as a function of cosmology (and TATT parameters); `map3_emulator` then does
  the line-of-sight integral with the lensing, NLA and source-distribution
  kernels. n(z), photo-z shifts, shear calibration and NLA never enter the
  networks. Milliseconds per evaluation.
* **Intrinsic alignments.** NLA is a term in the lensing kernels of the
  matter-bispectrum (bihalofit) part. TATT adds seven bispectrum components
  (`ddE dEd Edd dEE EdE EEd EEE`), each with its own emulator, with their
  tree-level NLA-like part removed so that nothing is counted twice.

## Workflow

Every run file takes its model from two `[DEFAULT]` keys that can be overridden
on the command line (use lower-case names): `cosmo` = `lcdm` | `wcdm` and
`ia` = `nla` | `tatt`, e.g. `-p DEFAULT.cosmo=wcdm DEFAULT.ia=tatt`.
Input files are set the same way (`2pt_file`, `map3_file`, `joint_cov_file`,
`emulator_dir`); their defaults point into `data/`, described in
[`data/README.md`](data/README.md).

### 0. Prepare the data files

```bash
python scripts/make_map3_file.py --map3 map3.npy --cov cov_map3.npy --nsim 1000 -o data/dv/map3.fits
python scripts/make_map3_file.py --joint-cov cov_joint.npy --nsim 1000 \
    --lengths xip=200 xim=200 map3=100 -o data/dv/cov_xip_xim_map3.fits
```

### 1. Fisher forecasts (before any chain)

Two interchangeable versions; the brute-force one needs no emulator. With NLA
a brute-force evaluation takes about a minute; brute-force TATT works too
(`-p DEFAULT.ia=tatt`) but costs tens of minutes and ~10 GB of memory per
evaluation, so TATT forecasts are better done with the emulators. Fisher
steps must stay inside the emulator training box.

```bash
# brute-force fastnc
mpirun -n 32 cosmosis --mpi config/fisher/joint-fastnc.ini
mpirun -n 32 cosmosis --mpi config/fisher/map3-fastnc.ini
# with the emulators
cosmosis config/fisher/joint-emulator.ini
cosmosis config/fisher/map3-emulator.ini

python scripts/fisher_summary.py output/fisher/fisher-map3-fastnc-lcdm-nla.txt \
    output/fisher/fisher-joint-fastnc-lcdm-nla.txt --labels Map3 2pt+Map3 \
    --plot output/fisher/forecast.png
```

`fisher_summary.py` reports marginalised and conditional errors and the
Ω_m–S_8 figure of merit. The Gaussian priors of the priors file are already
included by the CosmoSIS `fisher` sampler; `--add-prior` adds further ones and
`--fix` holds parameters fixed.

### 2. Emulator training sets

```bash
python scripts/make_training_samples.py reference_lcdm --n 2000 --chunks 4
python scripts/make_training_samples.py reference_wcdm --n 2000 --chunks 4
python scripts/make_training_samples.py tatt           --n 5000 --chunks 5

mpirun -n 64 cosmosis --mpi config/emulator/training-reference.ini -p DEFAULT.cosmo=lcdm DEFAULT.chunk=0
mpirun -n 64 cosmosis --mpi config/emulator/training-tatt.ini -p DEFAULT.component=ddE DEFAULT.chunk=0
# ... every chunk; cosmo = lcdm, wcdm; component = ddE dEd Edd dEE EdE EEd EEE
```

Parameter boxes are in [`config/emulator/designs.yaml`](config/emulator/designs.yaml).
One sample (64 slices) costs about 3 CPU-minutes. Designs are Sobol sequences
and can be extended later (`--skip`, `--first-chunk`).

### 3. Train and validate

```bash
python scripts/train_emulator.py reference_lcdm data/training/output/reference_lcdm_*.txt -o data/emulators/reference_lcdm
python scripts/train_emulator.py reference_wcdm data/training/output/reference_wcdm_*.txt -o data/emulators/reference_wcdm
for c in ddE dEd Edd dEE EdE EEd EEE; do
  python scripts/train_emulator.py tatt --component $c data/training/output/tatt_${c}_*.txt -o data/emulators/tatt_$c
done

python scripts/validate_emulator.py data/emulators/reference_lcdm      # network accuracy on held-out samples
```

End-to-end validation after the line-of-sight integration, against brute force:

```bash
cosmosis config/validation/predict.ini                       # fastnc -> output/validation/map3-fastnc-lcdm-nla.fits
cosmosis config/validation/predict.ini -p DEFAULT.theory=map3_emulator DEFAULT.tag=emulator \
    DEFAULT.map3_file=output/validation/map3-fastnc-lcdm-nla.fits
python scripts/compare_map3.py output/validation/map3-{fastnc,emulator}-lcdm-nla.fits
```

`predict.ini` with fastnc is also how synthetic (noiseless) 2pt and ⟨M_ap³⟩
data vectors are made.

### 4. Chains

```bash
mpirun -n 128 cosmosis --mpi config/inference/joint-2pt-map3.ini -p DEFAULT.cosmo=lcdm DEFAULT.ia=tatt
mpirun -n 128 cosmosis --mpi config/inference/map3-only.ini
mpirun -n 128 cosmosis --mpi config/inference/2pt-only.ini
```

The joint likelihood uses the full 2pt × ⟨M_ap³⟩ covariance. For
simulation-based covariances set `covariance_realizations` and
`free_parameters` (and optionally `percival = T`) in `[joint_like]` /
`[map3_like]`.

## Options worth knowing

| Where | Option | Meaning |
|---|---|---|
| `[map3_binning]` | `theta_range = 5 40` | scale cut on the aperture radii |
| | `sample_combinations = auto` / `1,1,2 2,3,4` | subset of tomographic triplets |
| | `filter_mode = all` | all 35 filter triplets instead of the 5 equal-aperture ones |
| `[fastnc]` | `baryon_model = takahashi` / `baccoemu` | baryonic feedback on the matter bispectrum (`[baryon_parameters]` `fb`, or `bacco_M_c`, …) |
| | `use_pixwin = T`, `nside` | HEALPix pixel window, for measurements made on maps |
| | `mean_separation_offset` | calibration of the effective bin centres against TreeCorr's `meand2/3` |
| `[map3_emulator]` | `out_of_range = reject` | points outside the training box get zero posterior |
| `[2pt_like]` | `angle_range_xip_1_1 = …` | 2pt scale cuts; propagated to the joint covariance automatically |

Baryonic feedback is available in the brute-force calculation only; emulating
it requires training sets generated with the corresponding `baryon_model`.

## Before using real data

* `config/priors/des-y6-priors.ini` contains **placeholder** shear-calibration
  and redshift priors.
* The sampling ranges in `config/values/` and the IA pivot redshift
  (`z_piv = 0.3`) are starting points, to be revisited with the Y6 analysis
  choices; the cosmological ranges must stay inside the emulator boxes.
* Check `nphi_bins`, `mean_separation_offset` and `use_pixwin` in `[fastnc]`
  against how the 3PCF is actually measured.
