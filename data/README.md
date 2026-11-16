# data/

Nothing in this directory is tracked by git (except this file). The run files
look for the following by default; every path can be changed in the
`[DEFAULT]` section of a run file or with `-p DEFAULT.<key>=<path>`.

```
data/
  dv/
    2pt.fits                 key 2pt_file        shear 2pt functions + n(z) + covariance
    map3.fits                key map3_file       <Map^3> data vector + covariance
    cov_xip_xim_map3.fits    key joint_cov_file  joint covariance of xip, xim, map3
  emulators/                 key emulator_dir
    reference_lcdm.{pkl,json}
    reference_wcdm.{pkl,json}
    tatt_{ddE,dEd,Edd,dEE,EdE,EEd,EEE}.{pkl,json}
  training/
    samples/                 Sobol designs (make_training_samples.py)
    output/                  CosmoSIS list-sampler outputs (training sets)
```

## Formats

**`2pt.fits`** — the standard DES `twopoint` format: extensions `xip`, `xim`
(with angular bin edges), `nz_source` and `COVMAT`. The source redshift
distributions used for ⟨M_ap³⟩ are read from this file.

**`map3.fits`** — written by `scripts/make_map3_file.py` or the `save_map3`
module (`desy6_3pcf.datavector.Map3Data`):

* binary table `map3` with columns `BIN1 BIN2 BIN3` (tomographic bins,
  1-based, `BIN1 ≤ BIN2 ≤ BIN3`), `THETA1 THETA2 THETA3` (aperture radii in
  arcmin) and `VALUE`;
* image `COVMAT` with the covariance, in the row order of the table, and the
  header key `NSIM` (number of simulations it was estimated from; 0 if
  analytic).

Default ordering: triplets (1,1,1), (1,1,2), …, (4,4,4); within each, the
filters 3′, 5′, 10′, 20′, 40′.

**`cov_xip_xim_map3.fits`** — image `COVMAT` with header keys
`NAME_0 = xip`, `STRT_0 = 0`, `NAME_1 = xim`, `STRT_1 = …`, `NAME_2 = map3`,
`STRT_2 = …` giving the first index of each block. The blocks have the
lengths and ordering of the *uncut* data files; scale cuts are applied by the
pipeline.

## Conventions

* The pipeline predicts ⟨M_ap³⟩ as TreeCorr's `GGGCorrelation.calculateMap3`
  defines it for a `LogSAS` 3PCF, `(mcmm + mmcm + mmmc + mmm) / 4`, and it is
  positive for the lensing signal. Check the sign of the measured data vector
  against a prediction before using it: some DES Y3 measurement products
  carried the opposite sign and were negated when packed.
* The 3PCF is assumed to be measured with `min_sep = 1`, `max_sep = 150`
  (arcmin), `nbins = 25`, `nphi_bins = 100`, `bin_type = 'LogSAS'`.
