# ERA5 Box Window-Sensitivity Scan

Scan setup:

- script: `clean_experiments/experiment_scale_gravity_einstein_box_era.py`
- input: `data/processed/wpwp_era5_2017_2019_experiment_M_input.nc`
- field set: `wind` (`u,v`)
- window length: `max_time=360` (about 3 months at 6-hour step)
- starts tested: `0, 120, 240, 360, 480, 600`
- permutations for slope p-value: `499`

Window-level summary (`time_start -> inertial R2_binned / p / PASS_ALL`):

- `0 -> 0.5201 / 0.012 / True`
- `120 -> 0.0361 / 0.294 / False`
- `240 -> 0.0856 / 0.156 / False`
- `360 -> 0.1201 / 0.142 / False`
- `480 -> 0.0046 / 0.328 / False`
- `600 -> 0.4541 / 0.016 / False`

Observed behavior:

- `time_start=0` passes all criteria.
- most middle windows do not pass inertial criteria (low `R2_binned`).
- `time_start=600` recovers high inertial `R2_binned`, but fails full combined pass criteria.

Interpretation:

- the Einstein-box atmospheric closure signal is present but regime-dependent;
- this matches the expectation that real-atmosphere nonstationarity and process mix
  can suppress the relation outside favorable windows.
