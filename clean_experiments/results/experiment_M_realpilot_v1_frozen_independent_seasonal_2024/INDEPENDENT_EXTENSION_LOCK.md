# Independent Extension Lock (Frozen Protocol)

- Base protocol: `clean_experiments/experiment_M_realpilot_v1_frozen.py`
- Feature set: unchanged from `v1` frozen
- Thresholds and ridge alpha: unchanged (`ridge_alpha=10.0`, fixed extraction thresholds)
- Event set: `clean_experiments/pilot_events_realpilot_v1_independent_seasonal_2024.csv`
- Data products: MRMS `MultiSensor_QPE_01H_Pass2_00.00`, ABI `C13`, GLM `GLM-L2-LCFA`

This run is intentionally out-of-sample relative to the locked positive run and is kept as a no-retuning robustness check.
