# Independent Geographic Extension Lock (Frozen Protocol)

- Base protocol: `clean_experiments/experiment_M_realpilot_v1_frozen.py`
- Feature set and thresholds: unchanged from frozen v1
- Fixed model/evaluation: `Ridge(alpha=10.0)`, leave-one-event-out CV, `n_perm=499`
- Event set: `clean_experiments/pilot_events_realpilot_v1_independent_geographic_southwest_2024.csv`
- Data products: MRMS `MultiSensor_QPE_01H_Pass2_00.00`, ABI C13, GLM LCFA

This run is an out-of-sample geographic robustness check with no retuning.
