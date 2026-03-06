# P2 l=8 Diagnostic Block (Finalized)

## Scope
Narrow diagnostics for dense `l=8` transfer in calibrated P2 (C009), with five checks:

1. matched-event control (sparse vs dense on same event pool)
2. local scalar ablation (`decoherence_alpha`, `w_occ`, `lambda_scale_power`)
3. operator attribution (`occ/sq/log/grad`)
4. resolution scan (`mrms_downsample`, active threshold)
5. regime split robustness

## Key outcomes

- Matched-event test: dense `l=8` degradation is not explained by event-pool shift alone.
- Scalar-only retuning: no recovery to stable significance.
- Operator attribution: strongest lift appears when dropping `sq` (diagnostic result).
- Resolution scan: `downsample=16, threshold=2.5` recovers borderline-significant gain.
- Regime split: degradation concentrates in calm regime; active/transition remain positive.

## Decision table

See: `l8_diagnostic_decision_table.csv`

- `dense_c009_l8`: `mae_gain=-1.330761e-07`, `perm_p=0.80`, `PASS_ALL=False`
- `operator_drop_sq`: `mae_gain=4.282879e-06`, `perm_p=0.02`, `PASS_ALL=True`
- `resolution_ds16_thr2.5`: `mae_gain=1.975572e-06`, `perm_p=0.05`, `PASS_ALL=True`
- `combo_drop_sq_ds16_thr2.5`: `mae_gain=6.980093e-06`, `perm_p=0.02`, `PASS_ALL=True`

## Program finalization

- Canonical Group A baseline remains C009.
- Retuned `l=8` configurations are retained as diagnostics, not promoted to baseline.
- Theory-motivated next step: P2-memory (retarded density matrix).

Primary finalization note:

- `report_P2_l8_resolution.md`
