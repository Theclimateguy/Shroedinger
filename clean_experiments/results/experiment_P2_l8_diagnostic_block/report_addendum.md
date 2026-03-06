# P2 l=8 Diagnostic Addendum (Program Sync)

This addendum aligns the diagnostic outputs with Group A finalization policy.

## Matched-event control
- Sparse matched (`16` events) remains significant.
- Dense (`240x16`) loses `l=8` significance under baseline C009.
- Conclusion: effect is not only event-pool substitution.

## Operator and resolution sensitivity
- `drop_sq` and `threshold=2.5` recover significance at `l=8` in targeted tests.
- Combined targeted check reaches: `mae_gain=6.980093e-06`, `perm_p=0.02`, `event_positive_frac=0.875`, `PASS_ALL=True`.

## Regime localization
- Zero-aware split shows failure localized in calm regime.
- Active-core and transition regimes stay positive at event level.

## Final policy
- Keep C009 as canonical baseline in Group A pipeline.
- Keep retuned `l=8` results as diagnostic/theory evidence.
- Promote P2-memory (retarded density matrix) as the next theory-close iteration.

Reference memo:
- `report_P2_l8_resolution.md`
