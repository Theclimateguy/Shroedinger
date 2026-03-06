# M-realpilot v1 Frozen: Independent Extension Check

## Locked baseline vs independent extension
- Baseline locked (expanded): mean_gain=0.001576, PASS_ALL=True
- Independent seasonal extension: mean_gain=-0.009710, PASS_ALL=False

## Satellite component robustness (independent extension)
- ABI-only mean_gain=-0.000609
- ABI+GLM mean_gain=-0.009710
- Delta (ABI+GLM - ABI-only)=-0.009101

## Interpretation
- In this independent slice, structural satellite signal is not detected under frozen protocol.
- This is evidence against pilot-time overfitting claims for the positive baseline run.