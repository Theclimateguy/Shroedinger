# Research Task: Regime Detection Under Frozen Observable Bridge

## Objective
Formalize the M-realpilot family as a conditional detectability study (not universal validity),
using one fixed protocol across all runs and reporting outcome vectors on a common perimeter.

## Frozen Operator
P_frozen = (fixed features, Ridge alpha=10.0, LOEO-CV, n_perm=499, target=next_p95).

## Outcome Vector
y = (mean_gain, p_time, p_event, q_plus, active_minus_calm, PASS_ALL).

## Main Hypothesis
There exists a non-empty regime set where frozen protocol detects structural signal.
Null/negative runs constrain applicability perimeter rather than automatically falsify the bridge.

## Decision Labels
- Detected
- Compatible but undetected
- Bridge tension

Protocol hash match: True (aded0e49e825a318a2de07f49faa9c877277d2e76f223277495bdcebfbe8f3f2).