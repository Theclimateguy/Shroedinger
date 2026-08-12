#!/usr/bin/env python3
"""Phase-5b: universality-as-collapse test for the curvature profile.

Protocol: docs/PROTOCOL_PHASE5_GAUGE_STRUCTURE.md (frozen 2026-08-12).
Pooled model: log Fnorm_{rw,b} = alpha*log(1+cape) + beta*land +
gamma*log(1+orog_std) + delta_b. PASS iff R^2 >= 0.5 AND residual
clustering diff < 0.5 x raw diff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B2b_heldout_invariants import (
        _perm_test,
        _zscore_cols,
    )
    from clean_experiments.experiment_B4_curvature_invariant import (
        region_covariates,
    )
except ImportError:
    from experiment_B2b_heldout_invariants import _perm_test, _zscore_cols  # type: ignore
    from experiment_B4_curvature_invariant import region_covariates  # type: ignore

SEED = 20260811
B4_DIR = _HERE / "results" / "experiment_B4_curvature_invariant"
B3_DIR = _HERE / "results" / "experiment_B3_scattering_benchmark"
OUT_DIR = _HERE / "results" / "experiment_B5_gauge_structure"


def main() -> None:
    rows = []
    for f in sorted(B4_DIR.glob("*__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append((r["region"], r["Fnorm_profile"]))
    for f in sorted(B3_DIR.glob("*__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "E2_curvature" in r:
            rows.append((r["region"], r["E2_curvature"]["Fnorm_profile"]))

    labels = np.asarray([r for r, _ in rows])
    prof = np.log(np.maximum(np.asarray([p for _, p in rows]), 1e-12))
    n, nb = prof.shape
    cov = region_covariates(Path("data/b4cov") if Path("data/b4cov").exists()
                            else _HERE.parent / "data/b4cov")

    x_cov = np.asarray([
        [np.log(1 + cov[r]["cape_mean"]), cov[r]["land_frac"],
         np.log(1 + cov[r]["orog_std"])] for r in labels
    ])
    # Design: 3 covariates (shared across bands) + band offsets
    y = prof.ravel()
    xc = np.repeat(x_cov, nb, axis=0)
    band_dummies = np.tile(np.eye(nb), (n, 1))
    a = np.hstack([xc, band_dummies])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    pred = a @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    resid = (y - pred).reshape(n, nb)
    rng = np.random.default_rng(SEED)
    raw = _perm_test(_zscore_cols(prof), labels, rng)
    res = _perm_test(_zscore_cols(resid), labels, rng)
    shrunk = bool(res["diff"] < 0.5 * raw["diff"])
    c5b = bool(r2 >= 0.5 and shrunk)

    out = {
        "pooled_R2": r2,
        "alpha_cape": float(coef[0]),
        "beta_land": float(coef[1]),
        "gamma_orog": float(coef[2]),
        "raw_clustering": raw,
        "residual_clustering": res,
        "shrinkage_ok": shrunk,
        "C5b_pass": c5b,
        "n_profiles": n,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "b5b_collapse.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
