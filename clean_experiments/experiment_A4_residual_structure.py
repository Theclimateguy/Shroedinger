#!/usr/bin/env python3
"""AUDIT-4: is the unexplained part of the global P map physics or noise?

Protocol: docs/PROTOCOL_AUDIT4_RESIDUAL_STRUCTURE.md (frozen 2026-08-19).
The last analysis before the papers; it asks one question and stops.

Uses the AUDIT-2b split-half shards and the frozen Phase-20 covariates.
No new data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B20_armB_global_map import (
        ALL_COV, OROG_EXCL_M, SEASONS, tile_covariates, tile_grid,
    )
except ImportError:
    from experiment_B20_armB_global_map import (  # type: ignore
        ALL_COV, OROG_EXCL_M, SEASONS, tile_covariates, tile_grid,
    )

A2B = _HERE / "results" / "experiment_A2b_splithalf"
ARMB = _HERE / "results" / "experiment_B20_armB_global_map"
OUT = _HERE / "results" / "experiment_A4_residual"
HALVES = ("odd", "even")
INT_COLS = ("P_cascade", "INT_sig2", "INT_flat", "INT_mu")


def z(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)


def ols_predict(X_fit, y_fit, X_pred):
    A = np.column_stack([np.ones(len(X_fit)), X_fit])
    coef, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    return np.column_stack([np.ones(len(X_pred)), X_pred]) @ coef


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = {t["tid"]: t for t in tile_grid()}
    sh = {(s, h): {r["tid"]: r for r in json.loads(
        (A2B / f"tiles_{s}2023_{h}.json").read_text(encoding="utf-8"))["tiles"]}
        for s in SEASONS for h in HALVES}
    armb = {s: {r["tid"]: r for r in json.loads(
        (ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))["tiles"]}
        for s in SEASONS}
    common = sorted(set.intersection(*[set(v) for v in sh.values()]))
    cov = tile_covariates([grid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]

    Xz = {n: z([cov[t][n] for t in keep]) for n in ALL_COV if n != "eke_syn"}
    Xz["eke_syn"] = z([np.mean([armb[s][t]["eke_syn"] for s in SEASONS]) for t in keep])
    X_cov = np.column_stack([Xz[n] for n in ALL_COV])

    def P(s, h):
        return np.array([sh[(s, h)][t]["P_anch_mean"] for t in keep])

    def INT(s, h):
        return np.column_stack([z([sh[(s, h)][t][f"{k}_anch_mean"] for t in keep])
                                for k in INT_COLS])

    res = {"protocol": "docs/PROTOCOL_AUDIT4_RESIDUAL_STRUCTURE.md",
           "n_tiles": len(keep)}

    # ---- R1: residual reliability ---------------------------------------
    res["R1"] = {}
    resid_store = {}
    for s in SEASONS:
        rc, rf = {}, {}
        for h, ho in (("odd", "even"), ("even", "odd")):
            y, yo = P(s, h), P(s, ho)
            rc[h] = y - ols_predict(X_cov, yo, X_cov)
            Xf_fit = np.column_stack([X_cov, INT(s, ho)])
            Xf_prd = np.column_stack([X_cov, INT(s, ho)])
            rf[h] = y - ols_predict(Xf_fit, yo, Xf_prd)
        def rel(d):
            r = float(np.corrcoef(d["odd"], d["even"])[0, 1])
            return {"r_half": r, "r_full_spearman_brown": 2 * r / (1 + r)}
        res["R1"][s] = {"residual_after_covariates": rel(rc),
                        "residual_after_covariates_and_intermittency": rel(rf)}
        resid_store[s] = {"cov": 0.5 * (rc["odd"] + rc["even"]),
                          "full": 0.5 * (rf["odd"] + rf["even"])}
    r_full_mean = float(np.mean([
        res["R1"][s]["residual_after_covariates_and_intermittency"]["r_full_spearman_brown"]
        for s in SEASONS]))
    res["R1"]["mean_r_full_after_intermittency"] = r_full_mean
    res["R1"]["mean_r_full_after_covariates"] = float(np.mean([
        res["R1"][s]["residual_after_covariates"]["r_full_spearman_brown"]
        for s in SEASONS]))
    res["VERDICT"] = ("RESIDUAL_IS_PHYSICAL" if r_full_mean >= 0.30
                      else "RESIDUAL_IS_NOISE")

    # ---- R2: where it lives ---------------------------------------------
    r_season_mean = 0.5 * (resid_store["JFM"]["full"] + resid_store["JAS"]["full"])
    order = np.argsort(r_season_mean)
    def tile_row(i):
        t = grid[keep[i]]
        return {"tid": t["tid"], "lat_c": round(t["lat_c"], 1),
                "lon_c": round(t["lon_c"], 1),
                "residual": round(float(r_season_mean[i]), 5)}
    res["R2"] = {
        "season_agreement_of_residual": float(spearmanr(
            resid_store["JFM"]["full"], resid_store["JAS"]["full"]).statistic),
        "top_negative": [tile_row(i) for i in order[:20]],
        "top_positive": [tile_row(i) for i in order[-20:][::-1]],
    }
    lat_c = np.array([grid[t]["lat_c"] for t in keep])
    prof = []
    for lo in np.arange(-60, 60, 10):
        m = (lat_c >= lo) & (lat_c < lo + 10)
        if m.any():
            prof.append({"lat_band": f"{int(lo)}..{int(lo)+10}",
                         "mean_residual": round(float(r_season_mean[m].mean()), 5),
                         "n": int(m.sum())})
    res["R2"]["zonal_profile"] = prof

    # ---- R3: is the residual spectral? ----------------------------------
    def armb_feat(name):
        return np.array([np.mean([armb[s][t]["F_fine"][name] for s in SEASONS])
                         for t in keep])
    res["R3"] = {name: float(spearmanr(armb_feat(name), r_season_mean).statistic)
                 for name in ("slope", "logvar_0", "logvar_1", "logvar_2", "logvar_3")}

    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    printable = {k: v for k, v in res.items() if k != "R2"}
    printable["R2_head"] = {"season_agreement": res["R2"]["season_agreement_of_residual"],
                            "zonal_profile": res["R2"]["zonal_profile"]}
    print(json.dumps(printable, indent=1))
    print("VERDICT:", res["VERDICT"])


if __name__ == "__main__":
    main()
