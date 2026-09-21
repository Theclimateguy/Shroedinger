#!/usr/bin/env python3
"""Phase 25: hydroclimatic candidates for the global P-map residual.

Protocol: docs/PROTOCOL_PHASE25_RESIDUAL_CANDIDATES.md (frozen 2026-08-24
before the candidate fields were downloaded). Candidates (tile means on the
frozen Phase-20 Arm-B grid, 902 tiles):

  monsoon_amp  log10(annual range of the monthly tp climatology + 0.1 mm/day)
  conv_frac    climatological cp/tp, clipped to [0, 1]
  shear        climatological mean of monthly |V850 - V500|
  pw           climatological mean total column water vapour

H25a: LOSO-sector increment of the 4-candidate block over the frozen
baseline [8 Phase-20 covariates + 4 intermittency statistics]; null = 999
longitude rotations (>= 30 deg) of the candidate block only.
H25b: Spearman of the AUDIT-4 residual against each candidate (rotation
null per candidate). C25-1: target consistency with the committed Arm-B
map. C25-2: spectral-slope placebo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B20_armB_global_map import (
        ALL_COV, OROG_EXCL_M, SEASONS, rotate_cov, tile_covariates, tile_grid,
    )
except ImportError:
    from experiment_B20_armB_global_map import (  # type: ignore
        ALL_COV, OROG_EXCL_M, SEASONS, rotate_cov, tile_covariates, tile_grid,
    )

SEED = 20260824
N_ROT = 999
MIN_ROT_DEG = 30.0
HYDRO = Path("data/b25hydro")
A2B = _HERE / "results" / "experiment_A2b_splithalf"
ARMB = _HERE / "results" / "experiment_B20_armB_global_map"
OUT = _HERE / "results" / "experiment_B25_residual_candidates"
HALVES = ("odd", "even")
INT_COLS = ("P_cascade", "INT_sig2", "INT_flat", "INT_mu")


def z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)


def tile_mean(field2d: np.ndarray, glat: np.ndarray, glon: np.ndarray,
              t: dict) -> float:
    la = (glat >= t["lat0"]) & (glat <= t["lat1"])
    lom = (glon + 360.0) % 360.0
    lo0, lo1 = (t["lon0"] + 360.0) % 360.0, (t["lon1"] + 360.0) % 360.0
    lo = (lom >= lo0) & (lom <= lo1) if lo0 <= lo1 else (lom >= lo0) | (lom <= lo1)
    return float(np.nanmean(field2d[np.ix_(la, lo)]))


def candidate_fields() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    sub = HYDRO / "_x_era5_monthly_single_hydro"
    ds1 = xr.open_dataset(sub / "data_stream-moda_stepType-avgad.nc")
    ds2 = xr.open_dataset(sub / "data_stream-moda_stepType-avgua.nc")
    dsw = xr.open_dataset(HYDRO / "era5_monthly_wind_850_500.nc")
    glat = ds1["latitude"].values
    glon = ds1["longitude"].values

    month = ds1["valid_time"].dt.month
    tp_clim = ds1["tp"].groupby(month).mean("valid_time").values * 1000.0  # mm/day
    monsoon = np.log10(tp_clim.max(0) - tp_clim.min(0) + 0.1)

    tp_mean = ds1["tp"].mean("valid_time").values
    cp_mean = ds1["cp"].mean("valid_time").values
    conv = np.clip(cp_mean / np.maximum(tp_mean, 1e-9), 0.0, 1.0)

    pw = ds2["tcwv"].mean("valid_time").values

    lev = dsw["pressure_level"].values
    i850 = int(np.argmin(np.abs(lev - 850))); i500 = int(np.argmin(np.abs(lev - 500)))
    du = dsw["u"].isel(pressure_level=i850) - dsw["u"].isel(pressure_level=i500)
    dv = dsw["v"].isel(pressure_level=i850) - dsw["v"].isel(pressure_level=i500)
    shear = np.sqrt(du ** 2 + dv ** 2).mean("valid_time").values

    for d in (ds1, ds2, dsw):
        d.close()
    return {"monsoon_amp": (monsoon, glat, glon),
            "conv_frac": (conv, glat, glon),
            "shear": (shear, glat, glon),
            "pw": (pw, glat, glon)}


def ols_pred(Xf: np.ndarray, yf: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(Xf)), Xf])
    c, *_ = np.linalg.lstsq(A, yf, rcond=None)
    return np.column_stack([np.ones(len(Xp)), Xp]) @ c


def loso_r2(X: np.ndarray, y: np.ndarray, sector: np.ndarray) -> float:
    press, ss = 0.0, ((y - y.mean()) ** 2).sum()
    for s in set(sector.tolist()):
        m = sector == s
        press += ((y[m] - ols_pred(X[~m], y[~m], X[m])) ** 2).sum()
    return 1.0 - press / ss


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    grid_all = tile_grid()
    grid = {t["tid"]: t for t in grid_all}
    sh = {(s, h): {r["tid"]: r for r in json.loads(
        (A2B / f"tiles_{s}2023_{h}.json").read_text(encoding="utf-8"))["tiles"]}
        for s in SEASONS for h in HALVES}
    armb = {s: {r["tid"]: r for r in json.loads(
        (ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))["tiles"]}
        for s in SEASONS}
    common = sorted(set.intersection(*[set(v) for v in sh.values()]))
    cov = tile_covariates([grid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    kept_tiles = [grid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}
    sector = np.array([int(((grid[t]["lon_c"] + 180.0) % 360.0) // 60) for t in keep])

    # baseline
    Xz = {n: z([cov[t][n] for t in keep]) for n in ALL_COV if n != "eke_syn"}
    Xz["eke_syn"] = z([np.mean([armb[s][t]["eke_syn"] for s in SEASONS])
                       for t in keep])
    X_cov = np.column_stack([Xz[n] for n in ALL_COV])
    INT_full = np.column_stack([z([np.mean([sh[(s, h)][t][f"{k}_anch_mean"]
                                            for s in SEASONS for h in HALVES])
                                   for t in keep]) for k in INT_COLS])
    base = np.column_stack([X_cov, INT_full])

    # targets
    y = np.array([np.mean([sh[(s, h)][t]["P_anch_mean"]
                           for s in SEASONS for h in HALVES]) for t in keep])
    y_map = np.array([np.mean([armb[s][t]["P_fine_anch_mean"] for s in SEASONS])
                      for t in keep])
    y_slope = np.array([np.mean([armb[s][t]["F_fine"]["slope"] for s in SEASONS])
                        for t in keep])

    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE25_RESIDUAL_CANDIDATES.md",
                 "n_tiles": len(keep)}
    res["C25_1_target_vs_committed_map_rho"] = float(
        spearmanr(y, y_map).statistic)

    # candidates
    fields = candidate_fields()
    cand_names = ["monsoon_amp", "conv_frac", "shear", "pw"]
    C_raw = {n: np.array([tile_mean(*fields[n], grid[t]) for t in keep])
             for n in cand_names}
    C = np.column_stack([z(C_raw[n]) for n in cand_names])

    # ---- H25a: increment with rotation null ------------------------------
    r2_base = loso_r2(base, y, sector)
    r2_full = loso_r2(np.column_stack([base, C]), y, sector)
    incr = r2_full - r2_base

    def rand_offsets(n: int) -> np.ndarray:
        # uniform over [30, 330] degrees
        return MIN_ROT_DEG + rng.random(n) * (360.0 - 2 * MIN_ROT_DEG)

    null = np.empty(N_ROT)
    offs = rand_offsets(N_ROT)
    for i, off in enumerate(offs):
        Crot = rotate_cov(C, kept_tiles, float(off), tid_index)
        null[i] = loso_r2(np.column_stack([base, Crot]), y, sector) - r2_base
    p_a = float((1 + np.sum(null >= incr)) / (N_ROT + 1))
    if incr >= 0.05 and p_a <= 0.01:
        verdict = "RESIDUAL_ATTRIBUTED"
    elif incr >= 0.03 and p_a < 0.05:
        verdict = "CANDIDATES_ADD"
    else:
        verdict = "NEGATIVE"
    res["H25a"] = {"r2_base": float(r2_base), "r2_full": float(r2_full),
                   "increment": float(incr), "p_rotation": p_a,
                   "null_q95": float(np.quantile(null, 0.95)),
                   "verdict": verdict}

    # ---- H25b: residual correlations -------------------------------------
    resid = {}
    for s in SEASONS:
        rf = {}
        for h, ho in (("odd", "even"), ("even", "odd")):
            yv = np.array([sh[(s, h)][t]["P_anch_mean"] for t in keep])
            yo = np.array([sh[(s, ho)][t]["P_anch_mean"] for t in keep])
            XI = np.column_stack([X_cov] + [z(
                [sh[(s, ho)][t][f"{k}_anch_mean"] for t in keep])[:, None]
                for k in INT_COLS])
            rf[h] = yv - ols_pred(XI, yo, XI)
        resid[s] = 0.5 * (rf["odd"] + rf["even"])
    r_mean = 0.5 * (resid["JFM"] + resid["JAS"])

    res["H25b"] = {}
    offs_b = rand_offsets(N_ROT)
    for j, n in enumerate(cand_names):
        rho = float(spearmanr(r_mean, C[:, j]).statistic)
        nullr = np.empty(N_ROT)
        col = C[:, j][:, None]
        for i, off in enumerate(offs_b):
            nullr[i] = spearmanr(
                r_mean, rotate_cov(col, kept_tiles, float(off), tid_index)[:, 0]
            ).statistic
        pj = float((1 + np.sum(np.abs(nullr) >= abs(rho))) / (N_ROT + 1))
        res["H25b"][n] = {"rho_residual": rho, "p_rotation_two_sided": pj}

    # ---- C25-2: spectral-slope placebo -----------------------------------
    r2b_s = loso_r2(base, y_slope, sector)
    r2f_s = loso_r2(np.column_stack([base, C]), y_slope, sector)
    res["C25_2_slope_placebo"] = {"r2_base": float(r2b_s),
                                  "r2_full": float(r2f_s),
                                  "increment": float(r2f_s - r2b_s)}

    res["candidate_loadings_vs_P"] = {
        n: float(spearmanr(y, C_raw[n]).statistic) for n in cand_names}

    (OUT / "summary.json").write_text(json.dumps(res, indent=1),
                                      encoding="utf-8")
    print(json.dumps(res, indent=1))
    print("PHASE25_VERDICT:", verdict)


if __name__ == "__main__":
    main()
