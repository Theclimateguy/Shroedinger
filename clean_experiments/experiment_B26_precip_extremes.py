#!/usr/bin/env python3
"""Phase 26: does the coupling map carry information on the geography of
precipitation extremes beyond CAPE, moisture and mean wetness?

Protocol: docs/PROTOCOL_PHASE26_PRECIP_EXTREMES.md (frozen 2026-09-21,
before any daily precipitation field was read).
Data: data/b26precip/annual_YYYY.npz (download_b26_precip_daily.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B20_armB_global_map import (  # noqa: E402
    ALL_COV, OROG_EXCL_M, SEASONS, rotate_cov, tile_covariates, tile_grid,
)
from experiment_B25_residual_candidates import (  # noqa: E402
    candidate_fields, loso_r2, ols_pred, tile_mean, z,
)

SEED = 20260921
N_ROT = 999
MIN_ROT_DEG = 30.0
PRECIP = _HERE.parent / "data" / "b26precip"
ARMB = _HERE / "results" / "experiment_B20_armB_global_map"
OUT = _HERE / "results" / "experiment_B26_precip_extremes"
E1, E2 = range(1981, 1991), range(2015, 2025)
BOXES = {  # land tiles only; fixed in the protocol (lat0, lat1, lon0, lon1 in 0-360)
    "N inner Eurasia 50-58N 40-140E": (50, 58, 40, 140),
    "East-European plain 50-58N 30-60E": (50, 58, 30, 60),
    "Siberia 50-58N 60-140E": (50, 58, 60, 140),
    "Europe 44-52N 0-30E": (44, 52, 0, 30),
    "Mediterranean 32-46N 0-40E": (32, 46, 0, 40),
    "East/South China 20-34N 100-122E": (20, 34, 100, 122),
    "North America 44-58N 235-290E": (44, 58, 235, 290),
}


def epoch_mean(years) -> dict[str, np.ndarray] | None:
    files = [PRECIP / f"annual_{y}.npz" for y in years]
    if not all(f.exists() for f in files):
        return None
    acc = {"rx1day": 0.0, "prcptot": 0.0, "wetdays": 0.0}
    for f in files:
        d = np.load(f)
        for k in acc:
            acc[k] = acc[k] + d[k].astype(np.float64)
        lat, lon = d["lat"], d["lon"]
    return {**{k: v / len(files) for k, v in acc.items()}, "lat": lat, "lon": lon}


def partial_spearman(x, y, controls) -> float:
    rx, ry = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in controls])
    ex = rx - ols_pred(C, rx, C)
    ey = ry - ols_pred(C, ry, C)
    return float(np.corrcoef(ex, ey)[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    grid = {t["tid"]: t for t in tile_grid()}
    armb = {s: {r["tid"]: r for r in json.loads(
        (ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))["tiles"]} for s in SEASONS}
    common = sorted(set(armb["JFM"]) & set(armb["JAS"]))
    cov = tile_covariates([grid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M
            and all(np.isfinite(armb[s][t]["P_fine_anch_mean"]) for s in SEASONS)]
    kept_tiles = [grid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}
    sector = np.array([int(((grid[t]["lon_c"] + 180.0) % 360.0) // 60) for t in keep])

    P = np.array([np.mean([armb[s][t]["P_fine_anch_mean"] for s in SEASONS]) for t in keep])
    slope = np.array([np.mean([armb[s][t]["F_fine"]["slope"] for s in SEASONS]) for t in keep])
    lai = np.array([cov[t]["lai_mean"] for t in keep])
    land = np.array([cov[t]["land_frac"] for t in keep])
    cape = np.array([cov[t]["cape_mean"] for t in keep])

    e2 = epoch_mean(E2)
    if e2 is None:
        sys.exit("E2 annual files incomplete - run download_b26_precip_daily.py")
    e1 = epoch_mean(E1)

    def tiles_of(field, ep):
        return np.array([tile_mean(field, ep["lat"], ep["lon"], grid[t]) for t in keep])

    rx2 = tiles_of(e2["rx1day"], e2)
    tot2 = tiles_of(e2["prcptot"], e2)
    wet2 = tiles_of(e2["wetdays"], e2)
    T1 = np.log10(rx2)
    T1b = np.log10(rx2 / np.maximum(tot2, 1e-6))
    pw_f = candidate_fields()["pw"]
    pw = np.array([tile_mean(*pw_f, grid[t]) for t in keep])

    Xz = {n: z([cov[t][n] for t in keep]) for n in ALL_COV if n != "eke_syn"}
    Xz["eke_syn"] = z([np.mean([armb[s][t]["eke_syn"] for s in SEASONS]) for t in keep])
    B0 = np.column_stack([Xz[n] for n in ALL_COV] + [z(pw), z(np.log10(tot2)), z(wet2)])

    def rand_offsets(n):
        return MIN_ROT_DEG + rng.random(n) * (360.0 - 2 * MIN_ROT_DEG)

    def increment(base, cand, y):
        c = z(cand)[:, None]
        r2b = loso_r2(base, y, sector)
        r2f = loso_r2(np.column_stack([base, c]), y, sector)
        inc = r2f - r2b
        null = np.empty(N_ROT)
        for i, off in enumerate(rand_offsets(N_ROT)):
            cr = rotate_cov(c, kept_tiles, float(off), tid_index)
            null[i] = loso_r2(np.column_stack([base, cr]), y, sector) - r2b
        p = float((1 + np.sum(null >= inc)) / (N_ROT + 1))
        return {"r2_base": float(r2b), "r2_full": float(r2f), "increment": float(inc),
                "p_rotation": p, "null_q95": float(np.quantile(null, 0.95))}

    def ladder(h):
        if h["increment"] >= 0.05 and h["p_rotation"] <= 0.01:
            return "P_ADDS_STRONG"
        if h["increment"] >= 0.03 and h["p_rotation"] < 0.05:
            return "P_ADDS"
        return "NEGATIVE"

    def rho_rot(x, y):
        rho = float(spearmanr(x, y).statistic)
        col = np.asarray(x, float)[:, None]
        null = np.array([spearmanr(rotate_cov(col, kept_tiles, float(o), tid_index)[:, 0], y).statistic
                         for o in rand_offsets(N_ROT)])
        return {"rho": rho, "p_rotation_two_sided": float((1 + np.sum(np.abs(null) >= abs(rho))) / (N_ROT + 1))}

    res: dict = {"seed": SEED, "protocol": "docs/PROTOCOL_PHASE26_PRECIP_EXTREMES.md",
                 "n_tiles": len(keep), "E1_available": e1 is not None}

    # ---- H26a --------------------------------------------------------------
    res["C26_3_base_r2_T1"] = float(loso_r2(B0, T1, sector))
    h = increment(B0, P, T1); h["verdict"] = ladder(h); res["H26a"] = h
    res["C26_1_slope_placebo_T1"] = increment(B0, slope, T1)
    res["C26_2_lai_control_T1"] = increment(B0, lai, T1)

    # ---- H26b (conditional) ---------------------------------------------
    T2 = None
    if e1 is not None:
        rx1 = tiles_of(e1["rx1day"], e1)
        T2 = np.log(rx2 / rx1)
        B0b = np.column_stack([B0, z(T1)])
        h = increment(B0b, P, T2); h["verdict"] = ladder(h); res["H26b"] = h
        res["C26_1_slope_placebo_T2"] = increment(B0b, slope, T2)
        res["C26_2_lai_control_T2"] = increment(B0b, lai, T2)
        res["T2_summary"] = {"median_ln_ratio": float(np.median(T2)),
                             "share_positive": float(np.mean(T2 > 0))}
    else:
        res["H26b"] = {"verdict": "NOT_RUN (E1 not available)"}

    # ---- R1 / R2 -------------------------------------------------------------
    targets = {"T1_rx1_clim": T1, "T1b_conc": T1b}
    if T2 is not None:
        targets["T2_rx1_change"] = T2
    for label, mask in (("R1_all", np.ones(len(keep), bool)), ("R2_land", land >= 0.5)):
        res[label] = {"n": int(mask.sum())}
        for tn, tv in targets.items():
            entry = {"partial_rho_given_cape_pw": partial_spearman(P[mask], tv[mask], [cape[mask], pw[mask]]),
                     "rho_cape": float(spearmanr(cape[mask], tv[mask]).statistic)}
            if label == "R1_all":
                entry.update(rho_rot(P, tv))
            else:
                entry["rho"] = float(spearmanr(P[mask], tv[mask]).statistic)
            res[label][tn] = entry

    # ---- R3 boxes -------------------------------------------------------------
    lat = np.array([grid[t]["lat_c"] for t in keep]); lon = np.array([grid[t]["lon_c"] for t in keep])
    res["R3_boxes"] = {}
    for name, (a0, a1, o0, o1) in BOXES.items():
        m = (lat >= a0) & (lat <= a1) & (lon >= o0) & (lon <= o1) & (land >= 0.5)
        row = {"n": int(m.sum()), "P": float(P[m].mean()), "cape": float(cape[m].mean()),
               "rx1day_E2_mm": float(rx2[m].mean()), "prcptot_E2_mm": float(tot2[m].mean())}
        if T2 is not None:
            row["rx1_change_pct"] = float(100 * (np.exp(T2[m].mean()) - 1))
        res["R3_boxes"][name] = row

    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
