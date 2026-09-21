#!/usr/bin/env python3
"""Descriptive quantities quoted in article 2 that no other artifact stores.

Three numbers appear in the manuscript as descriptive readouts rather than as
pre-registered test statistics, and so were not written to any summary.json:

  D1  variance decomposition of the Arm-A tile map  (between- vs within-region)
  D2  variance decomposition of the Arm-B global map
      (6-degree latitude bands / 60-degree longitude sectors within bands /
       residual inside a band-sector cell)
  D3  the post-hoc Congo/Amazon hold-out: the Arm-A attribution model fitted on
      ten regions, applied to the 24 tiles of the two deep-convection basins
  D4  the exploratory orographic candidate for the unexplained part of the
      global map ("distance to mountains adds nothing"): great-circle distance
      from the tile centre to the nearest terrain above 1200 m; its rank
      correlation with the AUDIT-4 residual and its leave-one-sector-out R^2
      increment over the frozen baseline [8 covariates + 4 intermittency
      statistics] (target: season-mean anchored P, as in Phase 25)

This script recomputes all three from the committed tile tables and writes
results/verify_article2_descriptives/summary.json.  Model spec for D3 is the
one used in experiment_B20_p_geography.stage_tests: five covariates,
z-scored on the full 144-tile sample, ordinary least squares with intercept.

Output: results/verify_article2_descriptives/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from experiment_B20_p_geography import (  # noqa: E402
    COVARS,
    OUT as OUT_A,
    static_covariates,
)
from experiment_B20_armB_global_map import (  # noqa: E402
    OROG_EXCL_M,
    OUT as OUT_B,
    SEASONS,
    tile_covariates,
    tile_grid,
)

OUT = _HERE / "results" / "verify_article2_descriptives"
HELDOUT = ("R7_CONGO", "R3_AMAZ")


def _armA_table():
    per_tile: dict[tuple, dict] = {}
    for f in sorted(OUT_A.glob("tiles_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for tid, t in d["tiles"].items():
            e = per_tile.setdefault((d["region"], tid),
                                    {"anch": [], "eke": [],
                                     "lat_bounds": t["lat_bounds"],
                                     "lon_bounds": t["lon_bounds"]})
            e["anch"].append(t["P_fine_anch_mean"])
            e["eke"].append(t["eke_syn"])
    keys = sorted(per_tile)
    stat = static_covariates({k: per_tile[k] for k in keys})
    y = np.asarray([float(np.median(per_tile[k]["anch"])) for k in keys])
    regions = np.asarray([k[0] for k in keys])
    Xr = np.column_stack([
        [stat[k]["orog_std"] for k in keys],
        [stat[k]["land_frac"] for k in keys],
        [stat[k]["coast_var"] for k in keys],
        [stat[k]["cape_mean"] for k in keys],
        [float(np.median(per_tile[k]["eke"])) for k in keys],
    ])
    mu, sd = Xr.mean(0), Xr.std(0)
    X = (Xr - mu) / np.where(sd < 1e-12, 1.0, sd)
    return keys, X, y, regions


def _share_between(y: np.ndarray, key: np.ndarray) -> float:
    means = np.asarray([y[key == k].mean() for k in key])
    return float(means.var() / y.var())


def d1_armA(y, regions):
    between = _share_between(y, regions)
    return {"n_tiles": int(y.size), "n_regions": int(np.unique(regions).size),
            "between_region_share": between,
            "within_region_share": 1.0 - between}


def d2_armB():
    grid = tile_grid()
    by = {t["tid"]: t for t in grid}
    seas = {}
    for name in SEASONS:
        d = json.loads((OUT_B / f"tiles_{name}2023.json").read_text(encoding="utf-8"))
        seas[name] = {r["tid"]: r for r in d["tiles"]}
    common = sorted(set(seas["JFM"]) & set(seas["JAS"]))
    cov = tile_covariates([by[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    y = np.asarray([0.5 * (seas["JFM"][t]["P_fine_anch_mean"]
                           + seas["JAS"][t]["P_fine_anch_mean"]) for t in keep])
    band = np.asarray([by[t]["row"] for t in keep])
    sector = np.floor(np.asarray([by[t]["lon_c"] for t in keep]) / 60.0).astype(int)
    s_band = _share_between(y, band)
    s_cell = _share_between(y, band * 10 + sector)
    return {"n_tiles": len(keep),
            "band_share": s_band,
            "sector_within_band_share": s_cell - s_band,
            "within_cell_share": 1.0 - s_cell}


def d3_heldout_basins(keys, X, y, regions):
    te = np.isin(regions, HELDOUT)
    tr = ~te
    A = np.column_stack([np.ones(tr.sum()), X[tr]])
    coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    pred = np.column_stack([np.ones(te.sum()), X[te]]) @ coef
    obs = y[te]
    rho = float(spearmanr(obs, pred).statistic)
    reg_te = regions[te]
    d_obs = float(obs[reg_te == "R7_CONGO"].mean() - obs[reg_te == "R3_AMAZ"].mean())
    d_pred = float(pred[reg_te == "R7_CONGO"].mean() - pred[reg_te == "R3_AMAZ"].mean())
    return {"heldout_regions": list(HELDOUT), "n_heldout_tiles": int(te.sum()),
            "rho_obs_vs_pred_within_heldout": rho,
            "observed_congo_minus_amazon": d_obs,
            "predicted_congo_minus_amazon": d_pred}


def d4_orography_distance():
    import xarray as xr
    from experiment_B20_armB_global_map import ALL_COV, INV, G0
    from experiment_A4_residual_structure import A2B, HALVES, INT_COLS, ols_predict, z
    from experiment_B25_residual_candidates import loso_r2

    grid = {t["tid"]: t for t in tile_grid()}
    sh = {(s, h): {r["tid"]: r for r in json.loads(
        (A2B / f"tiles_{s}2023_{h}.json").read_text(encoding="utf-8"))["tiles"]}
        for s in SEASONS for h in HALVES}
    armb = {s: {r["tid"]: r for r in json.loads(
        (OUT_B / f"tiles_{s}2023.json").read_text(encoding="utf-8"))["tiles"]} for s in SEASONS}
    common = sorted(set.intersection(*[set(v) for v in sh.values()]))
    cov = tile_covariates([grid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    sector = np.array([int(((grid[t]["lon_c"] + 180.0) % 360.0) // 60) for t in keep])

    # distance (km) from tile centre to the nearest 1-degree cell with terrain > 1200 m
    inv = xr.open_dataset(INV)
    zz = inv["z"].squeeze().values / G0
    la = inv["latitude"].values[::4]
    lo = inv["longitude"].values[::4]
    high = zz[::4, ::4] > OROG_EXCL_M
    inv.close()
    LA, LO = np.meshgrid(np.deg2rad(la), np.deg2rad(lo), indexing="ij")
    hla, hlo = LA[high], LO[high]
    dist = np.empty(len(keep))
    for i, t in enumerate(keep):
        a, b = np.deg2rad(grid[t]["lat_c"]), np.deg2rad(grid[t]["lon_c"])
        c = np.sin(a) * np.sin(hla) + np.cos(a) * np.cos(hla) * np.cos(hlo - b)
        dist[i] = 6371.0 * float(np.arccos(np.clip(c, -1.0, 1.0)).min())
    cand = z(np.log10(dist + 1.0))[:, None]

    Xz = {n: z([cov[t][n] for t in keep]) for n in ALL_COV if n != "eke_syn"}
    Xz["eke_syn"] = z([np.mean([armb[s][t]["eke_syn"] for s in SEASONS]) for t in keep])
    X_cov = np.column_stack([Xz[n] for n in ALL_COV])
    INT = np.column_stack([z([np.mean([sh[(s, h)][t][f"{k}_anch_mean"]
                                       for s in SEASONS for h in HALVES]) for t in keep])
                           for k in INT_COLS])
    base = np.column_stack([X_cov, INT])
    y = np.array([np.mean([sh[(s, h)][t]["P_anch_mean"] for s in SEASONS for h in HALVES])
                  for t in keep])
    r2b = loso_r2(base, y, sector)
    r2f = loso_r2(np.column_stack([base, cand]), y, sector)

    resid = []
    for s in SEASONS:
        rf = {}
        for h, ho in (("odd", "even"), ("even", "odd")):
            yv = np.array([sh[(s, h)][t]["P_anch_mean"] for t in keep])
            yo = np.array([sh[(s, ho)][t]["P_anch_mean"] for t in keep])
            XI = np.column_stack([X_cov] + [z([sh[(s, ho)][t][f"{k}_anch_mean"] for t in keep])[:, None]
                                            for k in INT_COLS])
            rf[h] = yv - ols_predict(XI, yo, XI)
        resid.append(0.5 * (rf["odd"] + rf["even"]))
    r_mean = 0.5 * (resid[0] + resid[1])
    return {"n_tiles": len(keep), "status": "exploratory, not preregistered",
            "median_distance_km": float(np.median(dist)),
            "rho_distance_vs_audit4_residual": float(spearmanr(dist, r_mean).statistic),
            "loso_r2_base": float(r2b), "loso_r2_with_distance": float(r2f),
            "loso_increment": float(r2f - r2b)}


def main() -> None:
    keys, X, y, regions = _armA_table()
    res = {"note": "descriptive readouts quoted in article 2; no thresholds, "
                   "no null models -- recomputation aid only",
           "covariates_armA": list(COVARS),
           "D1_armA_variance_decomposition": d1_armA(y, regions),
           "D2_armB_variance_decomposition": d2_armB(),
           "D3_congo_amazon_holdout": d3_heldout_basins(keys, X, y, regions),
           "D4_orography_distance_candidate": d4_orography_distance()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
