#!/usr/bin/env python3
"""AUDIT-1: prior-art head-to-head for the cross-scale coupling profile P.

Protocol: docs/PROTOCOL_AUDIT1_PRIOR_ART_HEADTOHEAD.md (frozen 2026-08-19).

Compares, on the identical Phase-20 Arm-B sample, machinery, mask and
surrogate realisations:

  P         Spearman rank correlation between adjacent band log-envelopes
            (the programme's estimator)
  P_lin     Pearson on the same envelopes (amplitude-amplitude coupling)
  R_AM_cyc  Pearson between the signed coarser band field (cyclonic
            convention) and the finer band log-envelope
            (Mathis-Hutchins-Marusic amplitude-modulation coefficient)
  R_AM_raw  the same without the cyclonic sign convention

Stages
  --stage tiles --season JFM|JAS   resumable; one atomic JSON shard per tile
  --stage tests                    scored questions A1a-A1d
  --stage all                      tiles (both seasons) then tests

The tiles stage is idempotent: an existing shard with a matching
CONFIG_VERSION is never recomputed, so the run can be killed and resumed
at any point.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask, phase_randomize,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        EPS, compute_vorticity, gaussian_bar_grouped,
    )
    from clean_experiments.experiment_B20_p_geography import (
        ELLS_FINE, N_FINE, N_SUR,
    )
    from clean_experiments.experiment_B20_armB_global_map import (
        ALL_COV, OROG_EXCL_M, N_ROT, ROT_MIN_DEG, SEASONS, SECTOR_DEG,
        SEED, SEED_SUR, GLOB, loso_r2, rotate_cov, tile_covariates, tile_grid,
    )
except ImportError:  # direct execution from clean_experiments/
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry, _interior_mask, phase_randomize,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        EPS, compute_vorticity, gaussian_bar_grouped,
    )
    from experiment_B20_p_geography import (  # type: ignore
        ELLS_FINE, N_FINE, N_SUR,
    )
    from experiment_B20_armB_global_map import (  # type: ignore
        ALL_COV, OROG_EXCL_M, N_ROT, ROT_MIN_DEG, SEASONS, SECTOR_DEG,
        SEED, SEED_SUR, GLOB, loso_r2, rotate_cov, tile_covariates, tile_grid,
    )

CONFIG_VERSION = "a1_v1"
OUT = _HERE / "results" / "experiment_A1_prior_art"
ARMB = _HERE / "results" / "experiment_B20_armB_global_map"
STATS = ("P", "P_lin", "R_AM_cyc", "R_AM_raw")


# --------------------------------------------------------------------------
# the four statistics, computed from one pass over the band ladder
# --------------------------------------------------------------------------

def quad_profiles(omega, dx_km, dy_km, mask, ells_km, lat_sign: float):
    """Return {stat: [n_steps]} of time-median values.

    Band ladder identical to envelope_rho_profile_ells (B18:162): the
    envelopes and their pairing are unchanged, so `P` here is the frozen
    estimator bit-for-bit.
    """
    n_steps = len(ells_km) - 1
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km)
            for ell in ells_km]
    sigs = [w - bars[0]]
    for i in range(n_steps):
        sigs.append(bars[i] - bars[i + 1])
    del bars
    envs = [np.log(gaussian_bar_grouped(np.abs(sigs[0]), ells_km[0] / np.sqrt(12.0),
                                        dx_km, dy_km) + EPS)]
    for i in range(n_steps):
        envs.append(np.log(gaussian_bar_grouped(
            np.abs(sigs[i + 1]), ells_km[i + 1] / np.sqrt(12.0),
            dx_km, dy_km) + EPS))

    env_m = [e[:, mask] for e in envs]
    sig_m = [s[:, mask] for s in sigs]
    del envs, sigs
    ranked = [np.argsort(np.argsort(m, axis=1), axis=1).astype(np.float32)
              for m in env_m]

    def _corr(a, b):
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        den = np.where(den < 1e-12, 1.0, den)
        return (a * b).sum(axis=1) / den

    out = {k: np.zeros(n_steps) for k in STATS}
    for i in range(n_steps):
        out["P"][i] = np.median(_corr(ranked[i], ranked[i + 1]))
        out["P_lin"][i] = np.median(_corr(env_m[i], env_m[i + 1]))
        out["R_AM_raw"][i] = np.median(_corr(sig_m[i + 1], env_m[i]))
        out["R_AM_cyc"][i] = lat_sign * out["R_AM_raw"][i]
    return {k: [float(x) for x in v] for k, v in out.items()}


# --------------------------------------------------------------------------
# tiles stage (resumable)
# --------------------------------------------------------------------------

_G: dict[str, object] = {}


def shard_path(season: str, tid: str) -> Path:
    return OUT / "tiles" / season / f"{tid}.json"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _tile_worker(t: dict) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    season = _G["season"]
    path = shard_path(season, t["tid"])
    if path.exists():
        return "cached"
    u, v = _G["u"], _G["v"]
    lat, lon = _G["lat"], _G["lon"]
    iy = np.where((lat >= t["lat0"] - 1e-9) & (lat <= t["lat1"] + 1e-9))[0]
    nx = len(lon)
    i0 = int(np.searchsorted(lon, t["lon0"] - 1e-9))
    n_cols = int(round((t["lon1"] - t["lon0"]) / 0.25))
    ix = (np.arange(i0, i0 + n_cols)) % nx
    la = lat[iy]
    lo = t["lon0"] + 0.25 * np.arange(n_cols)
    uu = u[:, iy][:, :, ix].astype(np.float32)
    vv = v[:, iy][:, :, ix].astype(np.float32)
    dx_km, dy_km = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1], dx_km, dy_km)
    lat_sign = float(np.sign(t["lat_c"])) or 1.0

    omega = compute_vorticity(uu, vv, la, lo)
    real = quad_profiles(omega, dx_km, dy_km, mask, ELLS_FINE, lat_sign)

    # identical surrogate realisations to Phase-20 Arm B
    tcode = zlib.crc32(f"glob_{t['tid']}_{season}".encode()) % 100000
    seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(N_SUR)
    sur = {k: np.empty((N_SUR, N_FINE)) for k in STATS}
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        us, vs = phase_randomize(uu, vv, rng)
        om_s = compute_vorticity(us, vs, la, lo)
        q = quad_profiles(om_s, dx_km, dy_km, mask, ELLS_FINE, lat_sign)
        for k in STATS:
            sur[k][j] = q[k]

    payload = {"tid": t["tid"], "season": season, "version": CONFIG_VERSION,
               "lat_c": t["lat_c"], "lon_c": t["lon_c"], "n_sur": N_SUR}
    for k in STATS:
        med = np.median(sur[k], axis=0)
        payload[f"{k}_real"] = real[k]
        payload[f"{k}_sur_median"] = [float(x) for x in med]
        payload[f"{k}_anch_mean"] = float(np.mean(np.asarray(real[k]) - med))
    _write_atomic(path, payload)
    return "done"


def stage_tiles(season: str, workers: int, limit: int | None = None) -> None:
    tiles = tile_grid()
    todo = [t for t in tiles if not shard_path(season, t["tid"]).exists()]
    if limit:
        todo = todo[:limit]
    print(f"{season}: {len(tiles)} tiles, {len(todo)} to compute", flush=True)
    if not todo:
        return
    import xarray as xr
    parts = [xr.open_dataset(GLOB / f"era5_wind850_global_{ym}.nc")
             for ym in SEASONS[season]]
    ds = xr.concat(parts, dim="valid_time")
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    print("loading u, v ...", flush=True)
    u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
    v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
    for p in parts:
        p.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]
    print(f"{season}: cube {u.shape}", flush=True)
    _G.update(u=u, v=v, lat=lat, lon=lon, season=season)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_tile_worker, todo, chunksize=2)):
            if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                el = time.time() - t0
                rate = (i + 1) / el
                print(f"[{i+1}/{len(todo)}] {el/60:.1f} min elapsed, "
                      f"{(len(todo)-i-1)/rate/60:.1f} min left", flush=True)


def stage_collect() -> None:
    """Merge the per-tile shards into one committed file per season."""
    for season in SEASONS:
        sh = load_shards(season)
        if not sh:
            continue
        path = OUT / f"tiles_{season}2023.json"
        _write_atomic(path, {"season": season, "version": CONFIG_VERSION,
                             "n_tiles": len(sh),
                             "tiles": [sh[t] for t in sorted(sh)]})
        print(f"{season}: merged {len(sh)} tiles -> {path.name}", flush=True)


def load_shards(season: str) -> dict[str, dict]:
    d = {}
    p = OUT / "tiles" / season
    for f in sorted(p.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("version") == CONFIG_VERSION:
            d[r["tid"]] = r
    return d


# --------------------------------------------------------------------------
# tests stage
# --------------------------------------------------------------------------

def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res = {"protocol": "docs/PROTOCOL_AUDIT1_PRIOR_ART_HEADTOHEAD.md",
           "version": CONFIG_VERSION, "seed": SEED}
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    sh = {s: load_shards(s) for s in SEASONS}
    armb = {}
    for s in SEASONS:
        d = json.loads((ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))
        armb[s] = {r["tid"]: r for r in d["tiles"]}
    common = sorted(set(sh["JFM"]) & set(sh["JAS"]) & set(armb["JFM"]) & set(armb["JAS"]))
    res["n_common"] = len(common)

    # ---- C-A1-0 pipeline identity ---------------------------------------
    a = np.array([np.mean([sh[s][t]["P_anch_mean"] for s in SEASONS]) for t in common])
    b = np.array([np.mean([armb[s][t]["P_fine_anch_mean"] for s in SEASONS]) for t in common])
    raw_diff = max(abs(x - y) for t in common for s in SEASONS
                   for x, y in zip(sh[s][t]["P_real"], armb[s][t]["P_fine_real"]))
    rho_id = float(spearmanr(a, b).statistic)
    res["C_A1_0"] = {"rho_vs_armB": rho_id, "max_abs_diff_raw": float(raw_diff),
                     "pass": bool(rho_id >= 0.99 and raw_diff <= 1e-6)}
    if not res["C_A1_0"]["pass"]:
        res["VERDICT"] = "PIPELINE_MISMATCH"
        return res

    cov = tile_covariates([by_tid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    tiles = [by_tid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}
    res["n_kept"] = len(keep)

    def target(stat: str) -> np.ndarray:
        return np.array([np.mean([sh[s][t][f"{stat}_anch_mean"] for s in SEASONS])
                         for t in keep])

    Y = {k: target(k) for k in STATS}
    eke = np.array([np.mean([armb[s][t]["eke_syn"] for s in SEASONS]) for t in keep])

    def z(x):
        sd = x.std()
        return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)

    Xz = {n: z(np.array([cov[t][n] for t in keep])) for n in
          ("orog_mean", "orog_std", "land_frac", "coast_var", "abs_lat",
           "sst_grad", "cape_mean")}
    Xz["eke_syn"] = z(eke)
    X_all = np.column_stack([Xz[n] for n in ALL_COV])
    sector = np.array([int(by_tid[t]["lon_c"] // SECTOR_DEG) for t in keep])

    # ---- A1a primary, A1c ------------------------------------------------
    def rho(x, y):
        return float(spearmanr(x, y).statistic)

    r_am = rho(Y["P"], Y["R_AM_cyc"])
    verdict = ("ESTIMATOR_IS_AM" if abs(r_am) >= 0.90 else
               "ESTIMATOR_RELATED" if abs(r_am) >= 0.60 else
               "ESTIMATOR_DISTINCT")
    res["A1a"] = {"rho_P_vs_R_AM_cyc": r_am,
                  "rho_P_vs_R_AM_raw": rho(Y["P"], Y["R_AM_raw"]),
                  "verdict": verdict}
    res["A1c"] = {"rho_P_vs_P_lin": rho(Y["P"], Y["P_lin"]),
                  "drop_copula_language": bool(abs(rho(Y["P"], Y["P_lin"])) >= 0.95)}

    # ---- A1b attribution transfer ---------------------------------------
    def rot_null(y, n=N_ROT):
        vals = np.empty(n)
        for i in range(n):
            off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
            vals[i] = loso_r2(rotate_cov(X_all, tiles, off, tid_index), y, sector)
        return vals

    res["A1b"] = {}
    for k in STATS:
        y = Y[k]
        r2 = loso_r2(X_all, y, sector)
        null = rot_null(y)
        res["A1b"][k] = {
            "loso_r2": float(r2),
            "p_rot": float((1 + np.sum(null >= r2)) / (N_ROT + 1)),
            "null_q95": float(np.quantile(null, 0.95)),
            "loadings": {n: rho(Xz[n], y) for n in ALL_COV},
        }
    res["A1b"]["armB_reference_r2"] = 0.5357542841968607

    # ---- A1d reported ----------------------------------------------------
    res["A1d"] = {"per_season": {}, "cross_season_reliability": {}}
    for s in SEASONS:
        ys = {k: np.array([sh[s][t][f"{k}_anch_mean"] for t in keep]) for k in STATS}
        res["A1d"]["per_season"][s] = {
            "rho_P_vs_R_AM_cyc": rho(ys["P"], ys["R_AM_cyc"]),
            "rho_P_vs_P_lin": rho(ys["P"], ys["P_lin"]),
        }
    for k in STATS:
        x1 = np.array([sh["JFM"][t][f"{k}_anch_mean"] for t in keep])
        x2 = np.array([sh["JAS"][t][f"{k}_anch_mean"] for t in keep])
        res["A1d"]["cross_season_reliability"][k] = {
            "pearson": float(np.corrcoef(x1, x2)[0, 1]),
            "spearman": rho(x1, x2),
        }
    res["VERDICT"] = verdict
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tiles", "collect", "tests", "all"],
                    default="all")
    ap.add_argument("--season", choices=["JFM", "JAS"], default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage in ("tiles", "all"):
        seasons = [args.season] if args.season else list(SEASONS)
        for s in seasons:
            stage_tiles(s, args.workers, args.limit)
    if args.stage in ("collect", "all"):
        stage_collect()
    if args.stage in ("tests", "all"):
        res = stage_tests()
        (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "A1b"}, indent=1))
        print("VERDICT:", res.get("VERDICT"))


if __name__ == "__main__":
    main()
