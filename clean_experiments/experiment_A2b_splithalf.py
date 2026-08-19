#!/usr/bin/env python3
"""AUDIT-2b: split-half decontamination of A2d and the map's true ceiling.

Protocol: docs/PROTOCOL_AUDIT2B_SPLITHALF_DECONTAMINATION.md (frozen
2026-08-19).

Each tile-season sample is split by time parity into two disjoint halves
(`odd`, `even`). Both halves span the whole season; only the sampling
noise is independent. Every AUDIT-1 and AUDIT-2 statistic is computed on
each half in one pass, each anchored on its own 99 phase surrogates.

Stages
  --stage tiles --season JFM|JAS --half odd|even   resumable, atomic shards
  --stage collect                                  merge shards per (season, half)
  --stage tests                                    B1-B4
  --stage all                                      all four tile runs, then tests
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
except ImportError:
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

CONFIG_VERSION = "a2b_v1"
OUT = _HERE / "results" / "experiment_A2b_splithalf"
ARMB = _HERE / "results" / "experiment_B20_armB_global_map"
HALVES = ("odd", "even")
STATS = ("P", "P_lin", "R_AM_cyc", "P_cascade", "INT_sig2", "INT_flat", "INT_mu")
INT_COLS = ("P_cascade", "INT_sig2", "INT_flat", "INT_mu")


# --------------------------------------------------------------------------
# one pass over the band ladder -> every AUDIT-1 / AUDIT-2 statistic
# --------------------------------------------------------------------------

def all_profiles(omega, dx_km, dy_km, mask, ells_km, lat_sign: float):
    n_steps = len(ells_km) - 1
    n_bands = n_steps + 1
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

    nt = w.shape[0]
    sig2 = np.empty((nt, n_bands))
    flat = np.empty((nt, n_bands))
    for b in range(n_bands):
        sig2[:, b] = env_m[b].var(axis=1)
        s = sig_m[b] - sig_m[b].mean(axis=1, keepdims=True)
        m2 = (s * s).mean(axis=1)
        m4 = (s * s * s * s).mean(axis=1)
        flat[:, b] = m4 / np.maximum(m2 * m2, 1e-30)

    ell_log = np.log(np.asarray(ells_km, dtype=float))
    ell_log = ell_log - ell_log.mean()
    denom_mu = float((ell_log * ell_log).sum())
    lf = np.log(np.maximum(flat, 1e-30))
    mu = -(lf * ell_log[None, :]).sum(axis=1) / denom_mu

    scalars = {
        "INT_sig2": float(np.median(sig2.mean(axis=1))),
        "INT_flat": float(np.median(np.log10(np.maximum(flat, 1e-30)).mean(axis=1))),
        "INT_mu": float(np.median(mu)),
    }
    out = {k: np.zeros(n_steps) for k in STATS}
    for i in range(n_steps):
        out["P"][i] = np.median(_corr(ranked[i], ranked[i + 1]))
        out["P_lin"][i] = np.median(_corr(env_m[i], env_m[i + 1]))
        out["R_AM_cyc"][i] = lat_sign * np.median(_corr(sig_m[i + 1], env_m[i]))
        out["P_cascade"][i] = np.median(
            np.sqrt(np.maximum(sig2[:, i + 1], 0.0) / np.maximum(sig2[:, i], 1e-30)))
        for k, v in scalars.items():
            out[k][i] = v
    return {k: [float(x) for x in v] for k, v in out.items()}


# --------------------------------------------------------------------------
# tiles stage (resumable)
# --------------------------------------------------------------------------

_G: dict[str, object] = {}


def shard_path(season: str, half: str, tid: str) -> Path:
    return OUT / "tiles" / f"{season}_{half}" / f"{tid}.json"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _tile_worker(t: dict) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    season, half = _G["season"], _G["half"]
    path = shard_path(season, half, t["tid"])
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
    sl = slice(1, None, 2) if half == "odd" else slice(0, None, 2)
    uu = u[sl][:, iy][:, :, ix].astype(np.float32)
    vv = v[sl][:, iy][:, :, ix].astype(np.float32)
    dx_km, dy_km = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1], dx_km, dy_km)
    lat_sign = float(np.sign(t["lat_c"])) or 1.0

    omega = compute_vorticity(uu, vv, la, lo)
    real = all_profiles(omega, dx_km, dy_km, mask, ELLS_FINE, lat_sign)

    tcode = zlib.crc32(f"glob_{t['tid']}_{season}_{half}".encode()) % 100000
    seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(N_SUR)
    sur = {k: np.empty((N_SUR, N_FINE)) for k in STATS}
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        us, vs = phase_randomize(uu, vv, rng)
        q = all_profiles(compute_vorticity(us, vs, la, lo), dx_km, dy_km,
                         mask, ELLS_FINE, lat_sign)
        for k in STATS:
            sur[k][j] = q[k]

    payload = {"tid": t["tid"], "season": season, "half": half,
               "version": CONFIG_VERSION, "n_steps_used": int(uu.shape[0])}
    for k in STATS:
        med = np.median(sur[k], axis=0)
        payload[f"{k}_real"] = real[k]
        payload[f"{k}_anch_mean"] = float(np.mean(np.asarray(real[k]) - med))
    _write_atomic(path, payload)
    return "done"


def stage_tiles(season: str, half: str, workers: int, limit: int | None) -> None:
    tiles = tile_grid()
    todo = [t for t in tiles if not shard_path(season, half, t["tid"]).exists()]
    if limit:
        todo = todo[:limit]
    print(f"{season}/{half}: {len(tiles)} tiles, {len(todo)} to compute", flush=True)
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
    _G.update(u=u, v=v, lat=lat, lon=lon, season=season, half=half)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_tile_worker, todo, chunksize=2)):
            if (i + 1) % 50 == 0 or (i + 1) == len(todo):
                el = time.time() - t0
                print(f"[{i+1}/{len(todo)}] {el/60:.1f} min, "
                      f"{(len(todo)-i-1)/((i+1)/el)/60:.1f} min left", flush=True)


def load_shards(season: str, half: str) -> dict[str, dict]:
    d = {}
    for f in sorted((OUT / "tiles" / f"{season}_{half}").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("version") == CONFIG_VERSION:
            d[r["tid"]] = r
    return d


def stage_collect() -> None:
    for season in SEASONS:
        for half in HALVES:
            sh = load_shards(season, half)
            if not sh:
                continue
            _write_atomic(OUT / f"tiles_{season}2023_{half}.json",
                          {"season": season, "half": half,
                           "version": CONFIG_VERSION, "n_tiles": len(sh),
                           "tiles": [sh[t] for t in sorted(sh)]})
            print(f"{season}/{half}: merged {len(sh)} tiles", flush=True)


# --------------------------------------------------------------------------
# tests stage
# --------------------------------------------------------------------------

def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res = {"protocol": "docs/PROTOCOL_AUDIT2B_SPLITHALF_DECONTAMINATION.md",
           "version": CONFIG_VERSION, "seed": SEED}
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    sh = {(s, h): load_shards(s, h) for s in SEASONS for h in HALVES}
    common = sorted(set.intersection(*[set(v) for v in sh.values()]))
    cov = tile_covariates([by_tid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    tiles = [by_tid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}
    sector = np.array([int(by_tid[t]["lon_c"] // SECTOR_DEG) for t in keep])
    res["n_common"], res["n_kept"] = len(common), len(keep)

    def col(stat, season, half):
        return np.array([sh[(season, half)][t][f"{stat}_anch_mean"] for t in keep])

    def z(x):
        sd = x.std()
        return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)

    armb = {}
    for s in SEASONS:
        d = json.loads((ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))
        armb[s] = {r["tid"]: r for r in d["tiles"]}
    Xz = {n: z(np.array([cov[t][n] for t in keep])) for n in ALL_COV if n != "eke_syn"}
    Xz["eke_syn"] = z(np.array([np.mean([armb[s][t]["eke_syn"] for s in SEASONS])
                                for t in keep]))
    X_all = np.column_stack([Xz[n] for n in ALL_COV])

    # ---- B1 ceiling ------------------------------------------------------
    res["B1"] = {}
    for s in SEASONS:
        a, b = col("P", s, "odd"), col("P", s, "even")
        r_half = float(np.corrcoef(a, b)[0, 1])
        r_full = 2 * r_half / (1 + r_half)
        res["B1"][s] = {"r_half": r_half, "spearman_half": float(spearmanr(a, b).statistic),
                        "r_full_spearman_brown": float(r_full),
                        "R2_ceiling_full_sample": float(r_full ** 2)}
    res["B1"]["mean_R2_ceiling"] = float(np.mean(
        [res["B1"][s]["R2_ceiling_full_sample"] for s in SEASONS]))
    res["B1"]["cross_season_ceiling_used_before"] = 0.603

    # ---- B2 scored primary ----------------------------------------------
    def increment(target, block, n_rot=N_ROT):
        base = loso_r2(X_all, target, sector)
        full = loso_r2(np.column_stack([X_all, block]), target, sector)
        inc = full - base
        null = np.empty(n_rot)
        for i in range(n_rot):
            off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
            null[i] = loso_r2(np.column_stack(
                [X_all, rotate_cov(block, tiles, off, tid_index)]),
                target, sector) - base
        return {"r2_base": float(base), "r2_full": float(full),
                "increment": float(inc),
                "p_rot": float((1 + np.sum(null >= inc)) / (n_rot + 1)),
                "null_q95": float(np.quantile(null, 0.95))}

    res["B2"] = {}
    incs = []
    for s in SEASONS:
        for h_t, h_p in (("odd", "even"), ("even", "odd")):
            block = np.column_stack([z(col(k, s, h_p)) for k in INT_COLS])
            r = increment(col("P", s, h_t), block)
            res["B2"][f"{s}_P{h_t}_INT{h_p}"] = r
            incs.append(r["increment"])
    res["B2"]["mean_increment"] = float(np.mean(incs))
    ps = [v["p_rot"] for k, v in res["B2"].items() if isinstance(v, dict)]
    res["B2"]["max_p"] = float(max(ps))
    res["B2"]["verdict"] = ("INTERMITTENCY_ADDS"
                            if res["B2"]["mean_increment"] > 0 and max(ps) < 0.05
                            else "INTERMITTENCY_NULL_AFTER_DECONTAMINATION")

    # ---- B3 contamination magnitude + attenuation correction -------------
    same = []
    for s in SEASONS:
        for h in HALVES:
            block = np.column_stack([z(col(k, s, h)) for k in INT_COLS])
            same.append(increment(col("P", s, h), block, n_rot=99)["increment"])
    rel_int = {k: float(np.mean([np.corrcoef(col(k, s, "odd"), col(k, s, "even"))[0, 1]
                                 for s in SEASONS])) for k in INT_COLS}
    res["B3"] = {"same_half_increment_mean": float(np.mean(same)),
                 "decontaminated_increment_mean": res["B2"]["mean_increment"],
                 "shared_noise_share": float(
                     1.0 - res["B2"]["mean_increment"] / max(np.mean(same), 1e-9)),
                 "split_half_reliability_of_block": rel_int}
    dis = []
    for s in SEASONS:
        for h_t, h_p in (("odd", "even"), ("even", "odd")):
            block = np.column_stack([z(col(k, s, h_p)) / max(np.sqrt(abs(rel_int[k])), 1e-6)
                                     for k in INT_COLS])
            dis.append(increment(col("P", s, h_t), block, n_rot=99)["increment"])
    res["B3"]["attenuation_corrected_increment_mean"] = float(np.mean(dis))

    # ---- B4 cross-half versions of the AUDIT-1/2 primaries ---------------
    res["B4"] = {}
    for k in ("R_AM_cyc", "P_lin", "P_cascade", "INT_sig2"):
        vals = []
        for s in SEASONS:
            for h_t, h_p in (("odd", "even"), ("even", "odd")):
                vals.append(float(spearmanr(col("P", s, h_t), col(k, s, h_p)).statistic))
        res["B4"][f"rho_P_vs_{k}_cross_half"] = float(np.mean(vals))
        res["B4"][f"rho_P_vs_{k}_cross_half_range"] = [float(min(vals)), float(max(vals))]
    res["VERDICT"] = res["B2"]["verdict"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tiles", "collect", "tests", "all"], default="all")
    ap.add_argument("--season", choices=["JFM", "JAS"], default=None)
    ap.add_argument("--half", choices=["odd", "even"], default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("tiles", "all"):
        for s in ([args.season] if args.season else list(SEASONS)):
            for h in ([args.half] if args.half else list(HALVES)):
                stage_tiles(s, h, args.workers, args.limit)
    if args.stage in ("collect", "all"):
        stage_collect()
    if args.stage in ("tests", "all"):
        res = stage_tests()
        (OUT / "summary_halves.json").write_text(json.dumps(res, indent=1),
                                                 encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "B2"}, indent=1))
        print("VERDICT:", res["VERDICT"])


if __name__ == "__main__":
    main()
