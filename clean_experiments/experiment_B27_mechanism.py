#!/usr/bin/env python3
"""Phase 27: mechanism of the inter-level coupling.

Frozen spec: docs/PROTOCOL_PHASE27_MECHANISM.md (2026-09-21). Two carriers
already on disk: 48 equal-km units (data/b3, full ladder) and the Phase-20
Arm-B global tile grid (data/b20global, fine ladder). Statistics: all-pair
anchored envelope correlations (S1), lag-2 partials (S2), deformation
index (S3), anchored linear log-envelope covariances (S4).

Stages: --stage units | tiles --season JFM|JAS | tests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B18_equal_km_regions import REGIONS, km_crop
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask, phase_randomize,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM, EPS, compute_vorticity, gaussian_bar_grouped,
    )
    from clean_experiments.experiment_B20_p_geography import ELLS_FINE, N_SUR
    from clean_experiments.experiment_B20_armB_global_map import (
        ALL_COV, GLOB, OROG_EXCL_M, SEASONS, rotate_cov, tile_covariates,
        tile_grid,
    )
except ImportError:
    from experiment_B18_equal_km_regions import REGIONS, km_crop  # type: ignore
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry, _interior_mask, phase_randomize,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM, EPS, compute_vorticity, gaussian_bar_grouped,
    )
    from experiment_B20_p_geography import ELLS_FINE, N_SUR  # type: ignore
    from experiment_B20_armB_global_map import (  # type: ignore
        ALL_COV, GLOB, OROG_EXCL_M, SEASONS, rotate_cov, tile_covariates,
        tile_grid,
    )

CONFIG_VERSION = "b27_v1"
PROTOCOL = Path("docs/PROTOCOL_PHASE27_MECHANISM.md")
SEED = 20260921
SEED_SUR = 20260811
OUT = _HERE / "results" / "experiment_B27_mechanism"
ARMB = _HERE / "results" / "experiment_B20_armB_global_map"
A3 = _HERE / "results" / "experiment_A3_robustness"
B3 = Path("data/b3")
WINDOWS = ("W9_2023JFM", "W10_2023JAS", "W11_2024JFM", "W12_2024JAS")
ELL_C_UNITS = 800.0
ELL_C_TILES = 400.0
LAG_STEPS = 20
FINE_F = (0, 1, 2)
N_ROT = 999
ROT_MIN_DEG = 30.0
N_BOOT_BLOCK = 1999
N_BOOT_REGION = 9999
BLOCK_DEG = 30.0


# --------------------------------------------------------------------------
# per-cube statistics
# --------------------------------------------------------------------------

def _zranks(m: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(m, axis=1), axis=1).astype(np.float32)
    r -= r.mean(axis=1, keepdims=True)
    n = np.sqrt((r * r).sum(axis=1, keepdims=True))
    return r / np.where(n < 1e-12, 1.0, n)


def _partial(r_ab, r_ac, r_bc):
    den = np.sqrt(np.maximum((1.0 - r_ac ** 2) * (1.0 - r_bc ** 2), 1e-12))
    return (r_ab - r_ac * r_bc) / den


def stat_names(n_env: int) -> list[str]:
    names = [f"rho_{i}_{j}" for i in range(n_env) for j in range(i + 1, n_env)]
    names += [f"part2_{i}" for i in range(n_env - 2)]
    for f in FINE_F:
        names += [f"D_{f}", f"V_{f}", f"Dlag_{f}"]
    names += [f"cov_{i}_{j}" for i in range(n_env) for j in range(i, n_env)]
    return names


def cube_stats(u, v, la, lo, dx_km, dy_km, mask, ells, ell_c) -> dict[str, list[float]]:
    """All Phase-27 statistics for one (nt, ny, nx) wind cube.

    Returns name -> [median over all steps, over first half, over second half].
    """
    n_env = len(ells)
    omega = compute_vorticity(u, v, la, lo)
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km) for ell in ells]
    envs = [np.log(gaussian_bar_grouped(np.abs(w - bars[0]), ells[0] / np.sqrt(12.0),
                                        dx_km, dy_km) + EPS)]
    for i in range(n_env - 1):
        envs.append(np.log(gaussian_bar_grouped(
            np.abs(bars[i] - bars[i + 1]), ells[i + 1] / np.sqrt(12.0),
            dx_km, dy_km) + EPS))
    del bars

    # coarse-flow strain and |vorticity|
    lat_sign = 1.0 if la[1] > la[0] else -1.0
    sig_c = ell_c / np.sqrt(12.0)
    uc = gaussian_bar_grouped(np.nan_to_num(u).astype(np.float32) / 1000.0,
                              sig_c, dx_km, dy_km)
    vc = gaussian_bar_grouped(np.nan_to_num(v).astype(np.float32) / 1000.0,
                              sig_c, dx_km, dy_km)
    ux = np.gradient(uc, axis=2) / dx_km[None, :, None]
    uy = np.gradient(uc, axis=1) / dy_km * lat_sign
    vx = np.gradient(vc, axis=2) / dx_km[None, :, None]
    vy = np.gradient(vc, axis=1) / dy_km * lat_sign
    S = np.sqrt((ux - vy) ** 2 + (vx + uy) ** 2)
    Z = np.abs(vx - uy)
    del uc, vc, ux, uy, vx, vy

    env_m = [e[:, mask] for e in envs]
    del envs
    S_m, Z_m = S[:, mask], Z[:, mask]
    S_l, Z_l = np.roll(S_m, -LAG_STEPS, axis=0), np.roll(Z_m, -LAG_STEPS, axis=0)

    zr = np.stack([_zranks(m) for m in env_m + [S_m, Z_m, S_l, Z_l]], axis=1)
    R = np.einsum("tkc,tlc->tkl", zr, zr)
    del zr
    iS, iZ, iSl, iZl = n_env, n_env + 1, n_env + 2, n_env + 3

    series: dict[str, np.ndarray] = {}
    for i in range(n_env):
        for j in range(i + 1, n_env):
            series[f"rho_{i}_{j}"] = R[:, i, j]
    for i in range(n_env - 2):
        series[f"part2_{i}"] = _partial(R[:, i, i + 2], R[:, i, i + 1], R[:, i + 1, i + 2])
    for f in FINE_F:
        series[f"D_{f}"] = _partial(R[:, f, iS], R[:, f, iZ], R[:, iS, iZ])
        series[f"V_{f}"] = _partial(R[:, f, iZ], R[:, f, iS], R[:, iS, iZ])
        series[f"Dlag_{f}"] = _partial(R[:, f, iSl], R[:, f, iZl], R[:, iSl, iZl])

    cen = np.stack([m - m.mean(axis=1, keepdims=True) for m in env_m], axis=1)
    C = np.einsum("tkc,tlc->tkl", cen, cen) / cen.shape[2]
    for i in range(n_env):
        for j in range(i, n_env):
            series[f"cov_{i}_{j}"] = C[:, i, j]

    nt = w.shape[0]
    h = nt // 2
    return {k: [float(np.median(s)), float(np.median(s[:h])), float(np.median(s[h:]))]
            for k, s in series.items()}


def anchored_payload(uu, vv, la, lo, ells, ell_c, seed_tag: str) -> dict:
    dx_km, dy_km = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ells[-1], dx_km, dy_km)
    real = cube_stats(uu, vv, la, lo, dx_km, dy_km, mask, ells, ell_c)
    names = list(real.keys())
    tcode = zlib.crc32(seed_tag.encode()) % 100000
    seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(N_SUR)
    sur = np.empty((N_SUR, len(names), 3))
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        us, vs = phase_randomize(uu, vv, rng)
        q = cube_stats(us, vs, la, lo, dx_km, dy_km, mask, ells, ell_c)
        sur[j] = np.asarray([q[k] for k in names])
    med = np.median(sur, axis=0)
    q05 = np.quantile(sur[:, :, 0], 0.05, axis=0)
    q95 = np.quantile(sur[:, :, 0], 0.95, axis=0)
    return {"version": CONFIG_VERSION, "names": names,
            "real": [real[k] for k in names],
            "sur_med": med.tolist(),
            "sur_q05": q05.tolist(), "sur_q95": q95.tolist(),
            "n_cells": [int(uu.shape[1]), int(uu.shape[2])],
            "n_interior": int(mask.sum()), "n_steps_time": int(uu.shape[0])}


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# units stage
# --------------------------------------------------------------------------

def _units_dir(factor: float) -> Path:
    return OUT / ("units" if factor == 1.0 else f"units_x{factor:g}")


def _unit_worker(job: tuple[str, str, float]) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    region, window, factor = job
    tag = f"{region}__{window}"
    path = _units_dir(factor) / f"{tag}.json"
    if path.exists():
        return "cached"
    import xarray as xr
    ds = xr.open_dataset(B3 / f"era5_wind850_{tag}.nc")
    u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
    v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    ds.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]
    la, lo, uu, vv = km_crop(lat, lon, u, v, region, factor)
    salt = "" if factor == 1.0 else f"_x{factor:g}"
    payload = anchored_payload(uu, vv, la, lo, ELLS_KM, ELL_C_UNITS, f"a3_{tag}{salt}")
    payload.update(tag=tag, region=region, window=window, extent_factor=factor)
    _write_atomic(path, payload)
    return "done"


def stage_units(workers: int, limit: int | None, factor: float = 1.0) -> None:
    jobs = [(r, w, factor) for r in REGIONS for w in WINDOWS]
    todo = [j for j in jobs if not (_units_dir(factor) / f"{j[0]}__{j[1]}.json").exists()]
    if limit:
        todo = todo[:limit]
    print(f"units: {len(jobs)} total, {len(todo)} to compute", flush=True)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_unit_worker, todo)):
            print(f"[{i+1}/{len(todo)}] {(time.time()-t0)/60:.1f} min", flush=True)


# --------------------------------------------------------------------------
# tiles stage
# --------------------------------------------------------------------------

_G: dict[str, object] = {}


def _tile_worker(t: dict) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    season = _G["season"]
    path = OUT / "tiles" / str(season) / f"{t['tid']}.json"
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
    payload = anchored_payload(uu, vv, la, lo, ELLS_FINE, ELL_C_TILES,
                               f"glob_{t['tid']}_{season}")
    payload.update(tid=t["tid"], season=season)
    _write_atomic(path, payload)
    return "done"


def stage_tiles(season: str, workers: int, limit: int | None) -> None:
    import xarray as xr
    tiles = tile_grid()
    todo = [t for t in tiles
            if not (OUT / "tiles" / season / f"{t['tid']}.json").exists()]
    if limit:
        todo = todo[:limit]
    print(f"{season}: {len(tiles)} tiles, {len(todo)} to compute", flush=True)
    if not todo:
        return
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
        for i, _ in enumerate(pool.imap_unordered(_tile_worker, todo, chunksize=4)):
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                el = (time.time() - t0) / 60
                print(f"[{i+1}/{len(todo)}] {el:.1f} min elapsed, "
                      f"{el/(i+1)*(len(todo)-i-1):.1f} min left", flush=True)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def _load(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    idx = {n: k for k, n in enumerate(d["names"])}
    real = np.asarray(d["real"])
    med = np.asarray(d["sur_med"])
    d["anch"] = {n: real[k] - med[k] for n, k in idx.items()}      # [full, H1, H2]
    d["floor"] = {n: d["sur_q95"][k] - d["sur_q05"][k] for n, k in idx.items()}
    return d


def _pairs(n_env: int, sep: int) -> list[str]:
    return [f"rho_{i}_{i+sep}" for i in range(n_env - sep)]


def _wilcoxon_greater(x: np.ndarray) -> float:
    return float(wilcoxon(x, alternative="greater").pvalue)


def _boot_median_ci(x: np.ndarray, rng, n: int) -> list[float]:
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def _classify(med: float, ci: list[float], p_gt: float, p_lt: float) -> str:
    if -0.05 <= ci[0] and ci[1] <= 0.05:
        return "LOCAL_CASCADE"
    if med > 0 and p_gt < 0.05:
        return "NONLOCAL"
    if med < 0 and p_lt < 0.05:
        return "SCREENED"
    return "INDETERMINATE"


def tests_units(rng) -> dict:
    n_env = len(ELLS_KM)
    units = {}
    for r in REGIONS:
        for w in WINDOWS:
            units[(r, w)] = _load(OUT / "units" / f"{r}__{w}.json")
    res: dict = {"n_units": len(units)}

    def unit_val(d, names, h=0):
        return float(np.mean([d["anch"][n][h] for n in names]))

    def region_vals(names):
        return np.asarray([np.mean([unit_val(units[(r, w)], names) for w in WINDOWS])
                           for r in REGIONS])

    # C27-1
    mine, stored = [], []
    for (r, w), d in units.items():
        f = A3 / "units" / f"{r}__{w}.json"
        if f.exists():
            mine.append(unit_val(d, _pairs(n_env, 1)))
            stored.append(json.loads(f.read_text())["P_anch_mean"])
    rho_c = float(spearmanr(mine, stored).statistic)
    res["C27_1_units"] = {"n": len(mine), "rho": rho_c,
                          "max_abs_diff": float(np.max(np.abs(np.asarray(mine) - np.asarray(stored)))),
                          "pass": bool(rho_c >= 0.99)}

    # A1 + separation profile
    prof = {}
    for sep in range(1, n_env):
        x = region_vals(_pairs(n_env, sep))
        prof[f"sep{sep}"] = {"region_median": float(np.median(x)),
                             "region_min": float(x.min()), "region_max": float(x.max()),
                             "p_wilcoxon_gt": _wilcoxon_greater(x)}
    x2 = region_vals(_pairs(n_env, 2))
    res["H27_A1"] = {"region_values": dict(zip(REGIONS, map(float, x2))),
                     "median": float(np.median(x2)),
                     "p_wilcoxon_gt": prof["sep2"]["p_wilcoxon_gt"],
                     "pass": bool(np.median(x2) > 0 and prof["sep2"]["p_wilcoxon_gt"] < 0.05),
                     "separation_profile": prof,
                     "per_pair_region_median": {
                         n: float(np.median(region_vals([n]))) for n in
                         [f"rho_{i}_{j}" for i in range(n_env) for j in range(i + 1, n_env)]}}

    # A2
    pn = [f"part2_{i}" for i in range(n_env - 2)]
    xp = region_vals(pn)
    ci = _boot_median_ci(xp, rng, N_BOOT_REGION)
    p_gt = _wilcoxon_greater(xp)
    p_lt = _wilcoxon_greater(-xp)
    res["H27_A2"] = {"region_values": dict(zip(REGIONS, map(float, xp))),
                     "median": float(np.median(xp)), "ci95": ci,
                     "p_gt": p_gt, "p_lt": p_lt,
                     "per_triplet_region_median": {
                         n: float(np.median(region_vals([n]))) for n in pn},
                     "class": _classify(float(np.median(xp)), ci, p_gt, p_lt)}

    # B2(i) units
    xd = region_vals([f"D_{f}" for f in FINE_F])
    xv = region_vals([f"V_{f}" for f in FINE_F])
    xl = region_vals([f"Dlag_{f}" for f in FINE_F])
    res["H27_B2i_units"] = {"D_region_values": dict(zip(REGIONS, map(float, xd))),
                            "D_median": float(np.median(xd)),
                            "p_wilcoxon_gt": _wilcoxon_greater(xd),
                            "V_median": float(np.median(xv)),
                            "Dlag_median": float(np.median(xl)),
                            "pass": bool(np.median(xd) > 0 and _wilcoxon_greater(xd) < 0.05)}

    # S4 (reported)
    lam, sig2 = [], []
    a_coarse = np.asarray(ELLS_KM)
    for (r, w), d in units.items():
        xs, ys = [], []
        for i in range(n_env):
            for j in range(i + 1, n_env):
                xs.append(-np.log(a_coarse[j]))
                ys.append(d["anch"][f"cov_{i}_{j}"][0])
        slope = float(np.polyfit(xs, ys, 1)[0])
        f = A3 / "units" / f"{r}__{w}.json"
        if f.exists():
            lam.append(slope)
            sig2.append(json.loads(f.read_text())["INT_sig2_real"][0])
    res["S4_reported"] = {"n": len(lam), "lambda2_median": float(np.median(lam)),
                          "rho_lambda2_INTsig2": float(spearmanr(lam, sig2).statistic)}

    res["noise_floor_q05_q95_width_median"] = {
        k: float(np.median([np.mean([d["floor"][n] for n in names])
                            for d in units.values()]))
        for k, names in {"sep1": _pairs(n_env, 1), "sep2": _pairs(n_env, 2),
                         "part2": pn, "D": [f"D_{f}" for f in FINE_F]}.items()}
    return res


def tests_tiles(rng) -> dict:
    n_env = len(ELLS_FINE)
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    shards = {s: {p.stem: _load(p) for p in sorted((OUT / "tiles" / s).glob("t*.json"))}
              for s in SEASONS}
    armb = {}
    for s in SEASONS:
        d = json.loads((ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))
        armb[s] = {r["tid"]: r for r in d["tiles"]}
    common = sorted(set.intersection(*[set(shards[s]) for s in SEASONS],
                                     *[set(armb[s]) for s in SEASONS]))
    cov = tile_covariates([by_tid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    tiles = [by_tid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}
    res: dict = {"n_common": len(common), "n_kept": len(keep)}

    def val(names, h=0):
        return np.asarray([np.mean([np.mean([shards[s][t]["anch"][n][h] for n in names])
                                    for s in SEASONS]) for t in keep])

    adj = _pairs(n_env, 1)
    # C27-1
    stored = np.asarray([np.mean([armb[s][t]["P_fine_anch_mean"] for s in SEASONS])
                         for t in keep])
    p_mean = val(adj)
    rho_c = float(spearmanr(p_mean, stored).statistic)
    res["C27_1_tiles"] = {"rho": rho_c,
                          "max_abs_diff": float(np.max(np.abs(p_mean - stored))),
                          "pass": bool(rho_c >= 0.99)}

    # blocks
    block = np.asarray([int(by_tid[t]["lon_c"] // BLOCK_DEG) * 10
                        + int((by_tid[t]["lat_c"] + 60.0) // BLOCK_DEG) for t in keep])
    ublocks = np.unique(block)
    members = [np.where(block == b)[0] for b in ublocks]

    def block_resample():
        pick = rng.integers(0, len(ublocks), size=len(ublocks))
        return np.concatenate([members[k] for k in pick])

    # covariates
    eke = np.asarray([np.mean([armb[s][t]["eke_syn"] for s in SEASONS]) for t in keep])
    cols = {n: np.asarray([cov[t][n] for t in keep]) for n in ALL_COV if n != "eke_syn"}
    cols["eke_syn"] = eke
    X = np.column_stack([(cols[n] - cols[n].mean()) / cols[n].std() for n in ALL_COV])
    A = np.column_stack([np.ones(len(keep)), X])
    i_cape, i_eke = ALL_COV.index("cape_mean") + 1, ALL_COV.index("eke_syn") + 1

    def betas(y, rows=None):
        a, yy = (A, y) if rows is None else (A[rows], y[rows])
        return np.linalg.lstsq(a, yy, rcond=None)[0]

    # B1
    Y = [val([n]) for n in adj]
    b = [betas(y) for y in Y]
    obs = {"beta_cape": [float(x[i_cape]) for x in b],
           "beta_eke": [float(x[i_eke]) for x in b],
           "P_step_median": [float(np.median(y)) for y in Y]}
    boot = np.empty((N_BOOT_BLOCK, 4))
    for k in range(N_BOOT_BLOCK):
        rows = block_resample()
        bb = [betas(y, rows) for y in Y]
        boot[k] = [bb[-1][i_cape] - bb[0][i_cape], bb[0][i_cape],
                   bb[-1][i_eke] - bb[0][i_eke], bb[1][i_cape]]
    q = lambda c: [float(np.quantile(boot[:, c], 0.025)), float(np.quantile(boot[:, c], 0.975))]
    d_cape = obs["beta_cape"][-1] - obs["beta_cape"][0]
    p_b1 = float((1 + np.sum(boot[:, 0] <= 0)) / (N_BOOT_BLOCK + 1))
    res["H27_B1"] = {**obs, "delta_cape": d_cape, "delta_cape_ci95": q(0),
                     "beta_cape_k1_ci95": q(1),
                     "delta_eke": obs["beta_eke"][-1] - obs["beta_eke"][0],
                     "delta_eke_ci95": q(2), "p_boot_delta_cape_le0": p_b1,
                     "profile_peaks_at_k2": bool(
                         obs["beta_cape"][1] < min(obs["beta_cape"][0], obs["beta_cape"][2])),
                     "pass": bool(q(0)[0] > 0 and q(1)[1] < 0)}

    # A1 / A2 tile replicates
    def med_ci(x):
        m = np.empty(N_BOOT_BLOCK)
        for k in range(N_BOOT_BLOCK):
            m[k] = np.median(x[block_resample()])
        return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]

    x2 = val(_pairs(n_env, 2))
    x3 = val(_pairs(n_env, 3))
    xp = val([f"part2_{i}" for i in range(n_env - 2)])
    ci_p = med_ci(xp)
    res["A1_tiles_replicate"] = {"sep1_median": float(np.median(p_mean)),
                                 "sep2_median": float(np.median(x2)), "sep2_ci95": med_ci(x2),
                                 "sep3_median": float(np.median(x3)), "sep3_ci95": med_ci(x3)}
    res["A2_tiles_replicate"] = {
        "median": float(np.median(xp)), "ci95": ci_p,
        "class": ("LOCAL_CASCADE" if (-0.05 <= ci_p[0] and ci_p[1] <= 0.05) else
                  "NONLOCAL" if ci_p[0] > 0 else "SCREENED" if ci_p[1] < 0 else "INDETERMINATE"),
        "rho_with_cape": float(spearmanr(xp, cols["cape_mean"]).statistic),
        "rho_with_eke": float(spearmanr(xp, eke).statistic)}

    # B2
    dn = [f"D_{f}" for f in FINE_F]
    D = val(dn)
    Dl = val([f"Dlag_{f}" for f in FINE_F])
    V = val([f"V_{f}" for f in FINE_F])
    ci_d = med_ci(D)
    res["H27_B2i_tiles"] = {"D_median": float(np.median(D)), "D_ci95": ci_d,
                            "V_median": float(np.median(V)),
                            "per_F_median": {n: float(np.median(val([n]))) for n in dn},
                            "pass": bool(ci_d[0] > 0)}
    res["C27_2"] = {"Dlag_median": float(np.median(Dl)),
                    "ratio": float(abs(np.median(Dl)) / max(abs(np.median(D)), 1e-12)),
                    "pass": bool(abs(np.median(Dl)) <= 0.25 * abs(np.median(D)))}

    D1, D2 = val(dn, 1), val(dn, 2)
    P1, P2 = val(adj, 1), val(adj, 2)

    def rho_x(d1, d2):
        return 0.5 * (spearmanr(d1, P2).statistic + spearmanr(d2, P1).statistic)

    obs_rho = float(rho_x(D1, D2))
    DD = np.column_stack([D1, D2])
    null = np.empty(N_ROT)
    for k in range(N_ROT):
        off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
        r = rotate_cov(DD, tiles, off, tid_index)
        null[k] = rho_x(r[:, 0], r[:, 1])
    p_rot = float((1 + np.sum(null >= obs_rho)) / (N_ROT + 1))
    res["H27_B2ii"] = {"rho_x": obs_rho, "p_rot": p_rot,
                       "null_q95": float(np.quantile(null, 0.95)),
                       "rho_same_half_reference": float(spearmanr(D, p_mean).statistic),
                       "pass": bool(obs_rho >= 0.20 and p_rot < 0.05)}

    bD = betas(D)
    nullb = np.empty((N_ROT, 2))
    for k in range(N_ROT):
        off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
        Ar = np.column_stack([np.ones(len(keep)), rotate_cov(X, tiles, off, tid_index)])
        br = np.linalg.lstsq(Ar, D, rcond=None)[0]
        nullb[k] = [br[i_cape], br[i_eke]]
    res["H27_B2iii_reported"] = {
        "beta_cape": float(bD[i_cape]), "beta_eke": float(bD[i_eke]),
        "p_rot_cape_2s": float((1 + np.sum(np.abs(nullb[:, 0]) >= abs(bD[i_cape]))) / (N_ROT + 1)),
        "p_rot_eke_2s": float((1 + np.sum(np.abs(nullb[:, 1]) >= abs(bD[i_eke]))) / (N_ROT + 1)),
        "rho_D_cape": float(spearmanr(D, cols["cape_mean"]).statistic),
        "rho_D_eke": float(spearmanr(D, eke).statistic)}

    res["noise_floor_q05_q95_width_median"] = {
        k: float(np.median([np.mean([shards[s][t]["floor"][n] for n in names
                                     for s in SEASONS]) for t in keep]))
        for k, names in {"sep1": adj, "sep2": _pairs(n_env, 2),
                         "part2": [f"part2_{i}" for i in range(n_env - 2)],
                         "D": dn}.items()}
    return res


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"protocol": str(PROTOCOL), "version": CONFIG_VERSION, "seed": SEED,
                 "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    res["units"] = tests_units(rng)
    res["tiles"] = tests_tiles(rng)

    controls_ok = (res["units"]["C27_1_units"]["pass"]
                   and res["tiles"]["C27_1_tiles"]["pass"])
    b2_ok = res["tiles"]["H27_B2ii"]["pass"] and res["tiles"]["C27_2"]["pass"]
    passed = {"A1": res["units"]["H27_A1"]["pass"],
              "B1": res["tiles"]["H27_B1"]["pass"],
              "B2ii": bool(b2_ok)}
    pv = {"A1": res["units"]["H27_A1"]["p_wilcoxon_gt"],
          "B1": res["tiles"]["H27_B1"]["p_boot_delta_cape_le0"],
          "B2ii": res["tiles"]["H27_B2ii"]["p_rot"]}
    order = sorted(pv, key=pv.get)
    holm, run = {}, 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * pv[name]))
        holm[name] = run
    res["primary"] = {"pass": passed, "p": pv, "p_holm": holm}
    res["A2_class"] = res["units"]["H27_A2"]["class"]
    if not controls_ok:
        res["VERDICT"] = "VOID_REPRODUCTION_FAILED"
    elif all(passed.values()):
        res["VERDICT"] = "MECHANISM_SUPPORTED"
    elif any(passed.values()):
        res["VERDICT"] = "PARTIAL(" + ",".join(k for k, v in passed.items() if v) + ")"
    else:
        res["VERDICT"] = "NOT_SUPPORTED"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"VERDICT": res["VERDICT"], "A2_class": res["A2_class"],
                      "primary": res["primary"]}, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["units", "tiles", "tests"])
    ap.add_argument("--season", choices=list(SEASONS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--extent", type=float, default=1.0,
                    help="km_crop factor for the units stage (Phase 28 uses 0.7)")
    args = ap.parse_args()
    if args.stage == "units":
        stage_units(args.workers, args.limit, args.extent)
    elif args.stage == "tiles":
        if not args.season:
            ap.error("--season required")
        stage_tiles(args.season, args.workers, args.limit)
    else:
        stage_tests()


if __name__ == "__main__":
    main()
