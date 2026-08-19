#!/usr/bin/env python3
"""AUDIT-3 (reduced): do the AUDIT-1/2 verdicts survive a bigger box, a
wider band ladder, another year and another season?

Protocol: docs/PROTOCOL_AUDIT3_SCALE_YEAR_ROBUSTNESS.md (frozen
2026-08-19). 12 equal-km Phase-18 boxes x 2 windows (W9_2023JFM,
W12_2024JAS) = 24 units, full ELLS_KM ladder (50-1600 km), 99 phase
surrogates per unit. No downloads.

Stages: --stage units (resumable, atomic shard per unit) | tests | all
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
    from clean_experiments.experiment_A2b_splithalf import all_profiles
    from clean_experiments.experiment_B18_equal_km_regions import REGIONS, km_crop
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask, phase_randomize,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM, compute_vorticity,
    )
    from clean_experiments.experiment_B20_p_geography import N_SUR
except ImportError:
    from experiment_A2b_splithalf import all_profiles  # type: ignore
    from experiment_B18_equal_km_regions import REGIONS, km_crop  # type: ignore
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry, _interior_mask, phase_randomize,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM, compute_vorticity,
    )
    from experiment_B20_p_geography import N_SUR  # type: ignore

CONFIG_VERSION = "a3_v1"
SEED_SUR = 20260811
OUT = _HERE / "results" / "experiment_A3_robustness"
B3 = Path("data/b3")
WINDOWS = ("W9_2023JFM", "W12_2024JAS")
STATS = ("P", "P_lin", "R_AM_cyc", "P_cascade", "INT_sig2", "INT_flat", "INT_mu")
N_STEPS = len(ELLS_KM) - 1


def shard_path(tag: str) -> Path:
    return OUT / "units" / f"{tag}.json"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _unit_worker(job: tuple[str, str]) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    region, window = job
    tag = f"{region}__{window}"
    path = shard_path(tag)
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
    la, lo, uu, vv = km_crop(lat, lon, u, v, region, 1.0)
    dx_km, dy_km = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_KM[-1], dx_km, dy_km)
    lat_sign = float(np.sign(0.5 * (la[0] + la[-1]))) or 1.0

    omega = compute_vorticity(uu, vv, la, lo)
    real = all_profiles(omega, dx_km, dy_km, mask, ELLS_KM, lat_sign)

    tcode = zlib.crc32(f"a3_{tag}".encode()) % 100000
    seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(N_SUR)
    sur = {k: np.empty((N_SUR, N_STEPS)) for k in STATS}
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        us, vs = phase_randomize(uu, vv, rng)
        q = all_profiles(compute_vorticity(us, vs, la, lo), dx_km, dy_km,
                         mask, ELLS_KM, lat_sign)
        for k in STATS:
            sur[k][j] = q[k]

    payload = {"tag": tag, "region": region, "window": window,
               "version": CONFIG_VERSION, "n_cells": [int(uu.shape[1]), int(uu.shape[2])],
               "n_steps_time": int(uu.shape[0])}
    for k in STATS:
        med = np.median(sur[k], axis=0)
        payload[f"{k}_real"] = real[k]
        payload[f"{k}_anch_mean"] = float(np.mean(np.asarray(real[k]) - med))
    _write_atomic(path, payload)
    return "done"


def stage_units(workers: int, limit: int | None) -> None:
    jobs = [(r, w) for r in REGIONS for w in WINDOWS]
    todo = [j for j in jobs if not shard_path(f"{j[0]}__{j[1]}").exists()]
    if limit:
        todo = todo[:limit]
    print(f"{len(jobs)} units, {len(todo)} to compute", flush=True)
    if not todo:
        return
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_unit_worker, todo)):
            el = time.time() - t0
            print(f"[{i+1}/{len(todo)}] {el/60:.1f} min, "
                  f"{(len(todo)-i-1)/((i+1)/el)/60:.1f} min left", flush=True)


def stage_tests() -> dict:
    rows = []
    for f in sorted((OUT / "units").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("version") == CONFIG_VERSION:
            rows.append(r)
    res = {"protocol": "docs/PROTOCOL_AUDIT3_SCALE_YEAR_ROBUSTNESS.md",
           "version": CONFIG_VERSION, "n_units": len(rows),
           "ells_km": ELLS_KM, "windows": list(WINDOWS)}
    if len(rows) < 24:
        res["note"] = "incomplete sample"
    P = np.array([r["P_anch_mean"] for r in rows])

    def rho_to(stat):
        x = np.array([r[f"{stat}_anch_mean"] for r in rows])
        return float(spearmanr(x, P).statistic)

    def ladder(rho, names):
        a = abs(rho)
        if 0.45 <= a <= 0.75:
            return "INCONCLUSIVE"
        return names[0] if a >= 0.90 else (names[1] if a >= 0.60 else names[2])

    r_am = rho_to("R_AM_cyc")
    r_ca = rho_to("P_cascade")
    res["A3a"] = {"rho_P_vs_R_AM_cyc": r_am,
                  "verdict": ladder(r_am, ("AM_IDENTITY_AT_BOX_SCALE", "AM_RELATED",
                                           "AM_DISTINCT_ROBUST"))}
    res["A3b"] = {"rho_P_vs_P_cascade": r_ca,
                  "verdict": ladder(r_ca, ("CASCADE_IDENTITY_AT_BOX_SCALE",
                                           "CASCADE_RELATED", "CASCADE_DISTINCT_ROBUST"))}
    res["A3c"] = {"rho_P_vs_P_lin": rho_to("P_lin"),
                  "rho_P_vs_INT_sig2": rho_to("INT_sig2"),
                  "rho_P_vs_INT_flat": rho_to("INT_flat"),
                  "rho_P_vs_INT_mu": rho_to("INT_mu")}
    # per-window split (reported)
    res["per_window"] = {}
    for w in WINDOWS:
        idx = [i for i, r in enumerate(rows) if r["window"] == w]
        if len(idx) < 6:
            continue
        Pw = P[idx]
        res["per_window"][w] = {
            k: float(spearmanr(np.array([rows[i][f"{k}_anch_mean"] for i in idx]), Pw).statistic)
            for k in ("R_AM_cyc", "P_cascade", "P_lin", "INT_sig2")}
    res["medians_real_vs_anchored"] = {
        k: {"real": float(np.median([np.mean(r[f"{k}_real"]) for r in rows])),
            "anchored": float(np.median([r[f"{k}_anch_mean"] for r in rows]))}
        for k in ("P", "P_cascade", "R_AM_cyc")}
    res["VERDICT"] = f"{res['A3a']['verdict']} + {res['A3b']['verdict']}"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["units", "tests", "all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("units", "all"):
        stage_units(args.workers, args.limit)
    if args.stage in ("tests", "all"):
        res = stage_tests()
        (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
