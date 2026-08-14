#!/usr/bin/env python3
"""Phase-6A: source-sink dynamics of tile-level curvature.

Protocol: docs/PROTOCOL_PHASE6_SOURCE_SINK_DYNAMICS.md (frozen 2026-08-12).
Model per tile (tile-demeaned within region-window):
  dC_t = a + alpha*E_t - beta*R_t - gamma*C_t
C = log tile fine-band ||F||, E = log(1+CAPE), R = log(precip+1e-6).
Fit set: R5-R8 (16 rws). Validation: R9-R12 (16 rws).
C6a1: beta>0 in >=12/16 fit rws. C6a2: alpha>0 in >=12/16.
C6a3: pooled-coefficient model beats per-rw AR(1)-only in one-step MSE
in >=12/16 validation rws.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B5a_source import (
        NTX,
        NTY,
        WARMUP,
        tile_fnorm_series,
        tile_slices,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    from experiment_B5a_source import (  # type: ignore
        NTX,
        NTY,
        WARMUP,
        tile_fnorm_series,
        tile_slices,
    )
    from experiment_scale_gravity_einstein_box_era import _load_vector_fields  # type: ignore

SEED = 20260811
FIT_REGIONS = ("R5_SPCZ", "R6_SATL", "R7_CONGO", "R8_AUS")
VAL_REGIONS = ("R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM")
OUT = _HERE / "results" / "experiment_B6_source_sink"


def _tile_means(path: Path, ny: int, nx: int) -> np.ndarray:
    ds = xr.open_dataset(path)
    var = next(v for v in ds.data_vars)
    da = ds[var].squeeze()
    tdim = next(d for d in da.dims if "time" in d or d == "valid_time")
    arr = np.nan_to_num(np.asarray(da.transpose(tdim, ...).values, dtype=float))
    ds.close()
    out = np.zeros((NTY * NTX, arr.shape[0]))
    for k, (sy, sx) in enumerate(tile_slices(ny, nx)):
        out[k] = arr[:, sy, sx].mean(axis=(1, 2))
    return out


def _series_for_rw(args_tuple) -> dict | None:
    wind, precip, cape = args_tuple
    os.environ["OMP_NUM_THREADS"] = "1"
    tag = wind.stem.replace("era5_wind850_", "")
    cached = OUT / f"b6a_series_{tag}.npz"
    if cached.exists():
        return {"tag": tag}
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=wind, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    c = np.log(tile_fnorm_series(u, v, lat, lon) + 1e-12)
    r = np.log(_tile_means(precip, u.shape[1], u.shape[2]) + 1e-6)
    e = np.log(1.0 + np.maximum(_tile_means(cape, u.shape[1], u.shape[2]), 0.0))
    nt = min(c.shape[1], r.shape[1], e.shape[1])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, C=c[:, WARMUP:nt], R=r[:, WARMUP:nt], E=e[:, WARMUP:nt])
    print(f"[{tag}] series cached", flush=True)
    return {"tag": tag}


def _design(tag: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / f"b6a_series_{tag}.npz")
    c, r, e = z["C"], z["R"], z["E"]
    c = c - c.mean(axis=1, keepdims=True)
    r = r - r.mean(axis=1, keepdims=True)
    e = e - e.mean(axis=1, keepdims=True)
    dc = (c[:, 1:] - c[:, :-1]).ravel()
    x = np.column_stack([
        np.ones_like(dc),
        e[:, :-1].ravel(),
        r[:, :-1].ravel(),
        c[:, :-1].ravel(),
    ])
    return x, dc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-dir", type=Path, default=Path("data/b2b"))
    parser.add_argument("--precip-dir", type=Path, default=Path("data/b5precip"))
    parser.add_argument("--cape-dir", type=Path, default=Path("data/b6cape"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    jobs = []
    for wf in sorted(args.wind_dir.glob("era5_wind850_*.nc")):
        tag = wf.stem.replace("era5_wind850_", "")
        pf = args.precip_dir / f"era5_precip_{tag}.nc"
        cf = args.cape_dir / f"era5_cape6h_{tag}.nc"
        if pf.exists() and cf.exists():
            jobs.append((wf, pf, cf))
    print(f"{len(jobs)} region-windows with wind+precip+cape", flush=True)
    if len(jobs) < 32:
        print("WARNING: fewer than 32 region-windows; criteria need all 32.")

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        pool.map(_series_for_rw, jobs)

    tags = [w.stem.replace("era5_wind850_", "") for w, _, _ in jobs]
    fit_tags = [t for t in tags if t.split("__")[0] in FIT_REGIONS]
    val_tags = [t for t in tags if t.split("__")[0] in VAL_REGIONS]

    per_rw = {}
    for t in fit_tags:
        x, y = _design(t)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        per_rw[t] = {"alpha": float(coef[1]), "beta": float(-coef[2]),
                     "gamma": float(-coef[3])}
    n_beta = sum(1 for v in per_rw.values() if v["beta"] > 0)
    n_alpha = sum(1 for v in per_rw.values() if v["alpha"] > 0)

    xs, ys = zip(*[_design(t) for t in fit_tags])
    x_pool, y_pool = np.vstack(xs), np.concatenate(ys)
    coef_pool, *_ = np.linalg.lstsq(x_pool, y_pool, rcond=None)

    n_val_win = 0
    val_detail = {}
    for t in val_tags:
        x, y = _design(t)
        mse_full = float(np.mean((y - x @ coef_pool) ** 2))
        x_ar = x[:, [0, 3]]
        coef_ar, *_ = np.linalg.lstsq(x_ar, y, rcond=None)
        mse_ar = float(np.mean((y - x_ar @ coef_ar) ** 2))
        win = mse_full < mse_ar
        n_val_win += int(win)
        val_detail[t] = {"mse_full": mse_full, "mse_ar1": mse_ar, "win": bool(win)}

    c6a1 = bool(n_beta >= 12 and len(fit_tags) >= 16)
    c6a2 = bool(n_alpha >= 12 and len(fit_tags) >= 16)
    c6a3 = bool(n_val_win >= 12 and len(val_tags) >= 16)
    summary = {
        "n_fit": len(fit_tags), "n_val": len(val_tags),
        "beta_positive": n_beta, "alpha_positive": n_alpha,
        "pooled": {"alpha": float(coef_pool[1]), "beta": float(-coef_pool[2]),
                   "gamma": float(-coef_pool[3]),
                   "relax_time_steps": float(1.0 / max(-coef_pool[3], 1e-9))},
        "validation_wins": n_val_win,
        "C6a1_discharge": c6a1, "C6a2_charge": c6a2, "C6a3_heldout": c6a3,
        "PASS": bool(c6a1 and c6a2 and c6a3),
        "per_rw_fit": per_rw, "validation_detail": val_detail,
    }
    (OUT / "b6a_source_sink.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("n_fit", "n_val", "beta_positive", "alpha_positive",
                       "pooled", "validation_wins", "C6a1_discharge",
                       "C6a2_charge", "C6a3_heldout", "PASS")}, indent=2))


if __name__ == "__main__":
    main()
