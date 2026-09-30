#!/usr/bin/env python3
"""EXPLORATORY (not frozen) test T9: what are the active objects?

Register and expectations (written before the run):
docs/EXPLORATION_PHYSICS_2026-09-29.md, test T9.

For the 12 regional boxes, windows 2023 JFM and JAS, Arm-A tiling (3 x 4):
spatial rank correlation, per 6-h step, between the log-envelope of the
850-hPa vorticity (bands 50-100, 100-200, 200-400 km) and organiser fields
smoothed at the same scale:
  O1  ln(precipitation rate)
  O2  ln|grad(total column water vapour)|   (moisture fronts)
  O3  ln|integrated vapour transport|
  O4  log-envelope of the divergence in the same band
On-disk data only: data/b3 (wind), data/b7budget/extracted (tp, tcwv),
data/b6ivt.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B18_equal_km_regions import REGIONS, km_crop  # noqa: E402
from experiment_B1_true_flux_baselines import (  # noqa: E402
    _grid_geometry, _interior_mask,
)
from experiment_B20_p_geography import (  # noqa: E402
    ELLS_FINE, TILE_COLS, TILE_ROWS, tile_slices,
)
from experiment_B2_scale_irreversibility import (  # noqa: E402
    compute_vorticity, gaussian_bar_grouped,
)
from explore_physics_tiles import (  # noqa: E402
    band_fields, compute_divergence, log_envelopes,
)

OUT = _HERE / "results" / "explore_physics"
ARMA = _HERE / "results" / "experiment_B20_p_geography"
WINDOWS = ("W9_2023JFM", "W10_2023JAS")
EPS = 1e-12
BANDS = (1, 2, 3)


def _open(path, names):
    import xarray as xr
    ds = xr.open_dataset(path)
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    out = [np.asarray(ds[n].squeeze().values, dtype=np.float32) for n in names]
    ds.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        out = [a[:, ::-1, :] for a in out]
    return lat, lon, out


def _spearman_rows(a, b):
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    r = np.where(den > 1e-9, (ra * rb).sum(axis=1) / np.maximum(den, 1e-9), np.nan)
    # steps where a field is (almost) constant carry no information
    bad = (np.ptp(a, axis=1) < 1e-9) | (np.ptp(b, axis=1) < 1e-9)
    r[bad] = np.nan
    return r


def _rank_r2(y, X):
    """Per-step R2 of rank(y) on rank(X columns)."""
    nt = y.shape[0]
    out = np.full(nt, np.nan)
    for t in range(nt):
        cols = []
        for x in X:
            if np.ptp(x[t]) < 1e-9:
                continue
            r = np.argsort(np.argsort(x[t])).astype(float)
            cols.append((r - r.mean()) / r.std())
        if not cols:
            continue
        ry = np.argsort(np.argsort(y[t])).astype(float)
        ry = (ry - ry.mean()) / ry.std()
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, ry, rcond=None)
        out[t] = 1.0 - np.mean((ry - A @ coef) ** 2)
    return out


def _one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    region, window = job
    tag = f"{region}__{window}"
    lat, lon, (u, v) = _open(f"data/b3/era5_wind850_{tag}.nc", ["u", "v"])
    base = Path(f"data/b7budget/extracted/era5_budget_{tag}")
    _, _, (tp,) = _open(base / "data_stream-oper_stepType-accum.nc", ["tp"])
    _, _, (tcwv,) = _open(base / "data_stream-oper_stepType-instant.nc", ["tcwv"])
    _, _, (qe, qn) = _open(f"data/b6ivt/era5_ivt_{tag}.nc", ["viwve", "viwvn"])
    nt = min(a.shape[0] for a in (u, tp, tcwv, qe))
    la, lo, u, v = km_crop(lat, lon, u[:nt], v[:nt], region, 1.0)
    _, _, tp, tcwv = km_crop(lat, lon, tp[:nt], tcwv[:nt], region, 1.0)
    _, _, qe, qn = km_crop(lat, lon, qe[:nt], qn[:nt], region, 1.0)
    arma = json.loads((ARMA / f"tiles_{tag}.json").read_text(encoding="utf-8"))["tiles"]
    rows = tile_slices(u.shape[1], TILE_ROWS)
    cols = tile_slices(u.shape[2], TILE_COLS)
    out = []
    for ri, rs in enumerate(rows):
        for ci, cs in enumerate(cols):
            tid = f"r{ri}c{ci}"
            tla, tlo = la[rs], lo[cs]
            dx_km, dy_km = _grid_geometry(tla, tlo)
            ny, nx = len(tla), len(tlo)
            mask = _interior_mask(ny, nx, ELLS_FINE[-1], dx_km, dy_km)
            uu, vv = u[:, rs, cs], v[:, rs, cs]
            om = np.nan_to_num(compute_vorticity(uu, vv, tla, tlo))
            dv = np.nan_to_num(compute_divergence(uu, vv, tla, tlo))
            b_v, _ = band_fields(om, dx_km, dy_km)
            b_d, _ = band_fields(dv, dx_km, dy_km)
            e_v = log_envelopes(b_v, dx_km, dy_km)
            e_d = log_envelopes(b_d, dx_km, dy_km)
            p = np.maximum(tp[:, rs, cs], 0.0) * 1000.0          # mm per hour
            w = tcwv[:, rs, cs]
            lat_sign = 1.0 if tla[1] > tla[0] else -1.0
            gx = np.gradient(w, axis=2) / dx_km[None, :, None]
            gy = np.gradient(w, axis=1) / dy_km * lat_sign
            gw = np.sqrt(gx ** 2 + gy ** 2)
            iv = np.sqrt(qe[:, rs, cs] ** 2 + qn[:, rs, cs] ** 2)
            res = {"region": region, "window": window, "tile": tid,
                   "lat_c": float(np.mean(tla)), "lon_c": float(np.mean(tlo)),
                   "P": float(arma[tid]["P_fine_anch_mean"]),
                   "eke_syn": float(arma[tid]["eke_syn"]),
                   "wet_frac": float(np.mean(p[:, mask] > 0.1)),
                   "tp_mean": float(np.mean(p[:, mask]))}
            acc = {k: [] for k in ("O1", "O2", "O3", "O4", "R2")}
            valid = {k: [] for k in ("O1",)}
            for b in BANDS:
                sig = ELLS_FINE[b] / np.sqrt(12.0)
                ev = e_v[b][:, mask]
                o1 = np.log(gaussian_bar_grouped(p, sig, dx_km, dy_km) + 0.01)[:, mask]
                o2 = np.log(gaussian_bar_grouped(gw, sig, dx_km, dy_km) + EPS)[:, mask]
                o3 = np.log(gaussian_bar_grouped(iv, sig, dx_km, dy_km) + EPS)[:, mask]
                o4 = e_d[b][:, mask]
                r1 = _spearman_rows(ev, o1)
                acc["O1"].append(np.nanmedian(r1))
                valid["O1"].append(np.mean(np.isfinite(r1)))
                acc["O2"].append(np.nanmedian(_spearman_rows(ev, o2)))
                acc["O3"].append(np.nanmedian(_spearman_rows(ev, o3)))
                acc["O4"].append(np.nanmedian(_spearman_rows(ev, o4)))
                acc["R2"].append(np.nanmedian(_rank_r2(ev, [o1, o2, o3])))
            for k, vals in acc.items():
                res[f"c_{k}"] = float(np.mean(vals))
                res[f"c_{k}_bands"] = [float(x) for x in vals]
            res["valid_frac_O1"] = float(np.mean(valid["O1"]))
            out.append(res)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, w) for r in REGIONS for w in WINDOWS]
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(3) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs)):
            rows.extend(r)
            print(f"[{i+1}/{len(jobs)}]", flush=True)
    (OUT / "objects_T9.json").write_text(json.dumps(rows), encoding="utf-8")
    print("done", len(rows))


if __name__ == "__main__":
    main()
