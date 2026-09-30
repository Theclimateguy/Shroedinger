#!/usr/bin/env python3
"""EXPLORATORY (not frozen): anchored fine-P on the programme's 12 regional
boxes (Arm-A tiling, 3 x 4 tiles per km box), with the isotropic-grid
correction (test T11) and per-band-pair output.

Register: docs/EXPLORATION_PHYSICS_2026-09-29.md (tests T4, T9, T11).
On-disk data only:
  --set levels   data/b5w500 (500 hPa) + matching 850 hPa windows (b1, b2b)
  --set y2023    data/b3 850 hPa, windows W9_2023JFM and W10_2023JAS
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B18_equal_km_regions import (  # noqa: E402
    REGIONS, envelope_rho_profile_ells, km_crop,
)
from experiment_B1_true_flux_baselines import (  # noqa: E402
    _grid_geometry, _interior_mask, phase_randomize,
)
from experiment_B20_p_geography import (  # noqa: E402
    ELLS_FINE, N_FINE, TILE_COLS, TILE_ROWS, tile_slices,
)
from experiment_B2_scale_irreversibility import compute_vorticity  # noqa: E402
from explore_physics_tiles import band_fields, compute_divergence  # noqa: E402

SEED_SUR = 20260811
OUT = _HERE / "results" / "explore_physics"


def jobs_levels():
    out = []
    for f500 in sorted(Path("data/b5w500").glob("era5_wind500_*.nc")):
        tag = f500.stem.replace("era5_wind500_", "")
        region, window = tag.split("__")
        for d in ("data/b1", "data/b2b", "data/b3"):
            f850 = Path(d) / f"era5_wind850_{tag}.nc"
            if f850.exists():
                out.append((region, window, 850, f850))
                out.append((region, window, 500, f500))
                break
    return out


def jobs_y2023():
    return [(r, w, 850, Path(f"data/b3/era5_wind850_{r}__{w}.nc"))
            for r in REGIONS for w in ("W9_2023JFM", "W10_2023JAS")]


def iso_resample(uu, vv, lo, lat_c):
    dlon = float(lo[1] - lo[0])
    n = len(lo)
    step = dlon / np.cos(np.deg2rad(lat_c))
    xo = np.arange(n) * dlon
    xn = np.arange(int(np.floor((n - 1) * dlon / step + 1e-9)) + 1) * step
    i0 = np.clip(np.searchsorted(xo, xn, side="right") - 1, 0, n - 2)
    wg = ((xn - xo[i0]) / dlon).astype(np.float32)
    return (uu[:, :, i0] * (1 - wg) + uu[:, :, i0 + 1] * wg,
            vv[:, :, i0] * (1 - wg) + vv[:, :, i0 + 1] * wg, lo[0] + xn)


def _one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    region, window, level, path, n_sur, iso = job
    ds = xr.open_dataset(path)
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
    v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
    ds.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]
    lat, lon, u, v = km_crop(lat, lon, u, v, region, 1.0)
    rows = tile_slices(u.shape[1], TILE_ROWS)
    cols = tile_slices(u.shape[2], TILE_COLS)
    res = []
    for ri, rs in enumerate(rows):
        for ci, cs in enumerate(cols):
            la, lo = lat[rs], lon[cs]
            uu, vv = u[:, rs, cs], v[:, rs, cs]
            lat_c, lon_c = float(np.mean(la)), float(np.mean(lo))
            if iso:
                uu, vv, lo = iso_resample(uu, vv, lo, lat_c)
            dx_km, dy_km = _grid_geometry(la, lo)
            mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1],
                                  dx_km, dy_km)
            om = compute_vorticity(uu, vv, la, lo)
            rho = np.median(envelope_rho_profile_ells(
                om, dx_km, dy_km, mask, ELLS_FINE), axis=0)
            tcode = zlib.crc32(
                f"xpreg_{region}_{window}_{level}_{ri}{ci}".encode()) % 100000
            seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(n_sur)
            sur = np.empty((n_sur, N_FINE))
            for j, s in enumerate(seeds):
                rng = np.random.default_rng(int(s))
                us, vs = phase_randomize(uu, vv, rng)
                sur[j] = np.median(envelope_rho_profile_ells(
                    compute_vorticity(us, vs, la, lo), dx_km, dy_km, mask,
                    ELLS_FINE), axis=0)
            b_v, _ = band_fields(np.nan_to_num(om), dx_km, dy_km)
            b_d, _ = band_fields(np.nan_to_num(
                compute_divergence(uu, vv, la, lo)), dx_km, dy_km)
            anch = rho - np.median(sur, axis=0)
            res.append({"region": region, "window": window, "level": level,
                        "tile": f"r{ri}c{ci}", "lat_c": lat_c, "lon_c": lon_c,
                        "P_real": rho.tolist(),
                        "P_sur": np.median(sur, axis=0).tolist(),
                        "P_anch": float(np.mean(anch)),
                        "P_c": float(np.mean(anch[1:])),
                        "vort_var": [float(np.mean(np.var(b[:, mask], axis=1)))
                                     for b in b_v],
                        "div_var": [float(np.mean(np.var(b[:, mask], axis=1)))
                                    for b in b_d]})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["levels", "y2023"], default="levels")
    ap.add_argument("--iso", type=int, default=1)
    ap.add_argument("--nsur", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / f"regional_{args.set}{'_iso' if args.iso else ''}.json"
    base = jobs_levels() if args.set == "levels" else jobs_y2023()
    jobs = [(*j, args.nsur, bool(args.iso)) for j in base]
    print(len(jobs), "jobs", flush=True)
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs)):
            rows.extend(r)
            print(f"[{i+1}/{len(jobs)}]", flush=True)
    out_f.write_text(json.dumps(rows), encoding="utf-8")
    print("done", len(rows))


if __name__ == "__main__":
    main()
