#!/usr/bin/env python3
"""EXPLORATORY (not frozen) test T15: boundary template versus universal
moving part in the inter-level coupling. No surrogates.

Register (expectations written before computation):
docs/EXPLORATION_PHYSICS_2026-09-29.md, section 11.

Log-envelopes E_i(x, t) of bands 50-100, 100-200, 200-400 km (vorticity and
temperature, isotropic grid). Time-mean spatial covariance splits exactly:
C_total = C_template + C_moving, where the template is the time-mean
envelope map. The template terms are cross-validated on two interleaved
sets of 5-day blocks.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # noqa: E402
from experiment_B20_armB_global_map import tile_grid  # noqa: E402
from experiment_B20_p_geography import ELLS_FINE  # noqa: E402
from experiment_B2_scale_irreversibility import compute_vorticity  # noqa: E402
from explore_physics_tiles import (  # noqa: E402
    BLOCK_DAYS, ERA5_MONTHS, GLOB, band_fields, log_envelopes,
)

OUT = _HERE / "results" / "explore_physics"
TDIR = Path("data/xp_physics")
_G: dict[str, object] = {}
BANDS = (1, 2, 3)


def split(envs, block):
    """envs: list of (nt, npix) for the 3 bands -> dict of 3x3 matrices."""
    e = [x - x.mean(axis=1, keepdims=True) for x in envs]          # remove tile mean per step
    n = len(e)
    tot = np.array([[np.mean(np.mean(e[i] * e[j], axis=1)) for j in range(n)] for i in range(n)])
    ma = [x[block == 0].mean(axis=0) for x in e]
    mb = [x[block == 1].mean(axis=0) for x in e]
    tpl = np.array([[0.5 * (np.mean(ma[i] * mb[j]) + np.mean(mb[i] * ma[j]))
                     for j in range(n)] for i in range(n)])
    return tot, tpl


def _tile_worker(t):
    os.environ["OMP_NUM_THREADS"] = "1"
    u, v, T = _G["u"], _G["v"], _G["T"]
    lat, lon, block = _G["lat"], _G["lon"], _G["block"]
    dlon = float(lon[1] - lon[0])
    iy = np.where((lat >= t["lat0"] - 1e-9) & (lat <= t["lat1"] + 1e-9))[0]
    i0 = int(np.searchsorted(lon, t["lon0"] - 1e-9))
    n_cols = max(4, int(round((t["lon1"] - t["lon0"]) / dlon)))
    ix = np.arange(i0, i0 + n_cols) % len(lon)
    la = lat[iy]
    step = dlon / np.cos(np.deg2rad(t["lat_c"]))
    xo = np.arange(n_cols) * dlon
    xn = np.arange(int(np.floor((n_cols - 1) * dlon / step + 1e-9)) + 1) * step
    j0 = np.clip(np.searchsorted(xo, xn, side="right") - 1, 0, n_cols - 2)
    wg = ((xn - xo[j0]) / dlon).astype(np.float32)

    def cut(a):
        b = a[:, iy][:, :, ix].astype(np.float32)
        return b[:, :, j0] * (1 - wg) + b[:, :, j0 + 1] * wg
    uu, vv = cut(u), cut(v)
    tt = cut(T) if T is not None else None
    lo = t["lon0"] + xn
    dx, dy = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1], dx, dy)
    om = np.nan_to_num(compute_vorticity(uu, vv, la, lo))
    ev = [e[:, mask].astype(np.float64) for e in log_envelopes(band_fields(om, dx, dy)[0], dx, dy)]
    ev = [ev[b] for b in BANDS]
    sets = [("vort", ev)]
    if tt is not None:
        et = [e[:, mask].astype(np.float64) for e in log_envelopes(band_fields(np.nan_to_num(tt), dx, dy)[0], dx, dy)]
        et = [et[b] for b in BANDS]
        sets += [("T", et), ("joint", ev + et)]
    out = {"tid": t["tid"]}
    for nm, envs in sets:
        tot, tpl = split(envs, block)
        out[f"{nm}_tot"] = tot.tolist()
        out[f"{nm}_tpl"] = tpl.tolist()
    return out


def main():
    import xarray as xr
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="JFM")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--model-level", type=int, default=0,
                    help="use the free-running IFS-HR run at this level (hPa)")
    a = ap.parse_args()
    if a.model_level:
        from explore_physics_tiles import load_cube
        u, v, lat, lon, _, block = load_cube("model", a.season, a.model_level)
        _G.update(u=u, v=v, T=None, lat=lat, lon=lon, block=block)
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(_tile_worker, tile_grid(), chunksize=8)
        (OUT / f"tiles_template_model{a.model_level}_{a.season}.json").write_text(
            json.dumps({"season": a.season, "tiles": res}), encoding="utf-8")
        print(f"TEMPLATE_MODEL{a.model_level}_{a.season}_DONE", len(res), flush=True)
        return
    out = OUT / f"tiles_template_{a.season}.json"
    pw = [xr.open_dataset(GLOB / f"era5_wind850_global_{ym}.nc") for ym in ERA5_MONTHS[a.season]]
    pt = [xr.open_dataset(TDIR / f"era5_t850_global_{ym}.nc") for ym in ERA5_MONTHS[a.season]]
    dw, dt = xr.concat(pw, dim="valid_time"), xr.concat(pt, dim="valid_time")
    lat = np.asarray(dw["latitude"].values, float)
    lon = np.asarray(dw["longitude"].values, float)
    u = np.asarray(dw["u"].squeeze().values, np.float32)
    v = np.asarray(dw["v"].squeeze().values, np.float32)
    T = np.asarray(dt["t"].squeeze().values, np.float32)
    tm = dw["valid_time"].values.astype("datetime64[D]")
    block = (((tm - tm[0]).astype(int) // BLOCK_DAYS) % 2).astype(int)
    if lat[0] > lat[-1]:
        lat, u, v, T = lat[::-1], u[:, ::-1], v[:, ::-1], T[:, ::-1]
    _G.update(u=np.ascontiguousarray(u), v=np.ascontiguousarray(v),
              T=np.ascontiguousarray(T), lat=lat, lon=lon, block=block)
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_tile_worker, tile_grid(), chunksize=8)
    out.write_text(json.dumps({"season": a.season, "bands_km": ["50-100", "100-200", "200-400"],
                               "tiles": res}), encoding="utf-8")
    print(f"TEMPLATE_{a.season}_DONE", len(res), flush=True)


if __name__ == "__main__":
    main()
