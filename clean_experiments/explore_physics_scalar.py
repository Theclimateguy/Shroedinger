#!/usr/bin/env python3
"""EXPLORATORY (not frozen) test T14: inter-level coupling in the 850-hPa
temperature field and between temperature and vorticity.

Register (expectations written before computation):
docs/EXPLORATION_PHYSICS_2026-09-29.md, section 9.

Corrected estimator only: isotropic grid (zonal step resampled to the
meridional one in km), bands 50-100, 100-200, 200-400 km (the sub-50 km
band is not used). Surrogates share one random phase field between u, v
and T at each step, so linear cross-spectra are kept.

Per tile: P_vort and P_T (pairs 50-100|100-200 and 100-200|200-400),
X_b = same-band coupling of the temperature and vorticity envelopes,
each also on two interleaved halves of alternating 5-day blocks.
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
sys.path.insert(0, str(_HERE))

from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # noqa: E402
from experiment_B20_armB_global_map import tile_grid  # noqa: E402
from experiment_B20_p_geography import ELLS_FINE  # noqa: E402
from experiment_B2_scale_irreversibility import compute_vorticity  # noqa: E402
from explore_physics_tiles import (  # noqa: E402
    BLOCK_DAYS, ERA5_MONTHS, GLOB, _corr_rows, _rank_rows, band_fields,
    log_envelopes,
)

SEED_SUR = 20260811
OUT = _HERE / "results" / "explore_physics"
TDIR = Path("data/xp_physics")
_G: dict[str, object] = {}
BANDS = (1, 2, 3)


def couplings(om, tt, dx, dy, mask):
    """Per-step couplings: vort pairs (2), T pairs (2), cross same-band (3)."""
    ev = [e[:, mask] for e in log_envelopes(band_fields(np.nan_to_num(om), dx, dy)[0], dx, dy)]
    et = [e[:, mask] for e in log_envelopes(band_fields(np.nan_to_num(tt), dx, dy)[0], dx, dy)]
    rv = {b: _rank_rows(ev[b]) for b in BANDS}
    rt = {b: _rank_rows(et[b]) for b in BANDS}
    cols = [_corr_rows(rv[1], rv[2]), _corr_rows(rv[2], rv[3]),
            _corr_rows(rt[1], rt[2]), _corr_rows(rt[2], rt[3])]
    cols += [_corr_rows(rv[b], rt[b]) for b in BANDS]
    return np.stack(cols, axis=1)                       # (nt, 7)


def summarise(c, block):
    return [np.median(c, axis=0), np.median(c[block == 0], axis=0),
            np.median(c[block == 1], axis=0)]


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
    uu, vv, tt = cut(u), cut(v), cut(T)
    lo = t["lon0"] + xn
    dx, dy = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1], dx, dy)
    real = summarise(couplings(compute_vorticity(uu, vv, la, lo), tt, dx, dy, mask), block)
    tcode = zlib.crc32(f"xpT_{t['tid']}_{_G['season']}".encode()) % 100000
    seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(_G["n_sur"])
    nt, ny, nx = uu.shape
    sur = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        us, vs, ts = np.empty_like(uu), np.empty_like(vv), np.empty_like(tt)
        my, mx = (2 * ny, 2 * nx) if _G["mirror"] else (ny, nx)

        def ext(a):
            if not _G["mirror"]:
                return a
            a2 = np.concatenate([a, a[::-1, :]], axis=0)
            return np.concatenate([a2, a2[:, ::-1]], axis=1)
        for k in range(nt):
            fac = np.exp(1j * np.angle(np.fft.rfft2(rng.standard_normal((my, mx)))))
            fac[0, 0] = 1.0
            us[k] = np.fft.irfft2(np.fft.rfft2(ext(uu[k])) * fac, s=(my, mx))[:ny, :nx]
            vs[k] = np.fft.irfft2(np.fft.rfft2(ext(vv[k])) * fac, s=(my, mx))[:ny, :nx]
            ts[k] = np.fft.irfft2(np.fft.rfft2(ext(tt[k])) * fac, s=(my, mx))[:ny, :nx]
        sur.append(summarise(couplings(compute_vorticity(us, vs, la, lo), ts, dx, dy, mask), block))
    sur = np.median(np.asarray(sur), axis=0)             # (3, 7)
    bt = band_fields(np.nan_to_num(tt - tt.mean(axis=(1, 2), keepdims=True)), dx, dy)[0]
    return {"tid": t["tid"],
            "real": [r.tolist() for r in real], "sur": sur.tolist(),
            "T_band_sd": [float(np.sqrt(np.mean(np.var(b[:, mask], axis=1)))) for b in bt]}


def main():
    import xarray as xr
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="JFM")
    ap.add_argument("--nsur", type=int, default=24)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mirror", type=int, default=0,
                    help="surrogates on the mirror extension (no edge leakage)")
    a = ap.parse_args()
    out = OUT / f"tiles_scalarT_iso{'_mirror' if a.mirror else ''}_{a.season}.json"
    if out.exists() and not a.limit:
        print("cached")
        return
    pw = [xr.open_dataset(GLOB / f"era5_wind850_global_{ym}.nc") for ym in ERA5_MONTHS[a.season]]
    pt = [xr.open_dataset(TDIR / f"era5_t850_global_{ym}.nc") for ym in ERA5_MONTHS[a.season]]
    dw, dt = xr.concat(pw, dim="valid_time"), xr.concat(pt, dim="valid_time")
    assert np.array_equal(dw["valid_time"].values, dt["valid_time"].values)
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
              T=np.ascontiguousarray(T), lat=lat, lon=lon, block=block,
              n_sur=a.nsur, season=a.season, mirror=bool(a.mirror))
    print(a.season, "cube", u.shape, flush=True)
    tiles = tile_grid()
    if a.limit:
        tiles = tiles[:: len(tiles) // a.limit][: a.limit]
    res = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_tile_worker, tiles, chunksize=4)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"[{a.season} {i+1}/{len(tiles)}]", flush=True)
    if a.limit:
        print(json.dumps(res[0])[:1500])
        return
    out.write_text(json.dumps({"season": a.season, "cols": [
        "vort_1_2", "vort_2_3", "T_1_2", "T_2_3", "X_1", "X_2", "X_3"],
        "rows": ["all", "block0", "block1"], "tiles": res}), encoding="utf-8")
    print(f"SCALAR_{a.season}_DONE", len(res), flush=True)


if __name__ == "__main__":
    main()
