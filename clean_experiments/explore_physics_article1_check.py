#!/usr/bin/env python3
"""EXPLORATORY (not frozen): does the grid-anisotropy artifact (T11) touch
Article 1? Article-1 profile (6-band ladder 50..1600 km, adjacent-pair
envelope coupling, phase-surrogate anchoring) on the 12 equal-km regional
boxes, windows 2023-2024 (data/b3), native grid versus isotropic grid
(zonal step resampled to the meridional step in km). 24 surrogates.

Register: docs/EXPLORATION_PHYSICS_2026-09-29.md.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from experiment_B18_equal_km_regions import (REGIONS, PRIMARY_WINDOWS,  # noqa: E402
                                             envelope_rho_profile_ells, km_crop)
from experiment_B1_true_flux_baselines import (_grid_geometry, _interior_mask,  # noqa: E402
                                               phase_randomize)
from experiment_B2_scale_irreversibility import ELLS_KM, compute_vorticity  # noqa: E402
from explore_physics_regional_levels import iso_resample  # noqa: E402

OUT = _HERE / "results" / "explore_physics" / "article1_check.json"
N_SUR = 24
SEED_SUR = 20260811


def profile(uu, vv, la, lo, tag):
    dx, dy = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_KM[-1], dx, dy)
    real = np.median(envelope_rho_profile_ells(
        compute_vorticity(uu, vv, la, lo), dx, dy, mask, ELLS_KM), axis=0)
    seeds = np.random.SeedSequence(
        SEED_SUR + zlib.crc32(tag.encode()) % 100000).generate_state(N_SUR)
    sur = []
    for s in seeds:
        us, vs = phase_randomize(uu, vv, np.random.default_rng(int(s)))
        sur.append(np.median(envelope_rho_profile_ells(
            compute_vorticity(us, vs, la, lo), dx, dy, mask, ELLS_KM), axis=0))
    return real, np.median(np.asarray(sur), axis=0), float(np.mean(dx)), dy


def _one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    region, window = job
    ds = xr.open_dataset(f"data/b3/era5_wind850_{region}__{window}.nc")
    lat = np.asarray(ds["latitude"].values, float)
    lon = np.asarray(ds["longitude"].values, float)
    u = np.asarray(ds["u"].squeeze().values, np.float32)
    v = np.asarray(ds["v"].squeeze().values, np.float32)
    ds.close()
    if lat[0] > lat[-1]:
        lat, u, v = lat[::-1], u[:, ::-1], v[:, ::-1]
    la, lo, u, v = km_crop(lat, lon, u, v, region, 1.0)
    out = {"region": region, "window": window, "lat_c": float(np.mean(la))}
    r, s, dx, dy = profile(u, v, la, lo, f"a1_{region}_{window}")
    out.update(native_real=r.tolist(), native_sur=s.tolist(), dx_native=dx, dy=dy)
    ui, vi, loi = iso_resample(u, v, lo, float(np.mean(la)))
    r, s, dx, _ = profile(ui, vi, la, loi, f"a1_{region}_{window}")
    out.update(iso_real=r.tolist(), iso_sur=s.tolist(), dx_iso=dx)
    return out


if __name__ == "__main__":
    jobs = [(r, w) for r in REGIONS for w in PRIMARY_WINDOWS]
    rows = []
    with mp.get_context("fork").Pool(9) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs)):
            rows.append(r)
            print(f"[{i+1}/{len(jobs)}] {r['region']} {r['window']}", flush=True)
    OUT.write_text(json.dumps(rows), encoding="utf-8")
    print("A1CHECK_DONE", len(rows))
