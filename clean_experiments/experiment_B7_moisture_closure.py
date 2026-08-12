#!/usr/bin/env python3
"""Phase-7: do the validated invariants carry moisture-budget closure
information beyond an honest baseline?

Protocol: docs/PROTOCOL_PHASE7_MOISTURE_CLOSURE.md (frozen 2026-08-12).
Residual r(t) = dW/dt + div(IVT) - (E - P), domain interior mean.
Spec note (logged in protocol Deviations): tp and e retrieved at synoptic
times are 1-hour accumulations; scaled by 6 as a 6-hour proxy. This inflates
|r| uniformly and does not affect the relative feature test.

C7: median held-out incremental R^2 (baseline + invariants vs baseline) > 0
across 24 region-windows AND sign test p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from math import comb
from pathlib import Path

import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        SCALE_EDGES_KM,
        compute_vorticity,
        envelope_rho_profile,
    )
    from clean_experiments.experiment_B6n2_ivt_profiles import load_ivt
    from clean_experiments.experiment_M_cosmo_flow import (
        _build_band_masks,
        _select_mode_indices,
        _xy_coordinates_m,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _build_coefficients,
        _load_vector_fields,
    )
except ImportError:
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        SCALE_EDGES_KM,
        compute_vorticity,
        envelope_rho_profile,
    )
    from experiment_B6n2_ivt_profiles import load_ivt  # type: ignore
    from experiment_M_cosmo_flow import (  # type: ignore
        _build_band_masks,
        _select_mode_indices,
        _xy_coordinates_m,
    )
    from experiment_scale_gravity_einstein_box_era import (  # type: ignore
        _build_coefficients,
        _load_vector_fields,
    )

SEED = 20260811
WINDOW, RIDGE, SHRINK, NMODES, WARMUP = 20, 1e-6, 0.05, 6, 19
RIDGE_REG = 1e-3
OUT = _HERE / "results" / "experiment_B7_moisture_closure"


def fnorm_series(u, v, lat, lon) -> np.ndarray:
    """Domain-level log fine-band ||F|| series (mean of bands 0-1)."""
    x_m, y_m = _xy_coordinates_m(lat, lon)
    dx = float(np.median(np.abs(np.diff(x_m / 1000.0))))
    dy = float(np.median(np.abs(np.diff(y_m / 1000.0))))
    masks, _, _, _, _, wl = _build_band_masks(
        ny=u.shape[1], nx=u.shape[2], dy_km=dy, dx_km=dx, scale_edges_km=SCALE_EDGES_KM
    )
    fields = {"u": u, "v": v}
    selected, _ = _select_mode_indices(
        mode_fields=fields, masks=masks, n_modes_per_var=NMODES, wavelength_km=wl
    )
    coeff = _build_coefficients(mode_fields=fields, selected=selected)
    scaled = []
    for arr in coeff[:3]:
        a = np.asarray(arr, dtype=np.complex128)
        rms = np.sqrt(np.mean(np.abs(a) ** 2, axis=0))
        scaled.append(a / np.where(rms < 1e-12, 1.0, rms)[None, :])
    nt = scaled[0].shape[0]
    fn = np.zeros(nt)
    cnt = 0
    for b in range(2):
        m1, m2 = scaled[b].shape[1], scaled[b + 1].shape[1]
        e1 = np.eye(m1, dtype=np.complex128)
        e2 = np.eye(m2, dtype=np.complex128)
        for t in range(nt):
            a = scaled[b][max(0, t - WINDOW + 1): t + 1]
            bb = scaled[b + 1][max(0, t - WINDOW + 1): t + 1]
            g1 = a.conj().T @ a + RIDGE * e1
            g2 = bb.conj().T @ bb + RIDGE * e2
            mf = np.linalg.solve(g1, a.conj().T @ bb)
            mr = np.linalg.solve(g2, bb.conj().T @ a)
            gf, gb = mf @ mf.conj().T, mr.conj().T @ mr
            comm = gf @ gb - gb @ gf
            f = 0.5 * (1j * comm + (1j * comm).conj().T)
            fn[t] += float(np.linalg.norm(f))
        cnt += 1
    return np.log(fn / cnt + 1e-12)


def _budget_fields(path: Path):
    ds = xr.open_dataset(path)
    names = {n.lower(): n for n in ds.data_vars}
    w_name = names.get("tcwv") or next(n for n in ds.data_vars if "tcwv" in n.lower() or "water_vapour" in n.lower())
    e_name = names.get("e") or next(n for n in ds.data_vars if n.lower() in ("e", "evap") or "evapo" in n.lower())
    p_name = names.get("tp") or next(n for n in ds.data_vars if n.lower() == "tp" or "precip" in n.lower())
    da = ds[w_name].squeeze()
    tdim = next(d for d in da.dims if "time" in d or d == "valid_time")
    def arr(n):
        return np.nan_to_num(np.asarray(ds[n].squeeze().transpose(tdim, ...).values, dtype=float))
    w, e, tp = arr(w_name), arr(e_name), arr(p_name)
    ds.close()
    return w, e, tp


def process_rw(args_tuple) -> dict | None:
    wind, ivt, budget = args_tuple
    os.environ["OMP_NUM_THREADS"] = "1"
    tag = wind.stem.replace("era5_wind850_", "")
    cached = OUT / f"{tag}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    print(f"[{tag}] computing", flush=True)

    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=wind, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], 1600.0, dx_km, dy_km)
    lat_sign = 1.0 if lat[1] > lat[0] else -1.0

    _, _, iu, iv = load_ivt(ivt)
    w, ev, tp = _budget_fields(budget)
    nt = min(u.shape[0], iu.shape[0], w.shape[0])

    # divergence of IVT in kg m-2 s-1
    div = np.gradient(iu[:nt], axis=2) / (dx_km[None, :, None] * 1000.0) \
        + np.gradient(iv[:nt], axis=1) / (dy_km * 1000.0) * lat_sign
    div_s = div[:, mask].mean(axis=1) * 6 * 3600.0            # kg m-2 / 6h
    dwdt = np.gradient(w[:nt, mask].mean(axis=1))             # kg m-2 / 6h (6h step)
    # tp, e: metres of water per hour at synoptic times; x1000 kg m-2, x6 proxy
    p_s = tp[:nt, mask].mean(axis=1) * 1000.0 * 6.0
    e_s = -ev[:nt, mask].mean(axis=1) * 1000.0 * 6.0          # ERA5 e negative up
    r = dwdt + div_s - (e_s - p_s)

    c_series = fnorm_series(u[:nt], v[:nt], lat, lon)
    omega = compute_vorticity(u[:nt], v[:nt], lat, lon)
    rho5 = envelope_rho_profile(omega, dx_km, dy_km, mask)[:, 4]

    t0 = WARMUP + 1
    idx = np.arange(t0, nt - 1)
    feats_base = np.column_stack([
        r[idx - 1], np.abs(r[idx - 1]), p_s[idx], np.abs(div_s[idx]),
        dwdt[idx], idx.astype(float),
    ])
    feats_inv = np.column_stack([
        c_series[idx], rho5[idx], c_series[idx - 1], rho5[idx - 1],
    ])
    y = r[idx]

    half = int(0.6 * len(y))

    def ridge_r2(x):
        xtr, xte = x[:half], x[half:]
        mu, sd = xtr.mean(0), xtr.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        xtr = np.hstack([np.ones((len(xtr), 1)), xtr])
        xte = np.hstack([np.ones((len(xte), 1)), xte])
        a = xtr.T @ xtr + RIDGE_REG * np.eye(xtr.shape[1])
        coef = np.linalg.solve(a, xtr.T @ y[:half])
        pred = xte @ coef
        ss_res = float(np.sum((y[half:] - pred) ** 2))
        ss_tot = float(np.sum((y[half:] - y[half:].mean()) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    r2_base = ridge_r2(feats_base)
    r2_full = ridge_r2(np.hstack([feats_base, feats_inv]))
    res = {
        "tag": tag, "region": tag.split("__")[0],
        "r2_base": r2_base, "r2_full": r2_full,
        "increment": r2_full - r2_base,
        "median_abs_residual_kg_m2_6h": float(np.median(np.abs(y))),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{tag}] inc={res['increment']:+.4f}", flush=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-dir", type=Path, default=Path("data/b3"))
    parser.add_argument("--ivt-dir", type=Path, default=Path("data/b6ivt"))
    parser.add_argument("--budget-dir", type=Path, default=Path("data/b7budget"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    jobs = []
    for wf in sorted(args.wind_dir.glob("era5_wind850_*.nc")):
        tag = wf.stem.replace("era5_wind850_", "")
        f_ivt = args.ivt_dir / f"era5_ivt_{tag}.nc"
        f_bud = args.budget_dir / f"era5_budget_{tag}.nc"
        if f_ivt.exists() and f_bud.exists():
            jobs.append((wf, f_ivt, f_bud))
    print(f"{len(jobs)} region-windows with wind+ivt+budget", flush=True)
    if not jobs:
        raise SystemExit("no complete region-windows yet")

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        results = [r for r in pool.map(process_rw, jobs) if r]

    incs = np.asarray([r["increment"] for r in results])
    n_pos = int(np.sum(incs > 0))
    n = len(incs)
    p_sign = float(sum(comb(n, k) for k in range(n_pos, n + 1)) / 2 ** n)
    c7 = bool(np.median(incs) > 0 and p_sign < 0.05 and n >= 24)
    summary = {
        "n_region_windows": n,
        "median_increment": float(np.median(incs)),
        "positive_increments": n_pos,
        "sign_test_p": p_sign,
        "median_r2_base": float(np.median([r["r2_base"] for r in results])),
        "median_r2_full": float(np.median([r["r2_full"] for r in results])),
        "C7_pass": c7,
        "PHASE7_VERDICT": "POSITIVE" if c7 else "NEGATIVE",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
