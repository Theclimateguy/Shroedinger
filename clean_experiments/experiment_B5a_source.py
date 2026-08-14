#!/usr/bin/env python3
"""Phase-5a: source localization — tiled curvature vs precipitation.

Protocol: docs/PROTOCOL_PHASE5_GAUGE_STRUCTURE.md (frozen 2026-08-12).
3x4 tiles per domain; per tile: curvature machinery with scale edges
[50,100,200,400] km, 4 modes/var; Fnorm_tile(t) = mean over band pairs.
C5a: Spearman over (tile,t) of log Fnorm vs log(precip+1e-6) positive with
block-permutation p<0.05 (999, time blocks of 20), in >=22 of 32 rws.
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
from scipy.stats import rankdata

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B4_curvature_invariant import (
        curvature_profiles,  # noqa: F401  (import check)
    )
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
WINDOW, RIDGE, SHRINK, NMODES, WARMUP = 20, 1e-6, 0.05, 4, 19
TILE_EDGES_KM = [50.0, 100.0, 200.0, 400.0]
NTY, NTX = 3, 4
OUT_DIR = _HERE / "results" / "experiment_B5_gauge_structure"


def tile_slices(ny: int, nx: int) -> list[tuple[slice, slice]]:
    ys = np.linspace(0, ny, NTY + 1, dtype=int)
    xs = np.linspace(0, nx, NTX + 1, dtype=int)
    return [(slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1]))
            for i in range(NTY) for j in range(NTX)]


def tile_fnorm_series(u, v, lat, lon) -> np.ndarray:
    """Fnorm time series per tile, shape (n_tiles, nt)."""
    nt = u.shape[0]
    out = np.zeros((NTY * NTX, nt))
    for k, (sy, sx) in enumerate(tile_slices(u.shape[1], u.shape[2])):
        ut, vt = u[:, sy, sx], v[:, sy, sx]
        lat_t, lon_t = lat[sy], lon[sx]
        x_m, y_m = _xy_coordinates_m(lat_t, lon_t)
        dx = float(np.median(np.abs(np.diff(x_m / 1000.0))))
        dy = float(np.median(np.abs(np.diff(y_m / 1000.0))))
        masks, _, _, _, _, wl = _build_band_masks(
            ny=ut.shape[1], nx=ut.shape[2], dy_km=dy, dx_km=dx,
            scale_edges_km=TILE_EDGES_KM,
        )
        fields = {"u": ut, "v": vt}
        selected, _ = _select_mode_indices(
            mode_fields=fields, masks=masks, n_modes_per_var=NMODES, wavelength_km=wl
        )
        coeff = _build_coefficients(mode_fields=fields, selected=selected)
        scaled = []
        for arr in coeff:
            a = np.asarray(arr, dtype=np.complex128)
            rms = np.sqrt(np.mean(np.abs(a) ** 2, axis=0))
            scaled.append(a / np.where(rms < 1e-12, 1.0, rms)[None, :])
        fnorm = np.zeros(nt)
        cnt = 0
        for b in range(len(scaled) - 1):
            m1, m2 = scaled[b].shape[1], scaled[b + 1].shape[1]
            if m1 == 0 or m2 == 0:
                continue
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
                fnorm[t] += float(np.linalg.norm(f))
            cnt += 1
        out[k] = fnorm / max(cnt, 1)
    return out


def tile_precip_series(precip_path: Path, ny: int, nx: int) -> np.ndarray:
    ds = xr.open_dataset(precip_path)
    var = next(v for v in ds.data_vars)
    da = ds[var].squeeze()
    tdim = next(d for d in da.dims if "time" in d or d == "valid_time")
    arr = np.nan_to_num(np.asarray(da.transpose(tdim, ...).values, dtype=float))
    ds.close()
    out = np.zeros((NTY * NTX, arr.shape[0]))
    for k, (sy, sx) in enumerate(tile_slices(ny, nx)):
        out[k] = arr[:, sy, sx].mean(axis=(1, 2))
    return out


def block_perm_p(x: np.ndarray, y: np.ndarray, rng, n_perm=999, block=20) -> tuple[float, float]:
    """Spearman over flattened (tile,t) with time-block permutation of y."""
    n_tiles, nt = x.shape
    rx = rankdata(x.ravel())
    ry = rankdata(y.ravel())
    obs = float(np.corrcoef(rx, ry)[0, 1])
    nb = int(np.ceil(nt / block))
    count = 0
    for _ in range(n_perm):
        order = rng.permutation(nb)
        idx = np.concatenate([np.arange(b * block, min((b + 1) * block, nt))
                              for b in order])
        yp = y[:, idx]
        ryp = rankdata(yp.ravel())
        r = float(np.corrcoef(rx[: len(ryp)], ryp)[0, 1])
        if r >= obs:
            count += 1
    return obs, (1 + count) / (n_perm + 1)


def process_rw(args_tuple) -> dict:
    wind_path, precip_path = args_tuple
    os.environ["OMP_NUM_THREADS"] = "1"
    tag = wind_path.stem.replace("era5_wind850_", "")
    cached = OUT_DIR / f"b5a_{tag}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=wind_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    fn = tile_fnorm_series(u, v, lat, lon)[:, WARMUP:]
    pr = tile_precip_series(precip_path, u.shape[1], u.shape[2])
    nt = min(fn.shape[1], pr.shape[1] - WARMUP)
    fn = fn[:, :nt]
    pr = pr[:, WARMUP: WARMUP + nt]
    rng = np.random.default_rng(SEED)
    rho, p = block_perm_p(np.log(fn + 1e-12), np.log(pr + 1e-6), rng)

    # centroids (descriptive)
    def centroid(m):
        w = m - m.min(axis=0, keepdims=True) + 1e-12
        ys, xs = np.divmod(np.arange(NTY * NTX), NTX)
        return (np.average(np.repeat(ys[:, None], m.shape[1], 1), 0, w),
                np.average(np.repeat(xs[:, None], m.shape[1], 1), 0, w))

    cfy, cfx = centroid(fn)
    cpy, cpx = centroid(pr)
    lags = {}
    for lag in range(-4, 5):
        if lag >= 0:
            a = np.concatenate([cfy[lag:], cfx[lag:]])
            b = np.concatenate([cpy[: len(cpy) - lag], cpx[: len(cpx) - lag]])
        else:
            a = np.concatenate([cfy[:lag], cfx[:lag]])
            b = np.concatenate([cpy[-lag:], cpx[-lag:]])
        lags[str(lag)] = float(np.corrcoef(a, b)[0, 1])

    res = {"tag": tag, "region": tag.split("__")[0],
           "spearman_tile_t": rho, "block_perm_p": p,
           "pass_rw": bool(rho > 0 and p < 0.05),
           "centroid_lag_corr": lags}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{tag}] rho={rho:.3f} p={p:.3f}", flush=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-dir", type=Path, default=Path("data/b2b"))
    parser.add_argument("--precip-dir", type=Path, default=Path("data/b5precip"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    jobs = []
    for wf in sorted(args.wind_dir.glob("era5_wind850_*.nc")):
        tag = wf.stem.replace("era5_wind850_", "")
        pf = args.precip_dir / f"era5_precip_{tag}.nc"
        if pf.exists():
            jobs.append((wf, pf))
    print(f"{len(jobs)} region-windows with precip", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(process_rw, jobs)

    n_pass = sum(1 for r in results if r["pass_rw"])
    c5a = bool(n_pass >= 22 and len(results) >= 32)
    lag_med = {str(l): float(np.median([r["centroid_lag_corr"][str(l)] for r in results]))
               for l in range(-4, 5)}
    out = {"n_region_windows": len(results), "rw_pass_count": n_pass,
           "C5a_pass": c5a, "median_spearman": float(np.median(
               [r["spearman_tile_t"] for r in results])),
           "median_centroid_lag_corr": lag_med}
    (OUT_DIR / "b5a_source.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
