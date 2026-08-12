#!/usr/bin/env python3
"""Phase-5c: integrability of the scale connection (path consistency vs null).

Protocol: docs/PROTOCOL_PHASE5_GAUGE_STRUCTURE.md (frozen 2026-08-12).
Path discrepancy between (scale-step then time-step) and (time-step then
scale-step); real dynamics must be closer to an integrable connection than
shared-phase surrogates: real median d < surrogate p05 in >=3/5 band pairs,
in >=32/48 region-windows.
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
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import phase_randomize
    from clean_experiments.experiment_B2_scale_irreversibility import SCALE_EDGES_KM
    from clean_experiments.experiment_B4_curvature_invariant import (
        confirmation_files,
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
    from experiment_B1_true_flux_baselines import phase_randomize  # type: ignore
    from experiment_B2_scale_irreversibility import SCALE_EDGES_KM  # type: ignore
    from experiment_B4_curvature_invariant import confirmation_files  # type: ignore
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
WINDOW, RIDGE, NMODES, WARMUP = 20, 1e-6, 6, 19
N_SURR = 99


def _coeffs(u, v, lat, lon) -> list[np.ndarray]:
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
    out = []
    for arr in coeff:
        a = np.asarray(arr, dtype=np.complex128)
        rms = np.sqrt(np.mean(np.abs(a) ** 2, axis=0))
        out.append(a / np.where(rms < 1e-12, 1.0, rms)[None, :])
    return out


def path_discrepancy(scaled: list[np.ndarray]) -> np.ndarray:
    """Median over t of d_{b,t} for the 5 band pairs."""
    nb = len(scaled)
    nt = scaled[0].shape[0]

    def t_map(a: np.ndarray, t: int) -> np.ndarray:
        i0 = max(0, t - WINDOW + 1)
        past, nxt = a[i0:t], a[i0 + 1: t + 1]
        m = a.shape[1]
        g = past.conj().T @ past + RIDGE * np.eye(m)
        return np.linalg.solve(g, past.conj().T @ nxt)

    def m_fwd(a: np.ndarray, b: np.ndarray, t: int) -> np.ndarray:
        i0 = max(0, t - WINDOW + 1)
        aa, bb = a[i0: t + 1], b[i0: t + 1]
        m = a.shape[1]
        g = aa.conj().T @ aa + RIDGE * np.eye(m)
        return np.linalg.solve(g, aa.conj().T @ bb)

    med = np.zeros(nb - 1)
    for bi in range(nb - 1):
        ds = []
        for t in range(WARMUP, nt - 1):
            mf_t = m_fwd(scaled[bi], scaled[bi + 1], t)
            mf_t1 = m_fwd(scaled[bi], scaled[bi + 1], t + 1)
            tb = t_map(scaled[bi], t)
            tb1 = t_map(scaled[bi + 1], t)
            a_map = mf_t @ tb1
            b_map = tb @ mf_t1
            na, nbn = np.linalg.norm(a_map), np.linalg.norm(b_map)
            if na + nbn > 1e-12:
                ds.append(float(np.linalg.norm(a_map - b_map) / (na + nbn)))
        med[bi] = float(np.median(ds))
    return med


_G: dict[str, object] = {}


def _init(u, v, lat, lon) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(u=u, v=v, lat=lat, lon=lon)


def _worker(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    us, vs = phase_randomize(_G["u"], _G["v"], rng)  # type: ignore[arg-type]
    return [float(x) for x in path_discrepancy(_coeffs(us, vs, _G["lat"], _G["lon"]))]  # type: ignore[arg-type]


def run_rw(nc_path: Path, out_dir: Path, workers: int) -> dict:
    tag = nc_path.stem.replace("era5_wind850_", "")
    cached = out_dir / f"b5c_{tag}.json"
    if cached.exists():
        print(f"[{tag}] cached", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))
    print(f"[{tag}] computing", flush=True)
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    d_real = path_discrepancy(_coeffs(u, v, lat, lon))
    seeds = [int(s) for s in np.random.SeedSequence(
        SEED + zlib.crc32(tag.encode()) % 100000).generate_state(N_SURR)]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, _init, (u, v, lat, lon)) as pool:
        sur = np.asarray(pool.map(_worker, seeds))
    p05 = np.percentile(sur, 5, axis=0)
    pairs_below = int(np.sum(d_real < p05))
    res = {
        "tag": tag, "region": tag.split("__")[0],
        "d_real": [float(x) for x in d_real],
        "d_surr_p05": [float(x) for x in p05],
        "d_surr_median": [float(x) for x in np.median(sur, axis=0)],
        "pairs_below_p05": pairs_below,
        "pass_rw": bool(pairs_below >= 3),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{tag}] pairs_below={pairs_below}/5", flush=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_HERE.parent)
    parser.add_argument("--out-dir", type=Path,
                        default=_HERE / "results" / "experiment_B5_gauge_structure")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    files = confirmation_files(args.repo_root)
    print(f"{len(files)} region-windows", flush=True)
    results = [run_rw(f, args.out_dir, args.workers) for f in files]

    n_pass = sum(1 for r in results if r["pass_rw"])
    c5c = bool(n_pass >= 32)
    summary = {
        "n_region_windows": len(results),
        "rw_pass_count": n_pass,
        "C5c_pass": c5c,
        "median_d_real": [float(x) for x in np.median(
            np.asarray([r["d_real"] for r in results]), axis=0)],
        "median_d_surr": [float(x) for x in np.median(
            np.asarray([r["d_surr_median"] for r in results]), axis=0)],
    }
    (args.out_dir / "b5c_integrability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
