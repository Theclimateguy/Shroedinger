#!/usr/bin/env python3
"""Phase-1 B1: does Lambda_b track the true cross-scale flux beyond cheap baselines?

Preregistered protocol: docs/PROTOCOL_PHASE1_LAMBDA_VS_FLUX.md (frozen 2026-08-11).

Target Pi_ell(t): physical-space Gaussian coarse-graining flux
    Pi_ell = -tau_ij d(bar u_i)/dx_j,  tau_ij = bar(u_i u_j) - bar(u_i) bar(u_j)
averaged over the domain interior (Germano/Eyink/Aluie a-priori method).

Predictors per (region-window, band):
    Lambda_b  - unchanged A15 pipeline, fixed global sign +1
    B1        - Smagorinsky proxy (0.17*ell)^2 * <|S_bar|^3>
    B2        - A15 spectral proxy <|k| E(k)>_b
    B3        - filtered enstrophy <omega_bar^2>
    B4        - resolved gradient <|grad u_bar|^2 + |grad v_bar|^2>

Statistics: raw Pearson/Spearman (no binning), chronological half-split OOS R^2,
moving-block bootstrap CI, and a spectrum-preserving phase-randomized surrogate
null for Lambda (full pipeline rerun per surrogate).

Specification choices logged in the protocol Deviations section:
    - Gaussian filter std sigma = ell / sqrt(12) (Pope's filter-width convention).
    - Interior exclusion ring capped at 25% of the domain extent per side
      (the frozen 2*ell ring empties the interior for ell=566 km).
    - First window-1 = 19 steps dropped from all statistics (rolling-window
      warm-up; matches the A15 t0 convention, applied to every predictor).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        _build_band_masks,
        _select_mode_indices,
        _xy_coordinates_m,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _build_coefficients,
        _compute_pi_proxy,
        _compute_rho_and_lambda,
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
        _compute_pi_proxy,
        _compute_rho_and_lambda,
        _load_vector_fields,
    )

SCALE_EDGES_KM = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]
INERTIAL_BANDS = [1, 2, 3]  # centers ~141, ~283, ~566 km
N_MODES_PER_VAR = 6
WINDOW = 20
RIDGE = 1e-6
COV_SHRINKAGE = 0.05
WARMUP = WINDOW - 1
SMAG_CS = 0.17
INTERIOR_CAP_FRAC = 0.25
BLOCK_LEN = 20
N_BOOT = 999
PREDICTORS = ["lambda", "B1_smag", "B2_a15proxy", "B3_enstrophy", "B4_gradient"]

REGION_WINDOWS = [
    f"era5_wind850_{r}__{w}.nc"
    for r in ("R1_WPWP", "R2_NATL")
    for w in ("W1_2017JFM", "W2_2017JAS", "W3_2018JFM", "W4_2019JAS")
]


def band_centers_km(edges: list[float]) -> list[float]:
    return [float(np.sqrt(lo * hi)) for lo, hi in zip(edges[:-1], edges[1:])]


# ---------------------------------------------------------------------------
# Metric-aware Gaussian filtering and the coarse-graining flux
# ---------------------------------------------------------------------------

def _grid_geometry(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, float]:
    """Return per-row zonal spacing dx_km[ny] and meridional spacing dy_km."""
    earth_r_km = 6371.0
    dlat = float(np.median(np.abs(np.diff(lat))))
    dlon = float(np.median(np.abs(np.diff(lon))))
    dy_km = earth_r_km * np.deg2rad(dlat)
    dx_km = earth_r_km * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon)
    return np.asarray(dx_km, dtype=float), float(dy_km)


def _gaussian_bar(field: np.ndarray, sigma_km: float, dx_km: np.ndarray, dy_km: float) -> np.ndarray:
    """Separable Gaussian filter with per-row zonal pixel sigma (2D field)."""
    out = gaussian_filter1d(field, sigma=sigma_km / dy_km, axis=0, mode="nearest")
    for iy in range(out.shape[0]):
        out[iy] = gaussian_filter1d(out[iy], sigma=sigma_km / dx_km[iy], mode="nearest")
    return out


def _gradients(field: np.ndarray, dx_km: np.ndarray, dy_km: float) -> tuple[np.ndarray, np.ndarray]:
    """d/dx (zonal), d/dy (meridional, northward-positive) in 1/km units."""
    ddy = np.gradient(field, axis=0) / dy_km
    ddx = np.gradient(field, axis=1) / dx_km[:, None]
    return ddx, ddy


def _interior_mask(ny: int, nx: int, ell_km: float, dx_km: np.ndarray, dy_km: float) -> np.ndarray:
    ring_km_y = min(2.0 * ell_km, INTERIOR_CAP_FRAC * ny * dy_km)
    ring_km_x = min(2.0 * ell_km, INTERIOR_CAP_FRAC * nx * float(np.mean(dx_km)))
    ry = int(np.ceil(ring_km_y / dy_km))
    rx = int(np.ceil(ring_km_x / float(np.mean(dx_km))))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[ry : ny - ry, rx : nx - rx] = True
    if not mask.any():
        raise ValueError(f"Interior mask empty for ell={ell_km} km")
    return mask


def flux_and_gridded_baselines(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    ells_km: list[float],
) -> dict[str, np.ndarray]:
    """Pi_ell and B1/B3/B4 series, shape (nt, n_ells). Units: km-based."""
    nt, ny, nx = u.shape
    dx_km, dy_km = _grid_geometry(lat, lon)
    # Fields in km/s so fluxes come out in km^2/s^3; correlations are unit-free.
    uk = u / 1000.0
    vk = v / 1000.0

    n_ells = len(ells_km)
    pi = np.zeros((nt, n_ells))
    b1 = np.zeros((nt, n_ells))
    b3 = np.zeros((nt, n_ells))
    b4 = np.zeros((nt, n_ells))
    masks = [_interior_mask(ny, nx, ell, dx_km, dy_km) for ell in ells_km]

    # Sign convention for d/dy: lat rows may run north->south; gradient along
    # axis 0 divided by positive dy_km needs the actual row direction.
    lat_step_sign = 1.0 if lat[1] > lat[0] else -1.0

    for t in range(nt):
        ut = np.nan_to_num(uk[t])
        vt = np.nan_to_num(vk[t])
        uu, uv, vv = ut * ut, ut * vt, vt * vt
        for j, ell in enumerate(ells_km):
            sigma_km = ell / np.sqrt(12.0)
            ub = _gaussian_bar(ut, sigma_km, dx_km, dy_km)
            vb = _gaussian_bar(vt, sigma_km, dx_km, dy_km)
            tau_xx = _gaussian_bar(uu, sigma_km, dx_km, dy_km) - ub * ub
            tau_xy = _gaussian_bar(uv, sigma_km, dx_km, dy_km) - ub * vb
            tau_yy = _gaussian_bar(vv, sigma_km, dx_km, dy_km) - vb * vb

            dudx, dudy = _gradients(ub, dx_km, dy_km)
            dvdx, dvdy = _gradients(vb, dx_km, dy_km)
            dudy *= lat_step_sign
            dvdy *= lat_step_sign

            flux_map = -(tau_xx * dudx + tau_xy * (dudy + dvdx) + tau_yy * dvdy)

            s_xx = dudx
            s_yy = dvdy
            s_xy = 0.5 * (dudy + dvdx)
            s_mag = np.sqrt(2.0 * (s_xx**2 + 2.0 * s_xy**2 + s_yy**2))
            smag_map = (SMAG_CS * ell) ** 2 * s_mag**3

            omega = dvdx - dudy
            grad_map = dudx**2 + dudy**2 + dvdx**2 + dvdy**2

            m = masks[j]
            pi[t, j] = float(np.mean(flux_map[m]))
            b1[t, j] = float(np.mean(smag_map[m]))
            b3[t, j] = float(np.mean(omega[m] ** 2))
            b4[t, j] = float(np.mean(grad_map[m]))

    return {"pi": pi, "B1_smag": b1, "B3_enstrophy": b3, "B4_gradient": b4}


# ---------------------------------------------------------------------------
# Lambda pipeline (A15, unchanged, fixed sign +1) and the A15 proxy baseline
# ---------------------------------------------------------------------------

def lambda_and_proxy(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lambda_mu, pi_proxy), each (nt, n_bands) over SCALE_EDGES_KM bands."""
    x_m, y_m = _xy_coordinates_m(lat, lon)
    dx_km = float(np.median(np.abs(np.diff(x_m / 1000.0))))
    dy_km = float(np.median(np.abs(np.diff(y_m / 1000.0))))
    masks, ky, kx, _, _, wavelength_km = _build_band_masks(
        ny=u.shape[1], nx=u.shape[2], dy_km=dy_km, dx_km=dx_km, scale_edges_km=SCALE_EDGES_KM
    )
    mode_fields = {"u": u, "v": v}
    selected, _ = _select_mode_indices(
        mode_fields=mode_fields,
        masks=masks,
        n_modes_per_var=N_MODES_PER_VAR,
        wavelength_km=wavelength_km,
    )
    coeff = _build_coefficients(mode_fields=mode_fields, selected=selected)
    lambda_mu, _, _ = _compute_rho_and_lambda(
        coeff_by_band=coeff, window=WINDOW, ridge=RIDGE, cov_shrinkage=COV_SHRINKAGE
    )
    pi_proxy = _compute_pi_proxy(u=u, v=v, masks=masks, ky=ky, kx=kx)
    return lambda_mu, pi_proxy  # lambda_sign fixed +1: no flip anywhere


# ---------------------------------------------------------------------------
# Phase-randomized surrogates (protocol: shared phase field for u and v)
# ---------------------------------------------------------------------------

def phase_randomize(u: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    nt, ny, nx = u.shape
    us = np.empty_like(u)
    vs = np.empty_like(v)
    for t in range(nt):
        w = rng.standard_normal((ny, nx))
        factor = np.exp(1j * np.angle(np.fft.rfft2(w)))
        factor[0, 0] = 1.0  # preserve the snapshot mean
        us[t] = np.fft.irfft2(np.fft.rfft2(u[t]) * factor, s=(ny, nx))
        vs[t] = np.fft.irfft2(np.fft.rfft2(v[t]) * factor, s=(ny, nx))
    return us, vs


_G: dict[str, object] = {}


def _surrogate_worker_init(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray, pi_true: np.ndarray) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _G["u"], _G["v"], _G["lat"], _G["lon"], _G["pi"] = u, v, lat, lon, pi_true


def _surrogate_worker(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    us, vs = phase_randomize(_G["u"], _G["v"], rng)  # type: ignore[arg-type]
    lam, _ = lambda_and_proxy(us, vs, _G["lat"], _G["lon"])  # type: ignore[arg-type]
    pi_true = _G["pi"]  # type: ignore[assignment]
    out = []
    for j, b in enumerate(INERTIAL_BANDS):
        r, _ = _pearson(lam[WARMUP:, b], pi_true[WARMUP:, j])  # type: ignore[index]
        out.append(r * r if np.isfinite(r) else 0.0)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 10:
        return np.nan, n
    xs, ys = x[m], y[m]
    sx, sy = np.std(xs), np.std(ys)
    if sx < 1e-300 or sy < 1e-300:
        return np.nan, n
    return float(np.mean((xs - xs.mean()) * (ys - ys.mean())) / (sx * sy)), n


def _block_bootstrap_ci(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT, block: int = BLOCK_LEN
) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[m], y[m]
    n = len(xs)
    if n < 3 * block:
        return np.nan, np.nan
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    rs = []
    for _ in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        r, _ = _pearson(xs[idx], ys[idx])
        if np.isfinite(r):
            rs.append(r)
    if len(rs) < 100:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _oos_r2(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[m], y[m]
    n = len(xs)
    if n < 40:
        return np.nan
    half = n // 2
    x_tr, y_tr, x_te, y_te = xs[:half], ys[:half], xs[half:], ys[half:]
    if np.std(x_tr) < 1e-300:
        return np.nan
    slope, intercept = np.polyfit(x_tr, y_tr, 1)
    pred = slope * x_te + intercept
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    if ss_tot < 1e-300:
        return np.nan
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Per region-window driver
# ---------------------------------------------------------------------------

def run_region_window(
    nc_path: Path, out_dir: Path, n_surrogates: int, workers: int, seed: int
) -> dict:
    tag = nc_path.stem.replace("era5_wind850_", "")
    print(f"[{tag}] loading", flush=True)
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path,
        field_set="wind",
        u_var=None,
        v_var=None,
        time_stride=1,
        lat_stride=1,
        lon_stride=1,
        time_start=0,
        max_time=None,
        crop_ny=None,
        crop_nx=None,
    )
    centers = band_centers_km(SCALE_EDGES_KM)
    ells = [centers[b] for b in INERTIAL_BANDS]

    print(f"[{tag}] true flux + gridded baselines", flush=True)
    grids = flux_and_gridded_baselines(u, v, lat, lon, ells)
    print(f"[{tag}] lambda pipeline", flush=True)
    lam, pi_proxy = lambda_and_proxy(u, v, lat, lon)

    rng = np.random.default_rng(seed)
    rows = []
    for j, b in enumerate(INERTIAL_BANDS):
        target = grids["pi"][WARMUP:, j]
        preds = {
            "lambda": lam[WARMUP:, b],
            "B1_smag": grids["B1_smag"][WARMUP:, j],
            "B2_a15proxy": pi_proxy[WARMUP:, b],
            "B3_enstrophy": grids["B3_enstrophy"][WARMUP:, j],
            "B4_gradient": grids["B4_gradient"][WARMUP:, j],
        }
        for name, x in preds.items():
            r, n = _pearson(x, target)
            rho = spearmanr(x, target, nan_policy="omit").statistic
            lo, hi = _block_bootstrap_ci(x, target, rng)
            rows.append(
                {
                    "region_window": tag,
                    "band": b,
                    "band_center_km": centers[b],
                    "predictor": name,
                    "n": n,
                    "pearson_r": r,
                    "raw_r2": r * r if np.isfinite(r) else np.nan,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "r_ci_lo": lo,
                    "r_ci_hi": hi,
                    "oos_r2": _oos_r2(x, target),
                }
            )

    print(f"[{tag}] surrogates n={n_surrogates}", flush=True)
    seeds = [int(s) for s in np.random.SeedSequence(seed + 7919).generate_state(n_surrogates)]
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=workers,
            initializer=_surrogate_worker_init,
            initargs=(u, v, lat, lon, grids["pi"]),
        ) as pool:
            sur = pool.map(_surrogate_worker, seeds)
    else:
        _surrogate_worker_init(u, v, lat, lon, grids["pi"])
        sur = [_surrogate_worker(s) for s in seeds]
    sur_arr = np.asarray(sur)  # (n_surrogates, n_inertial_bands)

    surrogate = {}
    for j, b in enumerate(INERTIAL_BANDS):
        obs = next(
            r["raw_r2"] for r in rows if r["band"] == b and r["predictor"] == "lambda"
        )
        thr = float(np.percentile(sur_arr[:, j], 95))
        surrogate[str(b)] = {
            "observed_r2": obs,
            "surrogate_p95_r2": thr,
            "surrogate_median_r2": float(np.median(sur_arr[:, j])),
            "pass": bool(np.isfinite(obs) and obs > thr),
        }

    result = {
        "tag": tag,
        "n_time": int(u.shape[0]),
        "rows": rows,
        "surrogate": surrogate,
        "n_surrogates": n_surrogates,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{tag}] done", flush=True)
    return result


# ---------------------------------------------------------------------------
# Consolidation against the frozen criteria C1-C4
# ---------------------------------------------------------------------------

def consolidate(results: list[dict], out_dir: Path) -> None:
    df = pd.DataFrame([r for res in results for r in res["rows"]])
    df.to_csv(out_dir / "all_rows.csv", index=False)

    lines = ["# Experiment B1: Lambda vs true cross-scale flux and baselines", ""]
    lines.append(f"Region-windows analyzed: {len(results)} of 8 required.")
    lines.append("")

    verdict_per_band: dict[int, dict] = {}
    for b in INERTIAL_BANDS:
        band = df[df["band"] == b]
        med = band.groupby("predictor")["raw_r2"].median()
        lam_rows = band[band["predictor"] == "lambda"]
        signs = np.sign(lam_rows["pearson_r"].dropna())
        dominant = int(max((signs > 0).sum(), (signs < 0).sum()))
        sur_pass = sum(
            1 for res in results if res["surrogate"].get(str(b), {}).get("pass", False)
        )
        oos_med = float(lam_rows["oos_r2"].median())
        c1 = bool(med.get("lambda", np.nan) > 0.05 and dominant >= 6)
        baselines = [p for p in PREDICTORS if p != "lambda"]
        c2 = bool(all(med.get("lambda", -np.inf) > med.get(p, -np.inf) for p in baselines))
        c3 = bool(sur_pass >= 5)
        c4 = bool(np.isfinite(oos_med) and oos_med > 0)
        verdict_per_band[b] = {
            "median_r2": {k: float(vv) for k, vv in med.items()},
            "sign_stability": f"{dominant}/{len(signs)}",
            "surrogate_pass_count": sur_pass,
            "lambda_oos_r2_median": oos_med,
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": c4,
        }

    bands_all_pass = [
        b for b, vb in verdict_per_band.items() if vb["C1"] and vb["C2"] and vb["C3"] and vb["C4"]
    ]
    overall = len(bands_all_pass) >= 2
    summary = {
        "per_band": {str(b): verdict_per_band[b] for b in INERTIAL_BANDS},
        "bands_passing_all": bands_all_pass,
        "PHASE1_VERDICT": "POSITIVE" if overall else "NEGATIVE",
        "n_region_windows": len(results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for b in INERTIAL_BANDS:
        vb = verdict_per_band[b]
        lines.append(f"## Band {b} (center {band_centers_km(SCALE_EDGES_KM)[b]:.0f} km)")
        lines.append("")
        lines.append("| predictor | median raw R^2 |")
        lines.append("|---|---|")
        for p in PREDICTORS:
            lines.append(f"| {p} | {vb['median_r2'].get(p, float('nan')):.4f} |")
        lines.append("")
        lines.append(
            f"- sign stability (lambda): {vb['sign_stability']}; "
            f"surrogate pass: {vb['surrogate_pass_count']}/{len(results)}; "
            f"lambda OOS R^2 median: {vb['lambda_oos_r2_median']:.4f}"
        )
        lines.append(
            f"- C1={vb['C1']} C2={vb['C2']} C3={vb['C3']} C4={vb['C4']}"
        )
        lines.append("")

    lines.append(f"## PHASE1_VERDICT: {summary['PHASE1_VERDICT']}")
    lines.append("")
    lines.append(
        "Criteria (frozen): C1 median raw R^2>0.05 with sign stability >=6/8; "
        "C2 lambda beats every baseline; C3 surrogate null passed >=5/8; "
        "C4 median OOS R^2>0; overall requires >=2 of 3 inertial bands."
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/b1"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_HERE / "results" / "experiment_B1_true_flux_baselines",
    )
    parser.add_argument("--n-surrogates", type=int, default=199)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--only", type=str, default=None, help="substring filter on files")
    args = parser.parse_args()

    files = [args.data_dir / f for f in REGION_WINDOWS if (args.data_dir / f).exists()]
    if args.only:
        files = [f for f in files if args.only in f.name]
    if not files:
        raise SystemExit(f"No input files in {args.data_dir}")
    print(f"{len(files)} region-window files found")

    results = []
    for f in files:
        results.append(
            run_region_window(f, args.out_dir, args.n_surrogates, args.workers, args.seed)
        )
    consolidate(results, args.out_dir)


if __name__ == "__main__":
    main()
