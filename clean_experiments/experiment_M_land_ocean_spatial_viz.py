#!/usr/bin/env python3
"""Spatial visualization for Experiment M land/ocean analysis (article-ready maps)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        K_BOLTZMANN,
        _build_band_masks,
        _compute_vorticity,
        _edge_order,
        _load_data,
        _time_to_seconds,
        _xy_coordinates_m,
    )
    from clean_experiments.experiment_O_spatial_variance import (
        _build_land_mask,
        _compute_lambda_local_batch,
        _load_m_band_amplitudes,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        K_BOLTZMANN,
        _build_band_masks,
        _compute_vorticity,
        _edge_order,
        _load_data,
        _time_to_seconds,
        _xy_coordinates_m,
    )
    from experiment_O_spatial_variance import (  # type: ignore
        _build_land_mask,
        _compute_lambda_local_batch,
        _load_m_band_amplitudes,
    )


EPS = 1e-12


def _solve_baseline(
    *,
    n: int,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    den = n * sxx - sx * sx
    b1 = np.zeros_like(sx)
    ok = np.abs(den) > 1e-10
    b1[ok] = (n * sxy[ok] - sx[ok] * sy[ok]) / den[ok]
    b0 = (sy - b1 * sx) / float(max(n, 1))
    return b0, b1


def _solve_full(
    *,
    n: int,
    sx1: np.ndarray,
    sx2: np.ndarray,
    sy: np.ndarray,
    sx11: np.ndarray,
    sx22: np.ndarray,
    sx12: np.ndarray,
    sx1y: np.ndarray,
    sx2y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npix = len(sx1)
    b0 = np.zeros(npix, dtype=float)
    b1 = np.zeros(npix, dtype=float)
    b2 = np.zeros(npix, dtype=float)
    for j in range(npix):
        lhs = np.array(
            [
                [float(n), float(sx1[j]), float(sx2[j])],
                [float(sx1[j]), float(sx11[j]), float(sx12[j])],
                [float(sx2[j]), float(sx12[j]), float(sx22[j])],
            ],
            dtype=float,
        )
        rhs = np.array([float(sy[j]), float(sx1y[j]), float(sx2y[j])], dtype=float)
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        b0[j], b1[j], b2[j] = float(beta[0]), float(beta[1]), float(beta[2])
    return b0, b1, b2


def _zscore_time(arr_tyx: np.ndarray) -> np.ndarray:
    mu = np.mean(arr_tyx, axis=0, keepdims=True)
    sd = np.std(arr_tyx, axis=0, keepdims=True)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return (arr_tyx - mu) / sd


def _plot_map(
    *,
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    land_mask: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
) -> None:
    arr = np.asarray(field, dtype=float)
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")

    vmin = np.nanquantile(arr, 0.02)
    vmax = np.nanquantile(arr, 0.98)
    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)

    im = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(lon2d, lat2d, land_mask.astype(float), levels=[0.5], colors="k", linewidths=0.6, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lon)), float(np.max(lon)))
    ax.set_ylim(float(np.min(lat)), float(np.max(lat)))
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_distribution(
    *,
    gain_map: np.ndarray,
    land_mask: np.ndarray,
    out_path: Path,
) -> None:
    g = np.asarray(gain_map, dtype=float)
    land = g[np.isfinite(g) & land_mask]
    ocean = g[np.isfinite(g) & (~land_mask)]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hist(land, bins=50, alpha=0.7, color="#8c564b", density=True, label=f"land (n={len(land)})")
    ax.hist(ocean, bins=50, alpha=0.6, color="#1f77b4", density=True, label=f"ocean (n={len(ocean)})")
    ax.axvline(float(np.nanmean(land)), color="#8c564b", linestyle="--", linewidth=1.2)
    ax.axvline(float(np.nanmean(ocean)), color="#1f77b4", linestyle="--", linewidth=1.2)
    ax.set_title("Spatial gain distribution: land vs ocean")
    ax.set_xlabel("Local gain in R2 (test year)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_experiment(
    *,
    input_nc: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    outdir: Path,
    scale_edges_km: list[float],
    train_end_year: int,
    test_year: int,
    batch_size: int,
    precip_factor: float,
    evap_factor: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    print("[M-spatial-viz] Step 1/6: load stable M-band amplitudes...", flush=True)
    band_amp, band_diag = _load_m_band_amplitudes(
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
    )
    band_diag.to_csv(outdir / "lambda_band_diagnostics.csv", index=False)

    print("[M-spatial-viz] Step 2/6: load fields and build target/control...", flush=True)
    loaded = _load_data(
        input_nc,
        var_overrides={
            "iwv": None,
            "ivt_u": None,
            "ivt_v": None,
            "precip": None,
            "evap": None,
            "u": None,
            "v": None,
            "temp": None,
            "pressure": None,
            "density": None,
            "temp_pl": None,
            "q_pl": None,
            "u_pl": None,
            "v_pl": None,
            "w_pl": None,
        },
        level_dim=None,
        level_index=0,
        time_stride=1,
        lat_stride=1,
        lon_stride=1,
        max_time=None,
    )
    f = loaded.fields
    nt, ny, nx = f["iwv"].shape
    npix = ny * nx
    if band_amp.shape[0] != nt:
        raise ValueError(f"M band length mismatch: {band_amp.shape[0]} vs nt={nt}")

    time_dt = pd.to_datetime(loaded.time)
    years = time_dt.year.to_numpy(dtype=int)
    train_mask = years <= int(train_end_year)
    test_mask = years == int(test_year)
    n_train = int(np.sum(train_mask))
    n_test = int(np.sum(test_mask))
    if n_train < 500 or n_test < 200:
        raise ValueError(f"Too few train/test samples: train={n_train}, test={n_test}")

    land_mask = _build_land_mask(loaded.lat, loaded.lon)

    time_s = _time_to_seconds(loaded.time)
    x_m, y_m = _xy_coordinates_m(loaded.lat, loaded.lon)
    dx_km = abs(float(np.median(np.diff(x_m))) / 1000.0)
    dy_km = abs(float(np.median(np.diff(y_m))) / 1000.0)
    masks, _, _, _, _, _ = _build_band_masks(
        ny=ny,
        nx=nx,
        dy_km=dy_km,
        dx_km=dx_km,
        scale_edges_km=scale_edges_km,
    )
    if len(masks) != band_amp.shape[1]:
        raise ValueError(f"Band mismatch: masks={len(masks)} vs M bands={band_amp.shape[1]}")

    eo_t = _edge_order(len(time_s))
    eo_x = _edge_order(len(x_m))
    eo_y = _edge_order(len(y_m))
    diwv_dt = np.gradient(f["iwv"], time_s, axis=0, edge_order=eo_t)
    div_ivt = np.gradient(f["ivt_u"], x_m, axis=2, edge_order=eo_x) + np.gradient(f["ivt_v"], y_m, axis=1, edge_order=eo_y)
    p_minus_e = precip_factor * f["precip"] - evap_factor * f["evap"]
    residual_local = diwv_dt + div_ivt + p_minus_e
    del diwv_dt
    del div_ivt
    del p_minus_e

    if "density" in f:
        n_density_local = f["density"]
    elif "pressure" in f and "temp" in f:
        n_density_local = f["pressure"] / (K_BOLTZMANN * np.maximum(f["temp"], 1e-6))
    else:
        n_density_local = np.ones_like(residual_local, dtype=float)
    ctrl_local = np.log(np.maximum(n_density_local, 1e-30))
    del n_density_local

    # Normalize per cell to match residual/control comparability across space.
    y_all = _zscore_time(np.asarray(residual_local, dtype=np.float64)).reshape(nt, npix)
    x1_all = _zscore_time(np.asarray(ctrl_local, dtype=np.float64)).reshape(nt, npix)
    del residual_local
    del ctrl_local

    print("[M-spatial-viz] Step 3/6: reconstruct Lambda_local and gather train/test stats...", flush=True)
    sx1_tr = np.zeros(npix, dtype=np.float64)
    sx2_tr = np.zeros(npix, dtype=np.float64)
    sy_tr = np.zeros(npix, dtype=np.float64)
    sx11_tr = np.zeros(npix, dtype=np.float64)
    sx22_tr = np.zeros(npix, dtype=np.float64)
    sx12_tr = np.zeros(npix, dtype=np.float64)
    sx1y_tr = np.zeros(npix, dtype=np.float64)
    sx2y_tr = np.zeros(npix, dtype=np.float64)

    x1_test = np.empty((n_test, npix), dtype=np.float32)
    x2_test = np.empty((n_test, npix), dtype=np.float32)
    y_test = np.empty((n_test, npix), dtype=np.float32)
    test_pos = 0

    for start in range(0, nt, batch_size):
        stop = min(start + batch_size, nt)
        sl = slice(start, stop)
        tb = stop - start

        u_blk = np.asarray(f["u"][sl], dtype=np.float64)
        v_blk = np.asarray(f["v"][sl], dtype=np.float64)
        zeta_blk = _compute_vorticity(u=u_blk, v=v_blk, x_m=x_m, y_m=y_m)
        lam_blk = _compute_lambda_local_batch(
            zeta_batch=zeta_blk,
            masks=masks,
            band_amp_batch=band_amp[start:stop],
        )
        x2_flat = lam_blk.reshape(tb, npix).astype(np.float64)
        y_flat = y_all[start:stop]
        x1_flat = x1_all[start:stop]

        rel_tr = np.where(train_mask[start:stop])[0]
        if len(rel_tr) > 0:
            ytr = y_flat[rel_tr]
            x1tr = x1_flat[rel_tr]
            x2tr = x2_flat[rel_tr]
            sy_tr += np.sum(ytr, axis=0)
            sx1_tr += np.sum(x1tr, axis=0)
            sx2_tr += np.sum(x2tr, axis=0)
            sx11_tr += np.sum(x1tr * x1tr, axis=0)
            sx22_tr += np.sum(x2tr * x2tr, axis=0)
            sx12_tr += np.sum(x1tr * x2tr, axis=0)
            sx1y_tr += np.sum(x1tr * ytr, axis=0)
            sx2y_tr += np.sum(x2tr * ytr, axis=0)

        rel_te = np.where(test_mask[start:stop])[0]
        if len(rel_te) > 0:
            n_add = len(rel_te)
            x1_test[test_pos : test_pos + n_add] = x1_flat[rel_te].astype(np.float32)
            x2_test[test_pos : test_pos + n_add] = x2_flat[rel_te].astype(np.float32)
            y_test[test_pos : test_pos + n_add] = y_flat[rel_te].astype(np.float32)
            test_pos += n_add

        print(f"[M-spatial-viz] progress: {stop}/{nt}", flush=True)

    if test_pos != n_test:
        raise RuntimeError(f"Stored test rows mismatch: {test_pos} vs {n_test}")

    print("[M-spatial-viz] Step 4/6: solve local regressions...", flush=True)
    b0_base, b1_base = _solve_baseline(
        n=n_train,
        sx=sx1_tr,
        sy=sy_tr,
        sxx=sx11_tr,
        sxy=sx1y_tr,
    )
    b0_full, b1_full, b2_full = _solve_full(
        n=n_train,
        sx1=sx1_tr,
        sx2=sx2_tr,
        sy=sy_tr,
        sx11=sx11_tr,
        sx22=sx22_tr,
        sx12=sx12_tr,
        sx1y=sx1y_tr,
        sx2y=sx2y_tr,
    )

    print("[M-spatial-viz] Step 5/6: compute test R2 gain map...", flush=True)
    y_test64 = y_test.astype(np.float64)
    sy_te = np.sum(y_test64, axis=0)
    sy2_te = np.sum(y_test64 * y_test64, axis=0)
    y_mean_te = sy_te / float(max(n_test, 1))
    sst_te = sy2_te - float(n_test) * y_mean_te * y_mean_te

    sse_b = np.zeros(npix, dtype=np.float64)
    sse_f = np.zeros(npix, dtype=np.float64)
    chunk = 512
    for j0 in range(0, npix, chunk):
        j1 = min(j0 + chunk, npix)
        x1 = x1_test[:, j0:j1].astype(np.float64)
        x2 = x2_test[:, j0:j1].astype(np.float64)
        yv = y_test64[:, j0:j1]
        yhat_b = b0_base[None, j0:j1] + b1_base[None, j0:j1] * x1
        yhat_f = b0_full[None, j0:j1] + b1_full[None, j0:j1] * x1 + b2_full[None, j0:j1] * x2
        sse_b[j0:j1] = np.sum((yv - yhat_b) ** 2, axis=0)
        sse_f[j0:j1] = np.sum((yv - yhat_f) ** 2, axis=0)

    valid = sst_te > 1e-12
    r2_b = np.full(npix, np.nan, dtype=np.float64)
    r2_f = np.full(npix, np.nan, dtype=np.float64)
    r2_b[valid] = 1.0 - sse_b[valid] / (sst_te[valid] + EPS)
    r2_f[valid] = 1.0 - sse_f[valid] / (sst_te[valid] + EPS)
    gain = r2_f - r2_b

    gain_map = gain.reshape(ny, nx)
    r2_base_map = r2_b.reshape(ny, nx)
    r2_full_map = r2_f.reshape(ny, nx)
    beta_lambda_map = b2_full.reshape(ny, nx)

    land_mean = float(np.nanmean(gain_map[land_mask]))
    ocean_mean = float(np.nanmean(gain_map[~land_mask]))
    delta = land_mean - ocean_mean
    pos_land = float(np.nanmean(gain_map[land_mask] > 0.0))
    pos_ocean = float(np.nanmean(gain_map[~land_mask] > 0.0))

    point_rows = []
    for iy in range(ny):
        for ix in range(nx):
            point_rows.append(
                {
                    "lat": float(loaded.lat[iy]),
                    "lon": float(loaded.lon[ix]),
                    "gain_r2": float(gain_map[iy, ix]),
                    "r2_base": float(r2_base_map[iy, ix]),
                    "r2_full": float(r2_full_map[iy, ix]),
                    "beta_lambda": float(beta_lambda_map[iy, ix]),
                    "is_land": bool(land_mask[iy, ix]),
                }
            )
    point_df = pd.DataFrame(point_rows)
    point_df.to_csv(outdir / "spatial_point_metrics.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "input_nc": str(input_nc),
                "m_timeseries_csv": str(m_timeseries_csv),
                "m_summary_csv": str(m_summary_csv),
                "m_mode_index_csv": str(m_mode_index_csv),
                "n_time": int(nt),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "ny": int(ny),
                "nx": int(nx),
                "n_bands": int(len(masks)),
                "train_end_year": int(train_end_year),
                "test_year": int(test_year),
                "land_mean_gain_r2": float(land_mean),
                "ocean_mean_gain_r2": float(ocean_mean),
                "delta_land_minus_ocean_gain_r2": float(delta),
                "positive_gain_frac_land": float(pos_land),
                "positive_gain_frac_ocean": float(pos_ocean),
                "median_gain_r2": float(np.nanmedian(gain_map)),
                "max_gain_r2": float(np.nanmax(gain_map)),
                "min_gain_r2": float(np.nanmin(gain_map)),
            }
        ]
    )
    summary.to_csv(outdir / "spatial_summary.csv", index=False)

    np.savez_compressed(
        outdir / "spatial_maps.npz",
        lat=np.asarray(loaded.lat, dtype=float),
        lon=np.asarray(loaded.lon, dtype=float),
        gain_map=gain_map,
        r2_base_map=r2_base_map,
        r2_full_map=r2_full_map,
        beta_lambda_map=beta_lambda_map,
        land_mask=land_mask.astype(np.int8),
    )

    print("[M-spatial-viz] Step 6/6: plots and report...", flush=True)
    _plot_map(
        field=gain_map,
        lat=np.asarray(loaded.lat, dtype=float),
        lon=np.asarray(loaded.lon, dtype=float),
        land_mask=land_mask,
        title="Experiment M: Local gain map (R2_full - R2_base)",
        cbar_label="Gain in R2 (test year)",
        out_path=outdir / "plot_spatial_gain_map.png",
        cmap="RdBu_r",
        symmetric=True,
    )
    _plot_map(
        field=beta_lambda_map,
        lat=np.asarray(loaded.lat, dtype=float),
        lon=np.asarray(loaded.lon, dtype=float),
        land_mask=land_mask,
        title="Experiment M: Local beta(Lambda) map",
        cbar_label="beta_lambda",
        out_path=outdir / "plot_spatial_beta_lambda_map.png",
        cmap="RdBu_r",
        symmetric=True,
    )
    _plot_distribution(
        gain_map=gain_map,
        land_mask=land_mask,
        out_path=outdir / "plot_spatial_gain_distribution_land_ocean.png",
    )

    report = [
        "# Experiment M Spatial Visualization (Land/Ocean)",
        "",
        "Train/test split used for map: train<=2018, test=2019.",
        "",
        "## Key metrics",
        f"- land mean gain (R2): {land_mean:.6f}",
        f"- ocean mean gain (R2): {ocean_mean:.6f}",
        f"- delta land-ocean: {delta:.6f}",
        f"- positive gain fraction land: {pos_land:.3f}",
        f"- positive gain fraction ocean: {pos_ocean:.3f}",
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[M-spatial-viz] done -> {outdir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
    )
    p.add_argument(
        "--m-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    p.add_argument(
        "--m-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_summary.csv"),
    )
    p.add_argument(
        "--m-mode-index-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_mode_index.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_land_ocean_spatial_viz"),
    )
    p.add_argument(
        "--scale-edges-km",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
    )
    p.add_argument("--train-end-year", type=int, default=2018)
    p.add_argument("--test-year", type=int, default=2019)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--precip-factor", type=float, default=1.0)
    p.add_argument("--evap-factor", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        input_nc=args.input_nc,
        m_timeseries_csv=args.m_timeseries_csv,
        m_summary_csv=args.m_summary_csv,
        m_mode_index_csv=args.m_mode_index_csv,
        outdir=args.outdir,
        scale_edges_km=list(args.scale_edges_km),
        train_end_year=args.train_end_year,
        test_year=args.test_year,
        batch_size=args.batch_size,
        precip_factor=args.precip_factor,
        evap_factor=args.evap_factor,
    )


if __name__ == "__main__":
    main()

