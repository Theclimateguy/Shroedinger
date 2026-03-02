#!/usr/bin/env python3
"""Experiment O: spatial visualization for reconstructed Lambda_local(t,y,x)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from clean_experiments.experiment_M_cosmo_flow import _build_band_masks, _compute_vorticity, _xy_coordinates_m
    from clean_experiments.experiment_O_spatial_variance import (
        _build_land_mask,
        _compute_lambda_local_batch,
        _find_coord_name,
        _find_var_name,
        _load_m_band_amplitudes,
        _to_datetime64,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _build_band_masks, _compute_vorticity, _xy_coordinates_m  # type: ignore
    from experiment_O_spatial_variance import (  # type: ignore
        _build_land_mask,
        _compute_lambda_local_batch,
        _find_coord_name,
        _find_var_name,
        _load_m_band_amplitudes,
        _to_datetime64,
    )


EPS = 1e-12


def _plot_map(
    *,
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str,
    symmetric: bool,
    contour_mask: np.ndarray | None = None,
) -> None:
    arr = np.asarray(field, dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")

    vmin = np.nanquantile(arr, 0.02)
    vmax = np.nanquantile(arr, 0.98)
    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)

    im = ax.pcolormesh(
        lon2d,
        lat2d,
        arr,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if contour_mask is not None:
        m = np.asarray(contour_mask, dtype=float)
        ax.contour(
            lon2d,
            lat2d,
            m,
            levels=[0.5],
            colors="black",
            linewidths=0.8,
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lon)), float(np.max(lon)))
    ax.set_ylim(float(np.min(lat)), float(np.max(lat)))
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_scatter(
    *,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[m]
    yv = yv[m]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.scatter(xv, yv, s=9, alpha=0.45, color="#1f77b4", edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(xv) >= 10 and np.std(xv) > 1e-12 and np.std(yv) > 1e-12:
        p = np.polyfit(xv, yv, deg=1)
        xx = np.linspace(float(np.min(xv)), float(np.max(xv)), 200)
        yy = p[0] * xx + p[1]
        ax.plot(xx, yy, color="#d62728", lw=1.5)
        r = float(np.corrcoef(xv, yv)[0, 1])
        ax.text(0.03, 0.97, f"r={r:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_domain_series(
    *,
    dom_df: pd.DataFrame,
    out_path: Path,
    roll_steps: int,
) -> None:
    x = pd.to_datetime(dom_df["time"])
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.plot(x, dom_df["lambda_domain_mean"], color="#1f77b4", lw=0.9, alpha=0.55, label="Domain")
    ax.plot(x, dom_df["lambda_west_mean"], color="#ff7f0e", lw=0.9, alpha=0.55, label="West")
    ax.plot(x, dom_df["lambda_east_mean"], color="#2ca02c", lw=0.9, alpha=0.55, label="East")

    if roll_steps > 1:
        ax.plot(
            x,
            dom_df["lambda_domain_mean"].rolling(roll_steps, min_periods=max(2, roll_steps // 3)).mean(),
            color="#1f77b4",
            lw=1.6,
            label=f"Domain rolling ({roll_steps} steps)",
        )

    ax.set_title("Lambda_local domain means (time series)")
    ax.set_ylabel("Lambda")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_visualization(
    *,
    input_nc: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    outdir: Path,
    scale_edges_km: list[float],
    west_split_lon: float,
    active_quantile: float,
    batch_size: int,
    rolling_steps: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    print("[lambda_viz] Step 1/5: load Experiment M band amplitudes...", flush=True)
    band_amp, band_diag = _load_m_band_amplitudes(
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
    )
    band_diag.to_csv(outdir / "lambda_band_diagnostics.csv", index=False)

    print("[lambda_viz] Step 2/5: open data and initialize accumulators...", flush=True)
    with Dataset(input_nc, mode="r") as ds:
        time_name = _find_coord_name(ds, ("valid_time", "time", "datetime", "date"), "time")
        lat_name = _find_coord_name(ds, ("latitude", "lat", "y", "rlat"), "latitude")
        lon_name = _find_coord_name(ds, ("longitude", "lon", "x", "rlon"), "longitude")

        u_name = _find_var_name(ds, ("u",), "u")
        v_name = _find_var_name(ds, ("v",), "v")
        conv_name = _find_var_name(ds, ("convective_precip", "cp", "cp_rate", "precip"), "convective precipitation")

        time_ns = _to_datetime64(ds.variables[time_name])
        time_dt = pd.to_datetime(time_ns)
        nt = len(time_dt)

        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)
        ny = len(lat)
        nx = len(lon)
        npix = ny * nx
        land_mask = _build_land_mask(lat, lon)

        if band_amp.shape[0] != nt:
            raise ValueError(f"M-band nt mismatch: {band_amp.shape[0]} vs {nt}")

        split_eff = float(west_split_lon)
        if not (np.any(lon <= split_eff) and np.any(lon > split_eff)):
            split_eff = float(np.median(lon))
        west_mask = np.repeat((lon[None, :] <= split_eff), ny, axis=0)
        east_mask = ~west_mask

        x_m, y_m = _xy_coordinates_m(lat, lon)
        dx_km = float(np.median(np.diff(x_m))) / 1000.0
        dy_km = float(np.median(np.diff(y_m))) / 1000.0
        masks, _, _, _, _, _ = _build_band_masks(
            ny=ny,
            nx=nx,
            dy_km=abs(dy_km),
            dx_km=abs(dx_km),
            scale_edges_km=scale_edges_km,
        )
        if len(masks) != band_amp.shape[1]:
            raise ValueError(
                f"Band mismatch: masks={len(masks)} vs M-band components={band_amp.shape[1]}. "
                "Use same scale edges as Experiment M run."
            )

        u_var = ds.variables[u_name]
        v_var = ds.variables[v_name]
        conv_var = ds.variables[conv_name]

        sum_l = np.zeros(npix, dtype=np.float64)
        sum_l2 = np.zeros(npix, dtype=np.float64)
        sum_abs_l = np.zeros(npix, dtype=np.float64)
        sum_c = np.zeros(npix, dtype=np.float64)
        sum_c2 = np.zeros(npix, dtype=np.float64)
        sum_lc = np.zeros(npix, dtype=np.float64)

        dom_rows: list[dict[str, float | str | int]] = []

        print("[lambda_viz] Step 3/5: stream over time and reconstruct Lambda_local...", flush=True)
        for start in range(0, nt, batch_size):
            stop = min(start + batch_size, nt)
            sl = slice(start, stop)
            tb = stop - start

            u_blk = np.asarray(u_var[sl, ...], dtype=np.float64)
            v_blk = np.asarray(v_var[sl, ...], dtype=np.float64)
            c_blk = np.asarray(conv_var[sl, ...], dtype=np.float64)

            u_blk = np.nan_to_num(u_blk, nan=0.0, posinf=0.0, neginf=0.0)
            v_blk = np.nan_to_num(v_blk, nan=0.0, posinf=0.0, neginf=0.0)
            c_blk = np.nan_to_num(c_blk, nan=0.0, posinf=0.0, neginf=0.0)

            zeta_blk = _compute_vorticity(u=u_blk, v=v_blk, x_m=x_m, y_m=y_m)
            lam_blk = _compute_lambda_local_batch(
                zeta_batch=zeta_blk,
                masks=masks,
                band_amp_batch=band_amp[start:stop],
            )

            lam_flat = lam_blk.reshape(tb, npix).astype(np.float64)
            c_flat = c_blk.reshape(tb, npix).astype(np.float64)

            sum_l += np.sum(lam_flat, axis=0)
            sum_l2 += np.sum(lam_flat * lam_flat, axis=0)
            sum_abs_l += np.sum(np.abs(lam_flat), axis=0)
            sum_c += np.sum(c_flat, axis=0)
            sum_c2 += np.sum(c_flat * c_flat, axis=0)
            sum_lc += np.sum(lam_flat * c_flat, axis=0)

            lam_domain = np.mean(lam_blk, axis=(1, 2))
            lam_west = np.mean(lam_blk[:, west_mask].reshape(tb, -1), axis=1)
            lam_east = np.mean(lam_blk[:, east_mask].reshape(tb, -1), axis=1)
            for i in range(tb):
                dom_rows.append(
                    {
                        "time_index": int(start + i),
                        "time": str(time_dt[start + i]),
                        "lambda_domain_mean": float(lam_domain[i]),
                        "lambda_west_mean": float(lam_west[i]),
                        "lambda_east_mean": float(lam_east[i]),
                    }
                )

            print(f"[lambda_viz] progress: {stop}/{nt}", flush=True)

    print("[lambda_viz] Step 4/5: finalize maps and metrics...", flush=True)
    mean_l = (sum_l / float(nt)).reshape(ny, nx)
    mean_abs_l = (sum_abs_l / float(nt)).reshape(ny, nx)
    var_l = np.maximum(sum_l2 / float(nt) - (sum_l / float(nt)) ** 2, 0.0)
    std_l = np.sqrt(var_l).reshape(ny, nx)
    conv_clim = (sum_c / float(nt)).reshape(ny, nx)

    num = sum_lc - (sum_l * sum_c / float(nt))
    den_l = np.maximum(sum_l2 - (sum_l * sum_l / float(nt)), 0.0)
    den_c = np.maximum(sum_c2 - (sum_c * sum_c / float(nt)), 0.0)
    den = np.sqrt(den_l * den_c)
    corr_l_conv = np.full(npix, np.nan, dtype=np.float64)
    ok = den > 1e-12
    corr_l_conv[ok] = num[ok] / den[ok]
    corr_l_conv = corr_l_conv.reshape(ny, nx)

    active_thr = float(np.nanquantile(conv_clim, active_quantile))
    active_mask = conv_clim >= active_thr
    quiet_mask = ~active_mask

    dom_df = pd.DataFrame(dom_rows)
    dom_df.to_csv(outdir / "lambda_domain_timeseries.csv", index=False)

    valid = np.isfinite(mean_abs_l) & np.isfinite(conv_clim)
    corr_mean_abs_vs_conv = (
        float(np.corrcoef(mean_abs_l[valid], conv_clim[valid])[0, 1]) if int(np.sum(valid)) > 20 else np.nan
    )
    valid_std = np.isfinite(std_l) & np.isfinite(conv_clim)
    corr_std_vs_conv = (
        float(np.corrcoef(std_l[valid_std], conv_clim[valid_std])[0, 1]) if int(np.sum(valid_std)) > 20 else np.nan
    )
    valid_corr_land = np.isfinite(corr_l_conv) & land_mask
    valid_corr_ocean = np.isfinite(corr_l_conv) & (~land_mask)
    mean_local_corr_land = float(np.nanmean(corr_l_conv[valid_corr_land])) if int(np.sum(valid_corr_land)) > 20 else np.nan
    mean_local_corr_ocean = float(np.nanmean(corr_l_conv[valid_corr_ocean])) if int(np.sum(valid_corr_ocean)) > 20 else np.nan
    median_local_corr_land = float(np.nanmedian(corr_l_conv[valid_corr_land])) if int(np.sum(valid_corr_land)) > 20 else np.nan
    median_local_corr_ocean = float(np.nanmedian(corr_l_conv[valid_corr_ocean])) if int(np.sum(valid_corr_ocean)) > 20 else np.nan

    max_abs_idx = np.unravel_index(int(np.nanargmax(mean_abs_l)), mean_abs_l.shape)
    max_std_idx = np.unravel_index(int(np.nanargmax(std_l)), std_l.shape)

    summary = pd.DataFrame(
        [
            {
                "input_nc": str(input_nc),
                "m_timeseries_csv": str(m_timeseries_csv),
                "m_summary_csv": str(m_summary_csv),
                "m_mode_index_csv": str(m_mode_index_csv),
                "n_time": int(nt),
                "ny": int(ny),
                "nx": int(nx),
                "n_bands": int(len(masks)),
                "west_split_lon": float(split_eff),
                "convective_source": "convective_precip/cp/cp_rate/precip",
                "active_quantile": float(active_quantile),
                "active_conv_threshold": float(active_thr),
                "active_area_frac": float(np.mean(active_mask)),
                "land_area_frac": float(np.mean(land_mask)),
                "mean_abs_lambda_active": float(np.nanmean(mean_abs_l[active_mask])),
                "mean_abs_lambda_quiet": float(np.nanmean(mean_abs_l[quiet_mask])),
                "std_lambda_active": float(np.nanmean(std_l[active_mask])),
                "std_lambda_quiet": float(np.nanmean(std_l[quiet_mask])),
                "corr_spatial_mean_abs_lambda_vs_conv_clim": float(corr_mean_abs_vs_conv),
                "corr_spatial_std_lambda_vs_conv_clim": float(corr_std_vs_conv),
                "mean_local_corr_lambda_conv_land": float(mean_local_corr_land),
                "mean_local_corr_lambda_conv_ocean": float(mean_local_corr_ocean),
                "median_local_corr_lambda_conv_land": float(median_local_corr_land),
                "median_local_corr_lambda_conv_ocean": float(median_local_corr_ocean),
                "domain_lambda_mean": float(np.mean(dom_df["lambda_domain_mean"])),
                "domain_lambda_std": float(np.std(dom_df["lambda_domain_mean"])),
                "west_lambda_mean": float(np.mean(dom_df["lambda_west_mean"])),
                "east_lambda_mean": float(np.mean(dom_df["lambda_east_mean"])),
                "max_mean_abs_lambda": float(np.nanmax(mean_abs_l)),
                "max_mean_abs_lambda_lat": float(lat[max_abs_idx[0]]),
                "max_mean_abs_lambda_lon": float(lon[max_abs_idx[1]]),
                "max_std_lambda": float(np.nanmax(std_l)),
                "max_std_lambda_lat": float(lat[max_std_idx[0]]),
                "max_std_lambda_lon": float(lon[max_std_idx[1]]),
            }
        ]
    )
    summary.to_csv(outdir / "lambda_spatial_summary.csv", index=False)

    np.savez_compressed(
        outdir / "lambda_spatial_maps.npz",
        lat=lat,
        lon=lon,
        lambda_mean_map=mean_l,
        lambda_mean_abs_map=mean_abs_l,
        lambda_std_map=std_l,
        lambda_conv_corr_map=corr_l_conv,
        convective_climatology_map=conv_clim,
        active_mask=active_mask.astype(np.int8),
        land_mask=land_mask.astype(np.int8),
    )

    print("[lambda_viz] Step 5/5: save visualizations...", flush=True)
    _plot_map(
        field=mean_l,
        lat=lat,
        lon=lon,
        title="Lambda_local mean map",
        cbar_label="mean(lambda_local)",
        out_path=outdir / "plot_lambda_mean_map.png",
        cmap="RdBu_r",
        symmetric=True,
        contour_mask=active_mask,
    )
    _plot_map(
        field=mean_abs_l,
        lat=lat,
        lon=lon,
        title="Lambda_local mean absolute map",
        cbar_label="mean(|lambda_local|)",
        out_path=outdir / "plot_lambda_mean_abs_map.png",
        cmap="YlOrRd",
        symmetric=False,
        contour_mask=active_mask,
    )
    _plot_map(
        field=std_l,
        lat=lat,
        lon=lon,
        title="Lambda_local temporal std map",
        cbar_label="std(lambda_local)",
        out_path=outdir / "plot_lambda_std_map.png",
        cmap="YlGnBu",
        symmetric=False,
        contour_mask=active_mask,
    )
    _plot_map(
        field=corr_l_conv,
        lat=lat,
        lon=lon,
        title="Local corr(lambda_local(t), convective_precip(t))",
        cbar_label="corr",
        out_path=outdir / "plot_lambda_conv_corr_map.png",
        cmap="RdBu_r",
        symmetric=True,
        contour_mask=land_mask,
    )
    _plot_map(
        field=conv_clim,
        lat=lat,
        lon=lon,
        title="Convective precipitation climatology",
        cbar_label="convective climatology",
        out_path=outdir / "plot_convective_climatology_map.png",
        cmap="YlGnBu",
        symmetric=False,
        contour_mask=None,
    )
    _plot_map(
        field=land_mask.astype(float),
        lat=lat,
        lon=lon,
        title="Land mask",
        cbar_label="1=land, 0=ocean",
        out_path=outdir / "plot_land_mask.png",
        cmap="Greys",
        symmetric=False,
        contour_mask=None,
    )
    _plot_domain_series(
        dom_df=dom_df,
        out_path=outdir / "plot_lambda_domain_timeseries.png",
        roll_steps=rolling_steps,
    )
    _plot_scatter(
        x=conv_clim,
        y=mean_abs_l,
        xlabel="convective climatology",
        ylabel="mean(|lambda_local|)",
        title="Spatial relation: convective climatology vs lambda amplitude",
        out_path=outdir / "plot_scatter_conv_vs_lambda_abs.png",
    )

    report_lines = [
        "# Lambda Spatial Visualization",
        "",
        "## Key spatial diagnostics",
        f"- active quantile: {active_quantile:.3f}",
        f"- active conv threshold: {active_thr:.6e}",
        f"- corr(mean_abs_lambda, convective_clim): {corr_mean_abs_vs_conv:.6f}",
        f"- corr(std_lambda, convective_clim): {corr_std_vs_conv:.6f}",
        f"- mean_abs_lambda active vs quiet: {float(np.nanmean(mean_abs_l[active_mask])):.6e} vs {float(np.nanmean(mean_abs_l[quiet_mask])):.6e}",
        f"- std_lambda active vs quiet: {float(np.nanmean(std_l[active_mask])):.6e} vs {float(np.nanmean(std_l[quiet_mask])):.6e}",
        f"- mean local corr(lambda,conv) land vs ocean: {mean_local_corr_land:.6f} vs {mean_local_corr_ocean:.6f}",
        f"- median local corr(lambda,conv) land vs ocean: {median_local_corr_land:.6f} vs {median_local_corr_ocean:.6f}",
        f"- max mean_abs_lambda at lat={float(lat[max_abs_idx[0]]):.3f}, lon={float(lon[max_abs_idx[1]]):.3f}",
        f"- max std_lambda at lat={float(lat[max_std_idx[0]]):.3f}, lon={float(lon[max_std_idx[1]]):.3f}",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("[lambda_viz] done.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
    )
    parser.add_argument(
        "--m-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    parser.add_argument(
        "--m-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_summary.csv"),
    )
    parser.add_argument(
        "--m-mode-index-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_mode_index.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_O_lambda_spatial_viz"),
    )
    parser.add_argument(
        "--scale-edges-km",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
    )
    parser.add_argument("--west-split-lon", type=float, default=140.0)
    parser.add_argument("--active-quantile", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--rolling-steps", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_visualization(
        input_nc=args.input_nc,
        m_timeseries_csv=args.m_timeseries_csv,
        m_summary_csv=args.m_summary_csv,
        m_mode_index_csv=args.m_mode_index_csv,
        outdir=args.outdir,
        scale_edges_km=list(args.scale_edges_km),
        west_split_lon=args.west_split_lon,
        active_quantile=args.active_quantile,
        batch_size=args.batch_size,
        rolling_steps=args.rolling_steps,
    )


if __name__ == "__main__":
    main()
