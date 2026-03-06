#!/usr/bin/env python3
"""F6c spatial panel visualization: patch-wise tail diagnostics for |Lambda_local(t,y,x)|."""

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
    from clean_experiments.experiment_F6b_era5_heavy_tails import F6bConfig, _scan_xmin_candidates
    from clean_experiments.experiment_F6c_clustered_subspace_tails import _extract_panel_lambda_and_conv
    from clean_experiments.experiment_O_spatial_variance import _build_land_mask
except ModuleNotFoundError:
    from experiment_F6b_era5_heavy_tails import F6bConfig, _scan_xmin_candidates  # type: ignore
    from experiment_F6c_clustered_subspace_tails import _extract_panel_lambda_and_conv  # type: ignore
    from experiment_O_spatial_variance import _build_land_mask  # type: ignore


def _build_patch_geometry(ny: int, nx: int, patch_size: int, patch_stride: int) -> tuple[np.ndarray, np.ndarray]:
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd.")
    half = patch_size // 2
    cy = np.arange(half, ny - half, patch_stride, dtype=int)
    cx = np.arange(half, nx - half, patch_stride, dtype=int)
    if len(cy) == 0 or len(cx) == 0:
        raise ValueError("Empty patch grid; adjust patch_size/patch_stride.")
    return cy, cx


def _plot_patch_map(
    field: np.ndarray,
    *,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    lat_full: np.ndarray,
    lon_full: np.ndarray,
    land_mask: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str = "viridis",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    arr = np.asarray(field, dtype=float)
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    lon2d, lat2d = np.meshgrid(lon_c, lat_c, indexing="xy")

    if vmin is None or vmax is None:
        q02 = float(np.nanquantile(arr, 0.02))
        q98 = float(np.nanquantile(arr, 0.98))
        if symmetric:
            m = max(abs(q02), abs(q98))
            vmin, vmax = -m, m
        else:
            vmin, vmax = q02, q98
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))

    im = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    lonf, latf = np.meshgrid(lon_full, lat_full, indexing="xy")
    ax.contour(lonf, latf, land_mask.astype(float), levels=[0.5], colors="k", linewidths=0.5, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lon_full)), float(np.max(lon_full)))
    ax.set_ylim(float(np.min(lat_full)), float(np.max(lat_full)))
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_soc_candidate_mask(
    mask: np.ndarray,
    *,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    lat_full: np.ndarray,
    lon_full: np.ndarray,
    land_mask: np.ndarray,
    out_path: Path,
) -> None:
    m = np.asarray(mask, dtype=float)
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    lon2d, lat2d = np.meshgrid(lon_c, lat_c, indexing="xy")
    im = ax.pcolormesh(lon2d, lat2d, m, shading="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    lonf, latf = np.meshgrid(lon_full, lat_full, indexing="xy")
    ax.contour(lonf, latf, land_mask.astype(float), levels=[0.5], colors="k", linewidths=0.5, alpha=0.8)
    ax.set_title("F6c spatial mask: strict SOC-candidate patches")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lon_full)), float(np.max(lon_full)))
    ax.set_ylim(float(np.min(lat_full)), float(np.max(lat_full)))
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label("1 = pass strict local criteria")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_alpha_vs_f5_composite(
    *,
    alpha_df: pd.DataFrame,
    f5_patch_csv: Path,
    out_path: Path,
) -> None:
    if not f5_patch_csv.exists():
        return
    f5 = pd.read_csv(f5_patch_csv)
    req = {"patch_iy", "patch_ix", "composite_delta"}
    if not req.issubset(set(f5.columns)):
        return

    m = alpha_df.merge(f5[list(req)], on=["patch_iy", "patch_ix"], how="inner")
    if len(m) < 20:
        return

    x = m["composite_delta"].to_numpy(float)
    y = m["alpha_mle"].to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 20:
        return

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.scatter(x, y, s=18, alpha=0.65, color="#1f77b4")
    if np.std(x) > 1e-12 and np.std(y) > 1e-12:
        p = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        yy = p[0] * xx + p[1]
        r = float(np.corrcoef(x, y)[0, 1])
        ax.plot(xx, yy, color="#d62728", linewidth=1.6)
        ax.text(0.03, 0.97, f"r={r:.3f}", transform=ax.transAxes, va="top")
    ax.axhspan(1.5, 2.0, color="#2ca02c", alpha=0.12)
    ax.set_xlabel("F5 composite fractal delta")
    ax.set_ylabel("Patch alpha (MLE)")
    ax.set_title("Patch alpha vs F5 fractal composite")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_spatial_viz(
    *,
    input_nc: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    outdir: Path,
    scale_edges_km: list[float],
    batch_size: int,
    patch_size: int,
    patch_stride: int,
    max_samples_patch: int,
    cfg_tail: F6bConfig,
    f5_patch_csv: Path,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260308)

    panel_abs, conv_mean, time_ns = _extract_panel_lambda_and_conv(
        input_nc=input_nc,
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
        scale_edges_km=scale_edges_km,
        batch_size=batch_size,
    )
    nt, npix = panel_abs.shape
    ny = 81
    nx = 161
    if npix != ny * nx:
        # fallback for potential alternate grid
        ny = int(round(np.sqrt(npix)))
        nx = int(npix // max(ny, 1))

    with Dataset(input_nc, mode="r") as ds:
        lat_name = "latitude" if "latitude" in ds.variables else "lat"
        lon_name = "longitude" if "longitude" in ds.variables else "lon"
        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)

    if len(lat) * len(lon) != panel_abs.shape[1]:
        raise ValueError("Grid mismatch between reconstructed panel and NetCDF coordinates.")

    panel3d = panel_abs.reshape(nt, len(lat), len(lon))
    land_mask = _build_land_mask(lat, lon)

    centers_y, centers_x = _build_patch_geometry(len(lat), len(lon), patch_size=patch_size, patch_stride=patch_stride)
    lat_c = lat[centers_y]
    lon_c = lon[centers_x]
    half = patch_size // 2

    rows = []
    for iy, cy in enumerate(centers_y):
        y0 = int(cy - half)
        y1 = int(cy + half + 1)
        for ix, cx in enumerate(centers_x):
            x0 = int(cx - half)
            x1 = int(cx + half + 1)
            vals = panel3d[:, y0:y1, x0:x1].reshape(-1).astype(float)
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            n_total = int(len(vals))
            if n_total == 0:
                rows.append(
                    {
                        "patch_iy": iy,
                        "patch_ix": ix,
                        "lat_center": float(lat[cy]),
                        "lon_center": float(lon[cx]),
                        "n_total": 0,
                        "n_used": 0,
                        "fit_available": False,
                    }
                )
                continue

            if n_total > max_samples_patch:
                idx = rng.choice(n_total, size=max_samples_patch, replace=False)
                vals_use = vals[idx]
            else:
                vals_use = vals

            cand, best = _scan_xmin_candidates(vals_use, regime=f"patch_{iy}_{ix}", cfg=cfg_tail)
            if best is None:
                rows.append(
                    {
                        "patch_iy": iy,
                        "patch_ix": ix,
                        "lat_center": float(lat[cy]),
                        "lon_center": float(lon[cx]),
                        "n_total": n_total,
                        "n_used": int(len(vals_use)),
                        "fit_available": False,
                    }
                )
                continue

            row = dict(best)
            row["patch_iy"] = iy
            row["patch_ix"] = ix
            row["lat_center"] = float(lat[cy])
            row["lon_center"] = float(lon[cx])
            row["n_total"] = n_total
            row["n_used"] = int(len(vals_use))
            row["fit_available"] = True
            row["pass_alpha_target_1p5_2p0"] = bool(1.5 <= float(row["alpha_mle"]) <= 2.0)
            row["pass_strict_patch"] = bool(
                bool(row["pass_dynamic_range"]) and bool(row["pass_llr"]) and bool(row["pass_alpha_target_1p5_2p0"])
            )
            rows.append(row)

        print(f"[F6c-spatial] processed patch row {iy + 1}/{len(centers_y)}", flush=True)

    patch_df = pd.DataFrame(rows)
    patch_df.to_csv(outdir / "experiment_F6c_spatial_patch_tail_metrics.csv", index=False)

    # Build maps
    nyc = len(centers_y)
    nxc = len(centers_x)
    alpha_map = np.full((nyc, nxc), np.nan, dtype=float)
    ks_map = np.full((nyc, nxc), np.nan, dtype=float)
    dyn_map = np.full((nyc, nxc), np.nan, dtype=float)
    p_map = np.full((nyc, nxc), np.nan, dtype=float)
    pass_map = np.zeros((nyc, nxc), dtype=float)
    nmap = np.full((nyc, nxc), np.nan, dtype=float)

    for _, r in patch_df.iterrows():
        i = int(r["patch_iy"])
        j = int(r["patch_ix"])
        nmap[i, j] = float(r.get("n_used", np.nan))
        if bool(r.get("fit_available", False)):
            alpha_map[i, j] = float(r.get("alpha_mle", np.nan))
            ks_map[i, j] = float(r.get("ks_distance", np.nan))
            dyn_map[i, j] = float(r.get("dynamic_range", np.nan))
            p_map[i, j] = float(r.get("llr_p_value", np.nan))
            pass_map[i, j] = 1.0 if bool(r.get("pass_strict_patch", False)) else 0.0

    _plot_patch_map(
        alpha_map,
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        title="F6c spatial alpha map (patch-wise MLE)",
        cbar_label="alpha",
        out_path=outdir / "plot_F6c_spatial_alpha_map.png",
        cmap="magma",
        symmetric=False,
    )
    _plot_patch_map(
        ks_map,
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        title="F6c spatial KS map (lower is better)",
        cbar_label="KS distance",
        out_path=outdir / "plot_F6c_spatial_ks_map.png",
        cmap="viridis",
        symmetric=False,
    )
    _plot_patch_map(
        dyn_map,
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        title="F6c spatial dynamic range map",
        cbar_label="x_max/x_min",
        out_path=outdir / "plot_F6c_spatial_dynamic_range_map.png",
        cmap="YlGnBu",
        symmetric=False,
    )
    _plot_patch_map(
        np.log10(np.maximum(p_map, 1e-300)),
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        title="F6c spatial log10 p-value map (LLR PL vs Exp)",
        cbar_label="log10(p)",
        out_path=outdir / "plot_F6c_spatial_log10p_map.png",
        cmap="cividis",
        symmetric=False,
    )
    _plot_soc_candidate_mask(
        pass_map,
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        out_path=outdir / "plot_F6c_spatial_soc_candidate_mask.png",
    )
    _plot_alpha_vs_f5_composite(
        alpha_df=patch_df,
        f5_patch_csv=f5_patch_csv,
        out_path=outdir / "plot_F6c_alpha_vs_f5_composite.png",
    )

    top_alpha = (
        patch_df[patch_df["fit_available"] == True]  # noqa: E712
        .sort_values("alpha_mle", ascending=True)
        .head(20)
        .copy()
    )
    top_alpha.to_csv(outdir / "experiment_F6c_spatial_top20_low_alpha.csv", index=False)

    n_fit = int(np.sum(patch_df["fit_available"].fillna(False).to_numpy(bool)))
    n_pass = int(np.sum(patch_df["pass_strict_patch"].fillna(False).to_numpy(bool)))
    alpha_vals = patch_df.loc[patch_df["fit_available"] == True, "alpha_mle"].to_numpy(float)  # noqa: E712
    alpha_min = float(np.nanmin(alpha_vals)) if len(alpha_vals) else float("nan")
    alpha_q10 = float(np.nanquantile(alpha_vals, 0.10)) if len(alpha_vals) else float("nan")
    alpha_med = float(np.nanmedian(alpha_vals)) if len(alpha_vals) else float("nan")

    summary = pd.DataFrame(
        [
            {
                "n_time": int(nt),
                "n_space": int(panel_abs.shape[1]),
                "patch_size": int(patch_size),
                "patch_stride": int(patch_stride),
                "n_patch_y": int(nyc),
                "n_patch_x": int(nxc),
                "n_patches_total": int(nyc * nxc),
                "n_patches_fit_available": int(n_fit),
                "n_patches_strict_soc_candidate": int(n_pass),
                "alpha_min": alpha_min,
                "alpha_q10": alpha_q10,
                "alpha_median": alpha_med,
                "alpha_target_low": 1.5,
                "alpha_target_high": 2.0,
            }
        ]
    )
    summary.to_csv(outdir / "experiment_F6c_spatial_summary.csv", index=False)

    report_lines = [
        "# F6c Spatial Panel Visualization",
        "",
        "Patch-wise tail fits over |Lambda_local(t,y,x)|.",
        f"- panel shape: nt={nt}, ny={len(lat)}, nx={len(lon)}",
        f"- patch grid: {nyc}x{nxc}, patch_size={patch_size}, stride={patch_stride}",
        f"- fit-available patches: {n_fit}/{nyc * nxc}",
        f"- strict SOC-candidate patches (dynamic+LLR+alpha in [1.5,2.0]): {n_pass}",
        f"- alpha min / q10 / median: {alpha_min:.4f} / {alpha_q10:.4f} / {alpha_med:.4f}",
        "",
        "## Main files",
        "- experiment_F6c_spatial_patch_tail_metrics.csv",
        "- experiment_F6c_spatial_summary.csv",
        "- experiment_F6c_spatial_top20_low_alpha.csv",
        "- plot_F6c_spatial_alpha_map.png",
        "- plot_F6c_spatial_ks_map.png",
        "- plot_F6c_spatial_dynamic_range_map.png",
        "- plot_F6c_spatial_log10p_map.png",
        "- plot_F6c_spatial_soc_candidate_mask.png",
        "- plot_F6c_alpha_vs_f5_composite.png",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "n_patches_total": int(nyc * nxc),
        "n_patches_fit_available": int(n_fit),
        "n_patches_soc_candidate": int(n_pass),
        "alpha_min": alpha_min,
        "alpha_q10": alpha_q10,
        "alpha_median": alpha_med,
    }


def main() -> None:
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
        "--out",
        type=Path,
        default=Path("clean_experiments/results/experiment_F6c_spatial_panel_viz"),
    )
    parser.add_argument(
        "--f5-patch-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_F5_spatial_fractal_maps/experiment_F5_spatial_patch_metrics.csv"),
    )
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--patch-stride", type=int, default=6)
    parser.add_argument("--max-samples-patch", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--scale-edges-km",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
    )
    parser.add_argument("--pmin", type=float, default=70.0)
    parser.add_argument("--pmax", type=float, default=99.0)
    parser.add_argument("--pgrid", type=int, default=100)
    parser.add_argument("--min-tail-points", type=int, default=80)
    parser.add_argument("--alpha-boot-iters", type=int, default=200)
    args = parser.parse_args()

    cfg_tail = F6bConfig(
        pmin=float(args.pmin),
        pmax=float(args.pmax),
        pgrid=max(20, int(args.pgrid)),
        min_tail_points=max(40, int(args.min_tail_points)),
        alpha_bootstrap_iters=max(0, int(args.alpha_boot_iters)),
    )

    summary = run_spatial_viz(
        input_nc=args.input_nc,
        m_timeseries_csv=args.m_timeseries_csv,
        m_summary_csv=args.m_summary_csv,
        m_mode_index_csv=args.m_mode_index_csv,
        outdir=args.out,
        scale_edges_km=[float(v) for v in args.scale_edges_km],
        batch_size=max(8, int(args.batch_size)),
        patch_size=max(5, int(args.patch_size)),
        patch_stride=max(1, int(args.patch_stride)),
        max_samples_patch=max(20000, int(args.max_samples_patch)),
        cfg_tail=cfg_tail,
        f5_patch_csv=args.f5_patch_csv,
    )
    print(summary)


if __name__ == "__main__":
    main()
