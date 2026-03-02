#!/usr/bin/env python3
"""Experiment O: masked spatial analysis (active convection + west WPWP)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from global_land_mask import globe


EPS = 1e-12


def _build_land_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    lon_wrapped = ((lon2d + 180.0) % 360.0) - 180.0
    return np.asarray(globe.is_land(lat2d, lon_wrapped), dtype=bool)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    m = np.isfinite(av) & np.isfinite(bv)
    if int(np.sum(m)) < 20:
        return np.nan
    if np.std(av[m]) < 1e-15 or np.std(bv[m]) < 1e-15:
        return np.nan
    return float(np.corrcoef(av[m], bv[m])[0, 1])


def _summarize_mask(df: pd.DataFrame, mask: np.ndarray, name: str) -> dict[str, float | str]:
    sub = df.loc[mask].copy()
    if sub.empty:
        return {
            "mask": name,
            "n_cells": 0,
            "mean_gain": np.nan,
            "median_gain": np.nan,
            "positive_gain_frac": np.nan,
            "max_gain": np.nan,
            "max_gain_lat": np.nan,
            "max_gain_lon": np.nan,
            "corr_gain_conv": np.nan,
            "mean_r2_base": np.nan,
            "mean_r2_full": np.nan,
        }

    imax = int(np.nanargmax(sub["gain_r2"].to_numpy(dtype=float)))
    row_max = sub.iloc[imax]
    return {
        "mask": name,
        "n_cells": int(len(sub)),
        "mean_gain": float(np.nanmean(sub["gain_r2"])),
        "median_gain": float(np.nanmedian(sub["gain_r2"])),
        "positive_gain_frac": float(np.nanmean(sub["gain_r2"].to_numpy(dtype=float) > 0.0)),
        "max_gain": float(row_max["gain_r2"]),
        "max_gain_lat": float(row_max["lat"]),
        "max_gain_lon": float(row_max["lon"]),
        "corr_gain_conv": _corr(sub["gain_r2"].to_numpy(dtype=float), sub["convective_clim"].to_numpy(dtype=float)),
        "mean_r2_base": float(np.nanmean(sub["r2_base"])),
        "mean_r2_full": float(np.nanmean(sub["r2_full"])),
    }


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
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    vmin = np.nanquantile(arr, 0.02)
    vmax = np.nanquantile(arr, 0.98)
    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)

    im = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if contour_mask is not None:
        m = np.asarray(contour_mask, dtype=float)
        ax.contour(lon2d, lat2d, m, levels=[0.5], colors="black", linewidths=0.8)

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


def _plot_gain_distributions(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    xs = np.arange(len(summary_df))
    ax.bar(xs - 0.16, summary_df["mean_gain"], width=0.32, label="mean gain", color="#1f77b4")
    ax.bar(xs + 0.16, summary_df["median_gain"], width=0.32, label="median gain", color="#ff7f0e")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(xs, summary_df["mask"], rotation=35, ha="right")
    ax.set_ylabel("R2 gain")
    ax.set_title("Masked gain summary")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_analysis(
    *,
    input_csv: Path,
    outdir: Path,
    active_quantile: float,
    west_split_lon: float,
    top_k: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    req = {"lat", "lon", "gain_r2", "r2_base", "r2_full", "convective_clim"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(miss)}")

    lat_vals = np.array(sorted(df["lat"].unique(), reverse=True), dtype=float)
    lon_vals = np.array(sorted(df["lon"].unique()), dtype=float)
    ny, nx = len(lat_vals), len(lon_vals)

    lat_to_i = {v: i for i, v in enumerate(lat_vals)}
    lon_to_i = {v: i for i, v in enumerate(lon_vals)}
    gain_map = np.full((ny, nx), np.nan, dtype=float)
    conv_map = np.full((ny, nx), np.nan, dtype=float)
    for _, r in df.iterrows():
        iy = lat_to_i[float(r["lat"])]
        ix = lon_to_i[float(r["lon"])]
        gain_map[iy, ix] = float(r["gain_r2"])
        conv_map[iy, ix] = float(r["convective_clim"])

    land_map = _build_land_mask(lat_vals, lon_vals)

    split_eff = float(west_split_lon)
    if not (np.any(lon_vals <= split_eff) and np.any(lon_vals > split_eff)):
        split_eff = float(np.median(lon_vals))

    active_thr = float(np.nanquantile(df["convective_clim"].to_numpy(dtype=float), active_quantile))

    west_mask = df["lon"].to_numpy(dtype=float) <= split_eff
    east_mask = ~west_mask
    active_mask = df["convective_clim"].to_numpy(dtype=float) >= active_thr
    quiet_mask = ~active_mask

    # land/ocean mapped to row-level df using lookup
    land_lookup = {(lat_vals[iy], lon_vals[ix]): bool(land_map[iy, ix]) for iy in range(ny) for ix in range(nx)}
    df_land = np.array([land_lookup[(float(r["lat"]), float(r["lon"]))] for _, r in df.iterrows()], dtype=bool)
    ocean_mask = ~df_land

    mask_defs = {
        "all": np.ones(len(df), dtype=bool),
        "west": west_mask,
        "east": east_mask,
        "active": active_mask,
        "quiet": quiet_mask,
        "active_west": active_mask & west_mask,
        "active_east": active_mask & east_mask,
        "active_land": active_mask & df_land,
        "active_ocean": active_mask & ocean_mask,
        "west_land": west_mask & df_land,
        "west_ocean": west_mask & ocean_mask,
    }

    rows = [_summarize_mask(df, m, name) for name, m in mask_defs.items()]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outdir / "masked_summary.csv", index=False)

    active_west_df = df.loc[mask_defs["active_west"]].copy().sort_values("gain_r2", ascending=False)
    active_west_df.head(top_k).to_csv(outdir / "top_hotspots_active_west.csv", index=False)

    # Map masks as 2D arrays.
    west_map = np.repeat((lon_vals[None, :] <= split_eff), ny, axis=0)
    active_map = conv_map >= active_thr
    active_west_map = active_map & west_map

    _plot_map(
        field=gain_map,
        lat=lat_vals,
        lon=lon_vals,
        title="Gain map (all cells)",
        cbar_label="R2_full - R2_base",
        out_path=outdir / "plot_gain_map_all.png",
        cmap="RdBu_r",
        symmetric=True,
        contour_mask=None,
    )
    _plot_map(
        field=np.where(active_map, gain_map, np.nan),
        lat=lat_vals,
        lon=lon_vals,
        title=f"Gain map (active mask q>={active_quantile:.2f})",
        cbar_label="R2 gain (active only)",
        out_path=outdir / "plot_gain_map_active.png",
        cmap="RdBu_r",
        symmetric=True,
        contour_mask=active_map,
    )
    _plot_map(
        field=np.where(active_west_map, gain_map, np.nan),
        lat=lat_vals,
        lon=lon_vals,
        title="Gain map (active + west)",
        cbar_label="R2 gain (active-west)",
        out_path=outdir / "plot_gain_map_active_west.png",
        cmap="RdBu_r",
        symmetric=True,
        contour_mask=active_west_map,
    )
    _plot_map(
        field=land_map.astype(float),
        lat=lat_vals,
        lon=lon_vals,
        title="Land mask",
        cbar_label="1=land, 0=ocean",
        out_path=outdir / "plot_land_mask.png",
        cmap="Greys",
        symmetric=False,
        contour_mask=None,
    )

    _plot_gain_distributions(summary_df, outdir / "plot_masked_gain_summary.png")

    # Key rows for report.
    def row(mask_name: str) -> pd.Series:
        return summary_df.loc[summary_df["mask"] == mask_name].iloc[0]

    r_all = row("all")
    r_active = row("active")
    r_active_west = row("active_west")

    report = [
        "# Experiment O Masked Spatial Analysis",
        "",
        f"- input: `{input_csv}`",
        f"- active quantile: {active_quantile:.3f}",
        f"- active threshold: {active_thr:.6e}",
        f"- west split lon: {split_eff:.3f}",
        "",
        "## Key gains",
        (
            f"- mean gain all/active/active_west: "
            f"{float(r_all['mean_gain']):.6e} / {float(r_active['mean_gain']):.6e} / {float(r_active_west['mean_gain']):.6e}"
        ),
        (
            f"- median gain all/active/active_west: "
            f"{float(r_all['median_gain']):.6e} / {float(r_active['median_gain']):.6e} / {float(r_active_west['median_gain']):.6e}"
        ),
        (
            f"- positive gain frac all/active/active_west: "
            f"{float(r_all['positive_gain_frac']):.3f} / {float(r_active['positive_gain_frac']):.3f} / {float(r_active_west['positive_gain_frac']):.3f}"
        ),
        (
            f"- corr(gain,conv) all/active/active_west: "
            f"{float(r_all['corr_gain_conv']):.6f} / {float(r_active['corr_gain_conv']):.6f} / {float(r_active_west['corr_gain_conv']):.6f}"
        ),
        (
            f"- max gain active_west: {float(r_active_west['max_gain']):.6e} "
            f"at lat={float(r_active_west['max_gain_lat']):.3f}, lon={float(r_active_west['max_gain_lon']):.3f}"
        ),
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_O_spatial_variance/spatial_point_metrics.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_O_spatial_active_west"),
    )
    p.add_argument("--active-quantile", type=float, default=0.90)
    p.add_argument("--west-split-lon", type=float, default=140.0)
    p.add_argument("--top-k", type=int, default=30)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    run_analysis(
        input_csv=a.input_csv,
        outdir=a.outdir,
        active_quantile=a.active_quantile,
        west_split_lon=a.west_split_lon,
        top_k=a.top_k,
    )


if __name__ == "__main__":
    main()
