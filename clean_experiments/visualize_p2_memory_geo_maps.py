#!/usr/bin/env python3
"""Geographic maps for P2-memory l=8 response on the common tile grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prepare_df(path: Path, *, require_memory: bool) -> pd.DataFrame:
    cols = [
        "event_id",
        "mrms_obs_time_utc",
        "scale_l",
        "tile_ix",
        "tile_iy",
        "lat_center",
        "lon_center",
        "lambda_local_raw",
        "fine_occ_mean",
    ]
    if require_memory:
        cols.extend(["memory_applied", "memory_eta_raw", "memory_history_l_raw", "memory_persistence_raw"])
    df = pd.read_csv(path, usecols=cols)
    df["scale_l"] = pd.to_numeric(df["scale_l"], errors="coerce")
    df = df[np.isfinite(df["scale_l"])].copy()
    df["scale_l_int"] = np.round(df["scale_l"]).astype(int)
    df["tile_ix_int"] = np.round(pd.to_numeric(df["tile_ix"], errors="coerce")).astype(int)
    df["tile_iy_int"] = np.round(pd.to_numeric(df["tile_iy"], errors="coerce")).astype(int)
    df["lat_center"] = pd.to_numeric(df["lat_center"], errors="coerce")
    df["lon_center"] = pd.to_numeric(df["lon_center"], errors="coerce")
    return df.reset_index(drop=True)


def _grid_from_df(df: pd.DataFrame, value_col: str, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    lon_idx = {float(v): i for i, v in enumerate(lons)}
    lat_idx = {float(v): i for i, v in enumerate(lats)}
    arr = np.full((len(lats), len(lons)), np.nan, dtype=float)
    for _, row in df.iterrows():
        iy = lat_idx.get(float(row["lat_center"]))
        ix = lon_idx.get(float(row["lon_center"]))
        if iy is None or ix is None:
            continue
        arr[iy, ix] = float(row[value_col])
    return arr


def _plot_map(
    *,
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    cbar_label: str,
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    arr = np.asarray(field, dtype=float)
    if vmin is None or vmax is None:
        q02 = float(np.nanquantile(arr, 0.02))
        q98 = float(np.nanquantile(arr, 0.98))
        if symmetric:
            m = max(abs(q02), abs(q98))
            vmin, vmax = -m, m
        else:
            vmin, vmax = q02, q98
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else 0.0
        vmax = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1.0

    lon2d, lat2d = np.meshgrid(lons, lats, indexing="xy")
    im = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lons)), float(np.max(lons)))
    ax.set_ylim(float(np.min(lats)), float(np.max(lats)))
    cb = plt.colorbar(im, ax=ax, shrink=0.88)
    cb.set_label(cbar_label)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline-tile-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv"),
    )
    p.add_argument(
        "--memory-tile-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory/memory_tile_dataset_best.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory_geo_viz"),
    )
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--active-quantile", type=float, default=0.67)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base = _prepare_df(args.baseline_tile_csv, require_memory=False)
    mem = _prepare_df(args.memory_tile_csv, require_memory=True)

    base = base[base["scale_l_int"] == int(args.scale)].copy()
    mem = mem[mem["scale_l_int"] == int(args.scale)].copy()
    keys = [
        "event_id",
        "mrms_obs_time_utc",
        "scale_l_int",
        "tile_ix_int",
        "tile_iy_int",
        "lat_center",
        "lon_center",
    ]
    merged = base.merge(mem, on=keys, suffixes=("_base", "_mem"), how="inner")
    merged["delta_lambda_raw"] = merged["lambda_local_raw_mem"] - merged["lambda_local_raw_base"]
    merged["memory_push_raw"] = (
        merged["memory_eta_raw"].to_numpy(dtype=float)
        * (merged["memory_history_l_raw"].to_numpy(dtype=float) - merged["fine_occ_mean_mem"].to_numpy(dtype=float))
    )

    event_cell = (
        merged.groupby(["event_id", "lat_center", "lon_center"], as_index=False)
        .agg(
            lambda_base_mean=("lambda_local_raw_base", "mean"),
            lambda_mem_mean=("lambda_local_raw_mem", "mean"),
            delta_lambda_mean=("delta_lambda_raw", "mean"),
            fine_occ_mean=("fine_occ_mean_mem", "mean"),
            memory_push_mean=("memory_push_raw", "mean"),
            memory_eta_mean=("memory_eta_raw", "mean"),
            memory_persistence_mean=("memory_persistence_raw", "mean"),
            memory_applied_frac=("memory_applied", "mean"),
        )
        .sort_values(["lat_center", "lon_center", "event_id"])
        .reset_index(drop=True)
    )
    event_cell.to_csv(outdir / "l8_event_cell_metrics.csv", index=False)

    occ_thresh = float(event_cell["fine_occ_mean"].quantile(float(args.active_quantile)))
    event_cell["is_active"] = event_cell["fine_occ_mean"].to_numpy(dtype=float) >= occ_thresh

    cell = (
        event_cell.groupby(["lat_center", "lon_center"], as_index=False)
        .agg(
            lambda_base_mean=("lambda_base_mean", "mean"),
            lambda_mem_mean=("lambda_mem_mean", "mean"),
            delta_lambda_mean=("delta_lambda_mean", "mean"),
            delta_lambda_se=("delta_lambda_mean", lambda x: float(np.std(np.asarray(x, dtype=float), ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
            event_positive_frac=("delta_lambda_mean", lambda x: float(np.mean(np.asarray(x, dtype=float) > 0.0))),
            active_frac=("is_active", "mean"),
            occ_mean=("fine_occ_mean", "mean"),
            memory_push_mean=("memory_push_mean", "mean"),
            memory_eta_mean=("memory_eta_mean", "mean"),
            memory_persistence_mean=("memory_persistence_mean", "mean"),
            memory_applied_frac=("memory_applied_frac", "mean"),
        )
        .sort_values(["lat_center", "lon_center"])
        .reset_index(drop=True)
    )

    active_cell = (
        event_cell[event_cell["is_active"]]
        .groupby(["lat_center", "lon_center"], as_index=False)
        .agg(delta_lambda_active=("delta_lambda_mean", "mean"))
    )
    calm_cell = (
        event_cell[~event_cell["is_active"]]
        .groupby(["lat_center", "lon_center"], as_index=False)
        .agg(delta_lambda_calm=("delta_lambda_mean", "mean"))
    )
    cell = cell.merge(active_cell, on=["lat_center", "lon_center"], how="left")
    cell = cell.merge(calm_cell, on=["lat_center", "lon_center"], how="left")
    cell["active_minus_calm"] = cell["delta_lambda_active"].to_numpy(dtype=float) - cell["delta_lambda_calm"].to_numpy(dtype=float)
    cell.to_csv(outdir / "l8_cell_summary.csv", index=False)

    lons = np.sort(cell["lon_center"].unique().astype(float))
    lats = np.sort(cell["lat_center"].unique().astype(float))

    baseline_map = _grid_from_df(cell, "lambda_base_mean", lons, lats)
    memory_map = _grid_from_df(cell, "lambda_mem_mean", lons, lats)
    delta_map = _grid_from_df(cell, "delta_lambda_mean", lons, lats)
    pos_map = _grid_from_df(cell, "event_positive_frac", lons, lats)
    active_map = _grid_from_df(cell, "delta_lambda_active", lons, lats)
    calm_map = _grid_from_df(cell, "delta_lambda_calm", lons, lats)
    contrast_map = _grid_from_df(cell, "active_minus_calm", lons, lats)

    np.savez_compressed(
        outdir / "l8_geo_maps.npz",
        lon=lons,
        lat=lats,
        baseline=baseline_map,
        memory=memory_map,
        delta=delta_map,
        event_positive_frac=pos_map,
        active_delta=active_map,
        calm_delta=calm_map,
        active_minus_calm=contrast_map,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.2))
    _plot_map(
        ax=axes[0, 0],
        lons=lons,
        lats=lats,
        field=baseline_map,
        title="l=8 baseline C009 mean lambda",
        cmap="viridis",
        cbar_label="mean lambda_local_raw",
        symmetric=False,
    )
    _plot_map(
        ax=axes[0, 1],
        lons=lons,
        lats=lats,
        field=memory_map,
        title="l=8 P2-memory mean lambda",
        cmap="viridis",
        cbar_label="mean lambda_local_raw",
        symmetric=False,
    )
    _plot_map(
        ax=axes[1, 0],
        lons=lons,
        lats=lats,
        field=delta_map,
        title="l=8 mean Delta lambda (memory - baseline)",
        cmap="RdBu_r",
        cbar_label="mean Delta lambda",
        symmetric=True,
    )
    _plot_map(
        ax=axes[1, 1],
        lons=lons,
        lats=lats,
        field=pos_map,
        title="l=8 event-positive fraction for Delta lambda",
        cmap="YlOrRd",
        cbar_label="fraction of events with Delta lambda > 0",
        symmetric=False,
        vmin=0.0,
        vmax=1.0,
    )
    fig.tight_layout()
    fig.savefig(outdir / "l8_memory_geo_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))
    _plot_map(
        ax=axes[0],
        lons=lons,
        lats=lats,
        field=active_map,
        title="l=8 active-core mean Delta lambda",
        cmap="RdBu_r",
        cbar_label="Delta lambda (active)",
        symmetric=True,
    )
    _plot_map(
        ax=axes[1],
        lons=lons,
        lats=lats,
        field=calm_map,
        title="l=8 calm mean Delta lambda",
        cmap="RdBu_r",
        cbar_label="Delta lambda (calm)",
        symmetric=True,
    )
    _plot_map(
        ax=axes[2],
        lons=lons,
        lats=lats,
        field=contrast_map,
        title="l=8 active minus calm Delta lambda",
        cmap="RdBu_r",
        cbar_label="active - calm",
        symmetric=True,
    )
    fig.tight_layout()
    fig.savefig(outdir / "l8_memory_geo_regimes.png", dpi=180)
    plt.close(fig)

    peak = cell.loc[cell["delta_lambda_mean"].idxmax()]
    low = cell.loc[cell["delta_lambda_mean"].idxmin()]
    top_cells = cell.nlargest(10, "delta_lambda_mean")[
        ["lat_center", "lon_center", "delta_lambda_mean", "event_positive_frac", "active_frac"]
    ].reset_index(drop=True)
    top_cells.to_csv(outdir / "l8_top_delta_cells.csv", index=False)

    report_lines = [
        "# P2-memory geographic maps",
        "",
        "## Setup",
        f"- baseline tile csv: `{args.baseline_tile_csv}`",
        f"- memory tile csv: `{args.memory_tile_csv}`",
        f"- scale: `{int(args.scale)}`",
        f"- active quantile for regime split: `{float(args.active_quantile):.2f}`",
        "",
        "## Headline",
        f"- peak mean `Delta lambda` at `(lat, lon)=({float(peak['lat_center']):.3f}, {float(peak['lon_center']):.3f})` with value `{float(peak['delta_lambda_mean']):.6f}`",
        f"- weakest cell at `(lat, lon)=({float(low['lat_center']):.3f}, {float(low['lon_center']):.3f})` with value `{float(low['delta_lambda_mean']):.6e}`",
        f"- peak cell event-positive fraction: `{float(peak['event_positive_frac']):.3f}`",
        f"- peak cell active fraction: `{float(peak['active_frac']):.3f}`",
        "",
        "## Interpretation",
        "- The main memory lift is geographically localized rather than uniform across the domain.",
        "- The strongest positive cells cluster in the central longitudes of the panel, consistent with the zonal profile peak near `lon ~ -95.4`.",
        "- The event-positive fraction map separates persistent geographic lift from isolated high-amplitude cells.",
        "- The active/calm split map shows whether the geographic lift is concentrated in active-core cells or leaks into calm background.",
        "",
        "## Artifacts",
        "- `l8_memory_geo_overview.png`",
        "- `l8_memory_geo_regimes.png`",
        "- `l8_cell_summary.csv`",
        "- `l8_event_cell_metrics.csv`",
        "- `l8_top_delta_cells.csv`",
        "- `l8_geo_maps.npz`",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
