#!/usr/bin/env python3
"""Spatial visualization for M-realpilot over coastline background.

Builds spatial maps from expanded MRMS+GOES pilot panel:
- MRMS active fraction
- GLM flash density
- simple convective coupling proxy
- event-level MAE gain markers from frozen M-realpilot run
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr


COAST_GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"


def _to_lon180(lon: np.ndarray) -> np.ndarray:
    x = np.asarray(lon, dtype=float)
    x = np.where(x > 180.0, x - 360.0, x)
    return x


def _read_mrms_grib(path_gz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with gzip.open(path_gz, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        ds = xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        try:
            if len(ds.data_vars) == 0:
                raise ValueError(f"No data vars in MRMS file: {path_gz}")
            var = next(iter(ds.data_vars))
            arr = np.asarray(ds[var].values, dtype=np.float32)
            lat = np.asarray(ds["latitude"].values, dtype=float)
            lon = _to_lon180(np.asarray(ds["longitude"].values, dtype=float))
        finally:
            ds.close()
        return arr, lat, lon
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_coastline(cache_path: Path) -> list[np.ndarray]:
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(COAST_GEOJSON_URL, timeout=60)
        r.raise_for_status()
        cache_path.write_text(r.text, encoding="utf-8")
        data = r.json()

    lines: list[np.ndarray] = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if gtype == "LineString":
            arr = np.asarray(coords, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                arr[:, 0] = _to_lon180(arr[:, 0])
                lines.append(arr[:, :2])
        elif gtype == "MultiLineString":
            for c in coords:
                arr = np.asarray(c, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    arr[:, 0] = _to_lon180(arr[:, 0])
                    lines.append(arr[:, :2])
    return lines


def _plot_coast(ax: plt.Axes, coast_lines: list[np.ndarray], lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    for arr in coast_lines:
        x = arr[:, 0]
        y = arr[:, 1]
        keep = (x >= lon_min - 2.0) & (x <= lon_max + 2.0) & (y >= lat_min - 2.0) & (y <= lat_max + 2.0)
        if int(keep.sum()) < 2:
            continue
        ax.plot(x[keep], y[keep], color="black", linewidth=0.45, alpha=0.7)


def _plot_field(
    ax: plt.Axes,
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str,
    coast_lines: list[np.ndarray],
) -> None:
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")
    q02 = float(np.nanquantile(field, 0.02))
    q98 = float(np.nanquantile(field, 0.98))
    if not np.isfinite(q02) or not np.isfinite(q98) or q98 <= q02:
        q02 = float(np.nanmin(field))
        q98 = float(np.nanmax(field))
    im = ax.pcolormesh(lon2d, lat2d, field, shading="auto", cmap=cmap, vmin=q02, vmax=q98)
    _plot_coast(ax, coast_lines, float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat)))
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lon)), float(np.max(lon)))
    ax.set_ylim(float(np.min(lat)), float(np.max(lat)))
    cb = plt.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(cbar_label)


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel_csv).sort_values(["event_id", "mrms_obs_time_utc"]).reset_index(drop=True)
    fold = pd.read_csv(args.fold_metrics_csv)
    feat = pd.read_csv(args.feature_dataset_csv) if args.feature_dataset_csv.exists() else pd.DataFrame()

    if len(panel) == 0:
        raise ValueError("Panel is empty.")

    print("[spatial] load coastline", flush=True)
    coast_lines = _download_coastline(outdir / "ne_110m_coastline.geojson")

    print("[spatial] initialize MRMS grid", flush=True)
    arr0, lat0, lon0 = _read_mrms_grib(Path(panel.iloc[0]["mrms_local_path"]))
    stride = max(1, int(args.mrms_stride))
    iy = np.arange(0, arr0.shape[0], stride, dtype=int)
    ix = np.arange(0, arr0.shape[1], stride, dtype=int)
    lat_ds = lat0[iy]
    lon_ds = lon0[ix]

    ny, nx = len(lat_ds), len(lon_ds)
    sum_rate = np.zeros((ny, nx), dtype=np.float64)
    count_valid = np.zeros((ny, nx), dtype=np.int32)
    count_active = np.zeros((ny, nx), dtype=np.int32)
    glm_counts = np.zeros((ny, nx), dtype=np.float64)

    # Event centroid accumulator (for gain markers).
    ev_w: dict[str, float] = {}
    ev_latw: dict[str, float] = {}
    ev_lonw: dict[str, float] = {}

    lat_edges = np.linspace(float(lat_ds.min()), float(lat_ds.max()), ny + 1)
    lon_edges = np.linspace(float(lon_ds.min()), float(lon_ds.max()), nx + 1)

    n_rows = len(panel)
    for i, r in panel.iterrows():
        if (i + 1) % 5 == 0 or i == 0 or i + 1 == n_rows:
            print(f"[spatial] processing row {i+1}/{n_rows}", flush=True)

        event_id = str(r["event_id"])

        arr, lat, lon = _read_mrms_grib(Path(r["mrms_local_path"]))
        arr = arr[np.ix_(iy, ix)]
        valid = np.isfinite(arr) & (arr >= 0.0)
        sum_rate += np.where(valid, arr, 0.0)
        count_valid += valid.astype(np.int32)
        active = valid & (arr >= args.active_threshold)
        count_active += active.astype(np.int32)

        if int(active.sum()) > 0:
            w = arr[active].astype(np.float64)
            lat2d, lon2d = np.meshgrid(lat_ds, lon_ds, indexing="ij")
            la = lat2d[active]
            lo = lon2d[active]
            ww = float(np.sum(w))
            if ww > 0.0:
                ev_w[event_id] = ev_w.get(event_id, 0.0) + ww
                ev_latw[event_id] = ev_latw.get(event_id, 0.0) + float(np.sum(w * la))
                ev_lonw[event_id] = ev_lonw.get(event_id, 0.0) + float(np.sum(w * lo))

        ds_glm = xr.open_dataset(r["glm_local_path"], engine="netcdf4")
        try:
            if "flash_lat" in ds_glm and "flash_lon" in ds_glm:
                gla = np.asarray(ds_glm["flash_lat"].values, dtype=float)
                glo = _to_lon180(np.asarray(ds_glm["flash_lon"].values, dtype=float))
                good = (
                    np.isfinite(gla)
                    & np.isfinite(glo)
                    & (gla >= float(lat_ds.min()))
                    & (gla <= float(lat_ds.max()))
                    & (glo >= float(lon_ds.min()))
                    & (glo <= float(lon_ds.max()))
                )
                if int(good.sum()) > 0:
                    h, _, _ = np.histogram2d(gla[good], glo[good], bins=[lat_edges, lon_edges])
                    glm_counts += h
        finally:
            ds_glm.close()

    mean_rate = np.full((ny, nx), np.nan, dtype=np.float64)
    active_frac = np.full((ny, nx), np.nan, dtype=np.float64)
    good_cells = count_valid > 0
    mean_rate[good_cells] = sum_rate[good_cells] / count_valid[good_cells]
    active_frac[good_cells] = count_active[good_cells] / count_valid[good_cells]
    glm_rate = glm_counts / max(1, len(panel))
    coupling = active_frac * np.log1p(glm_rate)

    ev_rows = []
    for ev, ww in ev_w.items():
        if ww <= 0.0:
            continue
        ev_rows.append(
            {
                "event_id": ev,
                "centroid_lat": ev_latw[ev] / ww,
                "centroid_lon": ev_lonw[ev] / ww,
                "weight_sum": ww,
            }
        )
    event_pos = pd.DataFrame(ev_rows)
    if len(event_pos) == 0:
        raise ValueError("No event centroids computed from active MRMS cells.")

    event_gain = fold[["event_id", "mae_gain"]].copy()
    event_map = event_pos.merge(event_gain, on="event_id", how="left")
    if len(feat) > 0:
        ag = feat.groupby("event_id", as_index=False)[["convective_coupling_index", "abi_cold_frac_235", "glm_flash_count_log"]].mean()
        event_map = event_map.merge(ag, on="event_id", how="left")
    event_map.to_csv(outdir / "event_spatial_metrics.csv", index=False)

    # Four-panel figure.
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    _plot_field(
        axs[0, 0],
        field=active_frac,
        lon=lon_ds,
        lat=lat_ds,
        title=f"MRMS active fraction (rate >= {args.active_threshold:g})",
        cbar_label="fraction",
        cmap="YlGnBu",
        coast_lines=coast_lines,
    )
    _plot_field(
        axs[0, 1],
        field=np.log1p(glm_rate),
        lon=lon_ds,
        lat=lat_ds,
        title="GLM flash density (log1p per frame)",
        cbar_label="log1p(density)",
        cmap="magma",
        coast_lines=coast_lines,
    )
    _plot_field(
        axs[1, 0],
        field=coupling,
        lon=lon_ds,
        lat=lat_ds,
        title="Convective coupling proxy (active_frac * log1p(glm_density))",
        cbar_label="coupling",
        cmap="viridis",
        coast_lines=coast_lines,
    )

    # Event-level gain markers on top of active-fraction background.
    ax = axs[1, 1]
    lon2d, lat2d = np.meshgrid(lon_ds, lat_ds, indexing="xy")
    bg = ax.pcolormesh(lon2d, lat2d, active_frac, shading="auto", cmap="Greys", alpha=0.35)
    _plot_coast(ax, coast_lines, float(np.min(lon_ds)), float(np.max(lon_ds)), float(np.min(lat_ds)), float(np.max(lat_ds)))
    sc = ax.scatter(
        event_map["centroid_lon"],
        event_map["centroid_lat"],
        c=event_map["mae_gain"],
        s=80,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )
    for _, rr in event_map.iterrows():
        ax.text(rr["centroid_lon"] + 0.25, rr["centroid_lat"] + 0.15, str(rr["event_id"]), fontsize=7, alpha=0.85)
    ax.set_title("Event-level MAE gain (frozen M-realpilot)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lon_ds)), float(np.max(lon_ds)))
    ax.set_ylim(float(np.min(lat_ds)), float(np.max(lat_ds)))
    plt.colorbar(sc, ax=ax, shrink=0.9, label="mae_gain")
    plt.colorbar(bg, ax=ax, shrink=0.9, label="MRMS active fraction (bg)")

    fig.suptitle("M-realpilot spatial diagnostics (expanded v1, coastline overlay)", fontsize=14)
    fig.savefig(outdir / "plot_M_realpilot_spatial_overview_coastline.png", dpi=180)
    plt.close(fig)

    report = "\n".join(
        [
            "# M-realpilot spatial visualization (coastline overlay)",
            "",
            f"- panel: `{args.panel_csv}`",
            f"- fold metrics: `{args.fold_metrics_csv}`",
            f"- feature dataset: `{args.feature_dataset_csv}`",
            f"- rows processed: {len(panel)}",
            f"- events: {panel['event_id'].nunique()}",
            f"- MRMS downsample stride: {stride}",
            f"- active threshold: {args.active_threshold}",
            "",
            "Outputs:",
            "- `plot_M_realpilot_spatial_overview_coastline.png`",
            "- `event_spatial_metrics.csv`",
            "- `ne_110m_coastline.geojson` (cached)",
        ]
    )
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(f"[spatial] done: {outdir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv"),
    )
    p.add_argument(
        "--fold-metrics-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_realpilot_v1_frozen_expanded/fold_metrics.csv"),
    )
    p.add_argument(
        "--feature-dataset-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_realpilot_v1_frozen_expanded/feature_dataset.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_realpilot_v1_frozen_expanded_spatial"),
    )
    p.add_argument("--mrms-stride", type=int, default=20)
    p.add_argument("--active-threshold", type=float, default=5.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
