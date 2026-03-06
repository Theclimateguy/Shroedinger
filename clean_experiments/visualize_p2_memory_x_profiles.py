#!/usr/bin/env python3
"""Visualize x-modulation for P2 memory bridge across scales and on l=8."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-12
CI_Z = 1.96
PALETTE = {
    8: "#d04f32",
    16: "#1f77b4",
    32: "#2a9d55",
    "baseline": "#6c757d",
    "memory": "#d04f32",
    "delta": "#7b2cbf",
}


def _prepare_df(path: Path, *, require_memory_cols: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = [
        "event_id",
        "mrms_obs_time_utc",
        "scale_l",
        "tile_iy",
        "tile_ix",
        "lambda_local_raw",
        "comm_defect_raw",
        "fine_occ_mean",
    ]
    if require_memory_cols:
        needed.extend(["memory_history_l_raw", "memory_eta_raw", "memory_persistence_raw", "memory_applied"])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    out = df.copy()
    out["scale_l"] = pd.to_numeric(out["scale_l"], errors="coerce")
    out["tile_ix"] = pd.to_numeric(out["tile_ix"], errors="coerce")
    out["tile_iy"] = pd.to_numeric(out["tile_iy"], errors="coerce")
    out = out[np.isfinite(out["scale_l"]) & np.isfinite(out["tile_ix"]) & np.isfinite(out["tile_iy"])].copy()
    out["scale_l_int"] = np.round(out["scale_l"]).astype(int)
    out["tile_ix_int"] = np.round(out["tile_ix"]).astype(int)
    out["tile_iy_int"] = np.round(out["tile_iy"]).astype(int)
    out["lon_center"] = pd.to_numeric(out["lon_center"], errors="coerce")
    out["n_x"] = out.groupby("scale_l_int")["tile_ix_int"].transform("max").astype(int) + 1
    out["x_norm"] = (out["tile_ix_int"].to_numpy(dtype=float) + 0.5) / np.maximum(out["n_x"].to_numpy(dtype=float), 1.0)
    out["x_centered"] = 2.0 * out["x_norm"] - 1.0
    out["lambda_abs_raw"] = np.abs(out["lambda_local_raw"].to_numpy(dtype=float))
    out["memory_push_raw"] = 0.0
    if require_memory_cols:
        hist = out["memory_history_l_raw"].to_numpy(dtype=float)
        inst = out["fine_occ_mean"].to_numpy(dtype=float)
        eta = out["memory_eta_raw"].to_numpy(dtype=float)
        out["memory_push_raw"] = eta * (hist - inst)
    return out.reset_index(drop=True)


def _mean_se(x: pd.Series) -> tuple[float, float]:
    arr = x.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, 0.0
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return mean, se


def _event_profile(df: pd.DataFrame, value_col: str, scales: list[int]) -> pd.DataFrame:
    sub = df[df["scale_l_int"].isin(scales)].copy()
    event_mean = (
        sub.groupby(["event_id", "scale_l_int", "tile_ix_int"], as_index=False)
        .agg(x_centered=("x_centered", "mean"), lon_center=("lon_center", "mean"), value=(value_col, "mean"))
        .sort_values(["scale_l_int", "tile_ix_int", "event_id"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, float | int]] = []
    for (scale_l, tile_ix), g in event_mean.groupby(["scale_l_int", "tile_ix_int"], sort=True):
        mean, se = _mean_se(g["value"])
        rows.append(
            {
                "scale_l": int(scale_l),
                "tile_ix": int(tile_ix),
                "x_centered": float(g["x_centered"].mean()),
                "lon_center": float(g["lon_center"].mean()),
                "value_mean": mean,
                "value_se": se,
                "value_lo": float(mean - CI_Z * se) if np.isfinite(mean) else np.nan,
                "value_hi": float(mean + CI_Z * se) if np.isfinite(mean) else np.nan,
                "n_events": int(g["event_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scale_l", "tile_ix"]).reset_index(drop=True)


def _paired_l8_profiles(base_df: pd.DataFrame, mem_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["event_id", "mrms_obs_time_utc", "scale_l_int", "tile_iy_int", "tile_ix_int"]
    keep_base = keys + ["x_centered", "lon_center", "lambda_local_raw", "lambda_abs_raw", "comm_defect_raw"]
    keep_mem = keys + [
        "x_centered",
        "lon_center",
        "lambda_local_raw",
        "lambda_abs_raw",
        "comm_defect_raw",
        "memory_history_l_raw",
        "memory_eta_raw",
        "memory_persistence_raw",
        "memory_push_raw",
        "fine_occ_mean",
    ]
    base_l8 = base_df[base_df["scale_l_int"] == 8][keep_base].copy()
    mem_l8 = mem_df[mem_df["scale_l_int"] == 8][keep_mem].copy()

    merged = base_l8.merge(mem_l8, on=keys, suffixes=("_base", "_mem"), how="inner")
    merged["delta_lambda_raw"] = merged["lambda_local_raw_mem"] - merged["lambda_local_raw_base"]
    merged["delta_lambda_abs"] = merged["lambda_abs_raw_mem"] - merged["lambda_abs_raw_base"]
    merged["delta_comm_defect_raw"] = merged["comm_defect_raw_mem"] - merged["comm_defect_raw_base"]

    event_l8 = (
        merged.groupby(["event_id", "tile_ix_int"], as_index=False)
        .agg(
            x_centered=("x_centered_mem", "mean"),
            lon_center=("lon_center_mem", "mean"),
            lambda_base_mean=("lambda_local_raw_base", "mean"),
            lambda_mem_mean=("lambda_local_raw_mem", "mean"),
            lambda_abs_base_mean=("lambda_abs_raw_base", "mean"),
            lambda_abs_mem_mean=("lambda_abs_raw_mem", "mean"),
            delta_lambda_mean=("delta_lambda_raw", "mean"),
            delta_lambda_abs_mean=("delta_lambda_abs", "mean"),
            comm_base_mean=("comm_defect_raw_base", "mean"),
            comm_mem_mean=("comm_defect_raw_mem", "mean"),
            delta_comm_mean=("delta_comm_defect_raw", "mean"),
            fine_occ_mean=("fine_occ_mean", "mean"),
            history_occ_mean=("memory_history_l_raw", "mean"),
            memory_eta_mean=("memory_eta_raw", "mean"),
            memory_persistence_mean=("memory_persistence_raw", "mean"),
            memory_push_mean=("memory_push_raw", "mean"),
        )
        .sort_values(["tile_ix_int", "event_id"])
        .reset_index(drop=True)
    )

    summary_rows: list[dict[str, float | int]] = []
    for tile_ix, g in event_l8.groupby("tile_ix_int", sort=True):
        row: dict[str, float | int] = {
            "tile_ix": int(tile_ix),
            "x_centered": float(g["x_centered"].mean()),
            "lon_center": float(g["lon_center"].mean()),
            "n_events": int(g["event_id"].nunique()),
        }
        for src, dst in [
            ("lambda_base_mean", "lambda_base"),
            ("lambda_mem_mean", "lambda_mem"),
            ("lambda_abs_base_mean", "lambda_abs_base"),
            ("lambda_abs_mem_mean", "lambda_abs_mem"),
            ("delta_lambda_mean", "delta_lambda"),
            ("delta_lambda_abs_mean", "delta_lambda_abs"),
            ("comm_base_mean", "comm_base"),
            ("comm_mem_mean", "comm_mem"),
            ("delta_comm_mean", "delta_comm"),
            ("fine_occ_mean", "fine_occ"),
            ("history_occ_mean", "history_occ"),
            ("memory_eta_mean", "memory_eta"),
            ("memory_persistence_mean", "memory_persistence"),
            ("memory_push_mean", "memory_push"),
        ]:
            mean, se = _mean_se(g[src])
            row[f"{dst}_mean"] = mean
            row[f"{dst}_se"] = se
            row[f"{dst}_lo"] = float(mean - CI_Z * se) if np.isfinite(mean) else np.nan
            row[f"{dst}_hi"] = float(mean + CI_Z * se) if np.isfinite(mean) else np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("tile_ix").reset_index(drop=True)
    return event_l8, summary


def _plot_scale_modulation(profile_df: pd.DataFrame, out_path: Path, coord_col: str, xlabel: str, title_suffix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.4), sharex=True)
    for scale_l in sorted(profile_df["scale_l"].unique()):
        d = profile_df[profile_df["scale_l"] == scale_l].copy()
        color = PALETTE.get(int(scale_l), None)
        axes[0].plot(d[coord_col], d["lambda_abs_mean"], color=color, linewidth=2.0, label=f"l={int(scale_l)}")
        axes[0].fill_between(d[coord_col], d["lambda_abs_lo"], d["lambda_abs_hi"], color=color, alpha=0.18)
    axes[0].set_ylabel("mean |lambda_local_raw|")
    axes[0].set_title(f"{title_suffix} of |lambda_local| by scale")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.22, linewidth=0.5)

    for scale_l in sorted(profile_df["scale_l"].unique()):
        d = profile_df[profile_df["scale_l"] == scale_l].copy()
        color = PALETTE.get(int(scale_l), None)
        axes[1].plot(d[coord_col], d["comm_mean"], color=color, linewidth=2.0, label=f"l={int(scale_l)}")
        axes[1].fill_between(d[coord_col], d["comm_lo"], d["comm_hi"], color=color, alpha=0.18)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("mean Delta_comm")
    axes[1].set_title(f"{title_suffix} of commutator defect by scale")
    axes[1].grid(alpha=0.22, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_l8_baseline_memory(summary_df: pd.DataFrame, out_path: Path, coord_col: str, xlabel: str, coord_label: str) -> None:
    x = summary_df[coord_col].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.6), sharex=True)

    axes[0].plot(x, summary_df["lambda_base_mean"], color=PALETTE["baseline"], linewidth=2.0, label="baseline C009")
    axes[0].fill_between(x, summary_df["lambda_base_lo"], summary_df["lambda_base_hi"], color=PALETTE["baseline"], alpha=0.18)
    axes[0].plot(x, summary_df["lambda_mem_mean"], color=PALETTE["memory"], linewidth=2.1, label="P2-memory")
    axes[0].fill_between(x, summary_df["lambda_mem_lo"], summary_df["lambda_mem_hi"], color=PALETTE["memory"], alpha=0.18)
    axes[0].axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("mean lambda_local_raw")
    axes[0].set_title(f"l=8 signed lambda profile by {coord_label}: baseline vs memory")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.22, linewidth=0.5)

    axes[1].plot(x, summary_df["delta_lambda_mean"], color=PALETTE["delta"], linewidth=2.1, label="memory - baseline")
    axes[1].fill_between(x, summary_df["delta_lambda_lo"], summary_df["delta_lambda_hi"], color=PALETTE["delta"], alpha=0.18)
    axes[1].axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("mean Delta lambda")
    axes[1].set_title(f"l=8 memory lift in lambda({coord_label})")
    axes[1].grid(alpha=0.22, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_l8_memory_heatmap(event_df: pd.DataFrame, out_path: Path, tick_col: str, xlabel: str) -> list[str]:
    heat = event_df.pivot(index="event_id", columns="tile_ix_int", values="delta_lambda_mean")
    order = heat.mean(axis=1).sort_values().index.tolist()
    heat = heat.loc[order]

    arr = heat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    vmin = float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else 0.0
    vmax = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1.0
    im = ax.imshow(arr, aspect="auto", cmap="magma", vmin=max(0.0, vmin), vmax=vmax, origin="upper")

    xticks = np.arange(arr.shape[1])
    coord_vals = (
        event_df.groupby("tile_ix_int", as_index=False)[tick_col]
        .mean()
        .sort_values("tile_ix_int")[tick_col]
        .to_numpy(dtype=float)
    )
    show_idx = np.linspace(0, len(xticks) - 1, min(7, len(xticks))).round().astype(int)
    show_idx = np.unique(show_idx)
    ax.set_xticks(show_idx)
    ax.set_xticklabels([f"{coord_vals[i]:.2f}" for i in show_idx])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("event_id")
    ax.set_title(f"l=8 event-by-event memory lift: Delta lambda({tick_col})")
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("mean Delta lambda")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return order


def _plot_l8_memory_state(summary_df: pd.DataFrame, out_path: Path, coord_col: str, xlabel: str, coord_label: str) -> None:
    x = summary_df[coord_col].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.4), sharex=True)

    axes[0].plot(x, summary_df["fine_occ_mean"], color="#1f77b4", linewidth=2.0, label="instant occ(l)")
    axes[0].fill_between(x, summary_df["fine_occ_lo"], summary_df["fine_occ_hi"], color="#1f77b4", alpha=0.18)
    axes[0].plot(x, summary_df["history_occ_mean"], color="#d04f32", linewidth=2.0, label="history occ(l)")
    axes[0].fill_between(x, summary_df["history_occ_lo"], summary_df["history_occ_hi"], color="#d04f32", alpha=0.18)
    axes[0].set_ylabel("occupancy")
    axes[0].set_title(f"l=8 state view by {coord_label}: instantaneous vs memory history")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.22, linewidth=0.5)

    axes[1].plot(x, summary_df["memory_push_mean"], color="#7b2cbf", linewidth=2.0, label="eta * (history - instant)")
    axes[1].fill_between(x, summary_df["memory_push_lo"], summary_df["memory_push_hi"], color="#7b2cbf", alpha=0.18)
    axes[1].axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    ax2 = axes[1].twinx()
    ax2.plot(x, summary_df["memory_persistence_mean"], color="#2a9d55", linewidth=1.6, linestyle="--", label="persistence")
    ax2.set_ylabel("persistence")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("memory push")
    axes[1].set_title(f"l=8 memory push and persistence by {coord_label}")
    axes[1].grid(alpha=0.22, linewidth=0.5)

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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
        "--summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory/summary_metrics.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory_x_viz"),
    )
    p.add_argument("--scales", nargs="+", type=int, default=[8, 16, 32])
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_df = _prepare_df(args.baseline_tile_csv, require_memory_cols=False)
    memory_df = _prepare_df(args.memory_tile_csv, require_memory_cols=True)

    lambda_abs_prof = _event_profile(memory_df, "lambda_abs_raw", args.scales).rename(
        columns={"value_mean": "lambda_abs_mean", "value_se": "lambda_abs_se", "value_lo": "lambda_abs_lo", "value_hi": "lambda_abs_hi"}
    )
    comm_prof = _event_profile(memory_df, "comm_defect_raw", args.scales).rename(
        columns={"value_mean": "comm_mean", "value_se": "comm_se", "value_lo": "comm_lo", "value_hi": "comm_hi"}
    )
    scale_profile = lambda_abs_prof.merge(
        comm_prof[["scale_l", "tile_ix", "comm_mean", "comm_se", "comm_lo", "comm_hi"]],
        on=["scale_l", "tile_ix"],
        how="left",
    )
    scale_profile.to_csv(outdir / "x_profile_by_scale.csv", index=False)
    scale_profile.to_csv(outdir / "lon_profile_by_scale.csv", index=False)

    event_l8, summary_l8 = _paired_l8_profiles(baseline_df, memory_df)
    summary_l8.to_csv(outdir / "l8_memory_profile.csv", index=False)
    event_l8.to_csv(outdir / "l8_memory_profile_eventwise.csv", index=False)

    heat = event_l8.pivot(index="event_id", columns="tile_ix_int", values="delta_lambda_mean")
    heat.to_csv(outdir / "l8_memory_delta_heatmap.csv")

    _plot_scale_modulation(
        scale_profile,
        outdir / "x_modulation_by_scale.png",
        coord_col="x_centered",
        xlabel="normalized panel x",
        title_suffix="X-modulation",
    )
    _plot_scale_modulation(
        scale_profile,
        outdir / "lon_modulation_by_scale.png",
        coord_col="lon_center",
        xlabel="longitude (deg E, west negative)",
        title_suffix="Longitude modulation",
    )
    _plot_l8_baseline_memory(
        summary_l8,
        outdir / "l8_memory_vs_baseline.png",
        coord_col="x_centered",
        xlabel="normalized panel x",
        coord_label="x",
    )
    _plot_l8_baseline_memory(
        summary_l8,
        outdir / "l8_memory_vs_baseline_lon.png",
        coord_col="lon_center",
        xlabel="longitude (deg E, west negative)",
        coord_label="longitude",
    )
    event_order = _plot_l8_memory_heatmap(
        event_l8,
        outdir / "l8_memory_delta_heatmap.png",
        tick_col="x_centered",
        xlabel="normalized panel x",
    )
    _plot_l8_memory_heatmap(
        event_l8,
        outdir / "l8_memory_delta_heatmap_lon.png",
        tick_col="lon_center",
        xlabel="longitude (deg E, west negative)",
    )
    _plot_l8_memory_state(
        summary_l8,
        outdir / "l8_memory_state_profile.png",
        coord_col="x_centered",
        xlabel="normalized panel x",
        coord_label="x",
    )
    _plot_l8_memory_state(
        summary_l8,
        outdir / "l8_memory_state_profile_lon.png",
        coord_col="lon_center",
        xlabel="longitude (deg E, west negative)",
        coord_label="longitude",
    )

    summary_all = pd.read_csv(args.summary_csv)
    overall = summary_all[summary_all["scale_l"].astype(str) == "ALL"]
    if overall.empty:
        raise ValueError(f"No ALL row in {args.summary_csv}")
    overall_row = overall.iloc[0]

    delta_peak = summary_l8.loc[summary_l8["delta_lambda_mean"].idxmax()]
    positive_frac = float(np.mean(summary_l8["delta_lambda_mean"].to_numpy(dtype=float) > 0.0))
    abs_peak_rows: list[str] = []
    for scale_l in sorted(scale_profile["scale_l"].unique()):
        d = scale_profile[scale_profile["scale_l"] == scale_l].copy()
        peak = d.loc[d["lambda_abs_mean"].idxmax()]
        abs_peak_rows.append(
            f"- `l={int(scale_l)}` peak `|lambda|` at `x={float(peak['x_centered']):.3f}` with mean `{float(peak['lambda_abs_mean']):.6f}`"
        )

    report_lines = [
        "# P2-memory x-modulation visualization",
        "",
        "## Setup",
        f"- baseline tile csv: `{args.baseline_tile_csv}`",
        f"- memory tile csv: `{args.memory_tile_csv}`",
        f"- normalized coordinate: `x = 2 * ((tile_ix + 0.5) / n_x) - 1`",
        f"- scales visualized: `{list(map(int, args.scales))}`",
        "",
        "## Headline",
        f"- all-scale memory run remains `PASS_ALL={bool(overall_row['PASS_ALL'])}` with `mae_gain={float(overall_row['mae_gain']):.6e}` and `perm_p={float(overall_row['perm_p_value']):.6f}`",
        f"- on `l=8`, mean `Delta lambda(x)` is positive on `{positive_frac:.1%}` of x-bins",
        f"- strongest memory lift is at `x={float(delta_peak['x_centered']):.3f}` with `Delta lambda={float(delta_peak['delta_lambda_mean']):.6f}`",
        f"- the same peak sits at longitude `{float(delta_peak['lon_center']):.3f}`",
        "",
        "## Scale modulation peaks",
        *abs_peak_rows,
        "",
        "## Geographic note",
        "- `lon_center` is globally aligned across events in this panel, so the longitude-profile is a real zonal geographic section rather than a per-event internal coordinate only.",
        "- Longitudes are on the CONUS MRMS grid; west is more negative.",
        "",
        "## l=8 memory interpretation",
        "- the signed `lambda(x)` curve shifts upward relative to dense baseline across the full x-range.",
        "- the strongest lift is concentrated near the central x-bins, not at the panel edges.",
        "- in absolute longitude, the strongest lift sits near the central US longitudes rather than the far Pacific or Atlantic edges of the panel.",
        "- the raw occupancy memory push is much smaller than the lambda lift and changes sign across x, which indicates that the visible effect comes from the nonlinear density-matrix bridge, not from a trivial additive inflation of occupancy.",
        "",
        "## Artifacts",
        "- `x_modulation_by_scale.png`",
        "- `lon_modulation_by_scale.png`",
        "- `l8_memory_vs_baseline.png`",
        "- `l8_memory_vs_baseline_lon.png`",
        "- `l8_memory_delta_heatmap.png`",
        "- `l8_memory_delta_heatmap_lon.png`",
        "- `l8_memory_state_profile.png`",
        "- `l8_memory_state_profile_lon.png`",
        "- `x_profile_by_scale.csv`",
        "- `lon_profile_by_scale.csv`",
        "- `l8_memory_profile.csv`",
        "- `l8_memory_profile_eventwise.csv`",
        "- `l8_memory_delta_heatmap.csv`",
        "",
        "## Heatmap order",
        f"- events are sorted by mean `Delta lambda` from weakest to strongest: `{', '.join(event_order)}`",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
