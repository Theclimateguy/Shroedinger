#!/usr/bin/env python3
"""Build applicability map table for frozen M-realpilot runs.

Axes requested:
- season
- region
- ABI-only metrics
- ABI+GLM metrics
- fraction of active windows
- mean GLM sparsity
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    outdir: Path
    events_csv: Path


def _month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def _season_label_from_events(events_df: pd.DataFrame) -> str:
    if "start_utc" not in events_df.columns or len(events_df) == 0:
        return "unknown"
    ts = pd.to_datetime(events_df["start_utc"], utc=True, errors="coerce")
    months = sorted({int(t.month) for t in ts.dropna()})
    if not months:
        return "unknown"
    seasons = sorted({_month_to_season(m) for m in months})
    return "+".join(seasons)


def _region_label_from_events(events_df: pd.DataFrame) -> str:
    if "region_tag" not in events_df.columns or len(events_df) == 0:
        return "unknown"
    vals = sorted({str(x) for x in events_df["region_tag"].dropna().astype(str).tolist() if str(x)})
    return "+".join(vals) if vals else "unknown"


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _active_frac(model_df: pd.DataFrame, summary_df: pd.DataFrame | None) -> float:
    if len(model_df) == 0 or "base_prev_p95" not in model_df.columns:
        return float("nan")
    if summary_df is not None and "active_threshold_p95" in summary_df.columns:
        thr = float(summary_df.iloc[0]["active_threshold_p95"])
    else:
        thr = float(np.nanquantile(model_df["base_prev_p95"].to_numpy(dtype=float), 0.67))
    arr = model_df["base_prev_p95"].to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr >= thr))


def _glm_sparsity(feature_df: pd.DataFrame) -> float:
    if len(feature_df) == 0 or "glm_flash_count" not in feature_df.columns:
        return float("nan")
    arr = feature_df["glm_flash_count"].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr <= 0.0))


def _extract_component_gain(stab_df: pd.DataFrame | None, variant: str) -> float:
    if stab_df is None or len(stab_df) == 0:
        return float("nan")
    sub = stab_df[stab_df["variant"].astype(str) == variant]
    if len(sub) == 0:
        return float("nan")
    return float(sub.iloc[0]["mean_mae_gain"])


def _extract_component_p(stab_df: pd.DataFrame | None, variant: str, col: str) -> float:
    if stab_df is None or len(stab_df) == 0:
        return float("nan")
    sub = stab_df[stab_df["variant"].astype(str) == variant]
    if len(sub) == 0:
        return float("nan")
    return float(sub.iloc[0][col])


def _build_row(spec: RunSpec) -> dict[str, object]:
    summary_path = spec.outdir / "summary_metrics.csv"
    model_path = spec.outdir / "modeling_dataset.csv"
    feat_path = spec.outdir / "feature_dataset.csv"
    stab_path = spec.outdir / "satellite_component_stability" / "satellite_component_summary.csv"

    events_df = _safe_read_csv(spec.events_csv)
    if events_df is None:
        events_df = pd.DataFrame()

    summary_df = _safe_read_csv(summary_path)
    model_df = _safe_read_csv(model_path)
    feat_df = _safe_read_csv(feat_path)
    stab_df = _safe_read_csv(stab_path)

    has_main = summary_df is not None and len(summary_df) > 0

    abi_only_gain = _extract_component_gain(stab_df, "ABI-only")
    abi_glm_gain = _extract_component_gain(stab_df, "ABI+GLM")

    row = {
        "run_id": spec.run_id,
        "label": spec.label,
        "season": _season_label_from_events(events_df),
        "region": _region_label_from_events(events_df),
        "n_events_manifest": int(len(events_df)) if len(events_df) > 0 else np.nan,
        "n_events_model": float(summary_df.iloc[0]["n_events"]) if has_main and "n_events" in summary_df.columns else np.nan,
        "active_window_frac": _active_frac(model_df, summary_df) if model_df is not None else np.nan,
        "mean_glm_sparsity": _glm_sparsity(feat_df) if feat_df is not None else np.nan,
        "abi_only_gain": abi_only_gain,
        "abi_only_p_time": _extract_component_p(stab_df, "ABI-only", "perm_p_time_shuffle"),
        "abi_only_p_event": _extract_component_p(stab_df, "ABI-only", "perm_p_event_shuffle"),
        "abi_glm_gain": abi_glm_gain,
        "abi_glm_p_time": _extract_component_p(stab_df, "ABI+GLM", "perm_p_time_shuffle"),
        "abi_glm_p_event": _extract_component_p(stab_df, "ABI+GLM", "perm_p_event_shuffle"),
        "delta_gain_glm_minus_abi": (
            float(abi_glm_gain - abi_only_gain)
            if np.isfinite(abi_glm_gain) and np.isfinite(abi_only_gain)
            else np.nan
        ),
        "main_mean_gain": float(summary_df.iloc[0]["mean_mae_gain"]) if has_main and "mean_mae_gain" in summary_df.columns else np.nan,
        "main_pass_all": bool(summary_df.iloc[0]["PASS_ALL"]) if has_main and "PASS_ALL" in summary_df.columns else np.nan,
        "main_event_positive_frac": float(summary_df.iloc[0]["event_positive_frac"]) if has_main and "event_positive_frac" in summary_df.columns else np.nan,
        "main_active_minus_calm": float(summary_df.iloc[0]["active_minus_calm"]) if has_main and "active_minus_calm" in summary_df.columns else np.nan,
    }
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_realpilot_applicability_map"),
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    specs = [
        RunSpec(
            run_id="v1_expanded_positive",
            label="Frozen expanded positive",
            outdir=Path("clean_experiments/results/experiment_M_realpilot_v1_expanded_positive"),
            events_csv=Path("clean_experiments/pilot_events_real_2024_us_convective_expanded_v1.csv"),
        ),
        RunSpec(
            run_id="v1_independent_seasonal_2024",
            label="Independent seasonal extension",
            outdir=Path("clean_experiments/results/experiment_M_realpilot_v1_frozen_independent_seasonal_2024"),
            events_csv=Path("clean_experiments/pilot_events_realpilot_v1_independent_seasonal_2024.csv"),
        ),
        RunSpec(
            run_id="v1_independent_geographic_southwest_2024",
            label="Independent geographic extension (Southwest)",
            outdir=Path("clean_experiments/results/experiment_M_realpilot_v1_frozen_independent_geographic_southwest_2024"),
            events_csv=Path("clean_experiments/pilot_events_realpilot_v1_independent_geographic_southwest_2024.csv"),
        ),
    ]

    rows = []
    for spec in specs:
        rows.append(_build_row(spec))

    out = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "applicability_map.csv"
    out.to_csv(out_csv, index=False)

    md_lines = [
        "# M-realpilot Applicability Map",
        "",
        "Rows summarize frozen runs across season/region with ABI-only vs ABI+GLM component checks.",
        "",
        "| run_id | season | region | main_mean_gain | main_pass_all | abi_only_gain | abi_glm_gain | delta_glm_minus_abi | active_window_frac | active_minus_calm | mean_glm_sparsity |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in out.iterrows():
        md_lines.append(
            "| {run_id} | {season} | {region} | {main_mean_gain:.6f} | {main_pass_all} | {abi_only_gain:.6f} | {abi_glm_gain:.6f} | {delta:.6f} | {active:.3f} | {active_minus_calm:.6f} | {sparse:.3f} |".format(
                run_id=r["run_id"],
                season=r["season"],
                region=r["region"],
                main_mean_gain=float(r["main_mean_gain"]) if np.isfinite(r["main_mean_gain"]) else float("nan"),
                main_pass_all=r["main_pass_all"],
                abi_only_gain=float(r["abi_only_gain"]) if np.isfinite(r["abi_only_gain"]) else float("nan"),
                abi_glm_gain=float(r["abi_glm_gain"]) if np.isfinite(r["abi_glm_gain"]) else float("nan"),
                delta=float(r["delta_gain_glm_minus_abi"]) if np.isfinite(r["delta_gain_glm_minus_abi"]) else float("nan"),
                active=float(r["active_window_frac"]) if np.isfinite(r["active_window_frac"]) else float("nan"),
                active_minus_calm=float(r["main_active_minus_calm"]) if np.isfinite(r["main_active_minus_calm"]) else float("nan"),
                sparse=float(r["mean_glm_sparsity"]) if np.isfinite(r["mean_glm_sparsity"]) else float("nan"),
            )
        )

    (args.outdir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Applicability map saved: {out_csv}")
    print(f"Report: {args.outdir / 'report.md'}")


if __name__ == "__main__":
    run(parse_args())
