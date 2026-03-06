#!/usr/bin/env python3
"""Compare ABI-only vs ABI+GLM structural signal on frozen M-realpilot modeling data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_M_realpilot_v1_frozen import _evaluate_logo, _permutation_test


BASELINE_COLS = ["base_prev_p95", "base_prev_area_gt5", "hour_sin", "hour_cos"]
STRUCT_ABI_ONLY = ["abi_cold_frac_235", "abi_grad_mean"]
STRUCT_ABI_GLM = ["abi_cold_frac_235", "abi_grad_mean", "glm_flash_count_log", "convective_coupling_index"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modeling-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--target-col", default="target_next_mrms_p95")
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--n-perm", type=int, default=499)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _check_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def _run_variant(
    df: pd.DataFrame,
    target_col: str,
    structural_cols: list[str],
    ridge_alpha: float,
    n_perm: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, oof_df, fold_df = _evaluate_logo(
        df,
        target_col=target_col,
        baseline_cols=BASELINE_COLS,
        structural_cols=structural_cols,
        alpha=ridge_alpha,
    )

    _, perm_time = _permutation_test(
        df,
        target_col=target_col,
        baseline_cols=BASELINE_COLS,
        structural_cols=structural_cols,
        alpha=ridge_alpha,
        n_perm=n_perm,
        seed=seed + 11,
        mode="time_shuffle",
    )
    _, perm_event = _permutation_test(
        df,
        target_col=target_col,
        baseline_cols=BASELINE_COLS,
        structural_cols=structural_cols,
        alpha=ridge_alpha,
        n_perm=n_perm,
        seed=seed + 23,
        mode="event_shuffle",
    )

    real_gain = float(summary["mean_mae_gain"])
    p_time = float((1.0 + (perm_time["mean_mae_gain"] >= real_gain).sum()) / (1.0 + len(perm_time)))
    p_event = float((1.0 + (perm_event["mean_mae_gain"] >= real_gain).sum()) / (1.0 + len(perm_event)))

    row = {
        "target": target_col,
        "n_events": float(df["event_id"].nunique()),
        "n_model_samples": float(len(df)),
        "mae_baseline": float(summary["mae_baseline"]),
        "mae_full": float(summary["mae_full"]),
        "mean_mae_gain": real_gain,
        "min_event_gain": float(summary["min_event_gain"]),
        "event_positive_frac": float(summary["event_positive_frac"]),
        "perm_p_time_shuffle": p_time,
        "perm_p_event_shuffle": p_event,
        "ridge_alpha": float(ridge_alpha),
        "n_perm": float(n_perm),
    }
    return row, oof_df, fold_df, perm_time, perm_event


def _plot_gain_compare(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df), dtype=float)
    gains = df["mean_mae_gain"].to_numpy(dtype=float)
    labels = df["variant"].tolist()
    colors = ["#377eb8" if "ABI+GLM" in v else "#4daf4a" for v in labels]
    ax.bar(x, gains, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Mean MAE gain")
    ax.set_title("Satellite component stability: ABI-only vs ABI+GLM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.modeling_csv)

    _check_cols(df, BASELINE_COLS + STRUCT_ABI_ONLY + STRUCT_ABI_GLM + [args.target_col, "event_id", "step_idx"], "modeling dataset")

    variants = [
        ("ABI-only", STRUCT_ABI_ONLY, 1001),
        ("ABI+GLM", STRUCT_ABI_GLM, 2001),
    ]

    rows = []
    for name, cols, seed_offset in variants:
        print(f"[variant] {name}: structural_cols={cols}", flush=True)
        row, oof, fold, perm_time, perm_event = _run_variant(
            df=df,
            target_col=args.target_col,
            structural_cols=cols,
            ridge_alpha=args.ridge_alpha,
            n_perm=args.n_perm,
            seed=args.seed + seed_offset,
        )
        row["variant"] = name
        row["structural_cols"] = ";".join(cols)
        rows.append(row)

        tag = name.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")
        oof.to_csv(outdir / f"oof_{tag}.csv", index=False)
        fold.to_csv(outdir / f"fold_{tag}.csv", index=False)
        perm_time.to_csv(outdir / f"perm_time_{tag}.csv", index=False)
        perm_event.to_csv(outdir / f"perm_event_{tag}.csv", index=False)

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "variant",
            "target",
            "n_events",
            "n_model_samples",
            "mae_baseline",
            "mae_full",
            "mean_mae_gain",
            "min_event_gain",
            "event_positive_frac",
            "perm_p_time_shuffle",
            "perm_p_event_shuffle",
            "ridge_alpha",
            "n_perm",
            "structural_cols",
        ]
    ]
    summary.to_csv(outdir / "satellite_component_summary.csv", index=False)

    abi_only = summary.loc[summary["variant"] == "ABI-only", "mean_mae_gain"].iloc[0]
    abi_glm = summary.loc[summary["variant"] == "ABI+GLM", "mean_mae_gain"].iloc[0]
    delta = float(abi_glm - abi_only)

    _plot_gain_compare(summary, outdir / "plot_satellite_component_gain.png")

    report = "\n".join(
        [
            "# Satellite Component Stability (Frozen M-realpilot)",
            "",
            f"- modeling dataset: `{args.modeling_csv}`",
            f"- target: `{args.target_col}`",
            f"- ridge_alpha: {args.ridge_alpha}",
            f"- permutations per variant: {args.n_perm}",
            "",
            "## Headline",
            f"- ABI-only mean gain: {float(abi_only):.6f}",
            f"- ABI+GLM mean gain: {float(abi_glm):.6f}",
            f"- delta (ABI+GLM - ABI-only): {delta:.6f}",
            "",
            "## Notes",
            "- Feature extraction and thresholds are inherited from frozen v1.",
            "- Comparison changes only the structural column subset inside the same CV/permutation protocol.",
        ]
    )
    (outdir / "report.md").write_text(report, encoding="utf-8")

    print("Satellite component stability done.", flush=True)
    print(f"Output: {outdir}", flush=True)
    print(f"ABI-only gain={float(abi_only):.6f} ABI+GLM gain={float(abi_glm):.6f} delta={delta:.6f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
