#!/usr/bin/env python3
"""Experiment H robust sweep: holographic truncation under dimension growth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_H_holographic import run_experiment
except ImportError:
    from experiment_H_holographic import run_experiment


def _sample_case(rng: np.random.Generator) -> dict[str, float | int | str]:
    growth = float(rng.uniform(0.70, 1.00))
    holo_growth = float(rng.uniform(0.35, min(0.80, growth - 0.05)))
    return {
        "k_layers": int(rng.choice([12, 16, 20, 24])),
        "mu_min": 0.0,
        "mu_max": float(rng.choice([4.5, 5.0, 6.0, 7.0])),
        "d0": int(rng.choice([2, 4, 6])),
        "growth": growth,
        "holo_growth": holo_growth,
        "d_max": int(rng.choice([128, 192, 256])),
        "mix_pure": float(rng.uniform(0.45, 0.80)),
        "weight_mode": str(rng.choice(["uniform", "inv_dim"])),
        "resolution_scan": "8,12,16,20,24",
    }


def run_robustness(
    outdir: Path,
    *,
    n_cases: int = 36,
    seed: int = 20260222,
    stability_gain_min: float = 1.0,
    trace_tol: float = 1e-11,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []
    cases_dir = outdir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    for case_id in range(n_cases):
        params = _sample_case(rng)
        case_seed = seed + 10000 + case_id
        case_dir = cases_dir / f"case_{case_id:03d}"

        _, _, _, summary_df = run_experiment(
            outdir=case_dir,
            seed=case_seed,
            k_layers=int(params["k_layers"]),
            mu_min=float(params["mu_min"]),
            mu_max=float(params["mu_max"]),
            d0=int(params["d0"]),
            growth=float(params["growth"]),
            holo_growth=float(params["holo_growth"]),
            d_max=int(params["d_max"]),
            mix_pure=float(params["mix_pure"]),
            weight_mode=str(params["weight_mode"]),
            resolution_scan=str(params["resolution_scan"]),
            write_detail_csv=False,
        )

        s = summary_df.iloc[0]
        row = {
            "case_id": case_id,
            "seed": case_seed,
            **params,
            "mean_cut_fraction": float(s["mean_cut_fraction"]),
            "lambda_raw_unreg": float(s["lambda_raw_unreg"]),
            "lambda_cut_unreg": float(s["lambda_cut_unreg"]),
            "lambda_raw_reg": float(s["lambda_raw_reg"]),
            "lambda_cut_reg": float(s["lambda_cut_reg"]),
            "trace_raw_max": float(s["trace_raw_max"]),
            "trace_cut_max": float(s["trace_cut_max"]),
            "resolution_stability_gain": float(s["resolution_stability_gain"]),
        }
        row["pass_trace_tol"] = bool(max(row["trace_raw_max"], row["trace_cut_max"]) <= trace_tol)
        row["pass_stability_gain"] = bool(row["resolution_stability_gain"] >= stability_gain_min)
        row["pass_all"] = bool(row["pass_trace_tol"] and row["pass_stability_gain"])
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    worst_df = results_df.sort_values(
        ["resolution_stability_gain", "mean_cut_fraction"], ascending=[True, False]
    ).head(min(12, len(results_df)))

    summary_df = pd.DataFrame(
        [
            {
                "n_cases": int(n_cases),
                "seed": int(seed),
                "trace_tol": float(trace_tol),
                "stability_gain_min": float(stability_gain_min),
                "fraction_pass_all": float(results_df["pass_all"].mean()),
                "n_failed": int((~results_df["pass_all"]).sum()),
                "median_stability_gain": float(results_df["resolution_stability_gain"].median()),
                "min_stability_gain": float(results_df["resolution_stability_gain"].min()),
                "max_trace_raw": float(results_df["trace_raw_max"].max()),
                "max_trace_cut": float(results_df["trace_cut_max"].max()),
                "mean_cut_fraction_mean": float(results_df["mean_cut_fraction"].mean()),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outdir / "robustness_results.csv", index=False)
    summary_df.to_csv(outdir / "robustness_summary.csv", index=False)
    worst_df.to_csv(outdir / "worst_12_by_stability_gain.csv", index=False)
    return results_df, summary_df, worst_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_H_holographic_robust",
        help="output directory",
    )
    parser.add_argument("--cases", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260222)
    parser.add_argument("--trace-tol", type=float, default=1e-11)
    parser.add_argument("--stability-gain-min", type=float, default=1.0)
    args = parser.parse_args()

    _, summary_df, worst_df = run_robustness(
        outdir=Path(args.out),
        n_cases=args.cases,
        seed=args.seed,
        trace_tol=args.trace_tol,
        stability_gain_min=args.stability_gain_min,
    )
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nWorst cases by stability gain:")
    print(
        worst_df[
            [
                "case_id",
                "resolution_stability_gain",
                "mean_cut_fraction",
                "trace_raw_max",
                "trace_cut_max",
                "pass_all",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6e}")
    )
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
