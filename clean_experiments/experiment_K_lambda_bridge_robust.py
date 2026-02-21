#!/usr/bin/env python3
"""Experiment K robust sweep: Lambda bridge with repeated-holdout and permutation checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_K_lambda_bridge import run_experiment
except ImportError:
    from experiment_K_lambda_bridge import run_experiment


def _sample_run(rng: np.random.Generator) -> dict[str, float | int]:
    return {
        "n_cases": int(rng.choice([48, 54])),
        "n_splits": int(rng.choice([8, 10])),
        "n_perm": int(rng.choice([100, 140])),
        "n_boot": int(rng.choice([180, 240])),
        "g2_steps": int(rng.choice([80, 90])),
        "g2_traj": int(rng.choice([300, 360])),
        "g2_layers": int(rng.choice([48, 64])),
        "ridge_alpha": float(rng.choice([0.02, 0.03, 0.05])),
        "min_test_r2_median": float(rng.choice([0.22, 0.25])),
        "min_test_r2_p25": float(rng.choice([0.00, 0.05])),
        "max_perm_p": 0.05,
        "min_sign_consistency": float(rng.choice([0.62, 0.65])),
    }


def run_robustness(
    outdir: Path,
    *,
    n_runs: int = 6,
    seed: int = 20260226,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for run_id in range(n_runs):
        params = _sample_run(rng)
        run_seed = seed + 1000 + run_id
        run_dir = runs_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        _, _, _, _, _, summary_df = run_experiment(
            outdir=run_dir,
            seed=run_seed,
            n_cases=int(params["n_cases"]),
            n_splits=int(params["n_splits"]),
            n_perm=int(params["n_perm"]),
            n_boot=int(params["n_boot"]),
            g2_steps=int(params["g2_steps"]),
            g2_traj=int(params["g2_traj"]),
            g2_layers=int(params["g2_layers"]),
            ridge_alpha=float(params["ridge_alpha"]),
            min_test_r2_median=float(params["min_test_r2_median"]),
            min_test_r2_p25=float(params["min_test_r2_p25"]),
            max_perm_p=float(params["max_perm_p"]),
            min_sign_consistency=float(params["min_sign_consistency"]),
        )
        s = summary_df.iloc[0]
        rows.append(
            {
                "run_id": int(run_id),
                "seed": int(run_seed),
                **params,
                "train_r2_mean": float(s["train_r2_mean"]),
                "test_r2_median": float(s["test_r2_median"]),
                "test_r2_p25": float(s["test_r2_p25"]),
                "test_r2_min": float(s["test_r2_min"]),
                "perm_p_value": float(s["perm_p_value"]),
                "perm_stat_p95": float(s["perm_stat_p95"]),
                "boot_min_sign_consistency_key": float(s["boot_min_sign_consistency_key"]),
                "boot_mean_sign_consistency_key": float(s["boot_mean_sign_consistency_key"]),
                "profile_min_sign_consistency_key": float(s["profile_min_sign_consistency_key"])
                if not pd.isna(s["profile_min_sign_consistency_key"])
                else np.nan,
                "profile_mean_sign_consistency_key": float(s["profile_mean_sign_consistency_key"])
                if not pd.isna(s["profile_mean_sign_consistency_key"])
                else np.nan,
                "pass_all": bool(s["pass_all"]),
            }
        )

    results = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    worst = results.sort_values(["test_r2_median", "perm_p_value"], ascending=[True, False]).head(min(10, len(results)))
    summary = pd.DataFrame(
        [
            {
                "n_runs": int(n_runs),
                "seed": int(seed),
                "fraction_pass_all": float(results["pass_all"].mean()),
                "n_failed": int((~results["pass_all"]).sum()),
                "median_test_r2_median": float(results["test_r2_median"].median()),
                "min_test_r2_median": float(results["test_r2_median"].min()),
                "median_test_r2_p25": float(results["test_r2_p25"].median()),
                "min_test_r2_p25": float(results["test_r2_p25"].min()),
                "median_perm_p": float(results["perm_p_value"].median()),
                "max_perm_p": float(results["perm_p_value"].max()),
                "median_boot_sign_consistency_mean": float(results["boot_mean_sign_consistency_key"].median()),
                "min_boot_sign_consistency_mean": float(results["boot_mean_sign_consistency_key"].min()),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(outdir / "robustness_results.csv", index=False)
    summary.to_csv(outdir / "robustness_summary.csv", index=False)
    worst.to_csv(outdir / "worst_10_runs.csv", index=False)
    return results, summary, worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_K_lambda_bridge_robust",
        help="output directory",
    )
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260226)
    args = parser.parse_args()

    _, summary, worst = run_robustness(
        outdir=Path(args.out),
        n_runs=args.runs,
        seed=args.seed,
    )

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nWorst runs:")
    print(
        worst[
            [
                "run_id",
                "test_r2_median",
                "test_r2_p25",
                "perm_p_value",
                "boot_mean_sign_consistency_key",
                "pass_all",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6e}")
    )
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
