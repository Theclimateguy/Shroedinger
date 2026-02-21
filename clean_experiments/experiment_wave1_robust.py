#!/usr/bin/env python3
"""Experiment B (wave-1) robust sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_wave1_user import run_experiment
except ImportError:
    from experiment_wave1_user import run_experiment


def _sample_case(rng: np.random.Generator) -> dict[str, float | int]:
    alpha1 = float(rng.uniform(0.12, 0.70))
    alpha2 = float(alpha1 + rng.uniform(0.08, 0.80))

    return {
        "nx": int(rng.choice([96, 128, 160, 200, 256, 320])),
        "alpha1_val": alpha1,
        "alpha2_val": alpha2,
        "alpha_mod_amp": float(rng.uniform(0.00, 0.35)),
        "theta1_base": float(rng.uniform(0.15, 0.80)),
        "theta1_amp": float(rng.uniform(0.05, 0.60)),
        "theta1_center": float(rng.uniform(0.20 * np.pi, 0.90 * np.pi)),
        "theta1_width": float(rng.uniform(0.15, 0.90)),
        "theta2_base": float(rng.uniform(0.15, 0.80)),
        "theta2_amp": float(rng.uniform(0.05, 0.60)),
        "theta2_center": float(rng.uniform(1.10 * np.pi, 1.95 * np.pi)),
        "theta2_width": float(rng.uniform(0.15, 0.90)),
        "n_angle_fine": int(rng.integers(24, 90)),
    }


def run_robustness(
    outdir: Path,
    n_cases: int = 120,
    seed: int = 20260220,
    spatial_tol: float = 1e-11,
    antisym_tol: float = 1e-11,
    angle_tol: float = 1e-11,
    ratio_std_tol: float = 1e-10,
    r2_min: float = 1.0 - 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | bool]] = []

    cases_dir = outdir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    for case_id in range(n_cases):
        params = _sample_case(rng)
        case_dir = cases_dir / f"case_{case_id:03d}"

        _, _, _, summary_df = run_experiment(
            outdir=case_dir,
            nx=int(params["nx"]),
            alpha1_val=float(params["alpha1_val"]),
            alpha2_val=float(params["alpha2_val"]),
            alpha_mod_amp=float(params["alpha_mod_amp"]),
            theta1_base=float(params["theta1_base"]),
            theta1_amp=float(params["theta1_amp"]),
            theta1_center=float(params["theta1_center"]),
            theta1_width=float(params["theta1_width"]),
            theta2_base=float(params["theta2_base"]),
            theta2_amp=float(params["theta2_amp"]),
            theta2_center=float(params["theta2_center"]),
            theta2_width=float(params["theta2_width"]),
            n_angle_fine=int(params["n_angle_fine"]),
            write_csv=True,
            verbose=False,
        )

        s = summary_df.iloc[0]
        row = {
            "case_id": case_id,
            "seed": seed,
            **params,
            "max_abs_spatial_err": float(s["max_abs_spatial_err"]),
            "max_abs_antisym_err": float(s["max_abs_antisym_err"]),
            "max_abs_angle_err": float(s["max_abs_angle_err"]),
            "rmse_angle": float(s["rmse_angle"]),
            "r2_angle": float(s["r2_angle"]),
            "ratio_mean": float(s["ratio_mean"]),
            "ratio_std": float(s["ratio_std"]),
        }
        row["pass_spatial_tol"] = bool(row["max_abs_spatial_err"] <= spatial_tol)
        row["pass_antisym_tol"] = bool(row["max_abs_antisym_err"] <= antisym_tol)
        row["pass_angle_tol"] = bool(row["max_abs_angle_err"] <= angle_tol)
        row["pass_ratio_std_tol"] = bool(row["ratio_std"] <= ratio_std_tol)
        row["pass_r2_min"] = bool(row["r2_angle"] >= r2_min)
        row["pass_all"] = bool(
            row["pass_spatial_tol"]
            and row["pass_antisym_tol"]
            and row["pass_angle_tol"]
            and row["pass_ratio_std_tol"]
            and row["pass_r2_min"]
        )
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    worst_by_angle_err = (
        results_df.sort_values(["max_abs_angle_err", "max_abs_spatial_err"], ascending=False)
        .head(min(12, len(results_df)))
        .reset_index(drop=True)
    )

    summary = {
        "n_cases": int(len(results_df)),
        "seed": int(seed),
        "spatial_tol": float(spatial_tol),
        "antisym_tol": float(antisym_tol),
        "angle_tol": float(angle_tol),
        "ratio_std_tol": float(ratio_std_tol),
        "r2_min": float(r2_min),
        "fraction_pass_all": float(results_df["pass_all"].mean()),
        "n_failed": int((~results_df["pass_all"]).sum()),
        "max_abs_spatial_err_max": float(results_df["max_abs_spatial_err"].max()),
        "max_abs_antisym_err_max": float(results_df["max_abs_antisym_err"].max()),
        "max_abs_angle_err_max": float(results_df["max_abs_angle_err"].max()),
        "max_ratio_std": float(results_df["ratio_std"].max()),
        "min_r2_angle": float(results_df["r2_angle"].min()),
        "p99_abs_angle_err": float(results_df["max_abs_angle_err"].quantile(0.99)),
        "p99_abs_spatial_err": float(results_df["max_abs_spatial_err"].quantile(0.99)),
    }
    summary_df = pd.DataFrame([summary])

    outdir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outdir / "robustness_results.csv", index=False)
    summary_df.to_csv(outdir / "robustness_summary.csv", index=False)
    worst_by_angle_err.to_csv(outdir / "worst_12_by_angle_err.csv", index=False)

    return results_df, summary_df, worst_by_angle_err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_B_wave1_robust",
        help="output directory",
    )
    parser.add_argument("--cases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260220)
    parser.add_argument("--spatial-tol", type=float, default=1e-11)
    parser.add_argument("--antisym-tol", type=float, default=1e-11)
    parser.add_argument("--angle-tol", type=float, default=1e-11)
    parser.add_argument("--ratio-std-tol", type=float, default=1e-10)
    parser.add_argument("--r2-min", type=float, default=1.0 - 1e-12)
    args = parser.parse_args()

    outdir = Path(args.out)
    _, summary_df, worst_df = run_robustness(
        outdir=outdir,
        n_cases=args.cases,
        seed=args.seed,
        spatial_tol=args.spatial_tol,
        antisym_tol=args.antisym_tol,
        angle_tol=args.angle_tol,
        ratio_std_tol=args.ratio_std_tol,
        r2_min=args.r2_min,
    )
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()
    print("Worst cases by max_abs_angle_err:")
    print(
        worst_df[
            [
                "case_id",
                "max_abs_angle_err",
                "max_abs_spatial_err",
                "max_abs_antisym_err",
                "ratio_std",
                "r2_angle",
                "pass_all",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.8e}")
    )
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
