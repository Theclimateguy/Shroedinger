#!/usr/bin/env python3
"""Experiment I robust sweep: continuum extrapolation of balance residual."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_I_continuum_conservation import run_experiment
except ImportError:
    from experiment_I_continuum_conservation import run_experiment


def _sample_case(rng: np.random.Generator) -> dict[str, float | int | str]:
    n_sites = int(rng.choice([4, 5]))
    kvals = "3,4,5,6" if n_sites == 4 else "3,4,5"
    dtvals = str(rng.choice(["0.12,0.06,0.03,0.015", "0.10,0.05,0.025,0.0125"]))
    return {
        "n_sites": n_sites,
        "k_values": kvals,
        "dt_values": dtvals,
        "n_steps": int(rng.choice([40, 50, 60])),
        "gamma_deph": float(rng.uniform(0.15, 0.80)),
        "eta0": float(rng.uniform(0.20, 1.60)),
        "delta_mu_span": 1.0,
        "intercept_tol": float(rng.choice([5e-4, 1e-3, 2e-3])),
    }


def run_robustness(
    outdir: Path,
    *,
    n_cases: int = 24,
    seed: int = 20260224,
    intercept_hard_tol: float = 5e-3,
    intercept_rel_tol: float = 8e-2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []
    cases_dir = outdir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    for case_id in range(n_cases):
        params = _sample_case(rng)
        case_seed = seed + 10000 + case_id
        case_dir = cases_dir / f"case_{case_id:03d}"

        _, _, ext_df = run_experiment(
            outdir=case_dir,
            seed=case_seed,
            n_sites=int(params["n_sites"]),
            k_values=str(params["k_values"]),
            dt_values=str(params["dt_values"]),
            n_steps=int(params["n_steps"]),
            gamma_deph=float(params["gamma_deph"]),
            eta0=float(params["eta0"]),
            delta_mu_span=float(params["delta_mu_span"]),
            intercept_tol=float(params["intercept_tol"]),
            write_dataset=False,
        )
        ext = ext_df.iloc[0]
        row = {
            "case_id": case_id,
            "seed": case_seed,
            **params,
            "intercept_max_abs_residual": float(ext["intercept_max_abs_residual"]),
            "fit_r2_max_abs_residual": float(ext["fit_r2_max_abs_residual"]),
            "max_residual_overall": float(ext["max_residual_overall"]),
        }
        row["intercept_scaled_tol"] = float(max(intercept_hard_tol, intercept_rel_tol * row["max_residual_overall"]))
        row["pass_hard_tol"] = bool(abs(row["intercept_max_abs_residual"]) <= intercept_hard_tol)
        row["pass_scaled_tol"] = bool(abs(row["intercept_max_abs_residual"]) <= row["intercept_scaled_tol"])
        row["pass_r2_nontrivial"] = bool(row["fit_r2_max_abs_residual"] >= 0.30)
        row["pass_all"] = bool(row["pass_scaled_tol"] and row["pass_r2_nontrivial"])
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    worst_df = results_df.sort_values(
        ["intercept_max_abs_residual", "max_residual_overall"], ascending=[False, False]
    ).head(min(12, len(results_df)))

    summary_df = pd.DataFrame(
        [
            {
                "n_cases": int(n_cases),
                "seed": int(seed),
                "intercept_hard_tol": float(intercept_hard_tol),
                "intercept_rel_tol": float(intercept_rel_tol),
                "fraction_pass_hard_tol": float(results_df["pass_hard_tol"].mean()),
                "fraction_pass_scaled_tol": float(results_df["pass_scaled_tol"].mean()),
                "fraction_pass_all": float(results_df["pass_all"].mean()),
                "n_failed": int((~results_df["pass_all"]).sum()),
                "median_abs_intercept": float(np.median(np.abs(results_df["intercept_max_abs_residual"]))),
                "max_abs_intercept": float(np.max(np.abs(results_df["intercept_max_abs_residual"]))),
                "median_fit_r2": float(np.median(results_df["fit_r2_max_abs_residual"])),
                "min_fit_r2": float(results_df["fit_r2_max_abs_residual"].min()),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outdir / "robustness_results.csv", index=False)
    summary_df.to_csv(outdir / "robustness_summary.csv", index=False)
    worst_df.to_csv(outdir / "worst_12_by_intercept.csv", index=False)
    return results_df, summary_df, worst_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_I_continuum_conservation_robust",
        help="output directory",
    )
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260224)
    parser.add_argument("--intercept-hard-tol", type=float, default=5e-3)
    parser.add_argument("--intercept-rel-tol", type=float, default=8e-2)
    args = parser.parse_args()

    _, summary_df, worst_df = run_robustness(
        outdir=Path(args.out),
        n_cases=args.cases,
        seed=args.seed,
        intercept_hard_tol=args.intercept_hard_tol,
        intercept_rel_tol=args.intercept_rel_tol,
    )
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nWorst cases by |intercept|:")
    print(
        worst_df[
            [
                "case_id",
                "intercept_max_abs_residual",
                "fit_r2_max_abs_residual",
                "max_residual_overall",
                "intercept_scaled_tol",
                "pass_hard_tol",
                "pass_scaled_tol",
                "pass_all",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6e}")
    )
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
