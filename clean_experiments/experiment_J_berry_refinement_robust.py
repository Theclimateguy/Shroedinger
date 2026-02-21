#!/usr/bin/env python3
"""Experiment J robust sweep: Berry-phase law across polar angles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_J_berry_refinement import _berry_phase_discrete, _wrapped_angle_distance
except ImportError:
    from experiment_J_berry_refinement import _berry_phase_discrete, _wrapped_angle_distance


def _parse_float_list(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    return sorted(set(vals))


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    vals = [v for v in vals if v >= 16]
    return sorted(set(vals))


def run_robustness(
    outdir: Path,
    *,
    theta_scan: str = "0.2,0.3,0.4,0.5,0.6,0.7,0.8",
    n_steps_scan: str = "64,96,128,192,256,384,512",
    abs_err_tol: float = 3e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thetas = [float(np.pi * x) for x in _parse_float_list(theta_scan)]
    ns = _parse_int_list(n_steps_scan)

    rows = []
    for theta in thetas:
        # For H = n·sigma and the lowest eigenstate in this convention,
        # the geometric phase is +Omega/2 = +pi(1-cos theta) modulo 2pi.
        target = np.pi * (1.0 - np.cos(theta))
        for n_steps in ns:
            gamma = _berry_phase_discrete(theta, n_steps)
            err = _wrapped_angle_distance(gamma, target)
            rows.append(
                {
                    "theta0": float(theta),
                    "theta0_over_pi": float(theta / np.pi),
                    "n_steps": int(n_steps),
                    "dphi": float(2.0 * np.pi / n_steps),
                    "target_phase": float(target),
                    "berry_phase": float(gamma),
                    "abs_error": float(err),
                }
            )

    df = pd.DataFrame(rows).sort_values(["theta0", "n_steps"]).reset_index(drop=True)
    grp = df.groupby("theta0", as_index=False).agg(
        theta0_over_pi=("theta0_over_pi", "first"),
        target_phase=("target_phase", "first"),
        finest_n_steps=("n_steps", "max"),
        finest_abs_error=("abs_error", "last"),
        max_abs_error=("abs_error", "max"),
        median_abs_error=("abs_error", "median"),
    )
    grp["pass_abs_err_tol"] = grp["finest_abs_error"] <= abs_err_tol

    summary = pd.DataFrame(
        [
            {
                "n_thetas": int(len(grp)),
                "n_steps_min": int(min(ns)),
                "n_steps_max": int(max(ns)),
                "abs_err_tol": float(abs_err_tol),
                "fraction_pass_abs_err_tol": float(grp["pass_abs_err_tol"].mean()),
                "max_finest_abs_error": float(grp["finest_abs_error"].max()),
                "median_finest_abs_error": float(grp["finest_abs_error"].median()),
                "pass_all": bool(grp["pass_abs_err_tol"].all()),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "robustness_results.csv", index=False)
    grp.to_csv(outdir / "robustness_by_theta.csv", index=False)
    summary.to_csv(outdir / "robustness_summary.csv", index=False)
    return grp, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_J_berry_refinement_robust",
        help="output directory",
    )
    parser.add_argument("--theta-scan", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--n-steps-scan", default="64,96,128,192,256,384,512")
    parser.add_argument("--abs-err-tol", type=float, default=3e-3)
    args = parser.parse_args()

    by_theta, summary = run_robustness(
        outdir=Path(args.out),
        theta_scan=args.theta_scan,
        n_steps_scan=args.n_steps_scan,
        abs_err_tol=args.abs_err_tol,
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nBy theta:")
    print(by_theta.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
