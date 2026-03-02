#!/usr/bin/env python3
"""Robustness sweep for Experiment L (matter fields)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_L_matter_fields import LParams, run_experiment
except ImportError:
    from experiment_L_matter_fields import LParams, run_experiment


def _sample_params(rng: np.random.Generator) -> LParams:
    n_sites = int(rng.choice([3, 4], p=[0.7, 0.3]))
    return LParams(
        n_sites=n_sites,
        t_hop=float(rng.uniform(0.45, 1.05)),
        mass=float(rng.uniform(0.05, 0.45)),
        omega_gauge=float(rng.uniform(0.15, 0.80)),
        theta_base=float(rng.uniform(0.20, 0.75)),
        theta_step=float(rng.uniform(0.05, 0.35)),
        gamma_matter=float(rng.uniform(0.0, 0.10)),
        gamma_gauge=float(rng.uniform(0.0, 0.10)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="clean_experiments/results/experiment_L_matter_fields_robust", help="output dir")
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30303)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--dt", type=float, default=0.025)
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | bool]] = []

    for case_id in range(args.cases):
        params = _sample_params(rng)
        case_seed = int(rng.integers(0, 2**31 - 1))
        _, _, summary_df = run_experiment(
            outdir=outdir / f"case_{case_id:03d}",
            params=params,
            seed=case_seed,
            n_samples=args.samples,
            n_steps=args.steps,
            dt=args.dt,
            write_csv=False,
            verbose=False,
        )
        s = summary_df.iloc[0]
        pass_case = bool(
            s["comm_QH_fro"] < 1e-10
            and s["max_abs_continuity_residual"] < 1e-10
            and s["max_total_charge_drift"] < 1e-7
        )
        rows.append(
            {
                "case_id": int(case_id),
                "case_seed": case_seed,
                "n_sites": int(params.n_sites),
                "t_hop": params.t_hop,
                "mass": params.mass,
                "omega_gauge": params.omega_gauge,
                "theta_base": params.theta_base,
                "theta_step": params.theta_step,
                "gamma_matter": params.gamma_matter,
                "gamma_gauge": params.gamma_gauge,
                "comm_QH_fro": float(s["comm_QH_fro"]),
                "max_abs_continuity_residual": float(s["max_abs_continuity_residual"]),
                "max_total_charge_drift": float(s["max_total_charge_drift"]),
                "pass_case": pass_case,
            }
        )
        print(
            f"[L-robust] case {case_id:02d}: "
            f"comm={float(s['comm_QH_fro']):.2e}, "
            f"cont={float(s['max_abs_continuity_residual']):.2e}, "
            f"drift={float(s['max_total_charge_drift']):.2e}, pass={pass_case}"
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(outdir / "robustness_results.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "cases": int(args.cases),
                "success_rate": float(result_df["pass_case"].mean()) if len(result_df) else float("nan"),
                "max_comm_QH_fro": float(result_df["comm_QH_fro"].max()) if len(result_df) else float("nan"),
                "max_abs_continuity_residual": float(result_df["max_abs_continuity_residual"].max())
                if len(result_df)
                else float("nan"),
                "max_total_charge_drift": float(result_df["max_total_charge_drift"].max()) if len(result_df) else float("nan"),
            }
        ]
    )
    summary.to_csv(outdir / "robustness_summary.csv", index=False)
    print(f"[L-robust] saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
