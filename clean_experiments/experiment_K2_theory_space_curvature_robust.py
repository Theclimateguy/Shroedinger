#!/usr/bin/env python3
"""Robustness sweep for Experiment K2 (theory-space curvature diagnostics)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_K2_theory_space_curvature import K2Params, run_experiment
except ImportError:
    from experiment_K2_theory_space_curvature import K2Params, run_experiment


def _sample_params(rng: np.random.Generator) -> K2Params:
    alpha1 = float(rng.uniform(0.18, 0.42))
    delta_alpha = float(rng.uniform(0.22, 0.72))
    alpha2 = alpha1 + delta_alpha
    theta1 = float(rng.uniform(0.38, 0.82))
    theta2 = float(rng.uniform(0.32, 0.88))
    phi_min = float(rng.uniform(0.08, 0.18))
    phi_max = float(rng.uniform(1.25, 1.52))
    return K2Params(
        alpha1=alpha1,
        alpha2=alpha2,
        theta1=theta1,
        theta2=theta2,
        u_min=0.15,
        u_max=0.85,
        phi_min=min(phi_min, phi_max - 0.15),
        phi_max=phi_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_K2_theory_space_curvature_robust",
        help="output directory",
    )
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-u", type=int, default=17)
    parser.add_argument("--n-phi", type=int, default=17)
    parser.add_argument("--h-deriv", type=float, default=4e-3)
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | bool]] = []

    for case_id in range(args.cases):
        case_seed = int(rng.integers(0, 2**31 - 1))
        params = _sample_params(rng)
        _, _, _, summary_df = run_experiment(
            outdir=outdir / f"case_{case_id:03d}",
            n_u=args.n_u,
            n_phi=args.n_phi,
            h_deriv=args.h_deriv,
            params=params,
            gauge_checks=4,
            seed=case_seed,
            write_csv=False,
            verbose=False,
        )
        s = summary_df.iloc[0]
        pass_case = bool(
            np.isfinite(s["corr_source_comm"])
            and np.isfinite(s["corr_source_fs_omega"])
            and s["corr_source_comm"] >= 0.92
            and abs(s["corr_source_fs_omega"]) >= 0.82
            and s["gauge_max_source_diff"] < 1e-8
            and s["gauge_max_response_metric_diff"] < 1e-8
            and s["avg_resp_ricci_finite_frac"] > 0.5
        )
        rows.append(
            {
                "case_id": int(case_id),
                "case_seed": case_seed,
                "alpha1": params.alpha1,
                "alpha2": params.alpha2,
                "theta1": params.theta1,
                "theta2": params.theta2,
                "phi_min": params.phi_min,
                "phi_max": params.phi_max,
                "corr_source_comm": float(s["corr_source_comm"]),
                "corr_source_fs_omega": float(s["corr_source_fs_omega"]),
                "corr_source_resp_det": float(s["corr_source_resp_det"]),
                "corr_source_resp_ricci": float(s["corr_source_resp_ricci"]),
                "gauge_max_source_diff": float(s["gauge_max_source_diff"]),
                "gauge_max_response_metric_diff": float(s["gauge_max_response_metric_diff"]),
                "max_corr_delta_coarse": float(s["max_corr_delta_coarse"]),
                "avg_resp_ricci_finite_frac": float(s["avg_resp_ricci_finite_frac"]),
                "pass_case": pass_case,
            }
        )
        print(
            f"[K2-robust] case {case_id:02d}: "
            f"corr_sc={float(s['corr_source_comm']):.4f}, "
            f"corr_so={float(s['corr_source_fs_omega']):.4f}, "
            f"pass={pass_case}"
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(outdir / "robustness_results.csv", index=False)

    summary = {
        "cases": int(args.cases),
        "success_rate": float(results_df["pass_case"].mean()) if len(results_df) else float("nan"),
        "min_corr_source_comm": float(results_df["corr_source_comm"].min()) if len(results_df) else float("nan"),
        "min_abs_corr_source_fs_omega": float(np.min(np.abs(results_df["corr_source_fs_omega"])))
        if len(results_df)
        else float("nan"),
        "max_gauge_source_diff": float(results_df["gauge_max_source_diff"].max()) if len(results_df) else float("nan"),
        "max_gauge_response_metric_diff": float(results_df["gauge_max_response_metric_diff"].max())
        if len(results_df)
        else float("nan"),
        "max_corr_delta_coarse": float(results_df["max_corr_delta_coarse"].max()) if len(results_df) else float("nan"),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(outdir / "robustness_summary.csv", index=False)
    print(f"[K2-robust] saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
