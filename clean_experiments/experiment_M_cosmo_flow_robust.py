#!/usr/bin/env python3
"""Robustness sweep for Experiment M (cosmo flow structural-scale test)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_M_cosmo_flow import run_experiment
except ImportError:
    from experiment_M_cosmo_flow import run_experiment


def _sample_case(rng: np.random.Generator, base_edges: list[float]) -> dict[str, float | int | list[float]]:
    scale = float(rng.choice([0.85, 1.0, 1.15]))
    edges = [float(x * scale) for x in base_edges]
    return {
        "scale_edges_km": edges,
        "n_modes_per_var": int(rng.choice([5, 6, 8])),
        "window": int(rng.choice([24, 36, 48, 60])),
        "peak_quantile": float(rng.choice([0.85, 0.90, 0.93])),
        "ridge_alpha": float(rng.choice([1e-6, 3e-6, 1e-5])),
        "cov_shrinkage": float(rng.choice([0.0, 0.1, 0.2])),
        "folds": int(rng.choice([5, 6])),
        "perm_block": int(rng.choice([12, 24, 36])),
        "precip_factor": float(rng.choice([1.0, 0.8, 1.2])),
        "evap_factor": float(rng.choice([1.0, 0.8, 1.2])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to .npz or NetCDF/Zarr input")
    parser.add_argument("--out", default="clean_experiments/results/experiment_M_cosmo_flow_robust", help="Output dir")

    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260307)
    parser.add_argument("--n-perm", type=int, default=80, help="Permutation count per case")

    parser.add_argument("--scale-edges-km", default="25,50,100,200,400,800,1600")
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--lat-stride", type=int, default=1)
    parser.add_argument("--lon-stride", type=int, default=1)
    parser.add_argument("--max-time", type=int, default=None)

    parser.add_argument("--level-dim", default=None)
    parser.add_argument("--level-index", type=int, default=0)

    parser.add_argument("--iwv-var", default=None)
    parser.add_argument("--ivt-u-var", default=None)
    parser.add_argument("--ivt-v-var", default=None)
    parser.add_argument("--precip-var", default=None)
    parser.add_argument("--evap-var", default=None)
    parser.add_argument("--u-var", default=None)
    parser.add_argument("--v-var", default=None)
    parser.add_argument("--temp-var", default=None)
    parser.add_argument("--pressure-var", default=None)
    parser.add_argument("--density-var", default=None)

    parser.add_argument("--strata-q", type=int, default=3)
    parser.add_argument("--min-mae-gain", type=float, default=0.03)
    parser.add_argument("--max-perm-p", type=float, default=0.05)
    parser.add_argument("--min-sign-consistency", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-strata-gain", type=float, default=0.0)
    parser.add_argument("--min-positive-strata-frac", type=float, default=1.0)

    parser.add_argument("--no-standardize-components", action="store_true")
    parser.add_argument(
        "--residual-mode",
        choices=["component_zscore", "physical_zscore", "physical_raw"],
        default=None,
    )
    parser.add_argument(
        "--coherence-mode",
        choices=["offdiag_fro", "relative_offdiag_fro"],
        default="offdiag_fro",
    )
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    var_overrides = {
        "iwv": args.iwv_var,
        "ivt_u": args.ivt_u_var,
        "ivt_v": args.ivt_v_var,
        "precip": args.precip_var,
        "evap": args.evap_var,
        "u": args.u_var,
        "v": args.v_var,
        "temp": args.temp_var,
        "pressure": args.pressure_var,
        "density": args.density_var,
    }

    base_edges = [float(x.strip()) for x in args.scale_edges_km.split(",") if x.strip()]

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | bool | str]] = []

    for case_id in range(args.cases):
        params = _sample_case(rng, base_edges)
        case_seed = int(rng.integers(0, 2**31 - 1))
        case_dir = outdir / f"case_{case_id:03d}"

        try:
            _, _, _, _, _, summary_df = run_experiment(
                input_path=Path(args.input),
                outdir=case_dir,
                scale_edges_km=params["scale_edges_km"],
                n_modes_per_var=int(params["n_modes_per_var"]),
                window=int(params["window"]),
                peak_quantile=float(params["peak_quantile"]),
                ridge_alpha=float(params["ridge_alpha"]),
                n_folds=int(params["folds"]),
                n_perm=args.n_perm,
                perm_block=int(params["perm_block"]),
                seed=case_seed,
                precip_factor=float(params["precip_factor"]),
                evap_factor=float(params["evap_factor"]),
                standardize_components=not args.no_standardize_components,
                residual_mode=args.residual_mode,
                cov_shrinkage=float(params["cov_shrinkage"]),
                coherence_mode=args.coherence_mode,
                strata_q=args.strata_q,
                min_mae_gain=args.min_mae_gain,
                max_perm_p=args.max_perm_p,
                min_sign_consistency=args.min_sign_consistency,
                min_strata_gain=args.min_strata_gain,
                min_positive_strata_frac=args.min_positive_strata_frac,
                time_stride=args.time_stride,
                lat_stride=args.lat_stride,
                lon_stride=args.lon_stride,
                max_time=args.max_time,
                level_dim=args.level_dim,
                level_index=args.level_index,
                var_overrides=var_overrides,
                verbose=False,
            )
            s = summary_df.iloc[0]
            row = {
                "case_id": int(case_id),
                "case_seed": case_seed,
                "status": "ok",
                "scale_edges_km": ",".join(f"{x:.2f}" for x in params["scale_edges_km"]),
                "n_modes_per_var": int(params["n_modes_per_var"]),
                "window": int(params["window"]),
                "peak_quantile": float(params["peak_quantile"]),
                "ridge_alpha": float(params["ridge_alpha"]),
                "cov_shrinkage": float(params["cov_shrinkage"]),
                "folds": int(params["folds"]),
                "perm_block": int(params["perm_block"]),
                "precip_factor": float(params["precip_factor"]),
                "evap_factor": float(params["evap_factor"]),
                "oof_gain_frac": float(s["oof_gain_frac"]),
                "perm_p_value": float(s["perm_p_value"]),
                "lambda_sign_consistency": float(s["lambda_sign_consistency"]),
                "strata_min_gain": float(s["strata_min_gain"]) if not pd.isna(s["strata_min_gain"]) else np.nan,
                "strata_positive_frac": float(s["strata_positive_frac"]) if "strata_positive_frac" in s and not pd.isna(s["strata_positive_frac"]) else np.nan,
                "pass_all": bool(s["pass_all"]),
            }
            print(
                f"[M-robust] case {case_id:02d}: gain={row['oof_gain_frac']:.4f}, "
                f"p={row['perm_p_value']:.4f}, sign={row['lambda_sign_consistency']:.3f}, pass={row['pass_all']}"
            )
        except Exception as exc:
            row = {
                "case_id": int(case_id),
                "case_seed": case_seed,
                "status": "error",
                "scale_edges_km": ",".join(f"{x:.2f}" for x in params["scale_edges_km"]),
                "n_modes_per_var": int(params["n_modes_per_var"]),
                "window": int(params["window"]),
                "peak_quantile": float(params["peak_quantile"]),
                "ridge_alpha": float(params["ridge_alpha"]),
                "cov_shrinkage": float(params["cov_shrinkage"]),
                "folds": int(params["folds"]),
                "perm_block": int(params["perm_block"]),
                "precip_factor": float(params["precip_factor"]),
                "evap_factor": float(params["evap_factor"]),
                "oof_gain_frac": np.nan,
                "perm_p_value": np.nan,
                "lambda_sign_consistency": np.nan,
                "strata_min_gain": np.nan,
                "strata_positive_frac": np.nan,
                "pass_all": False,
                "error": str(exc),
            }
            print(f"[M-robust] case {case_id:02d}: error={exc}")

        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "robustness_results.csv", index=False)

    ok = results[results["status"] == "ok"]
    summary = pd.DataFrame(
        [
            {
                "cases": int(args.cases),
                "cases_ok": int(len(ok)),
                "success_rate": float(ok["pass_all"].mean()) if len(ok) else float("nan"),
                "median_oof_gain": float(ok["oof_gain_frac"].median()) if len(ok) else float("nan"),
                "min_oof_gain": float(ok["oof_gain_frac"].min()) if len(ok) else float("nan"),
                "max_perm_p": float(ok["perm_p_value"].max()) if len(ok) else float("nan"),
                "median_lambda_sign_consistency": float(ok["lambda_sign_consistency"].median()) if len(ok) else float("nan"),
                "min_strata_gain": float(ok["strata_min_gain"].min()) if len(ok) else float("nan"),
            }
        ]
    )
    summary.to_csv(outdir / "robustness_summary.csv", index=False)

    print(f"[M-robust] saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
