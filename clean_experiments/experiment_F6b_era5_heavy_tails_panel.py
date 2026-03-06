#!/usr/bin/env python3
"""Experiment F6b-panel: heavy tails of panel |Lambda_local(t,y,x)| in ERA5/WPWP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

try:
    from clean_experiments.experiment_M_cosmo_flow import _build_band_masks, _compute_vorticity, _xy_coordinates_m
    from clean_experiments.experiment_O_spatial_variance import (
        _compute_lambda_local_batch,
        _find_coord_name,
        _find_var_name,
        _load_m_band_amplitudes,
        _to_datetime64,
    )
    from clean_experiments.experiment_F6b_era5_heavy_tails import (
        F6bConfig,
        _bootstrap_alpha_ci,
        _plot_global_tail,
        _plot_strata_compare,
        _safe_quantile,
        _scan_xmin_candidates,
        _zscore,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _build_band_masks, _compute_vorticity, _xy_coordinates_m  # type: ignore
    from experiment_O_spatial_variance import (  # type: ignore
        _compute_lambda_local_batch,
        _find_coord_name,
        _find_var_name,
        _load_m_band_amplitudes,
        _to_datetime64,
    )
    from experiment_F6b_era5_heavy_tails import (  # type: ignore
        F6bConfig,
        _bootstrap_alpha_ci,
        _plot_global_tail,
        _plot_strata_compare,
        _safe_quantile,
        _scan_xmin_candidates,
        _zscore,
    )


def _build_time_regime_masks(f5_df: pd.DataFrame, cfg: F6bConfig) -> dict[str, np.ndarray]:
    req = {"fractal_psd_beta", "fractal_variogram_slope", "residual_base_res0"}
    missing = [c for c in req if c not in f5_df.columns]
    if missing:
        raise KeyError(f"F5 dataset is missing required columns for stratification: {missing}")

    z_psd = _zscore(f5_df["fractal_psd_beta"].to_numpy(float))
    z_var = _zscore(f5_df["fractal_variogram_slope"].to_numpy(float))
    fractal_comp = 0.5 * (z_psd + z_var)
    residual_abs = np.abs(f5_df["residual_base_res0"].to_numpy(float))

    q_f_hi = _safe_quantile(fractal_comp, cfg.fractal_high_q)
    q_f_lo = _safe_quantile(fractal_comp, cfg.fractal_low_q)
    q_c_hi = _safe_quantile(residual_abs, cfg.conv_high_q)
    q_c_lo = _safe_quantile(residual_abs, cfg.calm_low_q)

    finite_f = np.isfinite(fractal_comp)
    finite_c = np.isfinite(residual_abs)

    masks = {
        "fractal_high_t": finite_f & (fractal_comp >= q_f_hi),
        "fractal_low_t": finite_f & (fractal_comp <= q_f_lo),
        "convective_fractal_t": finite_f & finite_c & (fractal_comp >= q_f_hi) & (residual_abs >= q_c_hi),
        "calm_t": finite_f & finite_c & (fractal_comp <= q_f_lo) & (residual_abs <= q_c_lo),
    }
    return masks


def _reconstruct_lambda_panel_abs(
    *,
    input_nc: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    scale_edges_km: list[float],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    band_amp, _ = _load_m_band_amplitudes(
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
    )

    with Dataset(input_nc, mode="r") as ds:
        time_name = _find_coord_name(ds, ("valid_time", "time", "datetime", "date"), "time")
        lat_name = _find_coord_name(ds, ("latitude", "lat", "y", "rlat"), "latitude")
        lon_name = _find_coord_name(ds, ("longitude", "lon", "x", "rlon"), "longitude")
        u_name = _find_var_name(ds, ("u",), "u")
        v_name = _find_var_name(ds, ("v",), "v")

        time_ns = _to_datetime64(ds.variables[time_name])
        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)
        nt = len(time_ns)
        ny = len(lat)
        nx = len(lon)
        npix = ny * nx

        if band_amp.shape[0] != nt:
            raise ValueError(f"band_amp nt={band_amp.shape[0]} does not match input nt={nt}.")

        x_m, y_m = _xy_coordinates_m(lat, lon)
        dx_km = float(np.median(np.diff(x_m))) / 1000.0
        dy_km = float(np.median(np.diff(y_m))) / 1000.0
        masks, _, _, _, _, _ = _build_band_masks(
            ny=ny,
            nx=nx,
            dy_km=abs(dy_km),
            dx_km=abs(dx_km),
            scale_edges_km=scale_edges_km,
        )
        if len(masks) != band_amp.shape[1]:
            raise ValueError(
                f"Band mismatch: local masks={len(masks)} vs M amplitudes={band_amp.shape[1]}. "
                "Use matching scale edges."
            )

        out = np.empty((nt, npix), dtype=np.float32)

        u_var = ds.variables[u_name]
        v_var = ds.variables[v_name]

        for start in range(0, nt, batch_size):
            stop = min(start + batch_size, nt)
            sl = slice(start, stop)

            u_blk = np.asarray(u_var[sl, ...], dtype=np.float64)
            v_blk = np.asarray(v_var[sl, ...], dtype=np.float64)
            u_blk = np.nan_to_num(u_blk, nan=0.0, posinf=0.0, neginf=0.0)
            v_blk = np.nan_to_num(v_blk, nan=0.0, posinf=0.0, neginf=0.0)

            zeta_blk = _compute_vorticity(u=u_blk, v=v_blk, x_m=x_m, y_m=y_m)
            lam_blk = _compute_lambda_local_batch(
                zeta_batch=zeta_blk,
                masks=masks,
                band_amp_batch=band_amp[start:stop],
            )
            out[start:stop] = np.abs(lam_blk.reshape(stop - start, npix)).astype(np.float32)
            print(f"[F6b-panel] reconstructed lambda_local: {stop}/{nt}", flush=True)

    return out, time_ns


def run_f6b_panel(
    *,
    input_nc: Path,
    f5_dataset_csv: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    outdir: Path,
    scale_edges_km: list[float],
    batch_size: int,
    max_samples_global: int,
    max_samples_regime: int,
    cfg: F6bConfig,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.rng_seed)

    f5_df = pd.read_csv(f5_dataset_csv).sort_values("time_index").reset_index(drop=True)
    if "time_index" not in f5_df.columns:
        raise KeyError("F5 dataset must contain `time_index` for panel-time alignment.")

    masks_t = _build_time_regime_masks(f5_df, cfg=cfg)

    lambda_panel_abs, time_ns = _reconstruct_lambda_panel_abs(
        input_nc=input_nc,
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
        scale_edges_km=scale_edges_km,
        batch_size=batch_size,
    )

    nt_panel = lambda_panel_abs.shape[0]
    if nt_panel != len(f5_df):
        raise ValueError(f"Time mismatch: panel nt={nt_panel}, F5 rows={len(f5_df)}")

    x_by_regime_full = {
        "global": lambda_panel_abs.reshape(-1),
        "fractal_high": lambda_panel_abs[masks_t["fractal_high_t"]].reshape(-1),
        "fractal_low": lambda_panel_abs[masks_t["fractal_low_t"]].reshape(-1),
        "convective_fractal": lambda_panel_abs[masks_t["convective_fractal_t"]].reshape(-1),
        "calm": lambda_panel_abs[masks_t["calm_t"]].reshape(-1),
    }
    rng = np.random.default_rng(cfg.rng_seed + 404)
    x_by_regime: dict[str, np.ndarray] = {}
    n_total_by_regime: dict[str, int] = {}
    n_used_by_regime: dict[str, int] = {}
    for regime, arr in x_by_regime_full.items():
        x_arr = np.asarray(arr, dtype=float)
        x_arr = x_arr[np.isfinite(x_arr) & (x_arr > 0.0)]
        n_total = int(len(x_arr))
        n_total_by_regime[regime] = n_total
        cap = int(max_samples_global) if regime == "global" else int(max_samples_regime)
        if cap > 0 and n_total > cap:
            idx = rng.choice(n_total, size=cap, replace=False)
            x_use = x_arr[idx]
        else:
            x_use = x_arr
        x_by_regime[regime] = np.asarray(x_use, dtype=float)
        n_used_by_regime[regime] = int(len(x_use))

    metrics_all: list[pd.DataFrame] = []
    best_rows: list[dict[str, float | bool | str]] = []
    best_map: dict[str, dict[str, float | bool] | None] = {}

    for regime, x_reg in x_by_regime.items():
        cand_df, best = _scan_xmin_candidates(x_reg, regime=regime, cfg=cfg)
        if not cand_df.empty:
            metrics_all.append(cand_df)
        if best is None:
            best_map[regime] = None
            best_rows.append(
                {
                    "regime": regime,
                    "n_total": int(n_total_by_regime[regime]),
                    "n_used": int(n_used_by_regime[regime]),
                    "fit_available": False,
                    "pass_all": False,
                }
            )
            continue

        xmin = float(best["xmin"])
        tail = np.asarray(x_reg, dtype=float)
        tail = tail[np.isfinite(tail) & (tail >= xmin)]
        alpha_lo, alpha_hi = _bootstrap_alpha_ci(
            tail,
            xmin,
            iters=cfg.alpha_bootstrap_iters,
            rng=rng,
        )

        row = dict(best)
        row["n_total"] = int(n_total_by_regime[regime])
        row["n_used"] = int(n_used_by_regime[regime])
        row["fit_available"] = True
        row["alpha_q025_boot"] = float(alpha_lo)
        row["alpha_q975_boot"] = float(alpha_hi)
        row["alpha_minus_alpha_pred"] = float(float(best["alpha_mle"]) - cfg.alpha_pred)
        best_rows.append(row)
        best_map[regime] = best

    metrics_df = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    best_df = pd.DataFrame(best_rows)
    metrics_df.to_csv(outdir / "experiment_F6b_panel_tail_metrics.csv", index=False)
    best_df.to_csv(outdir / "experiment_F6b_panel_best_fits.csv", index=False)

    global_best = best_map.get("global")
    if global_best is None:
        raise RuntimeError("Global fit unavailable for panel run.")

    _plot_global_tail(
        x=x_by_regime["global"],
        xmin=float(global_best["xmin"]),
        alpha=float(global_best["alpha_mle"]),
        out_path=outdir / "plot_F6b_panel_empirical_tail.png",
    )
    _plot_strata_compare(
        x_by_regime=x_by_regime,
        best_by_regime=best_map,
        out_path=outdir / "plot_F6b_panel_strata_tail_compare.png",
    )

    checks = {
        "dynamic_range_ge_10": bool(float(global_best["dynamic_range"]) >= cfg.min_dynamic_range),
        "llr_prefers_powerlaw": bool(
            (float(global_best["llr_pl_vs_exp"]) > 0.0)
            and np.isfinite(float(global_best["llr_p_value"]))
            and (float(global_best["llr_p_value"]) <= cfg.llr_p_max)
        ),
        "alpha_in_universality_band_1p3_2p0": bool(
            cfg.alpha_band_low <= float(global_best["alpha_mle"]) <= cfg.alpha_band_high
        ),
    }
    checks["pass_all"] = bool(all(checks.values()))

    conv = best_map.get("convective_fractal")
    calm = best_map.get("calm")
    if (conv is not None) and (calm is not None):
        extra = {
            "convective_alpha_leq_calm": bool(float(conv["alpha_mle"]) <= float(calm["alpha_mle"])),
            "convective_ks_leq_calm": bool(float(conv["ks_distance"]) <= float(calm["ks_distance"])),
        }
        extra["convective_heavier_or_better"] = bool(
            extra["convective_alpha_leq_calm"] or extra["convective_ks_leq_calm"]
        )
    else:
        extra = {"convective_heavier_or_better": False}

    verdict = {
        "checks": checks,
        "extra_regime_diagnostics": extra,
        "global_best_fit": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in global_best.items()},
        "inputs": {
            "input_nc": str(input_nc),
            "f5_dataset_csv": str(f5_dataset_csv),
            "m_timeseries_csv": str(m_timeseries_csv),
            "m_summary_csv": str(m_summary_csv),
            "m_mode_index_csv": str(m_mode_index_csv),
            "n_time": int(lambda_panel_abs.shape[0]),
            "n_space": int(lambda_panel_abs.shape[1]),
            "n_samples_global": int(lambda_panel_abs.size),
            "n_samples_global_used": int(n_used_by_regime["global"]),
            "percentile_scan": [float(cfg.pmin), float(cfg.pmax), int(cfg.pgrid)],
            "min_tail_points": int(cfg.min_tail_points),
            "alpha_pred_reference": float(cfg.alpha_pred),
            "scale_edges_km": [float(x) for x in scale_edges_km],
            "batch_size": int(batch_size),
            "max_samples_global": int(max_samples_global),
            "max_samples_regime": int(max_samples_regime),
        },
        "strata_counts": {
            "global_total": int(n_total_by_regime["global"]),
            "global_used": int(n_used_by_regime["global"]),
            "fractal_high_total": int(n_total_by_regime["fractal_high"]),
            "fractal_high_used": int(n_used_by_regime["fractal_high"]),
            "fractal_low_total": int(n_total_by_regime["fractal_low"]),
            "fractal_low_used": int(n_used_by_regime["fractal_low"]),
            "convective_fractal_total": int(n_total_by_regime["convective_fractal"]),
            "convective_fractal_used": int(n_used_by_regime["convective_fractal"]),
            "calm_total": int(n_total_by_regime["calm"]),
            "calm_used": int(n_used_by_regime["calm"]),
        },
    }
    (outdir / "experiment_F6b_panel_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    report_lines = [
        "# Experiment F6b-panel Report: Heavy Tails of |Lambda_local(t,y,x)|",
        "",
        "## Protocol",
        "- panel object is reconstructed exactly as in O spatial pipeline: Lambda_local(t,y,x)",
        "- x_min is selected strictly by minimum KS distance over percentile scan",
        "- alpha is estimated by continuous-tail MLE",
        "- model comparison uses LLR(Power-law vs Exponential)",
        f"- fitting sample caps: global={int(max_samples_global)}, regime={int(max_samples_regime)}",
        "",
        "## Global Best Fit",
        f"- xmin = {float(global_best['xmin']):.6e}",
        f"- alpha_emp = {float(global_best['alpha_mle']):.6f}",
        f"- alpha_pred(reference) = {cfg.alpha_pred:.6f}",
        f"- KS distance = {float(global_best['ks_distance']):.6f}",
        f"- dynamic range x_max/x_min = {float(global_best['dynamic_range']):.3f}",
        f"- n_tail = {int(global_best['n_tail'])}",
        f"- LLR(PL-Exp) = {float(global_best['llr_pl_vs_exp']):.6f}",
        f"- LLR p-value = {float(global_best['llr_p_value']):.6f}",
        "",
        "## Strict PASS Criteria",
        f"- dynamic_range_ge_10: {checks['dynamic_range_ge_10']}",
        f"- llr_prefers_powerlaw: {checks['llr_prefers_powerlaw']}",
        f"- alpha_in_universality_band_1p3_2p0: {checks['alpha_in_universality_band_1p3_2p0']}",
        f"- PASS_ALL: {checks['pass_all']}",
        "",
        "## Stratified Diagnostics",
        f"- convective_heavier_or_better: {extra['convective_heavier_or_better']}",
        "- per-regime best fits are saved in `experiment_F6b_panel_best_fits.csv`",
        "",
        "## Outputs",
        "- experiment_F6b_panel_tail_metrics.csv",
        "- experiment_F6b_panel_best_fits.csv",
        "- experiment_F6b_panel_verdict.json",
        "- plot_F6b_panel_empirical_tail.png",
        "- plot_F6b_panel_strata_tail_compare.png",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
    )
    parser.add_argument(
        "--f5-dataset-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_F5_lambda_struct_fractal_era5/experiment_F5_dataset.csv"),
    )
    parser.add_argument(
        "--m-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    parser.add_argument(
        "--m-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_summary.csv"),
    )
    parser.add_argument(
        "--m-mode-index-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_mode_index.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("clean_experiments/results/experiment_F6b_era5_heavy_tails_panel"),
    )
    parser.add_argument(
        "--scale-edges-km",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-samples-global", type=int, default=3_000_000)
    parser.add_argument("--max-samples-regime", type=int, default=1_200_000)
    parser.add_argument("--pmin", type=float, default=70.0)
    parser.add_argument("--pmax", type=float, default=99.0)
    parser.add_argument("--pgrid", type=int, default=120)
    parser.add_argument("--min-tail-points", type=int, default=80)
    parser.add_argument("--alpha-boot-iters", type=int, default=400)
    args = parser.parse_args()

    cfg = F6bConfig(
        pmin=float(args.pmin),
        pmax=float(args.pmax),
        pgrid=max(10, int(args.pgrid)),
        min_tail_points=max(40, int(args.min_tail_points)),
        alpha_bootstrap_iters=max(0, int(args.alpha_boot_iters)),
    )

    verdict = run_f6b_panel(
        input_nc=args.input_nc,
        f5_dataset_csv=args.f5_dataset_csv,
        m_timeseries_csv=args.m_timeseries_csv,
        m_summary_csv=args.m_summary_csv,
        m_mode_index_csv=args.m_mode_index_csv,
        outdir=args.out,
        scale_edges_km=[float(x) for x in args.scale_edges_km],
        batch_size=max(8, int(args.batch_size)),
        max_samples_global=max(200_000, int(args.max_samples_global)),
        max_samples_regime=max(100_000, int(args.max_samples_regime)),
        cfg=cfg,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
