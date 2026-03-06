#!/usr/bin/env python3
"""Experiment F6b: heavy-tail diagnostics of structural |Lambda| in ERA5/WPWP."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-15


@dataclass
class F6bConfig:
    pmin: float = 70.0
    pmax: float = 99.0
    pgrid: int = 120
    min_tail_points: int = 80

    min_dynamic_range: float = 10.0
    llr_p_max: float = 0.05
    alpha_band_low: float = 1.3
    alpha_band_high: float = 2.0
    alpha_pred: float = 1.54

    fractal_high_q: float = 0.70
    fractal_low_q: float = 0.30
    conv_high_q: float = 0.70
    calm_low_q: float = 0.30

    alpha_bootstrap_iters: int = 400
    rng_seed: int = 20260306


def _zscore(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x_arr))
    sd = float(np.nanstd(x_arr))
    if not np.isfinite(sd) or sd < 1e-14:
        return np.zeros_like(x_arr, dtype=float)
    return (x_arr - mu) / sd


def _safe_quantile(x: np.ndarray, q: float) -> float:
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    if len(x_arr) == 0:
        return float("nan")
    return float(np.quantile(x_arr, q))


def _alpha_mle_continuous(tail: np.ndarray, xmin: float) -> float:
    x = np.asarray(tail, dtype=float)
    log_sum = float(np.sum(np.log(x / (xmin + EPS))))
    if log_sum <= 0.0:
        return float("nan")
    return float(1.0 + len(x) / log_sum)


def _ks_distance_powerlaw(tail: np.ndarray, xmin: float, alpha: float) -> float:
    x = np.sort(np.asarray(tail, dtype=float))
    n = len(x)
    if n < 2 or not np.isfinite(alpha) or alpha <= 1.0:
        return float("nan")

    emp_cdf = np.arange(1, n + 1, dtype=float) / float(n)
    model_cdf = 1.0 - np.power(x / (xmin + EPS), 1.0 - alpha)
    return float(np.max(np.abs(emp_cdf - model_cdf)))


def _llr_powerlaw_vs_exponential(tail: np.ndarray, xmin: float, alpha: float) -> tuple[float, float]:
    x = np.asarray(tail, dtype=float)
    n = len(x)
    if n < 3 or not np.isfinite(alpha) or alpha <= 1.0:
        return float("nan"), float("nan")

    ll_pl = np.log(alpha - 1.0 + EPS) - np.log(xmin + EPS) - alpha * np.log(x / (xmin + EPS))

    shifted = x - xmin
    mean_shift = float(np.mean(shifted))
    lam = 1.0 / max(mean_shift, EPS)
    ll_exp = np.log(lam + EPS) - lam * shifted

    delta = ll_pl - ll_exp
    llr = float(np.sum(delta))

    sd = float(np.std(delta, ddof=1))
    if not np.isfinite(sd) or sd < 1e-14:
        return llr, float("nan")

    # Vuong-style asymptotic p-value for non-nested models.
    v_stat = llr / (sqrt(float(n)) * sd + EPS)
    p_value = float(erfc(abs(v_stat) / sqrt(2.0)))
    return llr, p_value


def _bootstrap_alpha_ci(
    tail: np.ndarray,
    xmin: float,
    *,
    iters: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    x = np.asarray(tail, dtype=float)
    n = len(x)
    if n < 20 or iters <= 0:
        return float("nan"), float("nan")

    draws = np.full(iters, np.nan, dtype=float)
    for i in range(iters):
        sample = x[rng.integers(0, n, size=n)]
        draws[i] = _alpha_mle_continuous(sample, xmin)

    good = draws[np.isfinite(draws)]
    if len(good) < 20:
        return float("nan"), float("nan")
    return float(np.quantile(good, 0.025)), float(np.quantile(good, 0.975))


def _scan_xmin_candidates(
    x: np.ndarray,
    regime: str,
    cfg: F6bConfig,
) -> tuple[pd.DataFrame, dict[str, float | bool] | None]:
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr) & (x_arr > 0.0)]
    if len(x_arr) < cfg.min_tail_points + 20:
        return pd.DataFrame(), None

    p_grid = np.linspace(cfg.pmin, cfg.pmax, max(2, int(cfg.pgrid)))
    rows: list[dict[str, float | int | bool | str]] = []

    for p in p_grid:
        xmin = float(np.percentile(x_arr, p))
        tail = x_arr[x_arr >= xmin]
        n_tail = int(len(tail))
        if n_tail < cfg.min_tail_points:
            continue

        alpha = _alpha_mle_continuous(tail, xmin)
        ks = _ks_distance_powerlaw(tail, xmin, alpha)
        dynamic_range = float(np.max(tail) / (xmin + EPS))
        llr, llr_p = _llr_powerlaw_vs_exponential(tail, xmin, alpha)

        rows.append(
            {
                "regime": regime,
                "percentile": float(p),
                "xmin": float(xmin),
                "n_tail": n_tail,
                "alpha_mle": float(alpha),
                "ks_distance": float(ks),
                "dynamic_range": float(dynamic_range),
                "llr_pl_vs_exp": float(llr),
                "llr_p_value": float(llr_p),
                "pass_dynamic_range": bool(dynamic_range >= cfg.min_dynamic_range),
                "pass_llr": bool(np.isfinite(llr_p) and (llr > 0.0) and (llr_p <= cfg.llr_p_max)),
                "pass_alpha_band": bool(cfg.alpha_band_low <= alpha <= cfg.alpha_band_high),
            }
        )

    cand_df = pd.DataFrame(rows)
    if cand_df.empty:
        return cand_df, None

    cand_df = cand_df.sort_values(["ks_distance", "percentile"], ascending=[True, True]).reset_index(drop=True)
    cand_df["is_best_ks"] = False
    cand_df.loc[0, "is_best_ks"] = True

    best = cand_df.iloc[0].to_dict()
    best["pass_all"] = bool(
        bool(best["pass_dynamic_range"]) and bool(best["pass_llr"]) and bool(best["pass_alpha_band"])
    )
    return cand_df, best


def _plot_global_tail(
    *,
    x: np.ndarray,
    xmin: float,
    alpha: float,
    out_path: Path,
    max_plot_points: int = 200_000,
    rng_seed: int = 20260306,
) -> None:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if len(vals) < 50 or not np.isfinite(xmin) or not np.isfinite(alpha):
        return

    rng = np.random.default_rng(rng_seed)
    if len(vals) > max_plot_points:
        idx = rng.choice(len(vals), size=max_plot_points, replace=False)
        vals_plot = vals[idx]
    else:
        vals_plot = vals

    xs = np.sort(vals_plot)
    n = len(xs)
    ccdf = np.arange(n, 0, -1, dtype=float) / float(n)

    tail = xs[xs >= xmin]
    n_tail = len(tail)
    if n_tail < 20:
        return
    ccdf_tail = np.arange(n_tail, 0, -1, dtype=float) / float(n_tail)

    x_fit = np.geomspace(float(xmin), float(np.max(tail)), 1000)
    fit = np.power(x_fit / (xmin + EPS), 1.0 - alpha)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.scatter(xs, ccdf, s=10, alpha=0.40, color="#4c78a8", label="Empirical CCDF")
    ax.scatter(tail, ccdf_tail, s=12, alpha=0.70, color="#f58518", label="Tail (x >= xmin)")
    ax.plot(x_fit, fit, color="#e45756", linewidth=2.0, label=f"Power-law MLE (alpha={alpha:.3f})")

    ax.axvline(xmin, color="#222222", linestyle="--", linewidth=1.5, label=f"xmin={xmin:.3e}")
    ax.axvspan(np.min(xs), xmin, color="#9e9e9e", alpha=0.15, label="Below xmin")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|Lambda_struct|")
    ax.set_ylabel("CCDF P(X >= x)")
    ax.set_title("F6b empirical tail of |Lambda_struct| (global)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_strata_compare(
    *,
    x_by_regime: dict[str, np.ndarray],
    best_by_regime: dict[str, dict[str, float | bool] | None],
    out_path: Path,
    max_plot_points: int = 160_000,
    rng_seed: int = 20260306,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    rng = np.random.default_rng(rng_seed + 17)
    palette = {
        "convective_fractal": "#d62728",
        "calm": "#1f77b4",
        "fractal_high": "#2ca02c",
        "fractal_low": "#9467bd",
    }
    drawn = 0
    for regime in ("convective_fractal", "calm", "fractal_high", "fractal_low"):
        x = x_by_regime.get(regime)
        best = best_by_regime.get(regime)
        if x is None or best is None:
            continue
        vals = np.asarray(x, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if len(vals) < 40:
            continue
        if len(vals) > max_plot_points:
            idx = rng.choice(len(vals), size=max_plot_points, replace=False)
            vals = vals[idx]

        xs = np.sort(vals)
        ccdf = np.arange(len(xs), 0, -1, dtype=float) / float(len(xs))
        alpha = float(best["alpha_mle"])
        label = f"{regime}: alpha={alpha:.3f}, ks={float(best['ks_distance']):.3f}"
        ax.plot(xs, ccdf, linewidth=1.6, alpha=0.9, color=palette.get(regime, None), label=label)
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|Lambda_struct|")
    ax.set_ylabel("CCDF P(X >= x)")
    ax.set_title("F6b strata comparison: convective/fractal vs calm")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_f6b(*, input_csv: Path, outdir: Path, cfg: F6bConfig) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.rng_seed)

    df = pd.read_csv(input_csv)
    if "lambda_abs" in df.columns:
        lambda_abs = np.asarray(df["lambda_abs"], dtype=float)
    elif "lambda_struct" in df.columns:
        lambda_abs = np.abs(np.asarray(df["lambda_struct"], dtype=float))
    else:
        raise KeyError("Expected `lambda_abs` or `lambda_struct` in input dataset.")

    df["lambda_abs_eval"] = lambda_abs
    finite_lambda = np.isfinite(df["lambda_abs_eval"].to_numpy(float)) & (df["lambda_abs_eval"].to_numpy(float) > 0.0)
    if int(np.sum(finite_lambda)) < cfg.min_tail_points + 20:
        raise ValueError("Not enough positive finite |Lambda| values for tail fitting.")

    # Fractal/convective stratification for optional regime diagnostics.
    if ("fractal_psd_beta" in df.columns) and ("fractal_variogram_slope" in df.columns):
        z_psd = _zscore(df["fractal_psd_beta"].to_numpy(float))
        z_var = _zscore(df["fractal_variogram_slope"].to_numpy(float))
        fractal_comp = 0.5 * (z_psd + z_var)
    else:
        fractal_comp = np.full(len(df), np.nan, dtype=float)

    if "residual_base_res0" in df.columns:
        residual_abs = np.abs(df["residual_base_res0"].to_numpy(float))
    else:
        residual_abs = np.full(len(df), np.nan, dtype=float)

    q_f_hi = _safe_quantile(fractal_comp, cfg.fractal_high_q)
    q_f_lo = _safe_quantile(fractal_comp, cfg.fractal_low_q)
    q_c_hi = _safe_quantile(residual_abs, cfg.conv_high_q)
    q_c_lo = _safe_quantile(residual_abs, cfg.calm_low_q)

    masks: dict[str, np.ndarray] = {
        "global": finite_lambda.copy(),
        "fractal_high": finite_lambda & np.isfinite(fractal_comp) & (fractal_comp >= q_f_hi),
        "fractal_low": finite_lambda & np.isfinite(fractal_comp) & (fractal_comp <= q_f_lo),
        "convective_fractal": finite_lambda
        & np.isfinite(fractal_comp)
        & np.isfinite(residual_abs)
        & (fractal_comp >= q_f_hi)
        & (residual_abs >= q_c_hi),
        "calm": finite_lambda
        & np.isfinite(fractal_comp)
        & np.isfinite(residual_abs)
        & (fractal_comp <= q_f_lo)
        & (residual_abs <= q_c_lo),
    }

    x_by_regime: dict[str, np.ndarray] = {}
    metrics_all: list[pd.DataFrame] = []
    best_rows: list[dict[str, float | bool | str]] = []
    best_map: dict[str, dict[str, float | bool] | None] = {}

    for regime, mask in masks.items():
        x_reg = df.loc[mask, "lambda_abs_eval"].to_numpy(float)
        x_by_regime[regime] = x_reg
        cand_df, best = _scan_xmin_candidates(x_reg, regime=regime, cfg=cfg)
        if not cand_df.empty:
            metrics_all.append(cand_df)

        if best is None:
            best_map[regime] = None
            best_rows.append(
                {
                    "regime": regime,
                    "n_total": int(np.sum(mask)),
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
        row["n_total"] = int(np.sum(mask))
        row["fit_available"] = True
        row["alpha_q025_boot"] = float(alpha_lo)
        row["alpha_q975_boot"] = float(alpha_hi)
        row["alpha_minus_alpha_pred"] = float(float(best["alpha_mle"]) - cfg.alpha_pred)
        best_rows.append(row)
        best_map[regime] = best

    metrics_df = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    best_df = pd.DataFrame(best_rows)

    metrics_path = outdir / "experiment_F6b_tail_metrics.csv"
    best_path = outdir / "experiment_F6b_best_fits.csv"
    metrics_df.to_csv(metrics_path, index=False)
    best_df.to_csv(best_path, index=False)

    global_best = best_map.get("global")
    if global_best is None:
        raise RuntimeError("Global fit unavailable; cannot evaluate F6b.")

    _plot_global_tail(
        x=x_by_regime["global"],
        xmin=float(global_best["xmin"]),
        alpha=float(global_best["alpha_mle"]),
        out_path=outdir / "plot_F6b_empirical_tail.png",
    )
    _plot_strata_compare(
        x_by_regime=x_by_regime,
        best_by_regime=best_map,
        out_path=outdir / "plot_F6b_strata_tail_compare.png",
    )

    # Global pass/fail criteria (strict, predeclared).
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

    # Additional (non-mandatory) regime expectation: convective/fractal should be heavier
    # (smaller alpha) and/or fit better (smaller KS) than calm.
    extra = {}
    conv = best_map.get("convective_fractal")
    calm = best_map.get("calm")
    if (conv is not None) and (calm is not None):
        extra["convective_alpha_leq_calm"] = bool(float(conv["alpha_mle"]) <= float(calm["alpha_mle"]))
        extra["convective_ks_leq_calm"] = bool(float(conv["ks_distance"]) <= float(calm["ks_distance"]))
        extra["convective_heavier_or_better"] = bool(
            extra["convective_alpha_leq_calm"] or extra["convective_ks_leq_calm"]
        )
    else:
        extra["convective_heavier_or_better"] = False

    verdict = {
        "checks": checks,
        "extra_regime_diagnostics": extra,
        "global_best_fit": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in global_best.items()},
        "inputs": {
            "input_csv": str(input_csv),
            "n_samples_global": int(np.sum(masks["global"])),
            "percentile_scan": [float(cfg.pmin), float(cfg.pmax), int(cfg.pgrid)],
            "min_tail_points": int(cfg.min_tail_points),
            "alpha_pred_reference": float(cfg.alpha_pred),
        },
        "strata_counts": {k: int(np.sum(v)) for k, v in masks.items()},
    }
    (outdir / "experiment_F6b_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    report_lines = [
        "# Experiment F6b Report: Heavy Tails of Structural |Lambda| in ERA5",
        "",
        "## Protocol",
        "- x_min is selected strictly by minimum KS distance (Clauset/Newman style scan).",
        "- alpha is estimated by continuous-tail MLE, not by log-log linear regression.",
        "- Tail model comparison uses LLR(Power-law vs Exponential) with Vuong-style p-value.",
        f"- Percentile scan: [{cfg.pmin:.1f}, {cfg.pmax:.1f}] with {cfg.pgrid} candidates.",
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
        "- Best fits by regime are saved in `experiment_F6b_best_fits.csv`.",
        "- Candidate scans are saved in `experiment_F6b_tail_metrics.csv`.",
        f"- convective_heavier_or_better: {extra['convective_heavier_or_better']}",
        "",
        "## Outputs",
        "- experiment_F6b_tail_metrics.csv",
        "- experiment_F6b_best_fits.csv",
        "- experiment_F6b_verdict.json",
        "- plot_F6b_empirical_tail.png",
        "- plot_F6b_strata_tail_compare.png",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="clean_experiments/results/experiment_F5_lambda_struct_fractal_era5/experiment_F5_dataset.csv",
        help="Input dataset with lambda_struct/lambda_abs and optional fractal surrogate columns.",
    )
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_F6b_era5_heavy_tails",
        help="Output directory.",
    )
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

    verdict = run_f6b(
        input_csv=Path(args.input_csv),
        outdir=Path(args.out),
        cfg=cfg,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
