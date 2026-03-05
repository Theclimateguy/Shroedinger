#!/usr/bin/env python3
"""Experiment F6: SOC avalanche statistics of vertical-flow coherence drops."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-15


@dataclass
class F6Config:
    n_mu: int = 128
    n_steps: int = 180_000
    burn_in: int = 3_000
    n_seeds: int = 8

    drive_amp: float = 0.020
    drive_noise: float = 0.10
    p_scale: float = 1.40
    cap_power: float = 2.20
    transfer_fraction: float = 0.96

    residual_coeff: float = 0.15
    fallback_release_frac: float = 0.55
    relax_coeff: float = 4e-4

    max_topplings: int = 80_000

    fit_qmin: float = 0.03
    fit_qmax: float = 1.0
    fit_bins: int = 55
    fit_min_bin_count: int = 8
    fit_min_points: int = 300

    r2_threshold: float = 0.95
    min_dynamic_range: float = 100.0
    alpha_rel_error_tol: float = 0.10
    epsilon_star_window: tuple[float, float] = (0.7, 1.4)

    failed_fit_r2_floor: float = 0.50
    max_failed_seed_share: float = 0.35


def _balance(epsilon: float) -> float:
    e = float(epsilon)
    return float(2.0 * np.sqrt(e) / (1.0 + e + EPS))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[m]
    y_arr = y_arr[m]
    if len(x_arr) < 3:
        return float("nan")
    if float(np.std(x_arr)) < 1e-14 or float(np.std(y_arr)) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    design = np.column_stack([x_arr, np.ones_like(x_arr)])
    slope, intercept = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    y_hat = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + EPS))
    return float(slope), float(intercept), r2


def _load_epsilon_inputs(
    f1_summary_csv: Path,
    f2_summary_csv: Path,
    eps_override: tuple[float, ...] | None,
) -> tuple[np.ndarray, float, float]:
    f1 = pd.read_csv(f1_summary_csv).sort_values("epsilon").reset_index(drop=True)
    f2 = pd.read_csv(f2_summary_csv).sort_values("epsilon").reset_index(drop=True)

    if f1.empty:
        raise ValueError(f"Empty F1 summary: {f1_summary_csv}")
    if f2.empty:
        raise ValueError(f"Empty F2 summary: {f2_summary_csv}")

    eps_star = float(f1.loc[f1["vertical_flow_abs_mean"].idxmin(), "epsilon"])

    idx = int((f2["epsilon"] - eps_star).abs().idxmin())
    y_rel = float(f2.iloc[idx]["delta_predicted_trace"])
    if not np.isfinite(y_rel) or y_rel <= 0:
        raise ValueError(f"Invalid y_rel from F2 at epsilon≈{eps_star}: {y_rel}")

    if eps_override is not None:
        eps_arr = np.asarray(sorted(set(float(x) for x in eps_override)), dtype=float)
    else:
        eps_arr = np.asarray(sorted(set(float(x) for x in f1["epsilon"].to_numpy(float))), dtype=float)

    if len(eps_arr) < 1:
        raise ValueError("Need at least 1 epsilon point for F6 scan")

    return eps_arr, eps_star, y_rel


def _simulate_soc(
    epsilon: float,
    seed: int,
    cfg: F6Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    mu = np.linspace(0.0, 1.0, cfg.n_mu)
    weights = np.exp(-1.1 * mu)
    weights = weights / np.sum(weights)
    zc = 1.0 + 0.05 * np.sin(4.0 * np.pi * mu)

    load = np.zeros(cfg.n_mu, dtype=float)
    p_cap = float(0.999 * (_balance(epsilon) ** cfg.cap_power))

    rows: list[dict[str, float | int]] = []
    event_id = 0

    for step in range(cfg.n_steps):
        i = int(rng.integers(0, cfg.n_mu))
        load[i] += cfg.drive_amp * (1.0 + cfg.drive_noise * rng.normal())
        load = np.maximum(load, 0.0)

        unstable = np.where(load > zc)[0]
        if len(unstable) == 0:
            continue

        load_ratio = float(np.mean(load / zc))
        p_branch = float(min(p_cap, max(0.0, cfg.p_scale * load_ratio)))

        stack = unstable.tolist()
        in_stack = np.zeros(cfg.n_mu, dtype=bool)
        in_stack[unstable] = True

        total_topplings = 0
        total_release = 0.0
        total_diss = 0.0
        delta_lambda_coh = 0.0

        while stack:
            j = int(stack.pop())
            in_stack[j] = False
            if load[j] <= zc[j]:
                continue

            release = float(load[j] - cfg.residual_coeff * zc[j])
            if release <= 0.0:
                release = float(cfg.fallback_release_frac * zc[j])

            load[j] -= release
            total_release += release
            total_topplings += 1
            # Coherence-drop proxy from local threshold release at mu_j.
            delta_lambda_coh += float(weights[j] * (release / (zc[j] + EPS)))

            for nb in (j - 1, j + 1):
                if 0 <= nb < cfg.n_mu:
                    if rng.random() < 0.5 * p_branch:
                        transferred = cfg.transfer_fraction * release
                        load[nb] += transferred
                        total_diss += (1.0 - cfg.transfer_fraction) * release
                        if load[nb] > zc[nb] and not in_stack[nb]:
                            stack.append(nb)
                            in_stack[nb] = True
                    else:
                        total_diss += 0.5 * release
                else:
                    total_diss += 0.5 * release

            if total_topplings > cfg.max_topplings:
                break

        load = np.maximum(load, 0.0)
        load *= 1.0 - cfg.relax_coeff * (1.08 - _balance(epsilon))

        delta_lambda = max(delta_lambda_coh, 0.0)

        if step >= cfg.burn_in and delta_lambda > 1e-12:
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "seed": int(seed),
                    "event_id": int(event_id),
                    "step": int(step),
                    "delta_lambda": float(delta_lambda),
                    "topplings": int(total_topplings),
                    "total_release": float(total_release),
                    "total_dissipation": float(total_diss),
                    "p_branch": float(p_branch),
                    "load_ratio_pre": float(load_ratio),
                }
            )
            event_id += 1

    return pd.DataFrame(rows)


def _fit_powerlaw_tail(events: pd.DataFrame, cfg: F6Config) -> dict[str, float | int | bool]:
    vals = events["delta_lambda"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]

    if len(vals) < cfg.fit_min_points:
        return {
            "n_events": int(len(vals)),
            "n_tail": 0,
            "xmin": np.nan,
            "xmax": np.nan,
            "dynamic_range": np.nan,
            "alpha_measured": np.nan,
            "fit_r2": np.nan,
            "slope_pdf": np.nan,
            "fit_ok": False,
        }

    xmin = float(np.quantile(vals, cfg.fit_qmin))
    xmax = float(np.quantile(vals, cfg.fit_qmax))

    tail = vals[(vals >= xmin) & (vals <= xmax)]
    if len(tail) < cfg.fit_min_points:
        return {
            "n_events": int(len(vals)),
            "n_tail": int(len(tail)),
            "xmin": xmin,
            "xmax": xmax,
            "dynamic_range": np.nan,
            "alpha_measured": np.nan,
            "fit_r2": np.nan,
            "slope_pdf": np.nan,
            "fit_ok": False,
        }

    dynamic = float(np.max(tail) / (np.min(tail) + EPS))

    edges = np.geomspace(np.min(tail), np.max(tail), cfg.fit_bins)
    hist, _ = np.histogram(tail, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    bin_width = np.diff(edges)
    dens = hist / (np.sum(hist) * bin_width + EPS)

    mask = (hist >= cfg.fit_min_bin_count) & (dens > 0.0) & np.isfinite(dens)
    x = np.log(centers[mask])
    y = np.log(dens[mask])

    if len(x) < 10:
        return {
            "n_events": int(len(vals)),
            "n_tail": int(len(tail)),
            "xmin": xmin,
            "xmax": xmax,
            "dynamic_range": dynamic,
            "alpha_measured": np.nan,
            "fit_r2": np.nan,
            "slope_pdf": np.nan,
            "fit_ok": False,
        }

    slope, _, r2 = _linear_fit_r2(x, y)
    alpha = float(-slope)

    return {
        "n_events": int(len(vals)),
        "n_tail": int(len(tail)),
        "xmin": xmin,
        "xmax": xmax,
        "dynamic_range": dynamic,
        "alpha_measured": alpha,
        "fit_r2": float(r2),
        "slope_pdf": float(slope),
        "fit_ok": True,
    }


def _plot_star_tail(
    *,
    events_star: pd.DataFrame,
    alpha_measured: float,
    alpha_pred: float,
    out_path: Path,
    cfg: F6Config,
) -> None:
    vals = np.asarray(events_star["delta_lambda"], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if len(vals) < cfg.fit_min_points:
        return

    xmin = float(np.quantile(vals, cfg.fit_qmin))
    xmax = float(np.quantile(vals, cfg.fit_qmax))
    tail = vals[(vals >= xmin) & (vals <= xmax)]
    if len(tail) < cfg.fit_min_points:
        return

    edges = np.geomspace(np.min(tail), np.max(tail), 60)
    hist, _ = np.histogram(tail, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    dens = hist / (np.sum(hist) * widths + EPS)
    mask = (hist > 0) & np.isfinite(dens)

    x = centers[mask]
    y = dens[mask]
    xlog = np.log(x)

    # measured and predicted reference lines through geometric center.
    x0 = float(np.exp(np.mean(np.log(tail))))
    y0 = float(np.exp(np.mean(np.log(y))))
    y_measured = y0 * (x / x0) ** (-alpha_measured)
    y_pred = y0 * (x / x0) ** (-alpha_pred)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.scatter(x, y, s=18, alpha=0.75, color="#1f77b4", label="Tail bins")
    ax.plot(x, y_measured, color="#d62728", linewidth=2.0, label=f"fit: alpha={alpha_measured:.3f}")
    ax.plot(x, y_pred, color="#2ca02c", linewidth=1.8, linestyle="--", label=f"pred: alpha={alpha_pred:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avalanche size ΔLambda_coh")
    ax.set_ylabel("P(ΔLambda_coh)")
    ax.set_title("F6 SOC tail fit at epsilon* (log-log)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_alpha_scan(
    eps_summary: pd.DataFrame,
    alpha_pred: float,
    eps_star: float,
    out_path: Path,
) -> None:
    d = eps_summary.sort_values("epsilon").reset_index(drop=True)
    x = d["epsilon"].to_numpy(float)
    a = d["alpha_mean"].to_numpy(float)
    lo = d["alpha_q025"].to_numpy(float)
    hi = d["alpha_q975"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x, a, color="#1f77b4", marker="o", linewidth=1.8, label="alpha_measured (mean)")
    ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.20, label="seed q2.5%-q97.5%")
    ax.axhline(alpha_pred, color="#d62728", linestyle="--", linewidth=1.8, label=f"alpha_pred={alpha_pred:.3f}")
    ax.axvline(eps_star, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"epsilon*={eps_star:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("alpha")
    ax.set_title("F6 alpha scan vs epsilon")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_f6(
    *,
    outdir: Path,
    f1_summary_csv: Path,
    f2_summary_csv: Path,
    eps_override: tuple[float, ...] | None,
    cfg: F6Config,
) -> dict[str, object]:
    eps_arr, eps_star, y_rel = _load_epsilon_inputs(
        f1_summary_csv=f1_summary_csv,
        f2_summary_csv=f2_summary_csv,
        eps_override=eps_override,
    )
    alpha_pred = float(1.0 + 1.0 / y_rel)

    events_all: list[pd.DataFrame] = []
    fit_rows: list[dict[str, float | int | bool]] = []

    for eps in eps_arr:
        for seed in range(cfg.n_seeds):
            events = _simulate_soc(epsilon=float(eps), seed=int(seed), cfg=cfg)
            events_all.append(events)

            fit = _fit_powerlaw_tail(events=events, cfg=cfg)
            fit_rows.append(
                {
                    "epsilon": float(eps),
                    "seed": int(seed),
                    **fit,
                    "p_branch_mean": float(events["p_branch"].mean()) if len(events) else np.nan,
                    "p_branch_q95": float(events["p_branch"].quantile(0.95)) if len(events) else np.nan,
                    "load_ratio_mean": float(events["load_ratio_pre"].mean()) if len(events) else np.nan,
                }
            )

        print(f"[F6] finished epsilon={eps:.3g}", flush=True)

    events_df = pd.concat(events_all, ignore_index=True)
    fit_df = pd.DataFrame(fit_rows)

    fit_df["fit_failed"] = (
        (~fit_df["fit_ok"].astype(bool))
        | (~np.isfinite(fit_df["fit_r2"]))
        | (fit_df["fit_r2"].fillna(-np.inf) < cfg.failed_fit_r2_floor)
    )

    eps_summary = (
        fit_df.groupby("epsilon", as_index=False)
        .agg(
            n_runs=("seed", "count"),
            failed_runs=("fit_failed", "sum"),
            alpha_mean=("alpha_measured", "mean"),
            alpha_median=("alpha_measured", "median"),
            alpha_q025=("alpha_measured", lambda s: float(np.nanquantile(s, 0.025))),
            alpha_q975=("alpha_measured", lambda s: float(np.nanquantile(s, 0.975))),
            fit_r2_mean=("fit_r2", "mean"),
            fit_r2_median=("fit_r2", "median"),
            dynamic_range_median=("dynamic_range", "median"),
            dynamic_range_min=("dynamic_range", "min"),
            n_events_mean=("n_events", "mean"),
            n_tail_mean=("n_tail", "mean"),
            p_branch_mean=("p_branch_mean", "mean"),
            p_branch_q95_mean=("p_branch_q95", "mean"),
            load_ratio_mean=("load_ratio_mean", "mean"),
        )
        .sort_values("epsilon")
        .reset_index(drop=True)
    )
    eps_summary["failed_share"] = eps_summary["failed_runs"] / np.maximum(eps_summary["n_runs"], 1)

    idx_star = int((eps_summary["epsilon"] - eps_star).abs().idxmin())
    star_row = eps_summary.iloc[idx_star]
    eps_star_used = float(star_row["epsilon"])

    alpha_measured = float(star_row["alpha_mean"])
    alpha_rel_err = float(abs(alpha_measured - alpha_pred) / (abs(alpha_pred) + EPS))

    fit_df_star = fit_df[np.isclose(fit_df["epsilon"].to_numpy(float), eps_star_used)].copy()
    events_star = events_df[np.isclose(events_df["epsilon"].to_numpy(float), eps_star_used)].copy()

    alpha_corr_dynamic = _safe_corr(
        fit_df_star["alpha_measured"].to_numpy(float),
        fit_df_star["dynamic_range"].to_numpy(float),
    )

    checks = {
        "epsilon_star_in_window": bool(cfg.epsilon_star_window[0] <= eps_star_used <= cfg.epsilon_star_window[1]),
        "tail_r2_gt_threshold": bool(float(star_row["fit_r2_mean"]) > cfg.r2_threshold),
        "tail_dynamic_range_ge_100": bool(float(star_row["dynamic_range_median"]) >= cfg.min_dynamic_range),
        "alpha_match_within_10pct": bool(alpha_rel_err <= cfg.alpha_rel_error_tol),
        "failed_seed_share_below_limit": bool(float(star_row["failed_share"]) <= cfg.max_failed_seed_share),
    }
    checks["pass_all"] = bool(all(checks.values()))

    outdir.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(outdir / "experiment_F6_avalanche_events.csv", index=False)
    fit_df.to_csv(outdir / "experiment_F6_powerlaw_fits.csv", index=False)
    eps_summary.to_csv(outdir / "experiment_F6_epsilon_summary.csv", index=False)

    _plot_star_tail(
        events_star=events_star,
        alpha_measured=alpha_measured,
        alpha_pred=alpha_pred,
        out_path=outdir / "plot_F6_star_tail_fit.png",
        cfg=cfg,
    )
    _plot_alpha_scan(
        eps_summary=eps_summary,
        alpha_pred=alpha_pred,
        eps_star=eps_star_used,
        out_path=outdir / "plot_F6_alpha_scan.png",
    )

    verdict = {
        "checks": checks,
        "inputs": {
            "f1_summary_csv": str(f1_summary_csv),
            "f2_summary_csv": str(f2_summary_csv),
            "n_mu": int(cfg.n_mu),
            "n_steps": int(cfg.n_steps),
            "burn_in": int(cfg.burn_in),
            "n_seeds": int(cfg.n_seeds),
            "eps_list": [float(x) for x in eps_arr],
            "transfer_fraction": float(cfg.transfer_fraction),
            "cap_power": float(cfg.cap_power),
            "p_scale": float(cfg.p_scale),
        },
        "theory": {
            "epsilon_star_from_F1": float(eps_star),
            "epsilon_star_used": float(eps_star_used),
            "y_rel_from_F2": float(y_rel),
            "alpha_pred": float(alpha_pred),
        },
        "main_result": {
            "alpha_measured_mean": float(alpha_measured),
            "alpha_measured_q025": float(star_row["alpha_q025"]),
            "alpha_measured_q975": float(star_row["alpha_q975"]),
            "alpha_rel_error": float(alpha_rel_err),
            "fit_r2_mean": float(star_row["fit_r2_mean"]),
            "fit_r2_median": float(star_row["fit_r2_median"]),
            "dynamic_range_median": float(star_row["dynamic_range_median"]),
            "dynamic_range_min": float(star_row["dynamic_range_min"]),
            "failed_seed_share": float(star_row["failed_share"]),
            "n_events_mean": float(star_row["n_events_mean"]),
            "n_tail_mean": float(star_row["n_tail_mean"]),
        },
        "diagnostics": {
            "alpha_vs_dynamic_corr_at_star": float(alpha_corr_dynamic),
        },
    }

    with open(outdir / "experiment_F6_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Experiment F6 Report: SOC Avalanche Signature",
        "",
        "## Goal",
        "- Test SOC signature in toy open system with slow drive + threshold relaxation in mu-space.",
        "- Observable: avalanche sizes as coherence drops ΔLambda_coh.",
        "- Theory target: alpha_pred = 1 + 1/y_rel, with y_rel from F2 linearized RG matrix.",
        "",
        "## Theory Inputs",
        f"- epsilon* from F1: {eps_star:.3f}",
        f"- epsilon* used in F6 scan: {eps_star_used:.3f}",
        f"- y_rel from F2: {y_rel:.6f}",
        f"- alpha_pred = 1 + 1/y_rel = {alpha_pred:.6f}",
        "",
        "## SOC Parameters",
        f"- transfer_fraction = {cfg.transfer_fraction:.4f}",
        f"- cap_power = {cfg.cap_power:.4f}",
        f"- p_cap(epsilon) = 0.999 * balance(epsilon)^cap_power",
        f"- p_scale = {cfg.p_scale:.4f}",
        "",
        "## Main Fit at epsilon*",
        f"- alpha_measured (mean over seeds) = {alpha_measured:.6f}",
        f"- alpha 95% seed interval = [{float(star_row['alpha_q025']):.6f}, {float(star_row['alpha_q975']):.6f}]",
        f"- relative error |alpha_measured-alpha_pred|/alpha_pred = {100.0 * alpha_rel_err:.3f}%",
        f"- fit R2 mean = {float(star_row['fit_r2_mean']):.6f}",
        f"- tail dynamic range (median) = {float(star_row['dynamic_range_median']):.3f}",
        f"- failed seed share = {float(star_row['failed_share']):.3f}",
        "",
        "## Criteria",
        f"- epsilon_star_in_window: {checks['epsilon_star_in_window']}",
        f"- tail_r2_gt_threshold ({cfg.r2_threshold}): {checks['tail_r2_gt_threshold']}",
        f"- tail_dynamic_range_ge_100 ({cfg.min_dynamic_range}): {checks['tail_dynamic_range_ge_100']}",
        f"- alpha_match_within_10pct ({cfg.alpha_rel_error_tol}): {checks['alpha_match_within_10pct']}",
        f"- failed_seed_share_below_limit ({cfg.max_failed_seed_share}): {checks['failed_seed_share_below_limit']}",
        f"- PASS_ALL: {checks['pass_all']}",
        "",
        "## Files",
        "- experiment_F6_avalanche_events.csv",
        "- experiment_F6_powerlaw_fits.csv",
        "- experiment_F6_epsilon_summary.csv",
        "- plot_F6_star_tail_fit.png",
        "- plot_F6_alpha_scan.png",
        "- experiment_F6_verdict.json",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[F6] saved outputs to: {outdir.resolve()}", flush=True)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--f1-summary-csv",
        default="clean_experiments/results/experiment_F1_fractal_emergence/experiment_F1_summary.csv",
    )
    parser.add_argument(
        "--f2-summary-csv",
        default="clean_experiments/results/experiment_F2_scale_covariance/experiment_F2_summary.csv",
    )
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_F6_soc_avalanches",
    )
    parser.add_argument(
        "--eps-list",
        default=None,
        help="optional comma-separated epsilon list; default takes F1 eps grid",
    )

    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=180000)
    parser.add_argument("--burn-in", type=int, default=3000)
    parser.add_argument("--transfer-fraction", type=float, default=0.96)
    parser.add_argument("--cap-power", type=float, default=2.20)
    parser.add_argument("--p-scale", type=float, default=1.40)

    args = parser.parse_args()

    eps_override = None
    if args.eps_list is not None:
        vals = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]
        if vals:
            eps_override = tuple(vals)

    cfg = F6Config(
        n_seeds=max(2, int(args.n_seeds)),
        n_steps=max(20_000, int(args.n_steps)),
        burn_in=max(1000, int(args.burn_in)),
        transfer_fraction=float(args.transfer_fraction),
        cap_power=float(args.cap_power),
        p_scale=float(args.p_scale),
    )

    verdict = run_f6(
        outdir=Path(args.out),
        f1_summary_csv=Path(args.f1_summary_csv),
        f2_summary_csv=Path(args.f2_summary_csv),
        eps_override=eps_override,
        cfg=cfg,
    )

    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
