#!/usr/bin/env python3
"""Extreme-event and amplitude analysis for Experiment M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from clean_experiments.experiment_M_cosmo_flow import _blocked_splits, _evaluate_splits, _permutation_test
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _blocked_splits, _evaluate_splits, _permutation_test  # type: ignore


def _pick_folds(n: int) -> int:
    if n >= 320:
        return 6
    if n >= 160:
        return 5
    if n >= 80:
        return 4
    return 3


def _pick_perm_block(n: int) -> int:
    if n >= 600:
        return 24
    if n >= 200:
        return 12
    return 8


def _bootstrap_mean_delta(high: np.ndarray, low: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(high) < 3 or len(low) < 3:
        return np.nan, np.nan
    boot = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        hi = rng.choice(high, size=len(high), replace=True)
        lo = rng.choice(low, size=len(low), replace=True)
        boot[i] = float(np.mean(hi) - np.mean(lo))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _evaluate_models_on_subset(
    *,
    subset_name: str,
    sub: pd.DataFrame,
    ridge_alpha: float,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = sub["residual_base_res0"].to_numpy(dtype=float)
    ctrl = sub["n_density_ctrl_z"].to_numpy(dtype=float)
    lam_h = sub["lambda_h"].to_numpy(dtype=float)
    lam_v = sub["lambda_v"].to_numpy(dtype=float)

    n = len(sub)
    n_folds = _pick_folds(n)
    perm_block = _pick_perm_block(n)
    splits = _blocked_splits(n, n_folds=n_folds)
    x_base = np.column_stack([ctrl])

    model_specs: list[tuple[str, np.ndarray, list[str], np.ndarray]] = [
        ("horizontal", np.column_stack([ctrl, lam_h]), ["ctrl", "lambda_h"], np.array([1], dtype=int)),
        ("vertical", np.column_stack([ctrl, lam_v]), ["ctrl", "lambda_v"], np.array([1], dtype=int)),
        ("combined", np.column_stack([ctrl, lam_h, lam_v]), ["ctrl", "lambda_h", "lambda_v"], np.array([1, 2], dtype=int)),
    ]

    rows: list[dict[str, float | int | str]] = []
    split_frames: list[pd.DataFrame] = []
    perm_frames: list[pd.DataFrame] = []

    for model_name, x_full, feature_names, permute_cols in model_specs:
        split_df, yhat_b, yhat_f = _evaluate_splits(
            y=y,
            x_base=x_base,
            x_full=x_full,
            base_feature_names=["ctrl"],
            full_feature_names=feature_names,
            splits=splits,
            ridge_alpha=ridge_alpha,
        )
        mae_b = float(np.mean(np.abs(y - yhat_b)))
        mae_f = float(np.mean(np.abs(y - yhat_f)))
        gain = float((mae_b - mae_f) / (mae_b + 1e-12))

        p_perm, perm_df, stat_real = _permutation_test(
            y=y,
            x_base=x_base,
            x_full=x_full,
            base_feature_names=["ctrl"],
            full_feature_names=feature_names,
            permute_cols=permute_cols,
            splits=splits,
            ridge_alpha=ridge_alpha,
            n_perm=n_perm,
            perm_block=perm_block,
            seed=seed,
        )

        split_df = split_df.copy()
        split_df.insert(0, "subset", subset_name)
        split_df.insert(1, "model", model_name)
        split_frames.append(split_df)

        perm_df = perm_df.copy()
        perm_df.insert(0, "subset", subset_name)
        perm_df.insert(1, "model", model_name)
        perm_frames.append(perm_df)

        rows.append(
            {
                "subset": subset_name,
                "model": model_name,
                "n_samples": int(n),
                "n_folds": int(n_folds),
                "perm_block": int(perm_block),
                "mae_base_oof": mae_b,
                "mae_model_oof": mae_f,
                "oof_gain_frac": gain,
                "split_gain_median": float(np.median(split_df["mae_gain_frac"])),
                "split_gain_q25": float(np.quantile(split_df["mae_gain_frac"], 0.25)),
                "split_gain_q75": float(np.quantile(split_df["mae_gain_frac"], 0.75)),
                "perm_stat_real_median_gain": stat_real,
                "perm_p_value": p_perm,
            }
        )

    return pd.DataFrame(rows), pd.concat(split_frames, ignore_index=True), pd.concat(perm_frames, ignore_index=True)


def _subset_table(df: pd.DataFrame, q_extreme: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    v = df["proxy_vorticity_abs_mean"].to_numpy(dtype=float)
    o = df["proxy_omega_q_850_300"].to_numpy(dtype=float)
    v_thr = float(np.quantile(v, q_extreme))
    o_thr = float(np.quantile(o, q_extreme))

    subsets = {
        "full": np.ones(len(df), dtype=bool),
        "vorticity_p90": v >= v_thr,
        "omegaq_p90": o >= o_thr,
        "union_p90": (v >= v_thr) | (o >= o_thr),
        "intersection_p90": (v >= v_thr) & (o >= o_thr),
        "non_union": ~((v >= v_thr) | (o >= o_thr)),
    }
    return subsets, {"vorticity_thr": v_thr, "omegaq_thr": o_thr}


def _amplitude_metrics(
    df: pd.DataFrame,
    subsets: dict[str, np.ndarray],
    q_tail: float,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_abs_lh = float(np.mean(np.abs(df["lambda_h"])))
    base_abs_lv = float(np.mean(np.abs(df["lambda_v"])))
    base_abs_y = float(np.mean(np.abs(df["residual_base_res0"])))

    summary_rows: list[dict[str, float | int | str]] = []
    tail_rows: list[dict[str, float | int | str]] = []

    for subset_idx, (subset_name, mask) in enumerate(subsets.items()):
        sub = df.loc[mask].copy()
        n = len(sub)
        if n < 20:
            continue

        y = sub["residual_base_res0"].to_numpy(dtype=float)
        lh = sub["lambda_h"].to_numpy(dtype=float)
        lv = sub["lambda_v"].to_numpy(dtype=float)

        abs_lh = float(np.mean(np.abs(lh)))
        abs_lv = float(np.mean(np.abs(lv)))
        abs_y = float(np.mean(np.abs(y)))
        corr_h = float(pearsonr(lh, y).statistic) if n > 2 else np.nan
        corr_v = float(pearsonr(lv, y).statistic) if n > 2 else np.nan

        summary_rows.append(
            {
                "subset": subset_name,
                "n_samples": int(n),
                "mean_abs_lambda_h": abs_lh,
                "mean_abs_lambda_h_ratio_to_full": abs_lh / (base_abs_lh + 1e-12),
                "mean_abs_lambda_v": abs_lv,
                "mean_abs_lambda_v_ratio_to_full": abs_lv / (base_abs_lv + 1e-12),
                "mean_abs_residual": abs_y,
                "mean_abs_residual_ratio_to_full": abs_y / (base_abs_y + 1e-12),
                "corr_lambda_h_residual": corr_h,
                "corr_lambda_v_residual": corr_v,
            }
        )

        for lam_idx, lam_name in enumerate(("lambda_h", "lambda_v")):
            lam = sub[lam_name].to_numpy(dtype=float)
            q_lo = float(np.quantile(lam, q_tail))
            q_hi = float(np.quantile(lam, 1.0 - q_tail))
            y_low = y[lam <= q_lo]
            y_high = y[lam >= q_hi]
            delta = float(np.mean(y_high) - np.mean(y_low))
            ci_seed = int(seed + 1000 * subset_idx + 100 * lam_idx)
            ci_lo, ci_hi = _bootstrap_mean_delta(y_high, y_low, n_boot=n_boot, seed=ci_seed)
            tail_rows.append(
                {
                    "subset": subset_name,
                    "lambda_name": lam_name,
                    "tail_q_low": q_tail,
                    "tail_q_high": 1.0 - q_tail,
                    "n_low": int(len(y_low)),
                    "n_high": int(len(y_high)),
                    "delta_mean_residual_high_minus_low": delta,
                    "abs_delta_mean_residual": abs(delta),
                    "delta_ci95_lo": ci_lo,
                    "delta_ci95_hi": ci_hi,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(tail_rows)


def _severity_deciles(df: pd.DataFrame) -> pd.DataFrame:
    rank_v = df["proxy_vorticity_abs_mean"].rank(pct=True).to_numpy(dtype=float)
    rank_o = df["proxy_omega_q_850_300"].rank(pct=True).to_numpy(dtype=float)
    severity = 0.5 * rank_v + 0.5 * rank_o

    work = df.copy()
    work["severity"] = severity
    work["severity_decile"] = pd.qcut(work["severity"], q=10, labels=False, duplicates="drop") + 1

    rows: list[dict[str, float | int]] = []
    for d in sorted(work["severity_decile"].dropna().unique().tolist()):
        sub = work[work["severity_decile"] == d]
        rows.append(
            {
                "severity_decile": int(d),
                "n": int(len(sub)),
                "mean_abs_residual": float(np.mean(np.abs(sub["residual_base_res0"]))),
                "mean_abs_lambda_h": float(np.mean(np.abs(sub["lambda_h"]))),
                "mean_abs_lambda_v": float(np.mean(np.abs(sub["lambda_v"]))),
                "mean_vorticity_proxy": float(np.mean(sub["proxy_vorticity_abs_mean"])),
                "mean_omegaq_proxy": float(np.mean(sub["proxy_omega_q_850_300"])),
            }
        )
    return pd.DataFrame(rows).sort_values("severity_decile").reset_index(drop=True)


def _plot_proxy_scatter(df: pd.DataFrame, thresholds: dict[str, float], out_path: Path) -> None:
    v = df["proxy_vorticity_abs_mean"].to_numpy(dtype=float)
    o = df["proxy_omega_q_850_300"].to_numpy(dtype=float)
    v_thr = thresholds["vorticity_thr"]
    o_thr = thresholds["omegaq_thr"]

    mask_union = (v >= v_thr) | (o >= o_thr)
    mask_inter = (v >= v_thr) & (o >= o_thr)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.scatter(v[~mask_union], o[~mask_union], s=10, alpha=0.25, color="#4c78a8", label="background")
    ax.scatter(v[mask_union], o[mask_union], s=15, alpha=0.45, color="#f58518", label="union P90")
    ax.scatter(v[mask_inter], o[mask_inter], s=18, alpha=0.75, color="#e45756", label="intersection P90")
    ax.axvline(v_thr, color="#333333", linestyle="--", linewidth=1.1)
    ax.axhline(o_thr, color="#333333", linestyle="--", linewidth=1.1)
    ax.set_xlabel("proxy_vorticity_abs_mean")
    ax.set_ylabel("proxy_omega_q_850_300")
    ax.set_title("Extreme-event mask in proxy space")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_subset_gains(model_df: pd.DataFrame, out_path: Path) -> None:
    plot_subsets = ["full", "vorticity_p90", "omegaq_p90", "union_p90", "intersection_p90", "non_union"]
    models = ["horizontal", "vertical", "combined"]
    width = 0.24
    x = np.arange(len(plot_subsets), dtype=float)
    palette = {"horizontal": "#4c78a8", "vertical": "#f58518", "combined": "#54a24b"}

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for i, model in enumerate(models):
        vals = []
        for subset in plot_subsets:
            sub = model_df[(model_df["subset"] == subset) & (model_df["model"] == model)]
            vals.append(float(sub["oof_gain_frac"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, color=palette[model], alpha=0.92, label=model)

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x, plot_subsets, rotation=20, ha="right")
    ax.set_ylabel("OOF gain fraction vs ctrl baseline")
    ax.set_title("Model gain across extreme subsets")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_tail_amplitude(tail_df: pd.DataFrame, out_path: Path) -> None:
    plot_subsets = ["full", "union_p90", "intersection_p90", "non_union"]
    labels = []
    vals = []
    errs_low = []
    errs_high = []
    colors = []

    cmap = {"lambda_h": "#4c78a8", "lambda_v": "#f58518"}

    for subset in plot_subsets:
        for lam in ("lambda_h", "lambda_v"):
            sub = tail_df[(tail_df["subset"] == subset) & (tail_df["lambda_name"] == lam)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            labels.append(f"{subset}\n{lam}")
            vals.append(float(row["abs_delta_mean_residual"]))
            lo = abs(float(row["delta_ci95_lo"]))
            hi = abs(float(row["delta_ci95_hi"]))
            errs_low.append(max(0.0, vals[-1] - min(lo, hi)))
            errs_high.append(max(0.0, max(lo, hi) - vals[-1]))
            colors.append(cmap[lam])

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    ax.bar(x, vals, color=colors, alpha=0.9)
    ax.errorbar(x, vals, yerr=np.vstack([errs_low, errs_high]), fmt="none", ecolor="#222222", capsize=4, elinewidth=1.0)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("|delta residual| between top/bottom lambda tails")
    ax.set_title("Tail-response amplitude by regime")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_severity_deciles(decile_df: pd.DataFrame, out_path: Path) -> None:
    d = decile_df.sort_values("severity_decile").reset_index(drop=True)
    x = d["severity_decile"].to_numpy(dtype=int)

    y_res = d["mean_abs_residual"].to_numpy(dtype=float)
    y_h = d["mean_abs_lambda_h"].to_numpy(dtype=float)
    y_v = d["mean_abs_lambda_v"].to_numpy(dtype=float)

    y_res_n = y_res / (np.mean(y_res) + 1e-12)
    y_h_n = y_h / (np.mean(y_h) + 1e-12)
    y_v_n = y_v / (np.mean(y_v) + 1e-12)

    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    ax.plot(x, y_res_n, marker="o", color="#e45756", label="|residual| (norm.)")
    ax.plot(x, y_h_n, marker="o", color="#4c78a8", label="|lambda_h| (norm.)")
    ax.plot(x, y_v_n, marker="o", color="#f58518", label="|lambda_v| (norm.)")
    ax.axhline(1.0, color="#444444", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Severity decile (combined vorticity/omega*q)")
    ax.set_ylabel("Normalized mean amplitude")
    ax.set_title("Amplitude trend across severity deciles")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_report(
    *,
    outdir: Path,
    model_df: pd.DataFrame,
    amp_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    key_subsets = ["full", "union_p90", "intersection_p90", "non_union"]
    lines = [
        "# Experiment M Extreme-Event Amplitude Analysis",
        "",
        "## Thresholds",
        f"- vorticity P90 threshold: {thresholds['vorticity_thr']:.10g}",
        f"- omega*q P90 threshold: {thresholds['omegaq_thr']:.10g}",
        "",
        "## Model gain summary",
    ]

    for subset in key_subsets:
        sub = model_df[model_df["subset"] == subset]
        if sub.empty:
            continue
        lines.append(f"- {subset}:")
        for _, row in sub.sort_values("oof_gain_frac", ascending=False).iterrows():
            lines.append(
                f"  {row['model']}: gain={row['oof_gain_frac']:.6f}, "
                f"p={row['perm_p_value']:.4f}, n={int(row['n_samples'])}"
            )

    lines.append("")
    lines.append("## Amplitude summary")
    for subset in key_subsets:
        sub = amp_df[amp_df["subset"] == subset]
        if sub.empty:
            continue
        row = sub.iloc[0]
        lines.append(
            f"- {subset}: |residual| ratio={row['mean_abs_residual_ratio_to_full']:.3f}, "
            f"|lambda_h| ratio={row['mean_abs_lambda_h_ratio_to_full']:.3f}, "
            f"|lambda_v| ratio={row['mean_abs_lambda_v_ratio_to_full']:.3f}"
        )

    lines.append("")
    lines.append("## Tail response amplitude")
    for subset in key_subsets:
        for lam in ("lambda_h", "lambda_v"):
            sub = tail_df[(tail_df["subset"] == subset) & (tail_df["lambda_name"] == lam)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            lines.append(
                f"- {subset} / {lam}: |delta|={row['abs_delta_mean_residual']:.6f}, "
                f"CI95=[{row['delta_ci95_lo']:.6f}, {row['delta_ci95_hi']:.6f}]"
            )

    (outdir / "extreme_amplitude_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_extreme_analysis(
    *,
    comparison_dataset: Path,
    outdir: Path,
    q_extreme: float,
    q_tail: float,
    ridge_alpha: float,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(comparison_dataset)

    required = [
        "residual_base_res0",
        "n_density_ctrl_z",
        "lambda_h",
        "lambda_v",
        "proxy_vorticity_abs_mean",
        "proxy_omega_q_850_300",
    ]
    mask = np.ones(len(df), dtype=bool)
    for c in required:
        mask &= np.isfinite(df[c].to_numpy(dtype=float))
    df = df.loc[mask].reset_index(drop=True)

    subsets, thresholds = _subset_table(df, q_extreme=q_extreme)
    with (outdir / "thresholds.json").open("w", encoding="utf-8") as f:
        json.dump({"q_extreme": q_extreme, "q_tail": q_tail, **thresholds}, f, indent=2)

    model_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    perm_frames: list[pd.DataFrame] = []

    for subset_name, subset_mask in subsets.items():
        sub = df.loc[subset_mask].reset_index(drop=True)
        if len(sub) < 36:
            continue
        print(f"[extremes] subset={subset_name} n={len(sub)}", flush=True)
        mdf, sdf, pdf = _evaluate_models_on_subset(
            subset_name=subset_name,
            sub=sub,
            ridge_alpha=ridge_alpha,
            n_perm=n_perm,
            seed=seed,
        )
        model_frames.append(mdf)
        split_frames.append(sdf)
        perm_frames.append(pdf)

    model_df = pd.concat(model_frames, ignore_index=True)
    split_df = pd.concat(split_frames, ignore_index=True)
    perm_df = pd.concat(perm_frames, ignore_index=True)

    amp_df, tail_df = _amplitude_metrics(df, subsets=subsets, q_tail=q_tail, n_boot=n_boot, seed=seed)
    decile_df = _severity_deciles(df)

    model_df.to_csv(outdir / "extreme_model_comparison.csv", index=False)
    split_df.to_csv(outdir / "extreme_model_splits.csv", index=False)
    perm_df.to_csv(outdir / "extreme_model_permutations.csv", index=False)
    amp_df.to_csv(outdir / "extreme_amplitude_summary.csv", index=False)
    tail_df.to_csv(outdir / "extreme_tail_response.csv", index=False)
    decile_df.to_csv(outdir / "extreme_severity_deciles.csv", index=False)

    _plot_proxy_scatter(df, thresholds=thresholds, out_path=outdir / "plot_extreme_proxy_scatter.png")
    _plot_subset_gains(model_df, out_path=outdir / "plot_extreme_model_gains.png")
    _plot_tail_amplitude(tail_df, out_path=outdir / "plot_extreme_tail_amplitude.png")
    _plot_severity_deciles(decile_df, out_path=outdir / "plot_extreme_severity_deciles.png")

    _save_report(outdir=outdir, model_df=model_df, amp_df=amp_df, tail_df=tail_df, thresholds=thresholds)
    print(f"[extremes] done -> {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dataset",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_horizontal_vertical_compare/comparison_dataset.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_extremes_amplitude"),
    )
    parser.add_argument("--q-extreme", type=float, default=0.9)
    parser.add_argument("--q-tail", type=float, default=0.1)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--n-perm", type=int, default=320)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260301)
    args = parser.parse_args()

    run_extreme_analysis(
        comparison_dataset=args.comparison_dataset,
        outdir=args.outdir,
        q_extreme=args.q_extreme,
        q_tail=args.q_tail,
        ridge_alpha=args.ridge_alpha,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
