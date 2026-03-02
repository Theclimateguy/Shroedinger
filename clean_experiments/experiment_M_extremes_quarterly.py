#!/usr/bin/env python3
"""Quarterly rolling-origin evaluation for Experiment M extreme calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from clean_experiments.experiment_M_cosmo_flow import _fit_ridge_scaled
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _fit_ridge_scaled  # type: ignore


def _compute_thresholds(df: pd.DataFrame, q_extreme: float) -> tuple[float, float]:
    v_thr = float(np.quantile(df["proxy_vorticity_abs_mean"].to_numpy(dtype=float), q_extreme))
    o_thr = float(np.quantile(df["proxy_omega_q_850_300"].to_numpy(dtype=float), q_extreme))
    return v_thr, o_thr


def _build_design(df: pd.DataFrame, *, model_name: str, v_thr: float, o_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctrl = df["n_density_ctrl_z"].to_numpy(dtype=float)
    lam_h = df["lambda_h"].to_numpy(dtype=float)
    lam_v = df["lambda_v"].to_numpy(dtype=float)
    y = df["residual_base_res0"].to_numpy(dtype=float)

    v = df["proxy_vorticity_abs_mean"].to_numpy(dtype=float)
    o = df["proxy_omega_q_850_300"].to_numpy(dtype=float)
    extreme = ((v >= v_thr) | (o >= o_thr)).astype(float)

    x_base = np.column_stack([ctrl])

    if model_name == "vertical_global":
        x_full = np.column_stack([ctrl, lam_v])
    elif model_name == "horizontal_global":
        x_full = np.column_stack([ctrl, lam_h])
    elif model_name == "vertical_regime":
        x_full = np.column_stack([ctrl, lam_v, extreme, ctrl * extreme, lam_v * extreme])
    elif model_name == "horizontal_regime":
        x_full = np.column_stack([ctrl, lam_h, extreme, ctrl * extreme, lam_h * extreme])
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    return y, x_base, x_full


def _gains(y: np.ndarray, yhat_base: np.ndarray, yhat_full: np.ndarray, extreme_mask: np.ndarray) -> tuple[float, float, float]:
    mae_base = float(np.mean(np.abs(y - yhat_base)))
    mae_full = float(np.mean(np.abs(y - yhat_full)))
    gain_all = float((mae_base - mae_full) / (mae_base + 1e-12))

    if int(extreme_mask.sum()) >= 8:
        y_e = y[extreme_mask]
        b_e = yhat_base[extreme_mask]
        f_e = yhat_full[extreme_mask]
        mae_be = float(np.mean(np.abs(y_e - b_e)))
        mae_fe = float(np.mean(np.abs(y_e - f_e)))
        gain_ext = float((mae_be - mae_fe) / (mae_be + 1e-12))
    else:
        gain_ext = np.nan

    non_mask = ~extreme_mask
    if int(non_mask.sum()) >= 8:
        y_n = y[non_mask]
        b_n = yhat_base[non_mask]
        f_n = yhat_full[non_mask]
        mae_bn = float(np.mean(np.abs(y_n - b_n)))
        mae_fn = float(np.mean(np.abs(y_n - f_n)))
        gain_non = float((mae_bn - mae_fn) / (mae_bn + 1e-12))
    else:
        gain_non = np.nan

    return gain_all, gain_ext, gain_non


def _block_bootstrap_gain(
    *,
    y: np.ndarray,
    yhat_base: np.ndarray,
    yhat_full: np.ndarray,
    block: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    starts = np.arange(0, n, block, dtype=int)
    gains = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        picks: list[np.ndarray] = []
        cur = 0
        while cur < n:
            s = int(rng.choice(starts))
            e = min(s + block, n)
            picks.append(np.arange(s, e, dtype=int))
            cur += e - s
        idx = np.concatenate(picks)[:n]
        y_i = y[idx]
        b_i = yhat_base[idx]
        f_i = yhat_full[idx]
        mae_b = float(np.mean(np.abs(y_i - b_i)))
        mae_f = float(np.mean(np.abs(y_i - f_i)))
        gains[i] = float((mae_b - mae_f) / (mae_b + 1e-12))
    return float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))


def _load_configs(selected_configs_path: Path) -> pd.DataFrame:
    if not selected_configs_path.exists():
        return pd.DataFrame(
            [
                {"model": "vertical_global", "q_extreme": 0.88, "ridge_alpha": 0.1},
                {"model": "horizontal_global", "q_extreme": 0.88, "ridge_alpha": 0.1},
                {"model": "vertical_regime", "q_extreme": 0.85, "ridge_alpha": 0.1},
                {"model": "horizontal_regime", "q_extreme": 0.85, "ridge_alpha": 0.1},
            ]
        )
    keep = ["model", "q_extreme", "ridge_alpha"]
    return pd.read_csv(selected_configs_path)[keep].copy()


def _plot_quarterly(df: pd.DataFrame, out_path: Path, value_col: str, title: str, ylabel: str) -> None:
    models = ["vertical_global", "horizontal_global", "vertical_regime", "horizontal_regime"]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    palette = {
        "vertical_global": "#4c78a8",
        "horizontal_global": "#72b7b2",
        "vertical_regime": "#f58518",
        "horizontal_regime": "#e45756",
    }
    quarter_order = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]
    x = np.arange(len(quarter_order), dtype=float)

    for model in models:
        d = df[df["model"] == model].copy()
        d = d.set_index("quarter").reindex(quarter_order).reset_index()
        y = d[value_col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=model, color=palette[model])

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x, quarter_order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ci(df: pd.DataFrame, out_path: Path) -> None:
    models = ["vertical_global", "horizontal_global", "vertical_regime", "horizontal_regime"]
    quarter_order = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]
    x_base = np.arange(len(quarter_order), dtype=float)
    width = 0.19
    offset = {
        "vertical_global": -1.5 * width,
        "horizontal_global": -0.5 * width,
        "vertical_regime": 0.5 * width,
        "horizontal_regime": 1.5 * width,
    }
    palette = {
        "vertical_global": "#4c78a8",
        "horizontal_global": "#72b7b2",
        "vertical_regime": "#f58518",
        "horizontal_regime": "#e45756",
    }

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for model in models:
        d = df[df["model"] == model].set_index("quarter").reindex(quarter_order).reset_index()
        y = d["gain_all"].to_numpy(dtype=float)
        lo = d["gain_ci95_lo"].to_numpy(dtype=float)
        hi = d["gain_ci95_hi"].to_numpy(dtype=float)
        x = x_base + offset[model]
        ax.bar(x, y, width=width, color=palette[model], alpha=0.9, label=model)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            elinewidth=1.0,
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x_base, quarter_order)
    ax.set_ylabel("Gain vs ctrl baseline")
    ax.set_title("Quarterly rolling-origin gain with CI (2019)")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_quarterly(
    *,
    dataset_path: Path,
    selected_configs_path: Path,
    outdir: Path,
    min_train_n: int,
    n_boot: int,
    seed: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(
        subset=[
            "time",
            "residual_base_res0",
            "n_density_ctrl_z",
            "lambda_h",
            "lambda_v",
            "proxy_vorticity_abs_mean",
            "proxy_omega_q_850_300",
        ]
    ).sort_values("time").reset_index(drop=True)

    configs = _load_configs(selected_configs_path)
    rows: list[dict[str, float | int | str]] = []

    quarter_starts = [
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2019-04-01"),
        pd.Timestamp("2019-07-01"),
        pd.Timestamp("2019-10-01"),
    ]
    quarter_ends = [
        pd.Timestamp("2019-04-01"),
        pd.Timestamp("2019-07-01"),
        pd.Timestamp("2019-10-01"),
        pd.Timestamp("2020-01-01"),
    ]
    quarter_labels = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]
    model_seed_offset = {
        "vertical_global": 11,
        "horizontal_global": 23,
        "vertical_regime": 37,
        "horizontal_regime": 53,
    }

    for _, cfg in configs.iterrows():
        model = str(cfg["model"])
        q_extreme = float(cfg["q_extreme"])
        ridge_alpha = float(cfg["ridge_alpha"])

        for qid, (q_start, q_end, q_label) in enumerate(zip(quarter_starts, quarter_ends, quarter_labels)):
            train_df = df[df["time"] < q_start].copy().reset_index(drop=True)
            test_df = df[(df["time"] >= q_start) & (df["time"] < q_end)].copy().reset_index(drop=True)
            if len(train_df) < min_train_n or test_df.empty:
                continue

            v_thr, o_thr = _compute_thresholds(train_df, q_extreme=q_extreme)
            y_tr, xb_tr, xf_tr = _build_design(train_df, model_name=model, v_thr=v_thr, o_thr=o_thr)
            y_te, xb_te, xf_te = _build_design(test_df, model_name=model, v_thr=v_thr, o_thr=o_thr)

            _, _, yhat_base = _fit_ridge_scaled(xb_tr, y_tr, xb_te, ridge_alpha)
            _, _, yhat_full = _fit_ridge_scaled(xf_tr, y_tr, xf_te, ridge_alpha)

            v_te = test_df["proxy_vorticity_abs_mean"].to_numpy(dtype=float)
            o_te = test_df["proxy_omega_q_850_300"].to_numpy(dtype=float)
            extreme_mask = (v_te >= v_thr) | (o_te >= o_thr)

            gain_all, gain_ext, gain_non = _gains(y_te, yhat_base, yhat_full, extreme_mask)
            ci_lo, ci_hi = _block_bootstrap_gain(
                y=y_te,
                yhat_base=yhat_base,
                yhat_full=yhat_full,
                block=24 if len(y_te) >= 120 else 12,
                n_boot=n_boot,
                seed=seed + 100 * qid + model_seed_offset.get(model, 0),
            )

            rows.append(
                {
                    "model": model,
                    "quarter": q_label,
                    "q_extreme": q_extreme,
                    "ridge_alpha": ridge_alpha,
                    "train_start": str(train_df["time"].min()),
                    "train_end": str(train_df["time"].max()),
                    "test_start": str(test_df["time"].min()),
                    "test_end": str(test_df["time"].max()),
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_test_extreme": int(extreme_mask.sum()),
                    "n_test_non_extreme": int((~extreme_mask).sum()),
                    "v_thr_train": v_thr,
                    "omegaq_thr_train": o_thr,
                    "gain_all": gain_all,
                    "gain_extreme": gain_ext,
                    "gain_non_extreme": gain_non,
                    "gain_ci95_lo": ci_lo,
                    "gain_ci95_hi": ci_hi,
                }
            )
            print(f"[quarterly] model={model} quarter={q_label} gain_all={gain_all:.6f}", flush=True)

    q_df = pd.DataFrame(rows).sort_values(["model", "quarter"]).reset_index(drop=True)
    q_df.to_csv(outdir / "quarterly_rolling_metrics.csv", index=False)

    summary = (
        q_df.groupby("model", as_index=False)
        .agg(
            quarters=("quarter", "count"),
            gain_all_mean=("gain_all", "mean"),
            gain_all_std=("gain_all", "std"),
            gain_extreme_mean=("gain_extreme", "mean"),
            gain_non_extreme_mean=("gain_non_extreme", "mean"),
            gain_min=("gain_all", "min"),
            gain_max=("gain_all", "max"),
        )
        .sort_values("gain_all_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(outdir / "quarterly_summary.csv", index=False)

    _plot_quarterly(
        q_df,
        out_path=outdir / "plot_quarterly_gain_all.png",
        value_col="gain_all",
        title="Quarterly Rolling-Origin Gain (All Points)",
        ylabel="Gain vs ctrl baseline",
    )
    _plot_quarterly(
        q_df,
        out_path=outdir / "plot_quarterly_gain_extreme.png",
        value_col="gain_extreme",
        title="Quarterly Rolling-Origin Gain (Extreme Subset)",
        ylabel="Extreme gain vs ctrl baseline",
    )
    _plot_quarterly(
        q_df,
        out_path=outdir / "plot_quarterly_gain_non_extreme.png",
        value_col="gain_non_extreme",
        title="Quarterly Rolling-Origin Gain (Non-Extreme Subset)",
        ylabel="Non-extreme gain vs ctrl baseline",
    )
    _plot_ci(q_df, out_path=outdir / "plot_quarterly_gain_ci.png")

    lines = [
        "# Quarterly Rolling-Origin Experiment (2019)",
        "",
        "Protocol:",
        "- For each quarter Q in 2019, train on all history before Q.",
        "- Thresholds for extreme mask are estimated on train only.",
        "- Evaluate gain on quarter holdout.",
        "",
        "## Summary by model",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['model']}: mean={r['gain_all_mean']:.6f}, std={r['gain_all_std']:.6f}, "
            f"ext_mean={r['gain_extreme_mean']:.6f}, non_ext_mean={r['gain_non_extreme_mean']:.6f}, "
            f"min={r['gain_min']:.6f}, max={r['gain_max']:.6f}"
        )
    (outdir / "quarterly_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[quarterly] done -> {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_horizontal_vertical_compare/comparison_dataset.csv"),
    )
    parser.add_argument(
        "--selected-configs",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_extremes_calibration/selected_configs.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_extremes_quarterly"),
    )
    parser.add_argument("--min-train-n", type=int, default=800)
    parser.add_argument("--n-boot", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260301)
    args = parser.parse_args()

    run_quarterly(
        dataset_path=args.dataset,
        selected_configs_path=args.selected_configs,
        outdir=args.outdir,
        min_train_n=args.min_train_n,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
