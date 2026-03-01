#!/usr/bin/env python3
"""Calibrate extreme-regime models with explicit overfitting controls."""

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

try:
    from clean_experiments.experiment_M_cosmo_flow import _blocked_splits, _fit_ridge_scaled
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _blocked_splits, _fit_ridge_scaled  # type: ignore


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    complexity: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "vertical_global": ModelSpec(
        name="vertical_global",
        description="ctrl + lambda_v",
        complexity=2,
    ),
    "horizontal_global": ModelSpec(
        name="horizontal_global",
        description="ctrl + lambda_h",
        complexity=2,
    ),
    "vertical_regime": ModelSpec(
        name="vertical_regime",
        description="ctrl + lambda_v + E + ctrl*E + lambda_v*E",
        complexity=5,
    ),
    "horizontal_regime": ModelSpec(
        name="horizontal_regime",
        description="ctrl + lambda_h + E + ctrl*E + lambda_h*E",
        complexity=5,
    ),
}


def _compute_thresholds(df: pd.DataFrame, q_extreme: float) -> tuple[float, float]:
    v_thr = float(np.quantile(df["proxy_vorticity_abs_mean"].to_numpy(dtype=float), q_extreme))
    o_thr = float(np.quantile(df["proxy_omega_q_850_300"].to_numpy(dtype=float), q_extreme))
    return v_thr, o_thr


def _build_design(df: pd.DataFrame, *, model_name: str, v_thr: float, o_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
        names = ["ctrl", "lambda_v"]
    elif model_name == "horizontal_global":
        x_full = np.column_stack([ctrl, lam_h])
        names = ["ctrl", "lambda_h"]
    elif model_name == "vertical_regime":
        x_full = np.column_stack([ctrl, lam_v, extreme, ctrl * extreme, lam_v * extreme])
        names = ["ctrl", "lambda_v", "extreme", "ctrl_x_extreme", "lambda_v_x_extreme"]
    elif model_name == "horizontal_regime":
        x_full = np.column_stack([ctrl, lam_h, extreme, ctrl * extreme, lam_h * extreme])
        names = ["ctrl", "lambda_h", "extreme", "ctrl_x_extreme", "lambda_h_x_extreme"]
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    return y, x_base, x_full, names


def _fold_metrics(
    *,
    y_true: np.ndarray,
    yhat_base: np.ndarray,
    yhat_full: np.ndarray,
    extreme_mask: np.ndarray,
) -> tuple[float, float, float]:
    mae_base = float(np.mean(np.abs(y_true - yhat_base)))
    mae_full = float(np.mean(np.abs(y_true - yhat_full)))
    gain_all = float((mae_base - mae_full) / (mae_base + 1e-12))

    if int(extreme_mask.sum()) >= 8:
        y_e = y_true[extreme_mask]
        b_e = yhat_base[extreme_mask]
        f_e = yhat_full[extreme_mask]
        mae_be = float(np.mean(np.abs(y_e - b_e)))
        mae_fe = float(np.mean(np.abs(y_e - f_e)))
        gain_ext = float((mae_be - mae_fe) / (mae_be + 1e-12))
    else:
        gain_ext = np.nan

    non_extreme_mask = ~extreme_mask
    if int(non_extreme_mask.sum()) >= 8:
        y_n = y_true[non_extreme_mask]
        b_n = yhat_base[non_extreme_mask]
        f_n = yhat_full[non_extreme_mask]
        mae_bn = float(np.mean(np.abs(y_n - b_n)))
        mae_fn = float(np.mean(np.abs(y_n - f_n)))
        gain_non = float((mae_bn - mae_fn) / (mae_bn + 1e-12))
    else:
        gain_non = np.nan

    return gain_all, gain_ext, gain_non


def _evaluate_cv_config(
    *,
    train_df: pd.DataFrame,
    model_name: str,
    q_extreme: float,
    ridge_alpha: float,
    n_folds: int,
    seed: int,
    extreme_weight: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    splits = _blocked_splits(len(train_df), n_folds=n_folds)
    rows: list[dict[str, float | int | str]] = []

    for fold_id, (tr_rel, va_rel) in enumerate(splits):
        df_tr = train_df.iloc[tr_rel].reset_index(drop=True)
        df_va = train_df.iloc[va_rel].reset_index(drop=True)

        v_thr, o_thr = _compute_thresholds(df_tr, q_extreme=q_extreme)
        y_tr, xb_tr, xf_tr, _ = _build_design(df_tr, model_name=model_name, v_thr=v_thr, o_thr=o_thr)
        y_va, xb_va, xf_va, _ = _build_design(df_va, model_name=model_name, v_thr=v_thr, o_thr=o_thr)

        _, _, yhat_base = _fit_ridge_scaled(xb_tr, y_tr, xb_va, ridge_alpha)
        _, _, yhat_full = _fit_ridge_scaled(xf_tr, y_tr, xf_va, ridge_alpha)

        v_va = df_va["proxy_vorticity_abs_mean"].to_numpy(dtype=float)
        o_va = df_va["proxy_omega_q_850_300"].to_numpy(dtype=float)
        extreme_mask = (v_va >= v_thr) | (o_va >= o_thr)

        gain_all, gain_ext, gain_non = _fold_metrics(
            y_true=y_va,
            yhat_base=yhat_base,
            yhat_full=yhat_full,
            extreme_mask=extreme_mask,
        )
        score = float(gain_all + extreme_weight * (gain_ext if np.isfinite(gain_ext) else 0.0))

        rows.append(
            {
                "fold_id": int(fold_id),
                "model": model_name,
                "q_extreme": float(q_extreme),
                "ridge_alpha": float(ridge_alpha),
                "n_train": int(len(df_tr)),
                "n_val": int(len(df_va)),
                "n_val_extreme": int(extreme_mask.sum()),
                "gain_all": gain_all,
                "gain_extreme": gain_ext,
                "gain_non_extreme": gain_non,
                "score": score,
            }
        )

    fold_df = pd.DataFrame(rows)
    stat = {
        "model": model_name,
        "q_extreme": float(q_extreme),
        "ridge_alpha": float(ridge_alpha),
        "cv_gain_all_median": float(np.nanmedian(fold_df["gain_all"])),
        "cv_gain_all_mean": float(np.nanmean(fold_df["gain_all"])),
        "cv_gain_extreme_median": float(np.nanmedian(fold_df["gain_extreme"])),
        "cv_gain_extreme_mean": float(np.nanmean(fold_df["gain_extreme"])),
        "cv_gain_non_extreme_median": float(np.nanmedian(fold_df["gain_non_extreme"])),
        "cv_score_mean": float(np.nanmean(fold_df["score"])),
        "cv_score_std": float(np.nanstd(fold_df["score"], ddof=1)),
        "cv_extreme_positive_frac": float(np.nanmean(fold_df["gain_extreme"] > 0.0)),
        "cv_overall_positive_frac": float(np.nanmean(fold_df["gain_all"] > 0.0)),
        "seed": int(seed),
    }
    return fold_df, stat


def _choose_config_one_se(df_stats: pd.DataFrame) -> pd.Series:
    if df_stats.empty:
        raise ValueError("No stats to choose from.")

    best_idx = int(df_stats["cv_score_mean"].idxmax())
    best_row = df_stats.loc[best_idx]
    n_folds = 6.0
    se = float(best_row["cv_score_std"] / np.sqrt(max(n_folds, 1.0)))
    threshold = float(best_row["cv_score_mean"] - se)

    eligible = df_stats[df_stats["cv_score_mean"] >= threshold].copy()
    if eligible.empty:
        return best_row

    eligible = eligible.sort_values(
        by=[
            "complexity",
            "ridge_alpha",
            "cv_score_mean",
            "cv_gain_all_median",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return eligible.iloc[0]


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
        pieces: list[np.ndarray] = []
        cur = 0
        while cur < n:
            s = int(rng.choice(starts))
            e = min(s + block, n)
            pieces.append(np.arange(s, e, dtype=int))
            cur += e - s
        idx = np.concatenate(pieces)[:n]
        yb = yhat_base[idx]
        yf = yhat_full[idx]
        yt = y[idx]
        mae_b = float(np.mean(np.abs(yt - yb)))
        mae_f = float(np.mean(np.abs(yt - yf)))
        gains[i] = float((mae_b - mae_f) / (mae_b + 1e-12))
    return float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))


def _fit_and_test(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    q_extreme: float,
    ridge_alpha: float,
    boot_n: int,
    seed: int,
) -> dict[str, float | str | int]:
    v_thr, o_thr = _compute_thresholds(train_df, q_extreme=q_extreme)
    y_tr, xb_tr, xf_tr, _ = _build_design(train_df, model_name=model_name, v_thr=v_thr, o_thr=o_thr)
    y_te, xb_te, xf_te, _ = _build_design(test_df, model_name=model_name, v_thr=v_thr, o_thr=o_thr)

    _, _, yhat_base = _fit_ridge_scaled(xb_tr, y_tr, xb_te, ridge_alpha)
    _, _, yhat_full = _fit_ridge_scaled(xf_tr, y_tr, xf_te, ridge_alpha)

    v_te = test_df["proxy_vorticity_abs_mean"].to_numpy(dtype=float)
    o_te = test_df["proxy_omega_q_850_300"].to_numpy(dtype=float)
    extreme_mask = (v_te >= v_thr) | (o_te >= o_thr)

    gain_all, gain_ext, gain_non = _fold_metrics(
        y_true=y_te,
        yhat_base=yhat_base,
        yhat_full=yhat_full,
        extreme_mask=extreme_mask,
    )
    ci_lo, ci_hi = _block_bootstrap_gain(
        y=y_te,
        yhat_base=yhat_base,
        yhat_full=yhat_full,
        block=24,
        n_boot=boot_n,
        seed=seed,
    )

    return {
        "model": model_name,
        "q_extreme": float(q_extreme),
        "ridge_alpha": float(ridge_alpha),
        "train_v_thr": float(v_thr),
        "train_omegaq_thr": float(o_thr),
        "n_test": int(len(test_df)),
        "n_test_extreme": int(extreme_mask.sum()),
        "n_test_non_extreme": int((~extreme_mask).sum()),
        "test_gain_all": gain_all,
        "test_gain_extreme": gain_ext,
        "test_gain_non_extreme": gain_non,
        "test_gain_ci95_lo": ci_lo,
        "test_gain_ci95_hi": ci_hi,
    }


def _plot_heatmap(stats_df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    d = stats_df[stats_df["model"] == model_name].copy()
    if d.empty:
        return
    piv = d.pivot_table(index="q_extreme", columns="ridge_alpha", values="cv_score_mean")
    piv = piv.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(piv.columns)), [f"{c:.0e}" for c in piv.columns], rotation=40, ha="right")
    ax.set_yticks(np.arange(len(piv.index)), [f"{q:.2f}" for q in piv.index])
    ax.set_xlabel("ridge_alpha")
    ax.set_ylabel("q_extreme")
    ax.set_title(f"CV score heatmap: {model_name}")
    fig.colorbar(im, ax=ax, label="cv_score_mean")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_test_gains(test_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = test_df.sort_values("test_gain_all", ascending=False).reset_index(drop=True)
    x = np.arange(len(plot_df), dtype=float)
    y = plot_df["test_gain_all"].to_numpy(dtype=float)
    lo = plot_df["test_gain_ci95_lo"].to_numpy(dtype=float)
    hi = plot_df["test_gain_ci95_hi"].to_numpy(dtype=float)
    err_low = np.maximum(0.0, y - lo)
    err_high = np.maximum(0.0, hi - y)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(x, y, color="#4c78a8", alpha=0.9)
    ax.errorbar(x, y, yerr=np.vstack([err_low, err_high]), fmt="none", ecolor="#222222", capsize=4, elinewidth=1.1)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x, plot_df["model"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Test gain (2019) vs ctrl baseline")
    ax.set_title("Out-of-time test gains with block-bootstrap CI")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_regime_gains(test_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = test_df.copy()
    models = plot_df["model"].tolist()
    x = np.arange(len(models), dtype=float)
    width = 0.23
    g_all = plot_df["test_gain_all"].to_numpy(dtype=float)
    g_ext = plot_df["test_gain_extreme"].to_numpy(dtype=float)
    g_non = plot_df["test_gain_non_extreme"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.bar(x - width, g_all, width=width, color="#4c78a8", label="all")
    ax.bar(x, g_ext, width=width, color="#e45756", label="extreme")
    ax.bar(x + width, g_non, width=width, color="#54a24b", label="non-extreme")
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("Test gain vs ctrl baseline")
    ax.set_title("Out-of-time gains by regime (2019)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_report(
    *,
    outdir: Path,
    selected_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_n: int,
    test_n: int,
) -> None:
    lines = [
        "# Experiment M Extreme Calibration (Anti-overfitting)",
        "",
        "## Protocol",
        "- Train/calibration: 2017-2018 only",
        "- Test: 2019 only (out-of-time)",
        "- Tuning: blocked CV on train; one-SE model selection (simpler + stronger regularization preferred)",
        "",
        f"- n_train={train_n}, n_test={test_n}",
        "",
        "## Selected Configs",
    ]
    for _, row in selected_df.iterrows():
        lines.append(
            f"- {row['model']}: q={row['q_extreme']:.2f}, alpha={row['ridge_alpha']:.0e}, "
            f"cv_score_mean={row['cv_score_mean']:.6f}, cv_gain_extreme_median={row['cv_gain_extreme_median']:.6f}"
        )

    lines.append("")
    lines.append("## Test (2019)")
    for _, row in test_df.sort_values("test_gain_all", ascending=False).iterrows():
        lines.append(
            f"- {row['model']}: gain_all={row['test_gain_all']:.6f} "
            f"[{row['test_gain_ci95_lo']:.6f}, {row['test_gain_ci95_hi']:.6f}], "
            f"gain_extreme={row['test_gain_extreme']:.6f}, "
            f"gain_non_extreme={row['test_gain_non_extreme']:.6f}, "
            f"n_extreme={int(row['n_test_extreme'])}"
        )

    (outdir / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration(
    *,
    dataset_path: Path,
    outdir: Path,
    q_grid: list[float],
    ridge_grid: list[float],
    n_folds: int,
    extreme_weight: float,
    boot_n: int,
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
    ).reset_index(drop=True)

    train_df = df[df["time"].dt.year <= 2018].copy().reset_index(drop=True)
    test_df = df[df["time"].dt.year == 2019].copy().reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("Expected split by years: train=2017-2018, test=2019.")

    fold_frames: list[pd.DataFrame] = []
    stat_rows: list[dict[str, float | str | int]] = []

    for model_name, spec in MODEL_SPECS.items():
        print(f"[cal] model={model_name}", flush=True)
        for q in q_grid:
            for alpha in ridge_grid:
                fold_df, stat = _evaluate_cv_config(
                    train_df=train_df,
                    model_name=model_name,
                    q_extreme=q,
                    ridge_alpha=alpha,
                    n_folds=n_folds,
                    seed=seed,
                    extreme_weight=extreme_weight,
                )
                fold_frames.append(fold_df)
                stat["complexity"] = int(spec.complexity)
                stat_rows.append(stat)

    folds_all = pd.concat(fold_frames, ignore_index=True)
    stats_df = pd.DataFrame(stat_rows)
    folds_all.to_csv(outdir / "calibration_folds.csv", index=False)
    stats_df.to_csv(outdir / "calibration_grid_stats.csv", index=False)

    selected_rows: list[pd.Series] = []
    test_rows: list[dict[str, float | str | int]] = []
    for model_name in MODEL_SPECS:
        sub_stats = stats_df[stats_df["model"] == model_name].copy().reset_index(drop=True)
        selected = _choose_config_one_se(sub_stats)
        selected_rows.append(selected)
        test_row = _fit_and_test(
            train_df=train_df,
            test_df=test_df,
            model_name=model_name,
            q_extreme=float(selected["q_extreme"]),
            ridge_alpha=float(selected["ridge_alpha"]),
            boot_n=boot_n,
            seed=seed + int(100 * len(test_rows)),
        )
        test_rows.append(test_row)

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    test_out_df = pd.DataFrame(test_rows).reset_index(drop=True)
    selected_df.to_csv(outdir / "selected_configs.csv", index=False)
    test_out_df.to_csv(outdir / "test_out_of_time_metrics.csv", index=False)

    for model_name in MODEL_SPECS:
        _plot_heatmap(stats_df, model_name=model_name, out_path=outdir / f"plot_cv_heatmap_{model_name}.png")
    _plot_test_gains(test_out_df, out_path=outdir / "plot_test_gains.png")
    _plot_regime_gains(test_out_df, out_path=outdir / "plot_test_regime_gains.png")

    meta = {
        "q_grid": q_grid,
        "ridge_grid": ridge_grid,
        "n_folds": n_folds,
        "extreme_weight": extreme_weight,
        "seed": seed,
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
    }
    with (outdir / "calibration_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _save_report(
        outdir=outdir,
        selected_df=selected_df,
        test_df=test_out_df,
        train_n=len(train_df),
        test_n=len(test_df),
    )
    print(f"[cal] done -> {outdir}")


def _parse_float_list(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected non-empty comma-separated list")
    return [float(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_horizontal_vertical_compare/comparison_dataset.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_extremes_calibration"),
    )
    parser.add_argument("--q-grid", type=str, default="0.85,0.88,0.90,0.92,0.95")
    parser.add_argument("--ridge-grid", type=str, default="1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--extreme-weight", type=float, default=0.6)
    parser.add_argument("--boot-n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260301)
    args = parser.parse_args()

    run_calibration(
        dataset_path=args.dataset,
        outdir=args.outdir,
        q_grid=_parse_float_list(args.q_grid),
        ridge_grid=_parse_float_list(args.ridge_grid),
        n_folds=args.n_folds,
        extreme_weight=args.extreme_weight,
        boot_n=args.boot_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
