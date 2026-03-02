#!/usr/bin/env python3
"""Experiment M4: Lambda necessity falsification (S1/S2/S3)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        _blocked_splits,
        _evaluate_splits,
        _fit_ridge_scaled,
        _permutation_test,
        _strata_table,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        _blocked_splits,
        _evaluate_splits,
        _fit_ridge_scaled,
        _permutation_test,
        _strata_table,
    )


@dataclass
class ModelEval:
    mae_base_oof: float
    mae_full_oof: float
    rmse_full_oof: float
    oof_gain_frac: float
    perm_p_value: float
    beta_lambda: float
    positive_strata_frac: float
    yhat_base_oof: np.ndarray
    yhat_full_oof: np.ndarray


def _sorted_band_cols(columns: list[str], prefix: str) -> list[str]:
    out = [c for c in columns if c.startswith(prefix)]
    if not out:
        return []
    out = sorted(out, key=lambda c: int(c.split(prefix, 1)[1]))
    return out


def _require_columns(df: pd.DataFrame, cols: list[str], source: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {source}: {missing}")


def _load_meta(summary_csv: Path) -> tuple[float, float, float]:
    s = pd.read_csv(summary_csv)
    if s.empty:
        raise ValueError(f"Summary file is empty: {summary_csv}")
    row = s.iloc[0]
    coherence_floor = float(row["coherence_floor"]) if "coherence_floor" in s.columns else 0.0
    coherence_power = float(row["coherence_power"]) if "coherence_power" in s.columns else 1.0
    coherence_blend = float(row["coherence_blend"]) if "coherence_blend" in s.columns else 1.0
    return coherence_floor, coherence_power, coherence_blend


def _load_band_dims(mode_index_csv: Path, n_bands: int) -> np.ndarray:
    md = pd.read_csv(mode_index_csv)
    _require_columns(md, ["band_id"], mode_index_csv)
    counts = md.groupby("band_id").size().to_dict()
    dims = np.asarray([float(max(int(counts.get(i, 1)), 1)) for i in range(n_bands)], dtype=float)
    return dims


def _compute_signal_mu(
    *,
    coh_mu: np.ndarray,
    entropy_mu: np.ndarray,
    band_dims: np.ndarray,
    coherence_floor: float,
    coherence_power: float,
    coherence_blend: float,
) -> np.ndarray:
    coh_eff = np.power(np.maximum(coh_mu + coherence_floor, 0.0), coherence_power)
    max_entropy = np.log(np.maximum(band_dims, 1.0))
    max_entropy = np.where(max_entropy < 1e-12, 1.0, max_entropy)
    diag_mix = np.clip(entropy_mu / max_entropy[None, :], 0.0, 1.0)
    signal_mu = coherence_blend * coh_eff + (1.0 - coherence_blend) * diag_mix
    return np.asarray(signal_mu, dtype=float)


def _build_lambda_placebo_mu(
    *,
    weights_mu: np.ndarray,
    signal_mu: np.ndarray,
    lambda_mu: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_bands = lambda_mu.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_bands)
    if n_bands > 1 and np.all(perm == np.arange(n_bands, dtype=int)):
        perm = np.roll(perm, 1)

    lambda_placebo = np.sum(weights_mu * signal_mu[:, perm] * lambda_mu[:, perm], axis=1)
    return np.asarray(lambda_placebo, dtype=float), np.asarray(perm, dtype=int)


def _evaluate_ctrl_model(
    *,
    y: np.ndarray,
    ctrl: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
) -> tuple[np.ndarray, float, float]:
    x = np.column_stack([ctrl])
    _, _, yhat = _evaluate_splits(
        y=y,
        x_base=x,
        x_full=x,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl"],
        splits=splits,
        ridge_alpha=ridge_alpha,
    )
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return yhat, mae, rmse


def _evaluate_lambda_model(
    *,
    y: np.ndarray,
    ctrl: np.ndarray,
    lam: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
    n_perm: int,
    perm_block: int,
    perm_seed: int,
    strata_q: int,
    feature_name: str,
) -> ModelEval:
    x_base = np.column_stack([ctrl])
    x_full = np.column_stack([ctrl, lam])
    split_df, yhat_base, yhat_full = _evaluate_splits(
        y=y,
        x_base=x_base,
        x_full=x_full,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl", feature_name],
        splits=splits,
        ridge_alpha=ridge_alpha,
    )

    mae_base = float(np.mean(np.abs(y - yhat_base)))
    mae_full = float(np.mean(np.abs(y - yhat_full)))
    rmse_full = float(np.sqrt(np.mean((y - yhat_full) ** 2)))
    gain = float((mae_base - mae_full) / (mae_base + 1e-12))

    if n_perm > 0:
        perm_p, _, _ = _permutation_test(
            y=y,
            x_base=x_base,
            x_full=x_full,
            base_feature_names=["ctrl"],
            full_feature_names=["ctrl", feature_name],
            permute_cols=np.array([1], dtype=int),
            splits=splits,
            ridge_alpha=ridge_alpha,
            n_perm=n_perm,
            perm_block=perm_block,
            seed=perm_seed,
        )
    else:
        perm_p = np.nan

    strata_df = _strata_table(
        y=y,
        yhat_base=yhat_base,
        yhat_full=yhat_full,
        n_ctrl=ctrl,
        q=strata_q,
    )
    positive_frac = float(np.mean(strata_df["mae_gain_frac"].to_numpy(dtype=float) >= 0.0)) if len(strata_df) else np.nan

    coef_full, _, _ = _fit_ridge_scaled(x_full, y, x_full, ridge_alpha)
    beta = float(coef_full[1]) if len(coef_full) > 1 else np.nan

    return ModelEval(
        mae_base_oof=mae_base,
        mae_full_oof=mae_full,
        rmse_full_oof=rmse_full,
        oof_gain_frac=gain,
        perm_p_value=float(perm_p),
        beta_lambda=beta,
        positive_strata_frac=positive_frac,
        yhat_base_oof=yhat_base,
        yhat_full_oof=yhat_full,
    )


def _info_criteria_from_oof(
    *,
    y: np.ndarray,
    yhat: np.ndarray,
    model: str,
    k: int,
) -> dict[str, float | int | str]:
    resid = y - yhat
    n = int(len(resid))
    sigma2 = float(max(np.mean(resid**2), 1e-12))
    log_l = float(-(n / 2.0) * (np.log(2.0 * np.pi * sigma2) + 1.0))
    aic = float(2.0 * k - 2.0 * log_l)
    bic = float(k * np.log(n) - 2.0 * log_l)
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {
        "model": model,
        "n": n,
        "k": int(k),
        "sigma2_hat": sigma2,
        "logL": log_l,
        "AIC": aic,
        "BIC": bic,
        "MAE": mae,
        "RMSE": rmse,
    }


def run_all(
    *,
    comparison_dataset: Path,
    timeseries_csv: Path,
    summary_csv: Path,
    mode_index_csv: Path,
    lambda_column: str,
    outdir: Path,
    ridge_alpha: float,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    strata_q: int,
    seed: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    cmp_df = pd.read_csv(comparison_dataset)
    _require_columns(
        cmp_df,
        ["time_index", "residual_base_res0", "n_density_ctrl_z", lambda_column],
        comparison_dataset,
    )

    ts_df = pd.read_csv(timeseries_csv)
    _require_columns(ts_df, ["time_index", "lambda_struct"], timeseries_csv)

    lambda_mu_cols = _sorted_band_cols(list(ts_df.columns), "lambda_mu_")
    weight_mu_cols = _sorted_band_cols(list(ts_df.columns), "weight_mu_")
    coh_mu_cols = _sorted_band_cols(list(ts_df.columns), "coh_mu_")
    entropy_mu_cols = _sorted_band_cols(list(ts_df.columns), "entropy_mu_")

    if not lambda_mu_cols:
        raise ValueError(f"No lambda_mu_* columns found in {timeseries_csv}")
    n_bands = len(lambda_mu_cols)
    if len(weight_mu_cols) != n_bands or len(coh_mu_cols) != n_bands:
        raise ValueError(
            "Mismatch in per-band columns. Expected same count for lambda_mu_*, weight_mu_*, coh_mu_*."
        )
    if len(entropy_mu_cols) not in (0, n_bands):
        raise ValueError("entropy_mu_* columns must be either absent or present for all bands.")
    has_entropy_mu = len(entropy_mu_cols) == n_bands

    merge_cols = ["time_index", "lambda_struct"] + lambda_mu_cols + weight_mu_cols + coh_mu_cols + entropy_mu_cols
    merged = cmp_df.merge(ts_df[merge_cols], on="time_index", how="inner")

    needed = ["residual_base_res0", "n_density_ctrl_z", lambda_column, "lambda_struct"] + lambda_mu_cols + weight_mu_cols + coh_mu_cols + entropy_mu_cols
    finite_mask = np.ones(len(merged), dtype=bool)
    for c in needed:
        finite_mask &= np.isfinite(merged[c].to_numpy(dtype=float))
    d = merged.loc[finite_mask].copy().reset_index(drop=True)
    if len(d) < max(40, n_folds * 8):
        raise ValueError(f"Not enough finite rows for evaluation: {len(d)}")

    y = d["residual_base_res0"].to_numpy(dtype=float)
    ctrl = d["n_density_ctrl_z"].to_numpy(dtype=float)
    lambda_real = d[lambda_column].to_numpy(dtype=float)

    lambda_mu = d[lambda_mu_cols].to_numpy(dtype=float)
    weights_mu = d[weight_mu_cols].to_numpy(dtype=float)
    coh_mu = d[coh_mu_cols].to_numpy(dtype=float)
    if has_entropy_mu:
        entropy_mu = d[entropy_mu_cols].to_numpy(dtype=float)
    else:
        entropy_mu = np.zeros_like(coh_mu)

    coherence_floor, coherence_power, coherence_blend = _load_meta(summary_csv)
    if not has_entropy_mu and coherence_blend < 1.0 - 1e-12:
        # Older runs may not store entropy_mu_*; in that case fall back to coherence-only signal.
        coherence_blend = 1.0
    band_dims = _load_band_dims(mode_index_csv, n_bands=n_bands)
    signal_mu = _compute_signal_mu(
        coh_mu=coh_mu,
        entropy_mu=entropy_mu,
        band_dims=band_dims,
        coherence_floor=coherence_floor,
        coherence_power=coherence_power,
        coherence_blend=coherence_blend,
    )

    lambda_real_recon = np.sum(weights_mu * signal_mu * lambda_mu, axis=1)
    lambda_placebo_mu, perm = _build_lambda_placebo_mu(
        weights_mu=weights_mu,
        signal_mu=signal_mu,
        lambda_mu=lambda_mu,
        seed=seed,
    )

    # F_comm control: trace of commutator-generated F_phys is zero in exact arithmetic.
    lambda_comm = np.zeros_like(lambda_real, dtype=float)

    splits = _blocked_splits(len(y), n_folds=n_folds)

    ctrl_yhat, ctrl_mae, ctrl_rmse = _evaluate_ctrl_model(
        y=y,
        ctrl=ctrl,
        splits=splits,
        ridge_alpha=ridge_alpha,
    )

    real_eval = _evaluate_lambda_model(
        y=y,
        ctrl=ctrl,
        lam=lambda_real,
        splits=splits,
        ridge_alpha=ridge_alpha,
        n_perm=n_perm,
        perm_block=perm_block,
        perm_seed=seed + 101,
        strata_q=strata_q,
        feature_name="lambda_real",
    )
    placebo_eval = _evaluate_lambda_model(
        y=y,
        ctrl=ctrl,
        lam=lambda_placebo_mu,
        splits=splits,
        ridge_alpha=ridge_alpha,
        n_perm=n_perm,
        perm_block=perm_block,
        perm_seed=seed + 211,
        strata_q=strata_q,
        feature_name="lambda_placebo_mu",
    )
    comm_eval = _evaluate_lambda_model(
        y=y,
        ctrl=ctrl,
        lam=lambda_comm,
        splits=splits,
        ridge_alpha=ridge_alpha,
        n_perm=0,
        perm_block=perm_block,
        perm_seed=seed + 307,
        strata_q=strata_q,
        feature_name="lambda_comm",
    )

    placebo_metrics = pd.DataFrame(
        [
            {
                "mae_gain_real": float(real_eval.oof_gain_frac),
                "mae_gain_placebo": float(placebo_eval.oof_gain_frac),
                "perm_p": float(placebo_eval.perm_p_value),
                "beta_lambda": float(real_eval.beta_lambda),
                "positive_strata_frac": float(real_eval.positive_strata_frac),
            }
        ]
    )
    placebo_metrics.to_csv(outdir / "placebo_mu_metrics.csv", index=False)

    comm_metrics = pd.DataFrame(
        [
            {
                "mae_gain_real": float(real_eval.oof_gain_frac),
                "mae_gain_comm": float(comm_eval.oof_gain_frac),
                "beta_real": float(real_eval.beta_lambda),
                "beta_comm": float(comm_eval.beta_lambda),
            }
        ]
    )
    comm_metrics.to_csv(outdir / "comm_control_metrics.csv", index=False)

    info_rows = [
        _info_criteria_from_oof(y=y, yhat=ctrl_yhat, model="ctrl", k=2),
        _info_criteria_from_oof(y=y, yhat=real_eval.yhat_full_oof, model="ctrl+lambda_real", k=3),
        _info_criteria_from_oof(y=y, yhat=placebo_eval.yhat_full_oof, model="ctrl+lambda_placebo_mu", k=3),
        _info_criteria_from_oof(y=y, yhat=comm_eval.yhat_full_oof, model="ctrl+lambda_comm", k=3),
    ]
    info_df = pd.DataFrame(info_rows)
    info_df.to_csv(outdir / "info_criteria.csv", index=False)

    summary_table = pd.DataFrame(
        [
            {
                "lambda_variant": "real",
                "mae_oof": float(real_eval.mae_full_oof),
                "rmse_oof": float(real_eval.rmse_full_oof),
                "gain_vs_ctrl": float(real_eval.oof_gain_frac),
                "perm_p": float(real_eval.perm_p_value),
                "beta": float(real_eval.beta_lambda),
                "positive_strata_frac": float(real_eval.positive_strata_frac),
            },
            {
                "lambda_variant": "placebo_mu",
                "mae_oof": float(placebo_eval.mae_full_oof),
                "rmse_oof": float(placebo_eval.rmse_full_oof),
                "gain_vs_ctrl": float(placebo_eval.oof_gain_frac),
                "perm_p": float(placebo_eval.perm_p_value),
                "beta": float(placebo_eval.beta_lambda),
                "positive_strata_frac": float(placebo_eval.positive_strata_frac),
            },
            {
                "lambda_variant": "comm",
                "mae_oof": float(comm_eval.mae_full_oof),
                "rmse_oof": float(comm_eval.rmse_full_oof),
                "gain_vs_ctrl": float(comm_eval.oof_gain_frac),
                "perm_p": np.nan,
                "beta": float(comm_eval.beta_lambda),
                "positive_strata_frac": float(comm_eval.positive_strata_frac),
            },
        ]
    )
    summary_table.to_csv(outdir / "lambda_summary_real_placebo_comm.csv", index=False)

    series_df = pd.DataFrame(
        {
            "time_index": d["time_index"].to_numpy(dtype=int),
            "residual_base_res0": y,
            "n_density_ctrl_z": ctrl,
            "lambda_real": lambda_real,
            "lambda_real_reconstructed": lambda_real_recon,
            "lambda_placebo_mu": lambda_placebo_mu,
            "lambda_comm": lambda_comm,
        }
    )
    series_df.to_csv(outdir / "lambda_falsification_timeseries.csv", index=False)

    recon_corr = float(np.corrcoef(lambda_real, lambda_real_recon)[0, 1]) if np.std(lambda_real_recon) > 1e-12 else np.nan
    recon_mae = float(np.mean(np.abs(lambda_real - lambda_real_recon)))
    meta = {
        "seed": int(seed),
        "n_rows": int(len(d)),
        "n_bands": int(n_bands),
        "lambda_column": str(lambda_column),
        "timeseries_csv": str(timeseries_csv),
        "summary_csv": str(summary_csv),
        "mode_index_csv": str(mode_index_csv),
        "coherence_floor": float(coherence_floor),
        "coherence_power": float(coherence_power),
        "coherence_blend": float(coherence_blend),
        "mu_permutation": perm.tolist(),
        "lambda_reconstruction_corr": recon_corr,
        "lambda_reconstruction_mae": recon_mae,
        "ctrl_mae_oof": float(ctrl_mae),
        "ctrl_rmse_oof": float(ctrl_rmse),
    }
    with (outdir / "falsification_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    aic_ctrl = float(info_df.loc[info_df["model"] == "ctrl", "AIC"].iloc[0])
    bic_ctrl = float(info_df.loc[info_df["model"] == "ctrl", "BIC"].iloc[0])
    aic_real = float(info_df.loc[info_df["model"] == "ctrl+lambda_real", "AIC"].iloc[0])
    bic_real = float(info_df.loc[info_df["model"] == "ctrl+lambda_real", "BIC"].iloc[0])
    delta_aic = aic_real - aic_ctrl
    delta_bic = bic_real - bic_ctrl

    report_lines = [
        "# Experiment M4 Lambda necessity falsification (S1/S2/S3)",
        "",
        "## S1 Placebo-mu (scale-layer permutation)",
        f"- mae_gain_real: {real_eval.oof_gain_frac:.6f}",
        f"- mae_gain_placebo: {placebo_eval.oof_gain_frac:.6f}",
        f"- perm_p(placebo): {placebo_eval.perm_p_value:.6f}",
        f"- beta_lambda(real): {real_eval.beta_lambda:.6f}",
        f"- positive_strata_frac(real): {real_eval.positive_strata_frac:.6f}",
        "",
        "## S2 F^comm control",
        f"- mae_gain_real: {real_eval.oof_gain_frac:.6f}",
        f"- mae_gain_comm: {comm_eval.oof_gain_frac:.6f}",
        f"- beta_real: {real_eval.beta_lambda:.6f}",
        f"- beta_comm: {comm_eval.beta_lambda:.6f}",
        "",
        "## S3 Information criteria (OOF residuals)",
        f"- Delta AIC (ctrl -> ctrl+lambda_real): {delta_aic:.6f}",
        f"- Delta BIC (ctrl -> ctrl+lambda_real): {delta_bic:.6f}",
        "",
        "## Note",
        "- In this pipeline, Lambda_comm is the commutator-trace control and evaluates to zero in exact arithmetic.",
    ]
    (outdir / "falsification_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[falsification] ΔAIC(ctrl -> ctrl+lambda_real) = {delta_aic:.6f}", flush=True)
    print(f"[falsification] ΔBIC(ctrl -> ctrl+lambda_real) = {delta_bic:.6f}", flush=True)
    print(f"[falsification] done -> {outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dataset",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_horizontal_vertical_compare/comparison_dataset.csv"),
    )
    parser.add_argument(
        "--timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--mode-index-csv", type=Path, default=None)
    parser.add_argument("--lambda-column", type=str, default="lambda_v")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_lambda_falsification_tests"),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--n-perm", type=int, default=180)
    parser.add_argument("--perm-block", type=int, default=24)
    parser.add_argument("--strata-q", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260302)
    args = parser.parse_args()

    summary_csv = args.summary_csv or (args.timeseries_csv.parent / "experiment_M_summary.csv")
    mode_index_csv = args.mode_index_csv or (args.timeseries_csv.parent / "experiment_M_mode_index.csv")

    run_all(
        comparison_dataset=args.comparison_dataset,
        timeseries_csv=args.timeseries_csv,
        summary_csv=summary_csv,
        mode_index_csv=mode_index_csv,
        lambda_column=args.lambda_column,
        outdir=args.outdir,
        ridge_alpha=args.ridge_alpha,
        n_folds=args.n_folds,
        n_perm=args.n_perm,
        perm_block=args.perm_block,
        strata_q=args.strata_q,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
