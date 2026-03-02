#!/usr/bin/env python3
"""Compare horizontal vs vertical Experiment M results and check artifact hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import linregress, pearsonr, t as student_t

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        _blocked_splits,
        _evaluate_splits,
        _permutation_test,
        _xy_coordinates_m,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        _blocked_splits,
        _evaluate_splits,
        _permutation_test,
        _xy_coordinates_m,
    )


def _find_coord_name(dataset: Dataset, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"Could not find coordinate for {label}. Tried: {candidates}.")


def _find_var_name(dataset: Dataset, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"Could not find variable for {label}. Tried: {candidates}.")


def _move_to_tyx(
    arr: np.ndarray,
    dims: tuple[str, ...],
    time_name: str,
    lat_name: str,
    lon_name: str,
) -> np.ndarray:
    t_axis = dims.index(time_name)
    lat_axis = dims.index(lat_name)
    lon_axis = dims.index(lon_name)
    out = np.moveaxis(arr, (t_axis, lat_axis, lon_axis), (0, 1, 2))
    out = np.squeeze(out)
    if out.ndim != 3:
        raise ValueError(f"Expected 3D array (time,lat,lon), got shape={out.shape} for dims={dims}")
    return out


def _move_to_tlxy(
    arr: np.ndarray,
    dims: tuple[str, ...],
    time_name: str,
    level_name: str,
    lat_name: str,
    lon_name: str,
) -> np.ndarray:
    t_axis = dims.index(time_name)
    l_axis = dims.index(level_name)
    lat_axis = dims.index(lat_name)
    lon_axis = dims.index(lon_name)
    out = np.moveaxis(arr, (t_axis, l_axis, lat_axis, lon_axis), (0, 1, 2, 3))
    out = np.squeeze(out)
    if out.ndim != 4:
        raise ValueError(f"Expected 4D array (time,level,lat,lon), got shape={out.shape} for dims={dims}")
    return out


def _compute_proxies_from_input(input_nc: Path, batch_size: int = 168) -> pd.DataFrame:
    with Dataset(input_nc, mode="r") as ds:
        time_name = _find_coord_name(ds, ("valid_time", "time", "datetime", "date"), "time")
        lat_name = _find_coord_name(ds, ("latitude", "lat", "y", "rlat"), "latitude")
        lon_name = _find_coord_name(ds, ("longitude", "lon", "x", "rlon"), "longitude")
        level_name = _find_coord_name(ds, ("pressure_level", "level", "plev", "isobaricInhPa"), "pressure level")

        u_name = _find_var_name(ds, ("u", "u10", "u100", "u850", "u1000"), "u")
        v_name = _find_var_name(ds, ("v", "v10", "v100", "v850", "v1000"), "v")
        w_name = _find_var_name(ds, ("w_pl", "w", "omega", "vertical_velocity"), "w_pl")
        q_name = _find_var_name(ds, ("q_pl", "specific_humidity", "q"), "q_pl")

        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)
        x_m, y_m = _xy_coordinates_m(lat=lat, lon=lon)
        eo_x = 2 if len(x_m) >= 3 else 1
        eo_y = 2 if len(y_m) >= 3 else 1

        levels = np.asarray(ds.variables[level_name][:], dtype=float)
        levels_hpa = levels / 100.0 if np.nanmax(np.abs(levels)) > 3000 else levels
        level_mask = (levels_hpa <= 850.0) & (levels_hpa >= 300.0)
        if int(level_mask.sum()) == 0:
            level_mask = np.ones_like(levels_hpa, dtype=bool)

        u_var = ds.variables[u_name]
        v_var = ds.variables[v_name]
        w_var = ds.variables[w_name]
        q_var = ds.variables[q_name]

        nt = int(u_var.shape[0])
        vorticity_proxy = np.full(nt, np.nan, dtype=float)
        omega_q_proxy = np.full(nt, np.nan, dtype=float)

        u_dims = tuple(u_var.dimensions)
        v_dims = tuple(v_var.dimensions)
        w_dims = tuple(w_var.dimensions)
        q_dims = tuple(q_var.dimensions)

        for start in range(0, nt, batch_size):
            stop = min(start + batch_size, nt)

            u_blk = _move_to_tyx(
                np.asarray(u_var[start:stop, ...], dtype=float),
                u_dims,
                time_name=time_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )
            v_blk = _move_to_tyx(
                np.asarray(v_var[start:stop, ...], dtype=float),
                v_dims,
                time_name=time_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )

            dv_dx = np.gradient(v_blk, x_m, axis=2, edge_order=eo_x)
            du_dy = np.gradient(u_blk, y_m, axis=1, edge_order=eo_y)
            zeta = dv_dx - du_dy
            vorticity_proxy[start:stop] = np.nanmean(np.abs(zeta), axis=(1, 2))

            w_blk = _move_to_tlxy(
                np.asarray(w_var[start:stop, ...], dtype=float),
                w_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )
            q_blk = _move_to_tlxy(
                np.asarray(q_var[start:stop, ...], dtype=float),
                q_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )
            omega_q_proxy[start:stop] = np.nanmean(-(w_blk[:, level_mask] * q_blk[:, level_mask]), axis=(1, 2, 3))

            print(f"[compare] proxy progress: {stop}/{nt}", flush=True)

    return pd.DataFrame(
        {
            "time_index": np.arange(nt, dtype=int),
            "proxy_vorticity_abs_mean": vorticity_proxy,
            "proxy_omega_q_850_300": omega_q_proxy,
        }
    )


def _load_core_timeseries(path: Path, label: str) -> pd.DataFrame:
    cols = [
        "time_index",
        "time",
        "residual_base_res0",
        "n_density_ctrl_z",
        "lambda_struct",
        "resid_base_oof",
        "resid_full_oof",
    ]
    df = pd.read_csv(path, usecols=cols)
    rename = {
        "lambda_struct": f"lambda_{label}",
        "resid_base_oof": f"resid_base_oof_{label}",
        "resid_full_oof": f"resid_full_oof_{label}",
    }
    return df.rename(columns=rename)


def _merge_horizontal_vertical(horizontal_path: Path, vertical_path: Path) -> pd.DataFrame:
    h = _load_core_timeseries(horizontal_path, label="h")
    v = _load_core_timeseries(vertical_path, label="v")
    merged = h.merge(
        v[
            [
                "time_index",
                "time",
                "residual_base_res0",
                "n_density_ctrl_z",
                "lambda_v",
                "resid_base_oof_v",
                "resid_full_oof_v",
            ]
        ],
        on="time_index",
        suffixes=("_h", "_v"),
    )
    merged = merged.rename(
        columns={
            "time_h": "time",
            "residual_base_res0_h": "residual_base_res0",
            "n_density_ctrl_z_h": "n_density_ctrl_z",
            "lambda_h": "lambda_h",
        }
    )

    if not np.allclose(
        merged["residual_base_res0"].to_numpy(dtype=float),
        merged["residual_base_res0_v"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError("Residual target mismatch between horizontal and vertical timeseries.")

    if not np.allclose(
        merged["n_density_ctrl_z"].to_numpy(dtype=float),
        merged["n_density_ctrl_z_v"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError("Control feature mismatch between horizontal and vertical timeseries.")

    merged = merged.drop(columns=["time_v", "residual_base_res0_v", "n_density_ctrl_z_v"])
    return merged.sort_values("time_index").reset_index(drop=True)


def _linear_ci(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n < 3:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)

    fit = linregress(x, y)
    y_hat = fit.intercept + fit.slope * x
    sse = np.sum((y - y_hat) ** 2)
    s_err = np.sqrt(sse / max(n - 2, 1))

    x_bar = np.mean(x)
    ssx = np.sum((x - x_bar) ** 2)
    if ssx < 1e-15:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)

    t_crit = float(student_t.ppf(1.0 - alpha / 2.0, df=max(n - 2, 1)))
    band = t_crit * s_err * np.sqrt((1.0 / n) + ((x_grid - x_bar) ** 2) / ssx)
    y_grid = fit.intercept + fit.slope * x_grid
    return y_grid - band, y_grid + band


def _plot_lambda_scatter(df: pd.DataFrame, out_path: Path, stats_row: dict[str, float]) -> None:
    x = df["lambda_h"].to_numpy(dtype=float)
    y = df["lambda_v"].to_numpy(dtype=float)

    fit = linregress(x, y)
    x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
    y_grid = fit.intercept + fit.slope * x_grid
    ci_lo, ci_hi = _linear_ci(x=x, y=y, x_grid=x_grid, alpha=0.05)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(x, y, s=14, alpha=0.35, color="#1f77b4", edgecolor="none", label="Time steps")
    ax.plot(x_grid, y_grid, color="#d62728", lw=2.0, label="Linear fit")
    if np.all(np.isfinite(ci_lo)) and np.all(np.isfinite(ci_hi)):
        ax.fill_between(x_grid, ci_lo, ci_hi, color="#d62728", alpha=0.15, label="95% CI")

    ax.set_xlabel("Lambda_horizontal")
    ax.set_ylabel("Lambda_vertical")
    ax.set_title("Experiment M: Horizontal vs Vertical Lambda")
    txt = (
        f"Pearson r={stats_row['pearson_r']:.4f}\n"
        f"R^2={stats_row['r2']:.4f}\n"
        f"p={stats_row['pearson_p']:.3g}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.85})
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _evaluate_model_set(
    *,
    y: np.ndarray,
    ctrl: np.ndarray,
    lambda_h: np.ndarray,
    lambda_v: np.ndarray,
    ridge_alpha: float,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    splits = _blocked_splits(len(y), n_folds=n_folds)

    model_specs: list[tuple[str, np.ndarray, list[str], np.ndarray]] = [
        ("horizontal", np.column_stack([ctrl, lambda_h]), ["ctrl", "lambda_h"], np.array([1], dtype=int)),
        ("vertical", np.column_stack([ctrl, lambda_v]), ["ctrl", "lambda_v"], np.array([1], dtype=int)),
        (
            "combined",
            np.column_stack([ctrl, lambda_h, lambda_v]),
            ["ctrl", "lambda_h", "lambda_v"],
            np.array([1, 2], dtype=int),
        ),
    ]

    x_base = np.column_stack([ctrl])
    model_rows: list[dict[str, float | str]] = []
    split_frames: list[pd.DataFrame] = []
    perm_frames: list[pd.DataFrame] = []
    preds: dict[str, np.ndarray] = {}
    yhat_base_ref: np.ndarray | None = None

    for model_name, x_full, feature_names, permute_cols in model_specs:
        split_df, yhat_base, yhat_full = _evaluate_splits(
            y=y,
            x_base=x_base,
            x_full=x_full,
            base_feature_names=["ctrl"],
            full_feature_names=feature_names,
            splits=splits,
            ridge_alpha=ridge_alpha,
        )

        if yhat_base_ref is None:
            yhat_base_ref = yhat_base.copy()

        mae_base = float(np.mean(np.abs(y - yhat_base)))
        mae_full = float(np.mean(np.abs(y - yhat_full)))
        gain = float((mae_base - mae_full) / (mae_base + 1e-12))

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
        split_df.insert(0, "model", model_name)
        split_frames.append(split_df)

        perm_df = perm_df.copy()
        perm_df.insert(0, "model", model_name)
        perm_frames.append(perm_df)

        model_rows.append(
            {
                "model": model_name,
                "mae_base_oof": mae_base,
                "mae_model_oof": mae_full,
                "oof_gain_frac": gain,
                "split_gain_median": float(np.median(split_df["mae_gain_frac"])),
                "split_gain_q25": float(np.quantile(split_df["mae_gain_frac"], 0.25)),
                "split_gain_q75": float(np.quantile(split_df["mae_gain_frac"], 0.75)),
                "perm_stat_real_median_gain": stat_real,
                "perm_p_value": p_perm,
            }
        )
        preds[model_name] = yhat_full

    if yhat_base_ref is None:
        raise RuntimeError("Internal error: base predictions were not computed.")

    preds["baseline_ctrl"] = yhat_base_ref

    model_df = pd.DataFrame(model_rows).sort_values("oof_gain_frac", ascending=False).reset_index(drop=True)
    split_out = pd.concat(split_frames, ignore_index=True)
    perm_out = pd.concat(perm_frames, ignore_index=True)
    return model_df, split_out, perm_out, preds


def _quartile_gain_table(
    *,
    y: np.ndarray,
    yhat_base: np.ndarray,
    yhat_by_model: dict[str, np.ndarray],
    proxy: np.ndarray,
    proxy_name: str,
    q: int = 4,
) -> pd.DataFrame:
    if q < 2:
        raise ValueError("q must be >=2")
    quant = np.quantile(proxy, np.linspace(0.0, 1.0, q + 1))
    rows: list[dict[str, float | int | str]] = []

    for i in range(q):
        lo = float(quant[i])
        hi = float(quant[i + 1])
        if i < q - 1:
            idx = np.where((proxy >= lo) & (proxy < hi))[0]
        else:
            idx = np.where((proxy >= lo) & (proxy <= hi))[0]
        if len(idx) == 0:
            continue

        mae_base = float(np.mean(np.abs(y[idx] - yhat_base[idx])))
        for model in ("horizontal", "vertical", "combined"):
            yhat = yhat_by_model[model]
            mae_model = float(np.mean(np.abs(y[idx] - yhat[idx])))
            gain = float((mae_base - mae_model) / (mae_base + 1e-12))
            rows.append(
                {
                    "proxy": proxy_name,
                    "quartile_id": int(i + 1),
                    "proxy_lo": lo,
                    "proxy_hi": hi,
                    "n": int(len(idx)),
                    "model": model,
                    "mae_base": mae_base,
                    "mae_model": mae_model,
                    "mae_gain_frac": gain,
                }
            )

    return pd.DataFrame(rows)


def _plot_model_gains(model_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = model_df.sort_values("oof_gain_frac", ascending=False).reset_index(drop=True)
    x = np.arange(len(plot_df), dtype=float)
    y = plot_df["oof_gain_frac"].to_numpy(dtype=float)
    err_low = y - plot_df["split_gain_q25"].to_numpy(dtype=float)
    err_high = plot_df["split_gain_q75"].to_numpy(dtype=float) - y

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    ax.bar(x, y, color=colors[: len(y)], width=0.65, alpha=0.85)
    ax.errorbar(x, y, yerr=np.vstack([err_low, err_high]), fmt="none", ecolor="#333333", elinewidth=1.1, capsize=4)
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_xticks(x, plot_df["model"].tolist())
    ax.set_ylabel("OOF gain fraction (vs ctrl baseline)")
    ax.set_title("Experiment M: Model Gain Comparison")
    for i, row in plot_df.iterrows():
        ax.text(i, row["oof_gain_frac"] + 0.0002, f"p={row['perm_p_value']:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_quartile_gains(q_df: pd.DataFrame, proxy_name: str, out_path: Path) -> None:
    d = q_df[q_df["proxy"] == proxy_name].copy()
    if d.empty:
        return

    quartiles = sorted(d["quartile_id"].unique().tolist())
    models = ["horizontal", "vertical", "combined"]
    width = 0.25
    x = np.arange(len(quartiles), dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    palette = {"horizontal": "#1f77b4", "vertical": "#ff7f0e", "combined": "#2ca02c"}

    for j, model in enumerate(models):
        yvals = []
        for qid in quartiles:
            sub = d[(d["quartile_id"] == qid) & (d["model"] == model)]
            yvals.append(float(sub["mae_gain_frac"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (j - 1) * width, yvals, width=width, label=model, color=palette[model], alpha=0.9)

    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_xticks(x, [f"Q{int(qid)}" for qid in quartiles])
    ax.set_ylabel("MAE gain fraction (vs ctrl baseline)")
    ax.set_title(f"Stratified Gain by {proxy_name}")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_delta_abs_error(y: np.ndarray, preds: dict[str, np.ndarray], out_path: Path) -> None:
    err_h = np.abs(y - preds["horizontal"])
    err_v = np.abs(y - preds["vertical"])
    delta = err_v - err_h

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.hist(delta, bins=42, color="#9467bd", alpha=0.78, edgecolor="white")
    ax.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_xlabel("abs_error_vertical - abs_error_horizontal")
    ax.set_ylabel("Count")
    ax.set_title("Per-step Error Difference: Vertical vs Horizontal")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _artifact_diagnosis(corr_df: pd.DataFrame, model_df: pd.DataFrame, q_df: pd.DataFrame) -> dict[str, float | str]:
    corr = corr_df.iloc[0]
    gain_h = float(model_df.loc[model_df["model"] == "horizontal", "oof_gain_frac"].iloc[0])
    gain_v = float(model_df.loc[model_df["model"] == "vertical", "oof_gain_frac"].iloc[0])
    gain_c = float(model_df.loc[model_df["model"] == "combined", "oof_gain_frac"].iloc[0])
    delta_best = gain_c - max(gain_h, gain_v)

    q_spread = q_df.pivot_table(index=["proxy", "quartile_id"], columns="model", values="mae_gain_frac")
    common = q_spread.dropna()
    if common.empty:
        hv_gap_mean = np.nan
    else:
        hv_gap_mean = float(np.mean(np.abs(common["horizontal"] - common["vertical"])))

    if corr["r2"] >= 0.3 and delta_best <= 0.0004:
        verdict = "likely_detectability_limit_with_redundant_physical_signal"
    elif corr["r2"] < 0.1 and delta_best <= 0.0002 and (np.isnan(hv_gap_mean) or hv_gap_mean < 0.0003):
        verdict = "artifact_risk_high"
    elif delta_best > 0.0008:
        verdict = "signals_are_complementary"
    else:
        verdict = "mixed_evidence_needs_more_permutation_power"

    return {
        "verdict": verdict,
        "r2_lambda_h_lambda_v": float(corr["r2"]),
        "combined_minus_best_single_gain": float(delta_best),
        "mean_abs_gap_horizontal_vs_vertical_strata_gain": float(hv_gap_mean),
        "gain_horizontal": gain_h,
        "gain_vertical": gain_v,
        "gain_combined": gain_c,
    }


def _load_context_summaries(summary_paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for label, path in summary_paths.items():
        if not path.exists():
            continue
        row = pd.read_csv(path).iloc[0].to_dict()
        rows.append(
            {
                "run": label,
                "oof_gain_frac": float(row.get("oof_gain_frac", np.nan)),
                "perm_p_value": float(row.get("perm_p_value", np.nan)),
                "strata_positive_frac": float(row.get("strata_positive_frac", np.nan)),
                "lambda_sign_consistency": float(row.get("lambda_sign_consistency", np.nan)),
                "pass_all": bool(row.get("pass_all", False)),
                "feature_set": str(row.get("feature_set", "n/a")),
            }
        )
    return pd.DataFrame(rows)


def _save_report(
    outdir: Path,
    corr_df: pd.DataFrame,
    model_df: pd.DataFrame,
    diagnosis: dict[str, float | str],
    context_df: pd.DataFrame,
) -> None:
    corr = corr_df.iloc[0]
    gain_h = float(model_df.loc[model_df["model"] == "horizontal", "oof_gain_frac"].iloc[0])
    gain_v = float(model_df.loc[model_df["model"] == "vertical", "oof_gain_frac"].iloc[0])
    gain_c = float(model_df.loc[model_df["model"] == "combined", "oof_gain_frac"].iloc[0])
    delta_best = float(diagnosis["combined_minus_best_single_gain"])

    lines = [
        "# Experiment M Horizontal vs Vertical Comparison",
        "",
        "## Test 1: Lambda correlation",
        f"- Pearson r: {float(corr['pearson_r']):.6f}",
        f"- R^2: {float(corr['r2']):.6f}",
        f"- p-value: {float(corr['pearson_p']):.6g}",
        "",
        "## Test 2: Combined model vs single models",
        f"- gain(ctrl+lambda_h): {gain_h:.6f}",
        f"- gain(ctrl+lambda_v): {gain_v:.6f}",
        f"- gain(ctrl+lambda_h+lambda_v): {gain_c:.6f}",
        f"- combined minus best single: {delta_best:.6f}",
        "",
        "## Diagnosis",
        f"- verdict: {diagnosis['verdict']}",
        f"- mean |gain_h - gain_v| across proxy quartiles: {float(diagnosis['mean_abs_gap_horizontal_vs_vertical_strata_gain']):.6f}",
        "",
        "## Context (existing runs)",
    ]

    if not context_df.empty:
        for _, row in context_df.iterrows():
            lines.append(
                f"- {row['run']}: gain={row['oof_gain_frac']:.6f}, p={row['perm_p_value']:.6g}, "
                f"strata_pos={row['strata_positive_frac']:.3f}, pass_all={bool(row['pass_all'])}"
            )
    else:
        lines.append("- context summaries are unavailable")

    report_path = outdir / "comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compare(
    *,
    horizontal_timeseries: Path,
    vertical_timeseries: Path,
    vertical_input_nc: Path,
    outdir: Path,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    ridge_alpha: float,
    seed: int,
    summary_paths: dict[str, Path],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df = _merge_horizontal_vertical(horizontal_timeseries, vertical_timeseries)
    proxies = _compute_proxies_from_input(vertical_input_nc)
    df = df.merge(proxies, on="time_index", how="left")

    valid_mask = np.isfinite(df["residual_base_res0"]) & np.isfinite(df["n_density_ctrl_z"])
    valid_mask &= np.isfinite(df["lambda_h"]) & np.isfinite(df["lambda_v"])
    valid_mask &= np.isfinite(df["proxy_vorticity_abs_mean"]) & np.isfinite(df["proxy_omega_q_850_300"])
    d = df.loc[valid_mask].reset_index(drop=True)
    d.to_csv(outdir / "comparison_dataset.csv", index=False)

    x = d["lambda_h"].to_numpy(dtype=float)
    y = d["lambda_v"].to_numpy(dtype=float)
    r, p = pearsonr(x, y)
    corr_df = pd.DataFrame(
        [
            {
                "n": int(len(d)),
                "pearson_r": float(r),
                "r2": float(r * r),
                "pearson_p": float(p),
                "slope": float(linregress(x, y).slope),
                "intercept": float(linregress(x, y).intercept),
            }
        ]
    )
    corr_df.to_csv(outdir / "lambda_correlation_stats.csv", index=False)

    y_arr = d["residual_base_res0"].to_numpy(dtype=float)
    ctrl = d["n_density_ctrl_z"].to_numpy(dtype=float)
    lam_h = d["lambda_h"].to_numpy(dtype=float)
    lam_v = d["lambda_v"].to_numpy(dtype=float)

    model_df, split_df, perm_df, preds = _evaluate_model_set(
        y=y_arr,
        ctrl=ctrl,
        lambda_h=lam_h,
        lambda_v=lam_v,
        ridge_alpha=ridge_alpha,
        n_folds=n_folds,
        n_perm=n_perm,
        perm_block=perm_block,
        seed=seed,
    )

    model_df.to_csv(outdir / "model_comparison.csv", index=False)
    split_df.to_csv(outdir / "model_split_metrics.csv", index=False)
    perm_df.to_csv(outdir / "model_permutation_metrics.csv", index=False)

    q_vort = _quartile_gain_table(
        y=y_arr,
        yhat_base=preds["baseline_ctrl"],
        yhat_by_model=preds,
        proxy=d["proxy_vorticity_abs_mean"].to_numpy(dtype=float),
        proxy_name="vorticity_abs_mean",
        q=4,
    )
    q_omega = _quartile_gain_table(
        y=y_arr,
        yhat_base=preds["baseline_ctrl"],
        yhat_by_model=preds,
        proxy=d["proxy_omega_q_850_300"].to_numpy(dtype=float),
        proxy_name="omega_q_850_300",
        q=4,
    )
    q_df = pd.concat([q_vort, q_omega], ignore_index=True)
    q_df.to_csv(outdir / "quartile_gain_comparison.csv", index=False)

    context_df = _load_context_summaries(summary_paths)
    context_df.to_csv(outdir / "context_run_summary.csv", index=False)

    diagnosis = _artifact_diagnosis(corr_df=corr_df, model_df=model_df, q_df=q_df)
    with (outdir / "artifact_diagnosis.json").open("w", encoding="utf-8") as f:
        json.dump(diagnosis, f, ensure_ascii=False, indent=2)

    _plot_lambda_scatter(d, outdir / "plot_lambda_scatter.png", corr_df.iloc[0].to_dict())
    _plot_model_gains(model_df, outdir / "plot_model_gains.png")
    _plot_quartile_gains(q_df, "vorticity_abs_mean", outdir / "plot_quartile_vorticity.png")
    _plot_quartile_gains(q_df, "omega_q_850_300", outdir / "plot_quartile_omega_q.png")
    _plot_delta_abs_error(y_arr, preds, outdir / "plot_delta_abs_error_vertical_minus_horizontal.png")

    _save_report(
        outdir=outdir,
        corr_df=corr_df,
        model_df=model_df,
        diagnosis=diagnosis,
        context_df=context_df,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizontal-timeseries",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated/experiment_M_timeseries.csv"),
    )
    parser.add_argument(
        "--vertical-timeseries",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    parser.add_argument(
        "--vertical-input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_horizontal_vertical_compare"),
    )
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--perm-block", type=int, default=24)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260301)
    args = parser.parse_args()

    summary_paths = {
        "horizontal_v3_calibrated": Path("clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated/experiment_M_summary.csv"),
        "vertical_v4_entropy_raw": Path("clean_experiments/results/experiment_M_cosmo_flow_v4_vertical_entropy/experiment_M_summary.csv"),
        "vertical_v4_macro_calibrated": Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_summary.csv"),
    }

    run_compare(
        horizontal_timeseries=args.horizontal_timeseries,
        vertical_timeseries=args.vertical_timeseries,
        vertical_input_nc=args.vertical_input_nc,
        outdir=args.outdir,
        n_folds=args.n_folds,
        n_perm=args.n_perm,
        perm_block=args.perm_block,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        summary_paths=summary_paths,
    )
    print(f"[compare] done -> {args.outdir}")


if __name__ == "__main__":
    main()
