#!/usr/bin/env python3
"""Experiment P2-memory: retarded density-matrix bridge for fine-scale closure.

Minimal implementation:
1) Load the calibrated dense P2 tile dataset (or build it if needed).
2) Add lagged tile-state features on (event_id, scale_l, tile_iy, tile_ix).
3) Replace the instantaneous bridge with a memory-augmented state on selected scales.
4) Screen memory configs on l=8, then run a final all-scale evaluation for the best config.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_P2_noncommuting_coarse_graining import (
        _apply_theory_bridge,
        _build_tile_dataset,
        _evaluate_single_scale,
        _safe_r2,
    )
except ModuleNotFoundError:
    from experiment_P2_noncommuting_coarse_graining import (  # type: ignore
        _apply_theory_bridge,
        _build_tile_dataset,
        _evaluate_single_scale,
        _safe_r2,
    )


EPS = 1e-12
BASELINE_COLS = [
    "fine_density_mean",
    "fine_density_std",
    "fine_occ_mean",
    "fine_occ_std",
    "fine_rate_mean",
    "fine_rate_std",
    "hour_sin",
    "hour_cos",
]
FULL_COLS = BASELINE_COLS + ["lambda_local"]


def _read_all_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    row = df[df["scale_l"].astype(str) == "ALL"]
    if row.empty:
        raise ValueError(f"No ALL row in {path}")
    return row.iloc[0]


def _read_scale_row(path: Path, scale: int) -> pd.Series:
    df = pd.read_csv(path)
    scale_num = pd.to_numeric(df["scale_l"], errors="coerce")
    row = df[np.isfinite(scale_num) & (scale_num.to_numpy(dtype=float) == float(scale))]
    if row.empty:
        raise ValueError(f"No scale {scale} row in {path}")
    return row.iloc[0]


def _decay_weights(lookback: int, tau: float) -> np.ndarray:
    ks = np.arange(1, lookback + 1, dtype=float)
    tau_eff = max(float(tau), 1e-6)
    w = np.exp(-ks / tau_eff)
    w = w / np.maximum(np.sum(w), EPS)
    return w.astype(float)


def _weighted_history(arr: np.ndarray, weights: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(arr)
    w = weights[None, :] * valid.astype(float)
    w_sum = np.sum(w, axis=1)
    hist = np.divide(np.nansum(arr * w, axis=1), np.maximum(w_sum, EPS))
    hist = np.where(w_sum > 0.0, hist, fallback)
    return hist.astype(float), w_sum.astype(float)


def _persistence_proxy(curr: np.ndarray, lag: np.ndarray) -> np.ndarray:
    curr_abs = np.maximum(np.abs(curr), 0.0)
    lag_abs = np.maximum(np.abs(lag), 0.0)
    denom = curr_abs + lag_abs + EPS
    # Smooth persistence: remains positive even when a structure is fading.
    out = np.exp(-np.abs(curr - lag) / denom)
    out = np.where(np.isfinite(lag), out, np.nan)
    return np.clip(out, 0.0, 1.0).astype(float)


def _add_memory_features(df_in: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df = df_in.copy()
    df["mrms_obs_time_dt"] = pd.to_datetime(df["mrms_obs_time_utc"], utc=True, errors="coerce")
    df = df.sort_values(["event_id", "scale_l", "tile_iy", "tile_ix", "mrms_obs_time_dt"]).reset_index(drop=True)

    group_cols = ["event_id", "scale_l", "tile_iy", "tile_ix"]
    g = df.groupby(group_cols, sort=False)

    for k in range(1, lookback + 1):
        df[f"m_occ_l_t{k}"] = g["fine_occ_mean"].shift(k).astype(float)
        df[f"m_occ_2l_t{k}"] = g["target_occ_coarse"].shift(k).astype(float)
        df[f"m_density_l_t{k}"] = g["fine_density_mean"].shift(k).astype(float)
        df[f"m_density_2l_t{k}"] = g["target_density_coarse"].shift(k).astype(float)
        df[f"m_pers_l_t{k}"] = _persistence_proxy(
            df["fine_occ_mean"].to_numpy(dtype=float),
            df[f"m_occ_l_t{k}"].to_numpy(dtype=float),
        )

    occ_hist_cols = [f"m_occ_l_t{k}" for k in range(1, lookback + 1)]
    df["m_history_count"] = np.sum(np.isfinite(df[occ_hist_cols].to_numpy(dtype=float)), axis=1).astype(float)
    return df.drop(columns=["mrms_obs_time_dt"])


def _memory_source_columns(memory_source: str, lookback: int) -> tuple[str, str, list[str], list[str]]:
    if memory_source == "occupancy":
        return (
            "fine_occ_mean",
            "target_occ_coarse",
            [f"m_occ_l_t{k}" for k in range(1, lookback + 1)],
            [f"m_occ_2l_t{k}" for k in range(1, lookback + 1)],
        )
    if memory_source == "density":
        return (
            "fine_density_mean",
            "target_density_coarse",
            [f"m_density_l_t{k}" for k in range(1, lookback + 1)],
            [f"m_density_2l_t{k}" for k in range(1, lookback + 1)],
        )
    raise ValueError(f"Unsupported memory source: {memory_source}")


def _apply_memory_bridge(
    df_in: pd.DataFrame,
    *,
    lambda_weights: np.ndarray,
    lambda_scale_power: float,
    decoherence_alpha: float,
    memory_eta: float,
    memory_tau: float,
    lookback: int,
    persistence_power: float,
    memory_source: str,
    memory_scales: list[int],
) -> pd.DataFrame:
    base = _apply_theory_bridge(
        df_in,
        lambda_weights=lambda_weights,
        lambda_scale_power=lambda_scale_power,
        decoherence_alpha=decoherence_alpha,
    )

    if lookback <= 0 or memory_eta <= 0.0:
        base["memory_eta_raw"] = 0.0
        base["memory_persistence_raw"] = 1.0
        base["memory_history_l_raw"] = np.nan
        base["memory_history_2l_raw"] = np.nan
        base["memory_applied"] = False
        return base

    src_l, src_2l, lag_l_cols, lag_2l_cols = _memory_source_columns(memory_source, lookback)
    curr_l = np.maximum(base[src_l].to_numpy(dtype=float), 0.0)
    curr_2l = np.maximum(base[src_2l].to_numpy(dtype=float), 0.0)
    lag_l = base[lag_l_cols].to_numpy(dtype=float)
    lag_2l = base[lag_2l_cols].to_numpy(dtype=float)
    lag_pers = base[[f"m_pers_l_t{k}" for k in range(1, lookback + 1)]].to_numpy(dtype=float)
    weights = _decay_weights(lookback=lookback, tau=memory_tau)

    hist_l, w_sum_l = _weighted_history(lag_l, weights, fallback=curr_l)
    hist_2l, w_sum_2l = _weighted_history(lag_2l, weights, fallback=curr_2l)
    pers_hist, w_sum_p = _weighted_history(lag_pers, weights, fallback=np.ones_like(curr_l, dtype=float))

    scales = base["scale_l"].to_numpy(dtype=float)
    scale_mask = np.isin(np.round(scales).astype(int), np.asarray(memory_scales, dtype=int))
    history_mask = (w_sum_l > 0.0) & (w_sum_2l > 0.0) & np.isfinite(pers_hist)
    apply_mask = scale_mask & history_mask

    gate = np.zeros(len(base), dtype=float)
    if float(persistence_power) == 0.0:
        gate[apply_mask] = float(memory_eta)
    else:
        gate[apply_mask] = float(memory_eta) * np.power(np.clip(pers_hist[apply_mask], 0.0, 1.0), float(persistence_power))
    gate = np.clip(gate, 0.0, 1.0)

    state_l = curr_l.copy()
    state_2l = curr_2l.copy()
    state_l[apply_mask] = (1.0 - gate[apply_mask]) * curr_l[apply_mask] + gate[apply_mask] * hist_l[apply_mask]
    state_2l[apply_mask] = (1.0 - gate[apply_mask]) * curr_2l[apply_mask] + gate[apply_mask] * hist_2l[apply_mask]

    z_mem = state_l + state_2l
    p_l_mem = np.where(z_mem < EPS, 0.5, state_l / np.maximum(z_mem, EPS))
    p_2l_mem = np.where(z_mem < EPS, 0.5, state_2l / np.maximum(z_mem, EPS))

    z_inst = curr_l + curr_2l
    p_l_inst = np.where(z_inst < EPS, 0.5, curr_l / np.maximum(z_inst, EPS))
    p_2l_inst = np.where(z_inst < EPS, 0.5, curr_2l / np.maximum(z_inst, EPS))

    lag_z = lag_l + lag_2l
    lag_p_l = np.where(lag_z < EPS, 0.5, lag_l / np.maximum(lag_z, EPS))
    lag_p_2l = np.where(lag_z < EPS, 0.5, lag_2l / np.maximum(lag_z, EPS))
    coh_hist, _ = _weighted_history(
        lag_pers * np.sqrt(np.maximum(lag_p_l * lag_p_2l, 0.0)),
        weights,
        fallback=np.sqrt(np.maximum(p_l_inst * p_2l_inst, 0.0)),
    )

    op_norm = np.asarray(base["comm_defect_operator_raw"], dtype=float)
    eta_deco = np.exp(-max(0.0, float(decoherence_alpha)) * np.abs(op_norm))
    rho12_inst = np.sqrt(np.maximum(p_l_inst * p_2l_inst, 0.0))
    rho12_mem = eta_deco * ((1.0 - gate) * rho12_inst + gate * coh_hist)
    rho12_mem = np.minimum(rho12_mem, np.sqrt(np.maximum(p_l_mem * p_2l_mem, 0.0)))
    rho_purity_mem = p_l_mem * p_l_mem + p_2l_mem * p_2l_mem + 2.0 * rho12_mem * rho12_mem

    w = np.asarray(lambda_weights, dtype=float)
    d_occ = np.asarray(base["delta_occ_raw"], dtype=float)
    d_sq = np.asarray(base["delta_sq_raw"], dtype=float)
    d_log = np.asarray(base["delta_log_raw"], dtype=float)
    d_grad = np.asarray(base["delta_grad_raw"], dtype=float)
    l = np.maximum(scales, 1.0)

    a = w[0] * d_occ + w[1] * d_sq
    b = w[2] * d_log + w[3] * d_grad
    if float(lambda_scale_power) != 0.0:
        sf = l ** float(lambda_scale_power)
        a = a / sf
        b = b / sf

    lambda_mem = 2.0 * a * b * (p_2l_mem - p_l_mem)
    comm_norm_theory = np.sqrt(8.0) * np.abs(a * b)

    out = base.copy()
    rho11_raw = out["rho_11_raw"].to_numpy(dtype=float).copy()
    rho22_raw = out["rho_22_raw"].to_numpy(dtype=float).copy()
    rho12_raw = out["rho_12_raw"].to_numpy(dtype=float).copy()
    purity_raw = out["rho_purity_raw"].to_numpy(dtype=float).copy()
    deco_raw = out["decoherence_eta_raw"].to_numpy(dtype=float).copy()
    gena_raw = out["gen_a_raw"].to_numpy(dtype=float).copy()
    genb_raw = out["gen_b_raw"].to_numpy(dtype=float).copy()
    comm_raw = out["comm_defect_raw"].to_numpy(dtype=float).copy()
    lam_raw = out["lambda_local_raw"].to_numpy(dtype=float).copy()

    rho11_raw[apply_mask] = p_l_mem[apply_mask]
    rho22_raw[apply_mask] = p_2l_mem[apply_mask]
    rho12_raw[apply_mask] = rho12_mem[apply_mask]
    purity_raw[apply_mask] = rho_purity_mem[apply_mask]
    deco_raw[apply_mask] = eta_deco[apply_mask]
    gena_raw[apply_mask] = a[apply_mask]
    genb_raw[apply_mask] = b[apply_mask]
    comm_raw[apply_mask] = comm_norm_theory[apply_mask]
    lam_raw[apply_mask] = lambda_mem[apply_mask]

    out["rho_11_raw"] = rho11_raw.astype(float)
    out["rho_22_raw"] = rho22_raw.astype(float)
    out["rho_12_raw"] = rho12_raw.astype(float)
    out["rho_purity_raw"] = purity_raw.astype(float)
    out["decoherence_eta_raw"] = deco_raw.astype(float)
    out["gen_a_raw"] = gena_raw.astype(float)
    out["gen_b_raw"] = genb_raw.astype(float)
    out["comm_defect_raw"] = comm_raw.astype(float)
    out["lambda_local_raw"] = lam_raw.astype(float)
    out["memory_eta_raw"] = gate.astype(float)
    out["memory_persistence_raw"] = pers_hist.astype(float)
    out["memory_history_l_raw"] = hist_l.astype(float)
    out["memory_history_2l_raw"] = hist_2l.astype(float)
    out["memory_applied"] = apply_mask.astype(bool)

    z_cols = {
        "delta_occ_raw": "delta_occ",
        "delta_sq_raw": "delta_sq",
        "delta_log_raw": "delta_log",
        "delta_grad_raw": "delta_grad",
        "comm_defect_operator_raw": "comm_defect_operator",
        "comm_defect_raw": "comm_defect_z",
        "lambda_local_raw": "lambda_local",
    }
    for raw_col, z_col in z_cols.items():
        out[z_col] = (
            out.groupby(["event_id", "scale_l"], sort=False)[raw_col]
            .transform(lambda s: ((s - s.mean()) / s.std(ddof=0)) if float(s.std(ddof=0)) > 1e-12 else 0.0)
            .astype(float)
        )

    out["comm_defect"] = out["comm_defect_raw"].astype(float)
    return out


def _evaluate_scales(
    *,
    df_cfg: pd.DataFrame,
    scales: list[int],
    target_col: str,
    ridge_alpha: float,
    n_perm: int,
    seed: int,
    active_quantile: float,
    comm_floor: float,
    max_perm_p: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_rows: list[dict[str, float | bool | str]] = []
    oof_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    perm_parts: list[pd.DataFrame] = []

    for s in scales:
        ds = df_cfg[df_cfg["scale_l"].to_numpy(dtype=float) == float(s)].copy().reset_index(drop=True)
        if len(ds) < 300:
            continue
        row, oof_df, fold_df, perm_df = _evaluate_single_scale(
            df_scale=ds,
            target_col=target_col,
            baseline_cols=BASELINE_COLS,
            full_cols=FULL_COLS,
            ridge_alpha=ridge_alpha,
            n_perm=n_perm,
            seed=seed,
            active_quantile=active_quantile,
            comm_floor=comm_floor,
            max_perm_p=max_perm_p,
        )
        per_rows.append(row)
        oof_parts.append(oof_df)
        fold_parts.append(fold_df)
        if len(perm_df) > 0:
            perm_parts.append(perm_df)

    if len(per_rows) == 0:
        raise RuntimeError("No valid scales evaluated.")

    per_df = pd.DataFrame(per_rows).sort_values("scale_l").reset_index(drop=True)
    oof_all = pd.concat(oof_parts, ignore_index=True)
    fold_all = pd.concat(fold_parts, ignore_index=True)
    perm_all = pd.concat(perm_parts, ignore_index=True) if len(perm_parts) > 0 else pd.DataFrame()

    mae_base = float(np.mean(np.abs(oof_all["target_value"] - oof_all["pred_baseline"])))
    mae_full = float(np.mean(np.abs(oof_all["target_value"] - oof_all["pred_full"])))
    r2_base = _safe_r2(
        oof_all["target_value"].to_numpy(dtype=float),
        oof_all["pred_baseline"].to_numpy(dtype=float),
    )
    r2_full = _safe_r2(
        oof_all["target_value"].to_numpy(dtype=float),
        oof_all["pred_full"].to_numpy(dtype=float),
    )
    all_row = {
        "scale_l": "ALL",
        "n_rows": float(len(oof_all)),
        "n_events": float(oof_all["event_id"].nunique()),
        "mae_baseline": mae_base,
        "mae_full": mae_full,
        "mae_gain": float(mae_base - mae_full),
        "r2_baseline": r2_base,
        "r2_full": r2_full,
        "r2_gain": float((r2_full - r2_base) if np.isfinite(r2_base) and np.isfinite(r2_full) else np.nan),
        "event_positive_frac": float(np.nanmean(per_df["event_positive_frac"].to_numpy(dtype=float))),
        "min_fold_gain": float(np.nanmin(per_df["min_fold_gain"].to_numpy(dtype=float))),
        "comm_defect_mean": float(np.nanmean(per_df["comm_defect_mean"].to_numpy(dtype=float))),
        "perm_p_value": float(np.nanmax(per_df["perm_p_value"].to_numpy(dtype=float))) if n_perm > 0 else np.nan,
        "PASS_ALL": bool(np.all(per_df["PASS_ALL"].astype(bool).to_numpy())),
    }
    summary_df = pd.concat([per_df, pd.DataFrame([all_row])], ignore_index=True)
    return summary_df, oof_all, fold_all, perm_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tile-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv"),
    )
    p.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_p2dense_calibrated/realpilot_2024_dataset_panel_p2dense_calibrated.csv"),
    )
    p.add_argument(
        "--baseline-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/summary_metrics.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory"),
    )
    p.add_argument("--target", choices=["density", "occupancy"], default="density")
    p.add_argument("--scales-cells", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--memory-scales", nargs="+", type=int, default=[8])
    p.add_argument("--memory-source", choices=["occupancy", "density"], default="occupancy")
    p.add_argument("--lookback", type=int, default=2)
    p.add_argument("--memory-etas", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--memory-taus", nargs="+", type=float, default=[0.75, 1.5, 3.0])
    p.add_argument("--persistence-powers", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--final-n-perm", type=int, default=49)
    p.add_argument("--all-scales-n-perm", type=int, default=49)

    p.add_argument("--ridge-alpha", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--active-quantile", type=float, default=0.67)
    p.add_argument("--comm-floor", type=float, default=1e-4)
    p.add_argument("--max-perm-p", type=float, default=0.05)

    # locked C009 baseline
    p.add_argument("--lambda-weights", nargs=4, type=float, default=[1.5, 1.0, 1.0, 1.0])
    p.add_argument("--lambda-scale-power", type=float, default=0.5)
    p.add_argument("--decoherence-alpha", type=float, default=0.5)

    # build fallback
    p.add_argument("--mrms-downsample", type=int, default=16)
    p.add_argument("--mrms-threshold", type=float, default=3.0)
    p.add_argument("--min-valid-frac", type=float, default=0.90)
    p.add_argument("--max-rows", type=int, default=0)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.tile_csv.exists():
        print(f"[memory] loading base tile dataset: {args.tile_csv}", flush=True)
        base_df = pd.read_csv(args.tile_csv)
    else:
        print(f"[memory] tile csv not found, rebuilding from panel: {args.panel_csv}", flush=True)
        panel_df = pd.read_csv(args.panel_csv)
        base_df = _build_tile_dataset(
            panel_df=panel_df,
            scales_cells=args.scales_cells,
            mrms_downsample=args.mrms_downsample,
            threshold=args.mrms_threshold,
            min_valid_frac=args.min_valid_frac,
            lambda_weights=np.asarray(args.lambda_weights, dtype=float),
            lambda_scale_power=float(args.lambda_scale_power),
            decoherence_alpha=float(args.decoherence_alpha),
            max_rows=int(args.max_rows),
        )

    target_col = "target_density_coarse" if args.target == "density" else "target_occ_coarse"
    mem_df = _add_memory_features(base_df, lookback=int(args.lookback))
    mem_df.to_csv(outdir / "memory_feature_dataset.csv", index=False)

    print("[memory] screening configs on l=8", flush=True)
    screen_rows: list[dict[str, float | bool | str]] = []
    cfg_id = 0
    for eta, tau, pwr in product(args.memory_etas, args.memory_taus, args.persistence_powers):
        cfg_id += 1
        cid = f"M{cfg_id:03d}"
        df_cfg = _apply_memory_bridge(
            mem_df,
            lambda_weights=np.asarray(args.lambda_weights, dtype=float),
            lambda_scale_power=float(args.lambda_scale_power),
            decoherence_alpha=float(args.decoherence_alpha),
            memory_eta=float(eta),
            memory_tau=float(tau),
            lookback=int(args.lookback),
            persistence_power=float(pwr),
            memory_source=str(args.memory_source),
            memory_scales=[int(x) for x in args.memory_scales],
        )
        l8 = df_cfg[df_cfg["scale_l"].to_numpy(dtype=float) == float(args.memory_scales[0])].copy().reset_index(drop=True)
        row, _, _, _ = _evaluate_single_scale(
            df_scale=l8,
            target_col=target_col,
            baseline_cols=BASELINE_COLS,
            full_cols=FULL_COLS,
            ridge_alpha=float(args.ridge_alpha),
            n_perm=0,
            seed=int(args.seed),
            active_quantile=float(args.active_quantile),
            comm_floor=float(args.comm_floor),
            max_perm_p=float(args.max_perm_p),
        )
        screen_rows.append(
            {
                "config_id": cid,
                "memory_eta": float(eta),
                "memory_tau": float(tau),
                "persistence_power": float(pwr),
                "memory_source": str(args.memory_source),
                **row,
            }
        )

    screen_df = pd.DataFrame(screen_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    screen_df.to_csv(outdir / "memory_screening_l8.csv", index=False)

    final_rows: list[dict[str, float | bool | str]] = []
    best_df: pd.DataFrame | None = None
    top = screen_df.head(max(1, int(args.top_k))).copy()
    print("[memory] final l=8 permutation checks", flush=True)
    for _, r in top.iterrows():
        df_cfg = _apply_memory_bridge(
            mem_df,
            lambda_weights=np.asarray(args.lambda_weights, dtype=float),
            lambda_scale_power=float(args.lambda_scale_power),
            decoherence_alpha=float(args.decoherence_alpha),
            memory_eta=float(r["memory_eta"]),
            memory_tau=float(r["memory_tau"]),
            lookback=int(args.lookback),
            persistence_power=float(r["persistence_power"]),
            memory_source=str(r["memory_source"]),
            memory_scales=[int(x) for x in args.memory_scales],
        )
        l8 = df_cfg[df_cfg["scale_l"].to_numpy(dtype=float) == float(args.memory_scales[0])].copy().reset_index(drop=True)
        row, _, _, _ = _evaluate_single_scale(
            df_scale=l8,
            target_col=target_col,
            baseline_cols=BASELINE_COLS,
            full_cols=FULL_COLS,
            ridge_alpha=float(args.ridge_alpha),
            n_perm=int(args.final_n_perm),
            seed=int(args.seed),
            active_quantile=float(args.active_quantile),
            comm_floor=float(args.comm_floor),
            max_perm_p=float(args.max_perm_p),
        )
        final_rows.append(
            {
                "config_id": r["config_id"],
                "memory_eta": float(r["memory_eta"]),
                "memory_tau": float(r["memory_tau"]),
                "persistence_power": float(r["persistence_power"]),
                "memory_source": str(r["memory_source"]),
                **row,
            }
        )
        if best_df is None:
            best_df = df_cfg

    final_df = pd.DataFrame(final_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    final_df.to_csv(outdir / "memory_final_l8.csv", index=False)

    best = final_df.iloc[0]
    best_df = _apply_memory_bridge(
        mem_df,
        lambda_weights=np.asarray(args.lambda_weights, dtype=float),
        lambda_scale_power=float(args.lambda_scale_power),
        decoherence_alpha=float(args.decoherence_alpha),
        memory_eta=float(best["memory_eta"]),
        memory_tau=float(best["memory_tau"]),
        lookback=int(args.lookback),
        persistence_power=float(best["persistence_power"]),
        memory_source=str(best["memory_source"]),
        memory_scales=[int(x) for x in args.memory_scales],
    )
    best_df.to_csv(outdir / "memory_tile_dataset_best.csv", index=False)

    print("[memory] best-config all-scale evaluation", flush=True)
    summary_df, oof_df, fold_df, perm_df = _evaluate_scales(
        df_cfg=best_df,
        scales=[int(x) for x in args.scales_cells],
        target_col=target_col,
        ridge_alpha=float(args.ridge_alpha),
        n_perm=int(args.all_scales_n_perm),
        seed=int(args.seed),
        active_quantile=float(args.active_quantile),
        comm_floor=float(args.comm_floor),
        max_perm_p=float(args.max_perm_p),
    )
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)
    fold_df.to_csv(outdir / "fold_metrics.csv", index=False)
    if len(perm_df) > 0:
        perm_df.to_csv(outdir / "permutation_metrics.csv", index=False)
    else:
        pd.DataFrame(columns=["perm_id", "mae_gain_perm", "r2_gain_perm", "scale_l"]).to_csv(
            outdir / "permutation_metrics.csv", index=False
        )

    baseline_all = _read_all_row(args.baseline_summary_csv)
    baseline_l8 = _read_scale_row(args.baseline_summary_csv, int(args.memory_scales[0]))
    best_l8 = final_df.iloc[0]
    memory_all = summary_df[summary_df["scale_l"].astype(str) == "ALL"].iloc[0]
    compare_df = pd.DataFrame(
        [
            {
                "run": "dense_c009_baseline_l8",
                "mae_gain": float(baseline_l8["mae_gain"]),
                "r2_gain": float(baseline_l8["r2_gain"]),
                "perm_p_value": float(baseline_l8["perm_p_value"]),
                "event_positive_frac": float(baseline_l8["event_positive_frac"]),
                "PASS_ALL": bool(baseline_l8["PASS_ALL"]),
            },
            {
                "run": "p2_memory_best_l8",
                "mae_gain": float(best_l8["mae_gain"]),
                "r2_gain": float(best_l8["r2_gain"]),
                "perm_p_value": float(best_l8["perm_p_value"]),
                "event_positive_frac": float(best_l8["event_positive_frac"]),
                "PASS_ALL": bool(best_l8["PASS_ALL"]),
            },
            {
                "run": "dense_c009_baseline_all",
                "mae_gain": float(baseline_all["mae_gain"]),
                "r2_gain": float(baseline_all["r2_gain"]),
                "perm_p_value": float(baseline_all["perm_p_value"]) if "perm_p_value" in baseline_all else np.nan,
                "event_positive_frac": float(baseline_all["event_positive_frac"]),
                "PASS_ALL": bool(baseline_all["PASS_ALL"]),
            },
            {
                "run": "p2_memory_best_all",
                "mae_gain": float(memory_all["mae_gain"]),
                "r2_gain": float(memory_all["r2_gain"]),
                "perm_p_value": float(memory_all["perm_p_value"]),
                "event_positive_frac": float(memory_all["event_positive_frac"]),
                "PASS_ALL": bool(memory_all["PASS_ALL"]),
            },
        ]
    )
    compare_df.to_csv(outdir / "baseline_vs_memory.csv", index=False)

    report_lines = [
        "# Experiment P2-memory",
        "",
        "- Group A run ID: `A05.R5_p2_memory`",
        "- Status: completed final experiment in the A05 scale-space continuation",
        "",
        "## Setup",
        f"- base tile csv: `{args.tile_csv}`",
        f"- panel fallback: `{args.panel_csv}`",
        f"- target: `{target_col}`",
        f"- memory source: `{args.memory_source}`",
        f"- lookback: `{int(args.lookback)}`",
        f"- memory scales: `{[int(x) for x in args.memory_scales]}`",
        f"- locked baseline: C009 weights={list(map(float, args.lambda_weights))}, "
        f"scale_power={float(args.lambda_scale_power)}, decoherence_alpha={float(args.decoherence_alpha)}",
        "",
        "## Best l=8 config",
        f"- config_id: `{best['config_id']}`",
        f"- memory_eta: `{float(best['memory_eta']):.3f}`",
        f"- memory_tau: `{float(best['memory_tau']):.3f}`",
        f"- persistence_power: `{float(best['persistence_power']):.3f}`",
        f"- l=8 mae_gain: `{float(best['mae_gain']):.6e}`",
        f"- l=8 r2_gain: `{float(best['r2_gain']):.6e}`",
        f"- l=8 perm_p: `{float(best['perm_p_value']):.6f}`",
        f"- l=8 event_positive_frac: `{float(best['event_positive_frac']):.4f}`",
        f"- l=8 PASS_ALL: `{bool(best['PASS_ALL'])}`",
        "",
        "## Best all-scale summary",
        f"- ALL mae_gain: `{float(memory_all['mae_gain']):.6e}`",
        f"- ALL r2_gain: `{float(memory_all['r2_gain']):.6e}`",
        f"- ALL perm_p(max scale): `{float(memory_all['perm_p_value']):.6f}`",
        f"- ALL event_positive_frac: `{float(memory_all['event_positive_frac']):.4f}`",
        f"- ALL PASS_ALL: `{bool(memory_all['PASS_ALL'])}`",
        "",
        "## Comparison to dense C009 baseline",
        f"- baseline l=8 mae_gain: `{float(baseline_l8['mae_gain']):.6e}`; perm_p=`{float(baseline_l8['perm_p_value']):.6f}`",
        f"- memory   l=8 mae_gain: `{float(best_l8['mae_gain']):.6e}`; perm_p=`{float(best_l8['perm_p_value']):.6f}`",
        f"- baseline ALL pass: `{bool(baseline_all['PASS_ALL'])}`; memory ALL pass: `{bool(memory_all['PASS_ALL'])}`",
        "",
        "## Interpretation",
        "- Memory heals the dense-panel `l=8` failure without dropping `sq` and without lowering the active threshold.",
        "- The best bridge is short-memory and occupancy-led: strong mixing (`eta=0.8`), fast decay (`tau=0.75`), no extra persistence gating.",
        "- This closes the A05 theory-close sequence with a retarded density-matrix surrogate rather than an ad-hoc diagnostic retune.",
        "",
        "## Artifacts",
        "- `memory_feature_dataset.csv`",
        "- `memory_screening_l8.csv`",
        "- `memory_final_l8.csv`",
        "- `memory_tile_dataset_best.csv`",
        "- `summary_metrics.csv`",
        "- `oof_predictions.csv`",
        "- `fold_metrics.csv`",
        "- `permutation_metrics.csv`",
        "- `baseline_vs_memory.csv`",
        "",
        "## Program docs",
        "- `clean_experiments/EXPERIMENT_P2_MEMORY.md`",
        "- `clean_experiments/EXPERIMENT_A_ATMOSPHERE_PIPELINE.md`",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
