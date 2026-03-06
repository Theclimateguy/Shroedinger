#!/usr/bin/env python3
"""Narrow diagnostic block for P2 transferability issues at l=8.

Checks implemented:
1) Matched-event test (sparse vs dense on same event pool).
2) Scale-local ablation at l=8 (alpha, w_occ, lambda_scale_power).
3) Operator attribution (occ/sq/log/grad).
4) Resolution check for l=8 (mrms_downsample x threshold grid).
5) Regime split for l=8 (active-core / transition / calm).
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


def _run_l8_eval(
    *,
    df_cfg: pd.DataFrame,
    target_col: str,
    ridge_alpha: float,
    n_perm: int,
    seed: int,
    active_quantile: float,
    comm_floor: float,
    max_perm_p: float,
) -> tuple[dict[str, float | bool | str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = df_cfg[df_cfg["scale_l"].to_numpy(dtype=float) == 8.0].copy().reset_index(drop=True)
    if len(ds) < 300:
        raise RuntimeError("Too few l=8 rows for evaluation.")
    return _evaluate_single_scale(
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


def _read_all_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    row = df[df["scale_l"].astype(str) == "ALL"]
    if row.empty:
        raise ValueError(f"No ALL row in {path}")
    return row.iloc[0]


def _bool(x: object) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _assign_activity_regime(occ: np.ndarray) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Split occupancy into calm/transition/active_core with zero-aware fallback."""
    q1 = float(np.nanquantile(occ, 1.0 / 3.0))
    q2 = float(np.nanquantile(occ, 2.0 / 3.0))

    reg = np.full(len(occ), "transition", dtype=object)
    method = "quantile_terciles"
    positive_q67 = float("nan")
    positive_count = int(np.sum(occ > 0.0))

    if np.isclose(q1, q2):
        pos = occ[occ > 0.0]
        if len(pos) >= 3:
            positive_q67 = float(np.nanquantile(pos, 2.0 / 3.0))
            reg[occ <= 0.0] = "calm"
            reg[(occ > 0.0) & (occ <= positive_q67)] = "transition"
            reg[occ > positive_q67] = "active_core"
            method = "zero_aware_positive_quantile"
        else:
            # Degenerate case: force a stable 3-way split by rank.
            ranks = pd.Series(occ).rank(method="first", pct=True).to_numpy(dtype=float)
            reg[ranks <= (1.0 / 3.0)] = "calm"
            reg[(ranks > (1.0 / 3.0)) & (ranks <= (2.0 / 3.0))] = "transition"
            reg[ranks > (2.0 / 3.0)] = "active_core"
            method = "rank_terciles_fallback"
    else:
        reg[occ <= q1] = "calm"
        reg[(occ > q1) & (occ <= q2)] = "transition"
        reg[occ > q2] = "active_core"

    meta = {
        "split_method": method,
        "q33_raw": q1,
        "q67_raw": q2,
        "positive_q67": positive_q67,
        "positive_count": positive_count,
    }
    return reg, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sparse-panel-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv"),
    )
    p.add_argument(
        "--dense-panel-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_p2dense_calibrated/realpilot_2024_dataset_panel_p2dense_calibrated.csv"),
    )
    p.add_argument(
        "--dense-events-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_p2dense_calibrated/stable_events_dense.csv"),
    )
    p.add_argument(
        "--sparse-tile-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining_calibrated/p2_tile_dataset.csv"),
    )
    p.add_argument(
        "--dense-tile-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv"),
    )
    p.add_argument(
        "--sparse-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining_calibrated/summary_metrics.csv"),
    )
    p.add_argument(
        "--dense-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/summary_metrics.csv"),
    )
    p.add_argument(
        "--dense-oof-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/oof_predictions.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_l8_diagnostic_block"),
    )

    p.add_argument("--target", choices=["density", "occupancy"], default="density")
    p.add_argument("--ridge-alpha", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--active-quantile", type=float, default=0.67)
    p.add_argument("--comm-floor", type=float, default=1e-4)
    p.add_argument("--max-perm-p", type=float, default=0.05)

    # C009 defaults
    p.add_argument("--alpha-c009", type=float, default=0.5)
    p.add_argument("--lambda-scale-power-c009", type=float, default=0.5)
    p.add_argument("--weights-c009", nargs=4, type=float, default=[1.5, 1.0, 1.0, 1.0])
    p.add_argument("--mrms-downsample-c009", type=int, default=16)
    p.add_argument("--mrms-threshold-c009", type=float, default=3.0)
    p.add_argument("--min-valid-frac", type=float, default=0.90)

    # l=8 local ablation
    p.add_argument("--ablate-alphas", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--ablate-w-occ", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.5, 2.0])
    p.add_argument("--ablate-scale-powers", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--ablate-top-k", type=int, default=6)
    p.add_argument("--ablate-n-perm-final", type=int, default=49)

    # operator attribution
    p.add_argument("--attrib-n-perm", type=int, default=49)

    # resolution scan (l=8 only)
    p.add_argument("--res-downsamples", nargs="+", type=int, default=[12, 16, 20])
    p.add_argument("--res-thresholds", nargs="+", type=float, default=[2.5, 3.0, 3.5])
    p.add_argument("--res-top-k", type=int, default=3)
    p.add_argument("--res-n-perm-final", type=int, default=19)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    target_col = "target_density_coarse" if args.target == "density" else "target_occ_coarse"
    c009_w = np.asarray(args.weights_c009, dtype=float)

    sparse_panel = pd.read_csv(args.sparse_panel_csv)
    dense_panel = pd.read_csv(args.dense_panel_csv)
    sparse_tile = pd.read_csv(args.sparse_tile_csv)
    dense_tile = pd.read_csv(args.dense_tile_csv)

    dense_events_df = pd.read_csv(args.dense_events_csv)
    dense_event_ids = sorted(set(dense_events_df["event_id"].astype(str)))
    sparse_matched_panel = sparse_panel[sparse_panel["event_id"].astype(str).isin(dense_event_ids)].copy()
    sparse_matched_panel = sparse_matched_panel.sort_values(["event_id", "mrms_obs_time_utc"]).reset_index(drop=True)
    sparse_matched_panel.to_csv(outdir / "matched_sparse_panel_16events.csv", index=False)

    # 1) Matched-event test
    print("[diag] 1/5 matched-event test", flush=True)
    sparse_matched_tile = sparse_tile[sparse_tile["event_id"].astype(str).isin(dense_event_ids)].copy()
    sparse_matched_tile = sparse_matched_tile.reset_index(drop=True)

    s_match_summary, s_match_oof, s_match_fold, s_match_perm = _evaluate_scales(
        df_cfg=sparse_matched_tile,
        scales=[8, 16, 32],
        target_col=target_col,
        ridge_alpha=args.ridge_alpha,
        n_perm=49,
        seed=args.seed,
        active_quantile=args.active_quantile,
        comm_floor=args.comm_floor,
        max_perm_p=args.max_perm_p,
    )
    s_match_summary.to_csv(outdir / "matched_event_summary_sparse16.csv", index=False)
    s_match_oof.to_csv(outdir / "matched_event_oof_sparse16.csv", index=False)
    s_match_fold.to_csv(outdir / "matched_event_fold_sparse16.csv", index=False)
    if len(s_match_perm) > 0:
        s_match_perm.to_csv(outdir / "matched_event_perm_sparse16.csv", index=False)

    sparse_all = _read_all_row(args.sparse_summary_csv)
    dense_all = _read_all_row(args.dense_summary_csv)
    sparse_matched_all = s_match_summary[s_match_summary["scale_l"].astype(str) == "ALL"].iloc[0]

    matched_comp = pd.DataFrame(
        [
            {
                "panel": "sparse_72x24",
                "rows": int(len(sparse_panel)),
                "events": int(sparse_panel["event_id"].nunique()),
                "mae_gain_all": float(sparse_all["mae_gain"]),
                "r2_gain_all": float(sparse_all["r2_gain"]),
                "perm_p_all_max_scale": float(
                    pd.read_csv(args.sparse_summary_csv).query("scale_l != 'ALL'")["perm_p_value"].max()
                ),
                "event_positive_frac": float(sparse_all["event_positive_frac"]),
                "min_fold_gain": float(sparse_all["min_fold_gain"]),
                "pass_all": _bool(sparse_all["PASS_ALL"]),
            },
            {
                "panel": "sparse_matched_16events",
                "rows": int(len(sparse_matched_panel)),
                "events": int(sparse_matched_panel["event_id"].nunique()),
                "mae_gain_all": float(sparse_matched_all["mae_gain"]),
                "r2_gain_all": float(sparse_matched_all["r2_gain"]),
                "perm_p_all_max_scale": float(
                    s_match_summary[s_match_summary["scale_l"].astype(str) != "ALL"]["perm_p_value"].max()
                ),
                "event_positive_frac": float(sparse_matched_all["event_positive_frac"]),
                "min_fold_gain": float(sparse_matched_all["min_fold_gain"]),
                "pass_all": _bool(sparse_matched_all["PASS_ALL"]),
            },
            {
                "panel": "dense_240x16",
                "rows": int(len(dense_panel)),
                "events": int(dense_panel["event_id"].nunique()),
                "mae_gain_all": float(dense_all["mae_gain"]),
                "r2_gain_all": float(dense_all["r2_gain"]),
                "perm_p_all_max_scale": float(
                    pd.read_csv(args.dense_summary_csv).query("scale_l != 'ALL'")["perm_p_value"].max()
                ),
                "event_positive_frac": float(dense_all["event_positive_frac"]),
                "min_fold_gain": float(dense_all["min_fold_gain"]),
                "pass_all": _bool(dense_all["PASS_ALL"]),
            },
        ]
    )
    matched_comp.to_csv(outdir / "matched_event_comparison.csv", index=False)

    # 2) Scale-local ablation at l=8 (dense)
    print("[diag] 2/5 scale-local l=8 ablation", flush=True)
    dense_l8_base = dense_tile[dense_tile["scale_l"].to_numpy(dtype=float) == 8.0].copy().reset_index(drop=True)
    if len(dense_l8_base) < 500:
        raise RuntimeError("Dense l=8 base dataset too small.")

    ab_screen_rows: list[dict[str, float | bool | str]] = []
    cfg_id = 0
    for alpha, w_occ, pwr in product(args.ablate_alphas, args.ablate_w_occ, args.ablate_scale_powers):
        cfg_id += 1
        cid = f"L8C{cfg_id:03d}"
        w = np.asarray([w_occ, 1.0, 1.0, 1.0], dtype=float)
        df_cfg = _apply_theory_bridge(
            dense_l8_base,
            lambda_weights=w,
            lambda_scale_power=float(pwr),
            decoherence_alpha=float(alpha),
        )
        row, _, _, _ = _run_l8_eval(
            df_cfg=df_cfg,
            target_col=target_col,
            ridge_alpha=args.ridge_alpha,
            n_perm=0,
            seed=args.seed,
            active_quantile=args.active_quantile,
            comm_floor=args.comm_floor,
            max_perm_p=args.max_perm_p,
        )
        ab_screen_rows.append(
            {
                "config_id": cid,
                "decoherence_alpha": float(alpha),
                "w_occ": float(w_occ),
                "lambda_scale_power": float(pwr),
                **row,
            }
        )
    ab_screen = pd.DataFrame(ab_screen_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ab_screen.to_csv(outdir / "l8_local_ablation_screening.csv", index=False)

    ab_top = ab_screen.head(max(1, int(args.ablate_top_k))).copy()
    ab_final_rows: list[dict[str, float | bool | str]] = []
    for _, r in ab_top.iterrows():
        w = np.asarray([float(r["w_occ"]), 1.0, 1.0, 1.0], dtype=float)
        df_cfg = _apply_theory_bridge(
            dense_l8_base,
            lambda_weights=w,
            lambda_scale_power=float(r["lambda_scale_power"]),
            decoherence_alpha=float(r["decoherence_alpha"]),
        )
        row, _, _, _ = _run_l8_eval(
            df_cfg=df_cfg,
            target_col=target_col,
            ridge_alpha=args.ridge_alpha,
            n_perm=int(args.ablate_n_perm_final),
            seed=args.seed,
            active_quantile=args.active_quantile,
            comm_floor=args.comm_floor,
            max_perm_p=args.max_perm_p,
        )
        ab_final_rows.append(
            {
                "config_id": r["config_id"],
                "decoherence_alpha": float(r["decoherence_alpha"]),
                "w_occ": float(r["w_occ"]),
                "lambda_scale_power": float(r["lambda_scale_power"]),
                **row,
            }
        )
    ab_final = pd.DataFrame(ab_final_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ab_final.to_csv(outdir / "l8_local_ablation_final.csv", index=False)

    # 3) Operator attribution at l=8 (dense, C009)
    print("[diag] 3/5 operator attribution l=8", flush=True)
    attrib_cfgs = [
        ("full_c009", [c009_w[0], c009_w[1], c009_w[2], c009_w[3]]),
        ("drop_occ", [0.0, c009_w[1], c009_w[2], c009_w[3]]),
        ("drop_sq", [c009_w[0], 0.0, c009_w[2], c009_w[3]]),
        ("drop_log", [c009_w[0], c009_w[1], 0.0, c009_w[3]]),
        ("drop_grad", [c009_w[0], c009_w[1], c009_w[2], 0.0]),
        ("only_occ", [c009_w[0], 0.0, 0.0, 0.0]),
        ("only_sq", [0.0, c009_w[1], 0.0, 0.0]),
        ("only_log", [0.0, 0.0, c009_w[2], 0.0]),
        ("only_grad", [0.0, 0.0, 0.0, c009_w[3]]),
    ]

    attrib_rows: list[dict[str, float | bool | str]] = []
    for tag, w_vec in attrib_cfgs:
        df_cfg = _apply_theory_bridge(
            dense_l8_base,
            lambda_weights=np.asarray(w_vec, dtype=float),
            lambda_scale_power=float(args.lambda_scale_power_c009),
            decoherence_alpha=float(args.alpha_c009),
        )
        row, _, _, _ = _run_l8_eval(
            df_cfg=df_cfg,
            target_col=target_col,
            ridge_alpha=args.ridge_alpha,
            n_perm=int(args.attrib_n_perm),
            seed=args.seed,
            active_quantile=args.active_quantile,
            comm_floor=args.comm_floor,
            max_perm_p=args.max_perm_p,
        )
        attrib_rows.append(
            {
                "config": tag,
                "w_occ": float(w_vec[0]),
                "w_sq": float(w_vec[1]),
                "w_log": float(w_vec[2]),
                "w_grad": float(w_vec[3]),
                **row,
            }
        )
    attrib_df = pd.DataFrame(attrib_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    attrib_df.to_csv(outdir / "l8_operator_attribution.csv", index=False)

    # 4) Resolution check for l=8 (dense panel)
    print("[diag] 4/5 resolution scan l=8", flush=True)
    res_screen_rows: list[dict[str, float | bool | str]] = []
    for downsample, thr in product(args.res_downsamples, args.res_thresholds):
        tile_df = _build_tile_dataset(
            panel_df=dense_panel,
            scales_cells=[8],
            mrms_downsample=int(downsample),
            threshold=float(thr),
            min_valid_frac=float(args.min_valid_frac),
            lambda_weights=c009_w,
            lambda_scale_power=float(args.lambda_scale_power_c009),
            decoherence_alpha=float(args.alpha_c009),
            max_rows=0,
        )
        row, _, _, _ = _run_l8_eval(
            df_cfg=tile_df,
            target_col=target_col,
            ridge_alpha=args.ridge_alpha,
            n_perm=0,
            seed=args.seed,
            active_quantile=args.active_quantile,
            comm_floor=args.comm_floor,
            max_perm_p=args.max_perm_p,
        )
        res_screen_rows.append(
            {
                "mrms_downsample": int(downsample),
                "mrms_threshold": float(thr),
                **row,
            }
        )
    res_screen = pd.DataFrame(res_screen_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    res_screen.to_csv(outdir / "l8_resolution_scan_screening.csv", index=False)

    res_top = res_screen.head(max(1, int(args.res_top_k))).copy()
    res_final_rows: list[dict[str, float | bool | str]] = []
    for _, r in res_top.iterrows():
        tile_df = _build_tile_dataset(
            panel_df=dense_panel,
            scales_cells=[8],
            mrms_downsample=int(r["mrms_downsample"]),
            threshold=float(r["mrms_threshold"]),
            min_valid_frac=float(args.min_valid_frac),
            lambda_weights=c009_w,
            lambda_scale_power=float(args.lambda_scale_power_c009),
            decoherence_alpha=float(args.alpha_c009),
            max_rows=0,
        )
        row, _, _, _ = _run_l8_eval(
            df_cfg=tile_df,
            target_col=target_col,
            ridge_alpha=args.ridge_alpha,
            n_perm=int(args.res_n_perm_final),
            seed=args.seed,
            active_quantile=args.active_quantile,
            comm_floor=args.comm_floor,
            max_perm_p=args.max_perm_p,
        )
        res_final_rows.append(
            {
                "mrms_downsample": int(r["mrms_downsample"]),
                "mrms_threshold": float(r["mrms_threshold"]),
                **row,
            }
        )
    res_final = pd.DataFrame(res_final_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    res_final.to_csv(outdir / "l8_resolution_scan_final.csv", index=False)

    # 5) Regime split at l=8 (dense baseline C009)
    print("[diag] 5/5 regime split l=8", flush=True)
    dense_oof = pd.read_csv(args.dense_oof_csv)
    l8_oof = dense_oof[dense_oof["scale_l"].to_numpy(dtype=float) == 8.0].copy().reset_index(drop=True)
    if len(l8_oof) == 0:
        raise RuntimeError("No l=8 rows in dense OOF.")
    occ = l8_oof["target_occ_coarse"].to_numpy(dtype=float)
    reg, regime_meta = _assign_activity_regime(occ)
    l8_oof["regime"] = reg

    l8_oof["lambda_abs"] = np.abs(l8_oof["lambda_local"].to_numpy(dtype=float))
    reg_rows = (
        l8_oof.groupby("regime", as_index=False)
        .agg(
            n_rows=("regime", "size"),
            point_gain_mean=("pointwise_gain", "mean"),
            point_gain_median=("pointwise_gain", "median"),
            comm_defect_mean=("comm_defect", "mean"),
            lambda_abs_mean=("lambda_abs", "mean"),
            target_occ_mean=("target_occ_coarse", "mean"),
        )
        .sort_values("regime")
        .reset_index(drop=True)
    )
    reg_rows["split_method"] = regime_meta["split_method"]
    reg_rows["q33_raw"] = regime_meta["q33_raw"]
    reg_rows["q67_raw"] = regime_meta["q67_raw"]
    reg_rows["positive_q67"] = regime_meta["positive_q67"]
    reg_rows["positive_count"] = regime_meta["positive_count"]
    reg_rows.to_csv(outdir / "l8_regime_split_summary.csv", index=False)

    event_reg = (
        l8_oof.groupby(["event_id", "regime"], as_index=False)
        .agg(point_gain_event=("pointwise_gain", "mean"), n_rows=("event_id", "size"))
    )
    event_reg["event_positive"] = event_reg["point_gain_event"] > 0.0
    reg_event_rob = (
        event_reg.groupby("regime", as_index=False)
        .agg(
            event_positive_frac=("event_positive", "mean"),
            events_n=("event_id", "nunique"),
            mean_event_gain=("point_gain_event", "mean"),
        )
        .sort_values("regime")
        .reset_index(drop=True)
    )
    reg_event_rob["split_method"] = regime_meta["split_method"]
    event_reg.to_csv(outdir / "l8_regime_split_event_table.csv", index=False)
    reg_event_rob.to_csv(outdir / "l8_regime_split_event_robustness.csv", index=False)

    # final report
    rep = [
        "# P2 l=8 Diagnostic Block",
        "",
        "## Matched-event test",
        f"- sparse full rows/events: {len(sparse_panel)}/{sparse_panel['event_id'].nunique()}",
        f"- sparse matched rows/events: {len(sparse_matched_panel)}/{sparse_matched_panel['event_id'].nunique()}",
        f"- dense rows/events: {len(dense_panel)}/{dense_panel['event_id'].nunique()}",
        f"- artifact: `{outdir / 'matched_event_comparison.csv'}`",
        "",
        "## l=8 local ablation",
        f"- screening configs: {len(ab_screen)}",
        f"- final top-k: {len(ab_final)}",
        f"- best final config: {ab_final.iloc[0]['config_id'] if len(ab_final) > 0 else 'NA'}",
        "",
        "## operator attribution",
        f"- tested configs: {len(attrib_df)}",
        "",
        "## resolution scan",
        f"- screening grid size: {len(res_screen)}",
        f"- final top-k: {len(res_final)}",
        "",
        "## regime split (l=8)",
        f"- split method: {regime_meta['split_method']}",
        f"- q33_raw={float(regime_meta['q33_raw']):.6f}, q67_raw={float(regime_meta['q67_raw']):.6f}",
        f"- positive_q67={float(regime_meta['positive_q67']):.6f}, positive_count={int(regime_meta['positive_count'])}",
        f"- artifacts: `{outdir / 'l8_regime_split_summary.csv'}`, `{outdir / 'l8_regime_split_event_robustness.csv'}`",
    ]
    (outdir / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print(f"Diagnostic block complete: {outdir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
