#!/usr/bin/env python3
"""Strict GKSL hybrid validation for Experiment M.

Protocol:
- candidate selection only on train years (default <= 2018),
- frozen model evaluation on test year (default 2019),
- optional external holdout year (default 2020),
- permutation tests on holdouts with fixed train-fitted coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        _block_permute,
        _blocked_splits,
        _evaluate_splits,
        _fit_ridge_scaled,
    )
    from clean_experiments.experiment_M_gksl_hybrid_bridge import (
        EPS,
        _active_calm_rows,
        _build_gksl_proxy,
        _phi_candidate_map,
        _pick_lambda_band,
        _parse_time_hours,
        _read_gksl_config,
        _safe_r2,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        _block_permute,
        _blocked_splits,
        _evaluate_splits,
        _fit_ridge_scaled,
    )
    from experiment_M_gksl_hybrid_bridge import (  # type: ignore
        EPS,
        _active_calm_rows,
        _build_gksl_proxy,
        _phi_candidate_map,
        _pick_lambda_band,
        _parse_time_hours,
        _read_gksl_config,
        _safe_r2,
    )


def _perm_p_fixed_model(
    *,
    y_eval: np.ndarray,
    yhat_base: np.ndarray,
    x_eval_full: np.ndarray,
    coef_full: np.ndarray,
    intercept_full: float,
    added_cols: np.ndarray,
    n_perm: int,
    perm_block: int,
    seed: int,
) -> tuple[float, float, pd.DataFrame]:
    yhat_real = intercept_full + x_eval_full @ coef_full
    mae_base = float(np.mean(np.abs(y_eval - yhat_base)))
    mae_full = float(np.mean(np.abs(y_eval - yhat_real)))
    real_gain = float((mae_base - mae_full) / (mae_base + EPS))

    rng = np.random.default_rng(seed)
    rows = []
    count_ge = 0
    for pid in range(int(n_perm)):
        x_perm = np.asarray(x_eval_full, dtype=float).copy()
        x_perm[:, added_cols] = _block_permute(x_perm[:, added_cols], block=int(perm_block), rng=rng)
        yhat_perm = intercept_full + x_perm @ coef_full
        mae_perm = float(np.mean(np.abs(y_eval - yhat_perm)))
        gain_perm = float((mae_base - mae_perm) / (mae_base + EPS))
        rows.append({"perm_id": int(pid), "gain_perm": gain_perm})
        if gain_perm >= real_gain:
            count_ge += 1
    p = float((count_ge + 1) / (int(n_perm) + 1))
    return p, real_gain, pd.DataFrame(rows)


def _materialize_phi_candidates(
    *,
    df: pd.DataFrame,
    gksl_cfg,
    band_pool: list[int],
    time_hours: np.ndarray,
    gksl_dt_cap_hours: float,
) -> dict[tuple[int, int, str], np.ndarray]:
    out: dict[tuple[int, int, str], np.ndarray] = {}
    for i in band_pool:
        for j in band_pool:
            if i >= j:
                continue
            sig_i = _pick_lambda_band(df, idx=int(i), fallback_col="lambda_struct")
            sig_j = _pick_lambda_band(df, idx=int(j), fallback_col="lambda_struct")
            proxy_df = _build_gksl_proxy(
                sig_l=sig_i,
                sig_2l=sig_j,
                time_hours=time_hours,
                cfg=gksl_cfg,
                dt_cap_hours=float(gksl_dt_cap_hours),
            )
            cand = _phi_candidate_map(proxy_df, sig_i, sig_j)
            for feat_name, phi_vec in cand.items():
                out[(int(i), int(j), str(feat_name))] = np.asarray(phi_vec, dtype=float)
    return out


def _cv_gain_for_candidate(
    *,
    y: np.ndarray,
    ctrl: np.ndarray,
    lam: np.ndarray,
    phi: np.ndarray,
    mask: np.ndarray,
    ridge_alpha: float,
    n_folds: int,
) -> tuple[float, float, float, int] | None:
    valid = mask & np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi)
    idx = np.where(valid)[0]
    if len(idx) < max(80, int(n_folds) * 8):
        return None
    y_tr = y[idx]
    x_base = np.column_stack([ctrl[idx], lam[idx]])
    x_full = np.column_stack([ctrl[idx], lam[idx], phi[idx]])
    splits = _blocked_splits(len(idx), n_folds=int(n_folds))
    split_df, yhat_b, yhat_f = _evaluate_splits(
        y=y_tr,
        x_base=x_base,
        x_full=x_full,
        base_feature_names=["ctrl", "lambda"],
        full_feature_names=["ctrl", "lambda", "phi_gksl"],
        splits=splits,
        ridge_alpha=float(ridge_alpha),
    )
    mae_b = float(np.mean(np.abs(y_tr - yhat_b)))
    mae_f = float(np.mean(np.abs(y_tr - yhat_f)))
    gain = float((mae_b - mae_f) / (mae_b + EPS))
    med = float(np.median(split_df["mae_gain_frac"].to_numpy(dtype=float)))
    mn = float(np.min(split_df["mae_gain_frac"].to_numpy(dtype=float)))
    return gain, med, mn, int(len(idx))


def _nested_select_candidate(
    *,
    y: np.ndarray,
    ctrl: np.ndarray,
    lam: np.ndarray,
    candidates: dict[tuple[int, int, str], np.ndarray],
    train_mask: np.ndarray,
    ridge_alpha: float,
    outer_folds: int,
    inner_folds: int,
) -> tuple[tuple[int, int, str] | None, pd.DataFrame, pd.DataFrame]:
    base_valid = train_mask & np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam)
    pool_idx = np.where(base_valid)[0]
    if len(pool_idx) < max(200, int(outer_folds) * 16):
        return None, pd.DataFrame(), pd.DataFrame()

    outer = _blocked_splits(len(pool_idx), n_folds=int(outer_folds))
    rows_outer: list[dict[str, float | int | str]] = []

    for fold_id, (tr_rel, te_rel) in enumerate(outer):
        tr_idx = pool_idx[tr_rel]
        te_idx = pool_idx[te_rel]
        if len(tr_idx) < 200 or len(te_idx) < 80:
            continue

        best_key: tuple[int, int, str] | None = None
        best_gain = -np.inf
        best_med = np.nan
        best_min = np.nan
        best_n = 0

        tr_mask_global = np.zeros_like(train_mask, dtype=bool)
        tr_mask_global[tr_idx] = True

        for key, phi_vec in candidates.items():
            stat = _cv_gain_for_candidate(
                y=y,
                ctrl=ctrl,
                lam=lam,
                phi=phi_vec,
                mask=tr_mask_global,
                ridge_alpha=float(ridge_alpha),
                n_folds=int(inner_folds),
            )
            if stat is None:
                continue
            gain, med, mn, n_eff = stat
            if gain > best_gain:
                best_gain = float(gain)
                best_med = float(med)
                best_min = float(mn)
                best_n = int(n_eff)
                best_key = key

        if best_key is None:
            continue

        i, j, feat = best_key
        phi_best = candidates[best_key]
        tr_eval = tr_idx[np.isfinite(phi_best[tr_idx])]
        te_eval = te_idx[np.isfinite(phi_best[te_idx])]
        if len(tr_eval) < 200 or len(te_eval) < 80:
            continue

        y_tr = y[tr_eval]
        y_te = y[te_eval]
        xl_tr = np.column_stack([ctrl[tr_eval], lam[tr_eval]])
        xl_te = np.column_stack([ctrl[te_eval], lam[te_eval]])
        xlp_tr = np.column_stack([ctrl[tr_eval], lam[tr_eval], phi_best[tr_eval]])
        xlp_te = np.column_stack([ctrl[te_eval], lam[te_eval], phi_best[te_eval]])

        _, _, yhat_l = _fit_ridge_scaled(xl_tr, y_tr, xl_te, float(ridge_alpha))
        _, _, yhat_lp = _fit_ridge_scaled(xlp_tr, y_tr, xlp_te, float(ridge_alpha))
        mae_l = float(np.mean(np.abs(y_te - yhat_l)))
        mae_lp = float(np.mean(np.abs(y_te - yhat_lp)))
        gain_outer = float((mae_l - mae_lp) / (mae_l + EPS))

        rows_outer.append(
            {
                "selection_mode": "nested_outer",
                "outer_fold": int(fold_id),
                "band_l_idx": int(i),
                "band_2l_idx": int(j),
                "feature_name": str(feat),
                "inner_gain_vs_lambda": float(best_gain),
                "inner_split_gain_median": float(best_med),
                "inner_split_gain_min": float(best_min),
                "inner_n": int(best_n),
                "outer_gain_vs_lambda": float(gain_outer),
                "outer_n_train": int(len(tr_eval)),
                "outer_n_test": int(len(te_eval)),
            }
        )

    outer_df = pd.DataFrame(rows_outer)
    if len(outer_df) == 0:
        return None, outer_df, pd.DataFrame()

    agg = (
        outer_df.groupby(["band_l_idx", "band_2l_idx", "feature_name"], as_index=False)
        .agg(
            n_selected=("outer_fold", "count"),
            outer_gain_median=("outer_gain_vs_lambda", "median"),
            outer_gain_mean=("outer_gain_vs_lambda", "mean"),
            inner_gain_median=("inner_gain_vs_lambda", "median"),
        )
        .sort_values(["outer_gain_median", "n_selected", "inner_gain_median"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    top = agg.iloc[0]
    key = (int(top["band_l_idx"]), int(top["band_2l_idx"]), str(top["feature_name"]))
    agg["selection_mode"] = "nested_aggregate"
    return key, outer_df, agg


def _select_phi_train(
    *,
    df: pd.DataFrame,
    train_mask: np.ndarray,
    gksl_final_csv: Path,
    gksl_config_id: str,
    selection_mode: str,
    fine_band_idx: int,
    coarse_band_idx: int,
    screen_band_idxs: list[int],
    lock_feature: str | None,
    ridge_alpha: float,
    n_folds: int,
    nested_outer_folds: int,
    nested_inner_folds: int,
    gksl_dt_cap_hours: float,
) -> tuple[pd.DataFrame, dict[str, int | str | float], pd.DataFrame]:
    gksl_cfg = _read_gksl_config(gksl_final_csv, prefer_config_id=gksl_config_id)
    y = np.asarray(df["residual_base_res0"], dtype=float)
    ctrl = np.asarray(df["n_density_ctrl_z"], dtype=float)
    lam = np.asarray(df["lambda_struct"], dtype=float)
    time_hours = _parse_time_hours(df["time"])
    band_pool = sorted(set([int(fine_band_idx), int(coarse_band_idx)] + [int(x) for x in screen_band_idxs]))

    if selection_mode == "locked" or lock_feature is not None:
        sel_i = int(fine_band_idx)
        sel_j = int(coarse_band_idx)
        sel_feat = str(lock_feature) if lock_feature is not None else "raw"
        selection_mode = "locked"
        screen_df = pd.DataFrame(
            [
                {
                    "selection_mode": "locked",
                    "band_l_idx": int(sel_i),
                    "band_2l_idx": int(sel_j),
                    "feature_name": str(sel_feat),
                    "train_n": int(np.sum(train_mask)),
                    "gain_vs_lambda": np.nan,
                    "split_gain_median": np.nan,
                    "split_gain_min": np.nan,
                }
            ]
        )
    elif selection_mode == "nested":
        candidates = _materialize_phi_candidates(
            df=df,
            gksl_cfg=gksl_cfg,
            band_pool=band_pool,
            time_hours=time_hours,
            gksl_dt_cap_hours=float(gksl_dt_cap_hours),
        )
        key, outer_df, agg_df = _nested_select_candidate(
            y=y,
            ctrl=ctrl,
            lam=lam,
            candidates=candidates,
            train_mask=train_mask,
            ridge_alpha=float(ridge_alpha),
            outer_folds=int(nested_outer_folds),
            inner_folds=int(nested_inner_folds),
        )
        if key is None:
            sel_i = int(fine_band_idx)
            sel_j = int(coarse_band_idx)
            sel_feat = "raw"
            selection_mode = "fallback_nested"
            screen_df = pd.DataFrame(
                [
                    {
                        "selection_mode": "fallback_nested",
                        "band_l_idx": int(sel_i),
                        "band_2l_idx": int(sel_j),
                        "feature_name": "raw",
                        "train_n": int(np.sum(train_mask)),
                        "gain_vs_lambda": np.nan,
                        "split_gain_median": np.nan,
                        "split_gain_min": np.nan,
                    }
                ]
            )
        else:
            sel_i, sel_j, sel_feat = key
            selection_mode = "nested_train_only"
            screen_rows: list[pd.DataFrame] = []
            if len(agg_df) > 0:
                screen_rows.append(agg_df)
            if len(outer_df) > 0:
                screen_rows.append(outer_df)
            screen_df = pd.concat(screen_rows, ignore_index=True) if len(screen_rows) > 0 else pd.DataFrame()
    else:
        candidates = _materialize_phi_candidates(
            df=df,
            gksl_cfg=gksl_cfg,
            band_pool=band_pool,
            time_hours=time_hours,
            gksl_dt_cap_hours=float(gksl_dt_cap_hours),
        )
        rows: list[dict[str, float | int | str]] = []
        for (i, j, feat_name), phi_vec in candidates.items():
            stat = _cv_gain_for_candidate(
                y=y,
                ctrl=ctrl,
                lam=lam,
                phi=phi_vec,
                mask=train_mask,
                ridge_alpha=float(ridge_alpha),
                n_folds=int(n_folds),
            )
            if stat is None:
                continue
            gain, med, mn, n_eff = stat
            rows.append(
                {
                    "selection_mode": "screened_train_only",
                    "band_l_idx": int(i),
                    "band_2l_idx": int(j),
                    "feature_name": str(feat_name),
                    "train_n": int(n_eff),
                    "gain_vs_lambda": float(gain),
                    "split_gain_median": float(med),
                    "split_gain_min": float(mn),
                }
            )
        if len(rows) == 0:
            sel_i = int(fine_band_idx)
            sel_j = int(coarse_band_idx)
            sel_feat = "raw"
            selection_mode = "fallback"
            screen_df = pd.DataFrame(
                [
                    {
                        "selection_mode": "fallback",
                        "band_l_idx": int(sel_i),
                        "band_2l_idx": int(sel_j),
                        "feature_name": "raw",
                        "train_n": int(np.sum(train_mask)),
                        "gain_vs_lambda": np.nan,
                        "split_gain_median": np.nan,
                        "split_gain_min": np.nan,
                    }
                ]
            )
        else:
            screen_df = pd.DataFrame(rows).sort_values(
                ["gain_vs_lambda", "split_gain_median", "split_gain_min"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            top = screen_df.iloc[0]
            sel_i = int(top["band_l_idx"])
            sel_j = int(top["band_2l_idx"])
            sel_feat = str(top["feature_name"])
            selection_mode = "screened_train_only"

    sig_l = _pick_lambda_band(df, idx=int(sel_i), fallback_col="lambda_struct")
    sig_2l = _pick_lambda_band(df, idx=int(sel_j), fallback_col="lambda_struct")
    phi_state = _build_gksl_proxy(
        sig_l=sig_l,
        sig_2l=sig_2l,
        time_hours=time_hours,
        cfg=gksl_cfg,
        dt_cap_hours=float(gksl_dt_cap_hours),
    )
    cand_map = _phi_candidate_map(phi_state, sig_l, sig_2l)
    if sel_feat not in cand_map:
        sel_feat = "raw"
    phi = np.asarray(cand_map[sel_feat], dtype=float)

    out = df.copy()
    out["phi_gksl"] = phi
    out["phi_gksl_state_raw"] = np.asarray(phi_state["phi_gksl_raw"], dtype=float)
    out["phi_gksl_state_popdiff"] = np.asarray(phi_state["phi_gksl_popdiff"], dtype=float)
    out["phi_gksl_state_coh_re"] = np.asarray(phi_state["phi_gksl_coh_re"], dtype=float)
    out["phi_gksl_state_eta"] = np.asarray(phi_state["phi_gksl_eta"], dtype=float)
    out["gksl_gamma_dephase"] = np.asarray(phi_state["gksl_gamma_dephase"], dtype=float)
    out["gksl_gamma_relax"] = np.asarray(phi_state["gksl_gamma_relax"], dtype=float)
    out["gksl_reset_kappa"] = np.asarray(phi_state["gksl_reset_kappa"], dtype=float)
    out["gksl_omega"] = np.asarray(phi_state["gksl_omega"], dtype=float)
    out["gksl_dt_hours"] = np.asarray(phi_state["gksl_dt_hours"], dtype=float)
    out["gksl_cptp_violation"] = np.asarray(phi_state["gksl_cptp_violation"], dtype=float)

    sel = {
        "selection_mode": selection_mode,
        "band_l_idx": int(sel_i),
        "band_2l_idx": int(sel_j),
        "feature_name": str(sel_feat),
        "gksl_config_id": str(gksl_cfg.config_id),
        "gksl_dephase_base": float(gksl_cfg.dephase_base),
        "gksl_dephase_comm_scale": float(gksl_cfg.dephase_comm_scale),
        "gksl_relax_base": float(gksl_cfg.relax_base),
        "gksl_relax_comm_scale": float(gksl_cfg.relax_comm_scale),
        "gksl_measurement_rate": float(gksl_cfg.measurement_rate),
        "gksl_hamiltonian_scale": float(gksl_cfg.hamiltonian_scale),
    }
    return out, sel, screen_df


def _fit_eval_models(
    *,
    df: pd.DataFrame,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    ridge_alpha: float,
    perm_block: int,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    y = np.asarray(df["residual_base_res0"], dtype=float)
    ctrl = np.asarray(df["n_density_ctrl_z"], dtype=float)
    lam = np.asarray(df["lambda_struct"], dtype=float)
    phi = np.asarray(df["phi_gksl"], dtype=float)

    valid = np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi) & train_mask
    train_idx = np.where(valid)[0]
    valid_eval = np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi) & eval_mask
    eval_idx = np.where(valid_eval)[0]
    if len(train_idx) < 200 or len(eval_idx) < 80:
        raise ValueError(f"Insufficient train/eval sizes: train={len(train_idx)}, eval={len(eval_idx)}")

    y_tr = y[train_idx]
    y_te = y[eval_idx]
    ctrl_tr = ctrl[train_idx]
    ctrl_te = ctrl[eval_idx]
    lam_tr = lam[train_idx]
    lam_te = lam[eval_idx]
    phi_tr = phi[train_idx]
    phi_te = phi[eval_idx]

    rng = np.random.default_rng(int(seed) + 100)
    phi_tr_shuf = _block_permute(phi_tr[:, None], block=int(perm_block), rng=rng).reshape(-1)
    phi_te_shuf = _block_permute(phi_te[:, None], block=int(perm_block), rng=rng).reshape(-1)

    xb_tr = ctrl_tr[:, None]
    xb_te = ctrl_te[:, None]
    xl_tr = np.column_stack([ctrl_tr, lam_tr])
    xl_te = np.column_stack([ctrl_te, lam_te])
    xp_tr = np.column_stack([ctrl_tr, phi_tr])
    xp_te = np.column_stack([ctrl_te, phi_te])
    xlp_tr = np.column_stack([ctrl_tr, lam_tr, phi_tr])
    xlp_te = np.column_stack([ctrl_te, lam_te, phi_te])
    xlp_shuf_tr = np.column_stack([ctrl_tr, lam_tr, phi_tr_shuf])
    xlp_shuf_te = np.column_stack([ctrl_te, lam_te, phi_te_shuf])

    c_b, i_b, yhat_b = _fit_ridge_scaled(xb_tr, y_tr, xb_te, float(ridge_alpha))
    c_l, i_l, yhat_l = _fit_ridge_scaled(xl_tr, y_tr, xl_te, float(ridge_alpha))
    c_p, i_p, yhat_p = _fit_ridge_scaled(xp_tr, y_tr, xp_te, float(ridge_alpha))
    c_lp, i_lp, yhat_lp = _fit_ridge_scaled(xlp_tr, y_tr, xlp_te, float(ridge_alpha))
    c_lps, i_lps, yhat_lps = _fit_ridge_scaled(xlp_shuf_tr, y_tr, xlp_shuf_te, float(ridge_alpha))

    mae_b = float(np.mean(np.abs(y_te - yhat_b)))
    mae_l = float(np.mean(np.abs(y_te - yhat_l)))
    mae_p = float(np.mean(np.abs(y_te - yhat_p)))
    mae_lp = float(np.mean(np.abs(y_te - yhat_lp)))
    mae_lps = float(np.mean(np.abs(y_te - yhat_lps)))

    p_l_abs, real_l_abs, perm_l_abs = _perm_p_fixed_model(
        y_eval=y_te,
        yhat_base=yhat_b,
        x_eval_full=xl_te,
        coef_full=np.asarray(c_l, dtype=float),
        intercept_full=float(i_l),
        added_cols=np.array([1], dtype=int),
        n_perm=int(n_perm),
        perm_block=int(perm_block),
        seed=int(seed) + 11,
    )
    p_p_abs, real_p_abs, perm_p_abs = _perm_p_fixed_model(
        y_eval=y_te,
        yhat_base=yhat_b,
        x_eval_full=xp_te,
        coef_full=np.asarray(c_p, dtype=float),
        intercept_full=float(i_p),
        added_cols=np.array([1], dtype=int),
        n_perm=int(n_perm),
        perm_block=int(perm_block),
        seed=int(seed) + 13,
    )
    p_lp_inc, real_lp_inc, perm_lp_inc = _perm_p_fixed_model(
        y_eval=y_te,
        yhat_base=yhat_l,
        x_eval_full=xlp_te,
        coef_full=np.asarray(c_lp, dtype=float),
        intercept_full=float(i_lp),
        added_cols=np.array([2], dtype=int),
        n_perm=int(n_perm),
        perm_block=int(perm_block),
        seed=int(seed) + 17,
    )
    p_lps_inc, real_lps_inc, perm_lps_inc = _perm_p_fixed_model(
        y_eval=y_te,
        yhat_base=yhat_l,
        x_eval_full=xlp_shuf_te,
        coef_full=np.asarray(c_lps, dtype=float),
        intercept_full=float(i_lps),
        added_cols=np.array([2], dtype=int),
        n_perm=int(n_perm),
        perm_block=int(perm_block),
        seed=int(seed) + 19,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "ERA5_only",
                "mae": mae_b,
                "r2": _safe_r2(y_te, yhat_b),
                "gain_vs_ctrl": 0.0,
                "gain_vs_lambda": np.nan,
                "perm_p_abs": np.nan,
                "perm_p_inc": np.nan,
            },
            {
                "model": "ERA5_plus_Lambda",
                "mae": mae_l,
                "r2": _safe_r2(y_te, yhat_l),
                "gain_vs_ctrl": float((mae_b - mae_l) / (mae_b + EPS)),
                "gain_vs_lambda": 0.0,
                "perm_p_abs": p_l_abs,
                "perm_p_inc": np.nan,
            },
            {
                "model": "ERA5_plus_Phi_GKSL",
                "mae": mae_p,
                "r2": _safe_r2(y_te, yhat_p),
                "gain_vs_ctrl": float((mae_b - mae_p) / (mae_b + EPS)),
                "gain_vs_lambda": float((mae_l - mae_p) / (mae_l + EPS)),
                "perm_p_abs": p_p_abs,
                "perm_p_inc": np.nan,
            },
            {
                "model": "ERA5_plus_Lambda_plus_Phi_GKSL",
                "mae": mae_lp,
                "r2": _safe_r2(y_te, yhat_lp),
                "gain_vs_ctrl": float((mae_b - mae_lp) / (mae_b + EPS)),
                "gain_vs_lambda": float((mae_l - mae_lp) / (mae_l + EPS)),
                "perm_p_abs": np.nan,
                "perm_p_inc": p_lp_inc,
            },
            {
                "model": "ERA5_plus_Lambda_plus_Phi_GKSL_shuffled",
                "mae": mae_lps,
                "r2": _safe_r2(y_te, yhat_lps),
                "gain_vs_ctrl": float((mae_b - mae_lps) / (mae_b + EPS)),
                "gain_vs_lambda": float((mae_l - mae_lps) / (mae_l + EPS)),
                "perm_p_abs": np.nan,
                "perm_p_inc": p_lps_inc,
            },
        ]
    )

    oof = pd.DataFrame(
        {
            "time_index": np.asarray(df.iloc[eval_idx]["time_index"], dtype=int),
            "time": df.iloc[eval_idx]["time"].astype(str).to_numpy(),
            "y": y_te,
            "ctrl": ctrl_te,
            "lambda_struct": lam_te,
            "phi_gksl": phi_te,
            "phi_gksl_shuffled": phi_te_shuf,
            "yhat_ctrl": yhat_b,
            "yhat_lambda": yhat_l,
            "yhat_phi": yhat_p,
            "yhat_lambda_phi": yhat_lp,
            "yhat_lambda_phi_shuffled": yhat_lps,
        }
    )

    active = _active_calm_rows(
        y=y_te,
        yhat_base=yhat_l,
        yhat_full=yhat_lp,
        yhat_shuffled=yhat_lps,
        activity=np.abs(lam_te),
        active_quantile=0.67,
    )

    perms = {
        "perm_lambda_abs": perm_l_abs,
        "perm_phi_abs": perm_p_abs,
        "perm_phi_given_lambda": perm_lp_inc,
        "perm_phi_shuffled_given_lambda": perm_lps_inc,
    }
    real = pd.DataFrame(
        [
            {"test_name": "lambda_abs_vs_ctrl", "real_gain": real_l_abs, "p_value": p_l_abs},
            {"test_name": "phi_abs_vs_ctrl", "real_gain": real_p_abs, "p_value": p_p_abs},
            {"test_name": "phi_given_lambda", "real_gain": real_lp_inc, "p_value": p_lp_inc},
            {"test_name": "phi_shuffled_given_lambda", "real_gain": real_lps_inc, "p_value": p_lps_inc},
        ]
    )
    return metrics, oof, active, perms, real


def _rolling_quarters_2019(
    *,
    df: pd.DataFrame,
    ridge_alpha: float,
) -> pd.DataFrame:
    t = pd.to_datetime(df["time"], errors="coerce")
    y = np.asarray(df["residual_base_res0"], dtype=float)
    ctrl = np.asarray(df["n_density_ctrl_z"], dtype=float)
    lam = np.asarray(df["lambda_struct"], dtype=float)
    phi = np.asarray(df["phi_gksl"], dtype=float)

    starts = [
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2019-04-01"),
        pd.Timestamp("2019-07-01"),
        pd.Timestamp("2019-10-01"),
    ]
    ends = [
        pd.Timestamp("2019-04-01"),
        pd.Timestamp("2019-07-01"),
        pd.Timestamp("2019-10-01"),
        pd.Timestamp("2020-01-01"),
    ]
    labels = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]

    rows = []
    for q0, q1, ql in zip(starts, ends, labels):
        tr_mask = (t < q0).to_numpy(dtype=bool)
        te_mask = ((t >= q0) & (t < q1)).to_numpy(dtype=bool)
        valid_tr = tr_mask & np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi)
        valid_te = te_mask & np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi)
        tr_idx = np.where(valid_tr)[0]
        te_idx = np.where(valid_te)[0]
        if len(tr_idx) < 200 or len(te_idx) < 80:
            continue
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        xb_tr = ctrl[tr_idx][:, None]
        xb_te = ctrl[te_idx][:, None]
        xl_tr = np.column_stack([ctrl[tr_idx], lam[tr_idx]])
        xl_te = np.column_stack([ctrl[te_idx], lam[te_idx]])
        xlp_tr = np.column_stack([ctrl[tr_idx], lam[tr_idx], phi[tr_idx]])
        xlp_te = np.column_stack([ctrl[te_idx], lam[te_idx], phi[te_idx]])

        _, _, yhat_b = _fit_ridge_scaled(xb_tr, y_tr, xb_te, float(ridge_alpha))
        _, _, yhat_l = _fit_ridge_scaled(xl_tr, y_tr, xl_te, float(ridge_alpha))
        _, _, yhat_lp = _fit_ridge_scaled(xlp_tr, y_tr, xlp_te, float(ridge_alpha))
        mae_b = float(np.mean(np.abs(y_te - yhat_b)))
        mae_l = float(np.mean(np.abs(y_te - yhat_l)))
        mae_lp = float(np.mean(np.abs(y_te - yhat_lp)))
        rows.append(
            {
                "quarter": ql,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "gain_lambda_vs_ctrl": float((mae_b - mae_l) / (mae_b + EPS)),
                "gain_lambda_phi_vs_lambda": float((mae_l - mae_lp) / (mae_l + EPS)),
                "gain_lambda_phi_vs_ctrl": float((mae_b - mae_lp) / (mae_b + EPS)),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--m-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_2017_2020q1_v4locked/experiment_M_timeseries.csv"),
    )
    p.add_argument(
        "--gksl-final-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory_gksl_cptp/gksl_final_l8.csv"),
    )
    p.add_argument("--outdir", type=Path, default=Path("clean_experiments/results/experiment_M_gksl_hybrid_strict"))
    p.add_argument("--train-end-year", type=int, default=2018)
    p.add_argument("--test-year", type=int, default=2019)
    p.add_argument("--external-year", type=int, default=2020)
    p.add_argument("--gksl-config-id", type=str, default="G001")
    p.add_argument(
        "--selection-mode",
        type=str,
        default="screened",
        choices=["screened", "nested", "locked"],
        help="Phi selection protocol on train split.",
    )
    p.add_argument("--fine-band-idx", type=int, default=0)
    p.add_argument("--coarse-band-idx", type=int, default=1)
    p.add_argument("--screen-band-idxs", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument(
        "--lock-feature",
        type=str,
        default=None,
        choices=["raw", "popdiff", "coh", "eta", "raw_x_eta", "raw_plus_pop", "dphi", "pop_x_delta"],
    )
    p.add_argument("--gksl-dt-cap-hours", type=float, default=6.0)
    p.add_argument("--ridge-alpha", type=float, default=1e-6)
    p.add_argument("--n-folds", type=int, default=6)
    p.add_argument("--nested-outer-folds", type=int, default=4)
    p.add_argument("--nested-inner-folds", type=int, default=4)
    p.add_argument("--n-perm", type=int, default=199)
    p.add_argument("--perm-block", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260307)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.m_timeseries_csv).sort_values("time_index").reset_index(drop=True)
    t = pd.to_datetime(df["time"], errors="coerce")
    if t.isna().all():
        raise ValueError("Could not parse time column in M timeseries.")
    years = t.dt.year.to_numpy(dtype=int)
    train_mask = years <= int(args.train_end_year)
    test_mask = years == int(args.test_year)
    ext_mask = years == int(args.external_year)

    if int(np.sum(train_mask)) < 400:
        raise ValueError(f"Too few train rows for years <= {args.train_end_year}: {int(np.sum(train_mask))}")
    if int(np.sum(test_mask)) < 200:
        raise ValueError(f"Too few test rows for year={args.test_year}: {int(np.sum(test_mask))}")

    enriched, sel, screen_df = _select_phi_train(
        df=df,
        train_mask=train_mask,
        gksl_final_csv=args.gksl_final_csv,
        gksl_config_id=str(args.gksl_config_id),
        selection_mode=str(args.selection_mode),
        fine_band_idx=int(args.fine_band_idx),
        coarse_band_idx=int(args.coarse_band_idx),
        screen_band_idxs=[int(x) for x in args.screen_band_idxs],
        lock_feature=args.lock_feature,
        ridge_alpha=float(args.ridge_alpha),
        n_folds=int(args.n_folds),
        nested_outer_folds=int(args.nested_outer_folds),
        nested_inner_folds=int(args.nested_inner_folds),
        gksl_dt_cap_hours=float(args.gksl_dt_cap_hours),
    )

    # Main strict test (2019).
    met_2019, oof_2019, act_2019, perms_2019, real_2019 = _fit_eval_models(
        df=enriched,
        train_mask=train_mask,
        eval_mask=test_mask,
        ridge_alpha=float(args.ridge_alpha),
        perm_block=int(args.perm_block),
        n_perm=int(args.n_perm),
        seed=int(args.seed) + 10,
    )

    # External holdout (2020, if available).
    met_ext = pd.DataFrame()
    oof_ext = pd.DataFrame()
    act_ext = pd.DataFrame()
    perms_ext: dict[str, pd.DataFrame] = {}
    real_ext = pd.DataFrame()
    if int(np.sum(ext_mask)) >= 200:
        met_ext, oof_ext, act_ext, perms_ext, real_ext = _fit_eval_models(
            df=enriched,
            train_mask=train_mask,
            eval_mask=ext_mask,
            ridge_alpha=float(args.ridge_alpha),
            perm_block=int(args.perm_block),
            n_perm=int(args.n_perm),
            seed=int(args.seed) + 20,
        )

    q2019 = _rolling_quarters_2019(df=enriched, ridge_alpha=float(args.ridge_alpha))

    # Save artifacts.
    enriched.to_csv(outdir / "strict_timeseries_with_phi.csv", index=False)
    screen_df.to_csv(outdir / "strict_phi_screening_train.csv", index=False)
    met_2019.to_csv(outdir / "strict_metrics_2019.csv", index=False)
    oof_2019.to_csv(outdir / "strict_oof_2019.csv", index=False)
    act_2019.to_csv(outdir / "strict_active_calm_2019.csv", index=False)
    real_2019.to_csv(outdir / "strict_perm_real_2019.csv", index=False)
    for k, v in perms_2019.items():
        v.to_csv(outdir / f"{k}_2019.csv", index=False)

    if len(met_ext) > 0:
        met_ext.to_csv(outdir / "strict_metrics_external.csv", index=False)
        oof_ext.to_csv(outdir / "strict_oof_external.csv", index=False)
        act_ext.to_csv(outdir / "strict_active_calm_external.csv", index=False)
        real_ext.to_csv(outdir / "strict_perm_real_external.csv", index=False)
        for k, v in perms_ext.items():
            v.to_csv(outdir / f"{k}_external.csv", index=False)
    q2019.to_csv(outdir / "strict_quarterly_2019.csv", index=False)

    meta = {
        "inputs": {
            "m_timeseries_csv": str(args.m_timeseries_csv),
            "gksl_final_csv": str(args.gksl_final_csv),
        },
        "split": {
            "train_end_year": int(args.train_end_year),
            "test_year": int(args.test_year),
            "external_year": int(args.external_year),
            "n_train": int(np.sum(train_mask)),
            "n_test": int(np.sum(test_mask)),
            "n_external": int(np.sum(ext_mask)),
        },
        "selection": sel,
        "runtime": {
            "ridge_alpha": float(args.ridge_alpha),
            "n_folds": int(args.n_folds),
            "nested_outer_folds": int(args.nested_outer_folds),
            "nested_inner_folds": int(args.nested_inner_folds),
            "n_perm": int(args.n_perm),
            "perm_block": int(args.perm_block),
            "seed": int(args.seed),
        },
    }
    (outdir / "strict_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    row_l = met_2019[met_2019["model"] == "ERA5_plus_Lambda"].iloc[0]
    row_lp = met_2019[met_2019["model"] == "ERA5_plus_Lambda_plus_Phi_GKSL"].iloc[0]
    row_lps = met_2019[met_2019["model"] == "ERA5_plus_Lambda_plus_Phi_GKSL_shuffled"].iloc[0]

    lines = [
        "# Experiment M GKSL Hybrid Strict",
        "",
        "## Protocol",
        f"- train years <= {int(args.train_end_year)}",
        f"- holdout test year = {int(args.test_year)}",
        f"- external holdout year = {int(args.external_year)}",
        "- phi candidate selected on train only (or locked by CLI)",
        "- permutation tests on holdouts with fixed train-fitted coefficients",
        "",
        "## Selection",
        f"- mode: `{sel['selection_mode']}`",
        f"- selected bands: `{sel['band_l_idx']}` / `{sel['band_2l_idx']}`",
        f"- selected feature: `{sel['feature_name']}`",
        f"- GKSL config: `{sel['gksl_config_id']}`",
        "",
        f"## Holdout {int(args.test_year)}",
        f"- ERA5+Lambda gain_vs_ctrl: `{float(row_l['gain_vs_ctrl']):.6f}`, p_abs=`{float(row_l['perm_p_abs']):.6f}`",
        f"- ERA5+Lambda+Phi gain_vs_ctrl: `{float(row_lp['gain_vs_ctrl']):.6f}`",
        f"- ERA5+Lambda+Phi gain_vs_lambda: `{float(row_lp['gain_vs_lambda']):.6f}`, p_inc=`{float(row_lp['perm_p_inc']):.6f}`",
        f"- shuffled control gain_vs_lambda: `{float(row_lps['gain_vs_lambda']):.6f}`, p_inc=`{float(row_lps['perm_p_inc']):.6f}`",
    ]
    if len(met_ext) > 0:
        row_l_e = met_ext[met_ext["model"] == "ERA5_plus_Lambda"].iloc[0]
        row_lp_e = met_ext[met_ext["model"] == "ERA5_plus_Lambda_plus_Phi_GKSL"].iloc[0]
        row_lps_e = met_ext[met_ext["model"] == "ERA5_plus_Lambda_plus_Phi_GKSL_shuffled"].iloc[0]
        lines.extend(
            [
                "",
                f"## External {int(args.external_year)}",
                f"- ERA5+Lambda gain_vs_ctrl: `{float(row_l_e['gain_vs_ctrl']):.6f}`, p_abs=`{float(row_l_e['perm_p_abs']):.6f}`",
                f"- ERA5+Lambda+Phi gain_vs_ctrl: `{float(row_lp_e['gain_vs_ctrl']):.6f}`",
                f"- ERA5+Lambda+Phi gain_vs_lambda: `{float(row_lp_e['gain_vs_lambda']):.6f}`, p_inc=`{float(row_lp_e['perm_p_inc']):.6f}`",
                f"- shuffled control gain_vs_lambda: `{float(row_lps_e['gain_vs_lambda']):.6f}`, p_inc=`{float(row_lps_e['perm_p_inc']):.6f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "- `strict_metrics_2019.csv`",
            "- `strict_oof_2019.csv`",
            "- `strict_active_calm_2019.csv`",
            "- `strict_quarterly_2019.csv`",
            "- `strict_phi_screening_train.csv`",
            "- `strict_meta.json`",
        ]
    )
    if len(met_ext) > 0:
        lines.append("- `strict_metrics_external.csv`")
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[M-GKSL-STRICT] done")
    print("\n[2019 metrics]")
    print(met_2019.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    if len(met_ext) > 0:
        print("\n[external metrics]")
        print(met_ext.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print(f"\nSaved: {outdir}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
