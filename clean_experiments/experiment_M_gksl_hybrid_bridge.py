#!/usr/bin/env python3
"""Experiment M hybrid bridge: add GKSL memory proxy from A05.R6 to M1 closure.

This script transfers canonical GKSL parameters from A05.R6 (best G001 config)
to M1 timeseries and evaluates hybrid ablations with the same blocked-CV
protocol used in M1:

1) ERA5 only (ctrl)
2) ERA5 + Lambda
3) ERA5 + Phi_GKSL
4) ERA5 + Lambda + Phi_GKSL
5) ERA5 + Lambda + Phi_GKSL_shuffled
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        _block_permute,
        _blocked_splits,
        _evaluate_splits,
        _permutation_test,
        _zscore,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        _block_permute,
        _blocked_splits,
        _evaluate_splits,
        _permutation_test,
        _zscore,
    )


EPS = 1e-12


@dataclass(frozen=True)
class GkslConfig:
    config_id: str
    dephase_base: float
    dephase_comm_scale: float
    relax_base: float
    relax_comm_scale: float
    measurement_rate: float
    hamiltonian_scale: float


def _safe_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    yt = np.asarray(y, dtype=float)
    yp = np.asarray(yhat, dtype=float)
    if len(yt) == 0:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    y_mu = float(np.mean(yt))
    ss_tot = float(np.sum((yt - y_mu) ** 2))
    if ss_tot < EPS:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _robust_unit_interval(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    if len(z) == 0:
        return z
    q05 = float(np.nanquantile(z, 0.05))
    q95 = float(np.nanquantile(z, 0.95))
    scale = max(q95 - q05, 1e-8)
    out = (z - q05) / scale
    return np.clip(out, 0.0, 1.0)


def _project_density_matrix(rho: np.ndarray) -> np.ndarray:
    h = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(h)
    vals = np.clip(np.real(vals), 0.0, None)
    tr = float(np.sum(vals))
    if tr <= EPS:
        vals = np.array([0.5, 0.5], dtype=float)
        tr = 1.0
    vals = vals / tr
    out = vecs @ np.diag(vals) @ vecs.conj().T
    out = 0.5 * (out + out.conj().T)
    return out


def _dm_from_inst(p_l: float, p_2l: float, rho12: float) -> np.ndarray:
    p_l_c = _clip01(p_l)
    p_2l_c = _clip01(p_2l)
    z = p_l_c + p_2l_c
    if z <= EPS:
        p_l_c, p_2l_c = 0.5, 0.5
    else:
        p_l_c, p_2l_c = p_l_c / z, p_2l_c / z
    cap = np.sqrt(max(p_l_c * p_2l_c, 0.0))
    c = float(np.clip(float(rho12), -cap, cap))
    return np.array([[p_l_c, c], [c, p_2l_c]], dtype=np.complex128)


def _apply_kraus(rho: np.ndarray, ks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((2, 2), dtype=np.complex128)
    for k in ks:
        out += k @ rho @ k.conj().T
    return out


def _unitary_sigma_z(phase: float) -> np.ndarray:
    e_neg = np.exp(-1j * float(phase))
    e_pos = np.exp(1j * float(phase))
    return np.array([[e_neg, 0.0], [0.0, e_pos]], dtype=np.complex128)


def _apply_dephase_channel(rho: np.ndarray, gamma_phi: float) -> np.ndarray:
    g = _clip01(gamma_phi)
    k0 = np.sqrt(max(1.0 - g, 0.0)) * np.eye(2, dtype=np.complex128)
    k1 = np.sqrt(g) * np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    k2 = np.sqrt(g) * np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    return _apply_kraus(rho, [k0, k1, k2])


def _apply_gad_channel(rho: np.ndarray, gamma_relax: float, p_eq: float) -> np.ndarray:
    g = _clip01(gamma_relax)
    p = _clip01(p_eq)
    k0 = np.sqrt(p) * np.array([[1.0, 0.0], [0.0, np.sqrt(max(1.0 - g, 0.0))]], dtype=np.complex128)
    k1 = np.sqrt(p) * np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=np.complex128)
    k2 = np.sqrt(max(1.0 - p, 0.0)) * np.array([[np.sqrt(max(1.0 - g, 0.0)), 0.0], [0.0, 1.0]], dtype=np.complex128)
    k3 = np.sqrt(max(1.0 - p, 0.0)) * np.array([[0.0, 0.0], [np.sqrt(g), 0.0]], dtype=np.complex128)
    return _apply_kraus(rho, [k0, k1, k2, k3])


def _density_metrics(rho: np.ndarray) -> tuple[float, float, float, float, float]:
    p_l = float(np.real(rho[0, 0]))
    p_2l = float(np.real(rho[1, 1]))
    c = complex(rho[0, 1])
    c_re = float(np.real(c))
    c_abs = float(np.abs(c))
    cap = np.sqrt(max(p_l * p_2l, 0.0))
    eta_eff = float(c_abs / max(cap, EPS)) if cap > EPS else 0.0
    tr_err = float(abs(np.trace(rho) - 1.0))
    det = float(np.real(p_l * p_2l - c_abs * c_abs))
    return p_l, p_2l, c_re, eta_eff, tr_err + max(0.0, -det)


def _read_gksl_config(final_csv: Path, prefer_config_id: str) -> GkslConfig:
    df = pd.read_csv(final_csv)
    if len(df) == 0:
        raise ValueError(f"Empty GKSL final csv: {final_csv}")

    sel = df[df["config_id"].astype(str) == str(prefer_config_id)]
    if len(sel) == 0:
        pass_mask = df["PASS_ALL"].astype(bool) if "PASS_ALL" in df.columns else np.zeros(len(df), dtype=bool)
        if int(np.sum(pass_mask)) > 0:
            sel = df[pass_mask].head(1)
        else:
            sel = df.head(1)
    row = sel.iloc[0]

    return GkslConfig(
        config_id=str(row["config_id"]),
        dephase_base=float(row["gksl_dephase_base"]),
        dephase_comm_scale=float(row["gksl_dephase_comm_scale"]),
        relax_base=float(row["gksl_relax_base"]),
        relax_comm_scale=float(row["gksl_relax_comm_scale"]),
        measurement_rate=float(row["gksl_measurement_rate"]),
        hamiltonian_scale=float(row["gksl_hamiltonian_scale"]),
    )


def _parse_time_hours(time_like: pd.Series) -> np.ndarray:
    t = pd.to_datetime(time_like, errors="coerce", utc=True)
    if t.isna().any():
        # fallback: equal spacing with nominal 6h cadence
        return np.arange(len(time_like), dtype=float) * 6.0
    dt_h = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    return np.asarray(dt_h, dtype=float)


def _pick_lambda_band(df: pd.DataFrame, idx: int, fallback_col: str = "lambda_struct") -> np.ndarray:
    col = f"lambda_mu_{idx:02d}"
    if col in df.columns:
        return np.asarray(df[col], dtype=float)
    cand = sorted([c for c in df.columns if c.startswith("lambda_mu_")])
    if len(cand) > idx:
        return np.asarray(df[cand[idx]], dtype=float)
    if fallback_col not in df.columns:
        raise ValueError(f"Could not resolve lambda band idx={idx}; fallback '{fallback_col}' missing.")
    return np.asarray(df[fallback_col], dtype=float)


def _build_gksl_proxy(
    *,
    sig_l: np.ndarray,
    sig_2l: np.ndarray,
    time_hours: np.ndarray,
    cfg: GkslConfig,
    dt_cap_hours: float,
) -> pd.DataFrame:
    x_l = np.asarray(sig_l, dtype=float)
    x_2l = np.asarray(sig_2l, dtype=float)
    th = np.asarray(time_hours, dtype=float)
    if not (len(x_l) == len(x_2l) == len(th)):
        raise ValueError("Signal/time length mismatch in GKSL proxy construction.")
    n = len(x_l)

    amp_l = _robust_unit_interval(np.abs(x_l))
    amp_2l = _robust_unit_interval(np.abs(x_2l))
    z = amp_l + amp_2l
    p_l_inst = np.full(n, 0.5, dtype=float)
    np.divide(amp_l, z, out=p_l_inst, where=z > EPS)
    p_2l_inst = 1.0 - p_l_inst

    a = _zscore(np.nan_to_num(x_l, nan=0.0, posinf=0.0, neginf=0.0))
    b = _zscore(np.nan_to_num(x_2l, nan=0.0, posinf=0.0, neginf=0.0))
    op_norm = np.abs(a - b)
    op_scale = float(np.nanquantile(op_norm, 0.75)) if n > 0 else 1.0
    op_scale = max(op_scale, 1e-6)
    op_norm_u = op_norm / op_scale

    inst_c = np.tanh(0.75 * a * b) * np.sqrt(np.maximum(p_l_inst * p_2l_inst, 0.0))

    p_l_out = np.zeros(n, dtype=float)
    p_2l_out = np.zeros(n, dtype=float)
    c_re_out = np.zeros(n, dtype=float)
    eta_out = np.zeros(n, dtype=float)
    phi_raw = np.zeros(n, dtype=float)
    gamma_phi = np.zeros(n, dtype=float)
    gamma_relax = np.zeros(n, dtype=float)
    reset_kappa = np.zeros(n, dtype=float)
    omega_raw = np.zeros(n, dtype=float)
    dt_hours_raw = np.zeros(n, dtype=float)
    cptp_err = np.zeros(n, dtype=float)

    rho_prev: np.ndarray | None = None
    t_prev = None
    for t in range(n):
        rho_inst = _dm_from_inst(
            p_l=float(p_l_inst[t]),
            p_2l=float(p_2l_inst[t]),
            rho12=float(inst_c[t]),
        )
        if t == 0 or rho_prev is None or t_prev is None:
            rho_next = rho_inst
            dt_h = 0.0
        else:
            dt_h = float(th[t] - t_prev)
            if not np.isfinite(dt_h) or dt_h <= 0.0:
                dt_h = 6.0
            dt_h = float(np.clip(dt_h, 1e-3, max(float(dt_cap_hours), 1e-3)))

            opn = float(op_norm_u[t]) if np.isfinite(op_norm_u[t]) else 0.0
            dephase_rate = max(0.0, float(cfg.dephase_base) + float(cfg.dephase_comm_scale) * opn)
            relax_rate = max(0.0, float(cfg.relax_base) + float(cfg.relax_comm_scale) * opn)
            g_phi = _clip01(1.0 - np.exp(-dephase_rate * dt_h))
            g_rel = _clip01(1.0 - np.exp(-relax_rate * dt_h))
            kappa = _clip01(1.0 - np.exp(-max(float(cfg.measurement_rate), 0.0) * dt_h))
            omega = max(float(cfg.hamiltonian_scale), 0.0) * np.sqrt(max(a[t] * a[t] + b[t] * b[t], 0.0))

            rho_pred = rho_prev
            if omega > 0.0:
                u = _unitary_sigma_z(omega * dt_h)
                rho_pred = u @ rho_pred @ u.conj().T
            rho_pred = _apply_dephase_channel(rho_pred, g_phi)
            p_eq = float(np.real(rho_inst[0, 0]))
            rho_pred = _apply_gad_channel(rho_pred, g_rel, p_eq)
            rho_next = _project_density_matrix((1.0 - kappa) * rho_pred + kappa * rho_inst)

            gamma_phi[t] = g_phi
            gamma_relax[t] = g_rel
            reset_kappa[t] = kappa
            omega_raw[t] = omega
            dt_hours_raw[t] = dt_h

        p1, p2, c_re, eta_eff, err = _density_metrics(rho_next)
        p_l_out[t] = p1
        p_2l_out[t] = p2
        c_re_out[t] = c_re
        eta_out[t] = eta_eff
        cptp_err[t] = err
        phi_raw[t] = float(2.0 * a[t] * b[t] * (p2 - p1))

        rho_prev = rho_next
        t_prev = float(th[t])

    out = pd.DataFrame(
        {
            "phi_gksl_raw": phi_raw,
            "phi_gksl": _zscore(phi_raw),
            "phi_gksl_popdiff": p_2l_out - p_l_out,
            "phi_gksl_coh_re": c_re_out,
            "phi_gksl_eta": eta_out,
            "gksl_gamma_dephase": gamma_phi,
            "gksl_gamma_relax": gamma_relax,
            "gksl_reset_kappa": reset_kappa,
            "gksl_omega": omega_raw,
            "gksl_dt_hours": dt_hours_raw,
            "gksl_cptp_violation": cptp_err,
            "proxy_sig_l": x_l,
            "proxy_sig_2l": x_2l,
        }
    )
    return out


def _phi_candidate_map(proxy_df: pd.DataFrame, sig_l: np.ndarray, sig_2l: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(proxy_df["phi_gksl_raw"], dtype=float)
    pop = np.asarray(proxy_df["phi_gksl_popdiff"], dtype=float)
    coh = np.asarray(proxy_df["phi_gksl_coh_re"], dtype=float)
    eta = np.asarray(proxy_df["phi_gksl_eta"], dtype=float)
    delta = _zscore(np.asarray(sig_l, dtype=float) - np.asarray(sig_2l, dtype=float))
    dphi = np.gradient(raw) if len(raw) > 2 else np.zeros_like(raw)
    return {
        "raw": _zscore(raw),
        "popdiff": _zscore(pop),
        "coh": _zscore(coh),
        "eta": _zscore(eta),
        "raw_x_eta": _zscore(raw * eta),
        "raw_plus_pop": _zscore(raw + 0.5 * pop),
        "dphi": _zscore(dphi),
        "pop_x_delta": _zscore(pop * np.abs(delta)),
    }


def _eval_pair(
    *,
    y: np.ndarray,
    x_base: np.ndarray,
    x_full: np.ndarray,
    base_feature_names: list[str],
    full_feature_names: list[str],
    ridge_alpha: float,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    permute_cols: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    splits = _blocked_splits(len(y), n_folds=n_folds)
    split_df, yhat_b, yhat_f = _evaluate_splits(
        y=y,
        x_base=x_base,
        x_full=x_full,
        base_feature_names=base_feature_names,
        full_feature_names=full_feature_names,
        splits=splits,
        ridge_alpha=ridge_alpha,
    )
    mae_b = float(np.mean(np.abs(y - yhat_b)))
    mae_f = float(np.mean(np.abs(y - yhat_f)))
    gain = float((mae_b - mae_f) / (mae_b + EPS))
    r2_b = _safe_r2(y, yhat_b)
    r2_f = _safe_r2(y, yhat_f)

    p_perm, perm_df, stat_real = _permutation_test(
        y=y,
        x_base=x_base,
        x_full=x_full,
        base_feature_names=base_feature_names,
        full_feature_names=full_feature_names,
        permute_cols=np.asarray(permute_cols, dtype=int),
        splits=splits,
        ridge_alpha=ridge_alpha,
        n_perm=n_perm,
        perm_block=perm_block,
        seed=seed,
    )

    split_gains = split_df["mae_gain_frac"].to_numpy(dtype=float)
    metrics = {
        "n_rows": float(len(y)),
        "mae_base_oof": mae_b,
        "mae_full_oof": mae_f,
        "oof_gain_frac": gain,
        "r2_base_oof": r2_b,
        "r2_full_oof": r2_f,
        "r2_gain": float((r2_f - r2_b) if np.isfinite(r2_b) and np.isfinite(r2_f) else np.nan),
        "split_gain_median": float(np.median(split_gains)),
        "split_gain_min": float(np.min(split_gains)),
        "perm_stat_real_median_gain": float(stat_real),
        "perm_p_value": float(p_perm),
    }
    return metrics, split_df, perm_df, yhat_b, yhat_f


def _active_calm_rows(
    *,
    y: np.ndarray,
    yhat_base: np.ndarray,
    yhat_full: np.ndarray,
    yhat_shuffled: np.ndarray,
    activity: np.ndarray,
    active_quantile: float,
) -> pd.DataFrame:
    thr = float(np.quantile(activity, active_quantile))
    is_active = activity >= thr
    rows = []
    for regime_name, mask in (("active", is_active), ("calm", ~is_active), ("all", np.ones(len(y), dtype=bool))):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        y_sub = y[idx]
        mae_base = float(np.mean(np.abs(y_sub - yhat_base[idx])))
        mae_full = float(np.mean(np.abs(y_sub - yhat_full[idx])))
        mae_shuf = float(np.mean(np.abs(y_sub - yhat_shuffled[idx])))
        gain = float((mae_base - mae_full) / (mae_base + EPS))
        gain_shuf = float((mae_base - mae_shuf) / (mae_base + EPS))
        rows.append(
            {
                "regime": regime_name,
                "n_rows": int(len(idx)),
                "activity_threshold": thr,
                "mae_base": mae_base,
                "mae_full": mae_full,
                "mae_shuffled": mae_shuf,
                "gain_full_frac": gain,
                "gain_shuffled_frac": gain_shuf,
                "active_minus_calm_full": np.nan,
                "active_minus_calm_shuffled": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if {"active", "calm"}.issubset(set(out["regime"].tolist())):
        ga = float(out.loc[out["regime"] == "active", "gain_full_frac"].iloc[0])
        gc = float(out.loc[out["regime"] == "calm", "gain_full_frac"].iloc[0])
        sa = float(out.loc[out["regime"] == "active", "gain_shuffled_frac"].iloc[0])
        sc = float(out.loc[out["regime"] == "calm", "gain_shuffled_frac"].iloc[0])
        out.loc[out["regime"] == "all", "active_minus_calm_full"] = ga - gc
        out.loc[out["regime"] == "all", "active_minus_calm_shuffled"] = sa - sc
    return out


def _surface_eval(
    *,
    surface_df: pd.DataFrame,
    cfg: GkslConfig,
    ridge_alpha: float,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    seed: int,
    dt_hours: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for sid, (surface, sub) in enumerate(surface_df.groupby("surface", sort=True)):
        sub = sub.sort_values("time_index").reset_index(drop=True)
        y = np.asarray(sub["residual_z"], dtype=float)
        ctrl = np.asarray(sub["ctrl_z"], dtype=float)
        lam = np.asarray(sub["lambda"], dtype=float)
        lam_lag = np.r_[lam[0], lam[:-1]]
        t_hours = np.arange(len(sub), dtype=float) * float(dt_hours)
        phi_df = _build_gksl_proxy(
            sig_l=lam,
            sig_2l=lam_lag,
            time_hours=t_hours,
            cfg=cfg,
            dt_cap_hours=dt_hours,
        )
        phi = np.asarray(phi_df["phi_gksl"], dtype=float)
        phi_shuf = _block_permute(phi[:, None], block=perm_block, rng=np.random.default_rng(seed + 170 + sid)).reshape(-1)

        valid = np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi) & np.isfinite(phi_shuf)
        yv = y[valid]
        xv_ctrl = ctrl[valid][:, None]
        xv_lam = np.column_stack([ctrl[valid], lam[valid]])
        xv_full = np.column_stack([ctrl[valid], lam[valid], phi[valid]])
        xv_shuf = np.column_stack([ctrl[valid], lam[valid], phi_shuf[valid]])
        if len(yv) < max(80, n_folds * 8):
            continue

        m_lam, _, _, yhat_ctrl, yhat_lam = _eval_pair(
            y=yv,
            x_base=xv_ctrl,
            x_full=xv_lam,
            base_feature_names=["ctrl"],
            full_feature_names=["ctrl", "lambda"],
            ridge_alpha=ridge_alpha,
            n_folds=n_folds,
            n_perm=n_perm,
            perm_block=perm_block,
            permute_cols=np.array([1], dtype=int),
            seed=seed + sid * 11,
        )
        m_inc, _, _, yhat_base_inc, yhat_full_inc = _eval_pair(
            y=yv,
            x_base=xv_lam,
            x_full=xv_full,
            base_feature_names=["ctrl", "lambda"],
            full_feature_names=["ctrl", "lambda", "phi_gksl"],
            ridge_alpha=ridge_alpha,
            n_folds=n_folds,
            n_perm=n_perm,
            perm_block=perm_block,
            permute_cols=np.array([2], dtype=int),
            seed=seed + sid * 11 + 1,
        )
        m_shuf, _, _, _, yhat_shuf = _eval_pair(
            y=yv,
            x_base=xv_lam,
            x_full=xv_shuf,
            base_feature_names=["ctrl", "lambda"],
            full_feature_names=["ctrl", "lambda", "phi_gksl_shuffled"],
            ridge_alpha=ridge_alpha,
            n_folds=n_folds,
            n_perm=n_perm,
            perm_block=perm_block,
            permute_cols=np.array([2], dtype=int),
            seed=seed + sid * 11 + 2,
        )

        mae_ctrl = float(np.mean(np.abs(yv - yhat_ctrl)))
        mae_lam = float(np.mean(np.abs(yv - yhat_lam)))
        mae_full = float(np.mean(np.abs(yv - yhat_full_inc)))
        mae_shuf = float(np.mean(np.abs(yv - yhat_shuf)))
        rows.append(
            {
                "surface": str(surface),
                "n_rows": float(len(yv)),
                "gain_lambda_vs_ctrl": float((mae_ctrl - mae_lam) / (mae_ctrl + EPS)),
                "gain_lambda_phi_vs_lambda": float((mae_lam - mae_full) / (mae_lam + EPS)),
                "gain_lambda_phi_shuffled_vs_lambda": float((mae_lam - mae_shuf) / (mae_lam + EPS)),
                "perm_p_lambda": float(m_lam["perm_p_value"]),
                "perm_p_phi_given_lambda": float(m_inc["perm_p_value"]),
                "perm_p_phi_shuffled_given_lambda": float(m_shuf["perm_p_value"]),
                "mae_ctrl": mae_ctrl,
                "mae_lambda": mae_lam,
                "mae_lambda_phi": mae_full,
                "mae_lambda_phi_shuffled": mae_shuf,
            }
        )
    return pd.DataFrame(rows).sort_values("surface").reset_index(drop=True) if rows else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--m-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    p.add_argument(
        "--m-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_summary.csv"),
    )
    p.add_argument(
        "--gksl-final-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory_gksl_cptp/gksl_final_l8.csv"),
    )
    p.add_argument(
        "--surface-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_land_ocean_split/surface_timeseries.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_gksl_hybrid_bridge"),
    )
    p.add_argument("--gksl-config-id", type=str, default="G001")
    p.add_argument("--fine-band-idx", type=int, default=0)
    p.add_argument("--coarse-band-idx", type=int, default=1)
    p.add_argument(
        "--lock-feature",
        type=str,
        default=None,
        choices=["raw", "popdiff", "coh", "eta", "raw_x_eta", "raw_plus_pop", "dphi", "pop_x_delta"],
        help="If set, skip screening and use this fixed phi candidate on (fine-band-idx, coarse-band-idx).",
    )
    p.add_argument("--screen-band-idxs", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--gksl-dt-cap-hours", type=float, default=6.0)
    p.add_argument("--surface-dt-hours", type=float, default=6.0)
    p.add_argument("--ridge-alpha", type=float, default=1e-6)
    p.add_argument("--n-folds", type=int, default=6)
    p.add_argument("--n-perm", type=int, default=140)
    p.add_argument("--perm-block", type=int, default=24)
    p.add_argument("--active-quantile", type=float, default=0.67)
    p.add_argument("--seed", type=int, default=20260307)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    m_df = pd.read_csv(args.m_timeseries_csv).sort_values("time_index").reset_index(drop=True)
    gksl_cfg = _read_gksl_config(args.gksl_final_csv, prefer_config_id=str(args.gksl_config_id))

    if "residual_base_res0" not in m_df.columns or "n_density_ctrl_z" not in m_df.columns or "lambda_struct" not in m_df.columns:
        raise ValueError("M timeseries csv must contain residual_base_res0, n_density_ctrl_z, lambda_struct.")

    t_hours = _parse_time_hours(m_df["time"])
    y = np.asarray(m_df["residual_base_res0"], dtype=float)
    ctrl = np.asarray(m_df["n_density_ctrl_z"], dtype=float)
    lam = np.asarray(m_df["lambda_struct"], dtype=float)

    # Step 1: choose phi candidate. By default we screen by incremental gain over Lambda.
    if args.lock_feature is not None:
        selection_mode = "locked"
        sel_i = int(args.fine_band_idx)
        sel_j = int(args.coarse_band_idx)
        sel_feat = str(args.lock_feature)
        screen_df = pd.DataFrame(
            [
                {
                    "selection_mode": "locked",
                    "band_l_idx": int(sel_i),
                    "band_2l_idx": int(sel_j),
                    "feature_name": str(sel_feat),
                    "n_valid": np.nan,
                    "mae_lambda_oof": np.nan,
                    "mae_lambda_phi_oof": np.nan,
                    "gain_vs_lambda": np.nan,
                    "split_gain_median": np.nan,
                    "split_gain_min": np.nan,
                    "corr_phi_lambda": np.nan,
                    "corr_phi_residual": np.nan,
                }
            ]
        )
    else:
        band_pool = sorted(set([int(args.fine_band_idx), int(args.coarse_band_idx)] + [int(x) for x in args.screen_band_idxs]))
        screen_rows: list[dict[str, float | int | str]] = []
        fallback_i = int(args.fine_band_idx)
        fallback_j = int(args.coarse_band_idx)
        for i in band_pool:
            for j in band_pool:
                if i >= j:
                    continue
                try:
                    sig_i = _pick_lambda_band(m_df, idx=int(i), fallback_col="lambda_struct")
                    sig_j = _pick_lambda_band(m_df, idx=int(j), fallback_col="lambda_struct")
                except Exception:
                    continue
                proxy_ij = _build_gksl_proxy(
                    sig_l=sig_i,
                    sig_2l=sig_j,
                    time_hours=t_hours,
                    cfg=gksl_cfg,
                    dt_cap_hours=float(args.gksl_dt_cap_hours),
                )
                cand_map = _phi_candidate_map(proxy_ij, sig_i, sig_j)
                for feat_name, phi_vec in cand_map.items():
                    valid = np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi_vec)
                    idv = np.where(valid)[0]
                    if len(idv) < max(80, int(args.n_folds) * 8):
                        continue
                    yv_s = y[idv]
                    x_base = np.column_stack([ctrl[idv], lam[idv]])
                    x_full = np.column_stack([ctrl[idv], lam[idv], phi_vec[idv]])
                    splits = _blocked_splits(len(idv), n_folds=int(args.n_folds))
                    split_df, yhat_b, yhat_f = _evaluate_splits(
                        y=yv_s,
                        x_base=x_base,
                        x_full=x_full,
                        base_feature_names=["ctrl", "lambda"],
                        full_feature_names=["ctrl", "lambda", "phi_gksl"],
                        splits=splits,
                        ridge_alpha=float(args.ridge_alpha),
                    )
                    mae_b = float(np.mean(np.abs(yv_s - yhat_b)))
                    mae_f = float(np.mean(np.abs(yv_s - yhat_f)))
                    screen_rows.append(
                        {
                            "selection_mode": "screened",
                            "band_l_idx": int(i),
                            "band_2l_idx": int(j),
                            "feature_name": str(feat_name),
                            "n_valid": int(len(idv)),
                            "mae_lambda_oof": mae_b,
                            "mae_lambda_phi_oof": mae_f,
                            "gain_vs_lambda": float((mae_b - mae_f) / (mae_b + EPS)),
                            "split_gain_median": float(np.median(split_df["mae_gain_frac"].to_numpy(dtype=float))),
                            "split_gain_min": float(np.min(split_df["mae_gain_frac"].to_numpy(dtype=float))),
                            "corr_phi_lambda": float(np.corrcoef(phi_vec[idv], lam[idv])[0, 1]) if np.std(phi_vec[idv]) > EPS else np.nan,
                            "corr_phi_residual": float(np.corrcoef(phi_vec[idv], y[idv])[0, 1]) if np.std(phi_vec[idv]) > EPS else np.nan,
                        }
                    )

        if len(screen_rows) == 0:
            sig_i = _pick_lambda_band(m_df, idx=fallback_i, fallback_col="lambda_struct")
            sig_j = _pick_lambda_band(m_df, idx=fallback_j, fallback_col="lambda_struct")
            proxy_fb = _build_gksl_proxy(
                sig_l=sig_i,
                sig_2l=sig_j,
                time_hours=t_hours,
                cfg=gksl_cfg,
                dt_cap_hours=float(args.gksl_dt_cap_hours),
            )
            phi_fb = _phi_candidate_map(proxy_fb, sig_i, sig_j)["raw"]
            screen_df = pd.DataFrame(
                [
                    {
                        "selection_mode": "fallback",
                        "band_l_idx": int(fallback_i),
                        "band_2l_idx": int(fallback_j),
                        "feature_name": "raw",
                        "n_valid": int(np.sum(np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi_fb))),
                        "mae_lambda_oof": np.nan,
                        "mae_lambda_phi_oof": np.nan,
                        "gain_vs_lambda": np.nan,
                        "split_gain_median": np.nan,
                        "split_gain_min": np.nan,
                        "corr_phi_lambda": np.nan,
                        "corr_phi_residual": np.nan,
                    }
                ]
            )
        else:
            screen_df = pd.DataFrame(screen_rows).sort_values(
                ["gain_vs_lambda", "split_gain_median", "split_gain_min"],
                ascending=[False, False, False],
            ).reset_index(drop=True)

        best_screen = screen_df.iloc[0]
        selection_mode = str(best_screen["selection_mode"])
        sel_i = int(best_screen["band_l_idx"])
        sel_j = int(best_screen["band_2l_idx"])
        sel_feat = str(best_screen["feature_name"])

    sig_l = _pick_lambda_band(m_df, idx=sel_i, fallback_col="lambda_struct")
    sig_2l = _pick_lambda_band(m_df, idx=sel_j, fallback_col="lambda_struct")
    phi_state_df = _build_gksl_proxy(
        sig_l=sig_l,
        sig_2l=sig_2l,
        time_hours=t_hours,
        cfg=gksl_cfg,
        dt_cap_hours=float(args.gksl_dt_cap_hours),
    )
    candidate_map = _phi_candidate_map(phi_state_df, sig_l, sig_2l)
    if sel_feat not in candidate_map:
        sel_feat = "raw"
    phi_selected = np.asarray(candidate_map[sel_feat], dtype=float)

    work = m_df.copy()
    work["phi_gksl"] = phi_selected
    work["phi_gksl_state_raw"] = np.asarray(phi_state_df["phi_gksl_raw"], dtype=float)
    work["phi_gksl_state_popdiff"] = np.asarray(phi_state_df["phi_gksl_popdiff"], dtype=float)
    work["phi_gksl_state_coh_re"] = np.asarray(phi_state_df["phi_gksl_coh_re"], dtype=float)
    work["phi_gksl_state_eta"] = np.asarray(phi_state_df["phi_gksl_eta"], dtype=float)
    work["gksl_gamma_dephase"] = np.asarray(phi_state_df["gksl_gamma_dephase"], dtype=float)
    work["gksl_gamma_relax"] = np.asarray(phi_state_df["gksl_gamma_relax"], dtype=float)
    work["gksl_reset_kappa"] = np.asarray(phi_state_df["gksl_reset_kappa"], dtype=float)
    work["gksl_omega"] = np.asarray(phi_state_df["gksl_omega"], dtype=float)
    work["gksl_dt_hours"] = np.asarray(phi_state_df["gksl_dt_hours"], dtype=float)
    work["gksl_cptp_violation"] = np.asarray(phi_state_df["gksl_cptp_violation"], dtype=float)
    work["proxy_sig_l"] = np.asarray(sig_l, dtype=float)
    work["proxy_sig_2l"] = np.asarray(sig_2l, dtype=float)
    work["phi_selected_feature"] = str(sel_feat)
    work["phi_selected_band_l_idx"] = int(sel_i)
    work["phi_selected_band_2l_idx"] = int(sel_j)

    rng = np.random.default_rng(int(args.seed) + 900)
    phi_shuf = _block_permute(np.asarray(work["phi_gksl"], dtype=float)[:, None], block=int(args.perm_block), rng=rng).reshape(-1)
    work["phi_gksl_shuffled"] = phi_shuf

    phi = np.asarray(work["phi_gksl"], dtype=float)
    phi_s = np.asarray(work["phi_gksl_shuffled"], dtype=float)

    common_valid = np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam) & np.isfinite(phi) & np.isfinite(phi_s)
    idx = np.where(common_valid)[0]
    if len(idx) < max(80, int(args.n_folds) * 8):
        raise ValueError(f"Too few valid rows for hybrid evaluation: {len(idx)}")

    yv = y[idx]
    ctrlv = ctrl[idx]
    lamv = lam[idx]
    phiv = phi[idx]
    phisv = phi_s[idx]

    x_ctrl = ctrlv[:, None]
    x_lam = np.column_stack([ctrlv, lamv])
    x_phi = np.column_stack([ctrlv, phiv])
    x_lam_phi = np.column_stack([ctrlv, lamv, phiv])
    x_lam_phi_shuf = np.column_stack([ctrlv, lamv, phisv])

    m_lambda, split_lambda, perm_lambda, yhat_ctrl_lam, yhat_lambda = _eval_pair(
        y=yv,
        x_base=x_ctrl,
        x_full=x_lam,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl", "lambda"],
        ridge_alpha=float(args.ridge_alpha),
        n_folds=int(args.n_folds),
        n_perm=int(args.n_perm),
        perm_block=int(args.perm_block),
        permute_cols=np.array([1], dtype=int),
        seed=int(args.seed) + 1,
    )
    m_phi, split_phi, perm_phi, _, yhat_phi = _eval_pair(
        y=yv,
        x_base=x_ctrl,
        x_full=x_phi,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl", "phi_gksl"],
        ridge_alpha=float(args.ridge_alpha),
        n_folds=int(args.n_folds),
        n_perm=int(args.n_perm),
        perm_block=int(args.perm_block),
        permute_cols=np.array([1], dtype=int),
        seed=int(args.seed) + 2,
    )
    m_lam_phi_abs, split_lam_phi_abs, perm_lam_phi_abs, _, yhat_lam_phi = _eval_pair(
        y=yv,
        x_base=x_ctrl,
        x_full=x_lam_phi,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl", "lambda", "phi_gksl"],
        ridge_alpha=float(args.ridge_alpha),
        n_folds=int(args.n_folds),
        n_perm=int(args.n_perm),
        perm_block=int(args.perm_block),
        permute_cols=np.array([1, 2], dtype=int),
        seed=int(args.seed) + 3,
    )
    m_lam_phi_inc, split_lam_phi_inc, perm_lam_phi_inc, _, _ = _eval_pair(
        y=yv,
        x_base=x_lam,
        x_full=x_lam_phi,
        base_feature_names=["ctrl", "lambda"],
        full_feature_names=["ctrl", "lambda", "phi_gksl"],
        ridge_alpha=float(args.ridge_alpha),
        n_folds=int(args.n_folds),
        n_perm=int(args.n_perm),
        perm_block=int(args.perm_block),
        permute_cols=np.array([2], dtype=int),
        seed=int(args.seed) + 4,
    )
    m_lam_phi_shuf, split_lam_phi_shuf, perm_lam_phi_shuf, _, yhat_lam_phi_shuf = _eval_pair(
        y=yv,
        x_base=x_lam,
        x_full=x_lam_phi_shuf,
        base_feature_names=["ctrl", "lambda"],
        full_feature_names=["ctrl", "lambda", "phi_gksl_shuffled"],
        ridge_alpha=float(args.ridge_alpha),
        n_folds=int(args.n_folds),
        n_perm=int(args.n_perm),
        perm_block=int(args.perm_block),
        permute_cols=np.array([2], dtype=int),
        seed=int(args.seed) + 5,
    )

    yhat_ctrl = yhat_ctrl_lam
    mae_ctrl = float(np.mean(np.abs(yv - yhat_ctrl)))
    mae_lambda = float(np.mean(np.abs(yv - yhat_lambda)))
    mae_phi = float(np.mean(np.abs(yv - yhat_phi)))
    mae_lam_phi = float(np.mean(np.abs(yv - yhat_lam_phi)))
    mae_lam_phi_shuf = float(np.mean(np.abs(yv - yhat_lam_phi_shuf)))

    r2_ctrl = _safe_r2(yv, yhat_ctrl)
    r2_lambda = _safe_r2(yv, yhat_lambda)
    r2_phi = _safe_r2(yv, yhat_phi)
    r2_lam_phi = _safe_r2(yv, yhat_lam_phi)
    r2_lam_phi_shuf = _safe_r2(yv, yhat_lam_phi_shuf)

    metrics_rows = [
        {
            "model": "ERA5_only",
            "mae_oof": mae_ctrl,
            "r2_oof": r2_ctrl,
            "gain_vs_ctrl": 0.0,
            "gain_vs_lambda": np.nan,
            "perm_p_abs": np.nan,
            "perm_p_inc": np.nan,
        },
        {
            "model": "ERA5_plus_Lambda",
            "mae_oof": mae_lambda,
            "r2_oof": r2_lambda,
            "gain_vs_ctrl": float((mae_ctrl - mae_lambda) / (mae_ctrl + EPS)),
            "gain_vs_lambda": 0.0,
            "perm_p_abs": float(m_lambda["perm_p_value"]),
            "perm_p_inc": np.nan,
        },
        {
            "model": "ERA5_plus_Phi_GKSL",
            "mae_oof": mae_phi,
            "r2_oof": r2_phi,
            "gain_vs_ctrl": float((mae_ctrl - mae_phi) / (mae_ctrl + EPS)),
            "gain_vs_lambda": float((mae_lambda - mae_phi) / (mae_lambda + EPS)),
            "perm_p_abs": float(m_phi["perm_p_value"]),
            "perm_p_inc": np.nan,
        },
        {
            "model": "ERA5_plus_Lambda_plus_Phi_GKSL",
            "mae_oof": mae_lam_phi,
            "r2_oof": r2_lam_phi,
            "gain_vs_ctrl": float((mae_ctrl - mae_lam_phi) / (mae_ctrl + EPS)),
            "gain_vs_lambda": float((mae_lambda - mae_lam_phi) / (mae_lambda + EPS)),
            "perm_p_abs": float(m_lam_phi_abs["perm_p_value"]),
            "perm_p_inc": float(m_lam_phi_inc["perm_p_value"]),
        },
        {
            "model": "ERA5_plus_Lambda_plus_Phi_GKSL_shuffled",
            "mae_oof": mae_lam_phi_shuf,
            "r2_oof": r2_lam_phi_shuf,
            "gain_vs_ctrl": float((mae_ctrl - mae_lam_phi_shuf) / (mae_ctrl + EPS)),
            "gain_vs_lambda": float((mae_lambda - mae_lam_phi_shuf) / (mae_lambda + EPS)),
            "perm_p_abs": np.nan,
            "perm_p_inc": float(m_lam_phi_shuf["perm_p_value"]),
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    active_df = _active_calm_rows(
        y=yv,
        yhat_base=yhat_lambda,
        yhat_full=yhat_lam_phi,
        yhat_shuffled=yhat_lam_phi_shuf,
        activity=np.abs(lamv),
        active_quantile=float(args.active_quantile),
    )

    corr_rows = [
        {
            "n_rows": int(len(yv)),
            "corr_residual_lambda": float(np.corrcoef(yv, lamv)[0, 1]) if np.std(lamv) > EPS else np.nan,
            "corr_residual_phi_gksl": float(np.corrcoef(yv, phiv)[0, 1]) if np.std(phiv) > EPS else np.nan,
            "corr_lambda_phi_gksl": float(np.corrcoef(lamv, phiv)[0, 1]) if np.std(lamv) > EPS and np.std(phiv) > EPS else np.nan,
            "corr_abs_residual_phi_gksl": float(np.corrcoef(np.abs(yv), phiv)[0, 1]) if np.std(phiv) > EPS else np.nan,
        }
    ]
    corr_df = pd.DataFrame(corr_rows)

    oof = pd.DataFrame(
        {
            "time_index": np.asarray(work.loc[idx, "time_index"], dtype=int),
            "time": work.loc[idx, "time"].astype(str).to_numpy(),
            "residual_base_res0": yv,
            "n_density_ctrl_z": ctrlv,
            "lambda_struct": lamv,
            "phi_gksl": phiv,
            "phi_gksl_shuffled": phisv,
            "yhat_ctrl": yhat_ctrl,
            "yhat_lambda": yhat_lambda,
            "yhat_phi": yhat_phi,
            "yhat_lambda_phi": yhat_lam_phi,
            "yhat_lambda_phi_shuffled": yhat_lam_phi_shuf,
        }
    )

    # Save primary artifacts.
    work.to_csv(outdir / "hybrid_timeseries_with_gksl.csv", index=False)
    screen_df.to_csv(outdir / "hybrid_phi_screening.csv", index=False)
    metrics_df.to_csv(outdir / "hybrid_model_metrics.csv", index=False)
    corr_df.to_csv(outdir / "hybrid_correlation_summary.csv", index=False)
    active_df.to_csv(outdir / "hybrid_active_calm_summary.csv", index=False)
    oof.to_csv(outdir / "hybrid_oof_predictions.csv", index=False)

    split_lambda.to_csv(outdir / "splits_era5_plus_lambda.csv", index=False)
    split_phi.to_csv(outdir / "splits_era5_plus_phi_gksl.csv", index=False)
    split_lam_phi_abs.to_csv(outdir / "splits_era5_plus_lambda_plus_phi_abs.csv", index=False)
    split_lam_phi_inc.to_csv(outdir / "splits_era5_plus_lambda_plus_phi_inc.csv", index=False)
    split_lam_phi_shuf.to_csv(outdir / "splits_era5_plus_lambda_plus_phi_shuffled_inc.csv", index=False)

    perm_lambda.to_csv(outdir / "permutation_era5_plus_lambda.csv", index=False)
    perm_phi.to_csv(outdir / "permutation_era5_plus_phi_gksl.csv", index=False)
    perm_lam_phi_abs.to_csv(outdir / "permutation_era5_plus_lambda_plus_phi_abs.csv", index=False)
    perm_lam_phi_inc.to_csv(outdir / "permutation_era5_plus_lambda_plus_phi_inc.csv", index=False)
    perm_lam_phi_shuf.to_csv(outdir / "permutation_era5_plus_lambda_plus_phi_shuffled_inc.csv", index=False)

    # Surface split extension (M3 artifacts).
    surface_summary = pd.DataFrame()
    if args.surface_timeseries_csv.exists():
        surface_df = pd.read_csv(args.surface_timeseries_csv)
        required = {"surface", "time_index", "residual_z", "ctrl_z", "lambda"}
        if required.issubset(set(surface_df.columns)):
            surface_summary = _surface_eval(
                surface_df=surface_df,
                cfg=gksl_cfg,
                ridge_alpha=float(args.ridge_alpha),
                n_folds=int(args.n_folds),
                n_perm=int(args.n_perm),
                perm_block=int(args.perm_block),
                seed=int(args.seed) + 200,
                dt_hours=float(args.surface_dt_hours),
            )
            surface_summary.to_csv(outdir / "hybrid_surface_summary.csv", index=False)

    baseline_m1_gain = np.nan
    if args.m_summary_csv.exists():
        msum = pd.read_csv(args.m_summary_csv)
        if "oof_gain_frac" in msum.columns and len(msum) > 0:
            baseline_m1_gain = float(msum["oof_gain_frac"].iloc[0])

    best_row = metrics_df.loc[metrics_df["model"] == "ERA5_plus_Lambda_plus_Phi_GKSL"].iloc[0]
    shuf_row = metrics_df.loc[metrics_df["model"] == "ERA5_plus_Lambda_plus_Phi_GKSL_shuffled"].iloc[0]
    lam_row = metrics_df.loc[metrics_df["model"] == "ERA5_plus_Lambda"].iloc[0]

    cfg_payload = {
        "gksl_config": {
            "config_id": gksl_cfg.config_id,
            "dephase_base": gksl_cfg.dephase_base,
            "dephase_comm_scale": gksl_cfg.dephase_comm_scale,
            "relax_base": gksl_cfg.relax_base,
            "relax_comm_scale": gksl_cfg.relax_comm_scale,
            "measurement_rate": gksl_cfg.measurement_rate,
            "hamiltonian_scale": gksl_cfg.hamiltonian_scale,
        },
        "inputs": {
            "m_timeseries_csv": str(args.m_timeseries_csv),
            "m_summary_csv": str(args.m_summary_csv),
            "gksl_final_csv": str(args.gksl_final_csv),
            "surface_timeseries_csv": str(args.surface_timeseries_csv),
        },
        "runtime": {
            "selection_mode": str(selection_mode),
            "requested_fine_band_idx": int(args.fine_band_idx),
            "requested_coarse_band_idx": int(args.coarse_band_idx),
            "screen_band_idxs": [int(x) for x in args.screen_band_idxs],
            "selected_band_l_idx": int(sel_i),
            "selected_band_2l_idx": int(sel_j),
            "selected_feature_name": str(sel_feat),
            "gksl_dt_cap_hours": float(args.gksl_dt_cap_hours),
            "ridge_alpha": float(args.ridge_alpha),
            "n_folds": int(args.n_folds),
            "n_perm": int(args.n_perm),
            "perm_block": int(args.perm_block),
            "active_quantile": float(args.active_quantile),
            "seed": int(args.seed),
            "n_common_valid": int(len(idx)),
        },
    }
    (outdir / "hybrid_config_used.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    max_cptp = float(np.nanmax(work["gksl_cptp_violation"].to_numpy(dtype=float)))
    mean_gamma = float(np.nanmean(work["gksl_gamma_dephase"].to_numpy(dtype=float)))
    mean_relax = float(np.nanmean(work["gksl_gamma_relax"].to_numpy(dtype=float)))
    mean_kappa = float(np.nanmean(work["gksl_reset_kappa"].to_numpy(dtype=float)))

    lines = [
        "# Experiment M GKSL Hybrid Bridge",
        "",
        "## Setup",
        f"- M1 timeseries: `{args.m_timeseries_csv}`",
        f"- M1 summary: `{args.m_summary_csv}`",
        f"- R6 config source: `{args.gksl_final_csv}`",
        f"- selected GKSL config: `{gksl_cfg.config_id}`",
        f"- phi selection mode: `{selection_mode}`",
        f"- selected proxy bands: `{sel_i}` / `{sel_j}`",
        f"- selected phi candidate: `{sel_feat}`",
        f"- common valid rows: `{int(len(idx))}`",
        "",
        "## Headline metrics",
        f"- baseline M1 oof_gain_frac (from summary): `{baseline_m1_gain:.6f}`",
        f"- ERA5+Lambda gain_vs_ctrl: `{float(lam_row['gain_vs_ctrl']):.6f}`, perm_p=`{float(lam_row['perm_p_abs']):.6f}`",
        f"- ERA5+Lambda+Phi_GKSL gain_vs_ctrl: `{float(best_row['gain_vs_ctrl']):.6f}`",
        f"- ERA5+Lambda+Phi_GKSL gain_vs_lambda: `{float(best_row['gain_vs_lambda']):.6f}`, "
        f"perm_p_inc=`{float(best_row['perm_p_inc']):.6f}`",
        f"- shuffled control gain_vs_lambda: `{float(shuf_row['gain_vs_lambda']):.6f}`, "
        f"perm_p_inc=`{float(shuf_row['perm_p_inc']):.6f}`",
        "",
        "## GKSL diagnostics",
        f"- max cptp violation proxy: `{max_cptp:.3e}`",
        f"- mean gamma_dephase: `{mean_gamma:.6f}`",
        f"- mean gamma_relax: `{mean_relax:.6f}`",
        f"- mean reset_kappa: `{mean_kappa:.6f}`",
    ]
    if len(surface_summary) > 0:
        lines.extend(
            [
                "",
                "## Surface split (land/ocean)",
            ]
        )
        for _, r in surface_summary.iterrows():
            lines.append(
                f"- {r['surface']}: gain_lambda_vs_ctrl={float(r['gain_lambda_vs_ctrl']):.6f}, "
                f"gain_phi_vs_lambda={float(r['gain_lambda_phi_vs_lambda']):.6f}, "
                f"gain_shuf_vs_lambda={float(r['gain_lambda_phi_shuffled_vs_lambda']):.6f}, "
                f"perm_p_phi={float(r['perm_p_phi_given_lambda']):.6f}"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "- `hybrid_model_metrics.csv`",
            "- `hybrid_phi_screening.csv`",
            "- `hybrid_oof_predictions.csv`",
            "- `hybrid_active_calm_summary.csv`",
            "- `hybrid_correlation_summary.csv`",
            "- `hybrid_timeseries_with_gksl.csv`",
            "- `hybrid_config_used.json`",
        ]
    )
    if len(surface_summary) > 0:
        lines.append("- `hybrid_surface_summary.csv`")

    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[M-GKSL] done")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    if len(surface_summary) > 0:
        print("\n[M-GKSL] surface summary:")
        print(surface_summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print(f"\nSaved: {outdir}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
