#!/usr/bin/env python3
"""Experiment P2-memory GKSL/CPTP: full density-matrix memory on l=8.

This experiment replaces lag-weighted surrogate memory with explicit state
propagation:

rho_t = Reset_t o GAD_t o Dephase_t o U_t (rho_{t-1})

where each block is CPTP. The model is evaluated with the same blocked
event-level CV and pass criteria as P2-memory surrogate.
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


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _project_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Project to nearest 2x2 PSD trace-1 Hermitian matrix."""
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
    """Generalized amplitude damping channel."""
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
    # For 2x2 Hermitian PSD state this should be >= 0.
    det = float(np.real(p_l * p_2l - c_abs * c_abs))
    return p_l, p_2l, c_re, eta_eff, tr_err + max(0.0, -det)


def _apply_gksl_cptp_bridge(
    df_in: pd.DataFrame,
    *,
    lambda_weights: np.ndarray,
    lambda_scale_power: float,
    decoherence_alpha: float,
    memory_scales: list[int],
    gksl_dephase_base: float,
    gksl_dephase_comm_scale: float,
    gksl_relax_base: float,
    gksl_relax_comm_scale: float,
    gksl_measurement_rate: float,
    gksl_hamiltonian_scale: float,
    gksl_dt_cap_hours: float,
) -> pd.DataFrame:
    base = _apply_theory_bridge(
        df_in,
        lambda_weights=lambda_weights,
        lambda_scale_power=lambda_scale_power,
        decoherence_alpha=decoherence_alpha,
    )
    n = len(base)
    if n == 0:
        return base

    memory_scales_set = {int(s) for s in memory_scales}
    scales = np.round(base["scale_l"].to_numpy(dtype=float)).astype(int)
    scale_apply_mask = np.isin(scales, np.asarray(sorted(memory_scales_set), dtype=int))

    rho11 = np.asarray(base["rho_11_raw"], dtype=float).copy()
    rho22 = np.asarray(base["rho_22_raw"], dtype=float).copy()
    rho12 = np.asarray(base["rho_12_raw"], dtype=float).copy()
    purity = np.asarray(base["rho_purity_raw"], dtype=float).copy()
    eta_eff = np.asarray(base["decoherence_eta_raw"], dtype=float).copy()
    comm_raw = np.asarray(base["comm_defect_raw"], dtype=float).copy()
    lam_raw = np.asarray(base["lambda_local_raw"], dtype=float).copy()
    gen_a = np.asarray(base["gen_a_raw"], dtype=float)
    gen_b = np.asarray(base["gen_b_raw"], dtype=float)
    op_norm = np.abs(np.asarray(base["comm_defect_operator_raw"], dtype=float))

    gamma_phi = np.zeros(n, dtype=float)
    gamma_relax = np.zeros(n, dtype=float)
    reset_kappa = np.zeros(n, dtype=float)
    omega_raw = np.zeros(n, dtype=float)
    dt_hours_raw = np.zeros(n, dtype=float)
    cptp_violation_raw = np.zeros(n, dtype=float)
    memory_applied = np.zeros(n, dtype=bool)

    base_rho11 = np.asarray(base["rho_11_raw"], dtype=float)
    base_rho22 = np.asarray(base["rho_22_raw"], dtype=float)
    base_rho12 = np.asarray(base["rho_12_raw"], dtype=float)

    ts = pd.to_datetime(base["mrms_obs_time_utc"], utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy(dtype=np.int64)

    work = base[["event_id", "scale_l", "tile_iy", "tile_ix"]].copy()
    work["row_id"] = np.arange(n, dtype=int)
    work["mrms_obs_ns"] = ts_ns
    work = work.sort_values(["event_id", "scale_l", "tile_iy", "tile_ix", "mrms_obs_ns"]).reset_index(drop=True)

    for _, g in work.groupby(["event_id", "scale_l", "tile_iy", "tile_ix"], sort=False):
        idx = g["row_id"].to_numpy(dtype=int)
        if len(idx) == 0:
            continue
        s_int = int(scales[idx[0]])
        if s_int not in memory_scales_set:
            continue

        tvals_ns = g["mrms_obs_ns"].to_numpy(dtype=np.int64)
        rho_prev: np.ndarray | None = None
        t_prev_ns: int | None = None

        for pos, row_idx in enumerate(idx):
            rho_inst = _dm_from_inst(
                p_l=float(base_rho11[row_idx]),
                p_2l=float(base_rho22[row_idx]),
                rho12=float(base_rho12[row_idx]),
            )

            if pos == 0 or rho_prev is None or t_prev_ns is None:
                rho_next = rho_inst
                dt_h = 0.0
            else:
                t_cur_ns = int(tvals_ns[pos])
                dt_h = float((t_cur_ns - t_prev_ns) / 3.6e12)
                if not np.isfinite(dt_h) or dt_h <= 0.0:
                    dt_h = 1.0
                dt_h = float(np.clip(dt_h, 1e-3, max(float(gksl_dt_cap_hours), 1e-3)))

                opn = float(op_norm[row_idx]) if np.isfinite(op_norm[row_idx]) else 0.0
                dephase_rate = max(0.0, float(gksl_dephase_base) + float(gksl_dephase_comm_scale) * opn)
                relax_rate = max(0.0, float(gksl_relax_base) + float(gksl_relax_comm_scale) * opn)
                g_phi = _clip01(1.0 - np.exp(-dephase_rate * dt_h))
                g_rel = _clip01(1.0 - np.exp(-relax_rate * dt_h))
                kappa = _clip01(1.0 - np.exp(-max(float(gksl_measurement_rate), 0.0) * dt_h))

                a = float(gen_a[row_idx]) if np.isfinite(gen_a[row_idx]) else 0.0
                b = float(gen_b[row_idx]) if np.isfinite(gen_b[row_idx]) else 0.0
                omega = max(float(gksl_hamiltonian_scale), 0.0) * np.sqrt(max(a * a + b * b, 0.0))

                rho_pred = rho_prev
                if omega > 0.0:
                    u = _unitary_sigma_z(omega * dt_h)
                    rho_pred = u @ rho_pred @ u.conj().T
                rho_pred = _apply_dephase_channel(rho_pred, g_phi)
                p_eq = float(np.real(rho_inst[0, 0]))
                rho_pred = _apply_gad_channel(rho_pred, g_rel, p_eq)
                rho_next = _project_density_matrix((1.0 - kappa) * rho_pred + kappa * rho_inst)

                gamma_phi[row_idx] = g_phi
                gamma_relax[row_idx] = g_rel
                reset_kappa[row_idx] = kappa
                omega_raw[row_idx] = omega
                dt_hours_raw[row_idx] = dt_h
                memory_applied[row_idx] = True

            p_l_n, p_2l_n, c_re, eta_n, cptp_err = _density_metrics(rho_next)
            rho11[row_idx] = p_l_n
            rho22[row_idx] = p_2l_n
            rho12[row_idx] = c_re
            purity[row_idx] = float(np.real(np.trace(rho_next @ rho_next)))
            eta_eff[row_idx] = eta_n
            cptp_violation_raw[row_idx] = cptp_err
            comm_raw[row_idx] = float(np.sqrt(8.0) * np.abs(gen_a[row_idx] * gen_b[row_idx]))
            lam_raw[row_idx] = float(2.0 * gen_a[row_idx] * gen_b[row_idx] * (p_2l_n - p_l_n))

            rho_prev = rho_next
            t_prev_ns = int(tvals_ns[pos])

    out = base.copy()
    # Overwrite only where GKSL/CPTP is active by scale.
    idx_apply = np.where(scale_apply_mask)[0]
    out.loc[idx_apply, "rho_11_raw"] = rho11[idx_apply]
    out.loc[idx_apply, "rho_22_raw"] = rho22[idx_apply]
    out.loc[idx_apply, "rho_12_raw"] = rho12[idx_apply]
    out.loc[idx_apply, "rho_purity_raw"] = purity[idx_apply]
    out.loc[idx_apply, "decoherence_eta_raw"] = eta_eff[idx_apply]
    out.loc[idx_apply, "comm_defect_raw"] = comm_raw[idx_apply]
    out.loc[idx_apply, "lambda_local_raw"] = lam_raw[idx_apply]

    out["memory_applied"] = memory_applied.astype(bool)
    out["gksl_gamma_dephase_raw"] = gamma_phi.astype(float)
    out["gksl_gamma_relax_raw"] = gamma_relax.astype(float)
    out["gksl_reset_kappa_raw"] = reset_kappa.astype(float)
    out["gksl_omega_raw"] = omega_raw.astype(float)
    out["gksl_dt_hours_raw"] = dt_hours_raw.astype(float)
    out["gksl_cptp_violation_raw"] = cptp_violation_raw.astype(float)

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
        "--surrogate-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory/summary_metrics.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory_gksl_cptp"),
    )
    p.add_argument("--target", choices=["density", "occupancy"], default="density")
    p.add_argument("--scales-cells", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--memory-scales", nargs="+", type=int, default=[8])

    p.add_argument("--gksl-dephase-bases", nargs="+", type=float, default=[0.4, 0.8])
    p.add_argument("--gksl-dephase-comm-scales", nargs="+", type=float, default=[0.4])
    p.add_argument("--gksl-relax-bases", nargs="+", type=float, default=[0.4, 0.8])
    p.add_argument("--gksl-relax-comm-scales", nargs="+", type=float, default=[0.0])
    p.add_argument("--gksl-measurement-rates", nargs="+", type=float, default=[0.8, 1.6])
    p.add_argument("--gksl-hamiltonian-scales", nargs="+", type=float, default=[0.2, 0.6])
    p.add_argument("--gksl-dt-cap-hours", type=float, default=6.0)

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
        print(f"[gksl] loading base tile dataset: {args.tile_csv}", flush=True)
        base_df = pd.read_csv(args.tile_csv)
    else:
        print(f"[gksl] tile csv not found, rebuilding from panel: {args.panel_csv}", flush=True)
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

    print("[gksl] screening GKSL/CPTP configs on l=8", flush=True)
    screen_rows: list[dict[str, float | bool | str]] = []
    cfg_id = 0
    grid = list(
        product(
        args.gksl_dephase_bases,
        args.gksl_dephase_comm_scales,
        args.gksl_relax_bases,
        args.gksl_relax_comm_scales,
        args.gksl_measurement_rates,
        args.gksl_hamiltonian_scales,
        )
    )
    n_cfg = len(grid)
    print(f"[gksl] screening configs: {n_cfg}", flush=True)
    for d_base, d_comm, r_base, r_comm, m_rate, h_scale in grid:
        cfg_id += 1
        cid = f"G{cfg_id:03d}"
        if cfg_id == 1 or cfg_id % 4 == 0 or cfg_id == n_cfg:
            print(f"[gksl] screening {cfg_id}/{n_cfg} ({cid})", flush=True)
        df_cfg = _apply_gksl_cptp_bridge(
            base_df,
            lambda_weights=np.asarray(args.lambda_weights, dtype=float),
            lambda_scale_power=float(args.lambda_scale_power),
            decoherence_alpha=float(args.decoherence_alpha),
            memory_scales=[int(x) for x in args.memory_scales],
            gksl_dephase_base=float(d_base),
            gksl_dephase_comm_scale=float(d_comm),
            gksl_relax_base=float(r_base),
            gksl_relax_comm_scale=float(r_comm),
            gksl_measurement_rate=float(m_rate),
            gksl_hamiltonian_scale=float(h_scale),
            gksl_dt_cap_hours=float(args.gksl_dt_cap_hours),
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
                "gksl_dephase_base": float(d_base),
                "gksl_dephase_comm_scale": float(d_comm),
                "gksl_relax_base": float(r_base),
                "gksl_relax_comm_scale": float(r_comm),
                "gksl_measurement_rate": float(m_rate),
                "gksl_hamiltonian_scale": float(h_scale),
                **row,
            }
        )

    screen_df = pd.DataFrame(screen_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    screen_df.to_csv(outdir / "gksl_screening_l8.csv", index=False)

    print("[gksl] final l=8 permutation checks", flush=True)
    final_rows: list[dict[str, float | bool | str]] = []
    top = screen_df.head(max(1, int(args.top_k))).copy()
    n_top = len(top)
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        print(f"[gksl] final config {i}/{n_top}: {r['config_id']}", flush=True)
        df_cfg = _apply_gksl_cptp_bridge(
            base_df,
            lambda_weights=np.asarray(args.lambda_weights, dtype=float),
            lambda_scale_power=float(args.lambda_scale_power),
            decoherence_alpha=float(args.decoherence_alpha),
            memory_scales=[int(x) for x in args.memory_scales],
            gksl_dephase_base=float(r["gksl_dephase_base"]),
            gksl_dephase_comm_scale=float(r["gksl_dephase_comm_scale"]),
            gksl_relax_base=float(r["gksl_relax_base"]),
            gksl_relax_comm_scale=float(r["gksl_relax_comm_scale"]),
            gksl_measurement_rate=float(r["gksl_measurement_rate"]),
            gksl_hamiltonian_scale=float(r["gksl_hamiltonian_scale"]),
            gksl_dt_cap_hours=float(args.gksl_dt_cap_hours),
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
                "gksl_dephase_base": float(r["gksl_dephase_base"]),
                "gksl_dephase_comm_scale": float(r["gksl_dephase_comm_scale"]),
                "gksl_relax_base": float(r["gksl_relax_base"]),
                "gksl_relax_comm_scale": float(r["gksl_relax_comm_scale"]),
                "gksl_measurement_rate": float(r["gksl_measurement_rate"]),
                "gksl_hamiltonian_scale": float(r["gksl_hamiltonian_scale"]),
                **row,
            }
        )

    final_df = pd.DataFrame(final_rows).sort_values(
        ["PASS_ALL", "mae_gain", "r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    final_df.to_csv(outdir / "gksl_final_l8.csv", index=False)

    best = final_df.iloc[0]
    best_df = _apply_gksl_cptp_bridge(
        base_df,
        lambda_weights=np.asarray(args.lambda_weights, dtype=float),
        lambda_scale_power=float(args.lambda_scale_power),
        decoherence_alpha=float(args.decoherence_alpha),
        memory_scales=[int(x) for x in args.memory_scales],
        gksl_dephase_base=float(best["gksl_dephase_base"]),
        gksl_dephase_comm_scale=float(best["gksl_dephase_comm_scale"]),
        gksl_relax_base=float(best["gksl_relax_base"]),
        gksl_relax_comm_scale=float(best["gksl_relax_comm_scale"]),
        gksl_measurement_rate=float(best["gksl_measurement_rate"]),
        gksl_hamiltonian_scale=float(best["gksl_hamiltonian_scale"]),
        gksl_dt_cap_hours=float(args.gksl_dt_cap_hours),
    )
    best_df.to_csv(outdir / "gksl_tile_dataset_best.csv", index=False)

    print("[gksl] best-config all-scale evaluation", flush=True)
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
    gksl_l8 = final_df.iloc[0]
    gksl_all = summary_df[summary_df["scale_l"].astype(str) == "ALL"].iloc[0]

    compare_rows = [
        {
            "run": "dense_c009_baseline_l8",
            "mae_gain": float(baseline_l8["mae_gain"]),
            "r2_gain": float(baseline_l8["r2_gain"]),
            "perm_p_value": float(baseline_l8["perm_p_value"]),
            "event_positive_frac": float(baseline_l8["event_positive_frac"]),
            "PASS_ALL": bool(baseline_l8["PASS_ALL"]),
        },
        {
            "run": "p2_memory_gksl_best_l8",
            "mae_gain": float(gksl_l8["mae_gain"]),
            "r2_gain": float(gksl_l8["r2_gain"]),
            "perm_p_value": float(gksl_l8["perm_p_value"]),
            "event_positive_frac": float(gksl_l8["event_positive_frac"]),
            "PASS_ALL": bool(gksl_l8["PASS_ALL"]),
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
            "run": "p2_memory_gksl_best_all",
            "mae_gain": float(gksl_all["mae_gain"]),
            "r2_gain": float(gksl_all["r2_gain"]),
            "perm_p_value": float(gksl_all["perm_p_value"]),
            "event_positive_frac": float(gksl_all["event_positive_frac"]),
            "PASS_ALL": bool(gksl_all["PASS_ALL"]),
        },
    ]

    if args.surrogate_summary_csv.exists():
        sur_all = _read_all_row(args.surrogate_summary_csv)
        sur_l8 = _read_scale_row(args.surrogate_summary_csv, int(args.memory_scales[0]))
        compare_rows.extend(
            [
                {
                    "run": "p2_memory_surrogate_l8",
                    "mae_gain": float(sur_l8["mae_gain"]),
                    "r2_gain": float(sur_l8["r2_gain"]),
                    "perm_p_value": float(sur_l8["perm_p_value"]),
                    "event_positive_frac": float(sur_l8["event_positive_frac"]),
                    "PASS_ALL": bool(sur_l8["PASS_ALL"]),
                },
                {
                    "run": "p2_memory_surrogate_all",
                    "mae_gain": float(sur_all["mae_gain"]),
                    "r2_gain": float(sur_all["r2_gain"]),
                    "perm_p_value": float(sur_all["perm_p_value"]),
                    "event_positive_frac": float(sur_all["event_positive_frac"]),
                    "PASS_ALL": bool(sur_all["PASS_ALL"]),
                },
            ]
        )
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(outdir / "baseline_vs_memory_models.csv", index=False)

    cptp_rows = best_df[best_df["memory_applied"]].copy()
    max_cptp_violation = float(np.nanmax(cptp_rows["gksl_cptp_violation_raw"].to_numpy(dtype=float))) if len(cptp_rows) > 0 else 0.0
    mean_gamma_phi = float(np.nanmean(cptp_rows["gksl_gamma_dephase_raw"].to_numpy(dtype=float))) if len(cptp_rows) > 0 else 0.0
    mean_gamma_relax = float(np.nanmean(cptp_rows["gksl_gamma_relax_raw"].to_numpy(dtype=float))) if len(cptp_rows) > 0 else 0.0
    mean_kappa = float(np.nanmean(cptp_rows["gksl_reset_kappa_raw"].to_numpy(dtype=float))) if len(cptp_rows) > 0 else 0.0

    report_lines = [
        "# Experiment P2-memory GKSL/CPTP",
        "",
        "## Setup",
        f"- base tile csv: `{args.tile_csv}`",
        f"- panel fallback: `{args.panel_csv}`",
        f"- target: `{target_col}`",
        f"- memory scales: `{[int(x) for x in args.memory_scales]}`",
        "- state propagation: `rho_t = Reset o GAD o Dephase o U (rho_{t-1})`",
        f"- locked baseline: C009 weights={list(map(float, args.lambda_weights))}, "
        f"scale_power={float(args.lambda_scale_power)}, decoherence_alpha={float(args.decoherence_alpha)}",
        "",
        "## Best l=8 config",
        f"- config_id: `{best['config_id']}`",
        f"- gksl_dephase_base: `{float(best['gksl_dephase_base']):.3f}`",
        f"- gksl_dephase_comm_scale: `{float(best['gksl_dephase_comm_scale']):.3f}`",
        f"- gksl_relax_base: `{float(best['gksl_relax_base']):.3f}`",
        f"- gksl_relax_comm_scale: `{float(best['gksl_relax_comm_scale']):.3f}`",
        f"- gksl_measurement_rate: `{float(best['gksl_measurement_rate']):.3f}`",
        f"- gksl_hamiltonian_scale: `{float(best['gksl_hamiltonian_scale']):.3f}`",
        f"- l=8 mae_gain: `{float(best['mae_gain']):.6e}`",
        f"- l=8 r2_gain: `{float(best['r2_gain']):.6e}`",
        f"- l=8 perm_p: `{float(best['perm_p_value']):.6f}`",
        f"- l=8 event_positive_frac: `{float(best['event_positive_frac']):.4f}`",
        f"- l=8 PASS_ALL: `{bool(best['PASS_ALL'])}`",
        "",
        "## Best all-scale summary",
        f"- ALL mae_gain: `{float(gksl_all['mae_gain']):.6e}`",
        f"- ALL r2_gain: `{float(gksl_all['r2_gain']):.6e}`",
        f"- ALL perm_p(max scale): `{float(gksl_all['perm_p_value']):.6f}`",
        f"- ALL event_positive_frac: `{float(gksl_all['event_positive_frac']):.4f}`",
        f"- ALL PASS_ALL: `{bool(gksl_all['PASS_ALL'])}`",
        "",
        "## CPTP diagnostics",
        f"- max cptp violation proxy (trace+psd): `{max_cptp_violation:.3e}`",
        f"- mean gamma_dephase (applied rows): `{mean_gamma_phi:.4f}`",
        f"- mean gamma_relax (applied rows): `{mean_gamma_relax:.4f}`",
        f"- mean reset_kappa (applied rows): `{mean_kappa:.4f}`",
        "",
        "## Comparison",
        f"- baseline l=8 mae_gain: `{float(baseline_l8['mae_gain']):.6e}`; perm_p=`{float(baseline_l8['perm_p_value']):.6f}`",
        f"- gksl     l=8 mae_gain: `{float(gksl_l8['mae_gain']):.6e}`; perm_p=`{float(gksl_l8['perm_p_value']):.6f}`",
        f"- baseline ALL pass: `{bool(baseline_all['PASS_ALL'])}`; gksl ALL pass: `{bool(gksl_all['PASS_ALL'])}`",
        "",
        "## Artifacts",
        "- `gksl_screening_l8.csv`",
        "- `gksl_final_l8.csv`",
        "- `gksl_tile_dataset_best.csv`",
        "- `summary_metrics.csv`",
        "- `oof_predictions.csv`",
        "- `fold_metrics.csv`",
        "- `permutation_metrics.csv`",
        "- `baseline_vs_memory_models.csv`",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
