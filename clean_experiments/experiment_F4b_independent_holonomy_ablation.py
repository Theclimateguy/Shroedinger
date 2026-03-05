#!/usr/bin/env python3
"""Experiment F4b: independent holonomy-fractal test with mandatory ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import linregress, spearmanr

try:
    from clean_experiments.common import I2, SX, SY, SZ, comm, lindblad_dissipator, random_su2
except ImportError:
    from common import I2, SX, SY, SZ, comm, lindblad_dissipator, random_su2


SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SP = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)


@dataclass
class F4bConfig:
    eps_list: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 0.70, 0.85, 1.00, 1.15, 1.40, 3.00, 10.00)
    n_mu: int = 192
    n_seeds: int = 50
    n_loops: int = 50
    d_top: float = 2.0
    min_loop_gap: int = 7
    epsilon_star_window: tuple[float, float] = (0.7, 1.4)
    corr_threshold: float = 0.9
    permutation_iters: int = 5000
    bootstrap_iters: int = 1200
    gauge_tests: int = 24
    gauge_tol: float = 1e-12
    df_agreement_tol: float = 0.35
    seed_df_agreement_ratio_min: float = 0.35
    seed_psd_r2_median_min: float = 0.35
    max_failed_seed_share: float = 0.35
    ablation_const_var_tol: float = 1e-20
    ablation_const_amp_tol: float = 1e-8
    # Independent open-system dynamics (not inherited from F3 kernels).
    h_geom_coeff: float = 0.65
    h_x_coeff: float = 0.18
    h_z_coeff: float = 0.22
    rel_rate_base: float = 0.45
    drv_rate_base: float = 0.45
    rate_power: float = 0.85
    dephase_base: float = 0.02
    dephase_balance_coeff: float = 0.28
    noise_base: float = 0.55
    # Edge coarse-graining factors, driven only by GKSL activity.
    eta_coeff: float = 0.90
    comm_coeff: float = 4.20
    twist_coeff: float = 2.40
    # D_f estimator normalization constants (fixed a priori for this grid).
    psd_lift: float = 1.0
    box_ref_dim: float = 0.87
    box_gain: float = 4.0
    # Ablation C geometry sweep.
    c2_geometry_variants: int = 6
    c2_epsilon_fixed: float = 1.0


def parse_eps_list(raw: str) -> tuple[float, ...]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty epsilon list")
    return tuple(vals)


def stabilize_density(rho: np.ndarray) -> np.ndarray:
    herm = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(herm)
    vals = np.clip(vals, 0.0, None)
    if float(np.sum(vals)) <= 1e-15:
        vals[0] = 1.0
    proj = vecs @ np.diag(vals) @ vecs.conj().T
    return proj / np.trace(proj)


def build_base_connection(mu: np.ndarray, variant_id: int = 0) -> list[np.ndarray]:
    phase = 0.35 * float(variant_id)
    amp = 1.0 + 0.06 * float(variant_id - 2)
    out = []
    for m in mu:
        a = amp * (0.72 * np.sin(2.0 * np.pi * m + phase) + 0.28 * np.sin(6.0 * np.pi * m - 0.4))
        b = amp * (0.66 * np.cos(3.0 * np.pi * m - 0.3 * phase) + 0.22 * np.sin(5.0 * np.pi * m + 0.2))
        c = 0.22 * np.sin(4.0 * np.pi * m + 0.5 * phase)
        out.append(a * SX + b * SY + c * SZ)
    return out


def build_commutative_connection(mu: np.ndarray) -> list[np.ndarray]:
    out = []
    for m in mu:
        scalar = 0.95 + 0.25 * np.sin(4.0 * np.pi * m)
        out.append(scalar * SZ)
    return out


def balance_factor(epsilon: float, cfg: F4bConfig) -> tuple[float, float, float]:
    gamma_rel = float(cfg.rel_rate_base / (epsilon**cfg.rate_power))
    gamma_drv = float(cfg.drv_rate_base * (epsilon**cfg.rate_power))
    bal = float(2.0 * np.sqrt(gamma_rel * gamma_drv) / (gamma_rel + gamma_drv + 1e-15))
    return gamma_rel, gamma_drv, bal


def simulate_trajectory(
    mu: np.ndarray,
    a_grid: list[np.ndarray],
    epsilon: float,
    seed: int,
    cfg: F4bConfig,
    mode: str = "open",
    permute_mu: bool = False,
) -> dict[str, np.ndarray | float]:
    if mode not in {"open", "unitary"}:
        raise ValueError(f"Unknown mode: {mode}")

    rng = np.random.default_rng(seed)
    n_mu = len(mu)
    dmu = float(mu[1] - mu[0])
    order = np.arange(n_mu)
    if permute_mu:
        order = rng.permutation(n_mu)

    vec = rng.normal(size=2) + 1j * rng.normal(size=2)
    vec = vec / (np.linalg.norm(vec) + 1e-15)
    rho = np.outer(vec, vec.conj())

    gamma_rel, gamma_drv, bal = balance_factor(epsilon=epsilon, cfg=cfg)
    if mode == "unitary":
        gamma_rel = 0.0
        gamma_drv = 0.0
        bal = 0.0
        gamma_phi = 0.0
        noise_amp = 0.0
    else:
        gamma_phi = float(cfg.dephase_base + cfg.dephase_balance_coeff * (bal**2))
        noise_amp = float(cfg.noise_base * (bal**4))

    purity = [float(np.real(np.trace(rho @ rho)))]
    coherence = [float(abs(rho[0, 1]))]

    for t in range(n_mu - 1):
        idx = int(order[t])
        m = float(mu[idx])
        a_mu = a_grid[idx]

        h_det = (
            cfg.h_geom_coeff * a_mu
            + cfg.h_x_coeff * np.cos(3.0 * np.pi * m + 0.2) * SX
            + cfg.h_z_coeff * np.sin(2.0 * np.pi * m - 0.15) * SZ
        )
        if mode == "open":
            h_det = h_det + noise_amp * rng.normal() * (0.62 * SX + 0.27 * SY + 0.11 * SZ)

        u = expm(-1j * h_det * dmu)
        rho = u @ rho @ u.conj().T

        if mode == "open":
            mod = 0.75 + 0.25 * np.cos(2.0 * np.pi * m + 0.17 * float(seed))
            g_rel = max(gamma_rel * mod, 0.0)
            g_drv = max(gamma_drv * (2.0 - mod), 0.0)
            g_phi = max(gamma_phi * (0.4 + 0.6 * np.sin(4.0 * np.pi * m) ** 2), 0.0)
            jumps = []
            if g_rel > 0.0:
                jumps.append(np.sqrt(g_rel) * SM)
            if g_drv > 0.0:
                jumps.append(np.sqrt(g_drv) * SP)
            if g_phi > 0.0:
                jumps.append(np.sqrt(g_phi) * SZ)
            if jumps:
                rho = rho + dmu * lindblad_dissipator(rho, jumps)

        rho = stabilize_density(rho)
        purity.append(float(np.real(np.trace(rho @ rho))))
        coherence.append(float(abs(rho[0, 1])))

    purity_arr = np.asarray(purity, dtype=float)
    coh_arr = np.asarray(coherence, dtype=float)
    dp = np.diff(purity_arr, prepend=purity_arr[0])
    dc = np.diff(coh_arr, prepend=coh_arr[0])
    activity = np.abs(dp) + np.abs(dc)
    activity = (activity - np.min(activity)) / (np.ptp(activity) + 1e-15)

    return {
        "purity": purity_arr,
        "coherence": coh_arr,
        "activity": activity,
        "balance": float(bal),
        "gamma_rel": float(gamma_rel),
        "gamma_drv": float(gamma_drv),
    }


def beta_from_psd(signal: np.ndarray) -> tuple[float, float]:
    s = np.asarray(signal, dtype=float)
    if float(np.std(s)) < 1e-10:
        return 1.0, 1.0
    centered = s - np.mean(s)
    n = len(centered)
    power = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = (
        (freqs > 0.0)
        & (np.arange(len(freqs)) >= 2)
        & (np.arange(len(freqs)) <= max(4, n // 4))
        & (power > 1e-20)
    )
    if int(np.sum(mask)) < 8:
        return 1.0, 0.0
    x = np.log(freqs[mask])
    y = np.log(power[mask])
    slope, intercept, _, _, _ = linregress(x, y)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-15))
    beta = float(-slope)
    return beta, r2


def box_dimension_trajectory(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = min(len(x_arr), len(y_arr))
    if n < 16:
        return 1.0, 1.0
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    if float(np.ptp(x_arr)) < 1e-10 or float(np.ptp(y_arr)) < 1e-10:
        return 1.0, 1.0

    xn = (x_arr - np.min(x_arr)) / (np.ptp(x_arr) + 1e-15)
    yn = (y_arr - np.min(y_arr)) / (np.ptp(y_arr) + 1e-15)

    bins_list = [4, 8, 12, 16, 24, 32, 48, 64]
    bins_list = [b for b in bins_list if b <= max(8, n // 2)]
    logs_n = []
    logs_boxes = []
    for nb in bins_list:
        ix = np.minimum((xn * nb).astype(int), nb - 1)
        iy = np.minimum((yn * nb).astype(int), nb - 1)
        boxes = len(set(zip(ix.tolist(), iy.tolist())))
        if boxes <= 0:
            continue
        logs_n.append(np.log(float(nb)))
        logs_boxes.append(np.log(float(boxes)))

    if len(logs_n) < 3:
        return 1.0, 0.0

    slope, intercept, _, _, _ = linregress(np.asarray(logs_n), np.asarray(logs_boxes))
    y_hat = slope * np.asarray(logs_n) + intercept
    ss_res = float(np.sum((np.asarray(logs_boxes) - y_hat) ** 2))
    ss_tot = float(np.sum((np.asarray(logs_boxes) - np.mean(logs_boxes)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-15))
    return float(slope), r2


def box_dimension_graph(signal: np.ndarray) -> tuple[float, float]:
    s = np.asarray(signal, dtype=float)
    n = len(s)
    if n < 16 or float(np.ptp(s)) < 1e-12:
        return 1.0, 1.0
    x = np.linspace(0.0, 1.0, n)
    y = (s - np.min(s)) / (np.ptp(s) + 1e-15)
    bins_list = [4, 8, 12, 16, 24, 32, 48, 64]
    bins_list = [b for b in bins_list if b <= max(8, n // 2)]
    logs_n = []
    logs_boxes = []
    for nb in bins_list:
        ix = np.minimum((x * nb).astype(int), nb - 1)
        iy = np.minimum((y * nb).astype(int), nb - 1)
        boxes = len(set(zip(ix.tolist(), iy.tolist())))
        if boxes <= 0:
            continue
        logs_n.append(np.log(float(nb)))
        logs_boxes.append(np.log(float(boxes)))
    if len(logs_n) < 3:
        return 1.0, 0.0
    slope, intercept, _, _, _ = linregress(np.asarray(logs_n), np.asarray(logs_boxes))
    y_hat = slope * np.asarray(logs_n) + intercept
    ss_res = float(np.sum((np.asarray(logs_boxes) - y_hat) ** 2))
    ss_tot = float(np.sum((np.asarray(logs_boxes) - np.mean(logs_boxes)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-15))
    return float(slope), r2


def estimate_df(purity: np.ndarray, coherence: np.ndarray, cfg: F4bConfig) -> dict[str, float]:
    pur = np.asarray(purity, dtype=float)
    coh = np.asarray(coherence, dtype=float)
    d1 = np.diff(coh, prepend=coh[0])
    signal_psd = np.abs(d1)
    beta, psd_r2 = beta_from_psd(signal=signal_psd)
    # Lift the 1D spectral estimate onto the 2D topological manifold.
    d_psd = float(cfg.psd_lift + (5.0 - beta) / 2.0)
    d_psd = float(np.clip(d_psd, cfg.d_top, cfg.d_top + 1.2))

    activity = np.abs(np.diff(pur, prepend=pur[0])) + np.abs(d1)
    d_curve, box_r2 = box_dimension_graph(signal=activity)
    d_box = float(cfg.d_top + np.clip(cfg.box_gain * (d_curve - cfg.box_ref_dim), 0.0, 1.2))

    d_mean = 0.5 * (d_psd + d_box)
    return {
        "beta_psd": float(beta),
        "psd_fit_r2": float(psd_r2),
        "D_f_psd": float(d_psd),
        "D_f_box": float(d_box),
        "box_fit_r2": float(box_r2),
        "D_f": float(d_mean),
        "D_f_minus_d_top": float(d_mean - cfg.d_top),
        "df_estimators_abs_diff": float(abs(d_psd - d_box)),
    }


def sample_loops(
    n_mu: int,
    n_loops: int,
    min_gap: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None) + 1e-6
    p = w / np.sum(w)
    loops: list[tuple[int, int, int]] = []
    used: set[tuple[int, int, int]] = set()
    tries = 0
    while len(loops) < n_loops and tries < 200000:
        tries += 1
        tri = np.sort(rng.choice(n_mu, size=3, replace=False, p=p))
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        if (j - i) < min_gap or (k - j) < min_gap:
            continue
        key = (i, j, k)
        if key in used:
            continue
        loops.append(key)
        used.add(key)

    # Guaranteed fill, if weighted sampling was too restrictive.
    while len(loops) < n_loops:
        tri = np.sort(rng.choice(n_mu, size=3, replace=False))
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        if (j - i) < min_gap or (k - j) < min_gap:
            continue
        key = (i, j, k)
        if key in used:
            continue
        loops.append(key)
        used.add(key)
    return loops


def edge_operator(
    a_grid: list[np.ndarray],
    activity: np.ndarray,
    balance: float,
    i: int,
    j: int,
    dmu: float,
    cfg: F4bConfig,
) -> np.ndarray:
    if i == j:
        return I2.copy()

    step = 1 if j > i else -1
    u_path = I2.copy()
    comm_int = np.zeros((2, 2), dtype=complex)
    acts = []
    for k in range(i, j, step):
        k_next = k + step
        lo, hi = sorted((k, k_next))
        act_k = 0.5 * float(activity[lo] + activity[hi])
        eta_k = float(1.0 + cfg.eta_coeff * (balance**3) * act_k)
        dt = float(step * dmu)
        u_path = expm(-1j * eta_k * a_grid[k] * dt) @ u_path
        comm_int = comm_int + float(step * dmu) * comm(a_grid[lo], a_grid[hi])
        acts.append(act_k)

    seg_act = float(np.mean(acts)) if acts else 0.0
    seg = float(abs(j - i) * dmu)
    comm_norm = float(np.linalg.norm(comm_int, ord="fro"))
    b_eff = float(balance**3)
    ai = a_grid[i]
    aj = a_grid[j]
    defect = cfg.comm_coeff * b_eff * seg_act * comm_int
    # Twist is allowed only when non-commutativity is present.
    twist_scale = cfg.twist_coeff * b_eff * seg_act * comm_norm * seg
    defect = defect + 1j * twist_scale * (ai - aj)
    return expm(defect) @ u_path


def loop_metrics(
    a_grid: list[np.ndarray],
    activity: np.ndarray,
    balance: float,
    loop: tuple[int, int, int],
    dmu: float,
    cfg: F4bConfig,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    i, j, k = loop
    edges = {
        (0, 1): edge_operator(a_grid, activity, balance, i, j, dmu, cfg),
        (1, 2): edge_operator(a_grid, activity, balance, j, k, dmu, cfg),
        (2, 0): edge_operator(a_grid, activity, balance, k, i, dmu, cfg),
        (0, 2): edge_operator(a_grid, activity, balance, i, k, dmu, cfg),
        (2, 1): edge_operator(a_grid, activity, balance, k, j, dmu, cfg),
        (1, 0): edge_operator(a_grid, activity, balance, j, i, dmu, cfg),
    }

    h_triangle_mat = edges[(2, 0)] @ edges[(1, 2)] @ edges[(0, 1)]
    h_placebo_mat = edges[(1, 0)] @ edges[(2, 1)] @ edges[(0, 2)]
    h_minimal_mat = edges[(1, 0)] @ edges[(0, 1)]

    metrics = {
        "h_triangle": float(np.linalg.norm(h_triangle_mat - I2, ord="fro")),
        "h_placebo": float(np.linalg.norm(h_placebo_mat - I2, ord="fro")),
        "h_minimal": float(np.linalg.norm(h_minimal_mat - I2, ord="fro")),
        "delta_order": float(np.linalg.norm(h_triangle_mat - h_placebo_mat, ord="fro")),
    }
    mats = {
        "triangle": h_triangle_mat,
        "placebo": h_placebo_mat,
        "minimal": h_minimal_mat,
    }
    return edges, metrics, mats


def run_regime(
    regime: str,
    eps_list: tuple[float, ...],
    mu: np.ndarray,
    a_grid: list[np.ndarray],
    cfg: F4bConfig,
    mode: str = "open",
    permute_mu: bool = False,
    compute_loops: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_mu = len(mu)
    dmu = float(mu[1] - mu[0])
    seed_rows: list[dict[str, float | int | str]] = []
    loop_rows: list[dict[str, float | int | str]] = []

    for eps in eps_list:
        epsilon = float(eps)
        for seed in range(cfg.n_seeds):
            sim = simulate_trajectory(
                mu=mu,
                a_grid=a_grid,
                epsilon=epsilon,
                seed=seed,
                cfg=cfg,
                mode=mode,
                permute_mu=permute_mu,
            )
            df_est = estimate_df(
                purity=np.asarray(sim["purity"], dtype=float),
                coherence=np.asarray(sim["coherence"], dtype=float),
                cfg=cfg,
            )
            seed_row = {
                "regime": regime,
                "epsilon": epsilon,
                "seed": int(seed),
                "balance": float(sim["balance"]),
                "gamma_rel": float(sim["gamma_rel"]),
                "gamma_drv": float(sim["gamma_drv"]),
                **df_est,
            }
            seed_rows.append(seed_row)

            if not compute_loops:
                continue

            rng_loop = np.random.default_rng(100000 * seed + int(round(1000 * epsilon)) + 17)
            loops = sample_loops(
                n_mu=n_mu,
                n_loops=cfg.n_loops,
                min_gap=cfg.min_loop_gap,
                weights=np.asarray(sim["activity"], dtype=float),
                rng=rng_loop,
            )
            for loop_id, tri in enumerate(loops):
                _, hol_metrics, _ = loop_metrics(
                    a_grid=a_grid,
                    activity=np.asarray(sim["activity"], dtype=float),
                    balance=float(sim["balance"]),
                    loop=tri,
                    dmu=dmu,
                    cfg=cfg,
                )
                loop_rows.append(
                    {
                        "regime": regime,
                        "epsilon": epsilon,
                        "seed": int(seed),
                        "loop_id": int(loop_id),
                        "mu_i": int(tri[0]),
                        "mu_j": int(tri[1]),
                        "mu_k": int(tri[2]),
                        "D_f": float(df_est["D_f"]),
                        "D_f_minus_d_top": float(df_est["D_f_minus_d_top"]),
                        **hol_metrics,
                    }
                )

    return pd.DataFrame(seed_rows), pd.DataFrame(loop_rows)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return float("nan")
    if float(np.std(x_arr)) < 1e-14 or float(np.std(y_arr)) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def permutation_pvalue(x: np.ndarray, y: np.ndarray, n_iter: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = abs(pearson_corr(x, y))
    if not np.isfinite(obs):
        return float("nan")
    cnt = 0
    for _ in range(n_iter):
        yp = rng.permutation(y)
        val = abs(pearson_corr(x, yp))
        if val >= obs:
            cnt += 1
    return float((cnt + 1) / (n_iter + 1))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return float("nan")
    if float(np.std(x_arr)) < 1e-14 or float(np.std(y_arr)) < 1e-14:
        return float("nan")
    return float(spearmanr(x_arr, y_arr).statistic)


def finite_mean(arr: pd.Series | np.ndarray) -> float:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def finite_quantile(arr: pd.Series | np.ndarray, q: float) -> float:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def partial_corr_linear(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    mask = (
        np.isfinite(x_arr)
        & np.isfinite(y_arr)
        & np.isfinite(z_arr)
        & (np.abs(x_arr) < 1e6)
        & (np.abs(y_arr) < 1e6)
        & (np.abs(z_arr) < 1e6)
    )
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    z_arr = z_arr[mask]
    if len(x_arr) < 4:
        return float("nan")
    if float(np.std(z_arr)) < 1e-14:
        return pearson_corr(x_arr, y_arr)
    z_scaled = (z_arr - np.mean(z_arr)) / (np.std(z_arr) + 1e-15)
    design = np.column_stack([z_scaled, np.ones_like(z_scaled)])
    cond = float(np.linalg.cond(design))
    if (not np.isfinite(cond)) or cond > 1e8:
        return pearson_corr(x_arr, y_arr)
    bx = np.linalg.lstsq(design, x_arr, rcond=None)[0]
    by = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    if (not np.all(np.isfinite(bx))) or (not np.all(np.isfinite(by))):
        return pearson_corr(x_arr, y_arr)
    rx = x_arr - design @ bx
    ry = y_arr - design @ by
    return pearson_corr(rx, ry)


def bootstrap_eps_star(
    delta_tensor: np.ndarray,
    eps_arr: np.ndarray,
    n_iter: int,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_eps, n_seeds, n_loops = delta_tensor.shape
    rows = []
    for b in range(n_iter):
        s_idx = rng.integers(0, n_seeds, size=n_seeds)
        l_idx = rng.integers(0, n_loops, size=n_loops)
        boot = np.take(np.take(delta_tensor, s_idx, axis=1), l_idx, axis=2)
        curve = np.mean(boot, axis=(1, 2))
        i_star = int(np.argmax(curve))
        rows.append(
            {
                "bootstrap_id": int(b),
                "epsilon_star": float(eps_arr[i_star]),
                "delta_order_star": float(curve[i_star]),
            }
        )
    return pd.DataFrame(rows)


def build_delta_tensor(loop_df: pd.DataFrame, eps_list: tuple[float, ...], cfg: F4bConfig) -> np.ndarray:
    eps_to_i = {float(e): i for i, e in enumerate(eps_list)}
    arr = np.full((len(eps_list), cfg.n_seeds, cfg.n_loops), np.nan, dtype=float)
    for row in loop_df.itertuples(index=False):
        i_eps = eps_to_i[float(row.epsilon)]
        arr[i_eps, int(row.seed), int(row.loop_id)] = float(row.delta_order)
    if np.isnan(arr).any():
        raise ValueError("Delta tensor has NaNs; check loop sampling coverage.")
    return arr


def gauge_checks_from_main(
    mu: np.ndarray,
    a_grid: list[np.ndarray],
    main_seed_df: pd.DataFrame,
    main_loop_df: pd.DataFrame,
    cfg: F4bConfig,
) -> pd.DataFrame:
    dmu = float(mu[1] - mu[0])
    rows = []
    rng = np.random.default_rng(2026)

    # Use seed=0 to keep gauge-check volume controlled but still across all eps.
    seed_id = 0
    for eps in sorted(main_seed_df["epsilon"].unique()):
        seed_row = main_seed_df[(main_seed_df["epsilon"] == eps) & (main_seed_df["seed"] == seed_id)].iloc[0]
        balance = float(seed_row["balance"])
        loop_subset = main_loop_df[(main_loop_df["epsilon"] == eps) & (main_loop_df["seed"] == seed_id)].head(12)

        sim = simulate_trajectory(
            mu=mu,
            a_grid=a_grid,
            epsilon=float(eps),
            seed=seed_id,
            cfg=cfg,
            mode="open",
            permute_mu=False,
        )
        activity = np.asarray(sim["activity"], dtype=float)

        for loop_row in loop_subset.itertuples(index=False):
            tri = (int(loop_row.mu_i), int(loop_row.mu_j), int(loop_row.mu_k))
            edges, hol_metrics, mats = loop_metrics(
                a_grid=a_grid,
                activity=activity,
                balance=balance,
                loop=tri,
                dmu=dmu,
                cfg=cfg,
            )
            for gauge_id in range(cfg.gauge_tests):
                g = (random_su2(rng), random_su2(rng), random_su2(rng))
                e_g = {}
                for (i, j), e_ij in edges.items():
                    e_g[(i, j)] = g[j] @ e_ij @ g[i].conj().T

                h_tri_g = e_g[(2, 0)] @ e_g[(1, 2)] @ e_g[(0, 1)]
                h_pl_g = e_g[(1, 0)] @ e_g[(2, 1)] @ e_g[(0, 2)]

                h_tri_base = float(np.linalg.norm(mats["triangle"] - I2, ord="fro"))
                h_tri_new = float(np.linalg.norm(h_tri_g - I2, ord="fro"))
                d_order_base = float(np.linalg.norm(mats["triangle"] - mats["placebo"], ord="fro"))
                d_order_new = float(np.linalg.norm(h_tri_g - h_pl_g, ord="fro"))

                rows.append(
                    {
                        "epsilon": float(eps),
                        "seed": int(seed_id),
                        "loop_id": int(loop_row.loop_id),
                        "gauge_id": int(gauge_id),
                        "abs_delta_h_triangle": float(abs(h_tri_new - h_tri_base)),
                        "abs_delta_delta_order": float(abs(d_order_new - d_order_base)),
                    }
                )
    return pd.DataFrame(rows)


def run_f4b(outdir: Path, cfg: F4bConfig) -> dict[str, object]:
    mu = np.linspace(0.0, 1.0, cfg.n_mu)
    a_base = build_base_connection(mu=mu)
    a_comm = build_commutative_connection(mu=mu)

    # Main regime: fixed geometry, varying epsilon only in open dynamics.
    main_seed_df, main_loop_df = run_regime(
        regime="main",
        eps_list=cfg.eps_list,
        mu=mu,
        a_grid=a_base,
        cfg=cfg,
        mode="open",
        permute_mu=False,
        compute_loops=True,
    )

    # Ablation A: no openness (unitary).
    abl_a_seed_df, abl_a_loop_df = run_regime(
        regime="ablation_A_unitary",
        eps_list=cfg.eps_list,
        mu=mu,
        a_grid=a_base,
        cfg=cfg,
        mode="unitary",
        permute_mu=False,
        compute_loops=True,
    )

    # Ablation B: commutative geometry with strong openness.
    abl_b_seed_df, abl_b_loop_df = run_regime(
        regime="ablation_B_commutative_geometry",
        eps_list=cfg.eps_list,
        mu=mu,
        a_grid=a_comm,
        cfg=cfg,
        mode="open",
        permute_mu=False,
        compute_loops=True,
    )

    # Ablation D: random mu permutation in GKSL trajectory (destroys scale causality).
    abl_d_seed_df, _ = run_regime(
        regime="ablation_D_mu_permutation",
        eps_list=cfg.eps_list,
        mu=mu,
        a_grid=a_base,
        cfg=cfg,
        mode="open",
        permute_mu=True,
        compute_loops=False,
    )

    # Ablation C2: fixed epsilon, varying geometry only.
    c2_rows = []
    for variant in range(cfg.c2_geometry_variants):
        a_var = build_base_connection(mu=mu, variant_id=variant)
        c2_seed_df, c2_loop_df = run_regime(
            regime=f"ablation_C2_geometry_variant_{variant}",
            eps_list=(cfg.c2_epsilon_fixed,),
            mu=mu,
            a_grid=a_var,
            cfg=cfg,
            mode="open",
            permute_mu=False,
            compute_loops=True,
        )
        c2_rows.append(
            {
                "geometry_variant": int(variant),
                "epsilon": float(cfg.c2_epsilon_fixed),
                "delta_order_mean": float(c2_loop_df["delta_order"].mean()),
                "delta_order_std": float(c2_loop_df["delta_order"].std(ddof=0)),
                "D_f_mean": float(c2_seed_df["D_f"].mean()),
                "D_f_std": float(c2_seed_df["D_f"].std(ddof=0)),
            }
        )
    c2_df = pd.DataFrame(c2_rows)

    # Epsilon-level summaries.
    main_eps_df = (
        main_loop_df.groupby("epsilon", as_index=False)
        .agg(
            delta_order_mean=("delta_order", "mean"),
            delta_order_std=("delta_order", "std"),
            h_triangle_mean=("h_triangle", "mean"),
            h_minimal_max=("h_minimal", "max"),
        )
        .merge(
            main_seed_df.groupby("epsilon", as_index=False).agg(
                D_f_mean=("D_f", "mean"),
                D_f_psd_mean=("D_f_psd", "mean"),
                D_f_box_mean=("D_f_box", "mean"),
                D_f_excess_mean=("D_f_minus_d_top", "mean"),
                df_estimators_abs_diff_mean=("df_estimators_abs_diff", "mean"),
                balance_mean=("balance", "mean"),
            ),
            on="epsilon",
            how="left",
        )
        .sort_values("epsilon")
        .reset_index(drop=True)
    )

    # Main-correlation metrics and tests.
    x = main_eps_df["delta_order_mean"].to_numpy(float)
    y = main_eps_df["D_f_excess_mean"].to_numpy(float)
    corr_main_pearson = pearson_corr(x, y)
    corr_main_spearman = spearman_corr(x, y)
    p_perm = permutation_pvalue(x=x, y=y, n_iter=cfg.permutation_iters, seed=7)
    eps_star = float(main_eps_df.iloc[int(np.argmax(x))]["epsilon"])

    # Seed diagnostics for D_f estimator stability.
    seed_diag_df = (
        main_seed_df.groupby("seed", as_index=False)
        .agg(
            n_eps=("epsilon", "count"),
            df_agreement_rate=("df_estimators_abs_diff", lambda s: float(np.mean(np.asarray(s, float) <= cfg.df_agreement_tol))),
            df_diff_mean=("df_estimators_abs_diff", "mean"),
            df_diff_max=("df_estimators_abs_diff", "max"),
            psd_r2_median=("psd_fit_r2", "median"),
            psd_r2_min=("psd_fit_r2", "min"),
            box_r2_median=("box_fit_r2", "median"),
            box_r2_min=("box_fit_r2", "min"),
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )
    seed_diag_df["failed_df_agreement"] = seed_diag_df["df_agreement_rate"] < cfg.seed_df_agreement_ratio_min
    seed_diag_df["failed_psd_quality"] = seed_diag_df["psd_r2_median"] < cfg.seed_psd_r2_median_min
    seed_diag_df["failed_any"] = seed_diag_df["failed_df_agreement"] | seed_diag_df["failed_psd_quality"]
    failed_seed_share = float(np.mean(seed_diag_df["failed_any"]))

    # Distribution of correlations across seeds.
    seed_corr_rows = []
    for seed in range(cfg.n_seeds):
        x_seed = (
            main_loop_df[main_loop_df["seed"] == seed]
            .groupby("epsilon", as_index=False)["delta_order"]
            .mean()
            .sort_values("epsilon")
        )
        y_seed = (
            main_seed_df[main_seed_df["seed"] == seed]
            .sort_values("epsilon")[["epsilon", "D_f_minus_d_top"]]
            .reset_index(drop=True)
        )
        xy = x_seed.merge(y_seed, on="epsilon", how="inner")
        if len(xy) < 4:
            continue
        seed_corr_rows.append(
            {
                "seed": int(seed),
                "pearson": pearson_corr(xy["delta_order"].to_numpy(float), xy["D_f_minus_d_top"].to_numpy(float)),
                "spearman": spearman_corr(xy["delta_order"], xy["D_f_minus_d_top"]),
            }
        )
    seed_corr_df = pd.DataFrame(seed_corr_rows, columns=["seed", "pearson", "spearman"]).merge(
        seed_diag_df, on="seed", how="left"
    )
    seed_corr_valid_df = seed_corr_df[~seed_corr_df["failed_any"]].copy()

    # Distribution of correlations across loop IDs.
    loop_corr_rows = []
    d_mean_by_eps = (
        main_seed_df.groupby("epsilon", as_index=False)["D_f_minus_d_top"]
        .mean()
        .sort_values("epsilon")
        .reset_index(drop=True)
    )
    for loop_id in range(cfg.n_loops):
        x_loop = (
            main_loop_df[main_loop_df["loop_id"] == loop_id]
            .groupby("epsilon", as_index=False)["delta_order"]
            .mean()
            .sort_values("epsilon")
            .reset_index(drop=True)
        )
        xy = x_loop.merge(d_mean_by_eps, on="epsilon", how="inner")
        if len(xy) < 4:
            continue
        loop_corr_rows.append(
            {
                "loop_id": int(loop_id),
                "pearson": pearson_corr(xy["delta_order"].to_numpy(float), xy["D_f_minus_d_top"].to_numpy(float)),
                "spearman": spearman_corr(xy["delta_order"], xy["D_f_minus_d_top"]),
            }
        )
    loop_corr_df = pd.DataFrame(loop_corr_rows)

    # Partial correlation corr(delta_order, Df_excess | epsilon) on seed-level points.
    seed_delta_mean = (
        main_loop_df.groupby(["epsilon", "seed"], as_index=False)["delta_order"]
        .mean()
        .rename(columns={"delta_order": "delta_order_seed_mean"})
    )
    seed_join = seed_delta_mean.merge(
        main_seed_df[["epsilon", "seed", "D_f_minus_d_top"]],
        on=["epsilon", "seed"],
        how="inner",
    )
    partial_corr = partial_corr_linear(
        x=seed_join["delta_order_seed_mean"].to_numpy(float),
        y=seed_join["D_f_minus_d_top"].to_numpy(float),
        z=np.log(seed_join["epsilon"].to_numpy(float)),
    )

    # Bootstrap epsilon* over seeds and loops.
    delta_tensor = build_delta_tensor(loop_df=main_loop_df, eps_list=cfg.eps_list, cfg=cfg)
    boot_df = bootstrap_eps_star(
        delta_tensor=delta_tensor,
        eps_arr=np.asarray(cfg.eps_list, dtype=float),
        n_iter=cfg.bootstrap_iters,
        seed=11,
    )
    eps_star_ci = (
        float(boot_df["epsilon_star"].quantile(0.025)),
        float(boot_df["epsilon_star"].quantile(0.975)),
    )
    eps_star_in_window_share = float(
        np.mean(
            (boot_df["epsilon_star"].to_numpy(float) >= cfg.epsilon_star_window[0])
            & (boot_df["epsilon_star"].to_numpy(float) <= cfg.epsilon_star_window[1])
        )
    )

    # Ablation summaries.
    def regime_corr_details(loop_df: pd.DataFrame, seed_df: pd.DataFrame) -> tuple[float, bool, pd.DataFrame]:
        agg = (
            loop_df.groupby("epsilon", as_index=False)["delta_order"]
            .mean()
            .merge(
                seed_df.groupby("epsilon", as_index=False)["D_f_minus_d_top"].mean(),
                on="epsilon",
                how="inner",
            )
            .sort_values("epsilon")
            .reset_index(drop=True)
        )
        if len(agg) < 4:
            return float("nan"), False, agg
        x_reg = agg["delta_order"].to_numpy(float)
        y_reg = agg["D_f_minus_d_top"].to_numpy(float)
        applicable = bool(np.std(x_reg) > 1e-14 and np.std(y_reg) > 1e-14)
        return pearson_corr(x_reg, y_reg), applicable, agg

    corr_abl_a, corr_abl_a_applicable, abl_a_eps_agg = regime_corr_details(abl_a_loop_df, abl_a_seed_df)
    corr_abl_b, corr_abl_b_applicable, abl_b_eps_agg = regime_corr_details(abl_b_loop_df, abl_b_seed_df)

    ablation_a_delta_var = float(abl_a_loop_df["delta_order"].var(ddof=0))
    ablation_a_delta_max = float(abl_a_loop_df["delta_order"].max())
    ablation_b_delta_var = float(abl_b_loop_df["delta_order"].var(ddof=0))
    ablation_b_delta_max = float(abl_b_loop_df["delta_order"].max())

    # For ablation D, use main holonomy (geometry object) vs permuted Df.
    d_perm_join = (
        seed_delta_mean.merge(
            abl_d_seed_df[["epsilon", "seed", "D_f_minus_d_top"]].rename(columns={"D_f_minus_d_top": "D_f_perm_excess"}),
            on=["epsilon", "seed"],
            how="inner",
        )
        .sort_values(["epsilon", "seed"])
        .reset_index(drop=True)
    )
    corr_abl_d = pearson_corr(d_perm_join["delta_order_seed_mean"].to_numpy(float), d_perm_join["D_f_perm_excess"].to_numpy(float))

    # Gauge robustness.
    gauge_df = gauge_checks_from_main(
        mu=mu,
        a_grid=a_base,
        main_seed_df=main_seed_df,
        main_loop_df=main_loop_df,
        cfg=cfg,
    )
    max_delta_h = float(gauge_df["abs_delta_h_triangle"].max())
    max_delta_order = float(gauge_df["abs_delta_delta_order"].max())

    # Checks.
    checks = {
        "H4b_1_corr_gt_0_9": bool(corr_main_pearson > cfg.corr_threshold),
        "H4b_1_perm_p_lt_0_05": bool(p_perm < 0.05),
        "H4b_2_eps_star_in_window": bool(cfg.epsilon_star_window[0] <= eps_star <= cfg.epsilon_star_window[1]),
        "H4b_2_bootstrap_majority_in_window": bool(eps_star_in_window_share > 0.5),
        "Ablation_A_constant_delta_order": bool(
            ablation_a_delta_var < cfg.ablation_const_var_tol and ablation_a_delta_max < cfg.ablation_const_amp_tol
        ),
        "H4b_3_commutative_delta_order_near_zero": bool(
            ablation_b_delta_var < cfg.ablation_const_var_tol and ablation_b_delta_max < cfg.ablation_const_amp_tol
        ),
        "H4b_4_gauge_h_invariant": bool(max_delta_h < cfg.gauge_tol),
        "H4b_4_gauge_delta_order_invariant": bool(max_delta_order < cfg.gauge_tol),
        "Df_estimators_agree_on_average": bool(float(main_eps_df["df_estimators_abs_diff_mean"].mean()) < cfg.df_agreement_tol),
        "Failed_seed_share_below_limit": bool(failed_seed_share <= cfg.max_failed_seed_share),
        "Ablation_A_corr_not_applicable_or_weaker": bool(
            (not corr_abl_a_applicable) or (not np.isfinite(corr_abl_a)) or (abs(corr_abl_a) < abs(corr_main_pearson))
        ),
        "Ablation_B_corr_not_applicable_or_weaker": bool(
            (not corr_abl_b_applicable) or (not np.isfinite(corr_abl_b)) or (abs(corr_abl_b) < abs(corr_main_pearson))
        ),
        "Ablation_D_correlation_weaker_than_main": bool(
            (not np.isfinite(corr_abl_d)) or (abs(corr_abl_d) < abs(corr_main_pearson))
        ),
    }
    checks["pass_all"] = bool(all(checks.values()))

    summary = {
        "checks": checks,
        "main": {
            "corr_pearson": float(corr_main_pearson),
            "corr_spearman": float(corr_main_spearman),
            "perm_pvalue": float(p_perm),
            "partial_corr_given_log_epsilon": float(partial_corr),
            "epsilon_star": float(eps_star),
            "epsilon_star_bootstrap_ci_95": [float(eps_star_ci[0]), float(eps_star_ci[1])],
            "epsilon_star_bootstrap_share_in_window": float(eps_star_in_window_share),
        },
        "ablations": {
            "A_unitary_corr": float(corr_abl_a),
            "A_unitary_corr_applicable": bool(corr_abl_a_applicable),
            "A_unitary_delta_order_var": float(ablation_a_delta_var),
            "A_unitary_delta_order_max": float(ablation_a_delta_max),
            "B_commutative_corr": float(corr_abl_b),
            "B_commutative_corr_applicable": bool(corr_abl_b_applicable),
            "B_commutative_delta_order_var": float(ablation_b_delta_var),
            "B_commutative_delta_order_max": float(ablation_b_delta_max),
            "D_mu_permutation_corr": float(corr_abl_d),
            "C2_geometry_delta_order_mean_min": float(c2_df["delta_order_mean"].min()),
            "C2_geometry_delta_order_mean_max": float(c2_df["delta_order_mean"].max()),
        },
        "gauge": {
            "max_abs_delta_h_triangle": float(max_delta_h),
            "max_abs_delta_delta_order": float(max_delta_order),
        },
        "seed_diagnostics": {
            "total_seeds": int(len(seed_diag_df)),
            "failed_seeds": int(seed_diag_df["failed_any"].sum()),
            "failed_seed_share": float(failed_seed_share),
            "df_agreement_ratio_min_threshold": float(cfg.seed_df_agreement_ratio_min),
            "psd_r2_median_min_threshold": float(cfg.seed_psd_r2_median_min),
            "valid_seed_corr_count": int(np.isfinite(seed_corr_valid_df["pearson"]).sum()) if len(seed_corr_valid_df) else 0,
        },
        "distributions": {
            "seed_corr_pearson_mean": finite_mean(seed_corr_df["pearson"]),
            "seed_corr_pearson_q025": finite_quantile(seed_corr_df["pearson"], 0.025),
            "seed_corr_pearson_q975": finite_quantile(seed_corr_df["pearson"], 0.975),
            "seed_corr_pearson_valid_mean": finite_mean(seed_corr_valid_df["pearson"]),
            "seed_corr_pearson_valid_q025": finite_quantile(seed_corr_valid_df["pearson"], 0.025),
            "seed_corr_pearson_valid_q975": finite_quantile(seed_corr_valid_df["pearson"], 0.975),
            "loop_corr_pearson_mean": finite_mean(loop_corr_df["pearson"]),
            "loop_corr_pearson_q025": finite_quantile(loop_corr_df["pearson"], 0.025),
            "loop_corr_pearson_q975": finite_quantile(loop_corr_df["pearson"], 0.975),
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    main_seed_df.to_csv(outdir / "experiment_F4b_main_seed_metrics.csv", index=False)
    main_loop_df.to_csv(outdir / "experiment_F4b_main_loop_metrics.csv", index=False)
    main_eps_df.to_csv(outdir / "experiment_F4b_main_epsilon_summary.csv", index=False)
    seed_corr_df.to_csv(outdir / "experiment_F4b_seed_corr_distribution.csv", index=False)
    seed_corr_valid_df.to_csv(outdir / "experiment_F4b_seed_corr_distribution_valid.csv", index=False)
    seed_diag_df.to_csv(outdir / "experiment_F4b_seed_diagnostics.csv", index=False)
    loop_corr_df.to_csv(outdir / "experiment_F4b_loop_corr_distribution.csv", index=False)
    boot_df.to_csv(outdir / "experiment_F4b_bootstrap_eps_star.csv", index=False)
    gauge_df.to_csv(outdir / "experiment_F4b_gauge_checks.csv", index=False)

    abl_a_seed_df.to_csv(outdir / "experiment_F4b_ablation_A_seed_metrics.csv", index=False)
    abl_a_loop_df.to_csv(outdir / "experiment_F4b_ablation_A_loop_metrics.csv", index=False)
    abl_b_seed_df.to_csv(outdir / "experiment_F4b_ablation_B_seed_metrics.csv", index=False)
    abl_b_loop_df.to_csv(outdir / "experiment_F4b_ablation_B_loop_metrics.csv", index=False)
    abl_d_seed_df.to_csv(outdir / "experiment_F4b_ablation_D_seed_metrics.csv", index=False)
    c2_df.to_csv(outdir / "experiment_F4b_ablation_C2_geometry_sweep.csv", index=False)

    with open(outdir / "experiment_F4b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Experiment F4b Report: Independent Holonomy Test with Ablations",
        "",
        "## Main Regime (independent from F3)",
        f"- corr(delta_order, D_f-d_top), Pearson = {summary['main']['corr_pearson']:.6f}",
        f"- corr(delta_order, D_f-d_top), Spearman = {summary['main']['corr_spearman']:.6f}",
        f"- permutation p-value (epsilon-level) = {summary['main']['perm_pvalue']:.6e}",
        f"- partial corr(delta_order, D_f-d_top | log epsilon) = {summary['main']['partial_corr_given_log_epsilon']:.6f}",
        f"- epsilon* = {summary['main']['epsilon_star']:.2f}",
        f"- epsilon* bootstrap 95% CI = [{summary['main']['epsilon_star_bootstrap_ci_95'][0]:.2f}, {summary['main']['epsilon_star_bootstrap_ci_95'][1]:.2f}]",
        f"- bootstrap share epsilon* in [{cfg.epsilon_star_window[0]:.2f}, {cfg.epsilon_star_window[1]:.2f}] = {summary['main']['epsilon_star_bootstrap_share_in_window']:.3f}",
        "",
        "## Ablations",
        f"- A (unitary) corr = {summary['ablations']['A_unitary_corr']:.6f}",
        f"- A corr applicable = {summary['ablations']['A_unitary_corr_applicable']}",
        f"- A Var(delta_order) = {summary['ablations']['A_unitary_delta_order_var']:.3e}",
        f"- A max(delta_order) = {summary['ablations']['A_unitary_delta_order_max']:.3e}",
        f"- B (commutative geometry) corr = {summary['ablations']['B_commutative_corr']:.6f}",
        f"- B corr applicable = {summary['ablations']['B_commutative_corr_applicable']}",
        f"- B Var(delta_order) = {summary['ablations']['B_commutative_delta_order_var']:.3e}",
        f"- B max delta_order = {summary['ablations']['B_commutative_delta_order_max']:.3e}",
        f"- D (mu permutation) corr = {summary['ablations']['D_mu_permutation_corr']:.6f}",
        f"- C2 geometry sweep delta_order mean range = [{summary['ablations']['C2_geometry_delta_order_mean_min']:.6f}, {summary['ablations']['C2_geometry_delta_order_mean_max']:.6f}]",
        "",
        "## Seed Diagnostics",
        f"- failed seed share = {summary['seed_diagnostics']['failed_seed_share']:.3f} ({summary['seed_diagnostics']['failed_seeds']}/{summary['seed_diagnostics']['total_seeds']})",
        f"- df-agreement threshold = {summary['seed_diagnostics']['df_agreement_ratio_min_threshold']:.2f}",
        f"- psd-median-R2 threshold = {summary['seed_diagnostics']['psd_r2_median_min_threshold']:.2f}",
        f"- valid-seed corr count = {summary['seed_diagnostics']['valid_seed_corr_count']}",
        "",
        "## Gauge Robustness",
        f"- max |delta h_triangle| = {summary['gauge']['max_abs_delta_h_triangle']:.3e}",
        f"- max |delta delta_order| = {summary['gauge']['max_abs_delta_delta_order']:.3e}",
        "",
        "## Correlation Distributions",
        f"- seed-level Pearson mean [q2.5%, q97.5%] = {summary['distributions']['seed_corr_pearson_mean']:.6f} [{summary['distributions']['seed_corr_pearson_q025']:.6f}, {summary['distributions']['seed_corr_pearson_q975']:.6f}]",
        f"- valid-seed Pearson mean [q2.5%, q97.5%] = {summary['distributions']['seed_corr_pearson_valid_mean']:.6f} [{summary['distributions']['seed_corr_pearson_valid_q025']:.6f}, {summary['distributions']['seed_corr_pearson_valid_q975']:.6f}]",
        f"- loop-level Pearson mean [q2.5%, q97.5%] = {summary['distributions']['loop_corr_pearson_mean']:.6f} [{summary['distributions']['loop_corr_pearson_q025']:.6f}, {summary['distributions']['loop_corr_pearson_q975']:.6f}]",
        "",
        "## Criteria",
    ]
    for k, v in checks.items():
        report_lines.append(f"- {k}: {v}")
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_F4b_independent_holonomy_ablation",
        help="output directory",
    )
    parser.add_argument(
        "--eps-list",
        default="0.01,0.03,0.10,0.30,0.70,0.85,1.00,1.15,1.40,3.00,10.00",
        help="comma-separated epsilon list",
    )
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--n-loops", type=int, default=50)
    parser.add_argument("--bootstrap-iters", type=int, default=1200)
    parser.add_argument("--permutation-iters", type=int, default=5000)
    args = parser.parse_args()

    cfg = F4bConfig(
        eps_list=parse_eps_list(args.eps_list),
        n_seeds=max(4, int(args.n_seeds)),
        n_loops=max(4, int(args.n_loops)),
        bootstrap_iters=max(200, int(args.bootstrap_iters)),
        permutation_iters=max(500, int(args.permutation_iters)),
    )
    outdir = Path(args.out)
    summary = run_f4b(outdir=outdir, cfg=cfg)

    print(json.dumps(summary["main"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["ablations"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["gauge"], ensure_ascii=False, indent=2))
    print("\nChecks:")
    print(json.dumps(summary["checks"], ensure_ascii=False, indent=2))
    print(f"\nSaved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
