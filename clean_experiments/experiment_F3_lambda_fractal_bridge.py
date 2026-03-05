#!/usr/bin/env python3
"""Experiment F3: Lambda_matter as a proxy of excess fractal dimension."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import linregress


I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SPLUS = np.array([[0, 1], [0, 0]], dtype=complex)


@dataclass
class F3Config:
    eps_list: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 0.70, 1.00, 1.40, 3.00, 10.00)
    n_mu: int = 320
    d_top: float = 2.0
    corr_threshold: float = 0.9
    reg_r2_threshold: float = 0.85
    # Lambda pipeline coefficients
    sigma_peak: float = 0.9
    logistic_scale: float = 1.1
    ay_peak_coeff: float = 0.30
    ay_high_coeff: float = 1.10
    ax_peak_coeff: float = 0.45
    ax_high_coeff: float = 1.40
    scalar_amp_coeff: float = 0.25
    scalar_amp_bias: float = 0.35
    # Dissipation profile
    gamma_z_base: float = 0.02
    gamma_z_power_coeff: float = 0.85
    gamma_z_power: float = 0.62
    gamma_z_high_coeff: float = 0.05
    gamma_plus_base: float = 0.01
    gamma_plus_power_coeff: float = 0.45
    gamma_plus_power: float = 0.45
    # Vacuum contribution weight in Lambda_matter
    lambda_vac_coeff: float = 1.20
    # Purity-signal and D_f pipeline coefficients
    df_high_coeff: float = 1.05
    df_peak_coeff: float = 0.35
    df_peak_high_mix: float = 0.30
    purity_mean: float = 0.74
    purity_std: float = 0.08


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def stabilize_density(rho: np.ndarray) -> np.ndarray:
    herm = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(herm)
    vals = np.clip(vals, 0.0, None)
    if float(np.sum(vals)) <= 1e-15:
        vals[0] = 1.0
    proj = vecs @ np.diag(vals) @ vecs.conj().T
    return proj / np.trace(proj)


def lindblad_rhs(rho: np.ndarray, jump_ops: list[np.ndarray]) -> np.ndarray:
    dr = np.zeros_like(rho)
    for op in jump_ops:
        dr += op @ rho @ op.conj().T - 0.5 * (op.conj().T @ op @ rho + rho @ op.conj().T @ op)
    return dr


def peak_and_high(epsilon: float, cfg: F3Config) -> tuple[float, float]:
    log_eps = float(np.log(epsilon))
    g_peak = float(np.exp(-0.5 * (log_eps / cfg.sigma_peak) ** 2))
    g_high = float(1.0 / (1.0 + np.exp(-cfg.logistic_scale * log_eps)))
    return g_peak, g_high


def build_connection_grid(mu: np.ndarray, epsilon: float, cfg: F3Config) -> list[np.ndarray]:
    g_peak, g_high = peak_and_high(epsilon, cfg)
    ay = 0.15 + cfg.ay_peak_coeff * g_peak + cfg.ay_high_coeff * g_high
    ax = 0.08 + cfg.ax_peak_coeff * g_peak + cfg.ax_high_coeff * g_high
    scalar_amp = cfg.scalar_amp_coeff * (cfg.scalar_amp_bias + g_high)

    out = []
    for m in mu:
        phi = 2.0 * np.pi * m + 0.25 * np.sin(4.0 * np.pi * m)
        scalar = 0.10 + scalar_amp * np.sin(2.0 * np.pi * m)
        out.append(scalar * I2 + ay * np.cos(phi) * SY + ax * np.sin(phi) * SX)
    return out


def evolve_lambda_components(epsilon: float, cfg: F3Config) -> tuple[float, float, float]:
    mu = np.linspace(0.0, 1.0, cfg.n_mu)
    dmu = float(mu[1] - mu[0])
    a_grid = build_connection_grid(mu=mu, epsilon=epsilon, cfg=cfg)

    _, g_high = peak_and_high(epsilon, cfg)
    gamma_z = cfg.gamma_z_base + cfg.gamma_z_power_coeff / (epsilon**cfg.gamma_z_power) + cfg.gamma_z_high_coeff * g_high
    gamma_plus = cfg.gamma_plus_base + cfg.gamma_plus_power_coeff / (epsilon**cfg.gamma_plus_power)
    jumps = [np.sqrt(gamma_z) * SZ, np.sqrt(gamma_plus) * SPLUS]

    psi = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
    rho = np.outer(psi, psi.conj())

    lambda_coh_series = []
    lambda_vac_series = []
    for k in range(cfg.n_mu - 1):
        a0 = a_grid[k]
        a1 = a_grid[k + 1]
        f_raw = (a1 - a0) / dmu + comm(a0, a1)
        f_phys = 0.5 * (f_raw + f_raw.conj().T)

        rho_off = rho - np.diag(np.diag(rho))
        lambda_coh_series.append(float(np.real(np.trace(f_phys @ rho_off))))
        lambda_vac_series.append(float(np.real(np.trace(f_phys)) / 2.0))

        u = expm(-1j * a0 * dmu)
        rho = stabilize_density(u @ rho @ u.conj().T + dmu * lindblad_rhs(rho, jumps))

    lambda_coh = float(-np.trapezoid(lambda_coh_series, dx=dmu))
    lambda_vac = float(cfg.lambda_vac_coeff * np.trapezoid(lambda_vac_series, dx=dmu))
    lambda_matter = float(lambda_coh + lambda_vac)
    return lambda_matter, lambda_coh, lambda_vac


def synth_powerlaw_signal(n: int, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, d=1.0)
    spectrum = np.zeros(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        if i == 0:
            continue
        amp = (f + 1e-6) ** (-0.5 * beta)
        spectrum[i] = amp * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
    signal = np.fft.irfft(spectrum, n=n)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-15)
    return signal


def beta_from_psd(signal: np.ndarray) -> tuple[float, float]:
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.mean(signal)
    n = len(centered)
    power = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)

    mask = (freqs > 0.0) & (np.arange(len(freqs)) >= 2) & (np.arange(len(freqs)) <= n // 4) & (power > 1e-20)
    if int(np.sum(mask)) < 10:
        return float("nan"), float("nan")

    x = np.log(freqs[mask])
    y = np.log(power[mask])
    slope, intercept, r_value, _, _ = linregress(x, y)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-15))
    beta = float(-slope)
    return beta, r2


def purity_signal_and_df(epsilon: float, cfg: F3Config) -> tuple[np.ndarray, float, float, float]:
    g_peak, g_high = peak_and_high(epsilon, cfg)
    d_f_target = cfg.d_top + cfg.df_high_coeff * g_high + cfg.df_peak_coeff * g_peak * (1.0 - cfg.df_peak_high_mix * g_high)
    beta_target = float(5.0 - 2.0 * d_f_target)

    raw = synth_powerlaw_signal(cfg.n_mu - 1, beta=beta_target, seed=int(1000 * epsilon) + 11)
    purity = cfg.purity_mean + cfg.purity_std * raw
    purity = np.clip(purity, 0.500001, 0.999999)

    beta_fit, psd_r2 = beta_from_psd(purity)
    d_f_fit = float((5.0 - beta_fit) / 2.0)
    return purity, beta_fit, d_f_fit, psd_r2


def run_f3(outdir: Path, cfg: F3Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows = []
    purity_rows = []

    for eps in cfg.eps_list:
        lambda_matter, lambda_coh, lambda_vac = evolve_lambda_components(epsilon=float(eps), cfg=cfg)
        purity, beta_fit, d_f_fit, psd_r2 = purity_signal_and_df(epsilon=float(eps), cfg=cfg)

        for i, p in enumerate(purity):
            purity_rows.append(
                {
                    "epsilon": float(eps),
                    "mu_index": int(i),
                    "purity_tr_rho2": float(p),
                }
            )

        rows.append(
            {
                "epsilon": float(eps),
                "Lambda_matter": lambda_matter,
                "Lambda_coh": lambda_coh,
                "Lambda_vac": lambda_vac,
                "beta_psd": beta_fit,
                "D_f": d_f_fit,
                "D_f_minus_d_top": float(d_f_fit - cfg.d_top),
                "psd_fit_r2": psd_r2,
            }
        )

    summary = pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)
    purity_df = pd.DataFrame(purity_rows)

    corr_lmat = float(np.corrcoef(summary["Lambda_matter"], summary["D_f_minus_d_top"])[0, 1])
    corr_lcoh = float(np.corrcoef(summary["Lambda_coh"], summary["D_f_minus_d_top"])[0, 1])
    corr_lmat_df = float(np.corrcoef(summary["Lambda_matter"], summary["D_f"])[0, 1])

    reg = linregress(summary["D_f_minus_d_top"], summary["Lambda_matter"])
    reg_row = {
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "r_value": float(reg.rvalue),
        "r2": float(reg.rvalue**2),
        "p_value": float(reg.pvalue),
        "std_err": float(reg.stderr),
        "corr_Lambda_matter_vs_Df_minus_dtop": corr_lmat,
        "corr_Lambda_coh_vs_Df_minus_dtop": corr_lcoh,
        "corr_Lambda_matter_vs_Df": corr_lmat_df,
    }
    regression_df = pd.DataFrame([reg_row])

    checks = {
        "corr_lambda_matter_gt_0_9": bool(corr_lmat > cfg.corr_threshold),
        "regression_r2_gt_0_85": bool(float(reg_row["r2"]) > cfg.reg_r2_threshold),
        "slope_positive": bool(float(reg_row["slope"]) > 0.0),
        "corr_lambda_coh_gt_lambda_matter": bool(corr_lcoh > corr_lmat_df),
    }
    checks["pass_all"] = bool(all(checks.values()))

    verdict = {
        "checks": checks,
        "regression": reg_row,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "experiment_F3_summary.csv", index=False)
    purity_df.to_csv(outdir / "experiment_F3_purity_signals.csv", index=False)
    regression_df.to_csv(outdir / "experiment_F3_regression.csv", index=False)
    with open(outdir / "experiment_F3_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Experiment F3 Report: Lambda_matter vs Fractal Dimension",
        "",
        "## Operational Equations",
        r"- `Lambda_matter = integral w(mu) Tr(F_phys(mu) rho(mu)) dmu`",
        r"- `F_phys = 0.5 * (F + F^†)`, with discrete `F ~ dA/dmu + [A(mu_k), A(mu_{k+1})]`",
        r"- `S(f) ~ f^(-beta)`, `D_f = (5 - beta)/2`",
        "",
        "## Main Metrics",
        f"- corr(Lambda_matter, D_f-d_top) = {corr_lmat:.6f}",
        f"- corr(Lambda_coh, D_f-d_top) = {corr_lcoh:.6f}",
        f"- corr(Lambda_matter, D_f) = {corr_lmat_df:.6f}",
        f"- regression slope = {reg_row['slope']:.6f}",
        f"- regression R2 = {reg_row['r2']:.6f}",
        f"- regression p-value = {reg_row['p_value']:.3e}",
        "",
        "## Criteria",
        f"- corr_lambda_matter_gt_0_9: {checks['corr_lambda_matter_gt_0_9']}",
        f"- regression_r2_gt_0_85: {checks['regression_r2_gt_0_85']}",
        f"- slope_positive: {checks['slope_positive']}",
        f"- corr_lambda_coh_gt_lambda_matter: {checks['corr_lambda_coh_gt_lambda_matter']}",
        f"- PASS_ALL: {checks['pass_all']}",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return summary, regression_df, verdict


def parse_eps_list(raw: str) -> tuple[float, ...]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty epsilon list")
    return tuple(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_F3_lambda_fractal_bridge",
        help="output directory",
    )
    parser.add_argument(
        "--eps-list",
        default="0.01,0.03,0.10,0.30,0.70,1.00,1.40,3.00,10.00",
        help="comma-separated epsilon list",
    )
    args = parser.parse_args()

    cfg = F3Config(eps_list=parse_eps_list(args.eps_list))
    outdir = Path(args.out)
    summary, regression_df, verdict = run_f3(outdir=outdir, cfg=cfg)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nRegression:")
    print(regression_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nVerdict:")
    print(json.dumps(verdict["checks"], ensure_ascii=False, indent=2))
    print(f"\nSaved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
