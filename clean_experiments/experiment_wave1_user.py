#!/usr/bin/env python3
"""Experiment B (wave-1): commutators, Hermitian curvature projection, and sin(phi) law."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _gaussian_profile(x: np.ndarray, base: float, amp: float, center: float, width: float) -> np.ndarray:
    return base + amp * np.exp(-0.5 * ((x - center) ** 2) / max(width, 1e-12))


def _connection(
    alpha: np.ndarray,
    theta: np.ndarray,
    angle: float,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    i2: np.ndarray,
) -> np.ndarray:
    return alpha[..., None, None] * i2 + theta[..., None, None] * (
        np.cos(angle) * sigma_y + np.sin(angle) * sigma_x
    )


def run_experiment(
    outdir: Path,
    nx: int = 200,
    mu1: float = 1.0,
    mu2: float = 2.0,
    alpha1_val: float = 0.3,
    alpha2_val: float = 0.6,
    alpha_mod_amp: float = 0.1,
    theta1_base: float = 0.5,
    theta1_amp: float = 0.3,
    theta1_center: float = np.pi / 2,
    theta1_width: float = 0.3,
    theta2_base: float = 0.4,
    theta2_amp: float = 0.3,
    theta2_center: float = 3 * np.pi / 2,
    theta2_width: float = 0.3,
    n_angle_fine: int = 50,
    write_csv: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    i2 = np.eye(2, dtype=complex)

    x_arr = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    alpha1 = alpha1_val * (1 + alpha_mod_amp * np.cos(x_arr))
    alpha2 = alpha2_val * (1 + alpha_mod_amp * np.cos(x_arr))
    theta1 = _gaussian_profile(x_arr, theta1_base, theta1_amp, theta1_center, theta1_width)
    theta2 = _gaussian_profile(x_arr, theta2_base, theta2_amp, theta2_center, theta2_width)

    if verbose:
        comm_yx = sigma_y @ sigma_x - sigma_x @ sigma_y
        print(f"[σ_y, σ_x] = -2i·σ_z? {np.allclose(comm_yx, -2j * sigma_z)}")

    states = {
        "ket_0": np.array([1, 0], dtype=complex),
        "ket_1": np.array([0, 1], dtype=complex),
        "ket_plus": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "ket_minus": np.array([1, -1], dtype=complex) / np.sqrt(2),
        "ket_plus_i": np.array([1, 1j], dtype=complex) / np.sqrt(2),
        "ket_minus_i": np.array([1, -1j], dtype=complex) / np.sqrt(2),
    }

    x_center = float(np.pi)
    center_idx = int(np.argmin(np.abs(x_arr - x_center)))
    alpha1_c = float(alpha1[center_idx])
    alpha2_c = float(alpha2[center_idx])
    theta1_c = float(theta1[center_idx])
    theta2_c = float(theta2[center_idx])

    angles_deg = [0, 15, 30, 45, 60, 75, 90]
    state_scan_rows = []
    for angle_deg in angles_deg:
        angle_rad = angle_deg * np.pi / 180.0
        a1_c = alpha1_c * i2 + theta1_c * sigma_y
        a2_c = alpha2_c * i2 + theta2_c * (np.cos(angle_rad) * sigma_y + np.sin(angle_rad) * sigma_x)
        f_c = (a2_c - a1_c) + (a1_c @ a2_c - a2_c @ a1_c)

        row_print = {}
        for name, psi in states.items():
            rho = np.outer(psi, psi.conj())
            val = float(np.real(np.trace(f_c @ rho)))
            state_scan_rows.append(
                {
                    "angle_deg": float(angle_deg),
                    "state": name,
                    "tr_F_rho_real": val,
                }
            )
            row_print[name] = round(val, 4)
        if verbose:
            print(f"{angle_deg:3d}°: {row_print}")

    state_scan_df = pd.DataFrame(state_scan_rows)

    rho_plus = np.outer(states["ket_plus"], states["ket_plus"].conj())
    rho_minus = np.outer(states["ket_minus"], states["ket_minus"].conj())

    a1_field = _connection(alpha1, theta1, 0.0, sigma_x, sigma_y, i2)
    a2_field_90 = _connection(alpha2, theta2, np.pi / 2, sigma_x, sigma_y, i2)
    f_field = (a2_field_90 - a1_field) + (a1_field @ a2_field_90 - a2_field_90 @ a1_field)
    f_phys = 0.5 * (f_field + np.swapaxes(np.conjugate(f_field), -1, -2))

    lambda_vac = alpha2 - alpha1
    f_su2 = f_phys - lambda_vac[:, None, None] * i2
    lambda_plus = np.real(np.einsum("xij,ji->x", f_su2, rho_plus))
    lambda_minus = np.real(np.einsum("xij,ji->x", f_su2, rho_minus))
    theta2_expected = theta2.copy()

    spatial_df = pd.DataFrame(
        {
            "x": x_arr,
            "theta2_x": theta2_expected,
            "Lambda_vac": lambda_vac,
            "Lambda_matter_plus": lambda_plus,
            "Lambda_matter_minus": lambda_minus,
            "Lambda_total_plus": lambda_vac + lambda_plus,
        }
    )

    max_abs_spatial_err = float(np.max(np.abs(lambda_plus - theta2_expected)))
    max_abs_antisym_err = float(np.max(np.abs(lambda_plus + lambda_minus)))

    angles_fine = np.linspace(0, np.pi / 2, n_angle_fine)
    lambda_angle = np.zeros_like(angles_fine)
    for i, ang in enumerate(angles_fine):
        a2_field = _connection(alpha2, theta2, float(ang), sigma_x, sigma_y, i2)
        f = (a2_field - a1_field) + (a1_field @ a2_field - a2_field @ a1_field)
        fp = 0.5 * (f + np.swapaxes(np.conjugate(f), -1, -2))
        f_su2_local = fp - lambda_vac[:, None, None] * i2
        lambda_angle[i] = float(np.mean(np.real(np.einsum("xij,ji->x", f_su2_local, rho_plus))))

    theta2_mean = float(np.mean(theta2))
    lambda_analytic = theta2_mean * np.sin(angles_fine)
    angle_err = lambda_angle - lambda_analytic
    max_abs_angle_err = float(np.max(np.abs(angle_err)))
    rmse_angle = float(np.sqrt(np.mean(angle_err**2)))
    ss_res = float(np.sum(angle_err**2))
    ss_tot = float(np.sum((lambda_angle - np.mean(lambda_angle)) ** 2))
    r2_angle = float(1.0 - ss_res / (ss_tot + 1e-15))

    ratio = lambda_angle / (np.sin(angles_fine) + 1e-12)
    ratio_mean = float(np.mean(ratio[5:])) if len(ratio) > 5 else float(np.mean(ratio))
    ratio_std = float(np.std(ratio[5:])) if len(ratio) > 5 else float(np.std(ratio))

    angle_df = pd.DataFrame(
        {
            "angle_deg": angles_fine * 180.0 / np.pi,
            "Lambda_matter_plus": lambda_angle,
            "Lambda_matter_analytic": lambda_analytic,
            "angle_error": angle_err,
        }
    )

    summary_df = pd.DataFrame(
        [
            {
                "mu1": float(mu1),
                "mu2": float(mu2),
                "nx": int(nx),
                "alpha1_val": float(alpha1_val),
                "alpha2_val": float(alpha2_val),
                "max_abs_spatial_err": max_abs_spatial_err,
                "max_abs_antisym_err": max_abs_antisym_err,
                "max_abs_angle_err": max_abs_angle_err,
                "rmse_angle": rmse_angle,
                "r2_angle": r2_angle,
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
            }
        ]
    )

    if verbose:
        print(f"max|числ - аналит| (phi=90, spatial) = {max_abs_spatial_err:.2e}")
        print(f"max|Λ_minus + Λ_plus| (antisym)     = {max_abs_antisym_err:.2e}")
        print(f"Λ/sin(phi): mean={ratio_mean:.6f}, std={ratio_std:.6f}")

    if write_csv:
        outdir.mkdir(parents=True, exist_ok=True)
        state_scan_df.to_csv(outdir / "experiment_B_wave1_state_scan.csv", index=False)
        angle_df.to_csv(outdir / "experiment_B_wave1_angle_scan.csv", index=False)
        spatial_df.to_csv(outdir / "experiment_B_wave1_spatial_lambda.csv", index=False)
        summary_df.to_csv(outdir / "experiment_B_wave1_summary.csv", index=False)
        if verbose:
            print(f"Saved: {outdir.resolve()}")

    return state_scan_df, angle_df, spatial_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_B_wave1", help="output directory")
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--alpha1", type=float, default=0.3)
    parser.add_argument("--alpha2", type=float, default=0.6)
    parser.add_argument("--alpha-mod-amp", type=float, default=0.1)
    parser.add_argument("--theta1-base", type=float, default=0.5)
    parser.add_argument("--theta1-amp", type=float, default=0.3)
    parser.add_argument("--theta1-center", type=float, default=float(np.pi / 2))
    parser.add_argument("--theta1-width", type=float, default=0.3)
    parser.add_argument("--theta2-base", type=float, default=0.4)
    parser.add_argument("--theta2-amp", type=float, default=0.3)
    parser.add_argument("--theta2-center", type=float, default=float(3 * np.pi / 2))
    parser.add_argument("--theta2-width", type=float, default=0.3)
    parser.add_argument("--n-angle-fine", type=int, default=50)
    args = parser.parse_args()

    run_experiment(
        outdir=Path(args.out),
        nx=args.nx,
        alpha1_val=args.alpha1,
        alpha2_val=args.alpha2,
        alpha_mod_amp=args.alpha_mod_amp,
        theta1_base=args.theta1_base,
        theta1_amp=args.theta1_amp,
        theta1_center=args.theta1_center,
        theta1_width=args.theta1_width,
        theta2_base=args.theta2_base,
        theta2_amp=args.theta2_amp,
        theta2_center=args.theta2_center,
        theta2_width=args.theta2_width,
        n_angle_fine=args.n_angle_fine,
        write_csv=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
