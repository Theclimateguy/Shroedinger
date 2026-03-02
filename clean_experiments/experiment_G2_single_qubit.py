#!/usr/bin/env python3
"""Experiment G2 (single-qubit): Clausius regression under profile scans."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.common import (
        SX,
        SY,
        SZ,
        build_u2_connection,
        transport_step,
        curvature_discrete,
        hermitian_part,
        lambda_matter,
        linear_regression,
    )
except ImportError:
    from common import SX, SY, SZ, build_u2_connection, transport_step, curvature_discrete, hermitian_part, lambda_matter, linear_regression


PROFILE_SET = ("constant", "gaussian", "oscillating")


def l1_coherence(coeffs: np.ndarray) -> np.ndarray:
    s = np.sum(np.abs(coeffs), axis=1)
    return (s * s - 1.0).astype(float)


def run_profile(
    omega_profile: str,
    out_prefix: Path,
    delta_mu: float = 2.0,
    k_layers: int = 64,
    omega0: float = 0.5,
    mu_c: float | None = None,
    sigma_g: float = 0.4,
    k_rg: float = 2.0,
    r_amp: float = 0.4,
    beta: float = 0.1,
    dmu_mcwf: float = 0.03,
    gamma_jump: float = 0.111,
    n_steps: int = 220,
    n_traj: int = 1200,
    alpha_state: float = np.pi / 4,
    delta_s_hor: float = 0.12,
    k_op: str = "sigma_x",
    mu0_for_k: float = 1.0,
    seed: int = 20260219,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    dmu_layer = delta_mu / k_layers
    mu = np.linspace(0.0, delta_mu, k_layers + 1)
    if mu_c is None:
        mu_c = delta_mu / 2.0

    if omega_profile == "constant":
        omega = omega0 * np.ones_like(mu)
    elif omega_profile == "gaussian":
        omega = omega0 * np.exp(-((mu - mu_c) ** 2) / (2.0 * sigma_g**2))
    elif omega_profile == "oscillating":
        omega = omega0 * np.cos(k_rg * mu)
    else:
        raise ValueError("omega_profile must be one of: constant, gaussian, oscillating")

    a_list = []
    cum_angle = 0.0
    for i in range(len(mu)):
        a_list.append(build_u2_connection(beta, r_amp * np.cos(cum_angle), r_amp * np.sin(cum_angle)))
        if i < k_layers:
            cum_angle += omega[i] * dmu_layer

    f_phys_list = []
    pointer_bases = []
    for k in range(k_layers):
        f = curvature_discrete(a_list[k], a_list[k + 1], dmu_layer)
        fp = hermitian_part(f)
        _, vecs = np.linalg.eigh(fp)
        f_phys_list.append(fp)
        pointer_bases.append(vecs)

    if k_op == "sigma_x":
        k_mat = SX
    elif k_op == "sigma_y":
        k_mat = SY
    elif k_op == "sigma_z":
        k_mat = SZ
    elif k_op == "A_mu0":
        k0 = int(np.clip(round(mu0_for_k / dmu_layer), 0, k_layers))
        k_mat = hermitian_part(a_list[k0])
    else:
        raise ValueError("k_op must be sigma_x, sigma_y, sigma_z, or A_mu0")

    s_hor = np.array([k * delta_s_hor for k in range(k_layers + 1)], dtype=float)
    w_layers = np.exp(-s_hor[:-1])
    w_layers = w_layers / np.sum(w_layers)

    p_jump = min(gamma_jump * dmu_mcwf, 0.8)

    psi0 = np.array([np.cos(alpha_state), np.sin(alpha_state)], dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)
    phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=n_traj))
    psis = phases[:, None] * psi0[None, :]
    layer = np.zeros(n_traj, dtype=int)

    def expvals_many(local_psis: np.ndarray, op: np.ndarray) -> np.ndarray:
        x = (op @ local_psis.T).T
        return np.real(np.sum(local_psis.conj() * x, axis=1))

    def blocks_entropy(local_psis: np.ndarray, local_layer: np.ndarray) -> float:
        entropy = 0.0
        eps = 1e-15
        for kk in range(k_layers + 1):
            idx = np.where(local_layer == kk)[0]
            if len(idx) == 0:
                continue
            psi_block = local_psis[idx]
            rho_block = (psi_block.conj().T @ psi_block) / n_traj
            vals = np.real_if_close(np.linalg.eigvalsh(hermitian_part(rho_block)))
            vals = np.maximum(vals, 0.0)
            if np.sum(vals) <= 0:
                continue
            entropy -= float(np.sum(vals * np.log(vals + eps)))
        return entropy

    def lambda_features(local_psis: np.ndarray, local_layer: np.ndarray) -> dict[str, float]:
        p_k = np.array([np.mean(local_layer == kk) for kk in range(k_layers + 1)], dtype=float)
        lam_cond = np.zeros(k_layers, dtype=float)
        coh_cond = np.zeros(k_layers, dtype=float)

        for kk in range(k_layers):
            idx = np.where(local_layer == kk)[0]
            if len(idx) == 0:
                continue
            psi_block = local_psis[idx]
            rho_cond = (psi_block.conj().T @ psi_block) / len(idx)
            lam_cond[kk] = lambda_matter(f_phys_list[kk], rho_cond)

            basis = pointer_bases[kk]
            coeffs = (basis.conj().T @ psi_block.T).T
            coh_cond[kk] = float(np.mean(l1_coherence(coeffs)))

        return {
            "Lambda_w": float(np.sum(w_layers * lam_cond)),
            "Lambda_unw": float(np.sum(p_k[:k_layers] * lam_cond)),
            "Coh_ptr_w": float(np.sum(w_layers * coh_cond)),
            "layer_mean": float(np.mean(local_layer)),
        }

    rows = []
    s_ens = blocks_entropy(psis, layer)
    for n in range(n_steps):
        k_before = expvals_many(psis, k_mat)
        layer_before = layer.copy()

        do_jump = rng.random(n_traj) < p_jump
        for kk in range(k_layers):
            idx = np.where((layer == kk) & do_jump)[0]
            if len(idx) == 0:
                continue

            basis = pointer_bases[kk]
            coeffs = (basis.conj().T @ psis[idx].T).T
            probs = np.abs(coeffs) ** 2
            probs = probs / np.sum(probs, axis=1, keepdims=True)

            rr = rng.random(len(idx))
            cdf = np.cumsum(probs, axis=1)
            choice = (cdf < rr[:, None]).sum(axis=1)

            ci = coeffs[np.arange(len(idx)), choice]
            phase = ci / (np.abs(ci) + 1e-15)
            evec = basis[:, choice].T
            psi_proj = phase[:, None] * evec

            u_step = transport_step(a_list[kk], dmu_layer)
            psi_new = (u_step @ psi_proj.T).T
            psi_new = psi_new / np.linalg.norm(psi_new, axis=1)[:, None]
            psis[idx] = psi_new
            layer[idx] = kk + 1

        k_after = expvals_many(psis, k_mat)
        s_ens_new = blocks_entropy(psis, layer)

        d_q_avg = float(np.mean(k_after - k_before))
        d_q_in = -d_q_avg
        d_s_hor_avg = float(np.mean(s_hor[layer] - s_hor[layer_before]))
        d_s_ens = float(s_ens_new - s_ens)
        s_ens = s_ens_new

        feats = lambda_features(psis, layer)
        rows.append(
            {
                "n": int(n),
                "omega_profile": omega_profile,
                "dmu_mcwf": float(dmu_mcwf),
                "Delta_mu": float(delta_mu),
                "K_layers": int(k_layers),
                "Gamma": float(gamma_jump),
                "p_jump": float(p_jump),
                "Ntraj": int(n_traj),
                "dS_hor_avg": d_s_hor_avg,
                "dQ_avg": d_q_avg,
                "dQ_in": d_q_in,
                "S_ens": float(s_ens),
                "dS_ens": d_s_ens,
                "jump_rate_emp": float(np.mean(layer != layer_before)),
                **feats,
            }
        )

    series = pd.DataFrame(rows)
    slope, intercept, r2 = linear_regression(series["dQ_in"].to_numpy(), series["dS_hor_avg"].to_numpy())
    fit = pd.DataFrame(
        [
            {
                "profile": omega_profile,
                "a_1_over_Teff": float(slope),
                "b": float(intercept),
                "R2": float(r2),
                "Teff": float(1.0 / slope) if abs(slope) > 1e-12 else np.nan,
                "mean_jump_rate": float(series["jump_rate_emp"].mean()),
                "mean_Lambda_w": float(series["Lambda_w"].mean()),
                "mean_Coh_ptr_w": float(series["Coh_ptr_w"].mean()),
            }
        ]
    )

    series.to_csv(f"{out_prefix}_timeseries.csv", index=False)
    fit.to_csv(f"{out_prefix}_fit.csv", index=False)
    return series, fit


def run_profile_scan(
    outdir: Path,
    profiles: tuple[str, ...] = PROFILE_SET,
    seed: int = 20260219,
    quick: bool = False,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)

    if quick:
        n_steps = 80
        n_traj = 300
    else:
        n_steps = 220
        n_traj = 1200

    summary_rows = []
    for i, profile in enumerate(profiles):
        prefix = outdir / f"G2_single_qubit_{profile}"
        _, fit = run_profile(
            omega_profile=profile,
            out_prefix=prefix,
            seed=seed + 10 * i,
            n_steps=n_steps,
            n_traj=n_traj,
        )
        summary_rows.append(
            {
                "profile": profile,
                "Teff": float(fit.loc[0, "Teff"]),
                "inv_Teff": float(fit.loc[0, "a_1_over_Teff"]),
                "R2": float(fit.loc[0, "R2"]),
                "mean_Lambda_w": float(fit.loc[0, "mean_Lambda_w"]),
                "mean_Coh_ptr_w": float(fit.loc[0, "mean_Coh_ptr_w"]),
                "mean_jump_rate": float(fit.loc[0, "mean_jump_rate"]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "experiment_G2_single_qubit_profiles_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_G2_single_qubit", help="output directory")
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--quick", action="store_true", help="run lightweight settings")
    args = parser.parse_args()

    outdir = Path(args.out)
    summary = run_profile_scan(outdir=outdir, seed=args.seed, quick=args.quick)
    print(summary.to_string(index=False))
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
