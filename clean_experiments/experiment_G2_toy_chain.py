#!/usr/bin/env python3
"""Experiment G2 (toy-chain): Clausius regression scan over epsilon."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.common import (
        SX,
        SZ,
        comm,
        kronN,
        op_on_site,
        normalize_vec,
        su2_from_axis_angle,
        axis_from_hermitian,
        projectors_axis,
        plaquette,
        f_phys_from_plaquette,
        partial_trace_site,
        bond_xx_yy,
        linear_regression,
    )
except ImportError:
    from common import (
        SX,
        SZ,
        comm,
        kronN,
        op_on_site,
        normalize_vec,
        su2_from_axis_angle,
        axis_from_hermitian,
        projectors_axis,
        plaquette,
        f_phys_from_plaquette,
        partial_trace_site,
        bond_xx_yy,
        linear_regression,
    )


def l1_coherence(coeffs: np.ndarray) -> np.ndarray:
    s = np.sum(np.abs(coeffs), axis=1)
    return (s * s - 1.0).astype(float)


def axis_mu(x: int, k: int) -> np.ndarray:
    return normalize_vec(
        [
            np.sin(0.7 * x + 0.9 * k) + 0.25 * np.cos(0.3 * x - 0.4 * k),
            np.cos(0.6 * x - 0.8 * k) + 0.15 * np.sin(0.9 * x + 0.2 * k),
            np.sin(0.5 * x + 0.5 * k) + 0.35,
        ]
    )


def axis_x(x: int, k: int) -> np.ndarray:
    return normalize_vec(
        [
            np.cos(0.8 * x + 0.4 * k) + 0.1,
            np.sin(0.5 * x - 0.7 * k) + 0.15 * np.cos(0.9 * x + 0.3 * k),
            np.cos(0.4 * x + 0.6 * k) - 0.25,
        ]
    )


def build_geometry(n_sites: int, k_layers: int) -> tuple[
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    xs = np.arange(n_sites)
    ks = np.arange(k_layers)

    omega = 0.25 + 0.45 * np.exp(-0.5 * ((ks - (k_layers - 1) / 2) / (0.35 * k_layers)) ** 2)
    angle_mu = 0.9 * omega / omega.max()
    angle_x = 0.55 * (0.75 + 0.25 * np.cos(2 * np.pi * ks / k_layers))

    ux: dict[tuple[int, int], np.ndarray] = {}
    umu: dict[tuple[int, int], np.ndarray] = {}
    for x in xs:
        for k in ks:
            umu[(x, k)] = su2_from_axis_angle(axis_mu(x, k), float(angle_mu[k]))
            ux[(x, k)] = su2_from_axis_angle(axis_x(x, k), float(angle_x[k]))

    f_site = {
        (x, k): f_phys_from_plaquette(plaquette(ux, umu, x, k, n_sites, k_layers))
        for x in xs
        for k in ks
    }

    u_chain = {k: kronN([umu[(x, k)] for x in xs]) for k in range(k_layers)}

    pointer_bases = {}
    for k in range(k_layers):
        f_avg = sum(f_site[(x, k)] for x in xs) / float(n_sites)
        axis = axis_from_hermitian(f_avg)
        _, _, vecs = projectors_axis(axis)
        pointer_bases[k] = kronN([vecs for _ in xs])

    return ux, umu, f_site, u_chain, pointer_bases


def run_one_epsilon(
    epsilon: float,
    seed: int,
    n_sites: int,
    k_layers: int,
    n_steps: int,
    n_traj: int,
    dmu_mcwf: float,
    gamma_jump: float,
    delta_s_hor: float,
    f_site: dict[tuple[int, int], np.ndarray],
    u_chain: dict[int, np.ndarray],
    pointer_bases: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    xs = np.arange(n_sites)

    dim = 2 ** n_sites
    h_bonds = [bond_xx_yy(i, i + 1, n_sites, 1.0) for i in range(n_sites - 1)]
    h_chain = sum(h_bonds)

    # Noise-sensitive boost-energy observable used in dQ_in.
    field = sum(op_on_site(SZ, i, n_sites) for i in xs) / float(n_sites)
    k_mat = h_chain + epsilon * field

    s_hor = np.array([k * delta_s_hor for k in range(k_layers + 1)], dtype=float)
    w_layers = np.exp(-s_hor[:-1])
    w_layers = w_layers / np.sum(w_layers)

    p_jump = min(gamma_jump * dmu_mcwf, 0.8)

    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0
    phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=n_traj))
    psis = phases[:, None] * psi0[None, :]
    layer = np.zeros(n_traj, dtype=int)

    # +/-1 signatures of computational basis for local Z phase kicks.
    bits = np.array([[(s >> (n_sites - 1 - q)) & 1 for q in range(n_sites)] for s in range(dim)], dtype=float)
    z_signs = 1.0 - 2.0 * bits

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
            vals = np.real_if_close(np.linalg.eigvalsh(0.5 * (rho_block + rho_block.conj().T)))
            vals = np.maximum(vals, 0.0)
            if np.sum(vals) <= 0:
                continue
            entropy -= float(np.sum(vals * np.log(vals + eps)))
        return entropy

    def layer_features(local_psis: np.ndarray, local_layer: np.ndarray) -> dict[str, float]:
        p_k = np.array([np.mean(local_layer == kk) for kk in range(k_layers + 1)], dtype=float)
        lam_cond = np.zeros(k_layers, dtype=float)
        coh_cond = np.zeros(k_layers, dtype=float)

        for kk in range(k_layers):
            idx = np.where(local_layer == kk)[0]
            if len(idx) == 0:
                continue

            psi_block = local_psis[idx]
            rho_cond = (psi_block.conj().T @ psi_block) / len(idx)

            lam_vals = []
            for x in xs:
                red = partial_trace_site(rho_cond, x, n_sites)
                lam_vals.append(float(np.real(np.trace(f_site[(x, kk)] @ red))))
            lam_cond[kk] = float(np.mean(lam_vals))

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

            psi_new = (u_chain[kk] @ psi_proj.T).T

            if epsilon > 0.0:
                xis = rng.normal(size=(len(idx), n_sites))
                diag = xis @ z_signs.T
                psi_new = psi_new * np.exp(-0.5j * epsilon * diag)

            psi_new = psi_new / np.linalg.norm(psi_new, axis=1)[:, None]
            psis[idx] = psi_new
            layer[idx] = kk + 1

        k_after = expvals_many(psis, k_mat)
        s_ens_new = blocks_entropy(psis, layer)

        d_q_avg = float(np.mean(k_after - k_before))
        d_q_in = -d_q_avg
        d_s_hor_avg = float(np.mean(s_hor[layer] - s_hor[layer_before]))

        feats = layer_features(psis, layer)
        rows.append(
            {
                "n": int(n),
                "epsilon": float(epsilon),
                "dS_hor_avg": d_s_hor_avg,
                "dQ_in": d_q_in,
                "dQ_avg": d_q_avg,
                "S_ens": float(s_ens_new),
                "dS_ens": float(s_ens_new - s_ens),
                "jump_rate_emp": float(np.mean(layer != layer_before)),
                **feats,
            }
        )
        s_ens = s_ens_new

    series = pd.DataFrame(rows)
    slope, intercept, r2 = linear_regression(series["dQ_in"].to_numpy(), series["dS_hor_avg"].to_numpy())
    fit = pd.DataFrame(
        [
            {
                "epsilon": float(epsilon),
                "a_1_over_Teff_from_Shor": float(slope),
                "r2_Shor": float(r2),
                "b_Shor": float(intercept),
                "mean_jump": float(series["jump_rate_emp"].mean()),
                "mean_Lambda_w": float(series["Lambda_w"].mean()),
                "mean_Coh_ptr_w": float(series["Coh_ptr_w"].mean()),
            }
        ]
    )
    return series, fit


def run_scan(
    outdir: Path,
    eps_list: tuple[float, ...] = (0.05, 0.15, 0.30, 0.60),
    seed: int = 20260219,
    n_sites: int = 3,
    k_layers: int = 6,
    n_steps: int = 280,
    n_traj: int = 900,
    dmu_mcwf: float = 0.03,
    gamma_jump: float = 0.31,
    delta_s_hor: float = 0.12,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if quick:
        n_steps = min(n_steps, 90)
        n_traj = min(n_traj, 250)

    outdir.mkdir(parents=True, exist_ok=True)

    _, _, f_site, u_chain, pointer_bases = build_geometry(n_sites=n_sites, k_layers=k_layers)

    all_series = []
    fit_rows = []
    for i, eps in enumerate(eps_list):
        series, fit = run_one_epsilon(
            epsilon=float(eps),
            seed=seed + 17 * i,
            n_sites=n_sites,
            k_layers=k_layers,
            n_steps=n_steps,
            n_traj=n_traj,
            dmu_mcwf=dmu_mcwf,
            gamma_jump=gamma_jump,
            delta_s_hor=delta_s_hor,
            f_site=f_site,
            u_chain=u_chain,
            pointer_bases=pointer_bases,
        )
        series.to_csv(outdir / f"G2_toy_chain_eps_{eps:.2f}_timeseries.csv", index=False)
        fit.to_csv(outdir / f"G2_toy_chain_eps_{eps:.2f}_fit.csv", index=False)

        all_series.append(series)
        fit_rows.append(fit.iloc[0].to_dict())

    fit_by_eps = pd.DataFrame(fit_rows).sort_values("epsilon").reset_index(drop=True)

    merged = pd.concat(all_series, ignore_index=True)
    slope, intercept, r2 = linear_regression(merged["dQ_in"].to_numpy(), merged["dS_hor_avg"].to_numpy())
    fit_global = pd.DataFrame(
        [
            {
                "a_1_over_Teff_global_from_Shor": float(slope),
                "b_global_Shor": float(intercept),
                "r2_global_Shor": float(r2),
                "mean_jump": float(merged["jump_rate_emp"].mean()),
                "mean_Lambda_w": float(merged["Lambda_w"].mean()),
            }
        ]
    )

    merged.to_csv(outdir / "experiment_G2_toy_chain_timeseries_all.csv", index=False)
    fit_by_eps.to_csv(outdir / "experiment_G2_toy_chain_fit_by_eps.csv", index=False)
    fit_global.to_csv(outdir / "experiment_G2_toy_chain_fit_global.csv", index=False)

    return merged, fit_by_eps, fit_global


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_G2_toy_chain", help="output directory")
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--quick", action="store_true", help="run lightweight settings")
    args = parser.parse_args()

    outdir = Path(args.out)
    _, fit_by_eps, fit_global = run_scan(outdir=outdir, seed=args.seed, quick=args.quick)

    print("Per-epsilon fit:")
    print(fit_by_eps.to_string(index=False))
    print("\nGlobal fit:")
    print(fit_global.to_string(index=False))
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
