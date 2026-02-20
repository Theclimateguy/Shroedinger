#!/usr/bin/env python3
"""Experiment D: full balance closure on the extended base (t, x, mu)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import norm

try:
    from clean_experiments.common import (
        SZ,
        comm,
        kronN,
        op_on_site,
        normalize_vec,
        su2_from_axis_angle,
        lindblad_dissipator,
        expect,
        normalize_rho,
        random_product_state_density,
        bond_xx_yy,
    )
except ImportError:
    from common import (
        SZ,
        comm,
        kronN,
        op_on_site,
        normalize_vec,
        su2_from_axis_angle,
        lindblad_dissipator,
        expect,
        normalize_rho,
        random_product_state_density,
        bond_xx_yy,
    )


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


def run_experiment(
    outdir: Path,
    seed: int = 20260218,
    n_samples: int = 120,
    n_sites: int = 4,
    k_layers: int = 4,
    gamma_deph: float = 0.35,
    eta0: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    d_chain = 2 ** n_sites
    d_total = k_layers * d_chain

    h_bonds = [bond_xx_yy(i, i + 1, n_sites, 1.0) for i in range(n_sites - 1)]
    h_chain = sum(h_bonds)
    j_bonds = [1j * comm(h_bonds[i], h_bonds[i + 1]) for i in range(n_sites - 2)]

    layer_basis = np.eye(k_layers, dtype=complex)
    projectors = [layer_basis[[k]].T @ layer_basis[[k]] for k in range(k_layers)]

    def lift_to_layer(op_chain: np.ndarray, k: int) -> np.ndarray:
        return np.kron(projectors[k], op_chain)

    h_total = sum(lift_to_layer(h_chain, k) for k in range(k_layers))

    jump_ops_deph = [
        np.sqrt(gamma_deph) * lift_to_layer(op_on_site(SZ, site, n_sites), k)
        for k in range(k_layers)
        for site in range(n_sites)
    ]

    u_chain = {k: kronN([umu[(x, k)] for x in xs]) for k in range(k_layers - 1)}

    jump_ops_mu = []
    for k in range(k_layers - 1):
        shift = np.zeros((k_layers, k_layers), dtype=complex)
        shift[k + 1, k] = 1.0
        jump_ops_mu.append(np.sqrt(eta0) * np.kron(shift, u_chain[k]))

    noncomm_norm = float("nan")
    if k_layers >= 3:
        noncomm_norm = float(norm(u_chain[0] @ u_chain[1] - u_chain[1] @ u_chain[0]))

    rows = []
    max_abs_residual = 0.0
    for sample_id in range(n_samples):
        rr = np.random.default_rng(seed + 10000 + sample_id)
        weights = rr.dirichlet(alpha=np.ones(k_layers))

        rho_total = np.zeros((d_total, d_total), dtype=complex)
        for k in range(k_layers):
            rho_k = random_product_state_density(rr, n_sites)
            block = slice(k * d_chain, (k + 1) * d_chain)
            rho_total[block, block] = weights[k] * rho_k
        rho_total = normalize_rho(rho_total)

        unitary_part = -1j * comm(h_total, rho_total)
        d_deph = lindblad_dissipator(rho_total, jump_ops_deph)
        d_mu = lindblad_dissipator(rho_total, jump_ops_mu)
        dr = unitary_part + d_deph + d_mu

        for k in range(k_layers):
            j_exp = [expect(lift_to_layer(j_bonds[i], k), rho_total) for i in range(n_sites - 2)]
            for b in range(n_sites - 1):
                observable = lift_to_layer(h_bonds[b], k)
                d_o_dt = expect(observable, dr)
                j_right = j_exp[b] if b <= n_sites - 3 else 0.0
                j_left = j_exp[b - 1] if b - 1 >= 0 else 0.0
                div_j = j_right - j_left

                src_deph = expect(observable, d_deph)
                src_mu = expect(observable, d_mu)
                residual = d_o_dt + div_j - (src_deph + src_mu)
                max_abs_residual = max(max_abs_residual, abs(residual))

                rows.append(
                    {
                        "sample": sample_id,
                        "k": k,
                        "bond": b,
                        "dO_dt": d_o_dt,
                        "div_x_j": div_j,
                        "src_deph": src_deph,
                        "src_mu": src_mu,
                        "residual": residual,
                        "abs_residual": abs(residual),
                    }
                )

    dataset = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "seed": seed,
                "Nsamples": n_samples,
                "L_sites": n_sites,
                "K_layers": k_layers,
                "gamma_deph": gamma_deph,
                "eta0": eta0,
                "noncomm_norm_[U0,U1]": noncomm_norm,
                "max_abs_balance_residual": float(max_abs_residual),
                "mean_abs_balance_residual": float(dataset["abs_residual"].mean()),
                "balance_closure_ok_1e-10": bool(max_abs_residual < 1e-10),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(outdir / "experiment_D_balance_dataset.csv", index=False)
    summary.to_csv(outdir / "experiment_D_balance_summary.csv", index=False)
    return dataset, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_D", help="output directory")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--k-layers", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--eta0", type=float, default=0.9)
    args = parser.parse_args()

    outdir = Path(args.out)
    _, summary = run_experiment(
        outdir=outdir,
        seed=args.seed,
        n_samples=args.samples,
        n_sites=args.n_sites,
        k_layers=args.k_layers,
        gamma_deph=args.gamma,
        eta0=args.eta0,
    )
    print(summary.to_string(index=False))
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
