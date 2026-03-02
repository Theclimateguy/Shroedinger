#!/usr/bin/env python3
"""Experiment E: coherence-driven rates vs Lambda-like predictors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.common import (
        SZ,
        comm,
        kronN,
        op_on_site,
        normalize_vec,
        su2_from_axis_angle,
        plaquette,
        f_phys_from_plaquette,
        lindblad_dissipator,
        expect,
        normalize_rho,
        random_product_state_density,
        bond_xx_yy,
        partial_trace_site,
        coherence_offdiag,
        train_test_r2,
    )
except ImportError:
    from common import (
        SZ,
        comm,
        kronN,
        op_on_site,
        normalize_vec,
        su2_from_axis_angle,
        plaquette,
        f_phys_from_plaquette,
        lindblad_dissipator,
        expect,
        normalize_rho,
        random_product_state_density,
        bond_xx_yy,
        partial_trace_site,
        coherence_offdiag,
        train_test_r2,
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
    n_samples: int = 500,
    n_sites: int = 4,
    k_layers: int = 4,
    eta0_list: tuple[float, ...] = (0.5, 0.9, 1.5),
    gamma_deph: float = 0.35,
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

    f_site = {
        (x, k): f_phys_from_plaquette(plaquette(ux, umu, x, k, n_sites, k_layers))
        for x in xs
        for k in ks
    }

    d_chain = 2 ** n_sites
    d_total = k_layers * d_chain

    h_bonds = [bond_xx_yy(i, i + 1, n_sites, 1.0) for i in range(n_sites - 1)]
    h_chain = sum(h_bonds)

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

    rows = []
    for eta0 in eta0_list:
        for sample_id in range(n_samples):
            rr = np.random.default_rng(seed + int(1000 * eta0) + sample_id)
            weights = rr.dirichlet(alpha=np.ones(k_layers))

            rho_total = np.zeros((d_total, d_total), dtype=complex)
            for k in range(k_layers):
                rho_k = random_product_state_density(rr, n_sites)
                block = slice(k * d_chain, (k + 1) * d_chain)
                rho_total[block, block] = weights[k] * rho_k
            rho_total = normalize_rho(rho_total)

            coherence_layer = np.zeros(k_layers, dtype=float)
            for k in range(k_layers):
                block = slice(k * d_chain, (k + 1) * d_chain)
                rho_k = rho_total[block, block]
                p_k = float(np.real(np.trace(rho_k)))
                if p_k < 1e-14:
                    continue
                rho_k_cond = rho_k / p_k
                local_coh = [coherence_offdiag(partial_trace_site(rho_k_cond, x, n_sites)) for x in xs]
                coherence_layer[k] = float(np.mean(local_coh))

            eta = [eta0 * min(1.0, coherence_layer[k] / 0.5) for k in range(k_layers - 1)]
            jump_ops_mu = []
            for k in range(k_layers - 1):
                shift = np.zeros((k_layers, k_layers), dtype=complex)
                shift[k + 1, k] = 1.0
                jump_ops_mu.append(np.sqrt(eta[k]) * np.kron(shift, u_chain[k]))

            unitary_part = -1j * comm(h_total, rho_total)
            d_deph = lindblad_dissipator(rho_total, jump_ops_deph)
            d_mu = lindblad_dissipator(rho_total, jump_ops_mu)
            _ = unitary_part + d_deph + d_mu

            for k in range(k_layers):
                block = slice(k * d_chain, (k + 1) * d_chain)
                rho_k = rho_total[block, block]
                p_k = float(np.real(np.trace(rho_k)))
                if p_k < 1e-14:
                    continue

                rho_k_cond = rho_k / p_k
                site_red = {x: partial_trace_site(rho_k_cond, x, n_sites) for x in xs}
                site_coh = {x: coherence_offdiag(site_red[x]) for x in xs}
                site_lambda = {x: float(np.real(np.trace(f_site[(x, k)] @ site_red[x]))) for x in xs}

                for b in range(n_sites - 1):
                    obs = lift_to_layer(h_bonds[b], k)
                    src_mu = expect(obs, d_mu)
                    rows.append(
                        {
                            "eta0": float(eta0),
                            "sample": sample_id,
                            "k": k,
                            "bond": b,
                            "abs_src_mu": abs(src_mu),
                            "coh_layer_mean": float(coherence_layer[k]),
                            "coh_sum": float(site_coh[b] + site_coh[b + 1]),
                            "absLambda_sum": float(abs(site_lambda[b]) + abs(site_lambda[b + 1])),
                        }
                    )

    raw = pd.DataFrame(rows)

    summaries = []
    for eta0 in eta0_list:
        sub = raw[raw["eta0"] == eta0]
        y = sub["abs_src_mu"].to_numpy(float)
        x_coh = sub[["coh_layer_mean", "coh_sum"]].to_numpy(float)
        x_lam = sub[["absLambda_sum"]].to_numpy(float)

        corr_coh = float(np.corrcoef(sub["coh_sum"], y)[0, 1])
        corr_lam = float(np.corrcoef(sub["absLambda_sum"], y)[0, 1])

        r2_coh = train_test_r2(x_coh, y, seed=11)
        r2_lam = train_test_r2(x_lam, y, seed=19)

        summaries.append(
            {
                "eta0": float(eta0),
                "corr_coh_sum": corr_coh,
                "corr_absLambda_sum": corr_lam,
                "R2_coh_features": r2_coh,
                "R2_absLambda_only": r2_lam,
                "coherence_features_better": bool(r2_coh > r2_lam),
            }
        )

    summary = pd.DataFrame(summaries)

    outdir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(outdir / "experiment_E_raw.csv", index=False)
    summary.to_csv(outdir / "experiment_E_summary.csv", index=False)
    return raw, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_E", help="output directory")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--k-layers", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.35)
    args = parser.parse_args()

    outdir = Path(args.out)
    _, summary = run_experiment(
        outdir=outdir,
        seed=args.seed,
        n_samples=args.samples,
        n_sites=args.n_sites,
        k_layers=args.k_layers,
        gamma_deph=args.gamma,
    )
    print(summary.to_string(index=False))
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
