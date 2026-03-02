#!/usr/bin/env python3
"""Experiment I: continuum extrapolation of balance residual from microscopic dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.common import (
        SZ,
        bond_xx_yy,
        comm,
        expect,
        kronN,
        lindblad_dissipator,
        normalize_rho,
        op_on_site,
        random_product_state_density,
    )
except ImportError:
    from common import (
        SZ,
        bond_xx_yy,
        comm,
        expect,
        kronN,
        lindblad_dissipator,
        normalize_rho,
        op_on_site,
        random_product_state_density,
    )


def _parse_float_list(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    return sorted(set(vals))


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    return sorted(set(vals))


def _build_model(n_sites: int, k_layers: int, gamma_deph: float, eta0: float):
    d_chain = 2**n_sites
    d_total = k_layers * d_chain

    h_bonds = [bond_xx_yy(i, i + 1, n_sites, 1.0) for i in range(n_sites - 1)]
    h_chain = sum(h_bonds)
    j_bonds = [1j * comm(h_bonds[i], h_bonds[i + 1]) for i in range(n_sites - 2)]

    layer_basis = np.eye(k_layers, dtype=complex)
    projectors = [layer_basis[[k]].T @ layer_basis[[k]] for k in range(k_layers)]

    def lift(op_chain: np.ndarray, k: int) -> np.ndarray:
        return np.kron(projectors[k], op_chain)

    h_total = sum(lift(h_chain, k) for k in range(k_layers))

    jump_ops_deph = [
        np.sqrt(gamma_deph) * lift(op_on_site(SZ, site, n_sites), k)
        for k in range(k_layers)
        for site in range(n_sites)
    ]

    # Vertical channel as nearest-layer jumps with identity transport on chain.
    jump_ops_mu = []
    for k in range(k_layers - 1):
        shift = np.zeros((k_layers, k_layers), dtype=complex)
        shift[k + 1, k] = 1.0
        jump_ops_mu.append(np.sqrt(eta0) * np.kron(shift, np.eye(d_chain, dtype=complex)))

    observables = {(k, b): lift(h_bonds[b], k) for k in range(k_layers) for b in range(n_sites - 1)}
    j_obs = {(k, i): lift(j_bonds[i], k) for k in range(k_layers) for i in range(n_sites - 2)}
    return h_total, jump_ops_deph, jump_ops_mu, observables, j_obs, d_chain, d_total


def _drho_dt(
    rho: np.ndarray,
    h_total: np.ndarray,
    jump_ops_deph: list[np.ndarray],
    jump_ops_mu: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unitary = -1j * comm(h_total, rho)
    d_deph = lindblad_dissipator(rho, jump_ops_deph)
    d_mu = lindblad_dissipator(rho, jump_ops_mu)
    return unitary + d_deph + d_mu, d_deph, d_mu


def _rk4_step(
    rho: np.ndarray,
    dt: float,
    h_total: np.ndarray,
    jump_ops_deph: list[np.ndarray],
    jump_ops_mu: list[np.ndarray],
) -> np.ndarray:
    def rhs(state: np.ndarray) -> np.ndarray:
        dr, _, _ = _drho_dt(state, h_total, jump_ops_deph, jump_ops_mu)
        return dr

    k1 = rhs(rho)
    k2 = rhs(rho + 0.5 * dt * k1)
    k3 = rhs(rho + 0.5 * dt * k2)
    k4 = rhs(rho + dt * k3)
    nxt = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    nxt = 0.5 * (nxt + nxt.conj().T)
    return normalize_rho(nxt)


def _fit_surface(dt: np.ndarray, dmu: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.column_stack([np.ones_like(dt), dt, dmu, dt * dmu])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-15))
    return beta, r2


def _run_case(
    *,
    seed: int,
    n_sites: int,
    k_layers: int,
    gamma_deph: float,
    eta0: float,
    dt: float,
    n_steps: int,
    delta_mu_span: float,
    case_id: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    (
        h_total,
        jump_ops_deph,
        jump_ops_mu,
        observables,
        j_obs,
        d_chain,
        d_total,
    ) = _build_model(n_sites, k_layers, gamma_deph, eta0)

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(alpha=np.ones(k_layers))
    rho = np.zeros((d_total, d_total), dtype=complex)
    for k in range(k_layers):
        block = slice(k * d_chain, (k + 1) * d_chain)
        rho_k = random_product_state_density(rng, n_sites)
        rho[block, block] = weights[k] * rho_k
    rho = normalize_rho(rho)

    rows: list[dict[str, float | int]] = []
    max_abs_residual = 0.0
    sum_abs_residual = 0.0
    n_res = 0

    for step in range(n_steps):
        t = step * dt
        dr, d_deph, d_mu = _drho_dt(rho, h_total, jump_ops_deph, jump_ops_mu)
        rho_next = _rk4_step(rho, dt, h_total, jump_ops_deph, jump_ops_mu)

        for k in range(k_layers):
            j_vals = [expect(j_obs[(k, i)], rho) for i in range(n_sites - 2)]
            for b in range(n_sites - 1):
                obs = observables[(k, b)]
                o_now = expect(obs, rho)
                o_next = expect(obs, rho_next)
                d_o_dt_fd = (o_next - o_now) / dt

                j_right = j_vals[b] if b <= n_sites - 3 else 0.0
                j_left = j_vals[b - 1] if b - 1 >= 0 else 0.0
                div_x = j_right - j_left

                src_deph = expect(obs, d_deph)
                src_mu = expect(obs, d_mu)
                residual = d_o_dt_fd + div_x - (src_deph + src_mu)
                abs_residual = abs(residual)

                max_abs_residual = max(max_abs_residual, abs_residual)
                sum_abs_residual += abs_residual
                n_res += 1

                rows.append(
                    {
                        "case_id": int(case_id),
                        "step": int(step),
                        "t": float(t),
                        "k": int(k),
                        "bond": int(b),
                        "dO_dt_fd": float(d_o_dt_fd),
                        "div_x_j": float(div_x),
                        "src_deph": float(src_deph),
                        "src_mu": float(src_mu),
                        "residual": float(residual),
                        "abs_residual": float(abs_residual),
                    }
                )

        rho = rho_next

    mean_abs_residual = float(sum_abs_residual / max(n_res, 1))
    dmu = float(delta_mu_span / k_layers)
    summary = {
        "case_id": int(case_id),
        "seed": int(seed),
        "n_sites": int(n_sites),
        "k_layers": int(k_layers),
        "dmu": dmu,
        "dt": float(dt),
        "n_steps": int(n_steps),
        "gamma_deph": float(gamma_deph),
        "eta0": float(eta0),
        "max_abs_residual": float(max_abs_residual),
        "mean_abs_residual": mean_abs_residual,
    }
    return pd.DataFrame(rows), summary


def run_experiment(
    outdir: Path,
    *,
    seed: int = 20260223,
    n_sites: int = 4,
    k_values: str = "3,4,5,6",
    dt_values: str = "0.12,0.06,0.03,0.015",
    n_steps: int = 60,
    gamma_deph: float = 0.35,
    eta0: float = 0.9,
    delta_mu_span: float = 1.0,
    intercept_tol: float = 3e-3,
    intercept_rel_tol: float = 8e-2,
    write_dataset: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ks = _parse_int_list(k_values)
    dts = _parse_float_list(dt_values)
    t_final = float(n_steps * min(dts))

    all_rows = []
    case_rows = []
    case_id = 0
    for k_idx, k_layers in enumerate(ks):
        seed_for_k = seed + 1000 * k_idx
        for dt in dts:
            steps_for_dt = max(1, int(round(t_final / dt)))
            df_case, case_summary = _run_case(
                seed=seed_for_k,
                n_sites=n_sites,
                k_layers=k_layers,
                gamma_deph=gamma_deph,
                eta0=eta0,
                dt=dt,
                n_steps=steps_for_dt,
                delta_mu_span=delta_mu_span,
                case_id=case_id,
            )
            all_rows.append(df_case)
            case_rows.append(case_summary)
            case_id += 1

    dataset = pd.concat(all_rows, ignore_index=True)
    case_df = pd.DataFrame(case_rows).sort_values(["k_layers", "dt"]).reset_index(drop=True)

    beta_max, r2_max = _fit_surface(
        case_df["dt"].to_numpy(dtype=float),
        case_df["dmu"].to_numpy(dtype=float),
        case_df["max_abs_residual"].to_numpy(dtype=float),
    )
    beta_mean, r2_mean = _fit_surface(
        case_df["dt"].to_numpy(dtype=float),
        case_df["dmu"].to_numpy(dtype=float),
        case_df["mean_abs_residual"].to_numpy(dtype=float),
    )

    extrapolation = pd.DataFrame(
        [
            {
                "seed": int(seed),
                "n_sites": int(n_sites),
                "n_cases": int(len(case_df)),
                "t_final": float(t_final),
                "gamma_deph": float(gamma_deph),
                "eta0": float(eta0),
                "intercept_max_abs_residual": float(beta_max[0]),
                "coef_dt_max_abs_residual": float(beta_max[1]),
                "coef_dmu_max_abs_residual": float(beta_max[2]),
                "coef_dt_dmu_max_abs_residual": float(beta_max[3]),
                "fit_r2_max_abs_residual": float(r2_max),
                "intercept_mean_abs_residual": float(beta_mean[0]),
                "coef_dt_mean_abs_residual": float(beta_mean[1]),
                "coef_dmu_mean_abs_residual": float(beta_mean[2]),
                "coef_dt_dmu_mean_abs_residual": float(beta_mean[3]),
                "fit_r2_mean_abs_residual": float(r2_mean),
                "max_residual_overall": float(case_df["max_abs_residual"].max()),
                "mean_residual_overall": float(case_df["mean_abs_residual"].mean()),
                "intercept_abs": float(abs(beta_max[0])),
                "intercept_abs_tol": float(intercept_tol),
                "intercept_rel_tol": float(intercept_rel_tol),
                "intercept_scaled_tol": float(
                    max(intercept_tol, intercept_rel_tol * float(case_df["max_abs_residual"].max()))
                ),
                "pass_intercept_tol": bool(abs(beta_max[0]) <= intercept_tol),
                "pass_intercept_scaled_tol": bool(
                    abs(beta_max[0])
                    <= max(intercept_tol, intercept_rel_tol * float(case_df["max_abs_residual"].max()))
                ),
                "pass_all": bool(
                    abs(beta_max[0])
                    <= max(intercept_tol, intercept_rel_tol * float(case_df["max_abs_residual"].max()))
                ),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    if write_dataset:
        dataset.to_csv(outdir / "experiment_I_fd_balance_dataset.csv", index=False)
    case_df.to_csv(outdir / "experiment_I_case_summary.csv", index=False)
    extrapolation.to_csv(outdir / "experiment_I_extrapolation.csv", index=False)
    return dataset, case_df, extrapolation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_I_continuum_conservation", help="output directory")
    parser.add_argument("--seed", type=int, default=20260223)
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--k-values", default="3,4,5,6")
    parser.add_argument("--dt-values", default="0.12,0.06,0.03,0.015")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--eta0", type=float, default=0.9)
    parser.add_argument("--delta-mu-span", type=float, default=1.0)
    parser.add_argument("--intercept-tol", type=float, default=3e-3)
    parser.add_argument("--intercept-rel-tol", type=float, default=8e-2)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="skip writing the full per-step dataset",
    )
    args = parser.parse_args()

    _, case_df, ext_df = run_experiment(
        outdir=Path(args.out),
        seed=args.seed,
        n_sites=args.n_sites,
        k_values=args.k_values,
        dt_values=args.dt_values,
        n_steps=args.n_steps,
        gamma_deph=args.gamma,
        eta0=args.eta0,
        delta_mu_span=args.delta_mu_span,
        intercept_tol=args.intercept_tol,
        intercept_rel_tol=args.intercept_rel_tol,
        write_dataset=not args.summary_only,
    )

    print("Case summary (top 12 by max residual):")
    print(
        case_df.sort_values("max_abs_residual", ascending=False)
        .head(min(12, len(case_df)))
        .to_string(index=False, float_format=lambda x: f"{x:.6e}")
    )
    print("\nContinuum extrapolation:")
    print(ext_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
