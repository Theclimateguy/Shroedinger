#!/usr/bin/env python3
"""Experiment L (Experiment 14): fermion+gauge embedding and Ward/continuity checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm, norm

try:
    from clean_experiments.common import I2, SX, SZ, expect, kronN, lindblad_dissipator, normalize_rho
except ImportError:
    from common import I2, SX, SZ, expect, kronN, lindblad_dissipator, normalize_rho


SM = np.array([[0, 1], [0, 0]], dtype=complex)


@dataclass(frozen=True)
class LParams:
    n_sites: int = 3
    t_hop: float = 0.8
    mass: float = 0.25
    omega_gauge: float = 0.35
    theta_base: float = 0.4
    theta_step: float = 0.2
    gamma_matter: float = 0.06
    gamma_gauge: float = 0.04


def _op_on_site(op: np.ndarray, site: int, n_qubits: int) -> np.ndarray:
    ops = [I2] * n_qubits
    ops[site] = op
    return kronN(ops)


def _jw_annihilation(site: int, n_sites: int) -> np.ndarray:
    ops: list[np.ndarray] = []
    for j in range(n_sites):
        if j < site:
            ops.append(SZ)
        elif j == site:
            ops.append(SM)
        else:
            ops.append(I2)
    return kronN(ops)


def _random_density(dim: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec = vec / np.linalg.norm(vec)
    return np.outer(vec, vec.conj())


def _build_system(params: LParams):
    n_sites = params.n_sites
    n_links = n_sites - 1
    d_m = 2**n_sites
    d_g = 2**n_links
    d_total = d_m * d_g

    id_m = np.eye(d_m, dtype=complex)
    id_g = np.eye(d_g, dtype=complex)

    c_ops = [_jw_annihilation(i, n_sites) for i in range(n_sites)]
    cd_ops = [c.conj().T for c in c_ops]
    n_ops = [cd_ops[i] @ c_ops[i] for i in range(n_sites)]

    sx_links = [_op_on_site(SX, link, n_links) for link in range(n_links)]
    sz_links = [_op_on_site(SZ, link, n_links) for link in range(n_links)]
    u_links = [expm(1j * (params.theta_base + params.theta_step * link) * sx_links[link]) for link in range(n_links)]

    q_local = [np.kron(n_ops[i], id_g) for i in range(n_sites)]
    q_total = sum(q_local)

    h_total = np.zeros((d_total, d_total), dtype=complex)
    for i in range(n_sites):
        h_total += params.mass * ((-1) ** i) * np.kron(n_ops[i], id_g)
    for link in range(n_links):
        hop = np.kron(cd_ops[link] @ c_ops[link + 1], u_links[link])
        h_total += -params.t_hop * (hop + hop.conj().T)
        h_total += 0.5 * params.omega_gauge * np.kron(id_m, sz_links[link])

    j_links = []
    for link in range(n_links):
        hop = np.kron(cd_ops[link] @ c_ops[link + 1], u_links[link])
        j_links.append(1j * params.t_hop * (hop - hop.conj().T))

    jump_matter = [np.sqrt(params.gamma_matter) * np.kron(n_ops[i], id_g) for i in range(n_sites)]
    jump_gauge = [np.sqrt(params.gamma_gauge) * np.kron(id_m, sz_links[link]) for link in range(n_links)]

    gauge_unitarity_max = float(max(norm(u @ u.conj().T - id_g, ord="fro") for u in u_links)) if u_links else 0.0

    return {
        "d_total": d_total,
        "h_total": h_total,
        "q_local": q_local,
        "q_total": q_total,
        "j_links": j_links,
        "jump_matter": jump_matter,
        "jump_gauge": jump_gauge,
        "gauge_unitarity_max": gauge_unitarity_max,
    }


def _rhs(
    rho: np.ndarray,
    h_total: np.ndarray,
    jump_matter: list[np.ndarray],
    jump_gauge: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unitary = -1j * (h_total @ rho - rho @ h_total)
    d_m = lindblad_dissipator(rho, jump_matter)
    d_g = lindblad_dissipator(rho, jump_gauge)
    return unitary + d_m + d_g, unitary, d_m, d_g


def _rk4_step(
    rho: np.ndarray,
    dt: float,
    h_total: np.ndarray,
    jump_matter: list[np.ndarray],
    jump_gauge: list[np.ndarray],
) -> np.ndarray:
    def f(state: np.ndarray) -> np.ndarray:
        return _rhs(state, h_total, jump_matter, jump_gauge)[0]

    k1 = f(rho)
    k2 = f(rho + 0.5 * dt * k1)
    k3 = f(rho + 0.5 * dt * k2)
    k4 = f(rho + dt * k3)
    nxt = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    nxt = 0.5 * (nxt + nxt.conj().T)
    return normalize_rho(nxt)


def run_experiment(
    outdir: Path,
    params: LParams = LParams(),
    *,
    seed: int = 20260229,
    n_samples: int = 12,
    n_steps: int = 80,
    dt: float = 0.02,
    write_csv: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    system = _build_system(params)

    h_total = system["h_total"]
    q_local = system["q_local"]
    q_total = system["q_total"]
    j_links = system["j_links"]
    jump_matter = system["jump_matter"]
    jump_gauge = system["jump_gauge"]
    d_total = int(system["d_total"])

    rows_local: list[dict[str, float | int]] = []
    rows_charge: list[dict[str, float | int]] = []
    rows_case: list[dict[str, float | int | bool]] = []

    comm_qh = float(norm(q_total @ h_total - h_total @ q_total, ord="fro"))
    max_abs_residual_global = 0.0
    max_charge_drift_global = 0.0

    for sample_id in range(n_samples):
        rng = np.random.default_rng(seed + sample_id * 17)
        rho = _random_density(d_total, rng)
        q0 = expect(q_total, rho)
        max_res_case = 0.0
        mean_acc_case = 0.0
        n_case = 0
        max_drift_case = 0.0

        for step in range(n_steps):
            t = step * dt
            dr, _, d_m, d_g = _rhs(rho, h_total, jump_matter, jump_gauge)

            q_tot_now = expect(q_total, rho)
            q_tot_dot = expect(q_total, dr)
            drift = abs(q_tot_now - q0)
            max_drift_case = max(max_drift_case, drift)

            rows_charge.append(
                {
                    "sample_id": int(sample_id),
                    "step": int(step),
                    "t": float(t),
                    "Q_total": float(q_tot_now),
                    "Q_total_dot": float(q_tot_dot),
                    "Q_total_drift": float(drift),
                }
            )

            for site in range(params.n_sites):
                dq_dt = expect(q_local[site], dr)
                j_out = expect(j_links[site], rho) if site < params.n_sites - 1 else 0.0
                j_in = expect(j_links[site - 1], rho) if site > 0 else 0.0
                div_j = j_out - j_in
                src_m = expect(q_local[site], d_m)
                src_g = expect(q_local[site], d_g)

                # Continuity convention: d q_i / dt - (j_out - j_in) = source_i.
                residual = dq_dt - div_j - (src_m + src_g)
                abs_residual = abs(residual)

                max_res_case = max(max_res_case, abs_residual)
                mean_acc_case += abs_residual
                n_case += 1

                rows_local.append(
                    {
                        "sample_id": int(sample_id),
                        "step": int(step),
                        "t": float(t),
                        "site": int(site),
                        "dq_dt": float(dq_dt),
                        "div_j": float(div_j),
                        "src_matter": float(src_m),
                        "src_gauge": float(src_g),
                        "residual": float(residual),
                        "abs_residual": float(abs_residual),
                    }
                )

            rho = _rk4_step(rho, dt, h_total, jump_matter, jump_gauge)

        mean_res_case = float(mean_acc_case / max(n_case, 1))
        max_abs_residual_global = max(max_abs_residual_global, max_res_case)
        max_charge_drift_global = max(max_charge_drift_global, max_drift_case)
        rows_case.append(
            {
                "sample_id": int(sample_id),
                "max_abs_continuity_residual": float(max_res_case),
                "mean_abs_continuity_residual": mean_res_case,
                "max_total_charge_drift": float(max_drift_case),
            }
        )

    local_df = pd.DataFrame(rows_local)
    charge_df = pd.DataFrame(rows_charge)
    case_df = pd.DataFrame(rows_case)

    mean_abs_residual_global = float(local_df["abs_residual"].mean()) if len(local_df) else float("nan")
    mean_charge_drift_global = float(case_df["max_total_charge_drift"].mean()) if len(case_df) else float("nan")

    ward_comm_tol = 1e-10
    continuity_tol = 1e-10
    charge_drift_tol = 1e-8

    summary_df = pd.DataFrame(
        [
            {
                "seed": int(seed),
                "n_samples": int(n_samples),
                "n_sites": int(params.n_sites),
                "n_links": int(params.n_sites - 1),
                "n_steps": int(n_steps),
                "dt": float(dt),
                "t_hop": float(params.t_hop),
                "mass": float(params.mass),
                "omega_gauge": float(params.omega_gauge),
                "theta_base": float(params.theta_base),
                "theta_step": float(params.theta_step),
                "gamma_matter": float(params.gamma_matter),
                "gamma_gauge": float(params.gamma_gauge),
                "comm_QH_fro": comm_qh,
                "gauge_link_unitarity_max": float(system["gauge_unitarity_max"]),
                "max_abs_continuity_residual": float(max_abs_residual_global),
                "mean_abs_continuity_residual": mean_abs_residual_global,
                "max_total_charge_drift": float(max_charge_drift_global),
                "mean_max_charge_drift_per_sample": mean_charge_drift_global,
                "pass_ward_comm": bool(comm_qh < ward_comm_tol),
                "pass_continuity": bool(max_abs_residual_global < continuity_tol),
                "pass_charge_drift": bool(max_charge_drift_global < charge_drift_tol),
                "pass_all": bool(
                    comm_qh < ward_comm_tol
                    and max_abs_residual_global < continuity_tol
                    and max_charge_drift_global < charge_drift_tol
                ),
            }
        ]
    )

    if write_csv:
        outdir.mkdir(parents=True, exist_ok=True)
        local_df.to_csv(outdir / "experiment_L_local_continuity_dataset.csv", index=False)
        charge_df.to_csv(outdir / "experiment_L_charge_timeseries.csv", index=False)
        case_df.to_csv(outdir / "experiment_L_case_summary.csv", index=False)
        summary_df.to_csv(outdir / "experiment_L_summary.csv", index=False)

    if verbose:
        s = summary_df.iloc[0]
        print(f"[L] ||[Q,H]||_F                 = {s['comm_QH_fro']:.3e}")
        print(f"[L] max continuity residual     = {s['max_abs_continuity_residual']:.3e}")
        print(f"[L] max total charge drift      = {s['max_total_charge_drift']:.3e}")
        print(f"[L] gauge-link unitarity maxdev = {s['gauge_link_unitarity_max']:.3e}")
        print(f"[L] pass_all                    = {bool(s['pass_all'])}")
        if write_csv:
            print(f"[L] saved: {outdir.resolve()}")

    return local_df, charge_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="clean_experiments/results/experiment_L_matter_fields", help="output directory")
    parser.add_argument("--seed", type=int, default=20260229)
    parser.add_argument("--samples", type=int, default=12, help="number of random initial states")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--n-sites", type=int, default=3)
    parser.add_argument("--t-hop", type=float, default=0.8)
    parser.add_argument("--mass", type=float, default=0.25)
    parser.add_argument("--omega-gauge", type=float, default=0.35)
    parser.add_argument("--theta-base", type=float, default=0.4)
    parser.add_argument("--theta-step", type=float, default=0.2)
    parser.add_argument("--gamma-matter", type=float, default=0.06)
    parser.add_argument("--gamma-gauge", type=float, default=0.04)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        samples = min(args.samples, 6)
        steps = min(args.steps, 40)
        dt = max(args.dt, 0.03)
    else:
        samples = args.samples
        steps = args.steps
        dt = args.dt

    params = LParams(
        n_sites=args.n_sites,
        t_hop=args.t_hop,
        mass=args.mass,
        omega_gauge=args.omega_gauge,
        theta_base=args.theta_base,
        theta_step=args.theta_step,
        gamma_matter=args.gamma_matter,
        gamma_gauge=args.gamma_gauge,
    )
    run_experiment(
        outdir=Path(args.out),
        params=params,
        seed=args.seed,
        n_samples=samples,
        n_steps=steps,
        dt=dt,
        write_csv=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
