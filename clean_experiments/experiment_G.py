#!/usr/bin/env python3
"""Experiment G: profile scan at fixed total phase (holonomy-controlled Lambda)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.common import (
        SX,
        build_u2_connection,
        curvature_discrete,
        hermitian_part,
        lambda_matter,
        qubit_state_density,
    )
except ImportError:
    from common import SX, build_u2_connection, curvature_discrete, hermitian_part, lambda_matter, qubit_state_density


def cumulative_trapezoid(y: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    return out


def run_experiment(
    outdir: Path,
    phi_target: float = 1.0,
    delta_mu: float = 2.0,
    k_layers: int = 128,
    beta: float = 0.1,
    r_amp: float = 0.4,
    alpha_state: float = np.pi / 4,
) -> pd.DataFrame:
    mu = np.linspace(0.0, delta_mu, k_layers + 1)
    dmu = delta_mu / k_layers

    mu_c = delta_mu / 2.0
    sigma_g = 0.35
    k_rg = 2.0

    raw_profiles = {
        "constant": np.ones_like(mu),
        "gaussian": np.exp(-((mu - mu_c) ** 2) / (2.0 * sigma_g**2)),
        "oscillating": np.cos(k_rg * mu),
    }

    rho0 = qubit_state_density(alpha_state)
    sx_expect = float(np.real(np.trace(SX @ rho0)))

    rows = []
    for name, shape in raw_profiles.items():
        shape_int = float(np.trapezoid(shape, mu))
        if abs(shape_int) < 1e-12:
            raise ValueError(f"Profile '{name}' has near-zero integral; cannot rescale to fixed phase")

        scale = phi_target / shape_int
        omega = scale * shape

        cum_angle = cumulative_trapezoid(omega, dmu)
        phi_total = float(cum_angle[-1])

        a_list = [build_u2_connection(beta, r_amp * np.cos(phi), r_amp * np.sin(phi)) for phi in cum_angle]

        lambda_num = 0.0
        for k in range(k_layers):
            f = curvature_discrete(a_list[k], a_list[k + 1], dmu)
            f_phys = hermitian_part(f)
            lambda_num += lambda_matter(f_phys, rho0) * dmu

        lambda_theory = r_amp * sx_expect * np.sin(phi_total)
        rel_error = abs(lambda_num - lambda_theory) / (abs(lambda_theory) + 1e-15)

        rows.append(
            {
                "profile": name,
                "scale": float(scale),
                "phi_total": phi_total,
                "phi_target": float(phi_target),
                "Lambda_numerical": float(lambda_num),
                "Lambda_theory": float(lambda_theory),
                "rel_error": float(rel_error),
                "rel_error_pct": float(100.0 * rel_error),
            }
        )

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "experiment_G_profile_fixed_phase.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_G", help="output directory")
    parser.add_argument("--phi-target", type=float, default=1.0)
    parser.add_argument("--delta-mu", type=float, default=2.0)
    parser.add_argument("--k-layers", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=float(np.pi / 4))
    args = parser.parse_args()

    outdir = Path(args.out)
    df = run_experiment(
        outdir=outdir,
        phi_target=args.phi_target,
        delta_mu=args.delta_mu,
        k_layers=args.k_layers,
        beta=args.beta,
        r_amp=args.r,
        alpha_state=args.alpha,
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
