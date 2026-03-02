#!/usr/bin/env python3
"""Experiment F: sinusoidal law and discretization convergence in the U(2) toy model."""

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


def run_experiment(
    outdir: Path,
    omega: float = 0.3,
    delta_mu: float = 2.0,
    beta: float = 0.1,
    r_amp: float = 0.4,
    alpha_state: float = np.pi / 4,
    k_values: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
) -> pd.DataFrame:
    rho0 = qubit_state_density(alpha_state)
    sx_expect = float(np.real(np.trace(SX @ rho0)))

    rows = []
    for k_layers in k_values:
        dmu = delta_mu / k_layers
        mu = np.linspace(0.0, delta_mu, k_layers + 1)

        a_list = []
        for m in mu:
            angle = omega * m
            a_list.append(build_u2_connection(beta, r_amp * np.cos(angle), r_amp * np.sin(angle)))

        lambda_num = 0.0
        for k in range(k_layers):
            f = curvature_discrete(a_list[k], a_list[k + 1], dmu)
            f_phys = hermitian_part(f)
            lambda_num += lambda_matter(f_phys, rho0) * dmu

        lambda_theory = r_amp * sx_expect * np.sin(omega * delta_mu)
        rel_error = abs(lambda_num - lambda_theory) / (abs(lambda_theory) + 1e-15)

        rows.append(
            {
                "K_layers": int(k_layers),
                "dmu": float(dmu),
                "Lambda_numerical": float(lambda_num),
                "Lambda_theory": float(lambda_theory),
                "rel_error": float(rel_error),
                "rel_error_pct": float(100.0 * rel_error),
            }
        )

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "experiment_F_sinusoidal_convergence.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_F", help="output directory")
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--delta-mu", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=float(np.pi / 4))
    args = parser.parse_args()

    outdir = Path(args.out)
    df = run_experiment(
        outdir=outdir,
        omega=args.omega,
        delta_mu=args.delta_mu,
        beta=args.beta,
        r_amp=args.r,
        alpha_state=args.alpha,
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
