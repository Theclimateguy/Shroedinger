#!/usr/bin/env python3
"""Experiment A: gauge invariance of Lambda and noncommutativity diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import norm

try:
    from clean_experiments.common import (
        normalize_vec,
        su2_from_axis_angle,
        random_su2,
        plaquette,
        f_phys_from_plaquette,
    )
except ImportError:
    from common import normalize_vec, su2_from_axis_angle, random_su2, plaquette, f_phys_from_plaquette


def axis_mu(x: int, k: int) -> np.ndarray:
    return normalize_vec(
        [
            np.sin(0.7 * x + 0.35 * k) + 0.3 * np.cos(0.2 * x - 0.6 * k),
            np.cos(0.5 * x - 0.55 * k) + 0.2 * np.sin(0.9 * x + 0.1 * k),
            np.sin(0.3 * x + 0.8 * k) + 0.4,
        ]
    )


def axis_x(x: int, k: int) -> np.ndarray:
    return normalize_vec(
        [
            np.cos(0.6 * x + 0.25 * k) + 0.1,
            np.sin(0.4 * x - 0.45 * k) + 0.2 * np.cos(0.9 * x + 0.15 * k),
            np.cos(0.2 * x + 0.75 * k) - 0.3,
        ]
    )


def run_experiment(
    outdir: Path,
    seed: int = 20260218,
    lx: int = 8,
    k_layers: int = 30,
    gauge_tests: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    xs = np.arange(lx)
    ks = np.arange(k_layers)

    omega = 0.22 + 0.40 * np.exp(-0.5 * ((ks - (k_layers - 1) / 2) / (0.16 * k_layers)) ** 2)
    angle_mu = 0.9 * omega / omega.max()
    angle_x = 0.55 * 0.9 * (0.6 + 0.4 * np.cos(2 * np.pi * ks / k_layers))

    ux: dict[tuple[int, int], np.ndarray] = {}
    umu: dict[tuple[int, int], np.ndarray] = {}
    for x in xs:
        for k in ks:
            umu[(x, k)] = su2_from_axis_angle(axis_mu(x, k), float(angle_mu[k]))
            ux[(x, k)] = su2_from_axis_angle(axis_x(x, k), float(angle_x[k]))

    comm_norms = []
    for x in xs:
        for k in range(k_layers - 1):
            comm_norms.append(norm(umu[(x, k)] @ umu[(x, k + 1)] - umu[(x, k + 1)] @ umu[(x, k)]))

    psi_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rho_plus = np.outer(psi_plus, psi_plus.conj())

    lambda0: dict[tuple[int, int], float] = {}
    for x in xs:
        for k in ks:
            p = plaquette(ux, umu, x, k, lx, k_layers)
            f_phys = f_phys_from_plaquette(p)
            lambda0[(x, k)] = float(np.real(np.trace(f_phys @ rho_plus)))

    points = np.array([(x, k) for x in xs for k in ks], dtype=int)
    picks = rng.choice(len(points), size=min(gauge_tests, len(points)), replace=False)

    deltas = []
    for idx in picks:
        x, k = points[idx]
        g = random_su2(rng)
        p = plaquette(ux, umu, int(x), int(k), lx, k_layers)
        f0 = f_phys_from_plaquette(p)
        f1 = g @ f0 @ g.conj().T
        r1 = g @ rho_plus @ g.conj().T
        lam1 = float(np.real(np.trace(f1 @ r1)))
        deltas.append(
            {
                "x": int(x),
                "k": int(k),
                "lambda_before": lambda0[(int(x), int(k))],
                "lambda_after": lam1,
                "delta_lambda": lam1 - lambda0[(int(x), int(k))],
                "abs_delta_lambda": abs(lam1 - lambda0[(int(x), int(k))]),
            }
        )

    per_point = pd.DataFrame(deltas)
    summary = pd.DataFrame(
        [
            {
                "seed": seed,
                "Lx": lx,
                "K_layers": k_layers,
                "mean_commutator_norm_adjacent_Umu": float(np.mean(comm_norms)),
                "max_commutator_norm_adjacent_Umu": float(np.max(comm_norms)),
                "mean_abs_delta_lambda_gauge": float(per_point["abs_delta_lambda"].mean()),
                "max_abs_delta_lambda_gauge": float(per_point["abs_delta_lambda"].max()),
                "gauge_invariant_within_tol": bool(per_point["abs_delta_lambda"].max() < 1e-12),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    per_point.to_csv(outdir / "experiment_A_gauge_samples.csv", index=False)
    summary.to_csv(outdir / "experiment_A_summary.csv", index=False)
    return per_point, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_A", help="output directory")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--lx", type=int, default=8)
    parser.add_argument("--k-layers", type=int, default=30)
    parser.add_argument("--gauge-tests", type=int, default=64)
    args = parser.parse_args()

    outdir = Path(args.out)
    _, summary = run_experiment(
        outdir=outdir,
        seed=args.seed,
        lx=args.lx,
        k_layers=args.k_layers,
        gauge_tests=args.gauge_tests,
    )
    print(summary.to_string(index=False))
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
