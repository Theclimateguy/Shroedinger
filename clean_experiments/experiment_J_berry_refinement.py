#!/usr/bin/env python3
"""Experiment J: Berry phase refinement (risk R3) under mu-step convergence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ham_from_angles(theta: float, phi: float) -> np.ndarray:
    # H = n·sigma for |n|=1 with n=(sinθ cosφ, sinθ sinφ, cosθ)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return nx * sx + ny * sy + nz * sz


def _lowest_eigenvector(h: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(h)
    idx = int(np.argmin(vals))
    v = vecs[:, idx]
    return v / np.linalg.norm(v)


def _berry_phase_discrete(theta0: float, n_steps: int) -> float:
    phis = np.linspace(0.0, 2.0 * np.pi, n_steps + 1)
    vecs = [_lowest_eigenvector(_ham_from_angles(theta0, phi)) for phi in phis]

    prod = 1.0 + 0.0j
    for k in range(n_steps):
        ov = np.vdot(vecs[k], vecs[k + 1])
        if abs(ov) <= 1e-15:
            continue
        prod *= ov / abs(ov)
    closing = np.vdot(vecs[-1], vecs[0])
    if abs(closing) > 1e-15:
        prod *= closing / abs(closing)

    gamma = -float(np.angle(prod))
    while gamma <= -np.pi:
        gamma += 2.0 * np.pi
    while gamma > np.pi:
        gamma -= 2.0 * np.pi
    return gamma


def _wrapped_angle_distance(a: float, b: float) -> float:
    return float(abs(np.angle(np.exp(1j * (a - b)))))


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    vals = [v for v in vals if v >= 8]
    return sorted(set(vals))


def run_experiment(
    outdir: Path,
    *,
    theta0: float = np.pi / 2.0,
    n_steps_scan: str = "16,24,32,48,64,96,128,192,256,384,512,768,1024",
    target_phase: float = -np.pi,
    err_tol: float = 2e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ns = _parse_int_list(n_steps_scan)
    rows = []
    for n_steps in ns:
        gamma = _berry_phase_discrete(theta0, n_steps)
        rows.append(
            {
                "n_steps": int(n_steps),
                "dphi": float(2.0 * np.pi / n_steps),
                "berry_phase": float(gamma),
                "target_phase": float(target_phase),
                "abs_error_to_target": float(_wrapped_angle_distance(gamma, target_phase)),
            }
        )

    df = pd.DataFrame(rows).sort_values("n_steps").reset_index(drop=True)
    finest = df.iloc[-1]
    tail = df.tail(min(5, len(df)))
    monotonic_tail = bool(np.all(np.diff(tail["abs_error_to_target"].to_numpy()) <= 1e-12))

    summary = pd.DataFrame(
        [
            {
                "theta0": float(theta0),
                "target_phase": float(target_phase),
                "n_points": int(len(df)),
                "finest_n_steps": int(finest["n_steps"]),
                "finest_berry_phase": float(finest["berry_phase"]),
                "finest_abs_error": float(finest["abs_error_to_target"]),
                "tail_monotonic_nonincreasing": monotonic_tail,
                "pass_err_tol": bool(float(finest["abs_error_to_target"]) <= err_tol),
                "pass_all": bool(float(finest["abs_error_to_target"]) <= err_tol),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "experiment_J_berry_scan.csv", index=False)
    summary.to_csv(outdir / "experiment_J_summary.csv", index=False)
    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_J_berry_refinement", help="output directory")
    parser.add_argument("--theta0", type=float, default=float(np.pi / 2.0))
    parser.add_argument("--n-steps-scan", default="16,24,32,48,64,96,128,192,256,384,512,768,1024")
    parser.add_argument("--target-phase", type=float, default=float(-np.pi))
    parser.add_argument("--err-tol", type=float, default=2e-3)
    args = parser.parse_args()

    df, summary = run_experiment(
        outdir=Path(args.out),
        theta0=args.theta0,
        n_steps_scan=args.n_steps_scan,
        target_phase=args.target_phase,
        err_tol=args.err_tol,
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
