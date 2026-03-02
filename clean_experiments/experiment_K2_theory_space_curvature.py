#!/usr/bin/env python3
"""Experiment K2 (Experiment 13): theory-space curvature diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

try:
    from clean_experiments.common import I2, SX, SY, random_su2
except ImportError:
    from common import I2, SX, SY, random_su2


@dataclass(frozen=True)
class K2Params:
    alpha1: float = 0.3
    alpha2: float = 0.7
    theta1: float = 0.6
    theta2: float = 0.55
    u_min: float = 0.15
    u_max: float = 0.85
    phi_min: float = 0.12
    phi_max: float = float(np.pi / 2 - 0.05)


def _normalize(vec: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-15:
        raise ValueError("Degenerate state vector with near-zero norm.")
    return vec / nrm


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size < 3:
        return float("nan")
    if float(np.std(x)) < 1e-14 or float(np.std(y)) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _connections(phi: float, params: K2Params, gauge: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    a1 = params.alpha1 * I2 + params.theta1 * SY
    a2 = params.alpha2 * I2 + params.theta2 * (np.cos(phi) * SY + np.sin(phi) * SX)
    if gauge is not None:
        a1 = gauge @ a1 @ gauge.conj().T
        a2 = gauge @ a2 @ gauge.conj().T
    return a1, a2


def _evolution_operator(u: float, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    return expm(-1j * u * a2) @ expm(-1j * u * a1)


def _source_vector_from_connections(
    u: float,
    a1: np.ndarray,
    a2: np.ndarray,
    probes: list[np.ndarray],
) -> np.ndarray:
    uop = _evolution_operator(u=u, a1=a1, a2=a2)
    f = (a2 - a1) + (a1 @ a2 - a2 @ a1)
    values: list[float] = []
    for probe in probes:
        psi = _normalize(uop @ probe)
        rho = np.outer(psi, psi.conj())
        values.append(float(np.real(np.trace(f @ rho))))
    return np.asarray(values, dtype=float)


def _source_vector(
    u: float,
    phi: float,
    params: K2Params,
    probes: list[np.ndarray],
    gauge: np.ndarray | None = None,
) -> np.ndarray:
    a1, a2 = _connections(phi=phi, params=params, gauge=gauge)
    return _source_vector_from_connections(u=u, a1=a1, a2=a2, probes=probes)


def _response_metric(
    u: float,
    phi: float,
    params: K2Params,
    probes: list[np.ndarray],
    h: float,
    gauge: np.ndarray | None = None,
) -> np.ndarray:
    du = (_source_vector(u + h, phi, params, probes, gauge) - _source_vector(u - h, phi, params, probes, gauge)) / (
        2.0 * h
    )
    dphi = (_source_vector(u, phi + h, params, probes, gauge) - _source_vector(u, phi - h, params, probes, gauge)) / (
        2.0 * h
    )
    g11 = float(du @ du)
    g22 = float(dphi @ dphi)
    g12 = float(du @ dphi)
    return np.array([[g11, g12], [g12, g22]], dtype=float)


def _state(
    u: float,
    phi: float,
    params: K2Params,
    reference_state: np.ndarray,
    gauge: np.ndarray | None = None,
) -> np.ndarray:
    a1, a2 = _connections(phi=phi, params=params, gauge=gauge)
    ref = gauge @ reference_state if gauge is not None else reference_state
    return _normalize(_evolution_operator(u=u, a1=a1, a2=a2) @ ref)


def _fs_metric_and_omega(
    u: float,
    phi: float,
    params: K2Params,
    reference_state: np.ndarray,
    h: float,
    gauge: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    psi = _state(u, phi, params, reference_state, gauge)
    d_u = (_state(u + h, phi, params, reference_state, gauge) - _state(u - h, phi, params, reference_state, gauge)) / (
        2.0 * h
    )
    d_phi = (
        _state(u, phi + h, params, reference_state, gauge) - _state(u, phi - h, params, reference_state, gauge)
    ) / (2.0 * h)

    # Project out local phase direction to get the gauge-invariant QGT blocks.
    d_u -= psi * np.vdot(psi, d_u)
    d_phi -= psi * np.vdot(psi, d_phi)

    g11 = float(np.real(np.vdot(d_u, d_u)))
    g22 = float(np.real(np.vdot(d_phi, d_phi)))
    g12 = float(np.real(np.vdot(d_u, d_phi)))
    omega = float(2.0 * np.imag(np.vdot(d_u, d_phi)))
    return np.array([[g11, g12], [g12, g22]], dtype=float), omega


def _scalar_curvature_2d(
    metric_field: np.ndarray,
    u_values: np.ndarray,
    phi_values: np.ndarray,
    det_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    n_u = len(u_values)
    n_phi = len(phi_values)

    det = np.linalg.det(metric_field)
    inv = np.full_like(metric_field, np.nan, dtype=float)
    for iu in range(n_u):
        for jp in range(n_phi):
            if det[iu, jp] > det_eps:
                inv[iu, jp] = np.linalg.inv(metric_field[iu, jp])

    du = float(u_values[1] - u_values[0])
    dphi = float(phi_values[1] - phi_values[0])

    dg = np.full((n_u, n_phi, 2, 2, 2), np.nan, dtype=float)
    for iu in range(1, n_u - 1):
        for jp in range(1, n_phi - 1):
            dg[iu, jp, 0] = (metric_field[iu + 1, jp] - metric_field[iu - 1, jp]) / (2.0 * du)
            dg[iu, jp, 1] = (metric_field[iu, jp + 1] - metric_field[iu, jp - 1]) / (2.0 * dphi)

    gamma = np.full((n_u, n_phi, 2, 2, 2), np.nan, dtype=float)
    for iu in range(1, n_u - 1):
        for jp in range(1, n_phi - 1):
            if not np.isfinite(inv[iu, jp]).all():
                continue
            for k in range(2):
                for i in range(2):
                    for j in range(2):
                        s = 0.0
                        for l in range(2):
                            s += inv[iu, jp, k, l] * (
                                dg[iu, jp, i, j, l] + dg[iu, jp, j, i, l] - dg[iu, jp, l, i, j]
                            )
                        gamma[iu, jp, k, i, j] = 0.5 * s

    dgamma = np.full((n_u, n_phi, 2, 2, 2, 2), np.nan, dtype=float)
    for iu in range(2, n_u - 2):
        for jp in range(2, n_phi - 2):
            dgamma[iu, jp, 0] = (gamma[iu + 1, jp] - gamma[iu - 1, jp]) / (2.0 * du)
            dgamma[iu, jp, 1] = (gamma[iu, jp + 1] - gamma[iu, jp - 1]) / (2.0 * dphi)

    ricci_scalar = np.full((n_u, n_phi), np.nan, dtype=float)
    for iu in range(2, n_u - 2):
        for jp in range(2, n_phi - 2):
            if not np.isfinite(inv[iu, jp]).all() or not np.isfinite(gamma[iu, jp]).all():
                continue

            ricci = np.zeros((2, 2), dtype=float)
            for i in range(2):
                for j in range(2):
                    val = 0.0
                    for k in range(2):
                        val += dgamma[iu, jp, k, k, i, j]
                        val -= dgamma[iu, jp, j, k, i, k]
                        for l in range(2):
                            val += gamma[iu, jp, k, i, j] * gamma[iu, jp, l, k, l]
                            val -= gamma[iu, jp, k, i, l] * gamma[iu, jp, l, j, k]
                    ricci[i, j] = val
            ricci_scalar[iu, jp] = float(np.sum(inv[iu, jp] * ricci))

    return ricci_scalar, det


def _core_scan(
    params: K2Params,
    n_u: int,
    n_phi: int,
    h_deriv: float,
    probes: list[np.ndarray],
    reference_state: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    u_values = np.linspace(params.u_min, params.u_max, n_u)
    phi_values = np.linspace(params.phi_min, params.phi_max, n_phi)

    src_field = np.zeros((n_u, n_phi, len(probes)), dtype=float)
    fs_metric_field = np.zeros((n_u, n_phi, 2, 2), dtype=float)
    fs_omega_field = np.zeros((n_u, n_phi), dtype=float)
    resp_metric_field = np.zeros((n_u, n_phi, 2, 2), dtype=float)
    comm_norm = np.zeros(n_phi, dtype=float)

    for jp, phi in enumerate(phi_values):
        a1, a2 = _connections(phi=phi, params=params, gauge=None)
        comm_norm[jp] = float(np.linalg.norm(a1 @ a2 - a2 @ a1, ord="fro"))
        for iu, u in enumerate(u_values):
            src_field[iu, jp] = _source_vector_from_connections(u=u, a1=a1, a2=a2, probes=probes)
            fs_metric, fs_omega = _fs_metric_and_omega(
                u=u,
                phi=phi,
                params=params,
                reference_state=reference_state,
                h=h_deriv,
                gauge=None,
            )
            fs_metric_field[iu, jp] = fs_metric
            fs_omega_field[iu, jp] = fs_omega
            resp_metric_field[iu, jp] = _response_metric(
                u=u,
                phi=phi,
                params=params,
                probes=probes,
                h=h_deriv,
                gauge=None,
            )

    fs_ricci, fs_det = _scalar_curvature_2d(fs_metric_field, u_values, phi_values)
    resp_ricci, resp_det = _scalar_curvature_2d(resp_metric_field, u_values, phi_values)

    rows: list[dict[str, float]] = []
    for jp, phi in enumerate(phi_values):
        src_slice = src_field[:, jp, :]
        src_spread_phi = float(np.ptp(src_slice))
        src_std_phi = float(np.std(src_slice))

        fs_mask = np.isfinite(fs_ricci[:, jp])
        resp_mask = np.isfinite(resp_ricci[:, jp])

        rows.append(
            {
                "phi": float(phi),
                "sin_phi": float(np.sin(phi)),
                "comm_norm_fro": float(comm_norm[jp]),
                "source_spread": src_spread_phi,
                "source_std": src_std_phi,
                "fs_omega_mean_abs": float(np.mean(np.abs(fs_omega_field[:, jp]))),
                "fs_ricci_mean_abs": float(np.mean(np.abs(fs_ricci[:, jp][fs_mask]))) if np.any(fs_mask) else float("nan"),
                "fs_ricci_finite_frac": float(np.mean(fs_mask)),
                "fs_det_mean": float(np.mean(fs_det[:, jp])),
                "resp_det_mean": float(np.mean(resp_det[:, jp])),
                "resp_ricci_mean_abs": float(np.mean(np.abs(resp_ricci[:, jp][resp_mask])))
                if np.any(resp_mask)
                else float("nan"),
                "resp_ricci_finite_frac": float(np.mean(resp_mask)),
            }
        )

    scan_df = pd.DataFrame(rows)

    corr = {
        "corr_source_comm": _corr(scan_df["source_spread"].to_numpy(), scan_df["comm_norm_fro"].to_numpy()),
        "corr_source_sin": _corr(scan_df["source_spread"].to_numpy(), np.abs(scan_df["sin_phi"].to_numpy())),
        "corr_source_fs_omega": _corr(scan_df["source_spread"].to_numpy(), scan_df["fs_omega_mean_abs"].to_numpy()),
        "corr_source_resp_det": _corr(scan_df["source_spread"].to_numpy(), scan_df["resp_det_mean"].to_numpy()),
        "corr_source_resp_ricci": _corr(scan_df["source_spread"].to_numpy(), scan_df["resp_ricci_mean_abs"].to_numpy()),
    }
    return scan_df, corr, u_values, phi_values


def _gauge_invariance_checks(
    params: K2Params,
    probes: list[np.ndarray],
    reference_state: np.ndarray,
    h_deriv: float,
    n_checks: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for idx in range(n_checks):
        u = float(rng.uniform(params.u_min + 0.03, params.u_max - 0.03))
        phi = float(rng.uniform(params.phi_min + 0.03, params.phi_max - 0.03))
        g = random_su2(rng)
        probes_g = [g @ p for p in probes]

        src_base = _source_vector(u, phi, params, probes, gauge=None)
        src_gauge = _source_vector(u, phi, params, probes_g, gauge=g)
        src_diff = float(np.max(np.abs(src_base - src_gauge)))

        fs_metric_base, fs_omega_base = _fs_metric_and_omega(
            u=u,
            phi=phi,
            params=params,
            reference_state=reference_state,
            h=h_deriv,
            gauge=None,
        )
        fs_metric_gauge, fs_omega_gauge = _fs_metric_and_omega(
            u=u,
            phi=phi,
            params=params,
            reference_state=reference_state,
            h=h_deriv,
            gauge=g,
        )
        fs_metric_diff = float(np.max(np.abs(fs_metric_base - fs_metric_gauge)))
        fs_omega_diff = float(abs(fs_omega_base - fs_omega_gauge))

        resp_base = _response_metric(u=u, phi=phi, params=params, probes=probes, h=h_deriv, gauge=None)
        resp_gauge = _response_metric(u=u, phi=phi, params=params, probes=probes_g, h=h_deriv, gauge=g)
        resp_diff = float(np.max(np.abs(resp_base - resp_gauge)))

        rows.append(
            {
                "check_id": int(idx),
                "u": u,
                "phi": phi,
                "max_abs_source_diff": src_diff,
                "max_abs_fs_metric_diff": fs_metric_diff,
                "abs_fs_omega_diff": fs_omega_diff,
                "max_abs_response_metric_diff": resp_diff,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    outdir: Path,
    n_u: int = 25,
    n_phi: int = 25,
    h_deriv: float = 3e-3,
    params: K2Params = K2Params(),
    gauge_checks: int = 8,
    seed: int = 1234,
    write_csv: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probes = [
        np.array([1.0, 0.0], dtype=complex),
        np.array([0.0, 1.0], dtype=complex),
        np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0),
        np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0),
    ]
    reference_state = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)

    scan_df, corr_main, _, _ = _core_scan(
        params=params,
        n_u=n_u,
        n_phi=n_phi,
        h_deriv=h_deriv,
        probes=probes,
        reference_state=reference_state,
    )

    coarse_n_u = max(11, n_u // 2)
    coarse_n_phi = max(11, n_phi // 2)
    _, corr_coarse, _, _ = _core_scan(
        params=params,
        n_u=coarse_n_u,
        n_phi=coarse_n_phi,
        h_deriv=1.2 * h_deriv,
        probes=probes,
        reference_state=reference_state,
    )

    conv_rows = []
    for key in sorted(corr_main.keys()):
        main_val = float(corr_main[key]) if np.isfinite(corr_main[key]) else float("nan")
        coarse_val = float(corr_coarse[key]) if np.isfinite(corr_coarse[key]) else float("nan")
        delta = float(abs(main_val - coarse_val)) if np.isfinite(main_val) and np.isfinite(coarse_val) else float("nan")
        conv_rows.append(
            {
                "metric": key,
                "main_value": main_val,
                "coarse_value": coarse_val,
                "abs_delta": delta,
            }
        )
    convergence_df = pd.DataFrame(conv_rows)

    gauge_df = _gauge_invariance_checks(
        params=params,
        probes=probes,
        reference_state=reference_state,
        h_deriv=h_deriv,
        n_checks=gauge_checks,
        seed=seed,
    )

    gauge_max_source = float(gauge_df["max_abs_source_diff"].max())
    gauge_max_fs_metric = float(gauge_df["max_abs_fs_metric_diff"].max())
    gauge_max_fs_omega = float(gauge_df["abs_fs_omega_diff"].max())
    gauge_max_resp_metric = float(gauge_df["max_abs_response_metric_diff"].max())

    finite_conv = convergence_df["abs_delta"].to_numpy(dtype=float)
    finite_conv = finite_conv[np.isfinite(finite_conv)]
    max_corr_delta = float(np.max(finite_conv)) if finite_conv.size else float("nan")

    avg_resp_ricci_finite = float(scan_df["resp_ricci_finite_frac"].mean())
    avg_fs_ricci_finite = float(scan_df["fs_ricci_finite_frac"].mean())

    pass_source_comm = bool(np.isfinite(corr_main["corr_source_comm"]) and corr_main["corr_source_comm"] >= 0.95)
    pass_source_fs_omega = bool(
        np.isfinite(corr_main["corr_source_fs_omega"]) and abs(corr_main["corr_source_fs_omega"]) >= 0.90
    )
    pass_gauge = bool(
        gauge_max_source < 1e-8 and gauge_max_fs_metric < 1e-8 and gauge_max_fs_omega < 1e-8 and gauge_max_resp_metric < 1e-8
    )
    pass_convergence = bool(np.isfinite(max_corr_delta) and max_corr_delta < 0.15)
    pass_curvature_defined = bool(avg_resp_ricci_finite > 0.55 and avg_fs_ricci_finite > 0.55)
    pass_all = bool(pass_source_comm and pass_source_fs_omega and pass_gauge and pass_convergence and pass_curvature_defined)

    summary_df = pd.DataFrame(
        [
            {
                "n_u": int(n_u),
                "n_phi": int(n_phi),
                "h_deriv": float(h_deriv),
                "corr_source_comm": float(corr_main["corr_source_comm"]),
                "corr_source_sin": float(corr_main["corr_source_sin"]),
                "corr_source_fs_omega": float(corr_main["corr_source_fs_omega"]),
                "corr_source_resp_det": float(corr_main["corr_source_resp_det"]),
                "corr_source_resp_ricci": float(corr_main["corr_source_resp_ricci"]),
                "avg_resp_ricci_finite_frac": avg_resp_ricci_finite,
                "avg_fs_ricci_finite_frac": avg_fs_ricci_finite,
                "gauge_max_source_diff": gauge_max_source,
                "gauge_max_fs_metric_diff": gauge_max_fs_metric,
                "gauge_max_fs_omega_diff": gauge_max_fs_omega,
                "gauge_max_response_metric_diff": gauge_max_resp_metric,
                "max_corr_delta_coarse": max_corr_delta,
                "pass_source_comm": pass_source_comm,
                "pass_source_fs_omega": pass_source_fs_omega,
                "pass_gauge": pass_gauge,
                "pass_convergence": pass_convergence,
                "pass_curvature_defined": pass_curvature_defined,
                "pass_all": pass_all,
            }
        ]
    )

    if write_csv:
        outdir.mkdir(parents=True, exist_ok=True)
        scan_df.to_csv(outdir / "experiment_K2_scan.csv", index=False)
        gauge_df.to_csv(outdir / "experiment_K2_gauge_checks.csv", index=False)
        convergence_df.to_csv(outdir / "experiment_K2_convergence.csv", index=False)
        summary_df.to_csv(outdir / "experiment_K2_summary.csv", index=False)

    if verbose:
        row = summary_df.iloc[0]
        print(f"[K2] corr(source, comm)      = {row['corr_source_comm']:.6f}")
        print(f"[K2] corr(source, |Omega_FS|)= {row['corr_source_fs_omega']:.6f}")
        print(f"[K2] corr(source, det G_resp)= {row['corr_source_resp_det']:.6f}")
        print(f"[K2] gauge max diff (source) = {row['gauge_max_source_diff']:.3e}")
        print(f"[K2] gauge max diff (resp G) = {row['gauge_max_response_metric_diff']:.3e}")
        print(f"[K2] max coarse delta        = {row['max_corr_delta_coarse']:.6f}")
        print(f"[K2] pass_all                = {bool(row['pass_all'])}")
        if write_csv:
            print(f"[K2] saved: {outdir.resolve()}")

    return scan_df, gauge_df, convergence_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_K2_theory_space_curvature",
        help="output directory",
    )
    parser.add_argument("--n-u", type=int, default=25, help="grid size along transport depth u")
    parser.add_argument("--n-phi", type=int, default=25, help="grid size along RG rotation angle phi")
    parser.add_argument("--h-deriv", type=float, default=3e-3, help="finite-difference step for derivatives")
    parser.add_argument("--gauge-checks", type=int, default=8, help="number of random gauge checks")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quick", action="store_true", help="quick smoke configuration")
    args = parser.parse_args()

    if args.quick:
        n_u = min(args.n_u, 17)
        n_phi = min(args.n_phi, 17)
        h_deriv = max(args.h_deriv, 4e-3)
        gauge_checks = min(args.gauge_checks, 4)
    else:
        n_u = args.n_u
        n_phi = args.n_phi
        h_deriv = args.h_deriv
        gauge_checks = args.gauge_checks

    run_experiment(
        outdir=Path(args.out),
        n_u=n_u,
        n_phi=n_phi,
        h_deriv=h_deriv,
        params=K2Params(),
        gauge_checks=gauge_checks,
        seed=args.seed,
        write_csv=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
