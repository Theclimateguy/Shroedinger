#!/usr/bin/env python3
"""Experiment F1: fractal emergence from balance competition in a two-scale GKSL toy model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

try:
    from clean_experiments.common import (
        SX,
        SY,
        SZ,
        bond_xx_yy,
        comm,
        kronN,
        lindblad_dissipator,
        op_on_site,
    )
except ImportError:
    from common import SX, SY, SZ, bond_xx_yy, comm, kronN, lindblad_dissipator, op_on_site


SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
X_SCALE = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SHIFT_01 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SHIFT_10 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)


@dataclass
class F1Config:
    eps_list: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 0.70, 1.00, 1.40, 3.00, 10.00)
    n_sites: int = 3
    dt: float = 0.003
    n_steps: int = 2200
    burn_in: int = 600
    sample_every: int = 12
    epsilon_internal_shift: float = 3.8
    omega_drive_ref: float = 0.8
    gamma_rel_ref: float = 1.6
    vertical_unitary_coeff: float = 0.07
    vertical_drift_coeff: float = 0.09
    drift_asymmetry: float = 0.75
    drift_rotation_angle: float = 0.8
    local_layer_detuning: float = 0.2
    source_offdiag_weight: float = 0.25
    jump_dephasing_weight: float = 1.0
    jump_relax_weight: float = 0.2
    flow_floor: float = 5e-5
    source_floor: float = 1e-3
    integer_tol: float = 0.10
    noninteger_tol: float = 0.10
    epsilon_star_window: tuple[float, float] = (0.7, 1.4)


def lift_to_layer(layer_op: np.ndarray, chain_op: np.ndarray) -> np.ndarray:
    return np.kron(layer_op, chain_op)


def stabilize_density(rho: np.ndarray) -> np.ndarray:
    herm = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(herm)
    vals = np.clip(vals, 0.0, None)
    if float(np.sum(vals)) <= 1e-15:
        vals[0] = 1.0
    proj = vecs @ np.diag(vals) @ vecs.conj().T
    return proj / np.trace(proj)


def entropy_effective_rank_log2(rho: np.ndarray, tol: float = 1e-15) -> float:
    vals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    vals = np.clip(vals, tol, 1.0)
    s = float(-np.sum(vals * np.log(vals)))
    return float(np.log2(np.exp(s)))


def build_static_geometry(cfg: F1Config) -> dict[str, np.ndarray]:
    d_chain = 2**cfg.n_sites

    h_chain = sum(bond_xx_yy(i, i + 1, cfg.n_sites, 1.0) for i in range(cfg.n_sites - 1))
    field_x = sum(op_on_site(SX, i, cfg.n_sites) for i in range(cfg.n_sites)) / float(cfg.n_sites)
    field_z = sum(op_on_site(SZ, i, cfg.n_sites) for i in range(cfg.n_sites)) / float(cfg.n_sites)

    observable_0 = lift_to_layer(P0, h_chain)
    observable_1 = lift_to_layer(P1, h_chain)

    f_chain = field_x + cfg.source_offdiag_weight * field_z
    drift_rotation = expm(-1j * cfg.drift_rotation_angle * field_x)

    psi_chain = np.zeros(d_chain, dtype=complex)
    psi_chain[0] = 1.0 / np.sqrt(2.0)
    psi_chain[-1] = 1.0 / np.sqrt(2.0)
    psi_scale = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    psi0 = np.kron(psi_scale, psi_chain)
    rho0 = np.outer(psi0, psi0.conj())

    return {
        "d_chain": d_chain,
        "h_chain": h_chain,
        "field_x": field_x,
        "field_z": field_z,
        "observable_0": observable_0,
        "observable_1": observable_1,
        "f_chain": f_chain,
        "drift_rotation": drift_rotation,
        "rho0": rho0,
    }


def run_one_epsilon(epsilon: float, cfg: F1Config, static: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, float]]:
    d_chain = int(static["d_chain"])
    h_chain = static["h_chain"]
    field_x = static["field_x"]
    field_z = static["field_z"]
    observable_0 = static["observable_0"]
    observable_1 = static["observable_1"]
    f_chain = static["f_chain"]
    drift_rotation = static["drift_rotation"]
    rho = static["rho0"].copy()

    eps_internal = float(epsilon * cfg.epsilon_internal_shift)
    omega_drive = float(cfg.omega_drive_ref * np.sqrt(eps_internal))
    gamma_rel = float(np.clip(cfg.gamma_rel_ref / np.sqrt(eps_internal), 0.05, 12.0))
    gamma_drift = float(cfg.vertical_drift_coeff / eps_internal)

    h0 = h_chain + omega_drive * field_x + cfg.local_layer_detuning * field_z
    h1 = h_chain + omega_drive * field_x - cfg.local_layer_detuning * field_z
    h_local = lift_to_layer(P0, h0) + lift_to_layer(P1, h1)
    h_vertical = (cfg.vertical_unitary_coeff * eps_internal) * lift_to_layer(X_SCALE, field_x)

    rel_jumps = []
    for site in range(cfg.n_sites):
        rel_jumps.extend(
            [
                np.sqrt(cfg.jump_dephasing_weight * gamma_rel) * lift_to_layer(P0, op_on_site(SZ, site, cfg.n_sites)),
                np.sqrt(cfg.jump_dephasing_weight * gamma_rel) * lift_to_layer(P1, op_on_site(SZ, site, cfg.n_sites)),
                np.sqrt(cfg.jump_relax_weight * gamma_rel) * lift_to_layer(P0, op_on_site(SM, site, cfg.n_sites)),
                np.sqrt(cfg.jump_relax_weight * gamma_rel) * lift_to_layer(P1, op_on_site(SM, site, cfg.n_sites)),
            ]
        )

    drift_jumps = [
        np.sqrt(gamma_drift) * lift_to_layer(SHIFT_10, drift_rotation),
        np.sqrt(gamma_drift * cfg.drift_asymmetry) * lift_to_layer(SHIFT_01, drift_rotation.conj().T),
    ]

    rows: list[dict[str, float]] = []
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    for step in range(cfg.n_steps):
        dr_local = -1j * comm(h_local, rho)
        dr_vertical_u = -1j * comm(h_vertical, rho)
        dr_source = lindblad_dissipator(rho, rel_jumps)
        dr_vertical_d = lindblad_dissipator(rho, drift_jumps)
        dr_total = np.nan_to_num(
            dr_local + dr_vertical_u + dr_source + dr_vertical_d,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if step >= cfg.burn_in and (step - cfg.burn_in) % cfg.sample_every == 0:
            dr_vertical = dr_vertical_u + dr_vertical_d
            flow0 = float(np.real(np.trace(observable_0 @ dr_vertical)))
            flow1 = float(np.real(np.trace(observable_1 @ dr_vertical)))
            vertical_flow = 0.5 * (flow1 - flow0)

            src0 = float(np.real(np.trace(observable_0 @ dr_source)))
            src1 = float(np.real(np.trace(observable_1 @ dr_source)))
            source_abs = 0.5 * (abs(src0) + abs(src1))

            block0 = rho[:d_chain, :d_chain]
            block1 = rho[d_chain:, d_chain:]
            p0 = float(np.real(np.trace(block0)))
            p1 = float(np.real(np.trace(block1)))
            if p0 > 1e-12:
                block0 = block0 / p0
            if p1 > 1e-12:
                block1 = block1 / p1

            block0_off = block0 - np.diag(np.diag(block0)) if p0 > 1e-12 else np.zeros_like(block0)
            block1_off = block1 - np.diag(np.diag(block1)) if p1 > 1e-12 else np.zeros_like(block1)
            lambda_coh = 0.0
            if p0 > 1e-12:
                lambda_coh += p0 * float(np.real(np.trace(f_chain @ block0_off)))
            if p1 > 1e-12:
                lambda_coh += p1 * float(np.real(np.trace(f_chain @ block1_off)))

            ds_layers = []
            if p0 > 1e-12:
                ds_layers.append(entropy_effective_rank_log2(block0))
            if p1 > 1e-12:
                ds_layers.append(entropy_effective_rank_log2(block1))
            ds_sample = float(np.mean(ds_layers)) if ds_layers else float("nan")

            rows.append(
                {
                    "step": int(step),
                    "epsilon": float(epsilon),
                    "epsilon_internal": float(eps_internal),
                    "vertical_flow": float(vertical_flow),
                    "vertical_flow_abs": float(abs(vertical_flow)),
                    "source_abs": float(source_abs),
                    "lambda_coh": float(lambda_coh),
                    "d_s_effrank_sample": ds_sample,
                }
            )

        rho = stabilize_density(rho + cfg.dt * dr_total)

    series = pd.DataFrame(rows)
    ds_mean = float(series["d_s_effrank_sample"].mean())
    ds_dist = float(abs(ds_mean - round(ds_mean)))
    summary_row = {
        "epsilon": float(epsilon),
        "epsilon_internal": float(eps_internal),
        "omega_drive": float(omega_drive),
        "gamma_rel": float(gamma_rel),
        "gamma_drift": float(gamma_drift),
        "vertical_flow_abs_mean": float(series["vertical_flow_abs"].mean()),
        "vertical_flow_abs_median": float(series["vertical_flow_abs"].median()),
        "source_abs_mean": float(series["source_abs"].mean()),
        "lambda_coh_rms": float(np.sqrt(np.mean(np.square(series["lambda_coh"].to_numpy(float))))),
        "d_s_effrank_mean": ds_mean,
        "d_s_nearest_integer": int(round(ds_mean)),
        "d_s_integer_distance": ds_dist,
    }
    return series, summary_row


def evaluate_f1_criteria(summary: pd.DataFrame, cfg: F1Config) -> dict[str, object]:
    summary = summary.sort_values("epsilon").reset_index(drop=True)

    idx_star = int(summary["vertical_flow_abs_mean"].idxmin())
    row_star = summary.iloc[idx_star]
    row_low = summary.iloc[int((summary["epsilon"] - min(cfg.eps_list)).abs().idxmin())]
    row_high = summary.iloc[int((summary["epsilon"] - max(cfg.eps_list)).abs().idxmin())]

    low_w, high_w = cfg.epsilon_star_window
    checks = {
        "epsilon_star_in_window": bool(low_w <= float(row_star["epsilon"]) <= high_w),
        "flow_star_is_min_vs_extremes": bool(
            float(row_star["vertical_flow_abs_mean"])
            < min(float(row_low["vertical_flow_abs_mean"]), float(row_high["vertical_flow_abs_mean"]))
        ),
        "flow_extremes_nonzero": bool(
            float(row_low["vertical_flow_abs_mean"]) > cfg.flow_floor
            and float(row_high["vertical_flow_abs_mean"]) > cfg.flow_floor
        ),
        "source_star_nonzero": bool(float(row_star["source_abs_mean"]) > cfg.source_floor),
        "d_s_star_noninteger": bool(float(row_star["d_s_integer_distance"]) > cfg.noninteger_tol),
        "d_s_extremes_near_integer": bool(
            float(row_low["d_s_integer_distance"]) < cfg.integer_tol
            and float(row_high["d_s_integer_distance"]) < cfg.integer_tol
        ),
    }
    checks["pass_all"] = bool(all(checks.values()))

    return {
        "epsilon_star": float(row_star["epsilon"]),
        "row_star": row_star.to_dict(),
        "row_low": row_low.to_dict(),
        "row_high": row_high.to_dict(),
        "checks": checks,
    }


def build_report(summary: pd.DataFrame, verdict: dict[str, object], cfg: F1Config) -> str:
    checks = verdict["checks"]
    row_star = verdict["row_star"]
    row_low = verdict["row_low"]
    row_high = verdict["row_high"]

    lines = [
        "# Experiment F1 Report: Fractal Emergence from Balance",
        "",
        "## Setup",
        "- Model: two-scale (`mu1, mu2`) GKSL dynamics with coherent pump + Lindblad dissipation + vertical transport.",
        "- Control parameter: `epsilon = tau_rel / tau_dr` scanned on `[0.01, 10]`.",
        "- Metrics:",
        "  - `vertical_flow_abs_mean` as proxy for `|partial_mu <J_O>|`.",
        "  - `source_abs_mean` as `|S_O|` proxy.",
        "  - `lambda_coh_rms` for `Lambda_coh`.",
        "  - `d_s_effrank_mean`: spectral-dimension proxy from the eigen-spectrum of `rho(mu)` via entropy effective rank.",
        "",
        "## Key Points",
        f"- Balance point (`epsilon*`, minimum flow): `{verdict['epsilon_star']:.2f}`.",
        f"- At `epsilon*`: flow=`{row_star['vertical_flow_abs_mean']:.6e}`, source=`{row_star['source_abs_mean']:.6e}`, d_s=`{row_star['d_s_effrank_mean']:.4f}`.",
        f"- Low extreme (`epsilon={row_low['epsilon']:.2f}`): flow=`{row_low['vertical_flow_abs_mean']:.6e}`, d_s=`{row_low['d_s_effrank_mean']:.4f}`.",
        f"- High extreme (`epsilon={row_high['epsilon']:.2f}`): flow=`{row_high['vertical_flow_abs_mean']:.6e}`, d_s=`{row_high['d_s_effrank_mean']:.4f}`.",
        "",
        "## F1 Criteria",
        f"- epsilon_star_in_window: `{checks['epsilon_star_in_window']}`",
        f"- flow_star_is_min_vs_extremes: `{checks['flow_star_is_min_vs_extremes']}`",
        f"- flow_extremes_nonzero: `{checks['flow_extremes_nonzero']}`",
        f"- source_star_nonzero: `{checks['source_star_nonzero']}`",
        f"- d_s_star_noninteger: `{checks['d_s_star_noninteger']}`",
        f"- d_s_extremes_near_integer: `{checks['d_s_extremes_near_integer']}`",
        f"- PASS_ALL: `{checks['pass_all']}`",
        "",
        "## Notes",
        "- `d_s_effrank_mean` is a finite-size spectral proxy suitable for two-scale toy runs; for dense continuum spectra, use full spectral-density fitting.",
    ]
    return "\n".join(lines) + "\n"


def run_f1(outdir: Path, cfg: F1Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    outdir.mkdir(parents=True, exist_ok=True)
    static = build_static_geometry(cfg)

    series_all = []
    summary_rows = []
    for eps in cfg.eps_list:
        series, row = run_one_epsilon(float(eps), cfg=cfg, static=static)
        series_all.append(series)
        summary_rows.append(row)

    timeseries = pd.concat(series_all, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("epsilon").reset_index(drop=True)
    verdict = evaluate_f1_criteria(summary=summary, cfg=cfg)

    timeseries.to_csv(outdir / "experiment_F1_timeseries.csv", index=False)
    summary.to_csv(outdir / "experiment_F1_summary.csv", index=False)

    with open(outdir / "experiment_F1_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    report = build_report(summary=summary, verdict=verdict, cfg=cfg)
    (outdir / "report.md").write_text(report, encoding="utf-8")
    return timeseries, summary, verdict


def parse_eps_list(eps_raw: str) -> tuple[float, ...]:
    values = [float(x.strip()) for x in eps_raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Empty epsilon list.")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_F1_fractal_emergence",
        help="output directory",
    )
    parser.add_argument(
        "--eps-list",
        default="0.01,0.03,0.10,0.30,0.70,1.00,1.40,3.00,10.00",
        help="comma-separated epsilon values",
    )
    parser.add_argument("--quick", action="store_true", help="run lightweight settings")
    args = parser.parse_args()

    cfg = F1Config(eps_list=parse_eps_list(args.eps_list))
    if args.quick:
        cfg.n_steps = 900
        cfg.burn_in = 240
        cfg.sample_every = 8

    outdir = Path(args.out)
    _, summary, verdict = run_f1(outdir=outdir, cfg=cfg)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nVerdict:")
    print(json.dumps(verdict["checks"], ensure_ascii=False, indent=2))
    print(f"\nSaved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
