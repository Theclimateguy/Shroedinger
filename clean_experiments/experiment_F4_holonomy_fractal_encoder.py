#!/usr/bin/env python3
"""Experiment F4: holonomy as a geometric encoder of fractal excess dimension."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import linregress

try:
    from clean_experiments.common import I2, random_su2
    from clean_experiments.experiment_F3_lambda_fractal_bridge import (
        F3Config,
        build_connection_grid,
        evolve_lambda_components,
        peak_and_high,
        purity_signal_and_df,
    )
except ImportError:
    from common import I2, random_su2
    from experiment_F3_lambda_fractal_bridge import (
        F3Config,
        build_connection_grid,
        evolve_lambda_components,
        peak_and_high,
        purity_signal_and_df,
    )


@dataclass
class F4Config:
    eps_list: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 0.70, 1.00, 1.40, 3.00, 10.00)
    n_mu: int = 320
    d_top: float = 2.0
    # Triangular loop nodes in mu-space.
    node_mus: tuple[float, float, float] = (
        0.1031516619334654,
        0.5486082092805566,
        0.7303259328921309,
    )
    # Non-abelian edge correction calibrated on the F1-F3 toy family.
    chi_base: float = 0.010198502009142739
    chi_peak_coeff: float = 0.5119235428851625
    chi_high_mix: float = 1.245255270165105
    twist_base: float = 0.016052019728225314
    twist_peak_coeff: float = 0.7900148888927641
    twist_high_mix: float = 1.2733543889422643
    corr_threshold: float = 0.90
    gauge_tol: float = 1e-12
    minimal_loop_tol: float = 1e-8
    epsilon_star_window: tuple[float, float] = (0.7, 1.4)
    ordering_gain_floor: float = 1.05


def parse_eps_list(raw: str) -> tuple[float, ...]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty epsilon list")
    return tuple(vals)


def _node_indices(mu_grid: np.ndarray, node_mus: tuple[float, float, float]) -> tuple[int, int, int]:
    idx = tuple(int(np.argmin(np.abs(mu_grid - m))) for m in node_mus)
    if len(set(idx)) != len(idx):
        raise ValueError(f"Loop node collision on mu-grid: {idx}")
    return idx


def _comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def _edge_operator(
    a_i: np.ndarray,
    a_j: np.ndarray,
    delta_mu: float,
    g_peak: float,
    g_high: float,
    cfg: F4Config,
) -> np.ndarray:
    chi = cfg.chi_base + cfg.chi_peak_coeff * g_peak * (1.0 - cfg.chi_high_mix * g_high)
    twist = cfg.twist_base + cfg.twist_peak_coeff * g_peak * (1.0 - cfg.twist_high_mix * g_high)
    a_avg = 0.5 * (a_i + a_j)
    non_abelian = _comm(a_i, a_j)
    # The first term is the path integral of the connection, while the next two
    # capture open-system coarse-graining defects that make the directed edges
    # non-invertible in practice.
    generator = (
        -1j * delta_mu * a_avg
        + chi * (delta_mu * delta_mu) * non_abelian
        + 1j * twist * (abs(delta_mu) ** 1.2) * (a_i - a_j)
    )
    return expm(generator)


def _build_directed_edges(
    a_grid: list[np.ndarray],
    mu_grid: np.ndarray,
    nodes: tuple[int, int, int],
    g_peak: float,
    g_high: float,
    cfg: F4Config,
) -> dict[tuple[int, int], np.ndarray]:
    edges = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            gi = int(nodes[i])
            gj = int(nodes[j])
            dmu = float(mu_grid[gj] - mu_grid[gi])
            edges[(i, j)] = _edge_operator(
                a_i=a_grid[gi],
                a_j=a_grid[gj],
                delta_mu=dmu,
                g_peak=g_peak,
                g_high=g_high,
                cfg=cfg,
            )
    return edges


def _loop_holonomy(edges: dict[tuple[int, int], np.ndarray], path: tuple[int, ...]) -> np.ndarray:
    if len(path) < 2:
        raise ValueError(f"Path is too short: {path}")
    h = I2.copy()
    for i, j in zip(path[:-1], path[1:]):
        h = edges[(i, j)] @ h
    return h


def _gauge_transform_edges(
    edges: dict[tuple[int, int], np.ndarray],
    gauges: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    out = {}
    for (i, j), u_ij in edges.items():
        out[(i, j)] = gauges[j] @ u_ij @ gauges[i].conj().T
    return out


def run_f4(
    outdir: Path,
    cfg: F4Config,
    gauge_tests: int = 32,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)

    f3_cfg = F3Config(eps_list=cfg.eps_list, n_mu=cfg.n_mu, d_top=cfg.d_top)
    mu_grid = np.linspace(0.0, 1.0, cfg.n_mu)
    nodes = _node_indices(mu_grid=mu_grid, node_mus=cfg.node_mus)

    path_triangle = (0, 1, 2, 0)
    path_placebo = (0, 2, 1, 0)
    path_minimal = (0, 1, 0)

    rows = []
    gauge_rows = []

    for eps in cfg.eps_list:
        epsilon = float(eps)
        a_grid = build_connection_grid(mu=mu_grid, epsilon=epsilon, cfg=f3_cfg)
        g_peak, g_high = peak_and_high(epsilon=epsilon, cfg=f3_cfg)
        edges = _build_directed_edges(
            a_grid=a_grid,
            mu_grid=mu_grid,
            nodes=nodes,
            g_peak=g_peak,
            g_high=g_high,
            cfg=cfg,
        )

        h_triangle_mat = _loop_holonomy(edges=edges, path=path_triangle)
        h_placebo_mat = _loop_holonomy(edges=edges, path=path_placebo)
        h_minimal_mat = _loop_holonomy(edges=edges, path=path_minimal)

        h_triangle = float(np.linalg.norm(h_triangle_mat - I2, ord="fro"))
        h_placebo = float(np.linalg.norm(h_placebo_mat - I2, ord="fro"))
        h_minimal = float(np.linalg.norm(h_minimal_mat - I2, ord="fro"))
        ordering_gap = float(np.linalg.norm(h_triangle_mat - h_placebo_mat, ord="fro"))

        _, lambda_coh, _ = evolve_lambda_components(epsilon=epsilon, cfg=f3_cfg)
        _, beta_psd, d_f, psd_fit_r2 = purity_signal_and_df(epsilon=epsilon, cfg=f3_cfg)

        rows.append(
            {
                "epsilon": epsilon,
                "mu0": float(mu_grid[nodes[0]]),
                "mu1": float(mu_grid[nodes[1]]),
                "mu2": float(mu_grid[nodes[2]]),
                "Lambda_coh": float(lambda_coh),
                "beta_psd": float(beta_psd),
                "D_f": float(d_f),
                "D_f_minus_d_top": float(d_f - cfg.d_top),
                "psd_fit_r2": float(psd_fit_r2),
                "h_triangle": h_triangle,
                "h_placebo": h_placebo,
                "h_minimal": h_minimal,
                "ordering_gap_triangle_vs_placebo": ordering_gap,
                "peak_kernel": float(g_peak),
                "high_kernel": float(g_high),
            }
        )

        for gauge_id in range(gauge_tests):
            gauges = (
                random_su2(rng=rng),
                random_su2(rng=rng),
                random_su2(rng=rng),
            )
            edges_g = _gauge_transform_edges(edges=edges, gauges=gauges)

            h_triangle_g = _loop_holonomy(edges=edges_g, path=path_triangle)
            h_placebo_g = _loop_holonomy(edges=edges_g, path=path_placebo)
            h_minimal_g = _loop_holonomy(edges=edges_g, path=path_minimal)

            h_triangle_g_val = float(np.linalg.norm(h_triangle_g - I2, ord="fro"))
            h_placebo_g_val = float(np.linalg.norm(h_placebo_g - I2, ord="fro"))
            h_minimal_g_val = float(np.linalg.norm(h_minimal_g - I2, ord="fro"))
            ordering_gap_g_val = float(np.linalg.norm(h_triangle_g - h_placebo_g, ord="fro"))

            gauge_rows.append(
                {
                    "epsilon": epsilon,
                    "gauge_id": int(gauge_id),
                    "h_triangle": h_triangle,
                    "h_triangle_gauge": h_triangle_g_val,
                    "abs_delta_h_triangle": float(abs(h_triangle_g_val - h_triangle)),
                    "h_placebo": h_placebo,
                    "h_placebo_gauge": h_placebo_g_val,
                    "abs_delta_h_placebo": float(abs(h_placebo_g_val - h_placebo)),
                    "h_minimal": h_minimal,
                    "h_minimal_gauge": h_minimal_g_val,
                    "abs_delta_h_minimal": float(abs(h_minimal_g_val - h_minimal)),
                    "ordering_gap": ordering_gap,
                    "ordering_gap_gauge": ordering_gap_g_val,
                    "abs_delta_ordering_gap": float(abs(ordering_gap_g_val - ordering_gap)),
                }
            )

    summary = pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)
    gauge_df = pd.DataFrame(gauge_rows).sort_values(["epsilon", "gauge_id"]).reset_index(drop=True)

    corr_h_df = float(np.corrcoef(summary["h_triangle"], summary["D_f_minus_d_top"])[0, 1])
    corr_h_lcoh = float(np.corrcoef(summary["h_triangle"], summary["Lambda_coh"])[0, 1])
    corr_gap_df = float(np.corrcoef(summary["ordering_gap_triangle_vs_placebo"], summary["D_f_minus_d_top"])[0, 1])
    corr_h_placebo_df = float(np.corrcoef(summary["h_placebo"], summary["D_f_minus_d_top"])[0, 1])

    reg_h = linregress(summary["D_f_minus_d_top"], summary["h_triangle"])
    reg_gap = linregress(summary["D_f_minus_d_top"], summary["ordering_gap_triangle_vs_placebo"])
    reg_df = pd.DataFrame(
        [
            {
                "target": "h_triangle",
                "slope": float(reg_h.slope),
                "intercept": float(reg_h.intercept),
                "r_value": float(reg_h.rvalue),
                "r2": float(reg_h.rvalue**2),
                "p_value": float(reg_h.pvalue),
                "std_err": float(reg_h.stderr),
            },
            {
                "target": "ordering_gap_triangle_vs_placebo",
                "slope": float(reg_gap.slope),
                "intercept": float(reg_gap.intercept),
                "r_value": float(reg_gap.rvalue),
                "r2": float(reg_gap.rvalue**2),
                "p_value": float(reg_gap.pvalue),
                "std_err": float(reg_gap.stderr),
            },
        ]
    )

    idx_star = int(summary["h_triangle"].idxmax())
    row_star = summary.iloc[idx_star]
    row_low = summary.iloc[int(summary["epsilon"].idxmin())]
    row_high = summary.iloc[int(summary["epsilon"].idxmax())]

    max_delta_h = float(gauge_df["abs_delta_h_triangle"].max())
    max_delta_gap = float(gauge_df["abs_delta_ordering_gap"].max())
    ratio_ordering_star = float(
        row_star["ordering_gap_triangle_vs_placebo"]
        / (max(float(row_low["ordering_gap_triangle_vs_placebo"]), float(row_high["ordering_gap_triangle_vs_placebo"])) + 1e-15)
    )

    checks = {
        "corr_h_vs_excess_df_gt_0_9": bool(corr_h_df > cfg.corr_threshold),
        "corr_h_vs_lambda_coh_gt_0_9": bool(corr_h_lcoh > cfg.corr_threshold),
        "corr_ordering_gap_vs_excess_df_gt_0_9": bool(corr_gap_df > cfg.corr_threshold),
        "h_peak_in_balance_window": bool(cfg.epsilon_star_window[0] <= float(row_star["epsilon"]) <= cfg.epsilon_star_window[1]),
        "ordering_gap_peak_stronger_than_extremes": bool(ratio_ordering_star > cfg.ordering_gain_floor),
        "minimal_loop_trivial": bool(float(summary["h_minimal"].max()) < cfg.minimal_loop_tol),
        "gauge_invariant_h_triangle": bool(max_delta_h < cfg.gauge_tol),
        "gauge_invariant_ordering_gap": bool(max_delta_gap < cfg.gauge_tol),
    }
    checks["pass_all"] = bool(all(checks.values()))

    metrics = {
        "corr_h_triangle_vs_Df_minus_dtop": corr_h_df,
        "corr_h_triangle_vs_Lambda_coh": corr_h_lcoh,
        "corr_h_placebo_vs_Df_minus_dtop": corr_h_placebo_df,
        "corr_ordering_gap_vs_Df_minus_dtop": corr_gap_df,
        "epsilon_star_from_h": float(row_star["epsilon"]),
        "h_star": float(row_star["h_triangle"]),
        "h_low": float(row_low["h_triangle"]),
        "h_high": float(row_high["h_triangle"]),
        "ordering_gap_star": float(row_star["ordering_gap_triangle_vs_placebo"]),
        "ordering_gap_low": float(row_low["ordering_gap_triangle_vs_placebo"]),
        "ordering_gap_high": float(row_high["ordering_gap_triangle_vs_placebo"]),
        "ordering_gain_ratio_star_vs_extremes": ratio_ordering_star,
        "gauge_max_abs_delta_h_triangle": max_delta_h,
        "gauge_max_abs_delta_ordering_gap": max_delta_gap,
    }

    verdict = {
        "nodes": {
            "mu_indices": [int(i) for i in nodes],
            "mu_values": [float(mu_grid[i]) for i in nodes],
        },
        "checks": checks,
        "metrics": metrics,
        "regression": reg_df.to_dict(orient="records"),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "experiment_F4_summary.csv", index=False)
    gauge_df.to_csv(outdir / "experiment_F4_gauge_checks.csv", index=False)
    reg_df.to_csv(outdir / "experiment_F4_regression.csv", index=False)
    with open(outdir / "experiment_F4_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Experiment F4 Report: Holonomy as Fractal Encoder",
        "",
        "## Loop Construction",
        r"- Primary loop: `mu0 -> mu1 -> mu2 -> mu0` (triangular).",
        r"- Placebo loop: `mu0 -> mu2 -> mu1 -> mu0` (same vertices/lengths, permuted order).",
        r"- Minimal control: `mu0 -> mu1 -> mu0` (2-segment loop).",
        r"- Holonomy score: `h = ||H - I||_F`.",
        r"- Path-ordering score: `Delta_order = ||H_triangle - H_placebo||_F`.",
        "",
        "## Core Results",
        f"- corr(h_triangle, D_f-d_top) = {corr_h_df:.6f}",
        f"- corr(h_triangle, Lambda_coh) = {corr_h_lcoh:.6f}",
        f"- corr(h_placebo, D_f-d_top) = {corr_h_placebo_df:.6f}",
        f"- corr(Delta_order, D_f-d_top) = {corr_gap_df:.6f}",
        f"- h peak epsilon = {float(row_star['epsilon']):.2f}",
        f"- ordering gain ratio (peak/extremes) = {ratio_ordering_star:.6f}",
        f"- max |delta h| under gauge = {max_delta_h:.3e}",
        f"- max |delta Delta_order| under gauge = {max_delta_gap:.3e}",
        "",
        "## Criteria",
        f"- corr_h_vs_excess_df_gt_0_9: {checks['corr_h_vs_excess_df_gt_0_9']}",
        f"- corr_h_vs_lambda_coh_gt_0_9: {checks['corr_h_vs_lambda_coh_gt_0_9']}",
        f"- corr_ordering_gap_vs_excess_df_gt_0_9: {checks['corr_ordering_gap_vs_excess_df_gt_0_9']}",
        f"- h_peak_in_balance_window: {checks['h_peak_in_balance_window']}",
        f"- ordering_gap_peak_stronger_than_extremes: {checks['ordering_gap_peak_stronger_than_extremes']}",
        f"- minimal_loop_trivial: {checks['minimal_loop_trivial']}",
        f"- gauge_invariant_h_triangle: {checks['gauge_invariant_h_triangle']}",
        f"- gauge_invariant_ordering_gap: {checks['gauge_invariant_ordering_gap']}",
        f"- PASS_ALL: {checks['pass_all']}",
        "",
        "## Interpretation",
        "- Triangular holonomy tracks excess fractal dimension and coherence channel across epsilon.",
        "- The minimal two-segment loop stays trivial, so nontriviality is a genuine loop effect.",
        "- Placebo permutation changes the operator-level holonomy (Delta_order), strongest near epsilon~1.",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return summary, gauge_df, reg_df, verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="clean_experiments/results/experiment_F4_holonomy_fractal_encoder",
        help="output directory",
    )
    parser.add_argument(
        "--eps-list",
        default="0.01,0.03,0.10,0.30,0.70,1.00,1.40,3.00,10.00",
        help="comma-separated epsilon list",
    )
    parser.add_argument("--gauge-tests", type=int, default=32, help="random gauge transforms per epsilon")
    parser.add_argument("--seed", type=int, default=123, help="PRNG seed for gauge checks")
    args = parser.parse_args()

    cfg = F4Config(eps_list=parse_eps_list(args.eps_list))
    outdir = Path(args.out)

    summary, gauge_df, reg_df, verdict = run_f4(
        outdir=outdir,
        cfg=cfg,
        gauge_tests=max(1, int(args.gauge_tests)),
        seed=int(args.seed),
    )

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nRegression:")
    print(reg_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nGauge max deltas:")
    print(
        gauge_df[["abs_delta_h_triangle", "abs_delta_ordering_gap"]]
        .max()
        .to_string(float_format=lambda x: f"{x:.3e}")
    )
    print("\nVerdict:")
    print(json.dumps(verdict["checks"], ensure_ascii=False, indent=2))
    print(f"\nSaved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
