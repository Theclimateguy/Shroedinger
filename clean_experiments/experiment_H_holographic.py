#!/usr/bin/env python3
"""Experiment H: holographic truncation under exponential layer-dimension growth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _random_hermitian(rng: np.random.Generator, dim: int, scale: float = 1.0) -> np.ndarray:
    g = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = 0.5 * (g + g.conj().T)
    return scale * h / np.sqrt(max(dim, 1))


def _random_density(rng: np.random.Generator, dim: int, mix_pure: float) -> np.ndarray:
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    rho_pure = np.outer(v, v.conj())
    rho = mix_pure * rho_pure + (1.0 - mix_pure) * np.eye(dim, dtype=complex) / dim
    return rho / np.trace(rho)


def _random_jump(rng: np.random.Generator, dim: int, scale: float = 0.6) -> np.ndarray:
    j = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return scale * j / np.sqrt(max(dim, 1))


def _lindblad_rhs(rho: np.ndarray, h: np.ndarray, jump: np.ndarray) -> np.ndarray:
    comm = h @ rho - rho @ h
    jd = jump.conj().T
    diss = jump @ rho @ jd - 0.5 * (jd @ jump @ rho + rho @ jd @ jump)
    return -1j * comm + diss


def _weights_from_mode(d_eff: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        w = np.ones_like(d_eff, dtype=float)
    elif mode == "dim":
        w = d_eff.astype(float)
    elif mode == "inv_dim":
        w = 1.0 / np.maximum(d_eff.astype(float), 1.0)
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")
    return w / np.sum(w)


def _dimension_profile(
    mu_grid: np.ndarray,
    d0: int,
    growth: float,
    holo_growth: float,
    d_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_shift = mu_grid - float(mu_grid[0])
    d_raw = np.clip(np.rint(d0 * np.exp(growth * mu_shift)), 2, d_max).astype(int)
    d_holo = np.clip(np.rint(d0 * np.exp(holo_growth * mu_shift)), 2, d_max).astype(int)
    d_eff = np.minimum(d_raw, d_holo)
    return d_raw, d_holo, d_eff


def _layer_case(
    seed: int,
    mu_grid: np.ndarray,
    d0: int,
    growth: float,
    holo_growth: float,
    d_max: int,
    mix_pure: float,
    weight_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    d_raw, d_holo, d_eff = _dimension_profile(mu_grid, d0, growth, holo_growth, d_max)
    weights = _weights_from_mode(d_eff, weight_mode)

    dim_df = pd.DataFrame(
        {
            "k": np.arange(len(mu_grid)),
            "mu": mu_grid,
            "d_raw": d_raw,
            "d_holo": d_holo,
            "d_eff": d_eff,
            "weight": weights,
            "cut_fraction": 1.0 - d_eff / np.maximum(d_raw, 1),
        }
    )

    rows: list[dict[str, float]] = []
    for k, (mu, d_r, d_e, wk) in enumerate(zip(mu_grid, d_raw, d_eff, weights)):
        rng = np.random.default_rng(seed + 7919 * (k + 1))

        f_raw = _random_hermitian(rng, d_r, scale=0.9)
        h_raw = _random_hermitian(rng, d_r, scale=0.7)
        rho_raw = _random_density(rng, d_r, mix_pure=mix_pure)
        j_raw = _random_jump(rng, d_r, scale=0.6)

        src_raw = float(np.real(np.trace(f_raw @ rho_raw)))
        src_raw_reg = src_raw / d_r
        dr_raw = _lindblad_rhs(rho_raw, h_raw, j_raw)
        tr_raw = float(abs(np.trace(dr_raw)))
        dyn_raw = float(np.linalg.norm(dr_raw))

        if d_e < d_r:
            f_cut = f_raw[:d_e, :d_e]
            h_cut = h_raw[:d_e, :d_e]
            j_cut = j_raw[:d_e, :d_e]
            rho_cut = rho_raw[:d_e, :d_e]
            tr_cut = np.trace(rho_cut)
            if abs(tr_cut) <= 1e-14:
                rho_cut = _random_density(rng, d_e, mix_pure=mix_pure)
            else:
                rho_cut = rho_cut / tr_cut
        else:
            f_cut = f_raw
            h_cut = h_raw
            j_cut = j_raw
            rho_cut = rho_raw

        src_cut = float(np.real(np.trace(f_cut @ rho_cut)))
        src_cut_reg = src_cut / d_e
        dr_cut = _lindblad_rhs(rho_cut, h_cut, j_cut)
        tr_cut = float(abs(np.trace(dr_cut)))
        dyn_cut = float(np.linalg.norm(dr_cut))

        rows.append(
            {
                "k": float(k),
                "mu": float(mu),
                "d_raw": float(d_r),
                "d_eff": float(d_e),
                "weight": float(wk),
                "source_raw": src_raw,
                "source_cut": src_cut,
                "source_raw_reg": src_raw_reg,
                "source_cut_reg": src_cut_reg,
                "trace_rhs_raw_abs": tr_raw,
                "trace_rhs_cut_abs": tr_cut,
                "dyn_norm_raw": dyn_raw,
                "dyn_norm_cut": dyn_cut,
            }
        )

    layer_df = pd.DataFrame(rows)
    summary = {
        "lambda_raw_unreg": float(np.sum(layer_df["weight"] * layer_df["source_raw"])),
        "lambda_cut_unreg": float(np.sum(layer_df["weight"] * layer_df["source_cut"])),
        "lambda_raw_reg": float(np.sum(layer_df["weight"] * layer_df["source_raw_reg"])),
        "lambda_cut_reg": float(np.sum(layer_df["weight"] * layer_df["source_cut_reg"])),
        "trace_raw_max": float(layer_df["trace_rhs_raw_abs"].max()),
        "trace_cut_max": float(layer_df["trace_rhs_cut_abs"].max()),
        "dyn_norm_raw_mean": float(layer_df["dyn_norm_raw"].mean()),
        "dyn_norm_cut_mean": float(layer_df["dyn_norm_cut"].mean()),
        "mean_d_raw": float(np.mean(d_raw)),
        "mean_d_eff": float(np.mean(d_eff)),
        "max_d_raw": float(np.max(d_raw)),
        "max_d_eff": float(np.max(d_eff)),
        "mean_cut_fraction": float(np.mean(1.0 - d_eff / np.maximum(d_raw, 1))),
    }
    return dim_df, layer_df, summary


def _parse_scan(scan: str) -> list[int]:
    vals = [int(x.strip()) for x in scan.split(",") if x.strip()]
    return sorted(set(v for v in vals if v >= 3))


def _rel_span(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float((np.max(arr) - np.min(arr)) / (abs(np.mean(arr)) + 1e-15))


def run_experiment(
    outdir: Path,
    *,
    seed: int = 20260222,
    k_layers: int = 20,
    mu_min: float = 0.0,
    mu_max: float = 6.0,
    d0: int = 4,
    growth: float = 0.9,
    holo_growth: float = 0.55,
    d_max: int = 256,
    mix_pure: float = 0.6,
    weight_mode: str = "uniform",
    resolution_scan: str = "8,12,16,20,24",
    write_detail_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mu_grid = np.linspace(mu_min, mu_max, k_layers)
    dim_df, layer_df, core = _layer_case(
        seed=seed,
        mu_grid=mu_grid,
        d0=d0,
        growth=growth,
        holo_growth=holo_growth,
        d_max=d_max,
        mix_pure=mix_pure,
        weight_mode=weight_mode,
    )

    scan_rows = []
    for idx, k_scan in enumerate(_parse_scan(resolution_scan)):
        mu_scan = np.linspace(mu_min, mu_max, k_scan)
        _, _, s = _layer_case(
            seed=seed + 100 * (idx + 1),
            mu_grid=mu_scan,
            d0=d0,
            growth=growth,
            holo_growth=holo_growth,
            d_max=d_max,
            mix_pure=mix_pure,
            weight_mode=weight_mode,
        )
        scan_rows.append({"k_layers": int(k_scan), **s})
    scan_df = pd.DataFrame(scan_rows).sort_values("k_layers").reset_index(drop=True)

    std_raw_unreg = float(scan_df["lambda_raw_unreg"].std(ddof=0))
    std_cut_reg = float(scan_df["lambda_cut_reg"].std(ddof=0))
    stability_gain = std_raw_unreg / (std_cut_reg + 1e-15)
    pass_trace = max(core["trace_raw_max"], core["trace_cut_max"]) <= 1e-11
    pass_stability = stability_gain > 1.0

    summary = pd.DataFrame(
        [
            {
                "seed": int(seed),
                "k_layers": int(k_layers),
                "mu_min": float(mu_min),
                "mu_max": float(mu_max),
                "d0": int(d0),
                "growth": float(growth),
                "holo_growth": float(holo_growth),
                "d_max": int(d_max),
                "mix_pure": float(mix_pure),
                "weight_mode": weight_mode,
                **core,
                "resolution_std_raw_unreg": std_raw_unreg,
                "resolution_std_cut_reg": std_cut_reg,
                "resolution_std_raw_reg": float(scan_df["lambda_raw_reg"].std(ddof=0)),
                "resolution_relspan_raw_unreg": _rel_span(scan_df["lambda_raw_unreg"]),
                "resolution_relspan_cut_reg": _rel_span(scan_df["lambda_cut_reg"]),
                "resolution_stability_gain": stability_gain,
                "pass_trace_1e-11": bool(pass_trace),
                "pass_stability_gain_gt_1": bool(pass_stability),
                "pass_all": bool(pass_trace and pass_stability),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    if write_detail_csv:
        dim_df.to_csv(outdir / "experiment_H_dimension_profile.csv", index=False)
        layer_df.to_csv(outdir / "experiment_H_layer_metrics.csv", index=False)
        scan_df.to_csv(outdir / "experiment_H_resolution_scan.csv", index=False)
    summary.to_csv(outdir / "experiment_H_summary.csv", index=False)
    return dim_df, layer_df, scan_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_H_holographic", help="output directory")
    parser.add_argument("--seed", type=int, default=20260222)
    parser.add_argument("--k-layers", type=int, default=20)
    parser.add_argument("--mu-min", type=float, default=0.0)
    parser.add_argument("--mu-max", type=float, default=6.0)
    parser.add_argument("--d0", type=int, default=4)
    parser.add_argument("--growth", type=float, default=0.9)
    parser.add_argument("--holo-growth", type=float, default=0.55)
    parser.add_argument("--d-max", type=int, default=256)
    parser.add_argument("--mix-pure", type=float, default=0.6)
    parser.add_argument(
        "--weight-mode",
        choices=["uniform", "dim", "inv_dim"],
        default="uniform",
    )
    parser.add_argument("--resolution-scan", default="8,12,16,20,24")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="write only experiment_H_summary.csv (no layer/scan tables)",
    )
    args = parser.parse_args()

    _, _, scan_df, summary_df = run_experiment(
        outdir=Path(args.out),
        seed=args.seed,
        k_layers=args.k_layers,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        d0=args.d0,
        growth=args.growth,
        holo_growth=args.holo_growth,
        d_max=args.d_max,
        mix_pure=args.mix_pure,
        weight_mode=args.weight_mode,
        resolution_scan=args.resolution_scan,
        write_detail_csv=not args.summary_only,
    )

    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    if not args.summary_only:
        print("\nResolution scan:")
        print(
            scan_df[
                [
                    "k_layers",
                    "lambda_raw_unreg",
                    "lambda_cut_unreg",
                    "lambda_raw_reg",
                    "lambda_cut_reg",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.6e}")
        )
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
