#!/usr/bin/env python3
"""Experiment F6c: clustered subspace tail fits for |Lambda_local(t,y,x)|."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from clean_experiments.experiment_M_cosmo_flow import _build_band_masks, _compute_vorticity, _xy_coordinates_m
    from clean_experiments.experiment_O_spatial_variance import (
        _compute_lambda_local_batch,
        _find_coord_name,
        _find_var_name,
        _load_m_band_amplitudes,
        _to_datetime64,
    )
    from clean_experiments.experiment_F6b_era5_heavy_tails import (
        F6bConfig,
        _bootstrap_alpha_ci,
        _plot_global_tail,
        _safe_quantile,
        _scan_xmin_candidates,
        _zscore,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _build_band_masks, _compute_vorticity, _xy_coordinates_m  # type: ignore
    from experiment_O_spatial_variance import (  # type: ignore
        _compute_lambda_local_batch,
        _find_coord_name,
        _find_var_name,
        _load_m_band_amplitudes,
        _to_datetime64,
    )
    from experiment_F6b_era5_heavy_tails import (  # type: ignore
        F6bConfig,
        _bootstrap_alpha_ci,
        _plot_global_tail,
        _safe_quantile,
        _scan_xmin_candidates,
        _zscore,
    )


@dataclass
class F6cConfig:
    n_clusters: int = 6
    cluster_max_iter: int = 80
    cluster_tol: float = 1e-5
    cluster_seed: int = 20260307
    min_cluster_time_points: int = 120

    max_samples_global: int = 2_500_000
    max_samples_cluster: int = 1_000_000

    alpha_target_low: float = 1.5
    alpha_target_high: float = 2.0
    corrected_p_max: float = 0.05
    p_correction: str = "bonferroni"  # fixed, predeclared


def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=float)
    i0 = int(rng.integers(0, n))
    centers[0] = x[i0]
    d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = d2 / (np.sum(d2) + 1e-15)
        idx = int(rng.choice(n, p=probs))
        centers[i] = x[idx]
        d2 = np.minimum(d2, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def _kmeans(x: np.ndarray, k: int, *, max_iter: int, tol: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(x, k, rng)
    labels = np.zeros(x.shape[0], dtype=int)

    for _ in range(max_iter):
        d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = centers.copy()
        for j in range(k):
            m = labels == j
            if np.any(m):
                new_centers[j] = np.mean(x[m], axis=0)
            else:
                new_centers[j] = x[int(rng.integers(0, x.shape[0]))]

        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        if shift < tol:
            break
    return labels, centers


def _label_cluster(fractal_z: float, conv_z: float, exchange_z: float) -> str:
    f = "highF" if fractal_z >= 0.4 else ("lowF" if fractal_z <= -0.4 else "midF")
    c = "highC" if conv_z >= 0.4 else ("lowC" if conv_z <= -0.4 else "midC")
    e = "highX" if exchange_z >= 0.4 else ("lowX" if exchange_z <= -0.4 else "midX")
    return f"{f}_{c}_{e}"


def _extract_panel_lambda_and_conv(
    *,
    input_nc: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    scale_edges_km: list[float],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_amp, _ = _load_m_band_amplitudes(
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
    )

    with Dataset(input_nc, mode="r") as ds:
        time_name = _find_coord_name(ds, ("valid_time", "time", "datetime", "date"), "time")
        lat_name = _find_coord_name(ds, ("latitude", "lat", "y", "rlat"), "latitude")
        lon_name = _find_coord_name(ds, ("longitude", "lon", "x", "rlon"), "longitude")
        u_name = _find_var_name(ds, ("u",), "u")
        v_name = _find_var_name(ds, ("v",), "v")
        conv_name = _find_var_name(ds, ("convective_precip", "cp", "cp_rate", "precip"), "convective precip")

        time_ns = _to_datetime64(ds.variables[time_name])
        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)
        nt = len(time_ns)
        ny = len(lat)
        nx = len(lon)
        npix = ny * nx

        if band_amp.shape[0] != nt:
            raise ValueError(f"M amplitude length mismatch: {band_amp.shape[0]} vs {nt}")

        x_m, y_m = _xy_coordinates_m(lat, lon)
        dx_km = float(np.median(np.diff(x_m))) / 1000.0
        dy_km = float(np.median(np.diff(y_m))) / 1000.0
        masks, _, _, _, _, _ = _build_band_masks(
            ny=ny,
            nx=nx,
            dy_km=abs(dy_km),
            dx_km=abs(dx_km),
            scale_edges_km=scale_edges_km,
        )
        if len(masks) != band_amp.shape[1]:
            raise ValueError(f"Band mismatch: masks={len(masks)} vs amp={band_amp.shape[1]}")

        panel_abs = np.empty((nt, npix), dtype=np.float32)
        conv_mean = np.empty(nt, dtype=np.float32)

        u_var = ds.variables[u_name]
        v_var = ds.variables[v_name]
        conv_var = ds.variables[conv_name]

        for start in range(0, nt, batch_size):
            stop = min(start + batch_size, nt)
            sl = slice(start, stop)
            tb = stop - start

            u_blk = np.asarray(u_var[sl, ...], dtype=np.float64)
            v_blk = np.asarray(v_var[sl, ...], dtype=np.float64)
            conv_blk = np.asarray(conv_var[sl, ...], dtype=np.float64)
            u_blk = np.nan_to_num(u_blk, nan=0.0, posinf=0.0, neginf=0.0)
            v_blk = np.nan_to_num(v_blk, nan=0.0, posinf=0.0, neginf=0.0)
            conv_blk = np.nan_to_num(conv_blk, nan=0.0, posinf=0.0, neginf=0.0)

            zeta_blk = _compute_vorticity(u=u_blk, v=v_blk, x_m=x_m, y_m=y_m)
            lam_blk = _compute_lambda_local_batch(
                zeta_batch=zeta_blk,
                masks=masks,
                band_amp_batch=band_amp[start:stop],
            )
            panel_abs[start:stop] = np.abs(lam_blk.reshape(tb, npix)).astype(np.float32)
            conv_mean[start:stop] = np.mean(conv_blk.reshape(tb, npix), axis=1).astype(np.float32)
            print(f"[F6c] panel reconstruction: {stop}/{nt}", flush=True)

    return panel_abs, conv_mean, time_ns


def _plot_cluster_alpha(best_df: pd.DataFrame, out_path: Path, alpha_ref: float) -> None:
    d = best_df.copy()
    d = d[d["fit_available"] == True]  # noqa: E712
    if d.empty:
        return
    d = d.sort_values("alpha_mle", ascending=True).reset_index(drop=True)

    x = np.arange(len(d))
    y = d["alpha_mle"].to_numpy(float)
    lo = d["alpha_q025_boot"].to_numpy(float)
    hi = d["alpha_q975_boot"].to_numpy(float)
    e_low = y - lo
    e_hi = hi - y

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.errorbar(x, y, yerr=[e_low, e_hi], fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=3)
    ax.axhline(alpha_ref, color="#d62728", linestyle="--", linewidth=1.6, label=f"alpha_pred={alpha_ref:.2f}")
    ax.axhspan(1.5, 2.0, color="#2ca02c", alpha=0.12, label="SOC target band [1.5,2.0]")
    ax.set_xticks(x)
    ax.set_xticklabels(d["cluster_label"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel("alpha (MLE)")
    ax.set_title("F6c cluster-wise tail exponents")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_f6c(
    *,
    input_nc: Path,
    f5_dataset_csv: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    outdir: Path,
    scale_edges_km: list[float],
    batch_size: int,
    cfg_tail: F6bConfig,
    cfg_cluster: F6cConfig,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg_cluster.cluster_seed)

    f5_df = pd.read_csv(f5_dataset_csv).sort_values("time_index").reset_index(drop=True)
    req_cols = {"fractal_psd_beta", "fractal_variogram_slope", "residual_base_res0"}
    miss = [c for c in req_cols if c not in f5_df.columns]
    if miss:
        raise KeyError(f"F5 dataset missing columns: {miss}")

    panel_abs, conv_mean, time_ns = _extract_panel_lambda_and_conv(
        input_nc=input_nc,
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
        scale_edges_km=scale_edges_km,
        batch_size=batch_size,
    )
    nt, npix = panel_abs.shape
    if nt != len(f5_df):
        raise ValueError(f"time mismatch: panel nt={nt}, f5 rows={len(f5_df)}")

    fractal_comp = 0.5 * (
        _zscore(f5_df["fractal_psd_beta"].to_numpy(float))
        + _zscore(f5_df["fractal_variogram_slope"].to_numpy(float))
    )
    conv_z = _zscore(conv_mean.astype(float))
    exchange_z = _zscore(np.abs(f5_df["residual_base_res0"].to_numpy(float)))
    feat = np.column_stack([fractal_comp, conv_z, exchange_z])
    valid_t = np.all(np.isfinite(feat), axis=1)
    if int(np.sum(valid_t)) < max(cfg_cluster.n_clusters * 40, 200):
        raise ValueError("Too few valid time points for clustering.")

    feat_valid = feat[valid_t]
    labels_valid, centers = _kmeans(
        feat_valid,
        cfg_cluster.n_clusters,
        max_iter=cfg_cluster.cluster_max_iter,
        tol=cfg_cluster.cluster_tol,
        seed=cfg_cluster.cluster_seed,
    )
    labels = np.full(nt, -1, dtype=int)
    labels[valid_t] = labels_valid

    # Cluster summary
    cluster_rows = []
    for cid in range(cfg_cluster.n_clusters):
        mt = labels == cid
        n_t = int(np.sum(mt))
        if n_t == 0:
            cluster_rows.append(
                {
                    "cluster_id": cid,
                    "n_time": 0,
                    "fractal_z_mean": np.nan,
                    "conv_z_mean": np.nan,
                    "exchange_z_mean": np.nan,
                    "cluster_label": f"empty_{cid}",
                    "valid_for_tail": False,
                }
            )
            continue
        f_mean = float(np.mean(fractal_comp[mt]))
        c_mean = float(np.mean(conv_z[mt]))
        x_mean = float(np.mean(exchange_z[mt]))
        lbl = _label_cluster(f_mean, c_mean, x_mean) + f"_k{cid}"
        cluster_rows.append(
            {
                "cluster_id": cid,
                "n_time": n_t,
                "fractal_z_mean": f_mean,
                "conv_z_mean": c_mean,
                "exchange_z_mean": x_mean,
                "cluster_label": lbl,
                "valid_for_tail": bool(n_t >= cfg_cluster.min_cluster_time_points),
            }
        )
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(outdir / "experiment_F6c_cluster_table.csv", index=False)

    # Build fit datasets
    sample_rows = []
    best_rows = []
    all_metrics = []

    # Global (fixed cap)
    global_vals = panel_abs.reshape(-1).astype(float)
    global_vals = global_vals[np.isfinite(global_vals) & (global_vals > 0.0)]
    n_total_global = int(len(global_vals))
    if n_total_global > cfg_cluster.max_samples_global:
        idx = rng.choice(n_total_global, size=cfg_cluster.max_samples_global, replace=False)
        global_use = global_vals[idx]
    else:
        global_use = global_vals

    g_cand, g_best = _scan_xmin_candidates(global_use, regime="global", cfg=cfg_tail)
    if g_best is None:
        raise RuntimeError("Global fit unavailable in F6c.")
    g_best = dict(g_best)
    g_best["cluster_id"] = -1
    g_best["cluster_label"] = "global"
    g_best["n_total"] = n_total_global
    g_best["n_used"] = int(len(global_use))
    g_best["fit_available"] = True
    g_best["alpha_target_band_low"] = cfg_cluster.alpha_target_low
    g_best["alpha_target_band_high"] = cfg_cluster.alpha_target_high
    g_best["pass_alpha_target_band"] = bool(
        cfg_cluster.alpha_target_low <= float(g_best["alpha_mle"]) <= cfg_cluster.alpha_target_high
    )
    g_tail = np.asarray(global_use)
    g_tail = g_tail[g_tail >= float(g_best["xmin"])]
    g_lo, g_hi = _bootstrap_alpha_ci(g_tail, float(g_best["xmin"]), iters=cfg_tail.alpha_bootstrap_iters, rng=rng)
    g_best["alpha_q025_boot"] = float(g_lo)
    g_best["alpha_q975_boot"] = float(g_hi)
    best_rows.append(g_best)
    all_metrics.append(g_cand.assign(cluster_id=-1, cluster_label="global"))
    sample_rows.append({"cluster_id": -1, "cluster_label": "global", "n_total": n_total_global, "n_used": int(len(global_use))})

    # Per-cluster fits
    for _, crow in cluster_df.iterrows():
        cid = int(crow["cluster_id"])
        lbl = str(crow["cluster_label"])
        mt = labels == cid
        n_t = int(np.sum(mt))
        if n_t == 0:
            continue
        vals = panel_abs[mt].reshape(-1).astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        n_total = int(len(vals))
        if n_total == 0:
            continue
        if n_total > cfg_cluster.max_samples_cluster:
            idx = rng.choice(n_total, size=cfg_cluster.max_samples_cluster, replace=False)
            vals_use = vals[idx]
        else:
            vals_use = vals
        sample_rows.append({"cluster_id": cid, "cluster_label": lbl, "n_total": n_total, "n_used": int(len(vals_use))})

        if (n_t < cfg_cluster.min_cluster_time_points) or (len(vals_use) < cfg_tail.min_tail_points + 20):
            best_rows.append(
                {
                    "cluster_id": cid,
                    "cluster_label": lbl,
                    "n_total": n_total,
                    "n_used": int(len(vals_use)),
                    "fit_available": False,
                }
            )
            continue

        cand, best = _scan_xmin_candidates(vals_use, regime=lbl, cfg=cfg_tail)
        if cand.empty or best is None:
            best_rows.append(
                {
                    "cluster_id": cid,
                    "cluster_label": lbl,
                    "n_total": n_total,
                    "n_used": int(len(vals_use)),
                    "fit_available": False,
                }
            )
            continue

        b = dict(best)
        b["cluster_id"] = cid
        b["cluster_label"] = lbl
        b["n_total"] = n_total
        b["n_used"] = int(len(vals_use))
        b["fit_available"] = True
        b["alpha_target_band_low"] = cfg_cluster.alpha_target_low
        b["alpha_target_band_high"] = cfg_cluster.alpha_target_high
        b["pass_alpha_target_band"] = bool(
            cfg_cluster.alpha_target_low <= float(b["alpha_mle"]) <= cfg_cluster.alpha_target_high
        )
        tail = np.asarray(vals_use)
        tail = tail[tail >= float(b["xmin"])]
        lo, hi = _bootstrap_alpha_ci(tail, float(b["xmin"]), iters=cfg_tail.alpha_bootstrap_iters, rng=rng)
        b["alpha_q025_boot"] = float(lo)
        b["alpha_q975_boot"] = float(hi)
        best_rows.append(b)
        all_metrics.append(cand.assign(cluster_id=cid, cluster_label=lbl))

    samples_df = pd.DataFrame(sample_rows)
    best_df = pd.DataFrame(best_rows)
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()

    # Multiplicity correction (predeclared, no data-driven choice).
    fit_mask = (best_df["fit_available"] == True) & (best_df["cluster_id"] >= 0)  # noqa: E712
    n_tests = int(np.sum(fit_mask))
    if n_tests > 0:
        corr = best_df.loc[fit_mask, "llr_p_value"].to_numpy(float) * float(n_tests)
        corr = np.clip(corr, 0.0, 1.0)
        best_df.loc[fit_mask, "llr_p_value_corrected"] = corr
        best_df.loc[~fit_mask, "llr_p_value_corrected"] = np.nan
    else:
        best_df["llr_p_value_corrected"] = np.nan

    best_df["pass_dynamic_range"] = best_df.get("dynamic_range", np.nan) >= cfg_tail.min_dynamic_range
    best_df["pass_llr_corrected"] = (
        (best_df.get("llr_pl_vs_exp", np.nan) > 0.0)
        & np.isfinite(best_df["llr_p_value_corrected"])
        & (best_df["llr_p_value_corrected"] <= cfg_cluster.corrected_p_max)
    )
    best_df["pass_alpha_target_band"] = best_df.get("pass_alpha_target_band", False).astype(bool)
    best_df["pass_cluster_all"] = (
        best_df["fit_available"].fillna(False).astype(bool)
        & best_df["pass_dynamic_range"].fillna(False).astype(bool)
        & best_df["pass_llr_corrected"].fillna(False).astype(bool)
        & best_df["pass_alpha_target_band"].fillna(False).astype(bool)
    )

    # Global result mirrors F6b strict criteria with target band [1.5,2.0] only for comparability.
    global_row = best_df[best_df["cluster_id"] == -1].iloc[0]
    checks = {
        "global_dynamic_range_ge_10": bool(global_row["dynamic_range"] >= cfg_tail.min_dynamic_range),
        "global_llr_prefers_powerlaw": bool(
            (float(global_row["llr_pl_vs_exp"]) > 0.0)
            and np.isfinite(float(global_row["llr_p_value"]))
            and (float(global_row["llr_p_value"]) <= cfg_tail.llr_p_max)
        ),
        "global_alpha_in_target_band_1p5_2p0": bool(
            cfg_cluster.alpha_target_low <= float(global_row["alpha_mle"]) <= cfg_cluster.alpha_target_high
        ),
        "any_cluster_passes_corrected_strict": bool(
            np.any((best_df["cluster_id"] >= 0) & best_df["pass_cluster_all"])
        ),
    }
    checks["pass_all"] = bool(
        checks["global_dynamic_range_ge_10"]
        and checks["global_llr_prefers_powerlaw"]
        and checks["any_cluster_passes_corrected_strict"]
    )

    # Winner cluster: minimum |alpha-alpha_pred| among strict-passing clusters.
    pass_df = best_df[(best_df["cluster_id"] >= 0) & (best_df["pass_cluster_all"] == True)]  # noqa: E712
    if not pass_df.empty:
        d = np.abs(pass_df["alpha_mle"].to_numpy(float) - cfg_tail.alpha_pred)
        i = int(np.argmin(d))
        winner = pass_df.iloc[i].to_dict()
    else:
        winner = None

    metrics_df.to_csv(outdir / "experiment_F6c_cluster_tail_metrics.csv", index=False)
    best_df.to_csv(outdir / "experiment_F6c_cluster_best_fits.csv", index=False)
    samples_df.to_csv(outdir / "experiment_F6c_cluster_sample_sizes.csv", index=False)

    _plot_global_tail(
        x=global_use,
        xmin=float(global_row["xmin"]),
        alpha=float(global_row["alpha_mle"]),
        out_path=outdir / "plot_F6c_global_empirical_tail.png",
    )
    _plot_cluster_alpha(best_df, out_path=outdir / "plot_F6c_cluster_alpha.png", alpha_ref=cfg_tail.alpha_pred)

    verdict = {
        "checks": checks,
        "n_clusters": int(cfg_cluster.n_clusters),
        "n_cluster_tests_corrected": int(n_tests),
        "p_correction": cfg_cluster.p_correction,
        "alpha_target_band": [float(cfg_cluster.alpha_target_low), float(cfg_cluster.alpha_target_high)],
        "global_best": {
            "alpha_mle": float(global_row["alpha_mle"]),
            "xmin": float(global_row["xmin"]),
            "n_tail": int(global_row["n_tail"]),
            "ks_distance": float(global_row["ks_distance"]),
            "dynamic_range": float(global_row["dynamic_range"]),
            "llr_pl_vs_exp": float(global_row["llr_pl_vs_exp"]),
            "llr_p_value": float(global_row["llr_p_value"]),
        },
        "winner_cluster": winner,
        "inputs": {
            "input_nc": str(input_nc),
            "f5_dataset_csv": str(f5_dataset_csv),
            "m_timeseries_csv": str(m_timeseries_csv),
            "m_summary_csv": str(m_summary_csv),
            "m_mode_index_csv": str(m_mode_index_csv),
            "n_time": int(nt),
            "n_space": int(npix),
            "max_samples_global": int(cfg_cluster.max_samples_global),
            "max_samples_cluster": int(cfg_cluster.max_samples_cluster),
            "min_cluster_time_points": int(cfg_cluster.min_cluster_time_points),
            "scale_edges_km": [float(v) for v in scale_edges_km],
        },
    }
    (outdir / "experiment_F6c_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    report_lines = [
        "# Experiment F6c Report: Clustered Subspace Tail Fits",
        "",
        "## Protocol (predeclared anti-overfit)",
        "- Clustering variables are fixed before fitting: fractal surrogate (z), convective activity (z), vertical exchange proxy (z).",
        f"- Number of clusters fixed: K={cfg_cluster.n_clusters}.",
        f"- Multiple testing correction fixed: {cfg_cluster.p_correction} across cluster LLR p-values.",
        f"- Target alpha band fixed: [{cfg_cluster.alpha_target_low:.2f}, {cfg_cluster.alpha_target_high:.2f}].",
        "",
        "## Global tail (reference)",
        f"- alpha_mle = {float(global_row['alpha_mle']):.6f}",
        f"- xmin = {float(global_row['xmin']):.6e}",
        f"- dynamic_range = {float(global_row['dynamic_range']):.3f}",
        f"- LLR(PL-Exp) = {float(global_row['llr_pl_vs_exp']):.6f}",
        f"- LLR p = {float(global_row['llr_p_value']):.6e}",
        "",
        "## Strict outcomes",
        f"- any_cluster_passes_corrected_strict = {checks['any_cluster_passes_corrected_strict']}",
        f"- PASS_ALL = {checks['pass_all']}",
        "",
        "## Files",
        "- experiment_F6c_cluster_table.csv",
        "- experiment_F6c_cluster_sample_sizes.csv",
        "- experiment_F6c_cluster_tail_metrics.csv",
        "- experiment_F6c_cluster_best_fits.csv",
        "- experiment_F6c_verdict.json",
        "- plot_F6c_global_empirical_tail.png",
        "- plot_F6c_cluster_alpha.png",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
    )
    parser.add_argument(
        "--f5-dataset-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_F5_lambda_struct_fractal_era5/experiment_F5_dataset.csv"),
    )
    parser.add_argument(
        "--m-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    parser.add_argument(
        "--m-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_summary.csv"),
    )
    parser.add_argument(
        "--m-mode-index-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_mode_index.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("clean_experiments/results/experiment_F6c_clustered_subspace_tails"),
    )
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--min-cluster-time-points", type=int, default=120)
    parser.add_argument("--max-samples-global", type=int, default=2_500_000)
    parser.add_argument("--max-samples-cluster", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--scale-edges-km",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
    )
    parser.add_argument("--pmin", type=float, default=70.0)
    parser.add_argument("--pmax", type=float, default=99.0)
    parser.add_argument("--pgrid", type=int, default=120)
    parser.add_argument("--min-tail-points", type=int, default=80)
    parser.add_argument("--alpha-boot-iters", type=int, default=400)
    parser.add_argument("--alpha-target-low", type=float, default=1.5)
    parser.add_argument("--alpha-target-high", type=float, default=2.0)
    args = parser.parse_args()

    cfg_tail = F6bConfig(
        pmin=float(args.pmin),
        pmax=float(args.pmax),
        pgrid=max(10, int(args.pgrid)),
        min_tail_points=max(40, int(args.min_tail_points)),
        alpha_bootstrap_iters=max(0, int(args.alpha_boot_iters)),
    )
    cfg_cluster = F6cConfig(
        n_clusters=max(2, int(args.n_clusters)),
        min_cluster_time_points=max(20, int(args.min_cluster_time_points)),
        max_samples_global=max(200_000, int(args.max_samples_global)),
        max_samples_cluster=max(80_000, int(args.max_samples_cluster)),
        alpha_target_low=float(args.alpha_target_low),
        alpha_target_high=float(args.alpha_target_high),
    )

    verdict = run_f6c(
        input_nc=args.input_nc,
        f5_dataset_csv=args.f5_dataset_csv,
        m_timeseries_csv=args.m_timeseries_csv,
        m_summary_csv=args.m_summary_csv,
        m_mode_index_csv=args.m_mode_index_csv,
        outdir=args.out,
        scale_edges_km=[float(v) for v in args.scale_edges_km],
        batch_size=max(8, int(args.batch_size)),
        cfg_tail=cfg_tail,
        cfg_cluster=cfg_cluster,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
