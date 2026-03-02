#!/usr/bin/env python3
"""Experiment M: land-vs-ocean residual-closure check with stable calibrated setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        K_BOLTZMANN,
        _blocked_splits,
        _build_band_masks,
        _compute_vorticity,
        _edge_order,
        _evaluate_splits,
        _load_data,
        _permutation_test,
        _time_to_seconds,
        _xy_coordinates_m,
        _zscore,
    )
    from clean_experiments.experiment_O_spatial_variance import (
        _build_land_mask,
        _compute_lambda_local_batch,
        _load_m_band_amplitudes,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        K_BOLTZMANN,
        _blocked_splits,
        _build_band_masks,
        _compute_vorticity,
        _edge_order,
        _evaluate_splits,
        _load_data,
        _permutation_test,
        _time_to_seconds,
        _xy_coordinates_m,
        _zscore,
    )
    from experiment_O_spatial_variance import (  # type: ignore
        _build_land_mask,
        _compute_lambda_local_batch,
        _load_m_band_amplitudes,
    )


def _masked_mean_t(arr_tyx: np.ndarray, mask_yx: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_yx, dtype=bool)
    if int(np.sum(m)) == 0:
        raise ValueError("Mask is empty.")
    return np.mean(arr_tyx[:, m], axis=1)


def _surface_metrics(
    *,
    y: np.ndarray,
    ctrl: np.ndarray,
    lam: np.ndarray,
    ridge_alpha: float,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    valid = np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam)
    yv = y[valid]
    x_base = ctrl[valid][:, None]
    x_full = np.column_stack([ctrl[valid], lam[valid]])
    if len(yv) < max(80, n_folds * 8):
        raise ValueError(f"Too few valid points after filtering: {len(yv)}")

    splits = _blocked_splits(len(yv), n_folds=n_folds)
    split_df, yhat_b, yhat_f = _evaluate_splits(
        y=yv,
        x_base=x_base,
        x_full=x_full,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl", "lambda"],
        splits=splits,
        ridge_alpha=ridge_alpha,
    )
    mae_b = float(np.mean(np.abs(yv - yhat_b)))
    mae_f = float(np.mean(np.abs(yv - yhat_f)))
    gain = float((mae_b - mae_f) / (mae_b + 1e-12))

    p_perm, perm_df, stat_real = _permutation_test(
        y=yv,
        x_base=x_base,
        x_full=x_full,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl", "lambda"],
        permute_cols=np.array([1], dtype=int),
        splits=splits,
        ridge_alpha=ridge_alpha,
        n_perm=n_perm,
        perm_block=perm_block,
        seed=seed,
    )
    split_gains = split_df["mae_gain_frac"].to_numpy(dtype=float)
    coef_lambda = split_df["coef_full_lambda"].to_numpy(dtype=float)
    sign_consistency = float(max(np.mean(coef_lambda > 0.0), np.mean(coef_lambda < 0.0)))
    corr_l_ctrl = float(np.corrcoef(lam[valid], ctrl[valid])[0, 1]) if np.std(lam[valid]) > 1e-14 else np.nan
    corr_l_y = float(np.corrcoef(lam[valid], yv)[0, 1]) if np.std(lam[valid]) > 1e-14 else np.nan

    metrics = {
        "n_valid": float(len(yv)),
        "mae_base_oof": mae_b,
        "mae_full_oof": mae_f,
        "oof_gain_frac": gain,
        "split_gain_median": float(np.median(split_gains)),
        "split_gain_q25": float(np.quantile(split_gains, 0.25)),
        "split_gain_q75": float(np.quantile(split_gains, 0.75)),
        "perm_stat_real_median_gain": float(stat_real),
        "perm_p_value": float(p_perm),
        "lambda_sign_consistency": sign_consistency,
        "corr_lambda_ctrl": corr_l_ctrl,
        "corr_lambda_target": corr_l_y,
    }
    return metrics, split_df, perm_df


def _plot_gain(summary_df: pd.DataFrame, out_path: Path) -> None:
    d = summary_df.copy().sort_values("surface")
    x = np.arange(len(d), dtype=float)
    y = d["oof_gain_frac"].to_numpy(dtype=float)
    c = ["#8c564b" if s == "land" else "#1f77b4" for s in d["surface"].tolist()]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(x, y, color=c, alpha=0.9)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x, d["surface"].tolist())
    ax.set_ylabel("OOF gain: residual~ctrl+Lambda vs residual~ctrl")
    ax.set_title("Experiment M: gain by underlying surface")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i, y[i], f"p={float(r['perm_p_value']):.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_timeseries(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.4), sharex=True)
    for ax, var, ttl in (
        (axes[0], "residual_z", "Residual z-score (surface-mean)"),
        (axes[1], "lambda", "Lambda local mean (surface-mean)"),
    ):
        for s, col in (("land", "#8c564b"), ("ocean", "#1f77b4")):
            sub = df[df["surface"] == s]
            ax.plot(sub["time_index"].to_numpy(dtype=int), sub[var].to_numpy(dtype=float), color=col, linewidth=1.1, alpha=0.9, label=s)
        ax.set_title(ttl)
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(loc="best")
    axes[-1].set_xlabel("time_index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_experiment(
    *,
    input_nc: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    outdir: Path,
    scale_edges_km: list[float],
    ridge_alpha: float,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    seed: int,
    batch_size: int,
    precip_factor: float,
    evap_factor: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    print("[M-land-ocean] Step 1/5: load stable M-band amplitudes...", flush=True)
    band_amp, band_diag = _load_m_band_amplitudes(
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
    )
    band_diag.to_csv(outdir / "lambda_band_diagnostics.csv", index=False)

    print("[M-land-ocean] Step 2/5: load input fields...", flush=True)
    loaded = _load_data(
        input_nc,
        var_overrides={
            "iwv": None,
            "ivt_u": None,
            "ivt_v": None,
            "precip": None,
            "evap": None,
            "u": None,
            "v": None,
            "temp": None,
            "pressure": None,
            "density": None,
            "temp_pl": None,
            "q_pl": None,
            "u_pl": None,
            "v_pl": None,
            "w_pl": None,
        },
        level_dim=None,
        level_index=0,
        time_stride=1,
        lat_stride=1,
        lon_stride=1,
        max_time=None,
    )
    f = loaded.fields
    nt, ny, nx = f["iwv"].shape
    if band_amp.shape[0] != nt:
        raise ValueError(f"Length mismatch: band_amp nt={band_amp.shape[0]} vs input nt={nt}")

    time_s = _time_to_seconds(loaded.time)
    x_m, y_m = _xy_coordinates_m(loaded.lat, loaded.lon)
    dx_km = abs(float(np.median(np.diff(x_m))) / 1000.0)
    dy_km = abs(float(np.median(np.diff(y_m))) / 1000.0)
    masks, _, _, _, _, _ = _build_band_masks(
        ny=ny,
        nx=nx,
        dy_km=dy_km,
        dx_km=dx_km,
        scale_edges_km=scale_edges_km,
    )
    if len(masks) != band_amp.shape[1]:
        raise ValueError(f"Band mismatch: masks={len(masks)} vs band_amp={band_amp.shape[1]}")

    land_mask = _build_land_mask(loaded.lat, loaded.lon)
    ocean_mask = ~land_mask

    print("[M-land-ocean] Step 3/5: build residual/ctrl time series by surface...", flush=True)
    eo_t = _edge_order(len(time_s))
    eo_x = _edge_order(len(x_m))
    eo_y = _edge_order(len(y_m))
    diwv_dt = np.gradient(f["iwv"], time_s, axis=0, edge_order=eo_t)
    div_ivt = np.gradient(f["ivt_u"], x_m, axis=2, edge_order=eo_x) + np.gradient(f["ivt_v"], y_m, axis=1, edge_order=eo_y)
    p_minus_e = precip_factor * f["precip"] - evap_factor * f["evap"]
    residual_local = diwv_dt + div_ivt + p_minus_e
    del diwv_dt
    del div_ivt
    del p_minus_e

    if "density" in f:
        n_density_local = f["density"]
        density_source = "density"
    elif "pressure" in f and "temp" in f:
        n_density_local = f["pressure"] / (K_BOLTZMANN * np.maximum(f["temp"], 1e-6))
        density_source = "pressure_over_kT"
    else:
        n_density_local = np.ones_like(residual_local, dtype=float)
        density_source = "ones"

    residual_land_z = _zscore(_masked_mean_t(residual_local, land_mask))
    residual_ocean_z = _zscore(_masked_mean_t(residual_local, ocean_mask))
    ctrl_land_z = _zscore(np.log(np.maximum(_masked_mean_t(n_density_local, land_mask), 1e-30)))
    ctrl_ocean_z = _zscore(np.log(np.maximum(_masked_mean_t(n_density_local, ocean_mask), 1e-30)))

    print("[M-land-ocean] Step 4/5: reconstruct Lambda_local and aggregate by surface...", flush=True)
    lam_land = np.zeros(nt, dtype=float)
    lam_ocean = np.zeros(nt, dtype=float)
    for start in range(0, nt, batch_size):
        stop = min(start + batch_size, nt)
        u_blk = np.asarray(f["u"][start:stop], dtype=float)
        v_blk = np.asarray(f["v"][start:stop], dtype=float)
        zeta_blk = _compute_vorticity(u=u_blk, v=v_blk, x_m=x_m, y_m=y_m)
        lam_blk = _compute_lambda_local_batch(
            zeta_batch=zeta_blk,
            masks=masks,
            band_amp_batch=band_amp[start:stop],
        )
        lam_land[start:stop] = _masked_mean_t(lam_blk, land_mask)
        lam_ocean[start:stop] = _masked_mean_t(lam_blk, ocean_mask)
        print(f"[M-land-ocean] lambda progress: {stop}/{nt}", flush=True)

    rows = []
    split_rows = []
    perm_rows = []
    ts_rows = []
    surfaces = {
        "land": (residual_land_z, ctrl_land_z, lam_land, land_mask),
        "ocean": (residual_ocean_z, ctrl_ocean_z, lam_ocean, ocean_mask),
    }
    for i, (name, (y, ctrl, lam, msk)) in enumerate(surfaces.items()):
        metrics, split_df, perm_df = _surface_metrics(
            y=y,
            ctrl=ctrl,
            lam=lam,
            ridge_alpha=ridge_alpha,
            n_folds=n_folds,
            n_perm=n_perm,
            perm_block=perm_block,
            seed=seed + 17 * i,
        )
        rows.append(
            {
                "surface": name,
                "n_cells": int(np.sum(msk)),
                "n_time": int(len(y)),
                "density_source": density_source,
                **metrics,
            }
        )
        split_df = split_df.copy()
        split_df.insert(0, "surface", name)
        split_rows.append(split_df)
        perm_df = perm_df.copy()
        perm_df.insert(0, "surface", name)
        perm_rows.append(perm_df)
        for t in range(len(y)):
            ts_rows.append(
                {
                    "time_index": int(t),
                    "surface": name,
                    "residual_z": float(y[t]),
                    "ctrl_z": float(ctrl[t]),
                    "lambda": float(lam[t]),
                }
            )

    summary_df = pd.DataFrame(rows).sort_values("surface").reset_index(drop=True)
    splits_df = pd.concat(split_rows, ignore_index=True)
    perm_df = pd.concat(perm_rows, ignore_index=True)
    ts_df = pd.DataFrame(ts_rows)

    gain_land = float(summary_df.loc[summary_df["surface"] == "land", "oof_gain_frac"].iloc[0])
    gain_ocean = float(summary_df.loc[summary_df["surface"] == "ocean", "oof_gain_frac"].iloc[0])
    delta = gain_land - gain_ocean

    compare = {
        "gain_land": gain_land,
        "gain_ocean": gain_ocean,
        "delta_land_minus_ocean": float(delta),
        "land_gt_ocean": bool(gain_land > gain_ocean),
    }
    if abs(gain_ocean) > 1e-12:
        compare["land_to_ocean_gain_ratio"] = float(gain_land / gain_ocean)
    else:
        compare["land_to_ocean_gain_ratio"] = float("nan")

    summary_df.to_csv(outdir / "surface_summary.csv", index=False)
    splits_df.to_csv(outdir / "surface_splits.csv", index=False)
    perm_df.to_csv(outdir / "surface_permutation.csv", index=False)
    ts_df.to_csv(outdir / "surface_timeseries.csv", index=False)
    with (outdir / "surface_comparison.json").open("w", encoding="utf-8") as fjs:
        json.dump(compare, fjs, ensure_ascii=False, indent=2)

    _plot_gain(summary_df, outdir / "plot_gain_land_vs_ocean.png")
    _plot_timeseries(ts_df, outdir / "plot_timeseries_land_ocean.png")
    plt.figure(figsize=(6.0, 4.8))
    plt.imshow(land_mask.astype(float), origin="lower", aspect="auto", cmap="Greens")
    plt.title("Land mask (1=land, 0=ocean)")
    plt.xlabel("lon index")
    plt.ylabel("lat index")
    plt.colorbar(shrink=0.85)
    plt.tight_layout()
    plt.savefig(outdir / "plot_land_mask.png", dpi=170)
    plt.close()

    lines = [
        "# Experiment M: Land/Ocean Split",
        "",
        "Stable base configuration reused from experiment_M_cosmo_flow_v4_macro_calibrated.",
        "",
        "## Result",
        f"- gain_land = {gain_land:.6f}",
        f"- gain_ocean = {gain_ocean:.6f}",
        f"- delta (land - ocean) = {delta:.6f}",
        f"- land > ocean: {bool(gain_land > gain_ocean)}",
        "",
        "## Surface details",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"- {r['surface']}: gain={float(r['oof_gain_frac']):.6f}, "
            f"perm_p={float(r['perm_p_value']):.6f}, "
            f"sign_consistency={float(r['lambda_sign_consistency']):.3f}, "
            f"corr(lambda,ctrl)={float(r['corr_lambda_ctrl']):.6f}"
        )
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[M-land-ocean] done -> {outdir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
    )
    p.add_argument(
        "--m-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    p.add_argument(
        "--m-summary-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_summary.csv"),
    )
    p.add_argument(
        "--m-mode-index-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_mode_index.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_land_ocean_split"),
    )
    p.add_argument(
        "--scale-edges-km",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
    )
    p.add_argument("--ridge-alpha", type=float, default=1e-6)
    p.add_argument("--n-folds", type=int, default=6)
    p.add_argument("--n-perm", type=int, default=140)
    p.add_argument("--perm-block", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260301)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--precip-factor", type=float, default=1.0)
    p.add_argument("--evap-factor", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        input_nc=args.input_nc,
        m_timeseries_csv=args.m_timeseries_csv,
        m_summary_csv=args.m_summary_csv,
        m_mode_index_csv=args.m_mode_index_csv,
        outdir=args.outdir,
        scale_edges_km=list(args.scale_edges_km),
        ridge_alpha=args.ridge_alpha,
        n_folds=args.n_folds,
        n_perm=args.n_perm,
        perm_block=args.perm_block,
        seed=args.seed,
        batch_size=args.batch_size,
        precip_factor=args.precip_factor,
        evap_factor=args.evap_factor,
    )


if __name__ == "__main__":
    main()
