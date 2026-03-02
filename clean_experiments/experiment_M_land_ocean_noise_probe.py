#!/usr/bin/env python3
"""Diagnostics for land/ocean paradox in Experiment M (noise and detectability checks)."""

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
        _build_band_masks,
        _compute_vorticity,
        _edge_order,
        _load_data,
        _time_to_seconds,
        _xy_coordinates_m,
        _zscore,
    )
    from clean_experiments.experiment_M_land_ocean_split import _masked_mean_t, _surface_metrics
    from clean_experiments.experiment_O_spatial_variance import (
        _build_land_mask,
        _compute_lambda_local_batch,
        _load_m_band_amplitudes,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        K_BOLTZMANN,
        _build_band_masks,
        _compute_vorticity,
        _edge_order,
        _load_data,
        _time_to_seconds,
        _xy_coordinates_m,
        _zscore,
    )
    from experiment_M_land_ocean_split import _masked_mean_t, _surface_metrics  # type: ignore
    from experiment_O_spatial_variance import (  # type: ignore
        _build_land_mask,
        _compute_lambda_local_batch,
        _load_m_band_amplitudes,
    )


def _roll_mean(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(window=window, min_periods=1).mean().to_numpy(dtype=float)


def _plot_gain(grid_df: pd.DataFrame, out_path: Path) -> None:
    targets = ["residual_full", "residual_no_pe", "p_minus_e_only", "residual_full_roll4"]
    surfaces = ["land", "ocean"]
    x = np.arange(len(targets), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = {"land": "#8c564b", "ocean": "#1f77b4"}
    for i, s in enumerate(surfaces):
        vals = []
        for t in targets:
            sub = grid_df[(grid_df["surface"] == s) & (grid_df["target"] == t)]
            vals.append(float(sub["oof_gain_frac"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=colors[s], alpha=0.9, label=s)
    ax.axhline(0.0, color="#444", linewidth=1.0)
    ax.set_xticks(x, ["full", "no P-E", "P-E only", "full roll4"])
    ax.set_ylabel("OOF gain")
    ax.set_title("Gain sensitivity by target definition")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_components(comp_df: pd.DataFrame, out_path: Path) -> None:
    m = comp_df.melt(id_vars=["surface"], value_vars=["std_diwv_dt", "std_div_ivt", "std_p_minus_e"], var_name="component", value_name="std")
    labels = ["dIWV/dt", "div(IVT)", "P-E"]
    comp_order = ["std_diwv_dt", "std_div_ivt", "std_p_minus_e"]
    x = np.arange(len(comp_order), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, s in enumerate(["land", "ocean"]):
        sub = m[m["surface"] == s].set_index("component")
        vals = [float(sub.loc[c, "std"]) for c in comp_order]
        ax.bar(x + (i - 0.5) * width, vals, width=width, alpha=0.9, label=s, color=("#8c564b" if s == "land" else "#1f77b4"))
    ax.set_xticks(x, labels)
    ax.set_ylabel("Std of surface-mean component")
    ax.set_title("Residual component variability by surface")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
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

    print("[M-noise-probe] Step 1/5: load data and stable M amplitudes...", flush=True)
    band_amp, _ = _load_m_band_amplitudes(
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
    )
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
        raise ValueError(f"Length mismatch: band_amp nt={band_amp.shape[0]} vs nt={nt}")

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

    print("[M-noise-probe] Step 2/5: build component time series by surface...", flush=True)
    eo_t = _edge_order(len(time_s))
    eo_x = _edge_order(len(x_m))
    eo_y = _edge_order(len(y_m))
    diwv_dt = np.gradient(f["iwv"], time_s, axis=0, edge_order=eo_t)
    div_ivt = np.gradient(f["ivt_u"], x_m, axis=2, edge_order=eo_x) + np.gradient(f["ivt_v"], y_m, axis=1, edge_order=eo_y)
    p_minus_e = precip_factor * f["precip"] - evap_factor * f["evap"]

    if "density" in f:
        n_density_local = f["density"]
    elif "pressure" in f and "temp" in f:
        n_density_local = f["pressure"] / (K_BOLTZMANN * np.maximum(f["temp"], 1e-6))
    else:
        n_density_local = np.ones_like(f["iwv"], dtype=float)

    series = {}
    for s_name, m in (("land", land_mask), ("ocean", ocean_mask)):
        di = _masked_mean_t(diwv_dt, m)
        dv = _masked_mean_t(div_ivt, m)
        pe = _masked_mean_t(p_minus_e, m)
        full = di + dv + pe
        no_pe = di + dv
        ctrl = _zscore(np.log(np.maximum(_masked_mean_t(n_density_local, m), 1e-30)))
        series[s_name] = {
            "diwv_dt": np.asarray(di, dtype=float),
            "div_ivt": np.asarray(dv, dtype=float),
            "p_minus_e": np.asarray(pe, dtype=float),
            "residual_full": np.asarray(full, dtype=float),
            "residual_no_pe": np.asarray(no_pe, dtype=float),
            "ctrl": np.asarray(ctrl, dtype=float),
        }

    print("[M-noise-probe] Step 3/5: reconstruct Lambda local means...", flush=True)
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
        print(f"[M-noise-probe] lambda progress: {stop}/{nt}", flush=True)
    series["land"]["lambda"] = lam_land
    series["ocean"]["lambda"] = lam_ocean

    print("[M-noise-probe] Step 4/5: run ML checks across target variants...", flush=True)
    target_defs = {
        "residual_full": lambda s: _zscore(s["residual_full"]),
        "residual_no_pe": lambda s: _zscore(s["residual_no_pe"]),
        "p_minus_e_only": lambda s: _zscore(s["p_minus_e"]),
        "residual_full_roll4": lambda s: _zscore(_roll_mean(s["residual_full"], window=4)),
    }

    rows = []
    split_rows = []
    perm_rows = []
    comp_rows = []
    for si, surface in enumerate(("land", "ocean")):
        s = series[surface]
        lam = np.asarray(s["lambda"], dtype=float)
        ctrl = np.asarray(s["ctrl"], dtype=float)
        comp_rows.append(
            {
                "surface": surface,
                "std_diwv_dt": float(np.std(s["diwv_dt"])),
                "std_div_ivt": float(np.std(s["div_ivt"])),
                "std_p_minus_e": float(np.std(s["p_minus_e"])),
                "std_residual_full": float(np.std(s["residual_full"])),
                "corr_lambda_diwv_dt": float(np.corrcoef(lam, s["diwv_dt"])[0, 1]) if np.std(lam) > 1e-14 else np.nan,
                "corr_lambda_div_ivt": float(np.corrcoef(lam, s["div_ivt"])[0, 1]) if np.std(lam) > 1e-14 else np.nan,
                "corr_lambda_p_minus_e": float(np.corrcoef(lam, s["p_minus_e"])[0, 1]) if np.std(lam) > 1e-14 else np.nan,
            }
        )
        for ti, (target_name, fn) in enumerate(target_defs.items()):
            y = fn(s)
            metrics, split_df, perm_df = _surface_metrics(
                y=y,
                ctrl=ctrl,
                lam=lam,
                ridge_alpha=ridge_alpha,
                n_folds=n_folds,
                n_perm=n_perm,
                perm_block=perm_block,
                seed=seed + 101 * si + 13 * ti,
            )
            rows.append(
                {
                    "surface": surface,
                    "target": target_name,
                    **metrics,
                }
            )
            split_df = split_df.copy()
            split_df.insert(0, "target", target_name)
            split_df.insert(0, "surface", surface)
            split_rows.append(split_df)
            perm_df = perm_df.copy()
            perm_df.insert(0, "target", target_name)
            perm_df.insert(0, "surface", surface)
            perm_rows.append(perm_df)

    res_df = pd.DataFrame(rows)
    comp_df = pd.DataFrame(comp_rows)
    splits_df = pd.concat(split_rows, ignore_index=True)
    perms_df = pd.concat(perm_rows, ignore_index=True)

    res_df.to_csv(outdir / "noise_probe_metrics.csv", index=False)
    comp_df.to_csv(outdir / "noise_probe_components.csv", index=False)
    splits_df.to_csv(outdir / "noise_probe_splits.csv", index=False)
    perms_df.to_csv(outdir / "noise_probe_permutation.csv", index=False)

    _plot_gain(res_df, outdir / "plot_noise_probe_gains.png")
    _plot_components(comp_df, outdir / "plot_noise_probe_component_std.png")

    # Compact verdict around colleagues' hypothesis.
    def _pick(surface: str, target: str, col: str) -> float:
        return float(res_df.loc[(res_df["surface"] == surface) & (res_df["target"] == target), col].iloc[0])

    verdict = {
        "full_target_gain_land": _pick("land", "residual_full", "oof_gain_frac"),
        "full_target_gain_ocean": _pick("ocean", "residual_full", "oof_gain_frac"),
        "no_pe_gain_land": _pick("land", "residual_no_pe", "oof_gain_frac"),
        "no_pe_gain_ocean": _pick("ocean", "residual_no_pe", "oof_gain_frac"),
        "pe_only_gain_land": _pick("land", "p_minus_e_only", "oof_gain_frac"),
        "pe_only_gain_ocean": _pick("ocean", "p_minus_e_only", "oof_gain_frac"),
        "full_roll4_gain_land": _pick("land", "residual_full_roll4", "oof_gain_frac"),
        "full_roll4_gain_ocean": _pick("ocean", "residual_full_roll4", "oof_gain_frac"),
    }
    with (outdir / "noise_probe_verdict.json").open("w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    lines = [
        "# Experiment M Land/Ocean Noise Probe",
        "",
        "Hypothesis checked: weak/negative land gain may be driven by noisy P-E closure and high-frequency residual noise.",
        "",
        "## Gains",
    ]
    for _, r in res_df.sort_values(["target", "surface"]).iterrows():
        lines.append(
            f"- {r['target']} / {r['surface']}: gain={float(r['oof_gain_frac']):.6f}, "
            f"perm_p={float(r['perm_p_value']):.6f}"
        )
    lines.append("")
    lines.append("## Component std by surface")
    for _, r in comp_df.sort_values("surface").iterrows():
        lines.append(
            f"- {r['surface']}: std(dIWV/dt)={float(r['std_diwv_dt']):.6e}, "
            f"std(divIVT)={float(r['std_div_ivt']):.6e}, std(P-E)={float(r['std_p_minus_e']):.6e}"
        )
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[M-noise-probe] done -> {outdir}", flush=True)


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
        default=Path("clean_experiments/results/experiment_M_land_ocean_noise_probe"),
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

