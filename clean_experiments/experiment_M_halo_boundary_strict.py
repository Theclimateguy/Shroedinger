#!/usr/bin/env python3
"""Strict halo-context boundary experiment for Experiment M.

Goal:
- build closure target only on the interior core C of WPWP domain,
- allow context features from halo ring H around C,
- test boundary bath observables (coarse->fine, fine->coarse exchange),
- run strict train/test/external protocol with blocked permutations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        _block_permute,
        _build_band_masks,
        _compute_vorticity,
        _fit_ridge_scaled,
        _load_data,
        _time_to_seconds,
        _xy_coordinates_m,
        _zscore,
    )
    from clean_experiments.experiment_M_gksl_hybrid_bridge import (
        _build_gksl_proxy,
        _phi_candidate_map,
        _read_gksl_config,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        _block_permute,
        _build_band_masks,
        _compute_vorticity,
        _fit_ridge_scaled,
        _load_data,
        _time_to_seconds,
        _xy_coordinates_m,
        _zscore,
    )
    from experiment_M_gksl_hybrid_bridge import (  # type: ignore
        _build_gksl_proxy,
        _phi_candidate_map,
        _read_gksl_config,
    )


EPS = 1e-12
K_BOLTZMANN = 1.380649e-23


def _safe_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    yt = np.asarray(y, dtype=float)
    yp = np.asarray(yhat, dtype=float)
    if len(yt) == 0:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    y_mu = float(np.mean(yt))
    ss_tot = float(np.sum((yt - y_mu) ** 2))
    if ss_tot < EPS:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _masked_mean(arr_tyx: np.ndarray, mask_yx: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(arr_tyx, dtype=float)
    mask = np.asarray(mask_yx, dtype=bool)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (time,lat,lon), got {arr.shape}")
    if mask.shape != arr.shape[1:]:
        raise ValueError(f"Mask shape mismatch: mask={mask.shape}, arr={arr.shape}")
    if int(np.sum(mask)) == 0:
        if allow_empty:
            return np.zeros(arr.shape[0], dtype=float)
        raise ValueError("Mask has zero cells.")
    return np.mean(arr[:, mask], axis=1)


def _build_core_mask(ny: int, nx: int, core_margin: int) -> tuple[np.ndarray, dict[str, int]]:
    m = int(core_margin)
    if m < 1:
        raise ValueError(f"core_margin must be >=1, got {m}")
    if 2 * m >= ny or 2 * m >= nx:
        raise ValueError(f"core_margin={m} too large for grid ({ny},{nx})")

    core = np.zeros((ny, nx), dtype=bool)
    y0, y1 = m, ny - m
    x0, x1 = m, nx - m
    core[y0:y1, x0:x1] = True
    bounds = {"core_y0": y0, "core_y1": y1, "core_x0": x0, "core_x1": x1}
    return core, bounds


def _local_halo_ring(
    *,
    ny: int,
    nx: int,
    core_bounds: dict[str, int],
    halo_width: int,
) -> np.ndarray:
    w = int(halo_width)
    if w < 0:
        raise ValueError(f"halo_width must be >=0, got {w}")
    ring = np.zeros((ny, nx), dtype=bool)
    if w == 0:
        return ring

    y0 = int(core_bounds["core_y0"])
    y1 = int(core_bounds["core_y1"])
    x0 = int(core_bounds["core_x0"])
    x1 = int(core_bounds["core_x1"])
    yy0 = max(0, y0 - w)
    yy1 = min(ny, y1 + w)
    xx0 = max(0, x0 - w)
    xx1 = min(nx, x1 + w)
    ring[yy0:yy1, xx0:xx1] = True
    ring[y0:y1, x0:x1] = False
    return ring


def _build_halo_mask(
    *,
    ny: int,
    nx: int,
    core_mask: np.ndarray,
    core_bounds: dict[str, int],
    halo_width: int,
    halo_mode: str,
    remote_exclusion_cells: int,
    misalign_shift_y: int,
    misalign_shift_x: int,
    seed: int,
) -> np.ndarray:
    local = _local_halo_ring(ny=ny, nx=nx, core_bounds=core_bounds, halo_width=int(halo_width))
    if halo_mode == "local":
        return local
    if int(np.sum(local)) == 0:
        return np.zeros((ny, nx), dtype=bool)

    if halo_mode == "misaligned":
        shifted = np.roll(np.roll(local, int(misalign_shift_y), axis=0), int(misalign_shift_x), axis=1)
        shifted &= ~core_mask
        return shifted

    if halo_mode == "remote":
        y0 = int(core_bounds["core_y0"])
        y1 = int(core_bounds["core_y1"])
        x0 = int(core_bounds["core_x0"])
        x1 = int(core_bounds["core_x1"])
        ex = int(max(remote_exclusion_cells, 0))
        yy0 = max(0, y0 - ex)
        yy1 = min(ny, y1 + ex)
        xx0 = max(0, x0 - ex)
        xx1 = min(nx, x1 + ex)

        forbidden = np.zeros((ny, nx), dtype=bool)
        forbidden[yy0:yy1, xx0:xx1] = True
        forbidden |= core_mask

        cand = np.argwhere(~forbidden)
        need = int(np.sum(local))
        if len(cand) < need:
            cand = np.argwhere((~core_mask) & (~local))
        if len(cand) < need:
            cand = np.argwhere(~core_mask)
        if len(cand) < need:
            raise ValueError(f"Not enough cells for remote halo: need={need}, have={len(cand)}")
        rng = np.random.default_rng(int(seed) + 777)
        pick = rng.choice(len(cand), size=need, replace=False)
        out = np.zeros((ny, nx), dtype=bool)
        out[cand[pick, 0], cand[pick, 1]] = True
        out &= ~core_mask
        return out

    raise ValueError(f"Unknown halo_mode='{halo_mode}'. Use local|remote|misaligned.")


def _edge_order(size: int) -> int:
    return 2 if size >= 3 else 1


def _compute_budget_core_halo(
    *,
    iwv: np.ndarray,
    ivt_u: np.ndarray,
    ivt_v: np.ndarray,
    precip: np.ndarray,
    evap: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    core_mask: np.ndarray,
    halo_mask: np.ndarray,
    precip_factor: float,
    evap_factor: float,
) -> dict[str, np.ndarray]:
    eo_t = _edge_order(len(time_s))
    eo_x = _edge_order(len(x_m))
    eo_y = _edge_order(len(y_m))

    diwv_dt = np.gradient(iwv, time_s, axis=0, edge_order=eo_t)
    div_ivt = np.gradient(ivt_u, x_m, axis=2, edge_order=eo_x) + np.gradient(ivt_v, y_m, axis=1, edge_order=eo_y)
    p_minus_e = precip_factor * precip - evap_factor * evap
    residual_map = diwv_dt + div_ivt + p_minus_e

    out = {
        "res_core_raw": _masked_mean(residual_map, core_mask),
        "res_halo_raw": _masked_mean(residual_map, halo_mask, allow_empty=True),
        "diwv_core_raw": _masked_mean(diwv_dt, core_mask),
        "div_core_raw": _masked_mean(div_ivt, core_mask),
        "pme_core_raw": _masked_mean(p_minus_e, core_mask),
    }
    return out


def _compute_band_ring_signals(
    *,
    vorticity: np.ndarray,
    mask_fine: np.ndarray,
    mask_coarse: np.ndarray,
    core_mask: np.ndarray,
    halo_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    nt, ny, nx = vorticity.shape
    has_halo = int(np.sum(halo_mask)) > 0
    fine_core = np.zeros(nt, dtype=float)
    fine_halo = np.zeros(nt, dtype=float)
    coarse_core = np.zeros(nt, dtype=float)
    coarse_halo = np.zeros(nt, dtype=float)

    for t in range(nt):
        z = np.asarray(vorticity[t], dtype=float)
        fhat = np.fft.rfft2(z - np.mean(z))

        fine_map = np.fft.irfft2(fhat * mask_fine, s=(ny, nx)).real
        coarse_map = np.fft.irfft2(fhat * mask_coarse, s=(ny, nx)).real

        af = np.abs(fine_map)
        ac = np.abs(coarse_map)
        fine_core[t] = float(np.mean(af[core_mask]))
        coarse_core[t] = float(np.mean(ac[core_mask]))
        if has_halo:
            fine_halo[t] = float(np.mean(af[halo_mask]))
            coarse_halo[t] = float(np.mean(ac[halo_mask]))
        else:
            fine_halo[t] = 0.0
            coarse_halo[t] = 0.0

    return {
        "fine_core": fine_core,
        "fine_halo": fine_halo,
        "coarse_core": coarse_core,
        "coarse_halo": coarse_halo,
    }


def _parse_time_hours(time_like: pd.Series) -> np.ndarray:
    t = pd.to_datetime(time_like, errors="coerce", utc=True)
    if t.isna().any():
        return np.arange(len(time_like), dtype=float) * 6.0
    return (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0


def _perm_p_fixed_model(
    *,
    y_eval: np.ndarray,
    yhat_base: np.ndarray,
    x_eval_full: np.ndarray,
    coef_full: np.ndarray,
    intercept_full: float,
    added_cols: np.ndarray,
    n_perm: int,
    perm_block: int,
    seed: int,
) -> tuple[float, float, pd.DataFrame]:
    yhat_real = intercept_full + x_eval_full @ coef_full
    mae_base = float(np.mean(np.abs(y_eval - yhat_base)))
    mae_full = float(np.mean(np.abs(y_eval - yhat_real)))
    real_gain = float((mae_base - mae_full) / (mae_base + EPS))

    rng = np.random.default_rng(seed)
    rows = []
    count_ge = 0
    for pid in range(int(n_perm)):
        x_perm = np.asarray(x_eval_full, dtype=float).copy()
        x_perm[:, added_cols] = _block_permute(x_perm[:, added_cols], block=int(perm_block), rng=rng)
        yhat_perm = intercept_full + x_perm @ coef_full
        mae_perm = float(np.mean(np.abs(y_eval - yhat_perm)))
        gain_perm = float((mae_base - mae_perm) / (mae_base + EPS))
        rows.append({"perm_id": int(pid), "gain_perm": gain_perm})
        if gain_perm >= real_gain:
            count_ge += 1
    p = float((count_ge + 1) / (int(n_perm) + 1))
    return p, real_gain, pd.DataFrame(rows)


def _evaluate_split(
    *,
    df: pd.DataFrame,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    model_specs: list[tuple[str, list[str], bool]],
    ridge_alpha: float,
    n_perm: int,
    perm_block: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    y_all = np.asarray(df["target_core"], dtype=float)

    all_cols = sorted({c for _, cols, _ in model_specs for c in cols})
    valid = np.isfinite(y_all)
    for c in all_cols:
        valid &= np.isfinite(np.asarray(df[c], dtype=float))

    rows = []
    pred = {"time_index": [], "time": [], "y": []}
    perm_tables: dict[str, pd.DataFrame] = {}
    perm_real_rows = []

    for model_id, (model_name, cols, use_shuffle) in enumerate(model_specs):
        mask_model = valid & train_mask
        tr_idx = np.where(mask_model)[0]
        te_idx = np.where(valid & eval_mask)[0]
        if len(tr_idx) < 200 or len(te_idx) < 80:
            rows.append(
                {
                    "model": model_name,
                    "n_train": int(len(tr_idx)),
                    "n_eval": int(len(te_idx)),
                    "mae": np.nan,
                    "r2": np.nan,
                    "gain_vs_core": np.nan,
                    "perm_p_abs": np.nan,
                }
            )
            continue

        y_tr = y_all[tr_idx]
        y_te = y_all[te_idx]

        x_core_tr = np.asarray(df.loc[tr_idx, ["ctrl_core"]], dtype=float)
        x_core_te = np.asarray(df.loc[te_idx, ["ctrl_core"]], dtype=float)
        _, _, yhat_core = _fit_ridge_scaled(x_core_tr, y_tr, x_core_te, float(ridge_alpha))

        x_tr = np.asarray(df.loc[tr_idx, cols], dtype=float).copy()
        x_te = np.asarray(df.loc[te_idx, cols], dtype=float).copy()
        if use_shuffle and x_tr.shape[1] > 1:
            rng_sh = np.random.default_rng(int(seed) + 1000 + int(model_id))
            for j in range(1, x_tr.shape[1]):
                x_tr[:, j] = _block_permute(x_tr[:, j : j + 1], block=int(perm_block), rng=rng_sh).reshape(-1)
                x_te[:, j] = _block_permute(x_te[:, j : j + 1], block=int(perm_block), rng=rng_sh).reshape(-1)

        coef, intercept, yhat = _fit_ridge_scaled(x_tr, y_tr, x_te, float(ridge_alpha))

        mae_core = float(np.mean(np.abs(y_te - yhat_core)))
        mae = float(np.mean(np.abs(y_te - yhat)))
        gain = float((mae_core - mae) / (mae_core + EPS))

        p_abs = np.nan
        real_gain = np.nan
        if x_te.shape[1] > 1 and int(n_perm) > 0:
            p_abs, real_gain, perm_df = _perm_p_fixed_model(
                y_eval=y_te,
                yhat_base=yhat_core,
                x_eval_full=x_te,
                coef_full=np.asarray(coef, dtype=float),
                intercept_full=float(intercept),
                added_cols=np.arange(1, x_te.shape[1], dtype=int),
                n_perm=int(n_perm),
                perm_block=int(perm_block),
                seed=int(seed) + 11 + int(model_id) * 17,
            )
            perm_tables[f"perm_{model_name}"] = perm_df
        perm_real_rows.append({"model": model_name, "real_gain_vs_core": real_gain, "perm_p_abs": p_abs})

        rows.append(
            {
                "model": model_name,
                "n_train": int(len(tr_idx)),
                "n_eval": int(len(te_idx)),
                "mae": mae,
                "r2": _safe_r2(y_te, yhat),
                "gain_vs_core": gain,
                "perm_p_abs": p_abs,
            }
        )

        pred[f"yhat_{model_name}"] = np.full(len(df), np.nan, dtype=float)
        pred[f"yhat_{model_name}"][te_idx] = yhat
        if "yhat_ERA_core" not in pred:
            pred["yhat_ERA_core"] = np.full(len(df), np.nan, dtype=float)
            pred["yhat_ERA_core"][te_idx] = yhat_core

    pred["time_index"] = np.asarray(df["time_index"], dtype=int)
    pred["time"] = df["time"].astype(str).to_numpy()
    pred["y"] = y_all
    pred_df = pd.DataFrame(pred)
    return pd.DataFrame(rows), pred_df, perm_tables, pd.DataFrame(perm_real_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2021_experiment_M_input.nc"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021"),
    )
    p.add_argument(
        "--gksl-final-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_memory_gksl_cptp/gksl_final_l8.csv"),
    )
    p.add_argument("--gksl-config-id", type=str, default="G001")
    p.add_argument("--phi-feature", type=str, default="raw", choices=["raw", "popdiff", "coh", "eta", "raw_x_eta", "raw_plus_pop", "dphi", "pop_x_delta"])
    p.add_argument("--gksl-dt-cap-hours", type=float, default=6.0)
    p.add_argument("--train-end-year", type=int, default=2019)
    p.add_argument("--test-year", type=int, default=2020)
    p.add_argument("--external-year", type=int, default=2021)
    p.add_argument("--core-margin-cells", type=int, default=10)
    p.add_argument("--halo-width-cells", type=int, default=8)
    p.add_argument("--halo-mode", type=str, default="local", choices=["local", "remote", "misaligned"])
    p.add_argument("--remote-exclusion-cells", type=int, default=12)
    p.add_argument("--misalign-shift-y", type=int, default=12)
    p.add_argument("--misalign-shift-x", type=int, default=18)
    p.add_argument("--fine-band-idx", type=int, default=0)
    p.add_argument("--coarse-band-idx", type=int, default=1)
    p.add_argument("--scale-edges-km", default="25,50,100,200,400,800,1600")
    p.add_argument("--target-mode", type=str, default="physical_zscore", choices=["physical_zscore", "physical_raw"])
    p.add_argument("--precip-factor", type=float, default=1.0)
    p.add_argument("--evap-factor", type=float, default=1.0)
    p.add_argument("--ridge-alpha", type=float, default=1e-6)
    p.add_argument("--n-perm", type=int, default=1999)
    p.add_argument("--perm-block", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260307)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--lat-stride", type=int, default=1)
    p.add_argument("--lon-stride", type=int, default=1)
    p.add_argument("--max-time", type=int, default=None)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    var_overrides = {
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
    }

    loaded = _load_data(
        args.input_nc,
        var_overrides=var_overrides,
        level_dim=None,
        level_index=0,
        time_stride=int(args.time_stride),
        lat_stride=int(args.lat_stride),
        lon_stride=int(args.lon_stride),
        max_time=args.max_time,
    )

    time = loaded.time
    lat = loaded.lat
    lon = loaded.lon
    f = loaded.fields
    iwv = f["iwv"]
    ivt_u = f["ivt_u"]
    ivt_v = f["ivt_v"]
    precip = f["precip"]
    evap = f["evap"]
    u = f["u"]
    v = f["v"]
    temp = f.get("temp")
    pressure = f.get("pressure")

    nt, ny, nx = iwv.shape
    core_mask, core_bounds = _build_core_mask(ny, nx, int(args.core_margin_cells))
    halo_mask = _build_halo_mask(
        ny=ny,
        nx=nx,
        core_mask=core_mask,
        core_bounds=core_bounds,
        halo_width=int(args.halo_width_cells),
        halo_mode=str(args.halo_mode),
        remote_exclusion_cells=int(args.remote_exclusion_cells),
        misalign_shift_y=int(args.misalign_shift_y),
        misalign_shift_x=int(args.misalign_shift_x),
        seed=int(args.seed),
    )
    core_frac = float(np.mean(core_mask))
    halo_frac = float(np.mean(halo_mask))
    halo_n_cells = int(np.sum(halo_mask))

    time_s = _time_to_seconds(time)
    x_m, y_m = _xy_coordinates_m(lat, lon)

    budget = _compute_budget_core_halo(
        iwv=iwv,
        ivt_u=ivt_u,
        ivt_v=ivt_v,
        precip=precip,
        evap=evap,
        time_s=time_s,
        x_m=x_m,
        y_m=y_m,
        core_mask=core_mask,
        halo_mask=halo_mask,
        precip_factor=float(args.precip_factor),
        evap_factor=float(args.evap_factor),
    )
    target_core_raw = np.asarray(budget["res_core_raw"], dtype=float)
    if args.target_mode == "physical_zscore":
        target_core = _zscore(target_core_raw)
    else:
        target_core = target_core_raw

    pme = float(args.precip_factor) * precip - float(args.evap_factor) * evap
    ivt_mag = np.sqrt(np.maximum(ivt_u * ivt_u + ivt_v * ivt_v, 0.0))
    iwv_halo = _masked_mean(iwv, halo_mask, allow_empty=True)
    ivtmag_halo = _masked_mean(ivt_mag, halo_mask, allow_empty=True)
    pme_halo = _masked_mean(pme, halo_mask, allow_empty=True)
    residual_halo = np.asarray(budget["res_halo_raw"], dtype=float)

    if temp is not None and pressure is not None:
        n_density = pressure / (K_BOLTZMANN * np.maximum(temp, 1e-6))
        n_density_core = _masked_mean(n_density, core_mask)
        ctrl_core = _zscore(np.log(np.maximum(n_density_core, 1e-30)))
        density_source = "pressure_over_kT_core"
    else:
        ctrl_core = _zscore(_masked_mean(iwv, core_mask))
        density_source = "iwv_core_fallback"

    vorticity = _compute_vorticity(u=u, v=v, x_m=x_m, y_m=y_m)
    dx_km = float(np.median(np.diff(x_m))) / 1000.0
    dy_km = float(np.median(np.diff(y_m))) / 1000.0
    edges = [float(x.strip()) for x in str(args.scale_edges_km).split(",") if x.strip()]
    masks, _, _, centers, mus, _ = _build_band_masks(
        ny=ny,
        nx=nx,
        dy_km=abs(dy_km),
        dx_km=abs(dx_km),
        scale_edges_km=edges,
    )
    fine_idx = int(args.fine_band_idx)
    coarse_idx = int(args.coarse_band_idx)
    if fine_idx < 0 or fine_idx >= len(masks) or coarse_idx < 0 or coarse_idx >= len(masks):
        raise ValueError(f"Band indices out of range: fine={fine_idx}, coarse={coarse_idx}, n_bands={len(masks)}")
    if fine_idx == coarse_idx:
        raise ValueError("fine_band_idx and coarse_band_idx must differ.")

    ring = _compute_band_ring_signals(
        vorticity=vorticity,
        mask_fine=masks[fine_idx],
        mask_coarse=masks[coarse_idx],
        core_mask=core_mask,
        halo_mask=halo_mask,
    )
    fine_core = np.asarray(ring["fine_core"], dtype=float)
    fine_halo = np.asarray(ring["fine_halo"], dtype=float)
    coarse_core = np.asarray(ring["coarse_core"], dtype=float)
    coarse_halo = np.asarray(ring["coarse_halo"], dtype=float)

    lambda_halo = _zscore(coarse_halo - fine_halo)
    bath_c2f = _zscore((coarse_halo - coarse_core) * fine_core)
    bath_f2c = _zscore((fine_halo - fine_core) * coarse_core)

    time_series = pd.Series(pd.to_datetime(time))
    time_hours = _parse_time_hours(time_series)
    gksl_cfg = _read_gksl_config(args.gksl_final_csv, prefer_config_id=str(args.gksl_config_id))
    proxy_df = _build_gksl_proxy(
        sig_l=fine_halo,
        sig_2l=coarse_halo,
        time_hours=time_hours,
        cfg=gksl_cfg,
        dt_cap_hours=float(args.gksl_dt_cap_hours),
    )
    phi_map = _phi_candidate_map(proxy_df, fine_halo, coarse_halo)
    phi_feature = str(args.phi_feature)
    if phi_feature not in phi_map:
        phi_feature = "raw"
    phi_halo = np.asarray(phi_map[phi_feature], dtype=float)

    # width=0 or degenerate halo mode: force halo-only channels to zero
    # to represent "no context" cleanly.
    if halo_n_cells == 0:
        iwv_halo = np.zeros(nt, dtype=float)
        ivtmag_halo = np.zeros(nt, dtype=float)
        pme_halo = np.zeros(nt, dtype=float)
        residual_halo = np.zeros(nt, dtype=float)
        lambda_halo = np.zeros(nt, dtype=float)
        bath_c2f = np.zeros(nt, dtype=float)
        bath_f2c = np.zeros(nt, dtype=float)
        phi_halo = np.zeros(nt, dtype=float)

    t_pd = pd.to_datetime(time)
    years = t_pd.year.to_numpy(dtype=int)
    df = pd.DataFrame(
        {
            "time_index": np.arange(nt, dtype=int),
            "time": t_pd.astype(str),
            "year": years,
            "target_core": target_core,
            "target_core_raw": target_core_raw,
            "residual_halo_raw": residual_halo,
            "ctrl_core": ctrl_core,
            "iwv_halo_z": _zscore(iwv_halo),
            "ivtmag_halo_z": _zscore(ivtmag_halo),
            "pme_halo_z": _zscore(pme_halo),
            "residual_halo_z": _zscore(residual_halo),
            "fine_core": fine_core,
            "fine_halo": fine_halo,
            "coarse_core": coarse_core,
            "coarse_halo": coarse_halo,
            "lambda_halo": lambda_halo,
            "bath_c2f": bath_c2f,
            "bath_f2c": bath_f2c,
            "phi_halo": phi_halo,
            "gksl_cptp_violation": np.asarray(proxy_df["gksl_cptp_violation"], dtype=float),
            "gksl_dt_hours": np.asarray(proxy_df["gksl_dt_hours"], dtype=float),
        }
    )

    model_specs = [
        ("ERA_core", ["ctrl_core"], False),
        ("ERA_window", ["ctrl_core", "iwv_halo_z", "ivtmag_halo_z", "pme_halo_z"], False),
        (
            "ERA_window_plus_Phi_H",
            ["ctrl_core", "iwv_halo_z", "ivtmag_halo_z", "pme_halo_z", "phi_halo"],
            False,
        ),
        (
            "ERA_window_plus_Lambda_H",
            ["ctrl_core", "iwv_halo_z", "ivtmag_halo_z", "pme_halo_z", "lambda_halo", "bath_c2f", "bath_f2c"],
            False,
        ),
        (
            "ERA_window_plus_Phi_H_plus_Lambda_H",
            [
                "ctrl_core",
                "iwv_halo_z",
                "ivtmag_halo_z",
                "pme_halo_z",
                "phi_halo",
                "lambda_halo",
                "bath_c2f",
                "bath_f2c",
            ],
            False,
        ),
        (
            "ERA_window_plus_Phi_H_plus_Lambda_H_shuffled",
            [
                "ctrl_core",
                "iwv_halo_z",
                "ivtmag_halo_z",
                "pme_halo_z",
                "phi_halo",
                "lambda_halo",
                "bath_c2f",
                "bath_f2c",
            ],
            True,
        ),
    ]

    train_mask = years <= int(args.train_end_year)
    test_mask = years == int(args.test_year)
    ext_mask = years == int(args.external_year)

    met_test, pred_test, perms_test, perm_real_test = _evaluate_split(
        df=df,
        train_mask=train_mask,
        eval_mask=test_mask,
        model_specs=model_specs,
        ridge_alpha=float(args.ridge_alpha),
        n_perm=int(args.n_perm),
        perm_block=int(args.perm_block),
        seed=int(args.seed) + 10,
    )
    met_ext, pred_ext, perms_ext, perm_real_ext = _evaluate_split(
        df=df,
        train_mask=train_mask,
        eval_mask=ext_mask,
        model_specs=model_specs,
        ridge_alpha=float(args.ridge_alpha),
        n_perm=int(args.n_perm),
        perm_block=int(args.perm_block),
        seed=int(args.seed) + 20,
    )

    df.to_csv(outdir / "halo_timeseries.csv", index=False)
    met_test.to_csv(outdir / "halo_metrics_test.csv", index=False)
    met_ext.to_csv(outdir / "halo_metrics_external.csv", index=False)
    pred_test.to_csv(outdir / "halo_predictions_test.csv", index=False)
    pred_ext.to_csv(outdir / "halo_predictions_external.csv", index=False)
    perm_real_test.to_csv(outdir / "halo_perm_real_test.csv", index=False)
    perm_real_ext.to_csv(outdir / "halo_perm_real_external.csv", index=False)
    for k, v in perms_test.items():
        v.to_csv(outdir / f"{k}_test.csv", index=False)
    for k, v in perms_ext.items():
        v.to_csv(outdir / f"{k}_external.csv", index=False)

    bath_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "n": int(np.sum(train_mask)),
                "corr_target_bath_c2f": float(np.corrcoef(df.loc[train_mask, "target_core"], df.loc[train_mask, "bath_c2f"])[0, 1]),
                "corr_target_bath_f2c": float(np.corrcoef(df.loc[train_mask, "target_core"], df.loc[train_mask, "bath_f2c"])[0, 1]),
            },
            {
                "split": "test",
                "n": int(np.sum(test_mask)),
                "corr_target_bath_c2f": float(np.corrcoef(df.loc[test_mask, "target_core"], df.loc[test_mask, "bath_c2f"])[0, 1]),
                "corr_target_bath_f2c": float(np.corrcoef(df.loc[test_mask, "target_core"], df.loc[test_mask, "bath_f2c"])[0, 1]),
            },
            {
                "split": "external",
                "n": int(np.sum(ext_mask)),
                "corr_target_bath_c2f": float(np.corrcoef(df.loc[ext_mask, "target_core"], df.loc[ext_mask, "bath_c2f"])[0, 1]),
                "corr_target_bath_f2c": float(np.corrcoef(df.loc[ext_mask, "target_core"], df.loc[ext_mask, "bath_f2c"])[0, 1]),
            },
        ]
    )
    bath_summary.to_csv(outdir / "halo_bath_summary.csv", index=False)

    meta = {
        "inputs": {
            "input_nc": str(args.input_nc),
            "gksl_final_csv": str(args.gksl_final_csv),
        },
        "split": {
            "train_end_year": int(args.train_end_year),
            "test_year": int(args.test_year),
            "external_year": int(args.external_year),
            "n_train": int(np.sum(train_mask)),
            "n_test": int(np.sum(test_mask)),
            "n_external": int(np.sum(ext_mask)),
        },
        "geometry": {
            "ny": int(ny),
            "nx": int(nx),
            "core_margin_cells": int(args.core_margin_cells),
            "halo_width_cells": int(args.halo_width_cells),
            "halo_mode": str(args.halo_mode),
            "halo_n_cells": int(halo_n_cells),
            "core_frac": float(core_frac),
            "halo_frac": float(halo_frac),
            **core_bounds,
        },
        "scale": {
            "scale_edges_km": edges,
            "fine_band_idx": int(fine_idx),
            "coarse_band_idx": int(coarse_idx),
            "fine_scale_center_km": float(centers[fine_idx]),
            "coarse_scale_center_km": float(centers[coarse_idx]),
            "fine_mu": float(mus[fine_idx]),
            "coarse_mu": float(mus[coarse_idx]),
        },
        "selection": {
            "phi_feature": str(phi_feature),
            "gksl_config_id": str(gksl_cfg.config_id),
            "gksl_dephase_base": float(gksl_cfg.dephase_base),
            "gksl_dephase_comm_scale": float(gksl_cfg.dephase_comm_scale),
            "gksl_relax_base": float(gksl_cfg.relax_base),
            "gksl_relax_comm_scale": float(gksl_cfg.relax_comm_scale),
            "gksl_measurement_rate": float(gksl_cfg.measurement_rate),
            "gksl_hamiltonian_scale": float(gksl_cfg.hamiltonian_scale),
            "density_source": density_source,
        },
        "runtime": {
            "target_mode": str(args.target_mode),
            "ridge_alpha": float(args.ridge_alpha),
            "n_perm": int(args.n_perm),
            "perm_block": int(args.perm_block),
            "seed": int(args.seed),
            "remote_exclusion_cells": int(args.remote_exclusion_cells),
            "misalign_shift_y": int(args.misalign_shift_y),
            "misalign_shift_x": int(args.misalign_shift_x),
        },
    }
    (outdir / "halo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _pick(dfm: pd.DataFrame, model: str, col: str) -> float:
        sel = dfm[dfm["model"] == model]
        if len(sel) == 0:
            return float("nan")
        return float(sel.iloc[0][col])

    lines = [
        "# Experiment M Halo Boundary Strict",
        "",
        "## Protocol",
        f"- train years <= {int(args.train_end_year)}",
        f"- holdout test year = {int(args.test_year)}",
        f"- external holdout year = {int(args.external_year)}",
        f"- scoring target: core-only residual (`target_mode={args.target_mode}`)",
        f"- core margin = {int(args.core_margin_cells)} cells",
        f"- halo mode = `{str(args.halo_mode)}`",
        f"- halo ring width = {int(args.halo_width_cells)} cells (n={halo_n_cells})",
        f"- selected bands (fine/coarse) = {fine_idx}/{coarse_idx} (centers {centers[fine_idx]:.1f}/{centers[coarse_idx]:.1f} km)",
        "",
        "## Model Ladder",
        "- ERA_core",
        "- ERA_window",
        "- ERA_window_plus_Phi_H",
        "- ERA_window_plus_Lambda_H (includes bath_c2f, bath_f2c)",
        "- ERA_window_plus_Phi_H_plus_Lambda_H",
        "- ERA_window_plus_Phi_H_plus_Lambda_H_shuffled",
        "",
        f"## Test {int(args.test_year)}",
        f"- ERA_window gain_vs_core: `{_pick(met_test, 'ERA_window', 'gain_vs_core'):.6f}` p_abs=`{_pick(met_test, 'ERA_window', 'perm_p_abs'):.6f}`",
        f"- +Phi_H gain_vs_core: `{_pick(met_test, 'ERA_window_plus_Phi_H', 'gain_vs_core'):.6f}` p_abs=`{_pick(met_test, 'ERA_window_plus_Phi_H', 'perm_p_abs'):.6f}`",
        f"- +Lambda_H gain_vs_core: `{_pick(met_test, 'ERA_window_plus_Lambda_H', 'gain_vs_core'):.6f}` p_abs=`{_pick(met_test, 'ERA_window_plus_Lambda_H', 'perm_p_abs'):.6f}`",
        f"- +Phi_H+Lambda_H gain_vs_core: `{_pick(met_test, 'ERA_window_plus_Phi_H_plus_Lambda_H', 'gain_vs_core'):.6f}` p_abs=`{_pick(met_test, 'ERA_window_plus_Phi_H_plus_Lambda_H', 'perm_p_abs'):.6f}`",
        "",
        f"## External {int(args.external_year)}",
        f"- ERA_window gain_vs_core: `{_pick(met_ext, 'ERA_window', 'gain_vs_core'):.6f}` p_abs=`{_pick(met_ext, 'ERA_window', 'perm_p_abs'):.6f}`",
        f"- +Phi_H gain_vs_core: `{_pick(met_ext, 'ERA_window_plus_Phi_H', 'gain_vs_core'):.6f}` p_abs=`{_pick(met_ext, 'ERA_window_plus_Phi_H', 'perm_p_abs'):.6f}`",
        f"- +Lambda_H gain_vs_core: `{_pick(met_ext, 'ERA_window_plus_Lambda_H', 'gain_vs_core'):.6f}` p_abs=`{_pick(met_ext, 'ERA_window_plus_Lambda_H', 'perm_p_abs'):.6f}`",
        f"- +Phi_H+Lambda_H gain_vs_core: `{_pick(met_ext, 'ERA_window_plus_Phi_H_plus_Lambda_H', 'gain_vs_core'):.6f}` p_abs=`{_pick(met_ext, 'ERA_window_plus_Phi_H_plus_Lambda_H', 'perm_p_abs'):.6f}`",
        "",
        "## Artifacts",
        "- `halo_metrics_test.csv`",
        "- `halo_metrics_external.csv`",
        "- `halo_bath_summary.csv`",
        "- `halo_timeseries.csv`",
        "- `halo_meta.json`",
    ]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[M-HALO-STRICT] done")
    print("\n[test metrics]")
    print(met_test.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\n[external metrics]")
    print(met_ext.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print(f"\nSaved: {outdir}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
