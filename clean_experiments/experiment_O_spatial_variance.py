#!/usr/bin/env python3
"""Experiment O (spatial): local Clausius regression with 2D Lambda diagnostics.

This script switches from domain-mean diagnostics to spatially resolved local regressions:
  baseline: dS_local ~ dQ_local
  full:     dS_local ~ dQ_local + Lambda_local

Lambda_local(t,y,x) is reconstructed from Experiment M band-level amplitudes and
local scale weights from band-limited vorticity fields.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_land_mask import globe

try:
    from clean_experiments.experiment_M_cosmo_flow import _build_band_masks, _compute_vorticity, _xy_coordinates_m
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _build_band_masks, _compute_vorticity, _xy_coordinates_m  # type: ignore


EPS = 1e-12
CP_D = 1004.64
L_V = 2.5e6
T0_K = 273.15


@dataclass(frozen=True)
class Taxon:
    name: str
    p_bot_hpa: float
    p_top_hpa: float


def _build_land_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    lon_wrapped = ((lon2d + 180.0) % 360.0) - 180.0
    mask = globe.is_land(lat2d, lon_wrapped)
    return np.asarray(mask, dtype=bool)


def _find_coord_name(dataset: Dataset, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"Could not find coordinate for {label}. Tried: {candidates}.")


def _find_var_name(dataset: Dataset, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"Could not find variable for {label}. Tried: {candidates}.")


def _to_datetime64(time_var) -> np.ndarray:
    raw = np.asarray(time_var[:])
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.astype("datetime64[ns]")
    if np.issubdtype(raw.dtype, np.number):
        units = getattr(time_var, "units", None)
        calendar = getattr(time_var, "calendar", "standard")
        if units is None:
            raise ValueError("Numeric time coordinate has no units.")
        dt = num2date(raw, units=units, calendar=calendar)
        arr = np.asarray(dt)
        if np.issubdtype(arr.dtype, np.datetime64):
            ts = pd.to_datetime(arr)
        else:
            ts = pd.to_datetime([str(x) for x in arr])
        return ts.to_numpy(dtype="datetime64[ns]")
    ts = pd.to_datetime(raw)
    return ts.to_numpy(dtype="datetime64[ns]")


def _parse_taxon(spec: str) -> Taxon:
    try:
        name, rng = spec.split(":")
        p_bot_s, p_top_s = rng.split("-")
        p_bot = float(p_bot_s)
        p_top = float(p_top_s)
    except Exception as exc:
        raise ValueError(f"Invalid taxon spec '{spec}'. Use NAME:PBOT-PTOP, e.g. FT:850-300") from exc
    if p_bot < p_top:
        raise ValueError(f"Taxon {name}: expected PBOT>=PTOP, got {p_bot}<{p_top}")
    return Taxon(name=name, p_bot_hpa=p_bot, p_top_hpa=p_top)


def _level_weights_pa(p_pa: np.ndarray) -> np.ndarray:
    p = np.asarray(p_pa, dtype=float)
    n = len(p)
    if n < 2:
        return np.ones_like(p)
    w = np.zeros(n, dtype=float)
    w[0] = 0.5 * abs(p[0] - p[1])
    w[-1] = 0.5 * abs(p[-2] - p[-1])
    if n >= 3:
        w[1:-1] = 0.5 * abs(p[:-2] - p[2:])
    w = np.where(w <= 0.0, 1.0, w)
    return w


def _load_m_band_amplitudes(
    *,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    ts = pd.read_csv(m_timeseries_csv)

    pat = re.compile(r"^lambda_mu_(\d+)$")
    lam_col_by_band: dict[int, str] = {}
    for c in ts.columns:
        m = pat.match(c)
        if m:
            lam_col_by_band[int(m.group(1))] = c
    band_ids = sorted(lam_col_by_band.keys())
    if len(band_ids) == 0:
        raise ValueError("No lambda_mu_* columns found in Experiment M timeseries.")

    lam_cols = [lam_col_by_band[b] for b in band_ids]
    coh_cols = [lam_col_by_band[b].replace("lambda_mu_", "coh_mu_") for b in band_ids]
    ent_cols = [lam_col_by_band[b].replace("lambda_mu_", "entropy_mu_") for b in band_ids]
    for c in lam_cols + coh_cols + ent_cols:
        if c not in ts.columns:
            raise ValueError(f"Missing required column '{c}' in {m_timeseries_csv}")

    lam_mu = ts[lam_cols].to_numpy(dtype=float)
    coh_mu = ts[coh_cols].to_numpy(dtype=float)
    ent_mu = ts[ent_cols].to_numpy(dtype=float)

    summary = pd.read_csv(m_summary_csv)
    if len(summary) != 1:
        raise ValueError(f"Expected one-row summary in {m_summary_csv}, got {len(summary)}")
    srow = summary.iloc[0]
    coherence_floor = float(srow["coherence_floor"])
    coherence_power = float(srow["coherence_power"])
    coherence_blend = float(srow["coherence_blend"])

    mode_idx = pd.read_csv(m_mode_index_csv)
    if "band_id" not in mode_idx.columns:
        raise ValueError(f"No 'band_id' in {m_mode_index_csv}")
    m_per_band = np.ones(len(band_ids), dtype=float)
    for i, bid in enumerate(band_ids):
        m_per_band[i] = float(max(int(np.sum(mode_idx["band_id"].to_numpy(dtype=int) == bid)), 1))

    max_entropy = np.log(np.maximum(m_per_band, 1.0))
    max_entropy = np.where(max_entropy < 1e-12, 1.0, max_entropy)

    coh_eff = np.power(np.maximum(coh_mu + coherence_floor, 0.0), coherence_power)
    diag_mix = np.clip(ent_mu / max_entropy[None, :], 0.0, 1.0)
    signal_mu = coherence_blend * coh_eff + (1.0 - coherence_blend) * diag_mix

    # Band amplitudes before spatial weighting.
    band_amp = signal_mu * lam_mu
    band_amp = np.nan_to_num(band_amp, nan=0.0, posinf=0.0, neginf=0.0)

    diag = pd.DataFrame(
        {
            "band_id": band_ids,
            "modes_count": m_per_band,
            "mean_abs_lambda_mu": np.mean(np.abs(lam_mu), axis=0),
            "mean_signal_mu": np.mean(signal_mu, axis=0),
            "mean_abs_band_amp": np.mean(np.abs(band_amp), axis=0),
        }
    )
    return band_amp, diag


def _compute_lambda_local_batch(
    *,
    zeta_batch: np.ndarray,
    masks: list[np.ndarray],
    band_amp_batch: np.ndarray,
) -> np.ndarray:
    # zeta_batch: [tb, y, x], band_amp_batch: [tb, nb]
    tb, ny, nx = zeta_batch.shape
    nb = len(masks)
    out = np.zeros((tb, ny, nx), dtype=np.float32)

    for i in range(tb):
        zeta = np.nan_to_num(zeta_batch[i], nan=0.0, posinf=0.0, neginf=0.0)
        zhat = np.fft.rfft2(zeta - np.mean(zeta))

        band_stack = np.zeros((nb, ny, nx), dtype=np.float64)
        for b, mask in enumerate(masks):
            z_band = np.fft.irfft2(zhat * mask, s=(ny, nx)).real
            band_stack[b] = np.abs(z_band)

        denom = np.sum(band_stack, axis=0)
        denom = np.where(denom < EPS, 1.0, denom)
        weights_local = band_stack / denom[None, :, :]
        lam = np.tensordot(band_amp_batch[i], weights_local, axes=(0, 0))
        out[i] = np.asarray(lam, dtype=np.float32)

    return out


def _solve_baseline(
    *,
    n: int,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    den = n * sxx - sx * sx
    b1 = np.zeros_like(sx)
    ok = np.abs(den) > 1e-10
    b1[ok] = (n * sxy[ok] - sx[ok] * sy[ok]) / den[ok]
    b0 = (sy - b1 * sx) / float(max(n, 1))
    return b0, b1


def _solve_full(
    *,
    n: int,
    sx1: np.ndarray,
    sx2: np.ndarray,
    sy: np.ndarray,
    sx11: np.ndarray,
    sx22: np.ndarray,
    sx12: np.ndarray,
    sx1y: np.ndarray,
    sx2y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npix = len(sx1)
    b0 = np.zeros(npix, dtype=float)
    b1 = np.zeros(npix, dtype=float)
    b2 = np.zeros(npix, dtype=float)

    for j in range(npix):
        lhs = np.array(
            [
                [float(n), float(sx1[j]), float(sx2[j])],
                [float(sx1[j]), float(sx11[j]), float(sx12[j])],
                [float(sx2[j]), float(sx12[j]), float(sx22[j])],
            ],
            dtype=float,
        )
        rhs = np.array([float(sy[j]), float(sx1y[j]), float(sx2y[j])], dtype=float)
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        b0[j], b1[j], b2[j] = float(beta[0]), float(beta[1]), float(beta[2])

    return b0, b1, b2


def _plot_map(
    *,
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str,
    symmetric: bool,
) -> None:
    arr = np.asarray(field, dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")

    vmin = np.nanquantile(arr, 0.02)
    vmax = np.nanquantile(arr, 0.98)
    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)

    im = ax.pcolormesh(
        lon2d,
        lat2d,
        arr,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lon)), float(np.max(lon)))
    ax.set_ylim(float(np.min(lat)), float(np.max(lat)))
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_experiment(
    *,
    input_nc: Path,
    m_timeseries_csv: Path,
    m_summary_csv: Path,
    m_mode_index_csv: Path,
    outdir: Path,
    taxon_spec: str,
    scale_edges_km: list[float],
    train_end_year: int,
    test_year: int,
    west_split_lon: float,
    batch_size: int,
    corr_threshold: float,
    hotspot_gain_threshold: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    print("[experiment_O_spatial] Step 1/6: load Experiment M band amplitudes...", flush=True)
    band_amp, band_diag = _load_m_band_amplitudes(
        m_timeseries_csv=m_timeseries_csv,
        m_summary_csv=m_summary_csv,
        m_mode_index_csv=m_mode_index_csv,
    )
    band_diag.to_csv(outdir / "lambda_band_diagnostics.csv", index=False)

    tx = _parse_taxon(taxon_spec)

    print("[experiment_O_spatial] Step 2/6: open input and prepare masks...", flush=True)
    with Dataset(input_nc, mode="r") as ds:
        time_name = _find_coord_name(ds, ("valid_time", "time", "datetime", "date"), "time")
        lat_name = _find_coord_name(ds, ("latitude", "lat", "y", "rlat"), "latitude")
        lon_name = _find_coord_name(ds, ("longitude", "lon", "x", "rlon"), "longitude")
        level_name = _find_coord_name(ds, ("pressure_level", "level", "plev", "isobaricInhPa"), "pressure level")

        temp_name = _find_var_name(ds, ("temp_pl", "t"), "temp_pl")
        q_name = _find_var_name(ds, ("q_pl", "q"), "q_pl")
        w_name = _find_var_name(ds, ("w_pl", "w", "omega"), "w_pl")
        u_name = _find_var_name(ds, ("u",), "u")
        v_name = _find_var_name(ds, ("v",), "v")

        conv_name = None
        for cand in ("convective_precip", "cp", "cp_rate", "precip"):
            if cand in ds.variables:
                conv_name = cand
                break
        if conv_name is None:
            raise ValueError("No convective/precip variable found for climatology map.")

        time_ns = _to_datetime64(ds.variables[time_name])
        time_dt = pd.to_datetime(time_ns)
        years = time_dt.year.to_numpy(dtype=int)

        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)
        nt = len(time_ns)
        ny = len(lat)
        nx = len(lon)
        npix = ny * nx
        land_mask = _build_land_mask(lat, lon)
        land_vec = land_mask.reshape(-1).astype(np.float64)
        land_count = float(np.sum(land_vec))
        ocean_count = float(npix - land_count)

        if band_amp.shape[0] != nt:
            raise ValueError(
                f"M timeseries length mismatch: band_amp nt={band_amp.shape[0]} vs input nt={nt}."
            )

        levels_raw = np.asarray(ds.variables[level_name][:], dtype=float)
        levels_hpa = levels_raw / 100.0 if np.nanmax(np.abs(levels_raw)) > 3000.0 else levels_raw
        levels_pa = levels_hpa * 100.0
        lvl_w = _level_weights_pa(levels_pa)
        lvl_mask = (levels_hpa <= tx.p_bot_hpa) & (levels_hpa >= tx.p_top_hpa)
        if int(np.sum(lvl_mask)) == 0:
            raise ValueError(
                f"Taxon {tx.name} has no levels in [{tx.p_top_hpa},{tx.p_bot_hpa}] hPa. Levels={levels_hpa.tolist()}"
            )
        wv = lvl_w[lvl_mask]
        wv = wv / (np.sum(wv) + EPS)

        x_m, y_m = _xy_coordinates_m(lat, lon)
        dx_km = float(np.median(np.diff(x_m))) / 1000.0
        dy_km = float(np.median(np.diff(y_m))) / 1000.0
        masks, _, _, centers, _, _ = _build_band_masks(
            ny=ny,
            nx=nx,
            dy_km=abs(dy_km),
            dx_km=abs(dx_km),
            scale_edges_km=scale_edges_km,
        )
        if len(masks) != band_amp.shape[1]:
            raise ValueError(
                f"Band mismatch: masks={len(masks)} vs M-band components={band_amp.shape[1]}. "
                "Use same scale edges as Experiment M run."
            )

        train_mask = years <= int(train_end_year)
        test_mask = years == int(test_year)
        n_train = int(np.sum(train_mask))
        n_test = int(np.sum(test_mask))
        if n_train < 500 or n_test < 200:
            raise ValueError(f"Too few train/test samples: n_train={n_train}, n_test={n_test}")

        # Training sufficient statistics.
        sx1_tr = np.zeros(npix, dtype=np.float64)
        sx2_tr = np.zeros(npix, dtype=np.float64)
        sy_tr = np.zeros(npix, dtype=np.float64)
        sx11_tr = np.zeros(npix, dtype=np.float64)
        sx22_tr = np.zeros(npix, dtype=np.float64)
        sx12_tr = np.zeros(npix, dtype=np.float64)
        sx1y_tr = np.zeros(npix, dtype=np.float64)
        sx2y_tr = np.zeros(npix, dtype=np.float64)

        # Panel-level land-aware matrix stats:
        # baseline features: [1, dQ]
        # full features: [1, dQ, Lambda, land, Lambda*land]
        xtx2_tr = np.zeros((2, 2), dtype=np.float64)
        xty2_tr = np.zeros(2, dtype=np.float64)
        xtx5_tr = np.zeros((5, 5), dtype=np.float64)
        xty5_tr = np.zeros(5, dtype=np.float64)
        xtx2_te = np.zeros((2, 2), dtype=np.float64)
        xty2_te = np.zeros(2, dtype=np.float64)
        xtx5_te = np.zeros((5, 5), dtype=np.float64)
        xty5_te = np.zeros(5, dtype=np.float64)
        n_panel_test = 0.0
        y_sum_panel_test = 0.0
        y2_sum_panel_test = 0.0

        # Test arrays for exact SSE/R2 without second full pass.
        x1_test = np.empty((n_test, npix), dtype=np.float32)
        x2_test = np.empty((n_test, npix), dtype=np.float32)
        y_test = np.empty((n_test, npix), dtype=np.float32)
        test_pos = 0

        # Additional diagnostics.
        conv_sum = np.zeros(npix, dtype=np.float64)
        lambda_domain_series = np.zeros(nt, dtype=np.float64)

        temp_var = ds.variables[temp_name]
        q_var = ds.variables[q_name]
        w_var = ds.variables[w_name]
        u_var = ds.variables[u_name]
        v_var = ds.variables[v_name]
        conv_var = ds.variables[conv_name]

        print("[experiment_O_spatial] Step 3/6: first pass (local series + train stats + test store)...", flush=True)
        for start in range(0, nt, batch_size):
            stop = min(start + batch_size, nt)
            sl = slice(start, stop)
            tb = stop - start

            temp_blk = np.asarray(temp_var[sl, ...], dtype=np.float64)
            q_blk = np.asarray(q_var[sl, ...], dtype=np.float64)
            w_blk = np.asarray(w_var[sl, ...], dtype=np.float64)
            u_blk = np.asarray(u_var[sl, ...], dtype=np.float64)
            v_blk = np.asarray(v_var[sl, ...], dtype=np.float64)
            conv_blk = np.asarray(conv_var[sl, ...], dtype=np.float64)

            temp_blk = np.nan_to_num(temp_blk, nan=0.0, posinf=0.0, neginf=0.0)
            q_blk = np.nan_to_num(q_blk, nan=0.0, posinf=0.0, neginf=0.0)
            w_blk = np.nan_to_num(w_blk, nan=0.0, posinf=0.0, neginf=0.0)
            u_blk = np.nan_to_num(u_blk, nan=0.0, posinf=0.0, neginf=0.0)
            v_blk = np.nan_to_num(v_blk, nan=0.0, posinf=0.0, neginf=0.0)
            conv_blk = np.nan_to_num(conv_blk, nan=0.0, posinf=0.0, neginf=0.0)

            temp_safe = np.clip(temp_blk, 150.0, 350.0)
            dS_lvl = w_blk * (L_V * q_blk / temp_safe)
            dQ_lvl = -w_blk * (CP_D * temp_safe + L_V * q_blk)

            dS_tyx = np.einsum("tlyx,l->tyx", dS_lvl[:, lvl_mask, :, :], wv, optimize=True)
            dQ_tyx = np.einsum("tlyx,l->tyx", dQ_lvl[:, lvl_mask, :, :], wv, optimize=True)

            zeta_blk = _compute_vorticity(u=u_blk, v=v_blk, x_m=x_m, y_m=y_m)
            lam_tyx = _compute_lambda_local_batch(
                zeta_batch=zeta_blk,
                masks=masks,
                band_amp_batch=band_amp[start:stop],
            )

            lambda_domain_series[start:stop] = np.mean(lam_tyx, axis=(1, 2))
            conv_sum += np.sum(conv_blk, axis=0).reshape(-1)

            y_flat = dS_tyx.reshape(tb, npix)
            x1_flat = dQ_tyx.reshape(tb, npix)
            x2_flat = lam_tyx.reshape(tb, npix)

            rel_train = np.where(train_mask[start:stop])[0]
            if len(rel_train) > 0:
                ytr = y_flat[rel_train].astype(np.float64)
                x1tr = x1_flat[rel_train].astype(np.float64)
                x2tr = x2_flat[rel_train].astype(np.float64)
                sy_tr += np.sum(ytr, axis=0)
                sx1_tr += np.sum(x1tr, axis=0)
                sx2_tr += np.sum(x2tr, axis=0)
                sx11_tr += np.sum(x1tr * x1tr, axis=0)
                sx22_tr += np.sum(x2tr * x2tr, axis=0)
                sx12_tr += np.sum(x1tr * x2tr, axis=0)
                sx1y_tr += np.sum(x1tr * ytr, axis=0)
                sx2y_tr += np.sum(x2tr * ytr, axis=0)

                mt = float(ytr.shape[0])
                n_loc = mt * float(npix)
                x1_sum = float(np.sum(x1tr))
                x2_sum = float(np.sum(x2tr))
                y_sum = float(np.sum(ytr))
                x11_sum = float(np.sum(x1tr * x1tr))
                x22_sum = float(np.sum(x2tr * x2tr))
                x12_sum = float(np.sum(x1tr * x2tr))
                x1y_sum = float(np.sum(x1tr * ytr))
                x2y_sum = float(np.sum(x2tr * ytr))
                x1_land_sum = float(np.sum(x1tr * land_vec[None, :]))
                x2_land_sum = float(np.sum(x2tr * land_vec[None, :]))
                y_land_sum = float(np.sum(ytr * land_vec[None, :]))
                x1x2_land_sum = float(np.sum(x1tr * x2tr * land_vec[None, :]))
                x22_land_sum = float(np.sum(x2tr * x2tr * land_vec[None, :]))
                x2y_land_sum = float(np.sum(x2tr * ytr * land_vec[None, :]))
                land_rep = mt * land_count

                xtx2_tr[0, 0] += n_loc
                xtx2_tr[0, 1] += x1_sum
                xtx2_tr[1, 1] += x11_sum
                xty2_tr[0] += y_sum
                xty2_tr[1] += x1y_sum

                xtx5_tr[0, 0] += n_loc
                xtx5_tr[0, 1] += x1_sum
                xtx5_tr[0, 2] += x2_sum
                xtx5_tr[0, 3] += land_rep
                xtx5_tr[0, 4] += x2_land_sum
                xtx5_tr[1, 1] += x11_sum
                xtx5_tr[1, 2] += x12_sum
                xtx5_tr[1, 3] += x1_land_sum
                xtx5_tr[1, 4] += x1x2_land_sum
                xtx5_tr[2, 2] += x22_sum
                xtx5_tr[2, 3] += x2_land_sum
                xtx5_tr[2, 4] += x22_land_sum
                xtx5_tr[3, 3] += land_rep
                xtx5_tr[3, 4] += x2_land_sum
                xtx5_tr[4, 4] += x22_land_sum

                xty5_tr[0] += y_sum
                xty5_tr[1] += x1y_sum
                xty5_tr[2] += x2y_sum
                xty5_tr[3] += y_land_sum
                xty5_tr[4] += x2y_land_sum

            rel_test = np.where(test_mask[start:stop])[0]
            if len(rel_test) > 0:
                n_add = len(rel_test)
                x1te = x1_flat[rel_test].astype(np.float32)
                x2te = x2_flat[rel_test].astype(np.float32)
                yte = y_flat[rel_test].astype(np.float32)
                x1_test[test_pos : test_pos + n_add] = x1te
                x2_test[test_pos : test_pos + n_add] = x2te
                y_test[test_pos : test_pos + n_add] = yte
                test_pos += n_add

                yte64 = yte.astype(np.float64)
                x1te64 = x1te.astype(np.float64)
                x2te64 = x2te.astype(np.float64)
                mt = float(yte64.shape[0])
                n_loc = mt * float(npix)
                x1_sum = float(np.sum(x1te64))
                x2_sum = float(np.sum(x2te64))
                y_sum = float(np.sum(yte64))
                x11_sum = float(np.sum(x1te64 * x1te64))
                x22_sum = float(np.sum(x2te64 * x2te64))
                x12_sum = float(np.sum(x1te64 * x2te64))
                x1y_sum = float(np.sum(x1te64 * yte64))
                x2y_sum = float(np.sum(x2te64 * yte64))
                x1_land_sum = float(np.sum(x1te64 * land_vec[None, :]))
                x2_land_sum = float(np.sum(x2te64 * land_vec[None, :]))
                y_land_sum = float(np.sum(yte64 * land_vec[None, :]))
                x1x2_land_sum = float(np.sum(x1te64 * x2te64 * land_vec[None, :]))
                x22_land_sum = float(np.sum(x2te64 * x2te64 * land_vec[None, :]))
                x2y_land_sum = float(np.sum(x2te64 * yte64 * land_vec[None, :]))
                land_rep = mt * land_count

                xtx2_te[0, 0] += n_loc
                xtx2_te[0, 1] += x1_sum
                xtx2_te[1, 1] += x11_sum
                xty2_te[0] += y_sum
                xty2_te[1] += x1y_sum

                xtx5_te[0, 0] += n_loc
                xtx5_te[0, 1] += x1_sum
                xtx5_te[0, 2] += x2_sum
                xtx5_te[0, 3] += land_rep
                xtx5_te[0, 4] += x2_land_sum
                xtx5_te[1, 1] += x11_sum
                xtx5_te[1, 2] += x12_sum
                xtx5_te[1, 3] += x1_land_sum
                xtx5_te[1, 4] += x1x2_land_sum
                xtx5_te[2, 2] += x22_sum
                xtx5_te[2, 3] += x2_land_sum
                xtx5_te[2, 4] += x22_land_sum
                xtx5_te[3, 3] += land_rep
                xtx5_te[3, 4] += x2_land_sum
                xtx5_te[4, 4] += x22_land_sum

                xty5_te[0] += y_sum
                xty5_te[1] += x1y_sum
                xty5_te[2] += x2y_sum
                xty5_te[3] += y_land_sum
                xty5_te[4] += x2y_land_sum

                n_panel_test += n_loc
                y_sum_panel_test += y_sum
                y2_sum_panel_test += float(np.sum(yte64 * yte64))

            print(f"[experiment_O_spatial] progress first pass: {stop}/{nt}", flush=True)

    if test_pos != n_test:
        raise RuntimeError(f"Test store mismatch: expected {n_test}, got {test_pos}")

    # Fill symmetric parts for panel normal equations.
    xtx2_tr = xtx2_tr + np.triu(xtx2_tr, 1).T
    xtx5_tr = xtx5_tr + np.triu(xtx5_tr, 1).T
    xtx2_te = xtx2_te + np.triu(xtx2_te, 1).T
    xtx5_te = xtx5_te + np.triu(xtx5_te, 1).T

    beta_panel_base = np.linalg.lstsq(xtx2_tr, xty2_tr, rcond=None)[0]
    beta_panel_full = np.linalg.lstsq(xtx5_tr, xty5_tr, rcond=None)[0]

    sse_panel_base = float(
        y2_sum_panel_test
        - 2.0 * float(beta_panel_base @ xty2_te)
        + float(beta_panel_base @ xtx2_te @ beta_panel_base)
    )
    sse_panel_full = float(
        y2_sum_panel_test
        - 2.0 * float(beta_panel_full @ xty5_te)
        + float(beta_panel_full @ xtx5_te @ beta_panel_full)
    )
    sst_panel_test = float(y2_sum_panel_test - (y_sum_panel_test * y_sum_panel_test) / max(n_panel_test, 1.0))
    panel_r2_base = float(1.0 - sse_panel_base / (sst_panel_test + EPS))
    panel_r2_full = float(1.0 - sse_panel_full / (sst_panel_test + EPS))
    panel_gain_r2 = float(panel_r2_full - panel_r2_base)

    print("[experiment_O_spatial] Step 4/6: solve local regressions...", flush=True)
    b0_base, b1_base = _solve_baseline(
        n=n_train,
        sx=sx1_tr,
        sy=sy_tr,
        sxx=sx11_tr,
        sxy=sx1y_tr,
    )
    b0_full, b1_full, b2_full = _solve_full(
        n=n_train,
        sx1=sx1_tr,
        sx2=sx2_tr,
        sy=sy_tr,
        sx11=sx11_tr,
        sx22=sx22_tr,
        sx12=sx12_tr,
        sx1y=sx1y_tr,
        sx2y=sx2y_tr,
    )

    print("[experiment_O_spatial] Step 5/6: compute local R2 maps on test year...", flush=True)
    y_test64 = y_test.astype(np.float64)
    sy_te = np.sum(y_test64, axis=0)
    sy2_te = np.sum(y_test64 * y_test64, axis=0)
    y_mean_te = sy_te / float(max(n_test, 1))
    sst_te = sy2_te - float(n_test) * y_mean_te * y_mean_te

    sse_base = np.zeros(npix, dtype=np.float64)
    sse_full = np.zeros(npix, dtype=np.float64)
    chunk = 512
    for j0 in range(0, npix, chunk):
        j1 = min(j0 + chunk, npix)
        x1 = x1_test[:, j0:j1].astype(np.float64)
        x2 = x2_test[:, j0:j1].astype(np.float64)
        yv = y_test64[:, j0:j1]

        pb = b0_base[None, j0:j1] + b1_base[None, j0:j1] * x1
        pf = b0_full[None, j0:j1] + b1_full[None, j0:j1] * x1 + b2_full[None, j0:j1] * x2

        sse_base[j0:j1] = np.sum((yv - pb) ** 2, axis=0)
        sse_full[j0:j1] = np.sum((yv - pf) ** 2, axis=0)

    r2_base = np.full(npix, np.nan, dtype=np.float64)
    r2_full = np.full(npix, np.nan, dtype=np.float64)
    valid_sst = sst_te > 1e-12
    r2_base[valid_sst] = 1.0 - sse_base[valid_sst] / (sst_te[valid_sst] + EPS)
    r2_full[valid_sst] = 1.0 - sse_full[valid_sst] / (sst_te[valid_sst] + EPS)
    gain = r2_full - r2_base

    gain_map = gain.reshape(ny, nx)
    r2_base_map = r2_base.reshape(ny, nx)
    r2_full_map = r2_full.reshape(ny, nx)
    beta_lambda_map = b2_full.reshape(ny, nx)
    conv_clim_map = (conv_sum / float(nt)).reshape(ny, nx)

    # Spatial criteria.
    valid = np.isfinite(gain_map) & np.isfinite(conv_clim_map)
    if int(np.sum(valid)) < 20:
        corr_gain_conv = np.nan
    else:
        corr_gain_conv = float(np.corrcoef(gain_map[valid], conv_clim_map[valid])[0, 1])

    max_gain = float(np.nanmax(gain_map))
    max_idx = np.unravel_index(int(np.nanargmax(gain_map)), gain_map.shape)
    max_lat = float(lat[max_idx[0]])
    max_lon = float(lon[max_idx[1]])

    split_eff = float(west_split_lon)
    if not (np.any(lon <= split_eff) and np.any(lon > split_eff)):
        split_eff = float(np.median(lon))

    west_mask = np.repeat((lon[None, :] <= split_eff), ny, axis=0)
    east_mask = ~west_mask
    west_mean_gain = float(np.nanmean(gain_map[west_mask]))
    east_mean_gain = float(np.nanmean(gain_map[east_mask]))
    land_mean_gain = float(np.nanmean(gain_map[land_mask]))
    ocean_mean_gain = float(np.nanmean(gain_map[~land_mask]))

    valid_land = np.isfinite(gain_map) & np.isfinite(conv_clim_map) & land_mask
    valid_ocean = np.isfinite(gain_map) & np.isfinite(conv_clim_map) & (~land_mask)
    corr_gain_conv_land = (
        float(np.corrcoef(gain_map[valid_land], conv_clim_map[valid_land])[0, 1]) if int(np.sum(valid_land)) > 20 else np.nan
    )
    corr_gain_conv_ocean = (
        float(np.corrcoef(gain_map[valid_ocean], conv_clim_map[valid_ocean])[0, 1]) if int(np.sum(valid_ocean)) > 20 else np.nan
    )

    success_corr = bool(np.isfinite(corr_gain_conv) and corr_gain_conv > corr_threshold)
    success_hotspot = bool(np.isfinite(max_gain) and max_gain > hotspot_gain_threshold)

    # Diagnostic: how close domain mean reconstructed lambda is to global lambda_struct.
    m_ts = pd.read_csv(m_timeseries_csv, usecols=["lambda_struct"])
    lam_global = m_ts["lambda_struct"].to_numpy(dtype=float)
    corr_recon_global = (
        float(np.corrcoef(lambda_domain_series, lam_global)[0, 1])
        if np.std(lambda_domain_series) > 1e-14 and np.std(lam_global) > 1e-14
        else np.nan
    )

    print("[experiment_O_spatial] Step 6/6: write artifacts...", flush=True)
    np.savez_compressed(
        outdir / "spatial_maps.npz",
        lat=lat,
        lon=lon,
        gain_map=gain_map,
        r2_base_map=r2_base_map,
        r2_full_map=r2_full_map,
        beta_lambda_map=beta_lambda_map,
        convective_climatology_map=conv_clim_map,
        land_mask=land_mask.astype(np.int8),
    )

    _plot_map(
        field=gain_map,
        lat=lat,
        lon=lon,
        title="Experiment O Spatial: Gain map (R2_full - R2_base)",
        cbar_label="Gain in R2",
        out_path=outdir / "plot_gain_map.png",
        cmap="RdBu_r",
        symmetric=True,
    )
    _plot_map(
        field=conv_clim_map,
        lat=lat,
        lon=lon,
        title=f"Convective climatology map ({conv_name})",
        cbar_label=conv_name,
        out_path=outdir / "plot_convective_climatology.png",
        cmap="YlGnBu",
        symmetric=False,
    )
    _plot_map(
        field=beta_lambda_map,
        lat=lat,
        lon=lon,
        title="Lambda coefficient map (full local regression)",
        cbar_label="beta_lambda",
        out_path=outdir / "plot_beta_lambda_map.png",
        cmap="RdBu_r",
        symmetric=True,
    )

    rows = []
    for iy in range(ny):
        for ix in range(nx):
            rows.append(
                {
                    "lat": float(lat[iy]),
                    "lon": float(lon[ix]),
                    "r2_base": float(r2_base_map[iy, ix]),
                    "r2_full": float(r2_full_map[iy, ix]),
                    "gain_r2": float(gain_map[iy, ix]),
                    "beta_dq_base": float(b1_base[iy * nx + ix]),
                    "beta_dq_full": float(b1_full[iy * nx + ix]),
                    "beta_lambda_full": float(beta_lambda_map[iy, ix]),
                    "convective_clim": float(conv_clim_map[iy, ix]),
                }
            )
    pd.DataFrame(rows).to_csv(outdir / "spatial_point_metrics.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "input_nc": str(input_nc),
                "m_timeseries_csv": str(m_timeseries_csv),
                "m_summary_csv": str(m_summary_csv),
                "m_mode_index_csv": str(m_mode_index_csv),
                "taxon": tx.name,
                "taxon_p_top_hpa": float(tx.p_top_hpa),
                "taxon_p_bot_hpa": float(tx.p_bot_hpa),
                "n_time": int(nt),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "ny": int(ny),
                "nx": int(nx),
                "n_bands": int(len(masks)),
                "convective_source": str(conv_name),
                "west_split_lon": float(split_eff),
                "corr_gain_vs_convective_clim": float(corr_gain_conv),
                "corr_threshold": float(corr_threshold),
                "pass_corr_threshold": bool(success_corr),
                "max_gain": float(max_gain),
                "hotspot_gain_threshold": float(hotspot_gain_threshold),
                "pass_hotspot_threshold": bool(success_hotspot),
                "max_gain_lat": float(max_lat),
                "max_gain_lon": float(max_lon),
                "west_mean_gain": float(west_mean_gain),
                "east_mean_gain": float(east_mean_gain),
                "land_area_frac": float(land_count / float(npix)),
                "land_mean_gain": float(land_mean_gain),
                "ocean_mean_gain": float(ocean_mean_gain),
                "corr_gain_vs_convective_clim_land": float(corr_gain_conv_land),
                "corr_gain_vs_convective_clim_ocean": float(corr_gain_conv_ocean),
                "panel_r2_base": float(panel_r2_base),
                "panel_r2_full_landaware": float(panel_r2_full),
                "panel_gain_r2_landaware": float(panel_gain_r2),
                "panel_beta_dq_base": float(beta_panel_base[1]),
                "panel_beta_dq_full": float(beta_panel_full[1]),
                "panel_beta_lambda_ocean": float(beta_panel_full[2]),
                "panel_beta_land_intercept_shift": float(beta_panel_full[3]),
                "panel_beta_lambda_land_delta": float(beta_panel_full[4]),
                "gain_positive_frac": float(np.nanmean(gain_map > 0.0)),
                "median_gain": float(np.nanmedian(gain_map)),
                "corr_lambda_recon_domainmean_vs_global_lambda_struct": float(corr_recon_global),
            }
        ]
    )
    summary.to_csv(outdir / "summary_metrics.csv", index=False)

    report = [
        "# Experiment O Spatial: Non-Averaged Diagnostics",
        "",
        "## Setup",
        f"- Input: `{input_nc}`",
        f"- Taxon: `{tx.name}` ({tx.p_bot_hpa}-{tx.p_top_hpa} hPa)",
        f"- Train <= {train_end_year}, test = {test_year}",
        f"- Lambda components from: `{m_timeseries_csv}`",
        f"- Convective climatology source: `{conv_name}`",
        "",
        "## Primary criteria",
        f"- spatial corr(gain_map, convective_clim) = {corr_gain_conv:.6f} (threshold > {corr_threshold:.3f})",
        f"- max(gain_map) = {max_gain:.6f} at lat={max_lat:.3f}, lon={max_lon:.3f} (threshold > {hotspot_gain_threshold:.3f})",
        "",
        "## Additional diagnostics",
        f"- west mean gain = {west_mean_gain:.6f}",
        f"- east mean gain = {east_mean_gain:.6f}",
        f"- land mean gain = {land_mean_gain:.6f}",
        f"- ocean mean gain = {ocean_mean_gain:.6f}",
        f"- corr(gain,conv) over land = {corr_gain_conv_land:.6f}",
        f"- corr(gain,conv) over ocean = {corr_gain_conv_ocean:.6f}",
        (
            "- panel (coords,time,scale + land criterion) gain = "
            f"{panel_gain_r2:.6f} "
            f"(R2 base={panel_r2_base:.6f}, R2 full={panel_r2_full:.6f})"
        ),
        (
            "- panel beta lambda ocean / land-delta = "
            f"{float(beta_panel_full[2]):.6f} / {float(beta_panel_full[4]):.6f}"
        ),
        f"- positive gain fraction = {float(np.nanmean(gain_map > 0.0)):.3f}",
        f"- median gain = {float(np.nanmedian(gain_map)):.6f}",
        (
            "- corr(domainmean lambda_local, global lambda_struct) = "
            f"{corr_recon_global:.6f}"
        ),
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("[experiment_O_spatial] done.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
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
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_O_spatial_variance"),
    )
    parser.add_argument("--taxon", type=str, default="FT:850-300")
    parser.add_argument(
        "--scale-edges-km",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
    )
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--west-split-lon", type=float, default=140.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--corr-threshold", type=float, default=0.30)
    parser.add_argument("--hotspot-gain-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        input_nc=args.input_nc,
        m_timeseries_csv=args.m_timeseries_csv,
        m_summary_csv=args.m_summary_csv,
        m_mode_index_csv=args.m_mode_index_csv,
        outdir=args.outdir,
        taxon_spec=args.taxon,
        scale_edges_km=list(args.scale_edges_km),
        train_end_year=args.train_end_year,
        test_year=args.test_year,
        west_split_lon=args.west_split_lon,
        batch_size=args.batch_size,
        corr_threshold=args.corr_threshold,
        hotspot_gain_threshold=args.hotspot_gain_threshold,
    )


if __name__ == "__main__":
    main()
