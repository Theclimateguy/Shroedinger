#!/usr/bin/env python3
"""Experiment F5 spatial follow-up: map fractal surrogates on ERA5/WPWP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        _edge_order,
        _find_var_name,
        _time_to_seconds,
        _to_tyx_da,
        _xy_coordinates_m,
    )
    from clean_experiments.experiment_O_spatial_variance import _build_land_mask
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        _edge_order,
        _find_var_name,
        _time_to_seconds,
        _to_tyx_da,
        _xy_coordinates_m,
    )
    from experiment_O_spatial_variance import _build_land_mask  # type: ignore


EPS = 1e-12


@dataclass
class SpatialConfig:
    quantile: float = 0.80
    max_samples_per_group: int = 600
    patch_size: int = 15
    patch_stride: int = 6
    vario_lags: tuple[int, ...] = (1, 2, 3, 4, 5)


def _load_fields(input_nc: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    if xr is None:
        raise ImportError("xarray is required for NetCDF input")

    ds = xr.open_dataset(input_nc)
    available = list(ds.data_vars)

    roles = ("iwv", "ivt_u", "ivt_v", "precip", "evap")
    names = {r: _find_var_name(available, r, None) for r in roles}

    fields: dict[str, np.ndarray] = {}
    t_coords = None
    lat_coords = None
    lon_coords = None

    for role in roles:
        da = ds[names[role]]
        da, t_dim, y_dim, x_dim = _to_tyx_da(da, level_dim=None, level_index=0)
        if t_coords is None:
            t_coords = np.asarray(da[t_dim].values)
            lat_coords = np.asarray(da[y_dim].values, dtype=float)
            lon_coords = np.asarray(da[x_dim].values, dtype=float)
        fields[role] = np.asarray(da.values, dtype=np.float32)

    ds.close()

    if t_coords is None or lat_coords is None or lon_coords is None:
        raise ValueError("Failed to load coordinates from NetCDF")

    return t_coords, lat_coords, lon_coords, fields, names


def _subsample_indices(idx: np.ndarray, max_samples: int) -> np.ndarray:
    if len(idx) <= max_samples:
        return idx
    pick = np.linspace(0, len(idx) - 1, num=max_samples, dtype=int)
    return idx[pick]


def _select_regimes(lambda_abs: np.ndarray, quantile: float, max_samples: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    q_hi = float(np.quantile(lambda_abs, quantile))
    q_lo = float(np.quantile(lambda_abs, 1.0 - quantile))

    idx_hi = np.where(lambda_abs >= q_hi)[0]
    idx_lo = np.where(lambda_abs <= q_lo)[0]

    idx_hi = _subsample_indices(idx_hi, max_samples)
    idx_lo = _subsample_indices(idx_lo, max_samples)

    if len(idx_hi) < 40 or len(idx_lo) < 40:
        raise ValueError(
            f"Too few samples for spatial regimes: high={len(idx_hi)}, low={len(idx_lo)}. "
            "Reduce quantile or increase max_samples_per_group."
        )

    return idx_hi, idx_lo, q_hi, q_lo


def _build_patch_geometry(ny: int, nx: int, patch_size: int, patch_stride: int) -> dict[str, np.ndarray | int]:
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    half = patch_size // 2

    centers_y = np.arange(half, ny - half, patch_stride, dtype=int)
    centers_x = np.arange(half, nx - half, patch_stride, dtype=int)
    starts_y = centers_y - half
    starts_x = centers_x - half

    if len(starts_y) == 0 or len(starts_x) == 0:
        raise ValueError("Patch geometry is empty; decrease patch_size or stride")

    py = patch_size
    px = patch_size
    ky = np.fft.fftfreq(py)[:, None]
    kx = np.fft.rfftfreq(px)[None, :]
    k = np.sqrt(ky * ky + kx * kx)
    mask_nonzero = k > 0.0
    kval = k[mask_nonzero]
    q_lo, q_hi = np.quantile(kval, [0.15, 0.80])
    mask_k = mask_nonzero & (k >= q_lo) & (k <= q_hi)
    x_psd = np.log(k[mask_k])

    lags = np.asarray([h for h in (1, 2, 3, 4, 5) if h < patch_size], dtype=float)
    if len(lags) < 3:
        raise ValueError("Need at least three valid lags for variogram slope")
    x_vario = np.log(lags)

    return {
        "patch_size": int(patch_size),
        "half": int(half),
        "centers_y": centers_y,
        "centers_x": centers_x,
        "starts_y": starts_y,
        "starts_x": starts_x,
        "mask_k": mask_k,
        "x_psd": x_psd,
        "lags": lags.astype(int),
        "x_vario": x_vario,
    }


def _extract_patches(field: np.ndarray, starts_y: np.ndarray, starts_x: np.ndarray, patch_size: int) -> np.ndarray:
    sw = np.lib.stride_tricks.sliding_window_view(field, (patch_size, patch_size))
    # sw shape: [ny-p+1, nx-p+1, p, p]
    patches = sw[np.ix_(starts_y, starts_x)]
    # [nyc, nxc, p, p] -> [npatch, p, p]
    return patches.reshape(-1, patch_size, patch_size)


def _slope_and_r2(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # y: [npatch, m], x: [m]
    mean_x = float(np.mean(x))
    xc = x - mean_x
    var_x = float(np.sum(xc * xc)) + EPS

    mean_y = np.mean(y, axis=1)
    yc = y - mean_y[:, None]
    cov = np.sum(yc * xc[None, :], axis=1)
    slope = cov / var_x

    y_hat = mean_y[:, None] + slope[:, None] * xc[None, :]
    ss_res = np.sum((y - y_hat) ** 2, axis=1)
    ss_tot = np.sum((y - mean_y[:, None]) ** 2, axis=1)
    r2 = 1.0 - ss_res / (ss_tot + EPS)
    return slope.astype(float), r2.astype(float)


def _estimate_patch_metrics(
    field: np.ndarray,
    *,
    starts_y: np.ndarray,
    starts_x: np.ndarray,
    patch_size: int,
    mask_k: np.ndarray,
    x_psd: np.ndarray,
    lags: np.ndarray,
    x_vario: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patches = _extract_patches(field, starts_y=starts_y, starts_x=starts_x, patch_size=patch_size)
    patches = patches.astype(float)

    centered = patches - np.mean(patches, axis=(1, 2), keepdims=True)
    pwr = np.abs(np.fft.rfft2(centered, axes=(-2, -1))) ** 2
    y_psd = np.log(pwr[:, mask_k] + EPS)
    slope_psd, r2_psd = _slope_and_r2(y=y_psd, x=x_psd)
    beta = -slope_psd

    gamma_list: list[np.ndarray] = []
    for h in lags:
        dx = patches[:, :, h:] - patches[:, :, :-h]
        dy = patches[:, h:, :] - patches[:, :-h, :]
        gamma_h = 0.25 * (
            np.mean(dx * dx, axis=(1, 2)) + np.mean(dy * dy, axis=(1, 2))
        )
        gamma_list.append(gamma_h.astype(float))

    gamma = np.stack(gamma_list, axis=1)
    y_vario = np.log(gamma + EPS)
    slope_vario, r2_vario = _slope_and_r2(y=y_vario, x=x_vario)

    return beta, r2_psd, slope_vario, r2_vario


def _residual_field_at(
    t: int,
    *,
    iwv: np.ndarray,
    ivt_u: np.ndarray,
    ivt_v: np.ndarray,
    precip: np.ndarray,
    evap: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> np.ndarray:
    nt = iwv.shape[0]
    if t <= 0:
        dt = max(float(time_s[1] - time_s[0]), EPS)
        diwv_dt = (iwv[1] - iwv[0]) / dt
    elif t >= nt - 1:
        dt = max(float(time_s[-1] - time_s[-2]), EPS)
        diwv_dt = (iwv[-1] - iwv[-2]) / dt
    else:
        dt = max(float(time_s[t + 1] - time_s[t - 1]), EPS)
        diwv_dt = (iwv[t + 1] - iwv[t - 1]) / dt

    eo_x = _edge_order(len(x_m))
    eo_y = _edge_order(len(y_m))
    div_ivt = np.gradient(ivt_u[t], x_m, axis=1, edge_order=eo_x) + np.gradient(
        ivt_v[t], y_m, axis=0, edge_order=eo_y
    )
    return np.asarray(diwv_dt + div_ivt + (precip[t] - evap[t]), dtype=float)


def _aggregate_group(
    indices: np.ndarray,
    *,
    iwv: np.ndarray,
    ivt_u: np.ndarray,
    ivt_v: np.ndarray,
    precip: np.ndarray,
    evap: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    geom: dict[str, np.ndarray | int],
    tag: str,
) -> dict[str, np.ndarray]:
    starts_y = np.asarray(geom["starts_y"], dtype=int)
    starts_x = np.asarray(geom["starts_x"], dtype=int)
    patch_size = int(geom["patch_size"])
    mask_k = np.asarray(geom["mask_k"], dtype=bool)
    x_psd = np.asarray(geom["x_psd"], dtype=float)
    lags = np.asarray(geom["lags"], dtype=int)
    x_vario = np.asarray(geom["x_vario"], dtype=float)

    nyc = len(starts_y)
    nxc = len(starts_x)
    npatch = nyc * nxc

    sum_beta = np.zeros(npatch, dtype=float)
    sumsq_beta = np.zeros(npatch, dtype=float)
    sum_r2_psd = np.zeros(npatch, dtype=float)
    cnt_beta = np.zeros(npatch, dtype=int)

    sum_vario = np.zeros(npatch, dtype=float)
    sumsq_vario = np.zeros(npatch, dtype=float)
    sum_r2_vario = np.zeros(npatch, dtype=float)
    cnt_vario = np.zeros(npatch, dtype=int)

    for i, t in enumerate(indices):
        field = _residual_field_at(
            int(t),
            iwv=iwv,
            ivt_u=ivt_u,
            ivt_v=ivt_v,
            precip=precip,
            evap=evap,
            time_s=time_s,
            x_m=x_m,
            y_m=y_m,
        )
        beta, r2_psd, vario, r2_vario = _estimate_patch_metrics(
            field,
            starts_y=starts_y,
            starts_x=starts_x,
            patch_size=patch_size,
            mask_k=mask_k,
            x_psd=x_psd,
            lags=lags,
            x_vario=x_vario,
        )

        m_beta = np.isfinite(beta)
        sum_beta[m_beta] += beta[m_beta]
        sumsq_beta[m_beta] += beta[m_beta] * beta[m_beta]
        sum_r2_psd[m_beta] += r2_psd[m_beta]
        cnt_beta[m_beta] += 1

        m_var = np.isfinite(vario)
        sum_vario[m_var] += vario[m_var]
        sumsq_vario[m_var] += vario[m_var] * vario[m_var]
        sum_r2_vario[m_var] += r2_vario[m_var]
        cnt_vario[m_var] += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(indices):
            print(f"[F5-map] {tag}: {i + 1}/{len(indices)} time slices", flush=True)

    beta_mean = np.where(cnt_beta > 0, sum_beta / np.maximum(cnt_beta, 1), np.nan)
    beta_var = np.where(cnt_beta > 0, sumsq_beta / np.maximum(cnt_beta, 1) - beta_mean * beta_mean, np.nan)
    beta_std = np.sqrt(np.maximum(beta_var, 0.0))
    psd_r2_mean = np.where(cnt_beta > 0, sum_r2_psd / np.maximum(cnt_beta, 1), np.nan)

    vario_mean = np.where(cnt_vario > 0, sum_vario / np.maximum(cnt_vario, 1), np.nan)
    vario_var = np.where(cnt_vario > 0, sumsq_vario / np.maximum(cnt_vario, 1) - vario_mean * vario_mean, np.nan)
    vario_std = np.sqrt(np.maximum(vario_var, 0.0))
    vario_r2_mean = np.where(cnt_vario > 0, sum_r2_vario / np.maximum(cnt_vario, 1), np.nan)

    return {
        "beta_mean": beta_mean.reshape(nyc, nxc),
        "beta_std": beta_std.reshape(nyc, nxc),
        "psd_r2_mean": psd_r2_mean.reshape(nyc, nxc),
        "count_beta": cnt_beta.reshape(nyc, nxc).astype(float),
        "vario_mean": vario_mean.reshape(nyc, nxc),
        "vario_std": vario_std.reshape(nyc, nxc),
        "vario_r2_mean": vario_r2_mean.reshape(nyc, nxc),
        "count_vario": cnt_vario.reshape(nyc, nxc).astype(float),
    }


def _zmap(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    m = np.isfinite(x)
    if int(np.sum(m)) == 0:
        return np.full_like(x, np.nan)
    mu = float(np.mean(x[m]))
    sd = float(np.std(x[m]))
    if sd < 1e-12:
        return x - mu
    out = np.full_like(x, np.nan)
    out[m] = (x[m] - mu) / sd
    return out


def _plot_map(
    field: np.ndarray,
    *,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    lat_full: np.ndarray,
    lon_full: np.ndarray,
    land_mask: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str,
    symmetric: bool,
) -> None:
    arr = np.asarray(field, dtype=float)
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    lon2d, lat2d = np.meshgrid(lon_c, lat_c, indexing="xy")

    vmin = float(np.nanquantile(arr, 0.02))
    vmax = float(np.nanquantile(arr, 0.98))
    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))

    im = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    lonf, latf = np.meshgrid(lon_full, lat_full, indexing="xy")
    ax.contour(lonf, latf, land_mask.astype(float), levels=[0.5], colors="k", linewidths=0.5, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(lon_full)), float(np.max(lon_full)))
    ax.set_ylim(float(np.min(lat_full)), float(np.max(lat_full)))

    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_panel(
    *,
    beta_hi: np.ndarray,
    beta_lo: np.ndarray,
    beta_delta: np.ndarray,
    vario_hi: np.ndarray,
    vario_lo: np.ndarray,
    vario_delta: np.ndarray,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    lat_full: np.ndarray,
    lon_full: np.ndarray,
    land_mask: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.0), constrained_layout=True)
    fields = [
        (beta_hi, "PSD beta (high |Lambda|)", "viridis", False),
        (beta_lo, "PSD beta (low |Lambda|)", "viridis", False),
        (beta_delta, "Delta beta (high-low)", "RdBu_r", True),
        (vario_hi, "Variogram slope (high |Lambda|)", "magma", False),
        (vario_lo, "Variogram slope (low |Lambda|)", "magma", False),
        (vario_delta, "Delta variogram (high-low)", "RdBu_r", True),
    ]

    lon2d, lat2d = np.meshgrid(lon_c, lat_c, indexing="xy")
    lonf, latf = np.meshgrid(lon_full, lat_full, indexing="xy")

    for ax, (arr, title, cmap, symmetric) in zip(axes.ravel(), fields):
        x = np.asarray(arr, dtype=float)
        vmin = float(np.nanquantile(x, 0.02))
        vmax = float(np.nanquantile(x, 0.98))
        if symmetric:
            vmax_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vmax_abs, vmax_abs
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))

        im = ax.pcolormesh(lon2d, lat2d, x, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.contour(lonf, latf, land_mask.astype(float), levels=[0.5], colors="k", linewidths=0.45, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        fig.colorbar(im, ax=ax, shrink=0.88)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_experiment(
    *,
    input_nc: Path,
    timeseries_csv: Path,
    outdir: Path,
    cfg: SpatialConfig,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(timeseries_csv)
    if "lambda_struct" not in ts.columns:
        raise KeyError(f"Missing lambda_struct in {timeseries_csv}")
    lambda_abs = np.abs(ts["lambda_struct"].to_numpy(dtype=float))

    print("[F5-map] loading ERA5/WPWP fields...", flush=True)
    time, lat, lon, fields, names = _load_fields(input_nc)

    nt = len(time)
    if len(lambda_abs) != nt:
        raise ValueError(f"Length mismatch: lambda {len(lambda_abs)} vs field time {nt}")

    idx_hi, idx_lo, q_hi, q_lo = _select_regimes(
        lambda_abs=lambda_abs,
        quantile=cfg.quantile,
        max_samples=cfg.max_samples_per_group,
    )

    print(
        f"[F5-map] selected regimes: high={len(idx_hi)} (|Lambda|>={q_hi:.3e}), "
        f"low={len(idx_lo)} (|Lambda|<={q_lo:.3e})",
        flush=True,
    )

    iwv = fields["iwv"]
    ivt_u = fields["ivt_u"]
    ivt_v = fields["ivt_v"]
    precip = fields["precip"]
    evap = fields["evap"]

    ny, nx = iwv.shape[1], iwv.shape[2]
    geom = _build_patch_geometry(ny=ny, nx=nx, patch_size=cfg.patch_size, patch_stride=cfg.patch_stride)
    centers_y = np.asarray(geom["centers_y"], dtype=int)
    centers_x = np.asarray(geom["centers_x"], dtype=int)
    lat_c = lat[centers_y]
    lon_c = lon[centers_x]

    time_s = _time_to_seconds(time)
    x_m, y_m = _xy_coordinates_m(lat, lon)
    land_mask = _build_land_mask(lat, lon)

    print("[F5-map] aggregating high-|Lambda| patch metrics...", flush=True)
    agg_hi = _aggregate_group(
        idx_hi,
        iwv=iwv,
        ivt_u=ivt_u,
        ivt_v=ivt_v,
        precip=precip,
        evap=evap,
        time_s=time_s,
        x_m=x_m,
        y_m=y_m,
        geom=geom,
        tag="high",
    )

    print("[F5-map] aggregating low-|Lambda| patch metrics...", flush=True)
    agg_lo = _aggregate_group(
        idx_lo,
        iwv=iwv,
        ivt_u=ivt_u,
        ivt_v=ivt_v,
        precip=precip,
        evap=evap,
        time_s=time_s,
        x_m=x_m,
        y_m=y_m,
        geom=geom,
        tag="low",
    )

    beta_hi = agg_hi["beta_mean"]
    beta_lo = agg_lo["beta_mean"]
    beta_delta = beta_hi - beta_lo

    vario_hi = agg_hi["vario_mean"]
    vario_lo = agg_lo["vario_mean"]
    vario_delta = vario_hi - vario_lo

    pooled_beta = np.sqrt(0.5 * (agg_hi["beta_std"] ** 2 + agg_lo["beta_std"] ** 2) + EPS)
    pooled_vario = np.sqrt(0.5 * (agg_hi["vario_std"] ** 2 + agg_lo["vario_std"] ** 2) + EPS)
    beta_effect = beta_delta / pooled_beta
    vario_effect = vario_delta / pooled_vario

    composite = 0.5 * (_zmap(beta_delta) + _zmap(vario_delta))
    sign_agree = np.sign(beta_delta) == np.sign(vario_delta)

    n_common = np.minimum(agg_hi["count_beta"], agg_lo["count_beta"])

    patch_rows: list[dict[str, float | int | bool]] = []
    for iy, yy in enumerate(centers_y):
        for ix, xx in enumerate(centers_x):
            patch_rows.append(
                {
                    "patch_iy": int(iy),
                    "patch_ix": int(ix),
                    "lat_center": float(lat[yy]),
                    "lon_center": float(lon[xx]),
                    "n_common": float(n_common[iy, ix]),
                    "beta_high": float(beta_hi[iy, ix]),
                    "beta_low": float(beta_lo[iy, ix]),
                    "beta_delta": float(beta_delta[iy, ix]),
                    "beta_effect": float(beta_effect[iy, ix]),
                    "vario_high": float(vario_hi[iy, ix]),
                    "vario_low": float(vario_lo[iy, ix]),
                    "vario_delta": float(vario_delta[iy, ix]),
                    "vario_effect": float(vario_effect[iy, ix]),
                    "psd_r2_high": float(agg_hi["psd_r2_mean"][iy, ix]),
                    "psd_r2_low": float(agg_lo["psd_r2_mean"][iy, ix]),
                    "vario_r2_high": float(agg_hi["vario_r2_mean"][iy, ix]),
                    "vario_r2_low": float(agg_lo["vario_r2_mean"][iy, ix]),
                    "composite_delta": float(composite[iy, ix]),
                    "sign_agree": bool(sign_agree[iy, ix]),
                }
            )

    patch_df = pd.DataFrame(patch_rows)
    patch_df.to_csv(outdir / "experiment_F5_spatial_patch_metrics.csv", index=False)

    np.savez_compressed(
        outdir / "experiment_F5_spatial_maps.npz",
        lat_centers=lat_c,
        lon_centers=lon_c,
        beta_high=beta_hi,
        beta_low=beta_lo,
        beta_delta=beta_delta,
        beta_effect=beta_effect,
        vario_high=vario_hi,
        vario_low=vario_lo,
        vario_delta=vario_delta,
        vario_effect=vario_effect,
        composite_delta=composite,
        sign_agree=sign_agree.astype(float),
        n_common=n_common,
    )

    print("[F5-map] rendering maps...", flush=True)
    _plot_panel(
        beta_hi=beta_hi,
        beta_lo=beta_lo,
        beta_delta=beta_delta,
        vario_hi=vario_hi,
        vario_lo=vario_lo,
        vario_delta=vario_delta,
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        out_path=outdir / "plot_F5_fractal_maps_panel.png",
    )

    _plot_map(
        composite,
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        title="F5 composite fractal delta (high-|Lambda| vs low-|Lambda|)",
        cbar_label="Composite z-delta",
        out_path=outdir / "plot_F5_fractal_composite_delta.png",
        cmap="RdBu_r",
        symmetric=True,
    )

    _plot_map(
        beta_effect,
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        title="F5 PSD beta effect size (high-low)",
        cbar_label="Effect size",
        out_path=outdir / "plot_F5_psd_effect_map.png",
        cmap="RdBu_r",
        symmetric=True,
    )

    _plot_map(
        vario_effect,
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        title="F5 variogram slope effect size (high-low)",
        cbar_label="Effect size",
        out_path=outdir / "plot_F5_variogram_effect_map.png",
        cmap="RdBu_r",
        symmetric=True,
    )

    _plot_map(
        sign_agree.astype(float),
        lat_c=lat_c,
        lon_c=lon_c,
        lat_full=lat,
        lon_full=lon,
        land_mask=land_mask,
        title="F5 estimator sign agreement (1=agree)",
        cbar_label="Agreement",
        out_path=outdir / "plot_F5_estimator_sign_agreement.png",
        cmap="viridis",
        symmetric=False,
    )

    valid_delta = np.isfinite(beta_delta) & np.isfinite(vario_delta)
    corr_delta = float(np.corrcoef(beta_delta[valid_delta], vario_delta[valid_delta])[0, 1]) if np.any(valid_delta) else float("nan")
    agree_frac = float(np.mean(sign_agree[valid_delta])) if np.any(valid_delta) else float("nan")

    top_hot = (
        patch_df.sort_values("composite_delta", ascending=False)
        .head(10)[["lat_center", "lon_center", "composite_delta", "beta_delta", "vario_delta"]]
        .reset_index(drop=True)
    )
    top_hot.to_csv(outdir / "experiment_F5_spatial_top_hotspots.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "input_nc": str(input_nc),
                "timeseries_csv": str(timeseries_csv),
                "iwv_var": names["iwv"],
                "ivt_u_var": names["ivt_u"],
                "ivt_v_var": names["ivt_v"],
                "precip_var": names["precip"],
                "evap_var": names["evap"],
                "n_time": int(nt),
                "q_high": q_hi,
                "q_low": q_lo,
                "n_high": int(len(idx_hi)),
                "n_low": int(len(idx_lo)),
                "patch_size": int(cfg.patch_size),
                "patch_stride": int(cfg.patch_stride),
                "n_patch_y": int(len(centers_y)),
                "n_patch_x": int(len(centers_x)),
                "corr_delta_beta_vs_vario": corr_delta,
                "sign_agreement_frac": agree_frac,
                "mean_beta_delta": float(np.nanmean(beta_delta)),
                "mean_vario_delta": float(np.nanmean(vario_delta)),
                "mean_composite_delta": float(np.nanmean(composite)),
            }
        ]
    )
    summary.to_csv(outdir / "experiment_F5_spatial_summary.csv", index=False)

    report_lines = [
        "# F5 Spatial Fractal Visualization",
        "",
        f"- input: `{input_nc}`",
        f"- high-|Lambda| threshold (q={cfg.quantile:.2f}): {q_hi:.6e}, n={len(idx_hi)}",
        f"- low-|Lambda| threshold (q={1.0-cfg.quantile:.2f}): {q_lo:.6e}, n={len(idx_lo)}",
        f"- patch geometry: size={cfg.patch_size}, stride={cfg.patch_stride}, grid={len(centers_y)}x{len(centers_x)}",
        "",
        "## Consistency",
        f"- corr(delta_beta, delta_variogram) = {corr_delta:.6f}",
        f"- sign agreement fraction = {agree_frac:.3f}",
        "",
        "## Mean Spatial Deltas (high-|Lambda| minus low-|Lambda|)",
        f"- mean delta PSD beta = {float(np.nanmean(beta_delta)):.6f}",
        f"- mean delta variogram slope = {float(np.nanmean(vario_delta)):.6f}",
        f"- mean composite delta = {float(np.nanmean(composite)):.6f}",
        "",
        "## Main figures",
        "- `plot_F5_fractal_maps_panel.png`",
        "- `plot_F5_fractal_composite_delta.png`",
        "- `plot_F5_psd_effect_map.png`",
        "- `plot_F5_variogram_effect_map.png`",
        "- `plot_F5_estimator_sign_agreement.png`",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[F5-map] outputs saved to: {outdir.resolve()}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc")
    p.add_argument(
        "--timeseries-csv",
        default="clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv",
    )
    p.add_argument("--out", default="clean_experiments/results/experiment_F5_spatial_fractal_maps")

    p.add_argument("--quantile", type=float, default=0.80)
    p.add_argument("--max-samples", type=int, default=600)
    p.add_argument("--patch-size", type=int, default=15)
    p.add_argument("--patch-stride", type=int, default=6)

    args = p.parse_args()

    cfg = SpatialConfig(
        quantile=float(args.quantile),
        max_samples_per_group=max(40, int(args.max_samples)),
        patch_size=max(7, int(args.patch_size)),
        patch_stride=max(1, int(args.patch_stride)),
    )

    run_experiment(
        input_nc=Path(args.input),
        timeseries_csv=Path(args.timeseries_csv),
        outdir=Path(args.out),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
