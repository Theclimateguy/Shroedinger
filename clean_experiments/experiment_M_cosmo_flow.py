#!/usr/bin/env python3
"""Experiment M (Experiment 15): structural-scale cosmological-flow proxy on atmospheric data.

This experiment implements a full multiscale pipeline:
1) define scale coordinate mu from wavelength bands,
2) extract modal coefficients a_{i,mu}(t) from Fourier bands,
3) build density matrices rho_mu(t) from rolling covariances,
4) derive structural Lambda proxy from interscale transport noncommutativity,
5) test whether Lambda improves moisture-balance residual closure out-of-sample.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:  # pragma: no cover - optional dependency
    xr = None


K_BOLTZMANN = 1.380649e-23
EARTH_RADIUS_M = 6_371_000.0


VAR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "iwv": (
        "iwv",
        "tcwv",
        "tciwv",
        "total_column_water_vapour",
        "total_column_water_vapor",
    ),
    "ivt_u": (
        "ivt_u",
        "viwve",
        "vertical_integral_of_eastward_water_vapour_flux",
        "vertical_integral_of_eastward_water_vapor_flux",
    ),
    "ivt_v": (
        "ivt_v",
        "viwvn",
        "vertical_integral_of_northward_water_vapour_flux",
        "vertical_integral_of_northward_water_vapor_flux",
    ),
    "precip": (
        "precip",
        "tp",
        "total_precipitation",
        "pr",
    ),
    "evap": (
        "evap",
        "e",
        "evaporation",
    ),
    "u": (
        "u",
        "u10",
        "u100",
        "u850",
        "u1000",
    ),
    "v": (
        "v",
        "v10",
        "v100",
        "v850",
        "v1000",
    ),
    "temp": (
        "temp",
        "t2m",
        "t",
        "temperature",
        "t850",
    ),
    "pressure": (
        "sp",
        "surface_pressure",
        "msl",
        "pressure",
        "p",
    ),
    "density": (
        "air_density",
        "rho_air",
        "n_air",
        "number_density",
    ),
}


@dataclass(frozen=True)
class LoadedData:
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    fields: dict[str, np.ndarray]
    variable_names: dict[str, str]


def _parse_float_list(value: str) -> list[float]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated float list.")
    vals = [float(x) for x in items]
    return vals


def _find_var_name(available: list[str], role: str, override: str | None) -> str:
    if override:
        if override not in available:
            raise KeyError(f"Variable '{override}' requested for role '{role}' not found in input dataset.")
        return override
    for cand in VAR_CANDIDATES[role]:
        if cand in available:
            return cand
    raise KeyError(
        f"No variable found for role '{role}'. Tried {VAR_CANDIDATES[role]}. "
        f"Provide --{role.replace('_', '-')}-var explicitly."
    )


def _infer_dim_name(dims: tuple[str, ...], candidates: tuple[str, ...], fallback: str | None = None) -> str:
    for c in candidates:
        if c in dims:
            return c
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not infer dimension name from candidates {candidates}; dims={dims}")


def _to_tyx_da(
    da,
    *,
    level_dim: str | None,
    level_index: int,
):
    if level_dim and level_dim in da.dims:
        da = da.isel({level_dim: level_index})

    # Squeeze singleton non-core dimensions automatically.
    for d in list(da.dims):
        if da.sizes[d] == 1:
            da = da.isel({d: 0})

    dims = tuple(da.dims)
    t_dim = _infer_dim_name(dims, ("time", "valid_time", "datetime", "date"), fallback=dims[0])
    y_dim = _infer_dim_name(dims, ("lat", "latitude", "y", "rlat"), fallback=dims[-2])
    x_dim = _infer_dim_name(dims, ("lon", "longitude", "x", "rlon"), fallback=dims[-1])

    keep = (t_dim, y_dim, x_dim)
    if len(set(keep)) != 3:
        raise ValueError(f"Could not map dimensions to (time,lat,lon). dims={dims}, mapped={keep}")

    da = da.transpose(t_dim, y_dim, x_dim)
    return da, t_dim, y_dim, x_dim


def _slice_tyx(
    arr: np.ndarray,
    *,
    time_stride: int,
    lat_stride: int,
    lon_stride: int,
    max_time: int | None,
) -> np.ndarray:
    arr = arr[::time_stride, ::lat_stride, ::lon_stride]
    if max_time is not None:
        arr = arr[:max_time]
    return np.asarray(arr, dtype=float)


def _load_from_xarray(
    path: Path,
    *,
    var_overrides: dict[str, str | None],
    level_dim: str | None,
    level_index: int,
    time_stride: int,
    lat_stride: int,
    lon_stride: int,
    max_time: int | None,
) -> LoadedData:
    if xr is None:
        raise ImportError(
            "xarray is required for NetCDF/Zarr input. Install xarray or provide a .npz input file."
        )

    ds = xr.open_dataset(path)
    available = list(ds.data_vars)

    variable_names: dict[str, str] = {}
    for role in ("iwv", "ivt_u", "ivt_v", "precip", "evap", "u", "v"):
        variable_names[role] = _find_var_name(available, role, var_overrides.get(role))

    # Optional controls.
    for opt_role in ("temp", "pressure", "density"):
        override = var_overrides.get(opt_role)
        if override is not None:
            variable_names[opt_role] = _find_var_name(available, opt_role, override)
        else:
            for cand in VAR_CANDIDATES[opt_role]:
                if cand in available:
                    variable_names[opt_role] = cand
                    break

    fields: dict[str, np.ndarray] = {}
    t_coords = None
    lat_coords = None
    lon_coords = None

    for role, name in variable_names.items():
        da = ds[name]
        da, t_dim, y_dim, x_dim = _to_tyx_da(da, level_dim=level_dim, level_index=level_index)

        if t_coords is None:
            t_coords = np.asarray(da[t_dim].values)
            lat_coords = np.asarray(da[y_dim].values)
            lon_coords = np.asarray(da[x_dim].values)

        values = _slice_tyx(
            da.values,
            time_stride=time_stride,
            lat_stride=lat_stride,
            lon_stride=lon_stride,
            max_time=max_time,
        )
        fields[role] = values

    if t_coords is None or lat_coords is None or lon_coords is None:
        raise ValueError("Failed to load coordinates from dataset.")

    t_coords = t_coords[::time_stride]
    lat_coords = lat_coords[::lat_stride]
    lon_coords = lon_coords[::lon_stride]
    if max_time is not None:
        t_coords = t_coords[:max_time]

    nt = len(t_coords)
    ny = len(lat_coords)
    nx = len(lon_coords)

    for role, arr in fields.items():
        if arr.shape != (nt, ny, nx):
            raise ValueError(
                f"Variable '{role}' has shape {arr.shape}, expected {(nt, ny, nx)} after slicing/alignment."
            )

    return LoadedData(time=t_coords, lat=lat_coords, lon=lon_coords, fields=fields, variable_names=variable_names)


def _load_from_npz(
    path: Path,
    *,
    var_overrides: dict[str, str | None],
    time_stride: int,
    lat_stride: int,
    lon_stride: int,
    max_time: int | None,
) -> LoadedData:
    npz = np.load(path, allow_pickle=False)
    available = list(npz.keys())

    variable_names: dict[str, str] = {}
    for role in ("iwv", "ivt_u", "ivt_v", "precip", "evap", "u", "v"):
        variable_names[role] = _find_var_name(available, role, var_overrides.get(role))

    for opt_role in ("temp", "pressure", "density"):
        override = var_overrides.get(opt_role)
        if override is not None:
            variable_names[opt_role] = _find_var_name(available, opt_role, override)
        else:
            for cand in VAR_CANDIDATES[opt_role]:
                if cand in available:
                    variable_names[opt_role] = cand
                    break

    fields: dict[str, np.ndarray] = {}
    for role, name in variable_names.items():
        arr = np.asarray(npz[name], dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"NPZ variable '{name}' for role '{role}' must have shape (time,lat,lon), got {arr.shape}")
        fields[role] = _slice_tyx(
            arr,
            time_stride=time_stride,
            lat_stride=lat_stride,
            lon_stride=lon_stride,
            max_time=max_time,
        )

    nt, ny, nx = fields["iwv"].shape

    if "time" in npz:
        time = np.asarray(npz["time"])[::time_stride]
        if max_time is not None:
            time = time[:max_time]
    else:
        time = np.arange(nt, dtype=float)

    if "lat" in npz:
        lat = np.asarray(npz["lat"], dtype=float)[::lat_stride]
    else:
        lat = np.arange(ny, dtype=float)

    if "lon" in npz:
        lon = np.asarray(npz["lon"], dtype=float)[::lon_stride]
    else:
        lon = np.arange(nx, dtype=float)

    if max_time is not None:
        time = time[:max_time]

    if len(time) != nt:
        raise ValueError(f"Time coordinate length mismatch: {len(time)} vs {nt}")
    if len(lat) != ny:
        raise ValueError(f"Lat coordinate length mismatch: {len(lat)} vs {ny}")
    if len(lon) != nx:
        raise ValueError(f"Lon coordinate length mismatch: {len(lon)} vs {nx}")

    return LoadedData(time=time, lat=lat, lon=lon, fields=fields, variable_names=variable_names)


def _load_data(
    path: Path,
    *,
    var_overrides: dict[str, str | None],
    level_dim: str | None,
    level_index: int,
    time_stride: int,
    lat_stride: int,
    lon_stride: int,
    max_time: int | None,
) -> LoadedData:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_from_npz(
            path,
            var_overrides=var_overrides,
            time_stride=time_stride,
            lat_stride=lat_stride,
            lon_stride=lon_stride,
            max_time=max_time,
        )
    if suffix in {".nc", ".nc4", ".netcdf"} or path.is_dir():
        return _load_from_xarray(
            path,
            var_overrides=var_overrides,
            level_dim=level_dim,
            level_index=level_index,
            time_stride=time_stride,
            lat_stride=lat_stride,
            lon_stride=lon_stride,
            max_time=max_time,
        )
    raise ValueError("Unsupported input format. Use .npz or NetCDF/Zarr path.")


def _time_to_seconds(time: np.ndarray) -> np.ndarray:
    t = np.asarray(time)
    if np.issubdtype(t.dtype, np.datetime64):
        sec = (t - t[0]).astype("timedelta64[s]").astype(float)
    elif np.issubdtype(t.dtype, np.number):
        sec = t.astype(float) - float(t.astype(float)[0])
    else:
        # Attempt datetime parsing via pandas.
        ts = pd.to_datetime(t)
        sec = (ts - ts[0]).total_seconds().to_numpy(dtype=float)

    if len(sec) < 3:
        raise ValueError("Need at least 3 time steps for gradients and blocked CV.")
    return sec


def _xy_coordinates_m(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    # Degree-based geophysical coordinates.
    if np.nanmax(np.abs(lat)) <= 90.0 and np.nanmax(np.abs(lon)) <= 360.0:
        lat0 = float(np.nanmean(lat))
        y_m = EARTH_RADIUS_M * np.deg2rad(lat - lat[0])
        x_m = EARTH_RADIUS_M * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon - lon[0])
    else:
        # Already projected coordinates.
        y_m = lat - lat[0]
        x_m = lon - lon[0]

    if len(y_m) < 3 or len(x_m) < 3:
        raise ValueError("Need at least 3x3 spatial grid for derivatives and multiscale decomposition.")

    return x_m, y_m


def _edge_order(size: int) -> int:
    return 2 if size >= 3 else 1


def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < eps:
        return x - mu
    return (x - mu) / sd


def _compute_budget_residual_series(
    *,
    iwv: np.ndarray,
    ivt_u: np.ndarray,
    ivt_v: np.ndarray,
    precip: np.ndarray,
    evap: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    precip_factor: float,
    evap_factor: float,
    residual_mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    eo_t = _edge_order(len(time_s))
    eo_x = _edge_order(len(x_m))
    eo_y = _edge_order(len(y_m))

    diwv_dt = np.gradient(iwv, time_s, axis=0, edge_order=eo_t)
    div_ivt = np.gradient(ivt_u, x_m, axis=2, edge_order=eo_x) + np.gradient(ivt_v, y_m, axis=1, edge_order=eo_y)
    p_minus_e = precip_factor * precip - evap_factor * evap

    diwv_dt_mean = np.mean(diwv_dt, axis=(1, 2))
    div_ivt_mean = np.mean(div_ivt, axis=(1, 2))
    p_minus_e_mean = np.mean(p_minus_e, axis=(1, 2))
    residual_physical_mean = diwv_dt_mean + div_ivt_mean + p_minus_e_mean

    if residual_mode == "component_zscore":
        res0 = _zscore(diwv_dt_mean) + _zscore(div_ivt_mean) + _zscore(p_minus_e_mean)
    elif residual_mode == "physical_zscore":
        res0 = _zscore(residual_physical_mean)
    elif residual_mode == "physical_raw":
        res0 = residual_physical_mean
    else:
        raise ValueError(
            f"Unknown residual_mode='{residual_mode}'. "
            "Use one of: component_zscore, physical_zscore, physical_raw."
        )

    components = {
        "d_iwv_dt_mean": diwv_dt_mean,
        "div_ivt_mean": div_ivt_mean,
        "p_minus_e_mean": p_minus_e_mean,
        "residual_physical_mean": residual_physical_mean,
    }
    return np.asarray(res0, dtype=float), components


def _compute_vorticity(u: np.ndarray, v: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    eo_x = _edge_order(len(x_m))
    eo_y = _edge_order(len(y_m))
    dv_dx = np.gradient(v, x_m, axis=2, edge_order=eo_x)
    du_dy = np.gradient(u, y_m, axis=1, edge_order=eo_y)
    return dv_dx - du_dy


def _count_local_peaks(field: np.ndarray, quantile: float) -> int:
    f = np.asarray(field, dtype=float)
    thr = float(np.quantile(f, quantile))
    if not np.isfinite(thr):
        return 1

    mask = f >= thr
    if int(mask.sum()) == 0:
        return 1

    local = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            local &= f >= np.roll(np.roll(f, dy, axis=0), dx, axis=1)

    n_peaks = int(local.sum())
    return max(n_peaks, 1)


def _build_band_masks(
    *,
    ny: int,
    nx: int,
    dy_km: float,
    dx_km: float,
    scale_edges_km: list[float],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[float], list[float], np.ndarray]:
    if len(scale_edges_km) < 2:
        raise ValueError("Need at least two scale edges.")

    edges = np.asarray(scale_edges_km, dtype=float)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("scale_edges_km must be strictly increasing.")

    ky = np.fft.fftfreq(ny, d=dy_km)[:, None]
    kx = np.fft.rfftfreq(nx, d=dx_km)[None, :]
    k_mag = np.sqrt(kx * kx + ky * ky)
    wavelength_km = np.full_like(k_mag, np.inf, dtype=float)
    nz = k_mag > 0.0
    wavelength_km[nz] = 1.0 / k_mag[nz]

    masks: list[np.ndarray] = []
    centers: list[float] = []
    mus: list[float] = []

    l_ref = float(edges[-1])
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        mask = (wavelength_km >= lo) & (wavelength_km < hi) & np.isfinite(wavelength_km)
        if int(mask.sum()) == 0:
            raise ValueError(
                f"Scale band [{lo},{hi}) km has no Fourier cells for current grid. "
                "Adjust --scale-edges-km or use finer grid."
            )
        masks.append(mask)
        center = float(np.sqrt(lo * hi))
        centers.append(center)
        mus.append(float(np.log2(l_ref / center)))

    return masks, ky[:, 0], kx[0, :], centers, mus, wavelength_km


def _select_mode_indices(
    *,
    mode_fields: dict[str, np.ndarray],
    masks: list[np.ndarray],
    n_modes_per_var: int,
    wavelength_km: np.ndarray,
) -> tuple[list[dict[str, list[tuple[int, int]]]], pd.DataFrame]:
    roles = list(mode_fields.keys())
    nt = next(iter(mode_fields.values())).shape[0]
    ny, nx_r = next(iter(mode_fields.values())).shape[1], np.fft.rfft2(next(iter(mode_fields.values()))[0]).shape[1]

    mean_power: dict[str, np.ndarray] = {r: np.zeros((ny, nx_r), dtype=float) for r in roles}

    for t in range(nt):
        for r in roles:
            field = mode_fields[r][t]
            fhat = np.fft.rfft2(field - np.mean(field))
            mean_power[r] += np.abs(fhat) ** 2

    for r in roles:
        mean_power[r] /= float(nt)

    selected: list[dict[str, list[tuple[int, int]]]] = []
    rows: list[dict[str, float | int | str]] = []

    for b, mask in enumerate(masks):
        by_role: dict[str, list[tuple[int, int]]] = {}
        idx_all = np.argwhere(mask)
        for r in roles:
            if len(idx_all) == 0:
                by_role[r] = []
                continue
            vals = mean_power[r][mask]
            order = np.argsort(vals)[::-1]
            keep = idx_all[order[: min(n_modes_per_var, len(order))]]
            idx_keep = [(int(iy), int(ix)) for iy, ix in keep]
            by_role[r] = idx_keep

            for rank, (iy, ix) in enumerate(idx_keep):
                rows.append(
                    {
                        "band_id": int(b),
                        "field": r,
                        "mode_rank": int(rank),
                        "ky_idx": int(iy),
                        "kx_idx": int(ix),
                        "mean_power": float(mean_power[r][iy, ix]),
                        "wavelength_km": float(wavelength_km[iy, ix]),
                    }
                )

        selected.append(by_role)

    meta_df = pd.DataFrame(rows)
    return selected, meta_df


def _build_coefficients_and_structures(
    *,
    mode_fields: dict[str, np.ndarray],
    vorticity: np.ndarray,
    masks: list[np.ndarray],
    selected: list[dict[str, list[tuple[int, int]]]],
    peak_quantile: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    roles = list(mode_fields.keys())
    nt, ny, nx = vorticity.shape

    coeff_sizes = [sum(len(selected[b][r]) for r in roles) for b in range(len(masks))]
    coeff_by_band = [np.zeros((nt, m), dtype=np.complex128) for m in coeff_sizes]
    n_struct = np.zeros((nt, len(masks)), dtype=float)

    for t in range(nt):
        fft_cache: dict[str, np.ndarray] = {}
        for r in roles:
            field = np.nan_to_num(mode_fields[r][t], nan=0.0, posinf=0.0, neginf=0.0)
            fft_cache[r] = np.fft.rfft2(field - np.mean(field))

        zeta = np.nan_to_num(vorticity[t], nan=0.0, posinf=0.0, neginf=0.0)
        fft_zeta = np.fft.rfft2(zeta - np.mean(zeta))

        for b, mask in enumerate(masks):
            vals: list[complex] = []
            for r in roles:
                fhat = fft_cache[r]
                vals.extend(fhat[iy, ix] for iy, ix in selected[b][r])
            if len(vals) > 0:
                coeff_by_band[b][t, :] = np.nan_to_num(np.asarray(vals, dtype=np.complex128), nan=0.0, posinf=0.0, neginf=0.0)

            zeta_band = np.fft.irfft2(fft_zeta * mask, s=(ny, nx)).real
            n_struct[t, b] = float(_count_local_peaks(np.abs(zeta_band), quantile=peak_quantile))

    n_struct = np.where(n_struct < 1.0, 1.0, n_struct)
    return coeff_by_band, n_struct


def _compute_rho_and_lambda(
    *,
    coeff_by_band: list[np.ndarray],
    n_struct: np.ndarray,
    window: int,
    ridge: float,
    cov_shrinkage: float,
    coherence_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= cov_shrinkage <= 1.0):
        raise ValueError(f"cov_shrinkage must be in [0,1], got {cov_shrinkage}")

    n_bands = len(coeff_by_band)
    nt = coeff_by_band[0].shape[0]

    coeff_scaled: list[np.ndarray] = []
    for b in range(n_bands):
        arr = np.nan_to_num(np.asarray(coeff_by_band[b], dtype=np.complex128), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[1] == 0:
            coeff_scaled.append(arr)
            continue
        rms = np.sqrt(np.mean(np.abs(arr) ** 2, axis=0))
        rms = np.where(rms < 1e-12, 1.0, rms)
        coeff_scaled.append(arr / rms[None, :])

    rho_cache: list[list[np.ndarray]] = [[np.eye(max(coeff_scaled[b].shape[1], 1), dtype=np.complex128)] * nt for b in range(n_bands)]
    coh = np.zeros((nt, n_bands), dtype=float)
    trace_err = np.zeros((nt, n_bands), dtype=float)
    min_eig = np.zeros((nt, n_bands), dtype=float)
    lambda_mu = np.zeros((nt, n_bands), dtype=float)

    for b in range(n_bands):
        m = coeff_scaled[b].shape[1]
        if m == 0:
            continue
        rho_cache[b] = [np.eye(m, dtype=np.complex128) / float(m) for _ in range(nt)]
        eye = np.eye(m, dtype=np.complex128)

        for t in range(nt):
            i0 = max(0, t - window + 1)
            a = np.nan_to_num(coeff_scaled[b][i0 : t + 1], nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                c = (a.conj().T @ a) / float(len(a))
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            c = 0.5 * (c + c.conj().T)
            if cov_shrinkage > 0.0:
                c = (1.0 - cov_shrinkage) * c + cov_shrinkage * np.diag(np.diag(c))
            c = c + ridge * eye
            tr = float(np.real(np.trace(c)))
            if tr < 1e-14:
                rho = eye / float(m)
            else:
                rho = c / tr
            rho = 0.5 * (rho + rho.conj().T)
            rho_cache[b][t] = rho

            eigs = np.linalg.eigvalsh(rho)
            min_eig[t, b] = float(np.min(np.real(eigs)))
            trace_err[t, b] = float(abs(np.real(np.trace(rho)) - 1.0))
            offdiag = rho - np.diag(np.diag(rho))
            offdiag_norm = float(np.linalg.norm(offdiag, ord="fro"))
            if coherence_mode == "offdiag_fro":
                coh[t, b] = offdiag_norm
            elif coherence_mode == "relative_offdiag_fro":
                coh[t, b] = offdiag_norm / (float(np.linalg.norm(rho, ord="fro")) + 1e-12)
            else:
                raise ValueError(
                    f"Unknown coherence_mode='{coherence_mode}'. "
                    "Use one of: offdiag_fro, relative_offdiag_fro."
                )

    for b in range(n_bands - 1):
        m1 = coeff_scaled[b].shape[1]
        m2 = coeff_scaled[b + 1].shape[1]
        if m1 == 0 or m2 == 0:
            continue

        eye1 = np.eye(m1, dtype=np.complex128)
        eye2 = np.eye(m2, dtype=np.complex128)

        for t in range(nt):
            i0 = max(0, t - window + 1)
            a = np.nan_to_num(coeff_scaled[b][i0 : t + 1], nan=0.0, posinf=0.0, neginf=0.0)
            bnext = np.nan_to_num(coeff_scaled[b + 1][i0 : t + 1], nan=0.0, posinf=0.0, neginf=0.0)

            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                g1 = a.conj().T @ a + ridge * eye1
                g2 = bnext.conj().T @ bnext + ridge * eye2
                rhs_fwd = a.conj().T @ bnext
                rhs_rev = bnext.conj().T @ a
            g1 = np.nan_to_num(g1, nan=0.0, posinf=0.0, neginf=0.0)
            g2 = np.nan_to_num(g2, nan=0.0, posinf=0.0, neginf=0.0)
            rhs_fwd = np.nan_to_num(rhs_fwd, nan=0.0, posinf=0.0, neginf=0.0)
            rhs_rev = np.nan_to_num(rhs_rev, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                m_fwd = np.linalg.solve(g1, rhs_fwd)  # m1 x m2
            except np.linalg.LinAlgError:
                m_fwd = np.linalg.pinv(g1, rcond=1e-10) @ rhs_fwd
            try:
                m_rev = np.linalg.solve(g2, rhs_rev)  # m2 x m1
            except np.linalg.LinAlgError:
                m_rev = np.linalg.pinv(g2, rcond=1e-10) @ rhs_rev

            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                g_fwd = m_fwd @ m_fwd.conj().T
                g_back = m_rev.conj().T @ m_rev
                comm = g_fwd @ g_back - g_back @ g_fwd
            g_fwd = np.nan_to_num(g_fwd, nan=0.0, posinf=0.0, neginf=0.0)
            g_back = np.nan_to_num(g_back, nan=0.0, posinf=0.0, neginf=0.0)
            comm = np.nan_to_num(comm, nan=0.0, posinf=0.0, neginf=0.0)
            f_phys = 0.5 * (1j * comm + (1j * comm).conj().T)

            rho = rho_cache[b][t]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                lam_val = np.real(np.trace(f_phys @ rho))
            lambda_mu[t, b] = float(np.nan_to_num(lam_val, nan=0.0, posinf=0.0, neginf=0.0))

    w = n_struct / np.sum(n_struct, axis=1, keepdims=True)
    lambda_struct = np.sum(w * coh * lambda_mu, axis=1)

    return lambda_struct, w, coh, lambda_mu, trace_err, min_eig


def _blocked_splits(n: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n < n_folds * 4:
        raise ValueError(f"Too few samples ({n}) for n_folds={n_folds}; increase data size or reduce folds.")

    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    all_idx = np.arange(n, dtype=int)
    for fs in fold_sizes:
        end = start + int(fs)
        test = np.arange(start, end, dtype=int)
        train = np.concatenate([all_idx[:start], all_idx[end:]])
        splits.append((train, test))
        start = end
    return splits


def _fit_ridge_scaled(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    mu = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    x_train_z = (x_train - mu) / sd
    x_eval_z = (x_eval - mu) / sd
    x_train_z = np.nan_to_num(x_train_z, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_z = np.nan_to_num(x_eval_z, nan=0.0, posinf=0.0, neginf=0.0)

    x1 = np.column_stack([np.ones(len(x_train_z)), x_train_z])
    reg = np.diag(np.r_[0.0, np.full(x_train_z.shape[1], ridge_alpha, dtype=float)])
    lhs = x1.T @ x1 + reg
    rhs = x1.T @ y_train

    try:
        beta_std = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta_std = np.linalg.pinv(lhs, rcond=1e-10) @ rhs

    if not np.all(np.isfinite(beta_std)):
        beta_std = np.linalg.pinv(lhs, rcond=1e-10) @ rhs
        beta_std = np.nan_to_num(beta_std, nan=0.0, posinf=0.0, neginf=0.0)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        y_eval = np.column_stack([np.ones(len(x_eval_z)), x_eval_z]) @ beta_std
    y_eval = np.nan_to_num(y_eval, nan=0.0, posinf=0.0, neginf=0.0)

    coef_raw = beta_std[1:] / sd
    intercept_raw = float(beta_std[0] - np.sum(beta_std[1:] * (mu / sd)))
    return coef_raw, intercept_raw, y_eval


def _evaluate_splits(
    *,
    y: np.ndarray,
    n_ctrl: np.ndarray,
    lambda_struct: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, float | int]] = []
    yhat_base_oof = np.full_like(y, np.nan, dtype=float)
    yhat_full_oof = np.full_like(y, np.nan, dtype=float)

    x_base = n_ctrl.reshape(-1, 1)
    x_full = np.column_stack([n_ctrl, lambda_struct])

    for split_id, (train_idx, test_idx) in enumerate(splits):
        y_train = y[train_idx]
        y_test = y[test_idx]

        xb_train, xb_test = x_base[train_idx], x_base[test_idx]
        xf_train, xf_test = x_full[train_idx], x_full[test_idx]

        coef_b, intercept_b, yhat_b = _fit_ridge_scaled(xb_train, y_train, xb_test, ridge_alpha)
        coef_f, intercept_f, yhat_f = _fit_ridge_scaled(xf_train, y_train, xf_test, ridge_alpha)

        yhat_base_oof[test_idx] = yhat_b
        yhat_full_oof[test_idx] = yhat_f

        mae_base = float(np.mean(np.abs(y_test - yhat_b)))
        mae_full = float(np.mean(np.abs(y_test - yhat_f)))
        gain = float((mae_base - mae_full) / (mae_base + 1e-12))

        rows.append(
            {
                "split_id": int(split_id),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "mae_base": mae_base,
                "mae_full": mae_full,
                "mae_gain_frac": gain,
                "coef_base_ctrl": float(coef_b[0]),
                "intercept_base": float(intercept_b),
                "coef_full_ctrl": float(coef_f[0]),
                "coef_full_lambda": float(coef_f[1]),
                "intercept_full": float(intercept_f),
            }
        )

    return pd.DataFrame(rows), yhat_base_oof, yhat_full_oof


def _block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    starts = list(range(0, n, block))
    blocks = [x[s : min(s + block, n)] for s in starts]
    rng.shuffle(blocks)
    return np.concatenate(blocks, axis=0)[:n]


def _permutation_test(
    *,
    y: np.ndarray,
    n_ctrl: np.ndarray,
    lambda_struct: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
    n_perm: int,
    perm_block: int,
    seed: int,
) -> tuple[float, pd.DataFrame, float]:
    rng = np.random.default_rng(seed)

    real_df, _, _ = _evaluate_splits(
        y=y,
        n_ctrl=n_ctrl,
        lambda_struct=lambda_struct,
        splits=splits,
        ridge_alpha=ridge_alpha,
    )
    stat_real = float(np.median(real_df["mae_gain_frac"].to_numpy(dtype=float)))

    rows = []
    count_ge = 0
    for pid in range(n_perm):
        lam_perm = _block_permute(lambda_struct, block=perm_block, rng=rng)
        perm_df, _, _ = _evaluate_splits(
            y=y,
            n_ctrl=n_ctrl,
            lambda_struct=lam_perm,
            splits=splits,
            ridge_alpha=ridge_alpha,
        )
        stat_perm = float(np.median(perm_df["mae_gain_frac"].to_numpy(dtype=float)))
        rows.append({"perm_id": int(pid), "stat_perm_median_gain": stat_perm})
        if stat_perm >= stat_real:
            count_ge += 1

    p_value = float((count_ge + 1) / (n_perm + 1))
    return p_value, pd.DataFrame(rows), stat_real


def _strata_table(
    *,
    y: np.ndarray,
    yhat_base: np.ndarray,
    yhat_full: np.ndarray,
    n_ctrl: np.ndarray,
    q: int,
) -> pd.DataFrame:
    if q < 2:
        raise ValueError("q for strata must be >=2")

    quant = np.quantile(n_ctrl, np.linspace(0.0, 1.0, q + 1))
    rows: list[dict[str, float | int]] = []

    for i in range(q):
        lo = float(quant[i])
        hi = float(quant[i + 1])
        if i < q - 1:
            idx = np.where((n_ctrl >= lo) & (n_ctrl < hi))[0]
        else:
            idx = np.where((n_ctrl >= lo) & (n_ctrl <= hi))[0]
        idx = idx[np.isfinite(yhat_base[idx]) & np.isfinite(yhat_full[idx])]
        if len(idx) == 0:
            continue

        y_sub = y[idx]
        mae_base = float(np.mean(np.abs(y_sub - yhat_base[idx])))
        mae_full = float(np.mean(np.abs(y_sub - yhat_full[idx])))
        gain = float((mae_base - mae_full) / (mae_base + 1e-12))

        rows.append(
            {
                "stratum_id": int(i),
                "ctrl_lo": lo,
                "ctrl_hi": hi,
                "n": int(len(idx)),
                "mae_base": mae_base,
                "mae_full": mae_full,
                "mae_gain_frac": gain,
            }
        )

    return pd.DataFrame(rows)


def run_experiment(
    *,
    input_path: Path,
    outdir: Path,
    scale_edges_km: list[float],
    n_modes_per_var: int,
    window: int,
    peak_quantile: float,
    ridge_alpha: float,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    seed: int,
    precip_factor: float,
    evap_factor: float,
    standardize_components: bool,
    residual_mode: str | None,
    cov_shrinkage: float,
    coherence_mode: str,
    strata_q: int,
    min_mae_gain: float,
    max_perm_p: float,
    min_sign_consistency: float,
    min_strata_gain: float,
    min_positive_strata_frac: float,
    time_stride: int,
    lat_stride: int,
    lon_stride: int,
    max_time: int | None,
    level_dim: str | None,
    level_index: int,
    var_overrides: dict[str, str | None],
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loaded = _load_data(
        input_path,
        var_overrides=var_overrides,
        level_dim=level_dim,
        level_index=level_index,
        time_stride=time_stride,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        max_time=max_time,
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

    nt, ny, nx = iwv.shape
    if verbose:
        print(f"[M] loaded data: nt={nt}, ny={ny}, nx={nx}")

    if not (0.0 <= min_positive_strata_frac <= 1.0):
        raise ValueError(f"min_positive_strata_frac must be in [0,1], got {min_positive_strata_frac}")

    if residual_mode is None:
        residual_mode = "component_zscore" if standardize_components else "physical_raw"

    time_s = _time_to_seconds(time)
    x_m, y_m = _xy_coordinates_m(lat, lon)

    res0, res_components = _compute_budget_residual_series(
        iwv=iwv,
        ivt_u=ivt_u,
        ivt_v=ivt_v,
        precip=precip,
        evap=evap,
        time_s=time_s,
        x_m=x_m,
        y_m=y_m,
        precip_factor=precip_factor,
        evap_factor=evap_factor,
        residual_mode=residual_mode,
    )

    vorticity = _compute_vorticity(u=u, v=v, x_m=x_m, y_m=y_m)

    density_source = "none"
    if "density" in f:
        n_density = np.mean(f["density"], axis=(1, 2))
        density_source = "density"
    elif "pressure" in f and "temp" in f:
        p = f["pressure"]
        t = f["temp"]
        n_density = np.mean(p / (K_BOLTZMANN * np.maximum(t, 1e-6)), axis=(1, 2))
        density_source = "pressure_over_kT"
    else:
        n_density = np.ones(nt, dtype=float)

    n_ctrl = _zscore(np.log(np.maximum(n_density, 1e-30)))

    dx_km = float(np.median(np.diff(x_m))) / 1000.0
    dy_km = float(np.median(np.diff(y_m))) / 1000.0

    masks, _, _, centers, mus, wavelength_km = _build_band_masks(
        ny=ny,
        nx=nx,
        dy_km=abs(dy_km),
        dx_km=abs(dx_km),
        scale_edges_km=scale_edges_km,
    )

    mode_fields = {
        "iwv": iwv,
        "ivt_u": ivt_u,
        "ivt_v": ivt_v,
        "p_minus_e": precip_factor * precip - evap_factor * evap,
        "vorticity": vorticity,
    }

    selected, mode_meta = _select_mode_indices(
        mode_fields=mode_fields,
        masks=masks,
        n_modes_per_var=n_modes_per_var,
        wavelength_km=wavelength_km,
    )
    coeff_by_band, n_struct = _build_coefficients_and_structures(
        mode_fields=mode_fields,
        vorticity=vorticity,
        masks=masks,
        selected=selected,
        peak_quantile=peak_quantile,
    )

    lambda_struct, w_mu, coh_mu, lambda_mu, trace_err_mu, min_eig_mu = _compute_rho_and_lambda(
        coeff_by_band=coeff_by_band,
        n_struct=n_struct,
        window=window,
        ridge=ridge_alpha,
        cov_shrinkage=cov_shrinkage,
        coherence_mode=coherence_mode,
    )

    valid_mask = np.isfinite(res0) & np.isfinite(lambda_struct) & np.isfinite(n_ctrl)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < max(40, n_folds * 8):
        raise ValueError(f"Not enough valid points after filtering: {len(valid_idx)}")

    y = res0[valid_idx]
    lam = lambda_struct[valid_idx]
    ctrl = n_ctrl[valid_idx]

    splits_rel = _blocked_splits(len(valid_idx), n_folds=n_folds)

    split_df, yhat_base_rel, yhat_full_rel = _evaluate_splits(
        y=y,
        n_ctrl=ctrl,
        lambda_struct=lam,
        splits=splits_rel,
        ridge_alpha=ridge_alpha,
    )

    p_perm, perm_df, stat_real = _permutation_test(
        y=y,
        n_ctrl=ctrl,
        lambda_struct=lam,
        splits=splits_rel,
        ridge_alpha=ridge_alpha,
        n_perm=n_perm,
        perm_block=perm_block,
        seed=seed + 117,
    )

    strata_df = _strata_table(
        y=y,
        yhat_base=yhat_base_rel,
        yhat_full=yhat_full_rel,
        n_ctrl=ctrl,
        q=strata_q,
    )

    full_coef, full_intercept, yhat_full_all_rel = _fit_ridge_scaled(
        np.column_stack([ctrl, lam]),
        y,
        np.column_stack([ctrl, lam]),
        ridge_alpha,
    )
    base_coef, base_intercept, yhat_base_all_rel = _fit_ridge_scaled(
        ctrl.reshape(-1, 1),
        y,
        ctrl.reshape(-1, 1),
        ridge_alpha,
    )

    # Expand OOF predictions back to full timeline.
    yhat_base_oof = np.full(nt, np.nan, dtype=float)
    yhat_full_oof = np.full(nt, np.nan, dtype=float)
    yhat_base_oof[valid_idx] = yhat_base_rel
    yhat_full_oof[valid_idx] = yhat_full_rel

    yhat_base_all = np.full(nt, np.nan, dtype=float)
    yhat_full_all = np.full(nt, np.nan, dtype=float)
    yhat_base_all[valid_idx] = yhat_base_all_rel
    yhat_full_all[valid_idx] = yhat_full_all_rel

    mae_base_oof = float(np.nanmean(np.abs(y - yhat_base_rel)))
    mae_full_oof = float(np.nanmean(np.abs(y - yhat_full_rel)))
    oof_gain = float((mae_base_oof - mae_full_oof) / (mae_base_oof + 1e-12))

    split_gains = split_df["mae_gain_frac"].to_numpy(dtype=float)
    coef_lambda = split_df["coef_full_lambda"].to_numpy(dtype=float)
    sign_consistency = float(max(np.mean(coef_lambda > 0.0), np.mean(coef_lambda < 0.0)))

    strata_min_gain = float(strata_df["mae_gain_frac"].min()) if len(strata_df) else float("nan")
    strata_positive_frac = (
        float(np.mean(strata_df["mae_gain_frac"].to_numpy(dtype=float) >= min_strata_gain)) if len(strata_df) else float("nan")
    )
    corr_lambda_density = float(np.corrcoef(lam, ctrl)[0, 1]) if np.std(lam) > 1e-12 and np.std(ctrl) > 1e-12 else float("nan")

    pass_mae = bool(oof_gain >= min_mae_gain)
    pass_perm = bool(p_perm <= max_perm_p)
    pass_sign = bool(sign_consistency >= min_sign_consistency)
    if np.isnan(strata_positive_frac):
        pass_strata = bool(min_strata_gain <= 0.0)
    else:
        pass_strata = bool(strata_positive_frac >= min_positive_strata_frac)
    pass_all = bool(pass_mae and pass_perm and pass_sign and pass_strata)

    summary_df = pd.DataFrame(
        [
            {
                "input_path": str(input_path),
                "n_time": int(nt),
                "n_valid": int(len(valid_idx)),
                "ny": int(ny),
                "nx": int(nx),
                "n_bands": int(len(masks)),
                "n_modes_per_var": int(n_modes_per_var),
                "window": int(window),
                "ridge_alpha": float(ridge_alpha),
                "residual_mode": str(residual_mode),
                "cov_shrinkage": float(cov_shrinkage),
                "coherence_mode": str(coherence_mode),
                "n_folds": int(n_folds),
                "n_perm": int(n_perm),
                "perm_block": int(perm_block),
                "density_source": density_source,
                "mae_base_oof": float(mae_base_oof),
                "mae_full_oof": float(mae_full_oof),
                "oof_gain_frac": float(oof_gain),
                "split_gain_median": float(np.median(split_gains)),
                "split_gain_min": float(np.min(split_gains)),
                "perm_stat_real_median_gain": float(stat_real),
                "perm_p_value": float(p_perm),
                "lambda_sign_consistency": float(sign_consistency),
                "strata_min_gain": float(strata_min_gain) if not np.isnan(strata_min_gain) else np.nan,
                "strata_positive_frac": float(strata_positive_frac) if not np.isnan(strata_positive_frac) else np.nan,
                "min_positive_strata_frac": float(min_positive_strata_frac),
                "corr_lambda_ctrl": corr_lambda_density,
                "coef_full_ctrl_global": float(full_coef[0]),
                "coef_full_lambda_global": float(full_coef[1]),
                "intercept_full_global": float(full_intercept),
                "coef_base_ctrl_global": float(base_coef[0]),
                "intercept_base_global": float(base_intercept),
                "pass_mae_gain": pass_mae,
                "pass_perm": pass_perm,
                "pass_sign_consistency": pass_sign,
                "pass_strata": pass_strata,
                "pass_all": pass_all,
            }
        ]
    )

    timeseries = pd.DataFrame(
        {
            "time_index": np.arange(nt, dtype=int),
            "time": pd.to_datetime(time).astype(str) if np.issubdtype(np.asarray(time).dtype, np.datetime64) else np.asarray(time),
            "residual_base_res0": res0,
            "lambda_struct": lambda_struct,
            "n_density_ctrl_z": n_ctrl,
            "d_iwv_dt_mean": res_components["d_iwv_dt_mean"],
            "div_ivt_mean": res_components["div_ivt_mean"],
            "p_minus_e_mean": res_components["p_minus_e_mean"],
            "residual_physical_mean": res_components["residual_physical_mean"],
            "yhat_base_oof": yhat_base_oof,
            "yhat_full_oof": yhat_full_oof,
            "resid_base_oof": res0 - yhat_base_oof,
            "resid_full_oof": res0 - yhat_full_oof,
            "yhat_base_global": yhat_base_all,
            "yhat_full_global": yhat_full_all,
            "resid_base_global": res0 - yhat_base_all,
            "resid_full_global": res0 - yhat_full_all,
        }
    )

    for b in range(len(masks)):
        bid = f"{b:02d}"
        timeseries[f"mu_{bid}"] = float(mus[b])
        timeseries[f"scale_center_km_{bid}"] = float(centers[b])
        timeseries[f"weight_mu_{bid}"] = w_mu[:, b]
        timeseries[f"coh_mu_{bid}"] = coh_mu[:, b]
        timeseries[f"lambda_mu_{bid}"] = lambda_mu[:, b]
        timeseries[f"n_struct_mu_{bid}"] = n_struct[:, b]
        timeseries[f"rho_trace_err_mu_{bid}"] = trace_err_mu[:, b]
        timeseries[f"rho_min_eig_mu_{bid}"] = min_eig_mu[:, b]

    coeff_df = pd.DataFrame(
        [
            {"term": "intercept_base", "value": float(base_intercept)},
            {"term": "ctrl_base", "value": float(base_coef[0])},
            {"term": "intercept_full", "value": float(full_intercept)},
            {"term": "ctrl_full", "value": float(full_coef[0])},
            {"term": "lambda_full", "value": float(full_coef[1])},
        ]
    )

    mode_meta = mode_meta.copy()
    for b in range(len(masks)):
        mode_meta.loc[mode_meta["band_id"] == b, "scale_center_km"] = float(centers[b])
        mode_meta.loc[mode_meta["band_id"] == b, "mu"] = float(mus[b])

    outdir.mkdir(parents=True, exist_ok=True)
    timeseries.to_csv(outdir / "experiment_M_timeseries.csv", index=False)
    split_df.to_csv(outdir / "experiment_M_splits.csv", index=False)
    perm_df.to_csv(outdir / "experiment_M_permutation.csv", index=False)
    strata_df.to_csv(outdir / "experiment_M_strata.csv", index=False)
    coeff_df.to_csv(outdir / "experiment_M_coefficients.csv", index=False)
    summary_df.to_csv(outdir / "experiment_M_summary.csv", index=False)
    mode_meta.to_csv(outdir / "experiment_M_mode_index.csv", index=False)

    return timeseries, split_df, perm_df, strata_df, coeff_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input .npz or NetCDF/Zarr dataset")
    parser.add_argument("--out", default="clean_experiments/results/experiment_M_cosmo_flow", help="Output directory")

    parser.add_argument("--scale-edges-km", default="25,50,100,200,400,800,1600")
    parser.add_argument("--n-modes-per-var", type=int, default=6)
    parser.add_argument("--window", type=int, default=24, help="Rolling window length in time steps")
    parser.add_argument("--peak-quantile", type=float, default=0.90)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)

    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--n-perm", type=int, default=140)
    parser.add_argument("--perm-block", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260301)

    parser.add_argument("--precip-factor", type=float, default=1.0)
    parser.add_argument("--evap-factor", type=float, default=1.0)
    parser.add_argument("--no-standardize-components", action="store_true", help="Use raw residual components")
    parser.add_argument(
        "--residual-mode",
        choices=["component_zscore", "physical_zscore", "physical_raw"],
        default=None,
        help=(
            "Residual construction mode. "
            "If omitted, legacy behavior is used: component_zscore unless --no-standardize-components."
        ),
    )

    parser.add_argument("--strata-q", type=int, default=3)
    parser.add_argument("--min-mae-gain", type=float, default=0.03)
    parser.add_argument("--max-perm-p", type=float, default=0.05)
    parser.add_argument("--min-sign-consistency", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-strata-gain", type=float, default=0.0)
    parser.add_argument(
        "--min-positive-strata-frac",
        type=float,
        default=1.0,
        help="Minimum fraction of strata with mae_gain_frac >= min-strata-gain.",
    )

    parser.add_argument("--cov-shrinkage", type=float, default=0.0, help="Covariance shrinkage alpha in [0,1].")
    parser.add_argument(
        "--coherence-mode",
        choices=["offdiag_fro", "relative_offdiag_fro"],
        default="offdiag_fro",
        help="How to compute per-band coherence from rho_mu(t).",
    )

    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--lat-stride", type=int, default=1)
    parser.add_argument("--lon-stride", type=int, default=1)
    parser.add_argument("--max-time", type=int, default=None)

    parser.add_argument("--level-dim", default=None)
    parser.add_argument("--level-index", type=int, default=0)

    parser.add_argument("--iwv-var", default=None)
    parser.add_argument("--ivt-u-var", default=None)
    parser.add_argument("--ivt-v-var", default=None)
    parser.add_argument("--precip-var", default=None)
    parser.add_argument("--evap-var", default=None)
    parser.add_argument("--u-var", default=None)
    parser.add_argument("--v-var", default=None)
    parser.add_argument("--temp-var", default=None)
    parser.add_argument("--pressure-var", default=None)
    parser.add_argument("--density-var", default=None)

    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    var_overrides = {
        "iwv": args.iwv_var,
        "ivt_u": args.ivt_u_var,
        "ivt_v": args.ivt_v_var,
        "precip": args.precip_var,
        "evap": args.evap_var,
        "u": args.u_var,
        "v": args.v_var,
        "temp": args.temp_var,
        "pressure": args.pressure_var,
        "density": args.density_var,
    }

    _, _, _, _, _, summary = run_experiment(
        input_path=Path(args.input),
        outdir=Path(args.out),
        scale_edges_km=_parse_float_list(args.scale_edges_km),
        n_modes_per_var=args.n_modes_per_var,
        window=args.window,
        peak_quantile=args.peak_quantile,
        ridge_alpha=args.ridge_alpha,
        n_folds=args.folds,
        n_perm=args.n_perm,
        perm_block=args.perm_block,
        seed=args.seed,
        precip_factor=args.precip_factor,
        evap_factor=args.evap_factor,
        standardize_components=not args.no_standardize_components,
        residual_mode=args.residual_mode,
        cov_shrinkage=args.cov_shrinkage,
        coherence_mode=args.coherence_mode,
        strata_q=args.strata_q,
        min_mae_gain=args.min_mae_gain,
        max_perm_p=args.max_perm_p,
        min_sign_consistency=args.min_sign_consistency,
        min_strata_gain=args.min_strata_gain,
        min_positive_strata_frac=args.min_positive_strata_frac,
        time_stride=args.time_stride,
        lat_stride=args.lat_stride,
        lon_stride=args.lon_stride,
        max_time=args.max_time,
        level_dim=args.level_dim,
        level_index=args.level_index,
        var_overrides=var_overrides,
        verbose=not args.quiet,
    )

    print("Summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
