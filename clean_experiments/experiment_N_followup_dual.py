#!/usr/bin/env python3
"""Experiment N follow-up: source-augmented and localized-lambda variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from clean_experiments.experiment_M_cosmo_flow import _blocked_splits, _xy_coordinates_m
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _blocked_splits, _xy_coordinates_m  # type: ignore


EPS = 1e-12
LAMBDA_RAW_KEYS = ("lambda_struct_raw", "lambda_low_raw", "lambda_mid_raw", "lambda_high_raw")
COMPONENTS = ("c1", "c2", "s_pe")


@dataclass(frozen=True)
class Term:
    name: str
    component: str
    factor: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    base_terms: tuple[Term, ...]
    full_terms: tuple[Term, ...]

    @property
    def complexity(self) -> int:
        return int(max(0, len(self.full_terms) - len(self.base_terms)))


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    description: str
    models: tuple[ModelSpec, ...]


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


def _move_to_tyx(
    arr: np.ndarray,
    dims: tuple[str, ...],
    time_name: str,
    lat_name: str,
    lon_name: str,
) -> np.ndarray:
    t_axis = dims.index(time_name)
    lat_axis = dims.index(lat_name)
    lon_axis = dims.index(lon_name)
    out = np.moveaxis(arr, (t_axis, lat_axis, lon_axis), (0, 1, 2))
    out = np.squeeze(out)
    if out.ndim != 3:
        raise ValueError(f"Expected 3D array (time,lat,lon), got shape={out.shape} for dims={dims}")
    return out


def _move_to_tlxy(
    arr: np.ndarray,
    dims: tuple[str, ...],
    time_name: str,
    level_name: str,
    lat_name: str,
    lon_name: str,
) -> np.ndarray:
    t_axis = dims.index(time_name)
    l_axis = dims.index(level_name)
    lat_axis = dims.index(lat_name)
    lon_axis = dims.index(lon_name)
    out = np.moveaxis(arr, (t_axis, l_axis, lat_axis, lon_axis), (0, 1, 2, 3))
    out = np.squeeze(out)
    if out.ndim != 4:
        raise ValueError(f"Expected 4D array (time,level,lat,lon), got shape={out.shape} for dims={dims}")
    return out


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


def _time_to_seconds(time_ns: np.ndarray) -> np.ndarray:
    t = np.asarray(time_ns)
    if not np.issubdtype(t.dtype, np.datetime64):
        raise ValueError("time array must be datetime64.")
    sec = (t - t[0]).astype("timedelta64[s]").astype(float)
    if len(sec) < 3:
        raise ValueError("Need at least 3 time steps.")
    return sec


def _edge_order(size: int) -> int:
    return 2 if size >= 3 else 1


def _mean_or_nan(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def _component_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, ci in enumerate(COMPONENTS):
        for j, cj in enumerate(COMPONENTS):
            if j < i:
                continue
            pairs.append((ci, cj))
    return pairs


def _compute_enhanced_moments(
    *,
    input_nc: Path,
    out_path: Path,
    batch_size: int,
    lat_stride: int,
    lon_stride: int,
    p_min_hpa: float,
    p_max_hpa: float,
    h_scale_hpa: float,
) -> pd.DataFrame:
    pairs = _component_pairs()
    with Dataset(input_nc, mode="r") as ds:
        time_name = _find_coord_name(ds, ("valid_time", "time", "datetime", "date"), "time")
        lat_name = _find_coord_name(ds, ("latitude", "lat", "y", "rlat"), "latitude")
        lon_name = _find_coord_name(ds, ("longitude", "lon", "x", "rlon"), "longitude")
        level_name = _find_coord_name(ds, ("pressure_level", "level", "plev", "isobaricInhPa"), "pressure level")

        q_name = _find_var_name(ds, ("q_pl", "specific_humidity", "q"), "q_pl")
        u_pl_name = _find_var_name(ds, ("u_pl",), "u_pl")
        v_pl_name = _find_var_name(ds, ("v_pl",), "v_pl")
        w_pl_name = _find_var_name(ds, ("w_pl", "w", "omega", "vertical_velocity"), "w_pl")
        u2_name = _find_var_name(ds, ("u",), "u")
        v2_name = _find_var_name(ds, ("v",), "v")
        precip_name = _find_var_name(ds, ("precip", "tp", "total_precipitation"), "precip")
        evap_name = _find_var_name(ds, ("evap", "e", "evaporation"), "evap")

        time_ns = _to_datetime64(ds.variables[time_name])
        time_s = _time_to_seconds(time_ns)

        lat = np.asarray(ds.variables[lat_name][:], dtype=float)[::lat_stride]
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)[::lon_stride]
        x_m, y_m = _xy_coordinates_m(lat=lat, lon=lon)

        levels_raw = np.asarray(ds.variables[level_name][:], dtype=float)
        levels_hpa = levels_raw / 100.0 if np.nanmax(np.abs(levels_raw)) > 3000 else levels_raw
        level_mask = (levels_hpa >= p_min_hpa) & (levels_hpa <= p_max_hpa)
        if int(level_mask.sum()) < 3:
            raise ValueError(
                f"Need at least 3 pressure levels in [{p_min_hpa}, {p_max_hpa}] hPa, got {int(level_mask.sum())}."
            )
        p_sel_pa = levels_hpa[level_mask] * 100.0
        p_norm = p_sel_pa / 100000.0
        h_scale_pa = h_scale_hpa * 100.0

        q_var = ds.variables[q_name]
        u_pl_var = ds.variables[u_pl_name]
        v_pl_var = ds.variables[v_pl_name]
        w_pl_var = ds.variables[w_pl_name]
        u2_var = ds.variables[u2_name]
        v2_var = ds.variables[v2_name]
        precip_var = ds.variables[precip_name]
        evap_var = ds.variables[evap_name]

        q_dims = tuple(q_var.dimensions)
        u_pl_dims = tuple(u_pl_var.dimensions)
        v_pl_dims = tuple(v_pl_var.dimensions)
        w_pl_dims = tuple(w_pl_var.dimensions)
        u2_dims = tuple(u2_var.dimensions)
        v2_dims = tuple(v2_var.dimensions)
        precip_dims = tuple(precip_var.dimensions)
        evap_dims = tuple(evap_var.dimensions)

        nt = int(q_var.shape[q_dims.index(time_name)])
        eo_t = _edge_order(len(time_s))
        eo_x = _edge_order(len(x_m))
        eo_y = _edge_order(len(y_m))
        eo_p = _edge_order(len(p_sel_pa))

        n_points = np.zeros(nt, dtype=float)
        m_rr = np.full(nt, np.nan, dtype=float)
        m_r = {c: np.full(nt, np.nan, dtype=float) for c in COMPONENTS}
        m_xx = {p: np.full(nt, np.nan, dtype=float) for p in pairs}
        proxy_vort = np.full(nt, np.nan, dtype=float)
        proxy_omegaq = np.full(nt, np.nan, dtype=float)
        pe_mean = np.full(nt, np.nan, dtype=float)

        for start in range(0, nt, batch_size):
            stop = min(start + batch_size, nt)
            t0 = max(0, start - 1)
            t1 = min(nt, stop + 1)
            c0 = start - t0
            c1 = c0 + (stop - start)

            q_blk = _move_to_tlxy(
                np.asarray(q_var[t0:t1, ...], dtype=float),
                q_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, level_mask, ::lat_stride, ::lon_stride]
            u_pl_blk = _move_to_tlxy(
                np.asarray(u_pl_var[t0:t1, ...], dtype=float),
                u_pl_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, level_mask, ::lat_stride, ::lon_stride]
            v_pl_blk = _move_to_tlxy(
                np.asarray(v_pl_var[t0:t1, ...], dtype=float),
                v_pl_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, level_mask, ::lat_stride, ::lon_stride]
            w_pl_blk = _move_to_tlxy(
                np.asarray(w_pl_var[t0:t1, ...], dtype=float),
                w_pl_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, level_mask, ::lat_stride, ::lon_stride]
            u2_blk = _move_to_tyx(
                np.asarray(u2_var[t0:t1, ...], dtype=float),
                u2_dims,
                time_name=time_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, ::lat_stride, ::lon_stride]
            v2_blk = _move_to_tyx(
                np.asarray(v2_var[t0:t1, ...], dtype=float),
                v2_dims,
                time_name=time_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, ::lat_stride, ::lon_stride]
            precip_blk = _move_to_tyx(
                np.asarray(precip_var[t0:t1, ...], dtype=float),
                precip_dims,
                time_name=time_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, ::lat_stride, ::lon_stride]
            evap_blk = _move_to_tyx(
                np.asarray(evap_var[t0:t1, ...], dtype=float),
                evap_dims,
                time_name=time_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, ::lat_stride, ::lon_stride]

            time_blk_s = time_s[t0:t1]
            dq_dt_blk = np.gradient(q_blk, time_blk_s, axis=0, edge_order=min(eo_t, _edge_order(len(time_blk_s))))

            q = q_blk[c0:c1]
            u_pl = u_pl_blk[c0:c1]
            v_pl = v_pl_blk[c0:c1]
            w_pl = w_pl_blk[c0:c1]
            dq_dt = dq_dt_blk[c0:c1]
            u2 = u2_blk[c0:c1]
            v2 = v2_blk[c0:c1]
            precip = precip_blk[c0:c1]
            evap = evap_blk[c0:c1]

            dq_dx = np.gradient(q, x_m, axis=3, edge_order=eo_x)
            dq_dy = np.gradient(q, y_m, axis=2, edge_order=eo_y)
            dq_dp = np.gradient(q, p_sel_pa, axis=1, edge_order=eo_p)
            domega_dp = np.gradient(w_pl, p_sel_pa, axis=1, edge_order=eo_p)

            r = dq_dt + u_pl * dq_dx + v_pl * dq_dy + w_pl * dq_dp
            c1_field = -(p_norm[None, :, None, None] * w_pl * dq_dp)
            c2_field = -(p_norm[None, :, None, None] * h_scale_pa * domega_dp * dq_dp)
            pe2d = precip - evap
            s_pe = np.broadcast_to(pe2d[:, None, :, :], q.shape)

            for i in range(stop - start):
                gi = start + i
                fields = {
                    "c1": c1_field[i],
                    "c2": c2_field[i],
                    "s_pe": s_pe[i],
                }
                valid = np.isfinite(r[i])
                for comp in COMPONENTS:
                    valid &= np.isfinite(fields[comp])
                if not np.any(valid):
                    continue
                rv = r[i][valid]
                n_points[gi] = float(rv.size)
                m_rr[gi] = _mean_or_nan(rv * rv)
                for comp in COMPONENTS:
                    xv = fields[comp][valid]
                    m_r[comp][gi] = _mean_or_nan(rv * xv)
                for ci, cj in pairs:
                    x1 = fields[ci][valid]
                    x2 = fields[cj][valid]
                    m_xx[(ci, cj)][gi] = _mean_or_nan(x1 * x2)

            dv_dx = np.gradient(v2, x_m, axis=2, edge_order=eo_x)
            du_dy = np.gradient(u2, y_m, axis=1, edge_order=eo_y)
            zeta = dv_dx - du_dy
            proxy_vort[start:stop] = np.nanmean(np.abs(zeta), axis=(1, 2))
            proxy_omegaq[start:stop] = np.nanmean(-(w_pl * q), axis=(1, 2, 3))
            pe_mean[start:stop] = np.nanmean(pe2d, axis=(1, 2))

            print(
                f"[N_followup] moments progress: {stop}/{nt} "
                f"(levels={int(level_mask.sum())}, grid={len(lat)}x{len(lon)})",
                flush=True,
            )

    out = pd.DataFrame(
        {
            "time_index": np.arange(nt, dtype=int),
            "time": pd.to_datetime(time_ns).astype(str),
            "n_points": n_points,
            "m_rr": m_rr,
            "proxy_vorticity_abs_mean": proxy_vort,
            "proxy_omega_q_850_300": proxy_omegaq,
            "p_minus_e_mean": pe_mean,
        }
    )
    for comp in COMPONENTS:
        out[f"m_r_{comp}"] = m_r[comp]
    for ci, cj in pairs:
        out[f"m_{ci}_{cj}"] = m_xx[(ci, cj)]

    out["time_dt"] = pd.to_datetime(out["time"])
    out["year"] = out["time_dt"].dt.year
    out["quarter"] = out["time_dt"].dt.quarter

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def _load_or_build_moments(
    *,
    moments_csv: Path,
    input_nc: Path,
    batch_size: int,
    lat_stride: int,
    lon_stride: int,
    p_min_hpa: float,
    p_max_hpa: float,
    h_scale_hpa: float,
    force_recompute: bool,
) -> pd.DataFrame:
    if moments_csv.exists() and not force_recompute:
        return pd.read_csv(moments_csv)
    return _compute_enhanced_moments(
        input_nc=input_nc,
        out_path=moments_csv,
        batch_size=batch_size,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        p_min_hpa=p_min_hpa,
        p_max_hpa=p_max_hpa,
        h_scale_hpa=h_scale_hpa,
    )


def _load_lambda_table(lambda_timeseries_csv: Path) -> pd.DataFrame:
    need_cols = ["time_index", "lambda_struct"] + [f"lambda_mu_{i:02d}" for i in range(6)]
    df = pd.read_csv(lambda_timeseries_csv)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing lambda columns in {lambda_timeseries_csv}: {missing}")
    out = df[need_cols].copy()
    out = out.rename(columns={"lambda_struct": "lambda_struct_raw"})
    out["lambda_low_raw"] = out[["lambda_mu_00", "lambda_mu_01"]].mean(axis=1)
    out["lambda_mid_raw"] = out[["lambda_mu_02", "lambda_mu_03"]].mean(axis=1)
    out["lambda_high_raw"] = out[["lambda_mu_04", "lambda_mu_05"]].mean(axis=1)
    return out[["time_index", *LAMBDA_RAW_KEYS]]


def _prepare_cache(df: pd.DataFrame) -> dict[str, object]:
    pairs = _component_pairs()
    cache: dict[str, object] = {
        "n_points": df["n_points"].to_numpy(dtype=float),
        "m_rr": df["m_rr"].to_numpy(dtype=float),
        "m_r": {c: df[f"m_r_{c}"].to_numpy(dtype=float) for c in COMPONENTS},
        "m_xx": {(ci, cj): df[f"m_{ci}_{cj}"].to_numpy(dtype=float) for ci, cj in pairs},
    }
    return cache


def _cov_key(ci: str, cj: str) -> tuple[str, str]:
    if COMPONENTS.index(ci) <= COMPONENTS.index(cj):
        return ci, cj
    return cj, ci


def _compute_extreme_mask(df: pd.DataFrame, train_idx: np.ndarray, q_extreme: float) -> tuple[np.ndarray, float, float]:
    vort = df["proxy_vorticity_abs_mean"].to_numpy(dtype=float)
    omegaq = df["proxy_omega_q_850_300"].to_numpy(dtype=float)
    v_thr = float(np.quantile(vort[train_idx], q_extreme))
    o_thr = float(np.quantile(omegaq[train_idx], q_extreme))
    extreme = ((vort >= v_thr) | (omegaq >= o_thr)).astype(float)
    return extreme, v_thr, o_thr


def _zscore_with_train(arr: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.nanmean(arr[train_idx]))
    sd = float(np.nanstd(arr[train_idx]))
    if sd < 1e-15:
        sd = 1.0
    return (arr - mu) / sd, mu, sd


def _build_factors(
    *,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    q_extreme: float,
    lambda_override: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    n = len(df)
    factors: dict[str, np.ndarray] = {"one": np.ones(n, dtype=float)}
    extreme, v_thr, o_thr = _compute_extreme_mask(df, train_idx, q_extreme=q_extreme)
    factors["extreme"] = extreme
    meta: dict[str, float] = {
        "v_thr_train": float(v_thr),
        "omegaq_thr_train": float(o_thr),
    }

    raw_cols = {}
    for key in LAMBDA_RAW_KEYS:
        if lambda_override is not None and key in lambda_override:
            raw_cols[key] = np.asarray(lambda_override[key], dtype=float)
        else:
            raw_cols[key] = df[key].to_numpy(dtype=float)

    for key, z_name in (
        ("lambda_struct_raw", "lambda_struct_z"),
        ("lambda_low_raw", "lambda_low_z"),
        ("lambda_mid_raw", "lambda_mid_z"),
        ("lambda_high_raw", "lambda_high_z"),
    ):
        z, mu, sd = _zscore_with_train(raw_cols[key], train_idx)
        factors[z_name] = z
        factors[f"{z_name}_ext"] = z * extreme
        meta[f"{key}_mu_train"] = float(mu)
        meta[f"{key}_sd_train"] = float(sd)

    return factors, meta


def _build_design(terms: tuple[Term, ...], factors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], list[str]]:
    n = len(next(iter(factors.values())))
    if len(terms) == 0:
        return np.zeros((n, 0), dtype=float), [], []
    f = np.zeros((n, len(terms)), dtype=float)
    comps: list[str] = []
    names: list[str] = []
    for j, t in enumerate(terms):
        if t.factor not in factors:
            raise KeyError(f"Unknown factor '{t.factor}' for term '{t.name}'")
        f[:, j] = factors[t.factor]
        comps.append(t.component)
        names.append(t.name)
    return f, comps, names


def _fit_theta(
    *,
    cache: dict[str, object],
    idx: np.ndarray,
    factors: np.ndarray,
    comps: list[str],
    ridge_alpha: float,
) -> np.ndarray:
    k = factors.shape[1]
    if k == 0:
        return np.zeros(0, dtype=float)

    n_w = np.asarray(cache["n_points"], dtype=float)[idx]
    m_r = cache["m_r"]  # type: ignore[assignment]
    m_xx = cache["m_xx"]  # type: ignore[assignment]

    a = np.zeros((k, k), dtype=float)
    b = np.zeros(k, dtype=float)
    for i in range(k):
        fi = factors[idx, i]
        mr_i = np.asarray(m_r[comps[i]], dtype=float)[idx]
        b[i] = -float(np.sum(n_w * fi * mr_i))
        for j in range(i, k):
            fj = factors[idx, j]
            key = _cov_key(comps[i], comps[j])
            mc = np.asarray(m_xx[key], dtype=float)[idx]
            a_ij = float(np.sum(n_w * fi * fj * mc))
            a[i, j] = a_ij
            a[j, i] = a_ij
    lhs = a + ridge_alpha * np.eye(k, dtype=float)
    try:
        theta = np.linalg.solve(lhs, b)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(lhs, rcond=1e-10) @ b
    return np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)


def _mean_square_series(
    *,
    cache: dict[str, object],
    theta: np.ndarray,
    factors: np.ndarray,
    comps: list[str],
) -> np.ndarray:
    ms = np.asarray(cache["m_rr"], dtype=float).copy()
    if len(theta) == 0:
        return ms
    m_r = cache["m_r"]  # type: ignore[assignment]
    m_xx = cache["m_xx"]  # type: ignore[assignment]
    n = len(ms)
    all_idx = np.arange(n, dtype=int)
    for i in range(len(theta)):
        mr = np.asarray(m_r[comps[i]], dtype=float)[all_idx]
        ms += 2.0 * theta[i] * factors[:, i] * mr
    for i in range(len(theta)):
        for j in range(i, len(theta)):
            key = _cov_key(comps[i], comps[j])
            mc = np.asarray(m_xx[key], dtype=float)[all_idx]
            term = theta[i] * theta[j] * factors[:, i] * factors[:, j] * mc
            if i == j:
                ms += term
            else:
                ms += 2.0 * term
    return np.clip(ms, 0.0, None)


def _weighted_rms(ms: np.ndarray, n_w: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) == 0:
        return np.nan
    ww = n_w[idx]
    vv = ms[idx]
    valid = np.isfinite(ww) & np.isfinite(vv) & (ww > 0)
    if int(np.sum(valid)) == 0:
        return np.nan
    return float(np.sqrt(np.sum(ww[valid] * vv[valid]) / (np.sum(ww[valid]) + EPS)))


def _gain_metrics(
    *,
    base_ms: np.ndarray,
    full_ms: np.ndarray,
    n_w: np.ndarray,
    eval_idx: np.ndarray,
    extreme_mask: np.ndarray,
) -> dict[str, float]:
    rms_base = _weighted_rms(base_ms, n_w, eval_idx)
    rms_full = _weighted_rms(full_ms, n_w, eval_idx)
    gain_all = float((rms_base - rms_full) / (rms_base + EPS))
    idx_ext = eval_idx[extreme_mask[eval_idx] > 0.5]
    idx_non = eval_idx[extreme_mask[eval_idx] <= 0.5]
    rms_base_ext = _weighted_rms(base_ms, n_w, idx_ext)
    rms_full_ext = _weighted_rms(full_ms, n_w, idx_ext)
    rms_base_non = _weighted_rms(base_ms, n_w, idx_non)
    rms_full_non = _weighted_rms(full_ms, n_w, idx_non)
    gain_ext = float((rms_base_ext - rms_full_ext) / (rms_base_ext + EPS)) if np.isfinite(rms_base_ext) else np.nan
    gain_non = float((rms_base_non - rms_full_non) / (rms_base_non + EPS)) if np.isfinite(rms_base_non) else np.nan
    return {
        "rms_base": float(rms_base),
        "rms_full": float(rms_full),
        "gain_all": float(gain_all),
        "gain_extreme": float(gain_ext),
        "gain_non_extreme": float(gain_non),
        "n_eval": int(len(eval_idx)),
        "n_eval_extreme": int(len(idx_ext)),
        "n_eval_non_extreme": int(len(idx_non)),
    }


def _evaluate_single(
    *,
    df: pd.DataFrame,
    cache: dict[str, object],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    model: ModelSpec,
    q_extreme: float,
    ridge_alpha: float,
    lambda_override: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    factors, meta = _build_factors(
        df=df,
        train_idx=train_idx,
        q_extreme=q_extreme,
        lambda_override=lambda_override,
    )
    f_base, c_base, n_base = _build_design(model.base_terms, factors)
    f_full, c_full, n_full = _build_design(model.full_terms, factors)
    th_base = _fit_theta(cache=cache, idx=train_idx, factors=f_base, comps=c_base, ridge_alpha=ridge_alpha)
    th_full = _fit_theta(cache=cache, idx=train_idx, factors=f_full, comps=c_full, ridge_alpha=ridge_alpha)
    ms_base = _mean_square_series(cache=cache, theta=th_base, factors=f_base, comps=c_base)
    ms_full = _mean_square_series(cache=cache, theta=th_full, factors=f_full, comps=c_full)
    n_w = np.asarray(cache["n_points"], dtype=float)
    metrics = _gain_metrics(
        base_ms=ms_base,
        full_ms=ms_full,
        n_w=n_w,
        eval_idx=eval_idx,
        extreme_mask=factors["extreme"],
    )
    return {
        "metrics": metrics,
        "meta": meta,
        "theta_base": th_base,
        "theta_full": th_full,
        "term_names_base": n_base,
        "term_names_full": n_full,
        "ms_base": ms_base,
        "ms_full": ms_full,
        "factors": factors,
    }


def _score(gain_all: float, gain_extreme: float, extreme_weight: float) -> float:
    ge = gain_extreme if np.isfinite(gain_extreme) else 0.0
    return float(gain_all + extreme_weight * ge)


def _evaluate_cv_config(
    *,
    df: pd.DataFrame,
    cache: dict[str, object],
    train_idx_full: np.ndarray,
    n_folds: int,
    model: ModelSpec,
    q_extreme: float,
    ridge_alpha: float,
    extreme_weight: float,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    rel_splits = _blocked_splits(len(train_idx_full), n_folds=n_folds)
    rows: list[dict[str, float | str | int]] = []
    for fold_id, (tr_rel, va_rel) in enumerate(rel_splits):
        tr_idx = train_idx_full[tr_rel]
        va_idx = train_idx_full[va_rel]
        out = _evaluate_single(
            df=df,
            cache=cache,
            train_idx=tr_idx,
            eval_idx=va_idx,
            model=model,
            q_extreme=q_extreme,
            ridge_alpha=ridge_alpha,
            lambda_override=None,
        )
        met = out["metrics"]  # type: ignore[assignment]
        meta = out["meta"]  # type: ignore[assignment]
        g_all = float(met["gain_all"])
        g_ext = float(met["gain_extreme"])
        rows.append(
            {
                "fold_id": int(fold_id),
                "model": model.name,
                "description": model.description,
                "complexity": int(model.complexity),
                "q_extreme": float(q_extreme),
                "ridge_alpha": float(ridge_alpha),
                "gain_all": g_all,
                "gain_extreme": g_ext,
                "gain_non_extreme": float(met["gain_non_extreme"]),
                "score": _score(g_all, g_ext, extreme_weight=extreme_weight),
                "v_thr_train": float(meta["v_thr_train"]),
                "omegaq_thr_train": float(meta["omegaq_thr_train"]),
            }
        )
    fold_df = pd.DataFrame(rows)
    stat: dict[str, float | str] = {
        "model": model.name,
        "description": model.description,
        "complexity": int(model.complexity),
        "q_extreme": float(q_extreme),
        "ridge_alpha": float(ridge_alpha),
        "cv_gain_all_mean": float(np.nanmean(fold_df["gain_all"])),
        "cv_gain_all_median": float(np.nanmedian(fold_df["gain_all"])),
        "cv_gain_extreme_mean": float(np.nanmean(fold_df["gain_extreme"])),
        "cv_gain_extreme_median": float(np.nanmedian(fold_df["gain_extreme"])),
        "cv_gain_non_extreme_median": float(np.nanmedian(fold_df["gain_non_extreme"])),
        "cv_score_mean": float(np.nanmean(fold_df["score"])),
        "cv_score_std": float(np.nanstd(fold_df["score"], ddof=1)),
        "cv_positive_gain_frac": float(np.nanmean(fold_df["gain_all"] > 0.0)),
        "cv_extreme_positive_frac": float(np.nanmean(fold_df["gain_extreme"] > 0.0)),
    }
    return fold_df, stat


def _choose_config_one_se(df_stats: pd.DataFrame, n_folds: int) -> pd.Series:
    if df_stats.empty:
        raise ValueError("No CV stats to choose from.")
    best_idx = int(df_stats["cv_score_mean"].idxmax())
    best = df_stats.loc[best_idx]
    se = float(best["cv_score_std"] / np.sqrt(max(float(n_folds), 1.0)))
    threshold = float(best["cv_score_mean"] - se)
    eligible = df_stats[df_stats["cv_score_mean"] >= threshold].copy()
    if eligible.empty:
        return best
    eligible = eligible.sort_values(
        by=["complexity", "ridge_alpha", "cv_score_mean", "cv_gain_all_median"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return eligible.iloc[0]


def _block_permute_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = list(range(0, n, block))
    blocks = [np.arange(s, min(s + block, n), dtype=int) for s in starts]
    rng.shuffle(blocks)
    return np.concatenate(blocks, axis=0)[:n]


def _bootstrap_rel_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = np.arange(0, n, block, dtype=int)
    parts: list[np.ndarray] = []
    cur = 0
    while cur < n:
        s = int(rng.choice(starts))
        e = min(s + block, n)
        p = np.arange(s, e, dtype=int)
        parts.append(p)
        cur += len(p)
    return np.concatenate(parts)[:n]


def _plot_hist(null_vals: np.ndarray, real_val: float, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.hist(null_vals, bins=24, color="#2ca02c", alpha=0.75, edgecolor="white")
    ax.axvline(real_val, color="#d62728", lw=2.0, label=f"real={real_val:.4f}")
    ax.set_xlabel("Gain")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_quarterly(df_q: pd.DataFrame, out_path: Path, title: str) -> None:
    if df_q.empty:
        return
    q = df_q["quarter"].to_numpy(dtype=int)
    g = df_q["gain_all"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(q, g, color="#1f77b4")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("2019 Quarter")
    ax.set_ylabel("Gain (RMS frac)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _run_quarterly(
    *,
    df: pd.DataFrame,
    cache: dict[str, object],
    model: ModelSpec,
    q_extreme: float,
    ridge_alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    time_dt = pd.to_datetime(df["time"])
    for quarter in (1, 2, 3, 4):
        test_mask = (df["year"] == 2019) & (df["quarter"] == quarter)
        if int(np.sum(test_mask)) == 0:
            continue
        q_start = pd.Timestamp(year=2019, month=3 * (quarter - 1) + 1, day=1)
        train_mask = time_dt < q_start
        train_idx = np.where(train_mask.to_numpy())[0]
        test_idx = np.where(test_mask.to_numpy())[0]
        if len(train_idx) < 500 or len(test_idx) < 40:
            continue
        out = _evaluate_single(
            df=df,
            cache=cache,
            train_idx=train_idx,
            eval_idx=test_idx,
            model=model,
            q_extreme=q_extreme,
            ridge_alpha=ridge_alpha,
            lambda_override=None,
        )
        met = out["metrics"]  # type: ignore[assignment]
        meta = out["meta"]  # type: ignore[assignment]
        theta = out["theta_full"]  # type: ignore[assignment]
        rows.append(
            {
                "quarter": int(quarter),
                "model": model.name,
                "q_extreme": float(q_extreme),
                "ridge_alpha": float(ridge_alpha),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "gain_all": float(met["gain_all"]),
                "gain_extreme": float(met["gain_extreme"]),
                "gain_non_extreme": float(met["gain_non_extreme"]),
                "rms_base": float(met["rms_base"]),
                "rms_full": float(met["rms_full"]),
                "v_thr_train": float(meta["v_thr_train"]),
                "omegaq_thr_train": float(meta["omegaq_thr_train"]),
                "theta_full": "|".join(f"{float(x):.8e}" for x in theta),
            }
        )
    return pd.DataFrame(rows)


def _run_single_experiment(
    *,
    exp: ExperimentSpec,
    df: pd.DataFrame,
    cache: dict[str, object],
    outdir: Path,
    train_end_year: int,
    test_year: int,
    n_folds: int,
    q_grid: list[float],
    ridge_grid: list[float],
    extreme_weight: float,
    n_perm: int,
    perm_block: int,
    n_boot: int,
    seed: int,
) -> dict[str, float | str]:
    outdir.mkdir(parents=True, exist_ok=True)
    train_idx = np.where(df["year"].to_numpy(dtype=int) <= train_end_year)[0]
    test_idx = np.where(df["year"].to_numpy(dtype=int) == test_year)[0]
    if len(train_idx) < n_folds * 16:
        raise ValueError(f"{exp.exp_id}: too few train samples for blocked CV.")
    if len(test_idx) < 100:
        raise ValueError(f"{exp.exp_id}: too few test samples.")

    print(f"[{exp.exp_id}] CV grid search...", flush=True)
    fold_rows: list[pd.DataFrame] = []
    stat_rows: list[dict[str, float | str]] = []
    for model in exp.models:
        for q_extreme in q_grid:
            for ridge_alpha in ridge_grid:
                fdf, stat = _evaluate_cv_config(
                    df=df,
                    cache=cache,
                    train_idx_full=train_idx,
                    n_folds=n_folds,
                    model=model,
                    q_extreme=float(q_extreme),
                    ridge_alpha=float(ridge_alpha),
                    extreme_weight=extreme_weight,
                )
                fold_rows.append(fdf)
                stat_rows.append(stat)
                print(
                    f"[{exp.exp_id}] model={model.name} q={q_extreme:.2f} alpha={ridge_alpha:.1e} "
                    f"score={float(stat['cv_score_mean']):.6f}",
                    flush=True,
                )
    cv_folds = pd.concat(fold_rows, ignore_index=True)
    cv_stats = pd.DataFrame(stat_rows)
    cv_folds.to_csv(outdir / "cv_folds.csv", index=False)
    cv_stats.to_csv(outdir / "cv_stats.csv", index=False)

    selected = _choose_config_one_se(cv_stats, n_folds=n_folds)
    pd.DataFrame([selected.to_dict()]).to_csv(outdir / "selected_config.csv", index=False)
    sel_model_name = str(selected["model"])
    sel_q = float(selected["q_extreme"])
    sel_alpha = float(selected["ridge_alpha"])
    sel_model = [m for m in exp.models if m.name == sel_model_name][0]
    print(
        f"[{exp.exp_id}] selected model={sel_model_name}, q_extreme={sel_q:.2f}, ridge_alpha={sel_alpha:.1e}",
        flush=True,
    )

    fit = _evaluate_single(
        df=df,
        cache=cache,
        train_idx=train_idx,
        eval_idx=test_idx,
        model=sel_model,
        q_extreme=sel_q,
        ridge_alpha=sel_alpha,
        lambda_override=None,
    )
    met = fit["metrics"]  # type: ignore[assignment]
    meta = fit["meta"]  # type: ignore[assignment]
    theta_base = fit["theta_base"]  # type: ignore[assignment]
    theta_full = fit["theta_full"]  # type: ignore[assignment]
    term_names_base = fit["term_names_base"]  # type: ignore[assignment]
    term_names_full = fit["term_names_full"]  # type: ignore[assignment]
    ms_base = np.asarray(fit["ms_base"], dtype=float)
    ms_full = np.asarray(fit["ms_full"], dtype=float)

    coef_rows = [{"part": "base", "term": n, "coef": float(c)} for n, c in zip(term_names_base, theta_base)]
    coef_rows += [{"part": "full", "term": n, "coef": float(c)} for n, c in zip(term_names_full, theta_full)]
    pd.DataFrame(coef_rows).to_csv(outdir / "coefficients.csv", index=False)

    rng = np.random.default_rng(seed)
    n_w = np.asarray(cache["n_points"], dtype=float)
    boot = np.zeros(n_boot, dtype=float)
    n_test = len(test_idx)
    for i in range(n_boot):
        rel = _bootstrap_rel_indices(n=n_test, block=perm_block, rng=rng)
        idx = test_idx[rel]
        rb = _weighted_rms(ms_base, n_w, idx)
        rf = _weighted_rms(ms_full, n_w, idx)
        boot[i] = float((rb - rf) / (rb + EPS))
    ci_lo = float(np.quantile(boot, 0.025))
    ci_hi = float(np.quantile(boot, 0.975))
    pd.DataFrame({"boot_id": np.arange(n_boot, dtype=int), "gain_all_boot": boot}).to_csv(
        outdir / "bootstrap_test_gain.csv", index=False
    )

    null = np.zeros(n_perm, dtype=float)
    ge_count = 0
    raw = {k: df[k].to_numpy(dtype=float) for k in LAMBDA_RAW_KEYS}
    for pid in range(n_perm):
        perm_idx = _block_permute_indices(len(df), block=perm_block, rng=rng)
        lam_override = {k: raw[k][perm_idx] for k in LAMBDA_RAW_KEYS}
        pfit = _evaluate_single(
            df=df,
            cache=cache,
            train_idx=train_idx,
            eval_idx=test_idx,
            model=sel_model,
            q_extreme=sel_q,
            ridge_alpha=sel_alpha,
            lambda_override=lam_override,
        )
        g = float(pfit["metrics"]["gain_all"])  # type: ignore[index]
        null[pid] = g
        if g >= float(met["gain_all"]):
            ge_count += 1
    p_value = float((ge_count + 1) / (n_perm + 1))
    pd.DataFrame({"perm_id": np.arange(n_perm, dtype=int), "gain_all_perm": null}).to_csv(
        outdir / "permutation_test.csv", index=False
    )

    quarterly = _run_quarterly(
        df=df,
        cache=cache,
        model=sel_model,
        q_extreme=sel_q,
        ridge_alpha=sel_alpha,
    )
    quarterly.to_csv(outdir / "quarterly_summary.csv", index=False)

    _plot_hist(
        null_vals=null,
        real_val=float(met["gain_all"]),
        out_path=outdir / "plot_permutation_gain.png",
        title=f"{exp.exp_id}: block-permutation null",
    )
    _plot_quarterly(quarterly, outdir / "plot_quarterly_gain.png", title=f"{exp.exp_id}: rolling-origin quarterly gain")

    q_mean = float(np.nanmean(quarterly["gain_all"])) if not quarterly.empty else np.nan
    q_min = float(np.nanmin(quarterly["gain_all"])) if not quarterly.empty else np.nan
    q_pos = float(np.nanmean(quarterly["gain_all"] > 0.0)) if not quarterly.empty else np.nan

    test_row = {
        "experiment_id": exp.exp_id,
        "description": exp.description,
        "selected_model": sel_model.name,
        "selected_q_extreme": sel_q,
        "selected_ridge_alpha": sel_alpha,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_gain_all": float(met["gain_all"]),
        "test_gain_extreme": float(met["gain_extreme"]),
        "test_gain_non_extreme": float(met["gain_non_extreme"]),
        "test_rms_base": float(met["rms_base"]),
        "test_rms_full": float(met["rms_full"]),
        "test_gain_ci95_lo": ci_lo,
        "test_gain_ci95_hi": ci_hi,
        "perm_p_value": p_value,
        "quarterly_mean_gain_all": q_mean,
        "quarterly_min_gain_all": q_min,
        "quarterly_positive_fraction": q_pos,
        "v_thr_train": float(meta["v_thr_train"]),
        "omegaq_thr_train": float(meta["omegaq_thr_train"]),
        "lambda_struct_mu_train": float(meta["lambda_struct_raw_mu_train"]),
        "lambda_struct_sd_train": float(meta["lambda_struct_raw_sd_train"]),
    }
    pd.DataFrame([test_row]).to_csv(outdir / "test_metrics.csv", index=False)

    report = [
        f"# {exp.exp_id}: {exp.description}",
        "",
        "## Selected config (one-SE)",
        f"- model: `{sel_model.name}` ({sel_model.description})",
        f"- q_extreme: {sel_q:.2f}",
        f"- ridge_alpha: {sel_alpha:.1e}",
        "",
        "## Out-of-time test (2019)",
        f"- gain_all: {float(met['gain_all']):.6f}",
        f"- gain_extreme: {float(met['gain_extreme']):.6f}",
        f"- gain_non_extreme: {float(met['gain_non_extreme']):.6f}",
        f"- CI95 gain_all: [{ci_lo:.6f}, {ci_hi:.6f}]",
        f"- permutation p-value: {p_value:.6f}",
        "",
        "## Quarterly rolling-origin (2019)",
        f"- mean gain_all: {q_mean:.6f}",
        f"- min gain_all: {q_min:.6f}",
        f"- positive-quarter fraction: {q_pos:.3f}",
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return test_row


def _experiment_specs() -> tuple[ExperimentSpec, ExperimentSpec]:
    base = (Term("src_pe", "s_pe", "one"),)

    exp11 = ExperimentSpec(
        exp_id="experiment_N11_source_proxy",
        description="Base source proxy + global/regime lambda corrections",
        models=(
            ModelSpec(
                name="n11_global_a",
                description="base + lambda_struct*c1",
                base_terms=base,
                full_terms=base + (Term("a", "c1", "lambda_struct_z"),),
            ),
            ModelSpec(
                name="n11_global_ab",
                description="base + lambda_struct*(c1,c2)",
                base_terms=base,
                full_terms=base + (
                    Term("a", "c1", "lambda_struct_z"),
                    Term("b", "c2", "lambda_struct_z"),
                ),
            ),
            ModelSpec(
                name="n11_regime_ab",
                description="base + lambda_struct*(c1,c2) + extreme interactions",
                base_terms=base,
                full_terms=base
                + (
                    Term("a", "c1", "lambda_struct_z"),
                    Term("b", "c2", "lambda_struct_z"),
                    Term("da_ext", "c1", "lambda_struct_z_ext"),
                    Term("db_ext", "c2", "lambda_struct_z_ext"),
                ),
            ),
        ),
    )

    exp12 = ExperimentSpec(
        exp_id="experiment_N12_localized_lambda",
        description="Base source proxy + localized multiscale lambda corrections",
        models=(
            ModelSpec(
                name="n12_ms_c1",
                description="base + (lambda_low,mid,high)*c1",
                base_terms=base,
                full_terms=base
                + (
                    Term("a_low", "c1", "lambda_low_z"),
                    Term("a_mid", "c1", "lambda_mid_z"),
                    Term("a_high", "c1", "lambda_high_z"),
                ),
            ),
            ModelSpec(
                name="n12_ms_c1c2",
                description="base + (lambda_low,mid,high)*(c1,c2)",
                base_terms=base,
                full_terms=base
                + (
                    Term("a_low", "c1", "lambda_low_z"),
                    Term("a_mid", "c1", "lambda_mid_z"),
                    Term("a_high", "c1", "lambda_high_z"),
                    Term("b_low", "c2", "lambda_low_z"),
                    Term("b_mid", "c2", "lambda_mid_z"),
                    Term("b_high", "c2", "lambda_high_z"),
                ),
            ),
            ModelSpec(
                name="n12_ms_regime",
                description="base + localized c1 terms + extreme interactions",
                base_terms=base,
                full_terms=base
                + (
                    Term("a_low", "c1", "lambda_low_z"),
                    Term("a_mid", "c1", "lambda_mid_z"),
                    Term("a_high", "c1", "lambda_high_z"),
                    Term("da_low_ext", "c1", "lambda_low_z_ext"),
                    Term("da_mid_ext", "c1", "lambda_mid_z_ext"),
                    Term("da_high_ext", "c1", "lambda_high_z_ext"),
                ),
            ),
        ),
    )
    return exp11, exp12


def run_dual_experiments(
    *,
    input_nc: Path,
    lambda_timeseries_csv: Path,
    moments_csv: Path,
    out_root: Path,
    train_end_year: int,
    test_year: int,
    batch_size: int,
    lat_stride: int,
    lon_stride: int,
    p_min_hpa: float,
    p_max_hpa: float,
    h_scale_hpa: float,
    n_folds: int,
    q_grid: list[float],
    ridge_grid: list[float],
    extreme_weight: float,
    n_perm: int,
    perm_block: int,
    n_boot: int,
    seed: int,
    force_recompute_moments: bool,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    print("[N_followup] building/loading enhanced moments...", flush=True)
    moment_df = _load_or_build_moments(
        moments_csv=moments_csv,
        input_nc=input_nc,
        batch_size=batch_size,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        p_min_hpa=p_min_hpa,
        p_max_hpa=p_max_hpa,
        h_scale_hpa=h_scale_hpa,
        force_recompute=force_recompute_moments,
    )
    lambda_df = _load_lambda_table(lambda_timeseries_csv=lambda_timeseries_csv)
    df = moment_df.merge(lambda_df, on="time_index", how="inner")
    if len(df) != len(moment_df):
        raise ValueError("Failed to align moments and lambda tables by time_index.")
    df = df.sort_values("time_index").reset_index(drop=True)
    df["time_dt"] = pd.to_datetime(df["time"])
    df["year"] = df["time_dt"].dt.year
    df["quarter"] = df["time_dt"].dt.quarter
    df.to_csv(out_root / "timeseries_enhanced.csv", index=False)
    cache = _prepare_cache(df)

    exp11, exp12 = _experiment_specs()
    rows: list[dict[str, float | str]] = []
    for exp in (exp11, exp12):
        print(f"[N_followup] running {exp.exp_id}...", flush=True)
        row = _run_single_experiment(
            exp=exp,
            df=df,
            cache=cache,
            outdir=out_root / exp.exp_id,
            train_end_year=train_end_year,
            test_year=test_year,
            n_folds=n_folds,
            q_grid=q_grid,
            ridge_grid=ridge_grid,
            extreme_weight=extreme_weight,
            n_perm=n_perm,
            perm_block=perm_block,
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(row)

    comp = pd.DataFrame(rows)
    comp.to_csv(out_root / "experiment_N_followup_comparison.csv", index=False)

    lines = [
        "# Experiment N Follow-up Comparison",
        "",
        "This table compares two follow-up branches:",
        "- `N11`: source-proxy baseline + global/regime lambda",
        "- `N12`: source-proxy baseline + localized multiscale lambda",
        "",
        "```text",
        comp.to_string(index=False),
        "```",
    ]
    (out_root / "experiment_N_followup_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[N_followup] done.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
    )
    parser.add_argument(
        "--lambda-timeseries-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    parser.add_argument(
        "--moments-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_N_followup/moments_enhanced.csv"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("clean_experiments/results/experiment_N_followup"),
    )
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lat-stride", type=int, default=1)
    parser.add_argument("--lon-stride", type=int, default=1)
    parser.add_argument("--p-min-hpa", type=float, default=300.0)
    parser.add_argument("--p-max-hpa", type=float, default=850.0)
    parser.add_argument("--h-scale-hpa", type=float, default=500.0)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--q-grid", type=float, nargs="+", default=[0.80, 0.85, 0.90])
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--extreme-weight", type=float, default=0.5)
    parser.add_argument("--n-perm", type=int, default=140)
    parser.add_argument("--perm-block", type=int, default=24)
    parser.add_argument("--n-boot", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompute-moments", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dual_experiments(
        input_nc=args.input_nc,
        lambda_timeseries_csv=args.lambda_timeseries_csv,
        moments_csv=args.moments_csv,
        out_root=args.out_root,
        train_end_year=args.train_end_year,
        test_year=args.test_year,
        batch_size=args.batch_size,
        lat_stride=args.lat_stride,
        lon_stride=args.lon_stride,
        p_min_hpa=args.p_min_hpa,
        p_max_hpa=args.p_max_hpa,
        h_scale_hpa=args.h_scale_hpa,
        n_folds=args.n_folds,
        q_grid=list(args.q_grid),
        ridge_grid=list(args.ridge_grid),
        extreme_weight=args.extreme_weight,
        n_perm=args.n_perm,
        perm_block=args.perm_block,
        n_boot=args.n_boot,
        seed=args.seed,
        force_recompute_moments=bool(args.force_recompute_moments),
    )


if __name__ == "__main__":
    main()
