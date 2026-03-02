#!/usr/bin/env python3
"""Experiment N: direct moisture-budget closure with Lambda coupling in p-coordinates."""

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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    features: tuple[tuple[str, int, str], ...]
    complexity: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "global_a": ModelSpec(
        name="global_a",
        description="R + a*(lambda*C1)",
        features=(("a", 0, "all"),),
        complexity=1,
    ),
    "global_ab": ModelSpec(
        name="global_ab",
        description="R + a*(lambda*C1) + b*(lambda*C2)",
        features=(("a", 0, "all"), ("b", 1, "all")),
        complexity=2,
    ),
    "regime_ab": ModelSpec(
        name="regime_ab",
        description="global_ab + extreme interactions",
        features=(
            ("a", 0, "all"),
            ("b", 1, "all"),
            ("da_ext", 0, "extreme"),
            ("db_ext", 1, "extreme"),
        ),
        complexity=4,
    ),
}


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
        # netCDF calendars may return cftime objects; stringify before pandas conversion.
        dt_arr = np.asarray(dt)
        if dt_arr.size == 0:
            raise ValueError("Empty time coordinate.")
        if np.issubdtype(dt_arr.dtype, np.datetime64):
            ts = pd.to_datetime(dt_arr)
        else:
            ts = pd.to_datetime([str(x) for x in dt_arr])
        return ts.to_numpy(dtype="datetime64[ns]")
    ts = pd.to_datetime(raw)
    return ts.to_numpy(dtype="datetime64[ns]")


def _time_to_seconds(time_ns: np.ndarray) -> np.ndarray:
    t = np.asarray(time_ns)
    if not np.issubdtype(t.dtype, np.datetime64):
        raise ValueError("time_ns must be datetime64.")
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


def _compute_moments(
    *,
    input_nc: Path,
    batch_size: int,
    lat_stride: int,
    lon_stride: int,
    p_min_hpa: float,
    p_max_hpa: float,
    h_scale_hpa: float,
) -> pd.DataFrame:
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
        p_sel_hpa = levels_hpa[level_mask]
        p_sel_pa = p_sel_hpa * 100.0
        p_norm = p_sel_pa / 100000.0
        h_scale_pa = h_scale_hpa * 100.0

        q_var = ds.variables[q_name]
        u_pl_var = ds.variables[u_pl_name]
        v_pl_var = ds.variables[v_pl_name]
        w_pl_var = ds.variables[w_pl_name]
        u2_var = ds.variables[u2_name]
        v2_var = ds.variables[v2_name]

        q_dims = tuple(q_var.dimensions)
        u_pl_dims = tuple(u_pl_var.dimensions)
        v_pl_dims = tuple(v_pl_var.dimensions)
        w_pl_dims = tuple(w_pl_var.dimensions)
        u2_dims = tuple(u2_var.dimensions)
        v2_dims = tuple(v2_var.dimensions)

        nt = int(q_var.shape[q_dims.index(time_name)])

        n_count = np.zeros(nt, dtype=float)
        m_rr = np.full(nt, np.nan, dtype=float)
        m_rc1 = np.full(nt, np.nan, dtype=float)
        m_rc2 = np.full(nt, np.nan, dtype=float)
        m_11 = np.full(nt, np.nan, dtype=float)
        m_22 = np.full(nt, np.nan, dtype=float)
        m_12 = np.full(nt, np.nan, dtype=float)
        proxy_vorticity = np.full(nt, np.nan, dtype=float)
        proxy_omega_q = np.full(nt, np.nan, dtype=float)

        eo_t = _edge_order(len(time_s))
        eo_x = _edge_order(len(x_m))
        eo_y = _edge_order(len(y_m))
        eo_p = _edge_order(len(p_sel_pa))

        for start in range(0, nt, batch_size):
            stop = min(start + batch_size, nt)
            t0 = max(0, start - 1)
            t1 = min(nt, stop + 1)

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

            time_blk_s = time_s[t0:t1]
            dq_dt_blk = np.gradient(q_blk, time_blk_s, axis=0, edge_order=min(eo_t, _edge_order(len(time_blk_s))))

            c0 = start - t0
            c1 = c0 + (stop - start)
            q = q_blk[c0:c1]
            u_pl = u_pl_blk[c0:c1]
            v_pl = v_pl_blk[c0:c1]
            w_pl = w_pl_blk[c0:c1]
            dq_dt = dq_dt_blk[c0:c1]
            u2 = u2_blk[c0:c1]
            v2 = v2_blk[c0:c1]

            dq_dx = np.gradient(q, x_m, axis=3, edge_order=eo_x)
            dq_dy = np.gradient(q, y_m, axis=2, edge_order=eo_y)
            dq_dp = np.gradient(q, p_sel_pa, axis=1, edge_order=eo_p)
            domega_dp = np.gradient(w_pl, p_sel_pa, axis=1, edge_order=eo_p)

            # Residual of standard moisture equation (without explicit source closure term).
            r_std = dq_dt + u_pl * dq_dx + v_pl * dq_dy + w_pl * dq_dp

            # Basis terms for Lambda-driven interscale correction.
            c1_field = -(p_norm[None, :, None, None] * w_pl * dq_dp)
            c2_field = -(p_norm[None, :, None, None] * h_scale_pa * domega_dp * dq_dp)

            for i in range(stop - start):
                gi = start + i
                r = r_std[i]
                c1_i = c1_field[i]
                c2_i = c2_field[i]
                valid = np.isfinite(r) & np.isfinite(c1_i) & np.isfinite(c2_i)
                if not np.any(valid):
                    continue
                rv = r[valid]
                c1v = c1_i[valid]
                c2v = c2_i[valid]
                n = float(rv.size)
                n_count[gi] = n
                m_rr[gi] = _mean_or_nan(rv * rv)
                m_rc1[gi] = _mean_or_nan(rv * c1v)
                m_rc2[gi] = _mean_or_nan(rv * c2v)
                m_11[gi] = _mean_or_nan(c1v * c1v)
                m_22[gi] = _mean_or_nan(c2v * c2v)
                m_12[gi] = _mean_or_nan(c1v * c2v)

            dv_dx = np.gradient(v2, x_m, axis=2, edge_order=eo_x)
            du_dy = np.gradient(u2, y_m, axis=1, edge_order=eo_y)
            zeta = dv_dx - du_dy
            proxy_vorticity[start:stop] = np.nanmean(np.abs(zeta), axis=(1, 2))
            proxy_omega_q[start:stop] = np.nanmean(-(w_pl * q), axis=(1, 2, 3))

            print(
                f"[experiment_N] moments progress: {stop}/{nt} "
                f"(levels={int(level_mask.sum())}, grid={len(lat)}x{len(lon)})",
                flush=True,
            )

    out = pd.DataFrame(
        {
            "time_index": np.arange(nt, dtype=int),
            "time": pd.to_datetime(time_ns).astype(str),
            "n_points": n_count,
            "m_rr": m_rr,
            "m_rc1": m_rc1,
            "m_rc2": m_rc2,
            "m_11": m_11,
            "m_22": m_22,
            "m_12": m_12,
            "proxy_vorticity_abs_mean": proxy_vorticity,
            "proxy_omega_q_850_300": proxy_omega_q,
        }
    )
    return out


def _load_lambda_series(lambda_timeseries_csv: Path) -> pd.DataFrame:
    usecols = ["time_index", "time", "lambda_struct"]
    df = pd.read_csv(lambda_timeseries_csv, usecols=usecols)
    if "lambda_struct" not in df.columns:
        raise ValueError(f"lambda_struct is missing in {lambda_timeseries_csv}.")
    return df.rename(columns={"lambda_struct": "lambda_raw"})


def _merge_timeseries(moment_df: pd.DataFrame, lambda_df: pd.DataFrame) -> pd.DataFrame:
    merged = moment_df.merge(lambda_df[["time_index", "lambda_raw"]], on="time_index", how="inner")
    if len(merged) != len(moment_df):
        raise ValueError("Failed to align moment and lambda series by time_index.")

    merged["time_dt"] = pd.to_datetime(merged["time"])
    merged["year"] = merged["time_dt"].dt.year
    merged["quarter"] = merged["time_dt"].dt.quarter

    good = np.isfinite(merged["m_rr"].to_numpy(dtype=float))
    if int(np.sum(good)) < int(0.8 * len(merged)):
        raise ValueError("Too many invalid budget moments.")
    return merged


def _lambda_standardize(lambda_raw: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.nanmean(lambda_raw[train_idx]))
    sd = float(np.nanstd(lambda_raw[train_idx]))
    if sd < 1e-15:
        sd = 1.0
    return (lambda_raw - mu) / sd, mu, sd


def _compute_extreme_mask(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    q_extreme: float,
) -> tuple[np.ndarray, float, float]:
    vort = df["proxy_vorticity_abs_mean"].to_numpy(dtype=float)
    omegaq = df["proxy_omega_q_850_300"].to_numpy(dtype=float)
    v_thr = float(np.quantile(vort[train_idx], q_extreme))
    o_thr = float(np.quantile(omegaq[train_idx], q_extreme))
    extreme = (vort >= v_thr) | (omegaq >= o_thr)
    return extreme.astype(float), v_thr, o_thr


def _build_feature_factors(
    *,
    lambda_z: np.ndarray,
    extreme_mask: np.ndarray,
    spec: ModelSpec,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n = len(lambda_z)
    k = len(spec.features)
    factors = np.zeros((n, k), dtype=float)
    comps = np.zeros(k, dtype=int)
    names: list[str] = []
    for j, (name, comp, gate) in enumerate(spec.features):
        gate_arr = np.ones(n, dtype=float) if gate == "all" else extreme_mask
        factors[:, j] = lambda_z * gate_arr
        comps[j] = int(comp)
        names.append(name)
    return factors, comps, names


def _pick_mr(df: pd.DataFrame, comp: int, idx: np.ndarray) -> np.ndarray:
    if comp == 0:
        return df["m_rc1"].to_numpy(dtype=float)[idx]
    return df["m_rc2"].to_numpy(dtype=float)[idx]


def _pick_mc(df: pd.DataFrame, ci: int, cj: int, idx: np.ndarray) -> np.ndarray:
    if ci == 0 and cj == 0:
        return df["m_11"].to_numpy(dtype=float)[idx]
    if ci == 1 and cj == 1:
        return df["m_22"].to_numpy(dtype=float)[idx]
    return df["m_12"].to_numpy(dtype=float)[idx]


def _fit_theta(
    *,
    df: pd.DataFrame,
    idx: np.ndarray,
    factors: np.ndarray,
    comps: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    n_w = df["n_points"].to_numpy(dtype=float)[idx]
    k = len(comps)
    a = np.zeros((k, k), dtype=float)
    b = np.zeros(k, dtype=float)

    for i in range(k):
        fi = factors[idx, i]
        mr_i = _pick_mr(df, int(comps[i]), idx)
        b[i] = -float(np.sum(n_w * fi * mr_i))
        for j in range(i, k):
            fj = factors[idx, j]
            mc_ij = _pick_mc(df, int(comps[i]), int(comps[j]), idx)
            a_ij = float(np.sum(n_w * fi * fj * mc_ij))
            a[i, j] = a_ij
            a[j, i] = a_ij

    lhs = a + ridge_alpha * np.eye(k, dtype=float)
    try:
        theta = np.linalg.solve(lhs, b)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(lhs, rcond=1e-10) @ b
    theta = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
    return theta


def _mean_square_series(
    *,
    df: pd.DataFrame,
    theta: np.ndarray,
    factors: np.ndarray,
    comps: np.ndarray,
) -> np.ndarray:
    ms = df["m_rr"].to_numpy(dtype=float).copy()
    if len(theta) == 0:
        return ms

    for i in range(len(theta)):
        mr_i = _pick_mr(df, int(comps[i]), np.arange(len(df), dtype=int))
        ms += 2.0 * theta[i] * factors[:, i] * mr_i

    for i in range(len(theta)):
        for j in range(i, len(theta)):
            mc = _pick_mc(df, int(comps[i]), int(comps[j]), np.arange(len(df), dtype=int))
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
    if int(valid.sum()) == 0:
        return np.nan
    return float(np.sqrt(np.sum(ww[valid] * vv[valid]) / (np.sum(ww[valid]) + EPS)))


def _gain_metrics(
    *,
    df: pd.DataFrame,
    ms_mod: np.ndarray,
    eval_idx: np.ndarray,
    extreme_mask: np.ndarray,
) -> dict[str, float]:
    n_w = df["n_points"].to_numpy(dtype=float)
    ms_base = df["m_rr"].to_numpy(dtype=float)

    rms_base = _weighted_rms(ms_base, n_w, eval_idx)
    rms_mod = _weighted_rms(ms_mod, n_w, eval_idx)
    gain_all = float((rms_base - rms_mod) / (rms_base + EPS))

    eval_ext = eval_idx[extreme_mask[eval_idx] > 0.5]
    eval_non = eval_idx[extreme_mask[eval_idx] <= 0.5]

    rms_base_ext = _weighted_rms(ms_base, n_w, eval_ext)
    rms_mod_ext = _weighted_rms(ms_mod, n_w, eval_ext)
    gain_ext = float((rms_base_ext - rms_mod_ext) / (rms_base_ext + EPS)) if np.isfinite(rms_base_ext) else np.nan

    rms_base_non = _weighted_rms(ms_base, n_w, eval_non)
    rms_mod_non = _weighted_rms(ms_mod, n_w, eval_non)
    gain_non = float((rms_base_non - rms_mod_non) / (rms_base_non + EPS)) if np.isfinite(rms_base_non) else np.nan

    return {
        "rms_base": rms_base,
        "rms_mod": rms_mod,
        "gain_all": gain_all,
        "gain_extreme": gain_ext,
        "gain_non_extreme": gain_non,
        "n_eval": int(len(eval_idx)),
        "n_eval_extreme": int(len(eval_ext)),
        "n_eval_non_extreme": int(len(eval_non)),
    }


def _safe_score(gain_all: float, gain_extreme: float, extreme_weight: float) -> float:
    ge = gain_extreme if np.isfinite(gain_extreme) else 0.0
    return float(gain_all + extreme_weight * ge)


def _evaluate_cv_config(
    *,
    df: pd.DataFrame,
    train_idx_full: np.ndarray,
    n_folds: int,
    spec: ModelSpec,
    q_extreme: float,
    ridge_alpha: float,
    extreme_weight: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rel_splits = _blocked_splits(len(train_idx_full), n_folds=n_folds)
    rows: list[dict[str, float | int | str]] = []

    lambda_raw = df["lambda_raw"].to_numpy(dtype=float)
    for fold_id, (tr_rel, va_rel) in enumerate(rel_splits):
        tr_idx = train_idx_full[tr_rel]
        va_idx = train_idx_full[va_rel]

        lambda_z, lam_mu, lam_sd = _lambda_standardize(lambda_raw, tr_idx)
        extreme_mask, v_thr, o_thr = _compute_extreme_mask(df, tr_idx, q_extreme)
        factors, comps, feat_names = _build_feature_factors(lambda_z=lambda_z, extreme_mask=extreme_mask, spec=spec)
        theta = _fit_theta(df=df, idx=tr_idx, factors=factors, comps=comps, ridge_alpha=ridge_alpha)
        ms_mod = _mean_square_series(df=df, theta=theta, factors=factors, comps=comps)
        metrics = _gain_metrics(df=df, ms_mod=ms_mod, eval_idx=va_idx, extreme_mask=extreme_mask)
        score = _safe_score(metrics["gain_all"], metrics["gain_extreme"], extreme_weight=extreme_weight)

        rows.append(
            {
                "fold_id": int(fold_id),
                "model": spec.name,
                "q_extreme": float(q_extreme),
                "ridge_alpha": float(ridge_alpha),
                "feature_names": "|".join(feat_names),
                "lambda_mu_train": float(lam_mu),
                "lambda_sd_train": float(lam_sd),
                "v_thr_train": float(v_thr),
                "omegaq_thr_train": float(o_thr),
                "gain_all": float(metrics["gain_all"]),
                "gain_extreme": float(metrics["gain_extreme"]),
                "gain_non_extreme": float(metrics["gain_non_extreme"]),
                "rms_base": float(metrics["rms_base"]),
                "rms_mod": float(metrics["rms_mod"]),
                "score": float(score),
            }
        )

    fold_df = pd.DataFrame(rows)
    stats = {
        "model": spec.name,
        "description": spec.description,
        "complexity": int(spec.complexity),
        "q_extreme": float(q_extreme),
        "ridge_alpha": float(ridge_alpha),
        "cv_gain_all_median": float(np.nanmedian(fold_df["gain_all"])),
        "cv_gain_all_mean": float(np.nanmean(fold_df["gain_all"])),
        "cv_gain_extreme_median": float(np.nanmedian(fold_df["gain_extreme"])),
        "cv_gain_extreme_mean": float(np.nanmean(fold_df["gain_extreme"])),
        "cv_gain_non_extreme_median": float(np.nanmedian(fold_df["gain_non_extreme"])),
        "cv_score_mean": float(np.nanmean(fold_df["score"])),
        "cv_score_std": float(np.nanstd(fold_df["score"], ddof=1)),
        "cv_positive_gain_frac": float(np.nanmean(fold_df["gain_all"] > 0.0)),
        "cv_extreme_positive_frac": float(np.nanmean(fold_df["gain_extreme"] > 0.0)),
    }
    return fold_df, stats


def _choose_config_one_se(cv_stats: pd.DataFrame, n_folds: int) -> pd.Series:
    if cv_stats.empty:
        raise ValueError("No CV stats to choose from.")
    best_idx = int(cv_stats["cv_score_mean"].idxmax())
    best_row = cv_stats.loc[best_idx]
    se = float(best_row["cv_score_std"] / np.sqrt(max(float(n_folds), 1.0)))
    threshold = float(best_row["cv_score_mean"] - se)
    eligible = cv_stats[cv_stats["cv_score_mean"] >= threshold].copy()
    if eligible.empty:
        return best_row
    eligible = eligible.sort_values(
        by=["complexity", "ridge_alpha", "cv_score_mean", "cv_gain_all_median"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return eligible.iloc[0]


def _block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    starts = list(range(0, n, block))
    blocks = [x[s : min(s + block, n)] for s in starts]
    rng.shuffle(blocks)
    return np.concatenate(blocks, axis=0)[:n]


def _block_bootstrap_rel_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = np.arange(0, n, block, dtype=int)
    parts: list[np.ndarray] = []
    cur = 0
    while cur < n:
        s = int(rng.choice(starts))
        e = min(s + block, n)
        part = np.arange(s, e, dtype=int)
        parts.append(part)
        cur += len(part)
    return np.concatenate(parts)[:n]


def _fit_eval_single(
    *,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    spec: ModelSpec,
    q_extreme: float,
    ridge_alpha: float,
    lambda_raw: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    lambda_z, lam_mu, lam_sd = _lambda_standardize(lambda_raw, train_idx)
    extreme_mask, v_thr, o_thr = _compute_extreme_mask(df, train_idx, q_extreme)
    factors, comps, _ = _build_feature_factors(lambda_z=lambda_z, extreme_mask=extreme_mask, spec=spec)
    theta = _fit_theta(df=df, idx=train_idx, factors=factors, comps=comps, ridge_alpha=ridge_alpha)
    ms_mod = _mean_square_series(df=df, theta=theta, factors=factors, comps=comps)
    metrics = _gain_metrics(df=df, ms_mod=ms_mod, eval_idx=eval_idx, extreme_mask=extreme_mask)
    return metrics, theta, ms_mod, extreme_mask, lam_mu, lam_sd, v_thr, o_thr


def _run_quarterly_rolling(
    *,
    df: pd.DataFrame,
    spec: ModelSpec,
    q_extreme: float,
    ridge_alpha: float,
) -> pd.DataFrame:
    lambda_raw = df["lambda_raw"].to_numpy(dtype=float)
    t = df["time_dt"]
    rows: list[dict[str, float | int | str]] = []
    for quarter in (1, 2, 3, 4):
        test_mask = (df["year"] == 2019) & (df["quarter"] == quarter)
        if int(test_mask.sum()) == 0:
            continue
        q_start = pd.Timestamp(year=2019, month=3 * (quarter - 1) + 1, day=1)
        train_mask = t < q_start
        train_idx = np.where(train_mask.to_numpy())[0]
        test_idx = np.where(test_mask.to_numpy())[0]
        if len(train_idx) < 500 or len(test_idx) < 40:
            continue
        metrics, theta, _, _, lam_mu, lam_sd, v_thr, o_thr = _fit_eval_single(
            df=df,
            train_idx=train_idx,
            eval_idx=test_idx,
            spec=spec,
            q_extreme=q_extreme,
            ridge_alpha=ridge_alpha,
            lambda_raw=lambda_raw,
        )
        rows.append(
            {
                "quarter": int(quarter),
                "model": spec.name,
                "q_extreme": float(q_extreme),
                "ridge_alpha": float(ridge_alpha),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "lambda_mu_train": float(lam_mu),
                "lambda_sd_train": float(lam_sd),
                "v_thr_train": float(v_thr),
                "omegaq_thr_train": float(o_thr),
                "gain_all": float(metrics["gain_all"]),
                "gain_extreme": float(metrics["gain_extreme"]),
                "gain_non_extreme": float(metrics["gain_non_extreme"]),
                "rms_base": float(metrics["rms_base"]),
                "rms_mod": float(metrics["rms_mod"]),
                "theta": "|".join(f"{x:.8e}" for x in theta),
            }
        )
    return pd.DataFrame(rows)


def _plot_quarterly(quarterly_df: pd.DataFrame, out_path: Path) -> None:
    if quarterly_df.empty:
        return
    q = quarterly_df["quarter"].to_numpy(dtype=int)
    g = quarterly_df["gain_all"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(q, g, color="#1f77b4")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("2019 Quarter")
    ax.set_ylabel("RMS gain fraction")
    ax.set_title("Experiment N: Rolling-origin quarterly gain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_permutation(null_gains: np.ndarray, real_gain: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.hist(null_gains, bins=24, color="#2ca02c", alpha=0.75, edgecolor="white")
    ax.axvline(real_gain, color="#d62728", lw=2.0, label=f"real={real_gain:.4f}")
    ax.set_xlabel("Test gain under permuted lambda")
    ax.set_ylabel("Count")
    ax.set_title("Experiment N: block-permutation null distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_experiment(
    *,
    input_nc: Path,
    lambda_timeseries_csv: Path,
    outdir: Path,
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
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print("[experiment_N] Step 1/4: computing budget moments from ERA5 vertical fields...", flush=True)
    moment_df = _compute_moments(
        input_nc=input_nc,
        batch_size=batch_size,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        p_min_hpa=p_min_hpa,
        p_max_hpa=p_max_hpa,
        h_scale_hpa=h_scale_hpa,
    )
    lambda_df = _load_lambda_series(lambda_timeseries_csv=lambda_timeseries_csv)
    df = _merge_timeseries(moment_df=moment_df, lambda_df=lambda_df).sort_values("time_index").reset_index(drop=True)
    df.to_csv(outdir / "budget_moments_timeseries.csv", index=False)

    train_idx = np.where(df["year"].to_numpy(dtype=int) <= train_end_year)[0]
    test_idx = np.where(df["year"].to_numpy(dtype=int) == test_year)[0]
    if len(train_idx) < n_folds * 16:
        raise ValueError("Too few train samples for blocked CV.")
    if len(test_idx) < 100:
        raise ValueError("Too few test samples.")
    print(
        f"[experiment_N] Split: train years <= {train_end_year} (n={len(train_idx)}), "
        f"test year = {test_year} (n={len(test_idx)})",
        flush=True,
    )

    print("[experiment_N] Step 2/4: blocked CV + one-SE model selection...", flush=True)
    cv_fold_rows: list[pd.DataFrame] = []
    cv_stats_rows: list[dict[str, float]] = []
    for model_name in ("global_a", "global_ab", "regime_ab"):
        spec = MODEL_SPECS[model_name]
        for q_extreme in q_grid:
            for ridge_alpha in ridge_grid:
                fold_df, stat = _evaluate_cv_config(
                    df=df,
                    train_idx_full=train_idx,
                    n_folds=n_folds,
                    spec=spec,
                    q_extreme=float(q_extreme),
                    ridge_alpha=float(ridge_alpha),
                    extreme_weight=extreme_weight,
                )
                cv_fold_rows.append(fold_df)
                cv_stats_rows.append(stat)
                print(
                    f"[experiment_N] CV done: model={model_name} q={q_extreme:.2f} alpha={ridge_alpha:.1e} "
                    f"score={stat['cv_score_mean']:.5f}",
                    flush=True,
                )

    cv_folds_df = pd.concat(cv_fold_rows, ignore_index=True)
    cv_stats_df = pd.DataFrame(cv_stats_rows)
    cv_folds_df.to_csv(outdir / "cv_folds.csv", index=False)
    cv_stats_df.to_csv(outdir / "cv_stats.csv", index=False)

    selected = _choose_config_one_se(cv_stats_df, n_folds=n_folds)
    sel_model = str(selected["model"])
    sel_q = float(selected["q_extreme"])
    sel_alpha = float(selected["ridge_alpha"])
    sel_spec = MODEL_SPECS[sel_model]
    pd.DataFrame([selected.to_dict()]).to_csv(outdir / "selected_config.csv", index=False)
    print(
        f"[experiment_N] selected model={sel_model}, q_extreme={sel_q:.2f}, ridge_alpha={sel_alpha:.1e}",
        flush=True,
    )

    print("[experiment_N] Step 3/4: final train/test fit, permutation, bootstrap...", flush=True)
    lambda_raw = df["lambda_raw"].to_numpy(dtype=float)
    test_metrics, theta, ms_mod, extreme_mask, lam_mu, lam_sd, v_thr, o_thr = _fit_eval_single(
        df=df,
        train_idx=train_idx,
        eval_idx=test_idx,
        spec=sel_spec,
        q_extreme=sel_q,
        ridge_alpha=sel_alpha,
        lambda_raw=lambda_raw,
    )
    ms_base = df["m_rr"].to_numpy(dtype=float)
    n_w = df["n_points"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    boot_gains = np.zeros(n_boot, dtype=float)
    n_test = len(test_idx)
    for i in range(n_boot):
        rel = _block_bootstrap_rel_indices(n=n_test, block=perm_block, rng=rng)
        idx_boot = test_idx[rel]
        rms_b = _weighted_rms(ms_base, n_w, idx_boot)
        rms_m = _weighted_rms(ms_mod, n_w, idx_boot)
        boot_gains[i] = float((rms_b - rms_m) / (rms_b + EPS))
    ci_lo = float(np.quantile(boot_gains, 0.025))
    ci_hi = float(np.quantile(boot_gains, 0.975))

    null_gains = np.zeros(n_perm, dtype=float)
    count_ge = 0
    for pid in range(n_perm):
        lam_perm = _block_permute(lambda_raw, block=perm_block, rng=rng)
        m_perm, _, _, _, _, _, _, _ = _fit_eval_single(
            df=df,
            train_idx=train_idx,
            eval_idx=test_idx,
            spec=sel_spec,
            q_extreme=sel_q,
            ridge_alpha=sel_alpha,
            lambda_raw=lam_perm,
        )
        g = float(m_perm["gain_all"])
        null_gains[pid] = g
        if g >= float(test_metrics["gain_all"]):
            count_ge += 1
    p_value = float((count_ge + 1) / (n_perm + 1))

    coef_rows = []
    for j, (name, _, _) in enumerate(sel_spec.features):
        coef_rows.append({"term": name, "coef": float(theta[j])})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(outdir / "coefficients.csv", index=False)

    test_row = {
        "model": sel_spec.name,
        "description": sel_spec.description,
        "q_extreme": sel_q,
        "ridge_alpha": sel_alpha,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_gain_all": float(test_metrics["gain_all"]),
        "test_gain_extreme": float(test_metrics["gain_extreme"]),
        "test_gain_non_extreme": float(test_metrics["gain_non_extreme"]),
        "test_rms_base": float(test_metrics["rms_base"]),
        "test_rms_mod": float(test_metrics["rms_mod"]),
        "test_gain_ci95_lo": ci_lo,
        "test_gain_ci95_hi": ci_hi,
        "perm_p_value": p_value,
        "lambda_mu_train": float(lam_mu),
        "lambda_sd_train": float(lam_sd),
        "v_thr_train": float(v_thr),
        "omegaq_thr_train": float(o_thr),
        "train_end_year": int(train_end_year),
        "test_year": int(test_year),
        "lat_stride": int(lat_stride),
        "lon_stride": int(lon_stride),
        "p_min_hpa": float(p_min_hpa),
        "p_max_hpa": float(p_max_hpa),
    }
    pd.DataFrame([test_row]).to_csv(outdir / "test_metrics.csv", index=False)
    pd.DataFrame({"perm_id": np.arange(n_perm, dtype=int), "gain_all_perm": null_gains}).to_csv(
        outdir / "permutation_test.csv", index=False
    )
    pd.DataFrame({"boot_id": np.arange(n_boot, dtype=int), "gain_all_boot": boot_gains}).to_csv(
        outdir / "bootstrap_test_gain.csv", index=False
    )

    print("[experiment_N] Step 4/4: quarterly rolling-origin backtest + report...", flush=True)
    quarterly_df = _run_quarterly_rolling(
        df=df,
        spec=sel_spec,
        q_extreme=sel_q,
        ridge_alpha=sel_alpha,
    )
    quarterly_df.to_csv(outdir / "quarterly_summary.csv", index=False)

    _plot_quarterly(quarterly_df, outdir / "plot_quarterly_gain.png")
    _plot_permutation(null_gains, real_gain=float(test_metrics["gain_all"]), out_path=outdir / "plot_permutation_gain.png")

    q_mean = float(np.nanmean(quarterly_df["gain_all"])) if not quarterly_df.empty else np.nan
    q_min = float(np.nanmin(quarterly_df["gain_all"])) if not quarterly_df.empty else np.nan
    q_pos_frac = float(np.nanmean(quarterly_df["gain_all"] > 0.0)) if not quarterly_df.empty else np.nan

    report = [
        "# Experiment N: Navier-Stokes Moisture Budget Closure",
        "",
        "## Setup",
        f"- Input: `{input_nc}`",
        f"- Lambda source: `{lambda_timeseries_csv}`",
        f"- Pressure layer: {p_min_hpa:.0f}-{p_max_hpa:.0f} hPa",
        f"- Spatial stride: lat={lat_stride}, lon={lon_stride}",
        f"- Train years: <= {train_end_year}, test year: {test_year}",
        "",
        "## Selected config (one-SE)",
        f"- model: `{sel_spec.name}` ({sel_spec.description})",
        f"- q_extreme: {sel_q:.2f}",
        f"- ridge_alpha: {sel_alpha:.1e}",
        "",
        "## Test (out-of-time)",
        f"- gain_all: {test_row['test_gain_all']:.6f}",
        f"- gain_extreme: {test_row['test_gain_extreme']:.6f}",
        f"- gain_non_extreme: {test_row['test_gain_non_extreme']:.6f}",
        f"- CI95 gain_all: [{ci_lo:.6f}, {ci_hi:.6f}]",
        f"- permutation p-value: {p_value:.6f}",
        "",
        "## Quarterly rolling-origin 2019",
        f"- mean gain_all: {q_mean:.6f}",
        f"- min gain_all: {q_min:.6f}",
        f"- positive-quarter fraction: {q_pos_frac:.3f}",
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("[experiment_N] done.", flush=True)


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
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_N_navier_stokes_budget"),
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
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2, 1e-1, 1.0])
    parser.add_argument("--extreme-weight", type=float, default=0.5)
    parser.add_argument("--n-perm", type=int, default=140)
    parser.add_argument("--perm-block", type=int, default=24)
    parser.add_argument("--n-boot", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        input_nc=args.input_nc,
        lambda_timeseries_csv=args.lambda_timeseries_csv,
        outdir=args.outdir,
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
    )


if __name__ == "__main__":
    main()
