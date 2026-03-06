#!/usr/bin/env python3
"""Experiment P1-lite: spatial occupancy cascade in scale space.

This script operationalizes a spatial (not temporal) closure test:

- Define scale coordinate via tile size `l` on MRMS fields.
- Build tile-level observables for `l -> 2l` transitions.
- Construct a local lambda proxy from non-commuting coarse-graining paths.
- Evaluate closure gain from adding lambda proxy under leave-one-event-out CV.

Primary target (default):
- coarse connected-component density of active precipitation mask.

Core outputs:
- summary_metrics.csv
- spatial_tile_dataset.csv
- oof_predictions.csv
- fold_metrics.csv
- permutation_metrics.csv
- spatial_tile_scan.csv
- lambda/comm/gain map PNGs per scale
- report.md
"""

from __future__ import annotations

import argparse
import gzip
import math
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import xarray as xr


EPS = 1e-12
LABEL_STRUCTURE = np.ones((3, 3), dtype=np.int8)


def _to_lon180(lon: np.ndarray) -> np.ndarray:
    x = np.asarray(lon, dtype=float)
    return np.where(x > 180.0, x - 360.0, x)


def _parse_utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    if len(y) < 2:
        return float("nan")
    if float(np.nanstd(y)) < 1e-12:
        return float("nan")
    try:
        return float(r2_score(y, y_pred))
    except ValueError:
        return float("nan")


def _group_zscore(values: pd.Series) -> pd.Series:
    x = values.to_numpy(dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    out = np.zeros_like(x, dtype=float)
    if np.isfinite(sd) and sd > 1e-12:
        out = (x - mu) / sd
    return pd.Series(out, index=values.index)


def _read_mrms_grib(path_gz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with gzip.open(path_gz, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        ds = xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        try:
            if len(ds.data_vars) == 0:
                raise ValueError(f"No data variables in MRMS file: {path_gz}")

            var_name = next(iter(ds.data_vars))
            arr = np.asarray(ds[var_name].values, dtype=np.float32)

            if "latitude" in ds:
                lat_raw = np.asarray(ds["latitude"].values, dtype=float)
            else:
                lat_raw = np.arange(arr.shape[0], dtype=float)

            if "longitude" in ds:
                lon_raw = np.asarray(ds["longitude"].values, dtype=float)
            else:
                lon_raw = np.arange(arr.shape[1], dtype=float)
        finally:
            ds.close()
    finally:
        tmp_path.unlink(missing_ok=True)

    if lat_raw.ndim == 2:
        lat = lat_raw[:, 0]
    elif lat_raw.ndim == 1:
        lat = lat_raw
    else:
        lat = np.arange(arr.shape[0], dtype=float)

    if lon_raw.ndim == 2:
        lon = lon_raw[0, :]
    elif lon_raw.ndim == 1:
        lon = lon_raw
    else:
        lon = np.arange(arr.shape[1], dtype=float)

    lon = _to_lon180(lon)

    if len(lat) != arr.shape[0]:
        lat = np.linspace(0.0, float(arr.shape[0] - 1), arr.shape[0], dtype=float)
    if len(lon) != arr.shape[1]:
        lon = np.linspace(0.0, float(arr.shape[1] - 1), arr.shape[1], dtype=float)

    return arr, lat, lon


def _count_components(mask: np.ndarray) -> int:
    if mask.size == 0 or int(np.sum(mask)) == 0:
        return 0
    _, ncomp = ndimage.label(mask, structure=LABEL_STRUCTURE)
    return int(ncomp)


def _tile_rows_for_scale(
    *,
    arr: np.ndarray,
    valid: np.ndarray,
    active: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    event_id: str,
    ts: pd.Timestamp,
    scale_l: int,
    min_valid_frac: float,
) -> list[dict[str, float | int | str]]:
    l = int(scale_l)
    l2 = int(2 * l)

    ny, nx = arr.shape
    ny2 = (ny // l2) * l2
    nx2 = (nx // l2) * l2
    if ny2 < l2 or nx2 < l2:
        return []

    rows: list[dict[str, float | int | str]] = []
    hour = int(ts.hour)
    hour_sin = float(math.sin(2.0 * math.pi * hour / 24.0))
    hour_cos = float(math.cos(2.0 * math.pi * hour / 24.0))

    min_coarse_valid = float(min_valid_frac * l2 * l2)
    min_fine_valid = float(min_valid_frac * l * l)

    for by in range(0, ny2, l2):
        for bx in range(0, nx2, l2):
            coarse_valid = valid[by : by + l2, bx : bx + l2]
            area_coarse = float(np.sum(coarse_valid))
            if area_coarse < min_coarse_valid:
                continue

            coarse_active = active[by : by + l2, bx : bx + l2] & coarse_valid
            coarse_vals = arr[by : by + l2, bx : bx + l2][coarse_valid]
            coarse_occ = float(np.sum(coarse_active) / max(area_coarse, 1.0))
            coarse_density = float(_count_components(coarse_active) / max(area_coarse, 1.0))
            coarse_rate_mean = float(np.nanmean(coarse_vals)) if coarse_vals.size > 0 else np.nan

            fine_density: list[float] = []
            fine_occ: list[float] = []
            fine_rate: list[float] = []
            ok = True
            for dy in (0, l):
                for dx in (0, l):
                    sy = by + dy
                    sx = bx + dx
                    sub_valid = valid[sy : sy + l, sx : sx + l]
                    area_sub = float(np.sum(sub_valid))
                    if area_sub < min_fine_valid:
                        ok = False
                        break

                    sub_active = active[sy : sy + l, sx : sx + l] & sub_valid
                    sub_vals = arr[sy : sy + l, sx : sx + l][sub_valid]

                    fine_density.append(float(_count_components(sub_active) / max(area_sub, 1.0)))
                    fine_occ.append(float(np.sum(sub_active) / max(area_sub, 1.0)))
                    fine_rate.append(float(np.nanmean(sub_vals)) if sub_vals.size > 0 else np.nan)
                if not ok:
                    break

            if not ok:
                continue

            fine_density_mean = float(np.mean(fine_density))
            fine_density_std = float(np.std(fine_density))
            fine_occ_mean = float(np.mean(fine_occ))
            fine_occ_std = float(np.std(fine_occ))
            fine_rate_mean = float(np.nanmean(np.asarray(fine_rate, dtype=float)))
            fine_rate_std = float(np.nanstd(np.asarray(fine_rate, dtype=float)))

            # Non-commuting proxy:
            # coarse(feature_density) vs average(feature_density over fine tiles)
            comm_defect = float(abs(fine_density_mean - coarse_density))
            lambda_local_raw = float(comm_defect / (abs(fine_density_mean) + abs(coarse_density) + EPS))

            cy = by + (l2 // 2)
            cx = bx + (l2 // 2)
            tile_iy = by // l2
            tile_ix = bx // l2

            rows.append(
                {
                    "event_id": event_id,
                    "mrms_obs_time_utc": ts.isoformat().replace("+00:00", "Z"),
                    "hour_utc": float(hour),
                    "hour_sin": hour_sin,
                    "hour_cos": hour_cos,
                    "scale_l": float(l),
                    "scale_2l": float(l2),
                    "tile_iy": float(tile_iy),
                    "tile_ix": float(tile_ix),
                    "lat_center": float(lat[cy]) if 0 <= cy < len(lat) else np.nan,
                    "lon_center": float(lon[cx]) if 0 <= cx < len(lon) else np.nan,
                    "fine_density_mean": fine_density_mean,
                    "fine_density_std": fine_density_std,
                    "fine_occ_mean": fine_occ_mean,
                    "fine_occ_std": fine_occ_std,
                    "fine_rate_mean": fine_rate_mean,
                    "fine_rate_std": fine_rate_std,
                    "target_density_coarse": coarse_density,
                    "target_occ_coarse": coarse_occ,
                    "target_rate_mean_coarse": coarse_rate_mean,
                    "comm_defect": comm_defect,
                    "lambda_local_raw": lambda_local_raw,
                }
            )

    return rows


def _build_tile_dataset(
    *,
    panel_df: pd.DataFrame,
    scales_cells: Sequence[int],
    mrms_downsample: int,
    mrms_threshold: float,
    min_valid_frac: float,
    max_rows: int,
) -> pd.DataFrame:
    work = panel_df.sort_values(["event_id", "mrms_obs_time_utc"]).reset_index(drop=True)
    if max_rows > 0:
        work = work.head(max_rows).copy()

    rows: list[dict[str, float | int | str]] = []
    n_rows = len(work)

    for i, (_, r) in enumerate(work.iterrows(), start=1):
        event_id = str(r["event_id"])
        ts = _parse_utc(str(r["mrms_obs_time_utc"]))

        print(f"[extract] row {i}/{n_rows} event={event_id} time={ts.isoformat()}", flush=True)

        arr, lat, lon = _read_mrms_grib(Path(str(r["mrms_local_path"])))

        step = max(1, int(mrms_downsample))
        arr_d = np.asarray(arr[::step, ::step], dtype=np.float32)
        lat_d = np.asarray(lat[::step], dtype=float)
        lon_d = np.asarray(lon[::step], dtype=float)

        valid = np.isfinite(arr_d) & (arr_d >= 0.0)
        active = valid & (arr_d >= float(mrms_threshold))

        for scale_l in scales_cells:
            rows.extend(
                _tile_rows_for_scale(
                    arr=arr_d,
                    valid=valid,
                    active=active,
                    lat=lat_d,
                    lon=lon_d,
                    event_id=event_id,
                    ts=ts,
                    scale_l=int(scale_l),
                    min_valid_frac=float(min_valid_frac),
                )
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No tile rows extracted. Relax scale/downsample/valid thresholds.")

    max_scale = float(np.max(df["scale_l"].to_numpy(dtype=float)))
    df["mu_log2"] = np.log2(max_scale / np.maximum(df["scale_l"].to_numpy(dtype=float), 1.0))

    df["lambda_local"] = (
        df.groupby(["event_id", "scale_l"], sort=False)["lambda_local_raw"].transform(_group_zscore).astype(float)
    )

    return df


def _impute_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_tr = np.where(np.isfinite(x_train), x_train, med)
    x_te = np.where(np.isfinite(x_test), x_test, med)
    return x_tr, x_te


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    x_tr, x_te = _impute_train_test(x_train, x_test)
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    model = Ridge(alpha=float(alpha))
    model.fit(x_tr_s, y_train)
    return np.asarray(model.predict(x_te_s), dtype=float)


def _evaluate_logo(
    *,
    df: pd.DataFrame,
    target_col: str,
    baseline_cols: Sequence[str],
    full_cols: Sequence[str],
    ridge_alpha: float,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    req = ["event_id", "mrms_obs_time_utc", "scale_l", "tile_iy", "tile_ix", target_col] + list(full_cols)
    extra = ["target_occ_coarse", "target_rate_mean_coarse", "comm_defect", "lambda_local", "mu_log2"]
    all_cols = list(dict.fromkeys(req + extra))
    data = df[all_cols].copy()

    finite_mask = np.isfinite(data[target_col].to_numpy(dtype=float))
    for c in full_cols:
        finite_mask &= np.isfinite(data[c].to_numpy(dtype=float))
    data = data.loc[finite_mask].reset_index(drop=True)

    if len(data) < 100:
        raise ValueError(f"Too few finite rows for modeling: {len(data)}")

    groups = data["event_id"].astype(str).to_numpy()
    unique_events = np.unique(groups)
    if len(unique_events) < 3:
        raise ValueError(f"Need >=3 events for blocked CV, got {len(unique_events)}")

    y = data[target_col].to_numpy(dtype=float)

    pred_base = np.full(len(data), np.nan, dtype=float)
    pred_full = np.full(len(data), np.nan, dtype=float)
    fold_rows: list[dict[str, float | str]] = []

    logo = LeaveOneGroupOut()
    for fold_id, (tr_idx, te_idx) in enumerate(logo.split(data, y, groups=groups), start=1):
        tr = data.iloc[tr_idx]
        te = data.iloc[te_idx]

        x_base_tr = tr[list(baseline_cols)].to_numpy(dtype=float)
        x_base_te = te[list(baseline_cols)].to_numpy(dtype=float)
        x_full_tr = tr[list(full_cols)].to_numpy(dtype=float)
        x_full_te = te[list(full_cols)].to_numpy(dtype=float)
        y_tr = tr[target_col].to_numpy(dtype=float)
        y_te = te[target_col].to_numpy(dtype=float)

        pb = _ridge_predict(x_base_tr, y_tr, x_base_te, alpha=ridge_alpha)
        pf = _ridge_predict(x_full_tr, y_tr, x_full_te, alpha=ridge_alpha)

        pred_base[te_idx] = pb
        pred_full[te_idx] = pf

        mae_b = float(mean_absolute_error(y_te, pb))
        mae_f = float(mean_absolute_error(y_te, pf))
        r2_b = _safe_r2(y_te, pb)
        r2_f = _safe_r2(y_te, pf)

        fold_rows.append(
            {
                "fold_id": float(fold_id),
                "event_id": str(te["event_id"].iloc[0]),
                "n_test": float(len(te_idx)),
                "mae_baseline": mae_b,
                "mae_full": mae_f,
                "mae_gain": mae_b - mae_f,
                "r2_baseline": r2_b,
                "r2_full": r2_f,
                "r2_gain": (r2_f - r2_b) if np.isfinite(r2_b) and np.isfinite(r2_f) else np.nan,
            }
        )

    oof = data[
        [
            "event_id",
            "mrms_obs_time_utc",
            "scale_l",
            "tile_iy",
            "tile_ix",
            "target_occ_coarse",
            "target_rate_mean_coarse",
            "comm_defect",
            "lambda_local",
            "mu_log2",
            target_col,
        ]
    ].copy()
    oof = oof.rename(columns={target_col: "target_value"})
    oof["pred_baseline"] = pred_base
    oof["pred_full"] = pred_full
    oof["abs_err_baseline"] = np.abs(oof["target_value"] - oof["pred_baseline"])
    oof["abs_err_full"] = np.abs(oof["target_value"] - oof["pred_full"])
    oof["pointwise_gain"] = oof["abs_err_baseline"] - oof["abs_err_full"]

    mae_base = float(mean_absolute_error(oof["target_value"], oof["pred_baseline"]))
    mae_full = float(mean_absolute_error(oof["target_value"], oof["pred_full"]))
    r2_base = _safe_r2(oof["target_value"].to_numpy(dtype=float), oof["pred_baseline"].to_numpy(dtype=float))
    r2_full = _safe_r2(oof["target_value"].to_numpy(dtype=float), oof["pred_full"].to_numpy(dtype=float))

    fold_df = pd.DataFrame(fold_rows)
    event_positive_frac = float((fold_df["mae_gain"] > 0.0).mean()) if len(fold_df) > 0 else np.nan
    min_fold_gain = float(fold_df["mae_gain"].min()) if len(fold_df) > 0 else np.nan

    summary = {
        "mae_baseline": mae_base,
        "mae_full": mae_full,
        "mae_gain": mae_base - mae_full,
        "r2_baseline": r2_base,
        "r2_full": r2_full,
        "r2_gain": (r2_full - r2_base) if np.isfinite(r2_base) and np.isfinite(r2_full) else np.nan,
        "event_positive_frac": event_positive_frac,
        "min_fold_gain": min_fold_gain,
    }
    return summary, oof, fold_df


def _permute_lambda_within_event(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    out = df["lambda_local"].to_numpy(dtype=float).copy()
    groups = df["event_id"].astype(str).to_numpy()
    for ev in np.unique(groups):
        idx = np.where(groups == ev)[0]
        if len(idx) <= 1:
            continue
        out[idx] = out[idx][rng.permutation(len(idx))]
    return out


def _permutation_test(
    *,
    df: pd.DataFrame,
    target_col: str,
    baseline_cols: Sequence[str],
    full_cols: Sequence[str],
    ridge_alpha: float,
    n_perm: int,
    seed: int,
    real_gain: float,
) -> tuple[pd.DataFrame, float, float, float]:
    if n_perm <= 0:
        return pd.DataFrame(columns=["perm_id", "mae_gain_perm", "r2_gain_perm"]), float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    for pid in range(1, n_perm + 1):
        dp = df.copy()
        dp["lambda_local"] = _permute_lambda_within_event(dp, rng=rng)

        s, _, _ = _evaluate_logo(
            df=dp,
            target_col=target_col,
            baseline_cols=baseline_cols,
            full_cols=full_cols,
            ridge_alpha=ridge_alpha,
        )
        rows.append(
            {
                "perm_id": float(pid),
                "mae_gain_perm": float(s["mae_gain"]),
                "r2_gain_perm": float(s["r2_gain"]),
            }
        )

        if pid % 10 == 0 or pid == n_perm:
            print(f"[perm] {pid}/{n_perm}", flush=True)

    perm_df = pd.DataFrame(rows)
    p_value = float((1.0 + float((perm_df["mae_gain_perm"] >= real_gain).sum())) / (1.0 + float(len(perm_df))))
    null_mean = float(np.nanmean(perm_df["mae_gain_perm"].to_numpy(dtype=float)))
    null_q95 = float(np.nanquantile(perm_df["mae_gain_perm"].to_numpy(dtype=float), 0.95))
    return perm_df, p_value, null_mean, null_q95


def _evaluate_single_scale(
    *,
    df_scale: pd.DataFrame,
    target_col: str,
    baseline_cols: Sequence[str],
    full_cols: Sequence[str],
    ridge_alpha: float,
    n_perm: int,
    seed: int,
    active_quantile: float,
    comm_floor: float,
    max_perm_p: float,
) -> tuple[dict[str, float | bool | str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scale_l = int(round(float(df_scale["scale_l"].iloc[0])))

    summary, oof, fold_df = _evaluate_logo(
        df=df_scale,
        target_col=target_col,
        baseline_cols=baseline_cols,
        full_cols=full_cols,
        ridge_alpha=ridge_alpha,
    )

    perm_df, p_value, null_mean, null_q95 = _permutation_test(
        df=df_scale,
        target_col=target_col,
        baseline_cols=baseline_cols,
        full_cols=full_cols,
        ridge_alpha=ridge_alpha,
        n_perm=n_perm,
        seed=seed + scale_l,
        real_gain=float(summary["mae_gain"]),
    )

    occ = oof["target_occ_coarse"].to_numpy(dtype=float)
    rate = oof["target_rate_mean_coarse"].to_numpy(dtype=float)

    active_source = "target_occ_coarse"
    active_threshold = float(np.nanquantile(occ, active_quantile))
    active_mask = occ > active_threshold

    occ_span = float(np.nanmax(occ) - np.nanmin(occ)) if np.any(np.isfinite(occ)) else 0.0
    occ_degenerate = (not np.isfinite(active_threshold)) or (occ_span < 1e-12)
    if occ_degenerate or int(np.sum(active_mask)) == 0 or int(np.sum(~active_mask)) == 0:
        active_source = "target_rate_mean_coarse"
        active_threshold = float(np.nanquantile(rate, active_quantile))
        active_mask = rate > active_threshold

    if int(np.sum(active_mask)) == 0 or int(np.sum(~active_mask)) == 0:
        # Last-resort split: top (1-q) fraction by coarse mean rate.
        active_source = "target_rate_mean_coarse_rank_fallback"
        order = np.argsort(np.nan_to_num(rate, nan=-np.inf))
        k = max(1, int(round((1.0 - active_quantile) * len(order))))
        active_mask = np.zeros(len(order), dtype=bool)
        active_mask[order[-k:]] = True

    point_gain = oof["pointwise_gain"].to_numpy(dtype=float)
    comm = oof["comm_defect"].to_numpy(dtype=float)

    mean_gain_active = float(np.nanmean(point_gain[active_mask])) if int(np.sum(active_mask)) > 0 else np.nan
    mean_gain_calm = float(np.nanmean(point_gain[~active_mask])) if int(np.sum(~active_mask)) > 0 else np.nan
    active_minus_calm = float(mean_gain_active - mean_gain_calm)

    comm_mean = float(np.nanmean(comm))
    comm_active = float(np.nanmean(comm[active_mask])) if int(np.sum(active_mask)) > 0 else np.nan
    comm_calm = float(np.nanmean(comm[~active_mask])) if int(np.sum(~active_mask)) > 0 else np.nan
    comm_active_minus_calm = float(comm_active - comm_calm)

    h1 = bool(np.isfinite(comm_mean) and comm_mean > comm_floor and np.isfinite(comm_active_minus_calm) and comm_active_minus_calm > 0.0)
    h2 = bool(float(summary["mae_gain"]) > 0.0 and (n_perm <= 0 or (np.isfinite(p_value) and p_value <= max_perm_p)))
    h3 = bool(np.isfinite(active_minus_calm) and active_minus_calm > 0.0)
    pass_all = bool(h1 and h2 and h3)

    row: dict[str, float | bool | str] = {
        "scale_l": float(scale_l),
        "n_rows": float(len(oof)),
        "n_events": float(oof["event_id"].nunique()),
        "mae_baseline": float(summary["mae_baseline"]),
        "mae_full": float(summary["mae_full"]),
        "mae_gain": float(summary["mae_gain"]),
        "r2_baseline": float(summary["r2_baseline"]),
        "r2_full": float(summary["r2_full"]),
        "r2_gain": float(summary["r2_gain"]),
        "event_positive_frac": float(summary["event_positive_frac"]),
        "min_fold_gain": float(summary["min_fold_gain"]),
        "comm_defect_mean": comm_mean,
        "comm_active_minus_calm": comm_active_minus_calm,
        "active_source": active_source,
        "active_quantile": float(active_quantile),
        "active_occ_threshold": active_threshold,
        "point_gain_active": mean_gain_active,
        "point_gain_calm": mean_gain_calm,
        "active_minus_calm": active_minus_calm,
        "perm_p_value": p_value,
        "perm_null_mean": null_mean,
        "perm_null_q95": null_q95,
        "H1_space": h1,
        "H2_space": h2,
        "H3_space": h3,
        "PASS_ALL": pass_all,
    }

    oof = oof.copy()
    oof["scale_l"] = float(scale_l)
    oof["is_active"] = active_mask.astype(float)

    fold_df = fold_df.copy()
    fold_df["scale_l"] = float(scale_l)

    if len(perm_df) > 0:
        perm_df = perm_df.copy()
        perm_df["scale_l"] = float(scale_l)

    return row, oof, fold_df, perm_df


def _grid_from_scan(scan_df: pd.DataFrame, value_col: str) -> np.ndarray:
    if len(scan_df) == 0:
        return np.zeros((1, 1), dtype=float)
    n_y = int(scan_df["tile_iy"].max()) + 1
    n_x = int(scan_df["tile_ix"].max()) + 1
    arr = np.full((n_y, n_x), np.nan, dtype=float)
    for _, r in scan_df.iterrows():
        iy = int(r["tile_iy"])
        ix = int(r["tile_ix"])
        arr[iy, ix] = float(r[value_col])
    return arr


def _plot_tile_map(
    *,
    field: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str,
    symmetric: bool,
) -> None:
    arr = np.asarray(field, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.6))

    vmin = float(np.nanquantile(arr, 0.02))
    vmax = float(np.nanquantile(arr, 0.98))
    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        vmin = float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else 0.0
        vmax = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1.0

    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("tile_ix")
    ax.set_ylabel("tile_iy")
    cb = fig.colorbar(im, ax=ax, shrink=0.88)
    cb.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_report(
    *,
    outdir: Path,
    panel_csv: Path,
    target_col: str,
    scales_cells: Sequence[int],
    summary_df: pd.DataFrame,
    n_perm: int,
    mrms_downsample: int,
    mrms_threshold: float,
) -> None:
    lines = [
        "# Experiment P1-lite: Spatial Occupancy Cascade",
        "",
        "## Setup",
        f"- panel: `{panel_csv}`",
        f"- target: `{target_col}`",
        f"- scales (l cells): `{', '.join(str(int(s)) for s in scales_cells)}`",
        f"- MRMS downsample: `{mrms_downsample}`",
        f"- MRMS active threshold: `{mrms_threshold}`",
        f"- permutations per scale: `{n_perm}`",
        "",
        "## Per-scale metrics",
    ]

    scale_rows = summary_df[summary_df["scale_l"] != "ALL"].copy()
    scale_rows = scale_rows.sort_values("scale_l", key=lambda s: s.astype(float))

    for _, r in scale_rows.iterrows():
        lines.extend(
            [
                (
                    f"- l={int(float(r['scale_l']))}: "
                    f"mae_gain={float(r['mae_gain']):.6f}, "
                    f"r2_gain={float(r['r2_gain']):.6f}, "
                    f"perm_p={float(r['perm_p_value']):.6f}, "
                    f"comm_mean={float(r['comm_defect_mean']):.6f}, "
                    f"PASS_ALL={bool(r['PASS_ALL'])}"
                )
            ]
        )

    if len(scale_rows) > 0:
        best_idx = int(np.nanargmax(scale_rows["mae_gain"].to_numpy(dtype=float)))
        best_row = scale_rows.iloc[best_idx]
        lines.extend(
            [
                "",
                "## Scale Sensitivity",
                (
                    f"- best scale by MAE gain: l={int(float(best_row['scale_l']))}, "
                    f"mae_gain={float(best_row['mae_gain']):.6f}, perm_p={float(best_row['perm_p_value']):.6f}"
                ),
                "- use this row as first candidate when lambda detectability is weak.",
            ]
        )

    lines.extend(
        [
            "",
            "## Data guidance",
            "- P1-lite runs on existing aligned MRMS files from realpilot panel.",
            "- P2 should add dense intra-event ABI/GLM sequences (not only one matched frame per hour).",
            "- P3 should add LES/CRM-like data (<1 km) for direct micro-structure thermodynamics.",
            "",
            "## Artifacts",
            "- `summary_metrics.csv`",
            "- `spatial_tile_dataset.csv`",
            "- `spatial_tile_scan.csv`",
            "- `oof_predictions.csv`",
            "- `fold_metrics.csv`",
            "- `permutation_metrics.csv`",
            "- `lambda_local_map_scale_*.png`",
            "- `comm_defect_map_scale_*.png`",
            "- `occupancy_gain_map_scale_*.png`",
        ]
    )

    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    panel_df = pd.read_csv(args.panel_csv)
    required = {"event_id", "mrms_obs_time_utc", "mrms_local_path"}
    missing = required - set(panel_df.columns)
    if missing:
        raise ValueError(f"Panel CSV missing columns: {sorted(missing)}")

    scales_cells = sorted({int(s) for s in args.scales_cells if int(s) >= 2})
    if len(scales_cells) == 0:
        raise ValueError("No valid scales provided.")

    print("[stage] 1/5 build spatial tile dataset", flush=True)
    tile_df = _build_tile_dataset(
        panel_df=panel_df,
        scales_cells=scales_cells,
        mrms_downsample=args.mrms_downsample,
        mrms_threshold=args.mrms_threshold,
        min_valid_frac=args.min_valid_frac,
        max_rows=args.max_rows,
    )
    tile_df.to_csv(outdir / "spatial_tile_dataset.csv", index=False)

    target_col = "target_density_coarse" if args.target == "density" else "target_occ_coarse"

    baseline_cols = [
        "fine_density_mean",
        "fine_density_std",
        "fine_occ_mean",
        "fine_occ_std",
        "fine_rate_mean",
        "fine_rate_std",
        "hour_sin",
        "hour_cos",
    ]
    full_cols = baseline_cols + ["lambda_local"]

    print("[stage] 2/5 evaluate per scale", flush=True)
    summary_rows: list[dict[str, float | bool | str]] = []
    oof_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    perm_parts: list[pd.DataFrame] = []

    for scale_l in scales_cells:
        print(f"[model] scale l={scale_l}", flush=True)
        df_s = tile_df[tile_df["scale_l"].to_numpy(dtype=float) == float(scale_l)].copy().reset_index(drop=True)
        if len(df_s) < 300:
            print(f"[model] skip scale l={scale_l}: too few rows ({len(df_s)})", flush=True)
            continue

        row, oof_df, fold_df, perm_df = _evaluate_single_scale(
            df_scale=df_s,
            target_col=target_col,
            baseline_cols=baseline_cols,
            full_cols=full_cols,
            ridge_alpha=args.ridge_alpha,
            n_perm=args.n_perm,
            seed=args.seed,
            active_quantile=args.active_quantile,
            comm_floor=args.comm_floor,
            max_perm_p=args.max_perm_p,
        )
        summary_rows.append({"scale_l": float(scale_l), **row})
        oof_parts.append(oof_df)
        fold_parts.append(fold_df)
        if len(perm_df) > 0:
            perm_parts.append(perm_df)

    if len(summary_rows) == 0:
        raise RuntimeError("No scales produced valid model results.")

    print("[stage] 3/5 aggregate outputs", flush=True)
    oof_all = pd.concat(oof_parts, ignore_index=True)
    fold_all = pd.concat(fold_parts, ignore_index=True)
    perm_all = pd.concat(perm_parts, ignore_index=True) if len(perm_parts) > 0 else pd.DataFrame()

    overall_mae_base = float(mean_absolute_error(oof_all["target_value"], oof_all["pred_baseline"]))
    overall_mae_full = float(mean_absolute_error(oof_all["target_value"], oof_all["pred_full"]))
    overall_r2_base = _safe_r2(oof_all["target_value"].to_numpy(dtype=float), oof_all["pred_baseline"].to_numpy(dtype=float))
    overall_r2_full = _safe_r2(oof_all["target_value"].to_numpy(dtype=float), oof_all["pred_full"].to_numpy(dtype=float))

    scale_pass = [bool(r["PASS_ALL"]) for r in summary_rows]
    overall_row: dict[str, float | bool | str] = {
        "scale_l": "ALL",
        "n_rows": float(len(oof_all)),
        "n_events": float(oof_all["event_id"].nunique()),
        "mae_baseline": overall_mae_base,
        "mae_full": overall_mae_full,
        "mae_gain": overall_mae_base - overall_mae_full,
        "r2_baseline": overall_r2_base,
        "r2_full": overall_r2_full,
        "r2_gain": (overall_r2_full - overall_r2_base) if np.isfinite(overall_r2_base) and np.isfinite(overall_r2_full) else np.nan,
        "event_positive_frac": float(np.nanmean(fold_all["mae_gain"].to_numpy(dtype=float) > 0.0)),
        "min_fold_gain": float(np.nanmin(fold_all["mae_gain"].to_numpy(dtype=float))),
        "comm_defect_mean": float(np.nanmean(oof_all["comm_defect"].to_numpy(dtype=float))),
        "comm_active_minus_calm": np.nan,
        "active_quantile": float(args.active_quantile),
        "active_occ_threshold": np.nan,
        "point_gain_active": np.nan,
        "point_gain_calm": np.nan,
        "active_minus_calm": np.nan,
        "perm_p_value": np.nan,
        "perm_null_mean": np.nan,
        "perm_null_q95": np.nan,
        "H1_space": bool(np.all([bool(r["H1_space"]) for r in summary_rows])),
        "H2_space": bool(np.all([bool(r["H2_space"]) for r in summary_rows])),
        "H3_space": bool(np.all([bool(r["H3_space"]) for r in summary_rows])),
        "PASS_ALL": bool(np.all(scale_pass)),
    }

    summary_df = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame([overall_row])], ignore_index=True)

    oof_all.to_csv(outdir / "oof_predictions.csv", index=False)
    fold_all.to_csv(outdir / "fold_metrics.csv", index=False)
    if len(perm_all) > 0:
        perm_all.to_csv(outdir / "permutation_metrics.csv", index=False)
    else:
        pd.DataFrame(columns=["perm_id", "mae_gain_perm", "r2_gain_perm", "scale_l"]).to_csv(
            outdir / "permutation_metrics.csv", index=False
        )
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)

    print("[stage] 4/5 build spatial scan + maps", flush=True)
    scan_df = (
        oof_all.groupby(["scale_l", "tile_iy", "tile_ix"], as_index=False)
        .agg(
            lambda_local_mean=("lambda_local", "mean"),
            comm_defect_mean=("comm_defect", "mean"),
            occupancy_gain_mean=("pointwise_gain", "mean"),
            n=("pointwise_gain", "size"),
        )
        .sort_values(["scale_l", "tile_iy", "tile_ix"])
        .reset_index(drop=True)
    )
    scan_df.to_csv(outdir / "spatial_tile_scan.csv", index=False)

    for scale_l in sorted(scan_df["scale_l"].unique()):
        d = scan_df[scan_df["scale_l"] == scale_l].copy()

        lambda_map = _grid_from_scan(d, "lambda_local_mean")
        comm_map = _grid_from_scan(d, "comm_defect_mean")
        gain_map = _grid_from_scan(d, "occupancy_gain_mean")

        sname = str(int(round(float(scale_l))))
        _plot_tile_map(
            field=lambda_map,
            title=f"Lambda local map (l={sname})",
            cbar_label="mean(lambda_local)",
            out_path=outdir / f"lambda_local_map_scale_{sname}.png",
            cmap="viridis",
            symmetric=False,
        )
        _plot_tile_map(
            field=comm_map,
            title=f"Commutation defect map (l={sname})",
            cbar_label="mean(comm_defect)",
            out_path=outdir / f"comm_defect_map_scale_{sname}.png",
            cmap="magma",
            symmetric=False,
        )
        _plot_tile_map(
            field=gain_map,
            title=f"Occupancy gain map (l={sname})",
            cbar_label="mean(pointwise_gain)",
            out_path=outdir / f"occupancy_gain_map_scale_{sname}.png",
            cmap="RdBu_r",
            symmetric=True,
        )

    print("[stage] 5/5 write report", flush=True)
    _write_report(
        outdir=outdir,
        panel_csv=args.panel_csv,
        target_col=target_col,
        scales_cells=scales_cells,
        summary_df=summary_df,
        n_perm=args.n_perm,
        mrms_downsample=args.mrms_downsample,
        mrms_threshold=args.mrms_threshold,
    )

    print("Experiment P1-lite complete.", flush=True)
    print(f"Output: {outdir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv"),
        help="Aligned MRMS+ABI+GLM panel; this script uses MRMS paths from this panel.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_P1_spatial_occupancy_cascade"),
    )
    p.add_argument("--target", choices=["density", "occupancy"], default="density")

    p.add_argument("--scales-cells", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--mrms-downsample", type=int, default=16)
    p.add_argument("--mrms-threshold", type=float, default=5.0)
    p.add_argument("--min-valid-frac", type=float, default=0.90)
    p.add_argument("--max-rows", type=int, default=0, help="If >0, limit panel rows for quick runs.")

    p.add_argument("--ridge-alpha", type=float, default=3.0)
    p.add_argument("--n-perm", type=int, default=49)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--active-quantile", type=float, default=0.67)
    p.add_argument("--comm-floor", type=float, default=1e-4)
    p.add_argument("--max-perm-p", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
