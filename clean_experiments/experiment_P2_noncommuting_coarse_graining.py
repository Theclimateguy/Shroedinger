#!/usr/bin/env python3
"""Experiment P2: noncommuting coarse-graining with scale-occupancy density matrices.

Core objects:
  Delta_comm(x,l,t) = || Pi_{l->2l} Phi - Phi Pi_{l->2l} ||
  rho_occ(x,l,t) from occupancy at scales {l, 2l}
  lambda_local ~ Re Tr(F_comm rho_occ)

Implemented on MRMS fields from realpilot panel:
- Pi: block-mean coarse-graining from l-pixels to 2x2 coarse lattice.
- Phi operators:
  1) threshold occupancy nonlinearity
  2) square nonlinearity
  3) log1p nonlinearity
  4) gradient-magnitude nonlinearity

For each 2l tile and time, we compute operator-wise commutation defects, build
rho_occ, and then compute lambda_local from a commutator-based F operator.
Then evaluate closure for coarse structure density with leave-one-event-out CV.
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
            var = next(iter(ds.data_vars))
            arr = np.asarray(ds[var].values, dtype=np.float32)

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


def _pool_mean(arr: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(arr, dtype=float)
    ny, nx = x.shape
    if (ny % block) != 0 or (nx % block) != 0:
        raise ValueError("Array shape must be divisible by block size.")

    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0)

    nyb = ny // block
    nxb = nx // block

    x_sum = x0.reshape(nyb, block, nxb, block).sum(axis=(1, 3))
    x_cnt = valid.reshape(nyb, block, nxb, block).sum(axis=(1, 3)).astype(float)

    x_mean = x_sum / np.maximum(x_cnt, 1.0)
    x_mean[x_cnt <= 0.0] = np.nan
    return x_mean.astype(float), x_cnt.astype(float)


def _op_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a)
    out[m] = (a[m] >= float(threshold)).astype(float)
    return out


def _op_square(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a)
    out[m] = a[m] * a[m]
    return out


def _op_log1p(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a)
    out[m] = np.log1p(np.maximum(a[m], 0.0))
    return out


def _op_gradmag(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a)
    if int(np.sum(m)) == 0:
        return out

    fill = float(np.nanmedian(a[m]))
    af = np.where(m, a, fill)
    gy, gx = np.gradient(af)
    g = np.hypot(gx, gy)
    out[m] = g[m]
    return out


def _operator_comm_defects(
    *,
    parent_arr: np.ndarray,
    parent_valid: np.ndarray,
    l: int,
    threshold: float,
    min_valid_frac: float,
) -> dict[str, float]:
    f = np.where(parent_valid, parent_arr, np.nan)
    coarse_field, coarse_cnt = _pool_mean(f, block=l)
    coarse_valid = coarse_cnt >= float(min_valid_frac * l * l)
    coarse_field = np.where(coarse_valid, coarse_field, np.nan)

    ops = {
        "occ": lambda z: _op_threshold(z, threshold=threshold),
        "sq": _op_square,
        "log": _op_log1p,
        "grad": _op_gradmag,
    }

    out: dict[str, float] = {}
    for name, op in ops.items():
        # Path A: Pi o Phi
        phi_fine = op(f)
        a_map, _ = _pool_mean(phi_fine, block=l)

        # Path B: Phi o Pi
        b_map = op(coarse_field)

        d = np.abs(a_map - b_map)
        if np.any(np.isfinite(d)):
            delta = float(np.nanmean(d))
        else:
            delta = float("nan")

        out[f"delta_{name}"] = delta

    vals = np.array([out["delta_occ"], out["delta_sq"], out["delta_log"], out["delta_grad"]], dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    out["delta_norm"] = float(np.sqrt(np.sum(vals * vals)))
    return out


def _density_matrix_lambda(
    *,
    fine_scale_density: float,
    coarse_scale_density: float,
    defects: dict[str, float],
    lambda_weights: np.ndarray,
    lambda_scale_power: float,
    decoherence_alpha: float,
    scale_l: int,
) -> dict[str, float]:
    # Scale-density populations on the scale pair {l, 2l}.
    n_l = float(max(0.0, fine_scale_density))
    n_2l = float(max(0.0, coarse_scale_density))
    z = n_l + n_2l
    if z < EPS:
        p_l = 0.5
        p_2l = 0.5
    else:
        p_l = n_l / z
        p_2l = n_2l / z

    # Decoherence from empirical noncommutativity.
    eta = float(np.exp(-max(0.0, decoherence_alpha) * abs(float(defects["delta_norm"]))))
    eta = float(np.clip(eta, 0.0, 1.0))

    rho = np.array(
        [
            [p_l, eta * math.sqrt(max(p_l * p_2l, 0.0))],
            [eta * math.sqrt(max(p_l * p_2l, 0.0)), p_2l],
        ],
        dtype=np.complex128,
    )

    # Two generators in theory-like form:
    # A = a * sigma_x, B = b * sigma_y, F = i[A,B].
    a = float(lambda_weights[0] * defects["delta_occ"] + lambda_weights[1] * defects["delta_sq"])
    b = float(lambda_weights[2] * defects["delta_log"] + lambda_weights[3] * defects["delta_grad"])
    if lambda_scale_power != 0.0:
        scale_fac = float(max(scale_l, 1) ** float(lambda_scale_power))
        a = float(a / scale_fac)
        b = float(b / scale_fac)

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)

    A = a * sigma_x
    B = b * sigma_y
    comm = A @ B - B @ A
    F = 1j * comm
    F = 0.5 * (F + F.conj().T)

    lambda_dm = float(np.real(np.trace(F @ rho)))
    comm_norm_theory = float(np.linalg.norm(comm, ord="fro"))
    purity = float(np.real(np.trace(rho @ rho)))

    return {
        "rho_11": float(np.real(rho[0, 0])),
        "rho_22": float(np.real(rho[1, 1])),
        "rho_12": float(np.real(rho[0, 1])),
        "rho_purity": purity,
        "decoherence_eta": eta,
        "gen_a": a,
        "gen_b": b,
        "comm_norm_theory": comm_norm_theory,
        "lambda_dm": lambda_dm,
    }


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
    threshold: float,
    min_valid_frac: float,
    lambda_weights: np.ndarray,
    lambda_scale_power: float,
    decoherence_alpha: float,
) -> list[dict[str, float | str]]:
    l = int(scale_l)
    l2 = int(2 * l)

    ny, nx = arr.shape
    ny2 = (ny // l2) * l2
    nx2 = (nx // l2) * l2
    if ny2 < l2 or nx2 < l2:
        return []

    hour = int(ts.hour)
    hour_sin = float(math.sin(2.0 * math.pi * hour / 24.0))
    hour_cos = float(math.cos(2.0 * math.pi * hour / 24.0))

    rows: list[dict[str, float | str]] = []
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

            target_density = float(_count_components(coarse_active) / max(area_coarse, 1.0))
            target_occ = float(np.sum(coarse_active) / max(area_coarse, 1.0))
            target_rate = float(np.nanmean(coarse_vals)) if coarse_vals.size > 0 else np.nan

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

            parent_arr = np.asarray(arr[by : by + l2, bx : bx + l2], dtype=float)
            parent_valid = np.asarray(valid[by : by + l2, bx : bx + l2], dtype=bool)
            defects = _operator_comm_defects(
                parent_arr=parent_arr,
                parent_valid=parent_valid,
                l=l,
                threshold=threshold,
                min_valid_frac=min_valid_frac,
            )

            dm = _density_matrix_lambda(
                fine_scale_density=float(np.mean(fine_density)),
                coarse_scale_density=target_density,
                defects=defects,
                lambda_weights=lambda_weights,
                lambda_scale_power=lambda_scale_power,
                decoherence_alpha=decoherence_alpha,
                scale_l=l,
            )
            lambda_raw = float(dm["lambda_dm"])

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
                    "fine_density_mean": float(np.mean(fine_density)),
                    "fine_density_std": float(np.std(fine_density)),
                    "fine_occ_mean": float(np.mean(fine_occ)),
                    "fine_occ_std": float(np.std(fine_occ)),
                    "fine_rate_mean": float(np.nanmean(np.asarray(fine_rate, dtype=float))),
                    "fine_rate_std": float(np.nanstd(np.asarray(fine_rate, dtype=float))),
                    "target_density_coarse": target_density,
                    "target_occ_coarse": target_occ,
                    "target_rate_mean_coarse": target_rate,
                    "delta_occ_raw": float(defects["delta_occ"]),
                    "delta_sq_raw": float(defects["delta_sq"]),
                    "delta_log_raw": float(defects["delta_log"]),
                    "delta_grad_raw": float(defects["delta_grad"]),
                    "comm_defect_operator_raw": float(defects["delta_norm"]),
                    "rho_11_raw": float(dm["rho_11"]),
                    "rho_22_raw": float(dm["rho_22"]),
                    "rho_12_raw": float(dm["rho_12"]),
                    "rho_purity_raw": float(dm["rho_purity"]),
                    "decoherence_eta_raw": float(dm["decoherence_eta"]),
                    "gen_a_raw": float(dm["gen_a"]),
                    "gen_b_raw": float(dm["gen_b"]),
                    "comm_defect_raw": float(dm["comm_norm_theory"]),
                    "lambda_local_raw": lambda_raw,
                }
            )

    return rows


def _build_tile_dataset(
    *,
    panel_df: pd.DataFrame,
    scales_cells: Sequence[int],
    mrms_downsample: int,
    threshold: float,
    min_valid_frac: float,
    lambda_weights: np.ndarray,
    lambda_scale_power: float,
    decoherence_alpha: float,
    max_rows: int,
) -> pd.DataFrame:
    work = panel_df.sort_values(["event_id", "mrms_obs_time_utc"]).reset_index(drop=True)
    if max_rows > 0:
        work = work.head(max_rows).copy()

    rows: list[dict[str, float | str]] = []
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
        active = valid & (arr_d >= float(threshold))

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
                    threshold=threshold,
                    min_valid_frac=min_valid_frac,
                    lambda_weights=lambda_weights,
                    lambda_scale_power=lambda_scale_power,
                    decoherence_alpha=decoherence_alpha,
                )
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No tile rows extracted; relax geometry/downsample thresholds.")

    max_scale = float(np.max(df["scale_l"].to_numpy(dtype=float)))
    df["mu_log2"] = np.log2(max_scale / np.maximum(df["scale_l"].to_numpy(dtype=float), 1.0))
    return _apply_theory_bridge(
        df,
        lambda_weights=lambda_weights,
        lambda_scale_power=lambda_scale_power,
        decoherence_alpha=decoherence_alpha,
    )


def _apply_theory_bridge(
    df_in: pd.DataFrame,
    *,
    lambda_weights: np.ndarray,
    lambda_scale_power: float,
    decoherence_alpha: float,
) -> pd.DataFrame:
    df = df_in.copy()

    w = np.asarray(lambda_weights, dtype=float)
    if w.shape != (4,):
        raise ValueError("lambda_weights must have shape (4,) for [occ, sq, log, grad].")

    d_occ = np.asarray(df["delta_occ_raw"], dtype=float)
    d_sq = np.asarray(df["delta_sq_raw"], dtype=float)
    d_log = np.asarray(df["delta_log_raw"], dtype=float)
    d_grad = np.asarray(df["delta_grad_raw"], dtype=float)
    l = np.maximum(np.asarray(df["scale_l"], dtype=float), 1.0)

    op_norm = np.asarray(df["comm_defect_operator_raw"], dtype=float)
    op_norm = np.where(np.isfinite(op_norm), op_norm, 0.0)

    n_l = np.maximum(np.asarray(df["fine_density_mean"], dtype=float), 0.0)
    n_2l = np.maximum(np.asarray(df["target_density_coarse"], dtype=float), 0.0)
    z = n_l + n_2l
    p_l = np.where(z < EPS, 0.5, n_l / np.maximum(z, EPS))
    p_2l = np.where(z < EPS, 0.5, n_2l / np.maximum(z, EPS))

    eta = np.exp(-max(0.0, float(decoherence_alpha)) * np.abs(op_norm))
    eta = np.clip(eta, 0.0, 1.0)

    rho_12 = eta * np.sqrt(np.maximum(p_l * p_2l, 0.0))
    rho_purity = p_l * p_l + p_2l * p_2l + 2.0 * rho_12 * rho_12

    a = w[0] * d_occ + w[1] * d_sq
    b = w[2] * d_log + w[3] * d_grad
    if float(lambda_scale_power) != 0.0:
        sf = l ** float(lambda_scale_power)
        a = a / sf
        b = b / sf

    comm_norm_theory = np.sqrt(8.0) * np.abs(a * b)
    lambda_dm = 2.0 * a * b * (p_2l - p_l)

    df["rho_11_raw"] = p_l.astype(float)
    df["rho_22_raw"] = p_2l.astype(float)
    df["rho_12_raw"] = rho_12.astype(float)
    df["rho_purity_raw"] = rho_purity.astype(float)
    df["decoherence_eta_raw"] = eta.astype(float)
    df["gen_a_raw"] = a.astype(float)
    df["gen_b_raw"] = b.astype(float)
    df["comm_defect_raw"] = comm_norm_theory.astype(float)
    df["lambda_local_raw"] = lambda_dm.astype(float)

    z_cols = {
        "delta_occ_raw": "delta_occ",
        "delta_sq_raw": "delta_sq",
        "delta_log_raw": "delta_log",
        "delta_grad_raw": "delta_grad",
        "comm_defect_operator_raw": "comm_defect_operator",
        "comm_defect_raw": "comm_defect_z",
        "lambda_local_raw": "lambda_local",
    }
    for raw_col, z_col in z_cols.items():
        df[z_col] = (
            df.groupby(["event_id", "scale_l"], sort=False)[raw_col]
            .transform(_group_zscore)
            .astype(float)
        )

    # Raw commutator norm is used for physics-facing diagnostics.
    df["comm_defect"] = df["comm_defect_raw"].astype(float)
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
    extra = [
        "target_occ_coarse",
        "target_rate_mean_coarse",
        "delta_occ",
        "delta_sq",
        "delta_log",
        "delta_grad",
        "comm_defect_operator",
        "comm_defect",
        "lambda_local",
        "rho_11_raw",
        "rho_22_raw",
        "rho_12_raw",
        "rho_purity_raw",
        "decoherence_eta_raw",
        "gen_a_raw",
        "gen_b_raw",
        "mu_log2",
    ]
    cols = list(dict.fromkeys(req + extra))

    data = df[cols].copy()

    finite_mask = np.isfinite(data[target_col].to_numpy(dtype=float))
    for c in full_cols:
        finite_mask &= np.isfinite(data[c].to_numpy(dtype=float))
    data = data.loc[finite_mask].reset_index(drop=True)

    if len(data) < 100:
        raise ValueError(f"Too few finite rows for modeling: {len(data)}")

    groups = data["event_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < 3:
        raise ValueError("Need at least 3 events for LOEO CV.")

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
            "delta_occ",
            "delta_sq",
            "delta_log",
            "delta_grad",
            "comm_defect_operator",
            "comm_defect",
            "lambda_local",
            "rho_11_raw",
            "rho_22_raw",
            "rho_12_raw",
            "rho_purity_raw",
            "decoherence_eta_raw",
            "gen_a_raw",
            "gen_b_raw",
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
    summary = {
        "mae_baseline": mae_base,
        "mae_full": mae_full,
        "mae_gain": mae_base - mae_full,
        "r2_baseline": r2_base,
        "r2_full": r2_full,
        "r2_gain": (r2_full - r2_base) if np.isfinite(r2_base) and np.isfinite(r2_full) else np.nan,
        "event_positive_frac": float((fold_df["mae_gain"] > 0.0).mean()) if len(fold_df) > 0 else np.nan,
        "min_fold_gain": float(fold_df["mae_gain"].min()) if len(fold_df) > 0 else np.nan,
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
        empty = pd.DataFrame(columns=["perm_id", "mae_gain_perm", "r2_gain_perm"])
        return empty, float("nan"), float("nan"), float("nan")

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


def _active_mask_with_fallback(
    *,
    target_occ: np.ndarray,
    target_rate: np.ndarray,
    active_quantile: float,
) -> tuple[np.ndarray, str, float]:
    occ = np.asarray(target_occ, dtype=float)
    rate = np.asarray(target_rate, dtype=float)

    source = "target_occ_coarse"
    thr = float(np.nanquantile(occ, active_quantile))
    mask = occ > thr

    occ_span = float(np.nanmax(occ) - np.nanmin(occ)) if np.any(np.isfinite(occ)) else 0.0
    if (not np.isfinite(thr)) or (occ_span < 1e-12) or int(np.sum(mask)) == 0 or int(np.sum(~mask)) == 0:
        source = "target_rate_mean_coarse"
        thr = float(np.nanquantile(rate, active_quantile))
        mask = rate > thr

    if int(np.sum(mask)) == 0 or int(np.sum(~mask)) == 0:
        source = "target_rate_mean_coarse_rank_fallback"
        order = np.argsort(np.nan_to_num(rate, nan=-np.inf))
        k = max(1, int(round((1.0 - active_quantile) * len(order))))
        mask = np.zeros(len(order), dtype=bool)
        mask[order[-k:]] = True

    return mask, source, thr


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

    active_mask, active_source, active_thr = _active_mask_with_fallback(
        target_occ=oof["target_occ_coarse"].to_numpy(dtype=float),
        target_rate=oof["target_rate_mean_coarse"].to_numpy(dtype=float),
        active_quantile=active_quantile,
    )

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
        "active_threshold": active_thr,
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
        arr[int(r["tile_iy"]), int(r["tile_ix"])] = float(r[value_col])
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
    fig, ax = plt.subplots(figsize=(7.2, 5.8))

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


def _operator_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for s, g in df.groupby("scale_l", sort=True):
        x_occ = g["delta_occ_raw"].to_numpy(dtype=float)
        x_sq = g["delta_sq_raw"].to_numpy(dtype=float)
        x_log = g["delta_log_raw"].to_numpy(dtype=float)
        x_grad = g["delta_grad_raw"].to_numpy(dtype=float)
        x_norm_op = g["comm_defect_operator_raw"].to_numpy(dtype=float)
        x_norm_dm = g["comm_defect_raw"].to_numpy(dtype=float)
        x_purity = g["rho_purity_raw"].to_numpy(dtype=float)
        x_eta = g["decoherence_eta_raw"].to_numpy(dtype=float)
        x_lam = g["lambda_local_raw"].to_numpy(dtype=float)

        rows.append(
            {
                "scale_l": float(s),
                "n_rows": float(len(g)),
                "mean_delta_occ": float(np.nanmean(x_occ)),
                "mean_delta_sq": float(np.nanmean(x_sq)),
                "mean_delta_log": float(np.nanmean(x_log)),
                "mean_delta_grad": float(np.nanmean(x_grad)),
                "mean_delta_norm_operator": float(np.nanmean(x_norm_op)),
                "mean_delta_norm_theory": float(np.nanmean(x_norm_dm)),
                "q90_delta_norm_theory": float(np.nanquantile(x_norm_dm, 0.90)),
                "q99_delta_norm_theory": float(np.nanquantile(x_norm_dm, 0.99)),
                "mean_rho_purity": float(np.nanmean(x_purity)),
                "mean_decoherence_eta": float(np.nanmean(x_eta)),
                "mean_lambda_dm": float(np.nanmean(x_lam)),
            }
        )

    return pd.DataFrame(rows)


def _write_report(
    *,
    outdir: Path,
    panel_csv: Path,
    target_col: str,
    scales_cells: Sequence[int],
    summary_df: pd.DataFrame,
    op_summary_df: pd.DataFrame,
    threshold: float,
    downsample: int,
    n_perm: int,
    lambda_weights: np.ndarray,
    lambda_scale_power: float,
    decoherence_alpha: float,
) -> None:
    lines = [
        "# Experiment P2: Noncommuting Coarse-Graining",
        "",
        "## Setup",
        f"- panel: `{panel_csv}`",
        f"- target: `{target_col}`",
        f"- scales (l cells): `{', '.join(str(int(s)) for s in scales_cells)}`",
        f"- MRMS downsample: `{downsample}`",
        f"- active threshold: `{threshold}`",
        f"- generator weights [occ,sq,log,grad]: `{lambda_weights.tolist()}`",
        f"- generator scale power: `{lambda_scale_power}`",
        f"- decoherence alpha: `{decoherence_alpha}`",
        f"- permutations per scale: `{n_perm}`",
        "",
        "## Formal object",
        "- Delta_comm(x,l,t) = ||Pi_{l->2l} Phi - Phi Pi_{l->2l}||",
        "- Phi set: threshold, square, log1p, gradient-magnitude",
        "- rho_occ from occupancy populations on scales {l,2l}",
        "- lambda_local = Re Tr(F_comm rho_occ), with F_comm = i[A,B]",
        "",
        "## Per-scale closure metrics",
    ]

    per = summary_df[summary_df["scale_l"] != "ALL"].copy()
    per = per.sort_values("scale_l", key=lambda s: s.astype(float))
    for _, r in per.iterrows():
        lines.append(
            (
                f"- l={int(float(r['scale_l']))}: "
                f"mae_gain={float(r['mae_gain']):.6f}, "
                f"r2_gain={float(r['r2_gain']):.6f}, "
                f"perm_p={float(r['perm_p_value']):.6f}, "
                f"comm_mean={float(r['comm_defect_mean']):.6f}, "
                f"PASS_ALL={bool(r['PASS_ALL'])}"
            )
        )

    lines.extend(["", "## Operator defect summary"])
    for _, r in op_summary_df.sort_values("scale_l").iterrows():
        lines.append(
            (
                f"- l={int(float(r['scale_l']))}: "
                f"mean(delta_occ)={float(r['mean_delta_occ']):.6f}, "
                f"mean(delta_sq)={float(r['mean_delta_sq']):.6f}, "
                f"mean(delta_log)={float(r['mean_delta_log']):.6f}, "
                f"mean(delta_grad)={float(r['mean_delta_grad']):.6f}, "
                f"mean(norm_theory)={float(r['mean_delta_norm_theory']):.6f}, "
                f"q90(norm_theory)={float(r['q90_delta_norm_theory']):.6f}, "
                f"mean(rho_purity)={float(r['mean_rho_purity']):.6f}"
            )
        )

    if len(per) > 0:
        best_idx = int(np.nanargmax(per["mae_gain"].to_numpy(dtype=float)))
        best = per.iloc[best_idx]
        lines.extend(
            [
                "",
                "## Scale priority",
                (
                    f"- best scale by MAE gain: l={int(float(best['scale_l']))}, "
                    f"mae_gain={float(best['mae_gain']):.6f}, perm_p={float(best['perm_p_value']):.6f}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "- `summary_metrics.csv`",
            "- `comm_operator_summary.csv`",
            "- `p2_tile_dataset.csv`",
            "- `spatial_tile_scan.csv`",
            "- `oof_predictions.csv`",
            "- `fold_metrics.csv`",
            "- `permutation_metrics.csv`",
            "- `lambda_local_map_scale_*.png`",
            "- `comm_defect_map_scale_*.png`",
            "- `comm_gain_map_scale_*.png`",
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

    weights = np.asarray(args.lambda_weights, dtype=float)
    if weights.shape != (4,):
        raise ValueError("--lambda-weights must provide exactly 4 values: occ sq log grad")

    print("[stage] 1/5 build P2 tile dataset", flush=True)
    tile_df = _build_tile_dataset(
        panel_df=panel_df,
        scales_cells=scales_cells,
        mrms_downsample=args.mrms_downsample,
        threshold=args.mrms_threshold,
        min_valid_frac=args.min_valid_frac,
        lambda_weights=weights,
        lambda_scale_power=args.lambda_scale_power,
        decoherence_alpha=args.decoherence_alpha,
        max_rows=args.max_rows,
    )
    tile_df.to_csv(outdir / "p2_tile_dataset.csv", index=False)

    op_summary_df = _operator_summary(tile_df)
    op_summary_df.to_csv(outdir / "comm_operator_summary.csv", index=False)

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
        raise RuntimeError("No scales produced valid model outputs.")

    print("[stage] 3/5 aggregate outputs", flush=True)
    oof_all = pd.concat(oof_parts, ignore_index=True)
    fold_all = pd.concat(fold_parts, ignore_index=True)
    perm_all = pd.concat(perm_parts, ignore_index=True) if len(perm_parts) > 0 else pd.DataFrame()

    overall_mae_base = float(mean_absolute_error(oof_all["target_value"], oof_all["pred_baseline"]))
    overall_mae_full = float(mean_absolute_error(oof_all["target_value"], oof_all["pred_full"]))
    overall_r2_base = _safe_r2(oof_all["target_value"].to_numpy(dtype=float), oof_all["pred_baseline"].to_numpy(dtype=float))
    overall_r2_full = _safe_r2(oof_all["target_value"].to_numpy(dtype=float), oof_all["pred_full"].to_numpy(dtype=float))

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
        "active_source": "",
        "active_quantile": float(args.active_quantile),
        "active_threshold": np.nan,
        "point_gain_active": np.nan,
        "point_gain_calm": np.nan,
        "active_minus_calm": np.nan,
        "perm_p_value": np.nan,
        "perm_null_mean": np.nan,
        "perm_null_q95": np.nan,
        "H1_space": bool(np.all([bool(r["H1_space"]) for r in summary_rows])),
        "H2_space": bool(np.all([bool(r["H2_space"]) for r in summary_rows])),
        "H3_space": bool(np.all([bool(r["H3_space"]) for r in summary_rows])),
        "PASS_ALL": bool(np.all([bool(r["PASS_ALL"]) for r in summary_rows])),
    }

    summary_df = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame([overall_row])], ignore_index=True)

    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    oof_all.to_csv(outdir / "oof_predictions.csv", index=False)
    fold_all.to_csv(outdir / "fold_metrics.csv", index=False)
    if len(perm_all) > 0:
        perm_all.to_csv(outdir / "permutation_metrics.csv", index=False)
    else:
        pd.DataFrame(columns=["perm_id", "mae_gain_perm", "r2_gain_perm", "scale_l"]).to_csv(
            outdir / "permutation_metrics.csv", index=False
        )

    print("[stage] 4/5 spatial scan + maps", flush=True)
    scan_df = (
        oof_all.groupby(["scale_l", "tile_iy", "tile_ix"], as_index=False)
        .agg(
            lambda_local_mean=("lambda_local", "mean"),
            comm_defect_mean=("comm_defect", "mean"),
            comm_gain_mean=("pointwise_gain", "mean"),
            delta_occ_mean=("delta_occ", "mean"),
            delta_sq_mean=("delta_sq", "mean"),
            delta_log_mean=("delta_log", "mean"),
            delta_grad_mean=("delta_grad", "mean"),
            n=("pointwise_gain", "size"),
        )
        .sort_values(["scale_l", "tile_iy", "tile_ix"])
        .reset_index(drop=True)
    )
    scan_df.to_csv(outdir / "spatial_tile_scan.csv", index=False)

    for scale_l in sorted(scan_df["scale_l"].unique()):
        d = scan_df[scan_df["scale_l"] == scale_l].copy()
        sid = str(int(round(float(scale_l))))

        lambda_map = _grid_from_scan(d, "lambda_local_mean")
        comm_map = _grid_from_scan(d, "comm_defect_mean")
        gain_map = _grid_from_scan(d, "comm_gain_mean")

        _plot_tile_map(
            field=lambda_map,
            title=f"P2 lambda local map (l={sid})",
            cbar_label="mean(lambda_local)",
            out_path=outdir / f"lambda_local_map_scale_{sid}.png",
            cmap="viridis",
            symmetric=False,
        )
        _plot_tile_map(
            field=comm_map,
            title=f"P2 comm defect map (l={sid})",
            cbar_label="mean(Delta_comm)",
            out_path=outdir / f"comm_defect_map_scale_{sid}.png",
            cmap="magma",
            symmetric=False,
        )
        _plot_tile_map(
            field=gain_map,
            title=f"P2 comm gain map (l={sid})",
            cbar_label="mean(pointwise_gain)",
            out_path=outdir / f"comm_gain_map_scale_{sid}.png",
            cmap="RdBu_r",
            symmetric=True,
        )

    print("[stage] 5/5 report", flush=True)
    _write_report(
        outdir=outdir,
        panel_csv=args.panel_csv,
        target_col=target_col,
        scales_cells=scales_cells,
        summary_df=summary_df,
        op_summary_df=op_summary_df,
        threshold=args.mrms_threshold,
        downsample=args.mrms_downsample,
        n_perm=args.n_perm,
        lambda_weights=weights,
        lambda_scale_power=args.lambda_scale_power,
        decoherence_alpha=args.decoherence_alpha,
    )

    print("Experiment P2 complete.", flush=True)
    print(f"Output: {outdir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_noncommuting_coarse_graining"),
    )
    p.add_argument("--target", choices=["density", "occupancy"], default="density")

    p.add_argument("--scales-cells", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--mrms-downsample", type=int, default=16)
    p.add_argument("--mrms-threshold", type=float, default=3.0)
    p.add_argument("--min-valid-frac", type=float, default=0.90)
    p.add_argument("--max-rows", type=int, default=0)

    p.add_argument("--lambda-weights", nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0])
    p.add_argument("--lambda-scale-power", type=float, default=0.0)
    p.add_argument("--decoherence-alpha", type=float, default=4.0)

    p.add_argument("--ridge-alpha", type=float, default=3.0)
    p.add_argument("--n-perm", type=int, default=49)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--active-quantile", type=float, default=0.67)
    p.add_argument("--comm-floor", type=float, default=1e-4)
    p.add_argument("--max-perm-p", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
