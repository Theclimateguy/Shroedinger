#!/usr/bin/env python3
"""Experiment F5: Structural-scale Lambda signal and multiscale fractal surrogates on ERA5/WPWP."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

try:
    from clean_experiments.experiment_M_cosmo_flow import (
        _blocked_splits,
        _edge_order,
        _evaluate_splits,
        _find_var_name,
        _permutation_test,
        _time_to_seconds,
        _to_tyx_da,
        _xy_coordinates_m,
        _zscore,
    )
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import (  # type: ignore
        _blocked_splits,
        _edge_order,
        _evaluate_splits,
        _find_var_name,
        _permutation_test,
        _time_to_seconds,
        _to_tyx_da,
        _xy_coordinates_m,
        _zscore,
    )


@dataclass
class F5Config:
    min_mae_gain: float = 0.002
    max_perm_p: float = 0.05
    min_strata_positive_frac: float = 0.8

    fractal_block: int = 72
    folds: int = 6
    oot_frac: float = 0.2

    fractal_corr_min: float = 0.20
    fractal_sign_frac_min: float = 0.80
    fractal_oot_corr_min: float = 0.15

    fractal_agreement_spearman_min: float = 0.60
    fractal_agreement_abs_zdiff_max: float = 1.50

    comm_const_var_tol: float = 1e-20
    comm_const_amp_tol: float = 1e-8

    comm_gain_ratio_max: float = 0.50
    poly_gain_ratio_max: float = 0.50

    perm_seed: int = 20260306


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[m]
    y_arr = y_arr[m]
    if len(x_arr) < 3:
        return float("nan")
    if float(np.std(x_arr)) < 1e-14 or float(np.std(y_arr)) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(m)) < 3:
        return float("nan")
    xs = pd.Series(x_arr[m]).rank(method="average").to_numpy(dtype=float)
    ys = pd.Series(y_arr[m]).rank(method="average").to_numpy(dtype=float)
    return _safe_corr(xs, ys)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[m]
    y_arr = y_arr[m]
    if len(x_arr) < 3:
        return float("nan"), float("nan")
    if float(np.std(x_arr)) < 1e-14:
        return 0.0, float(np.mean(y_arr))
    design = np.column_stack([x_arr, np.ones_like(x_arr)])
    coef = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def _fit_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = _linear_fit(x, y)
    if not np.isfinite(slope):
        return float("nan"), float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[m]
    y_arr = y_arr[m]
    y_hat = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-15))
    return slope, r2


def _load_required_fields(input_nc: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    if xr is None:
        raise ImportError("xarray is required for NetCDF input.")

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
        raise ValueError("Failed to read time/lat/lon coordinates.")

    return t_coords, lat_coords, lon_coords, fields, names


def _psd_beta_and_r2(
    field: np.ndarray,
    *,
    k_flat: np.ndarray,
    base_mask: np.ndarray,
    min_points: int,
) -> tuple[float, float]:
    f = np.asarray(field, dtype=float)
    centered = f - float(np.mean(f))
    power = np.abs(np.fft.rfft2(centered)) ** 2
    p_flat = power.reshape(-1)

    p_use = p_flat[base_mask]
    m = p_use > 1e-20
    if int(np.sum(m)) < min_points:
        return float("nan"), float("nan")

    x = np.log(k_flat[base_mask][m])
    y = np.log(p_use[m])
    slope, r2 = _fit_slope_r2(x, y)
    return float(-slope), float(r2)


def _variogram_slope_and_r2(field: np.ndarray, lags: tuple[int, ...]) -> tuple[float, float]:
    f = np.asarray(field, dtype=float)
    hs: list[float] = []
    gs: list[float] = []

    for h in lags:
        if h >= min(f.shape):
            continue
        dx = f[:, h:] - f[:, :-h]
        dy = f[h:, :] - f[:-h, :]
        g = 0.25 * (float(np.mean(dx * dx)) + float(np.mean(dy * dy)))
        if np.isfinite(g) and g > 0.0:
            hs.append(float(h))
            gs.append(float(g))

    if len(gs) < 4:
        return float("nan"), float("nan")

    x = np.log(np.asarray(hs, dtype=float))
    y = np.log(np.asarray(gs, dtype=float))
    slope, r2 = _fit_slope_r2(x, y)
    return float(slope), float(r2)


def _compute_fractal_surrogates(
    *,
    time: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    fields: dict[str, np.ndarray],
) -> pd.DataFrame:
    iwv = fields["iwv"]
    ivt_u = fields["ivt_u"]
    ivt_v = fields["ivt_v"]
    precip = fields["precip"]
    evap = fields["evap"]

    nt, ny, nx = iwv.shape
    time_s = _time_to_seconds(time)
    x_m, y_m = _xy_coordinates_m(lat, lon)

    eo_x = _edge_order(len(x_m))
    eo_y = _edge_order(len(y_m))

    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.rfftfreq(nx)[None, :]
    k_mag = np.sqrt(ky * ky + kx * kx)
    k_flat = k_mag.reshape(-1)
    k_nonzero = k_flat > 0.0
    q_lo, q_hi = np.quantile(k_flat[k_nonzero], [0.10, 0.80])
    k_band = (k_flat >= q_lo) & (k_flat <= q_hi) & k_nonzero

    lags = (1, 2, 4, 8, 12, 16, 24, 32)

    beta = np.full(nt, np.nan, dtype=float)
    beta_r2 = np.full(nt, np.nan, dtype=float)
    vg = np.full(nt, np.nan, dtype=float)
    vg_r2 = np.full(nt, np.nan, dtype=float)

    for t in range(nt):
        if t == 0:
            dt = max(float(time_s[1] - time_s[0]), 1e-12)
            diwv_dt = (iwv[1] - iwv[0]) / dt
        elif t == nt - 1:
            dt = max(float(time_s[-1] - time_s[-2]), 1e-12)
            diwv_dt = (iwv[-1] - iwv[-2]) / dt
        else:
            dt = max(float(time_s[t + 1] - time_s[t - 1]), 1e-12)
            diwv_dt = (iwv[t + 1] - iwv[t - 1]) / dt

        div_ivt = np.gradient(ivt_u[t], x_m, axis=1, edge_order=eo_x) + np.gradient(
            ivt_v[t], y_m, axis=0, edge_order=eo_y
        )
        residual_field = np.asarray(diwv_dt + div_ivt + (precip[t] - evap[t]), dtype=float)

        b, b_r2 = _psd_beta_and_r2(
            residual_field,
            k_flat=k_flat,
            base_mask=k_band,
            min_points=20,
        )
        v, v_r2 = _variogram_slope_and_r2(residual_field, lags=lags)
        beta[t] = b
        beta_r2[t] = b_r2
        vg[t] = v
        vg_r2[t] = v_r2

        if (t + 1) % 600 == 0:
            print(f"[F5] fractal surrogate progress: {t + 1}/{nt}", flush=True)

    return pd.DataFrame(
        {
            "time_index": np.arange(nt, dtype=int),
            "fractal_psd_beta": beta,
            "fractal_psd_r2": beta_r2,
            "fractal_variogram_slope": vg,
            "fractal_variogram_r2": vg_r2,
        }
    )


def _choose_folds(n: int, want: int) -> int:
    max_safe = max(2, n // 4)
    return max(2, min(want, max_safe))


def _blocked_sign_and_oot_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    oot_frac: float,
) -> dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[m]
    y_arr = y_arr[m]

    if len(x_arr) < 12:
        return {
            "corr": float("nan"),
            "spearman": float("nan"),
            "fold_sign_frac": float("nan"),
            "fold_test_corr_mean": float("nan"),
            "fold_test_corr_min": float("nan"),
            "oot_train_slope": float("nan"),
            "oot_test_corr": float("nan"),
        }

    corr = _safe_corr(x_arr, y_arr)
    rho = _safe_spearman(x_arr, y_arr)

    n_fold_eff = _choose_folds(len(x_arr), folds)
    split_signs: list[float] = []
    test_corrs: list[float] = []

    splits = _blocked_splits(len(x_arr), n_fold_eff)
    for train_idx, test_idx in splits:
        slope, _ = _linear_fit(x_arr[train_idx], y_arr[train_idx])
        r_test = _safe_corr(x_arr[test_idx], y_arr[test_idx])
        test_corrs.append(r_test)
        if np.isfinite(slope) and np.isfinite(r_test) and abs(r_test) > 1e-14:
            split_signs.append(1.0 if np.sign(slope) == np.sign(r_test) else 0.0)

    sign_frac = float(np.mean(split_signs)) if split_signs else float("nan")

    n_test = max(4, int(round(oot_frac * len(x_arr))))
    n_test = min(n_test, len(x_arr) - 4)
    n_train = len(x_arr) - n_test
    if n_train < 4:
        oot_slope = float("nan")
        oot_corr = float("nan")
    else:
        oot_slope, _ = _linear_fit(x_arr[:n_train], y_arr[:n_train])
        oot_corr = _safe_corr(x_arr[n_train:], y_arr[n_train:])

    return {
        "corr": float(corr),
        "spearman": float(rho),
        "fold_sign_frac": float(sign_frac),
        "fold_test_corr_mean": float(np.nanmean(test_corrs)) if test_corrs else float("nan"),
        "fold_test_corr_min": float(np.nanmin(test_corrs)) if test_corrs else float("nan"),
        "oot_train_slope": float(oot_slope),
        "oot_test_corr": float(oot_corr),
    }


def _build_block_table(dataset: pd.DataFrame, block: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    n = len(dataset)

    for s in range(0, n, block):
        e = min(s + block, n)
        sub = dataset.iloc[s:e]

        valid = np.isfinite(sub["fractal_psd_beta"].to_numpy(float)) & np.isfinite(
            sub["fractal_variogram_slope"].to_numpy(float)
        )
        if int(np.sum(valid)) < max(8, block // 4):
            continue

        mae_base = float(np.nanmean(np.abs(sub["resid_base_oof"].to_numpy(float))))
        mae_full = float(np.nanmean(np.abs(sub["resid_full_oof"].to_numpy(float))))
        oof_gain_block = float((mae_base - mae_full) / (mae_base + 1e-12))

        rows.append(
            {
                "block_id": int(len(rows)),
                "start": int(s),
                "stop": int(e),
                "n": int(e - s),
                "lambda_abs_mean": float(np.nanmean(np.abs(sub["lambda_struct"].to_numpy(float)))),
                "lambda_std": float(np.nanstd(sub["lambda_struct"].to_numpy(float))),
                "oof_gain_block": oof_gain_block,
                "fractal_psd_beta": float(np.nanmean(sub["fractal_psd_beta"].to_numpy(float))),
                "fractal_variogram_slope": float(np.nanmean(sub["fractal_variogram_slope"].to_numpy(float))),
            }
        )

    return pd.DataFrame(rows)


def _polynomial_placebo_eval(
    *,
    dataset: pd.DataFrame,
    ridge_alpha: float,
    n_folds: int,
    n_perm: int,
    perm_block: int,
    seed: int,
) -> dict[str, float]:
    y = dataset["residual_base_res0"].to_numpy(dtype=float)
    ctrl = dataset["n_density_ctrl_z"].to_numpy(dtype=float)
    lam = dataset["lambda_struct"].to_numpy(dtype=float)

    valid = np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(lam)
    yv = y[valid]
    ctrl_v = ctrl[valid]
    lam_v = lam[valid]

    ctrl_z = _zscore(ctrl_v)
    pseudo_raw = ctrl_z**2 + 0.30 * ctrl_z**3
    pseudo_z = _zscore(pseudo_raw)
    pseudo = float(np.mean(lam_v)) + float(np.std(lam_v) + 1e-12) * pseudo_z

    x_base = np.column_stack([ctrl_v])
    x_full = np.column_stack([ctrl_v, pseudo])

    n_fold_eff = _choose_folds(len(yv), n_folds)
    splits = _blocked_splits(len(yv), n_folds=n_fold_eff)

    split_df, yhat_b, yhat_f = _evaluate_splits(
        y=yv,
        x_base=x_base,
        x_full=x_full,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl", "poly_placebo"],
        splits=splits,
        ridge_alpha=ridge_alpha,
    )

    mae_base = float(np.mean(np.abs(yv - yhat_b)))
    mae_full = float(np.mean(np.abs(yv - yhat_f)))
    gain = float((mae_base - mae_full) / (mae_base + 1e-12))

    perm_p, _, stat_real = _permutation_test(
        y=yv,
        x_base=x_base,
        x_full=x_full,
        base_feature_names=["ctrl"],
        full_feature_names=["ctrl", "poly_placebo"],
        permute_cols=np.array([1], dtype=int),
        splits=splits,
        ridge_alpha=ridge_alpha,
        n_perm=n_perm,
        perm_block=perm_block,
        seed=seed,
    )

    return {
        "mae_base_oof": mae_base,
        "mae_full_oof": mae_full,
        "oof_gain_frac": gain,
        "perm_stat_real_median_gain": float(stat_real),
        "perm_p_value": float(perm_p),
        "split_gain_median": float(np.median(split_df["mae_gain_frac"].to_numpy(float))),
        "split_gain_min": float(np.min(split_df["mae_gain_frac"].to_numpy(float))),
    }


def _constant_metric_diag(arr: np.ndarray, var_tol: float, amp_tol: float) -> dict[str, float | bool]:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "var": float("nan"),
            "max_abs": float("nan"),
            "corr_applicable": False,
            "constant_metric": True,
        }
    v = float(np.var(x))
    m = float(np.max(np.abs(x)))
    constant = bool((v < var_tol) and (m < amp_tol))
    return {
        "var": v,
        "max_abs": m,
        "corr_applicable": bool(not constant),
        "constant_metric": constant,
    }


def run_f5(
    *,
    input_nc: Path,
    m_dir: Path,
    falsification_dir: Path,
    land_ocean_dir: Path,
    noise_probe_dir: Path,
    outdir: Path,
    cfg: F5Config,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    summary_csv = m_dir / "experiment_M_summary.csv"
    timeseries_csv = m_dir / "experiment_M_timeseries.csv"
    permutation_csv = m_dir / "experiment_M_permutation.csv"
    strata_csv = m_dir / "experiment_M_strata.csv"

    summary_df = pd.read_csv(summary_csv)
    ts_df = pd.read_csv(timeseries_csv)
    perm_df = pd.read_csv(permutation_csv)
    strata_df = pd.read_csv(strata_csv)

    if summary_df.empty:
        raise ValueError(f"Empty summary: {summary_csv}")

    s0 = summary_df.iloc[0]
    real_gain = float(s0["oof_gain_frac"])
    perm_p = float(s0["perm_p_value"])
    strata_positive_frac = float(s0["strata_positive_frac"])

    h1_pass = bool(
        (real_gain >= cfg.min_mae_gain)
        and (perm_p <= cfg.max_perm_p)
        and (strata_positive_frac >= cfg.min_strata_positive_frac)
    )

    print("[F5] loading ERA5/WPWP fields for external fractal surrogates...", flush=True)
    time, lat, lon, fields, field_names = _load_required_fields(input_nc)

    print("[F5] computing residual-field fractal surrogates (PSD + variogram)...", flush=True)
    fractal_df = _compute_fractal_surrogates(time=time, lat=lat, lon=lon, fields=fields)

    dataset = ts_df.merge(fractal_df, on="time_index", how="inner")

    keep_cols = [
        "time_index",
        "time",
        "residual_base_res0",
        "n_density_ctrl_z",
        "lambda_struct",
        "yhat_base_oof",
        "yhat_full_oof",
        "resid_base_oof",
        "resid_full_oof",
        "fractal_psd_beta",
        "fractal_psd_r2",
        "fractal_variogram_slope",
        "fractal_variogram_r2",
    ]
    missing = [c for c in keep_cols if c not in dataset.columns]
    if missing:
        raise KeyError(f"Missing required columns in merged dataset: {missing}")
    dataset = dataset[keep_cols].copy()
    dataset["lambda_abs"] = np.abs(dataset["lambda_struct"].to_numpy(dtype=float))

    agreement_spearman = _safe_spearman(
        dataset["fractal_psd_beta"].to_numpy(float),
        dataset["fractal_variogram_slope"].to_numpy(float),
    )
    z1 = _zscore(dataset["fractal_psd_beta"].to_numpy(float))
    z2 = _zscore(dataset["fractal_variogram_slope"].to_numpy(float))
    agreement_abs_zdiff_mean = float(np.nanmean(np.abs(z1 - z2)))
    fractal_estimators_agree = bool(
        (agreement_spearman >= cfg.fractal_agreement_spearman_min)
        and (agreement_abs_zdiff_mean <= cfg.fractal_agreement_abs_zdiff_max)
    )

    block_df = _build_block_table(dataset, block=cfg.fractal_block)
    if len(block_df) < 16:
        raise ValueError(
            f"Too few valid blocks ({len(block_df)}) after block aggregation. "
            "Decrease --fractal-block or inspect surrogate NaNs."
        )

    relation_rows: list[dict[str, float | str | bool]] = []
    h2_targets = ("lambda_abs_mean", "oof_gain_block")
    h2_features = ("fractal_psd_beta", "fractal_variogram_slope")
    surrogate_pass_map: dict[str, bool] = {k: False for k in h2_features}

    for feat in h2_features:
        for target in h2_targets:
            m = _blocked_sign_and_oot_metrics(
                block_df[feat].to_numpy(float),
                block_df[target].to_numpy(float),
                folds=cfg.folds,
                oot_frac=cfg.oot_frac,
            )
            pass_rel = bool(
                np.isfinite(m["corr"])
                and np.isfinite(m["fold_sign_frac"])
                and np.isfinite(m["oot_test_corr"])
                and (m["corr"] >= cfg.fractal_corr_min)
                and (m["fold_sign_frac"] >= cfg.fractal_sign_frac_min)
                and (m["oot_test_corr"] >= cfg.fractal_oot_corr_min)
            )
            surrogate_pass_map[feat] = bool(surrogate_pass_map[feat] or pass_rel)
            relation_rows.append(
                {
                    "feature": feat,
                    "target": target,
                    "corr": float(m["corr"]),
                    "spearman": float(m["spearman"]),
                    "fold_sign_frac": float(m["fold_sign_frac"]),
                    "fold_test_corr_mean": float(m["fold_test_corr_mean"]),
                    "fold_test_corr_min": float(m["fold_test_corr_min"]),
                    "oot_train_slope": float(m["oot_train_slope"]),
                    "oot_test_corr": float(m["oot_test_corr"]),
                    "pass_relation": bool(pass_rel),
                }
            )

    relation_df = pd.DataFrame(relation_rows)
    h2_pass = bool(any(surrogate_pass_map.values()))

    perm_vals = perm_df["stat_perm_median_gain"].to_numpy(dtype=float)
    perm_null_median = float(np.median(perm_vals))
    perm_null_q95 = float(np.quantile(perm_vals, 0.95))
    placebo_time_degrades = bool((real_gain > perm_null_q95) and (perm_p <= cfg.max_perm_p))

    ridge_alpha = float(s0["ridge_alpha"])
    n_folds = int(s0["n_folds"])
    n_perm = int(s0["n_perm"])
    perm_block = int(s0["perm_block"])
    poly = _polynomial_placebo_eval(
        dataset=dataset,
        ridge_alpha=ridge_alpha,
        n_folds=n_folds,
        n_perm=n_perm,
        perm_block=perm_block,
        seed=cfg.perm_seed + 701,
    )
    poly_degrades = bool(poly["oof_gain_frac"] <= cfg.poly_gain_ratio_max * real_gain)

    comm_csv = falsification_dir / "comm_control_metrics.csv"
    comm_df = pd.read_csv(comm_csv)
    if comm_df.empty:
        raise ValueError(f"Empty commutative control metrics: {comm_csv}")
    comm_row = comm_df.iloc[0]
    comm_gain = float(comm_row["mae_gain_comm"])
    comm_degrades = bool(comm_gain <= cfg.comm_gain_ratio_max * real_gain)

    comm_ts_csv = falsification_dir / "lambda_falsification_timeseries.csv"
    comm_ts = pd.read_csv(comm_ts_csv)
    if "lambda_comm" not in comm_ts.columns:
        raise KeyError(f"Missing lambda_comm in {comm_ts_csv}")
    comm_diag = _constant_metric_diag(
        comm_ts["lambda_comm"].to_numpy(float),
        var_tol=cfg.comm_const_var_tol,
        amp_tol=cfg.comm_const_amp_tol,
    )

    if bool(comm_diag["corr_applicable"]):
        comm_corr = _safe_corr(comm_ts["lambda_comm"].to_numpy(float), comm_ts["residual_base_res0"].to_numpy(float))
    else:
        comm_corr = float("nan")

    surface_summary_csv = land_ocean_dir / "surface_summary.csv"
    surface_df = pd.read_csv(surface_summary_csv)
    noise_metrics_csv = noise_probe_dir / "noise_probe_metrics.csv"
    noise_df = pd.read_csv(noise_metrics_csv)

    land_row = surface_df[surface_df["surface"] == "land"].iloc[0]
    ocean_row = surface_df[surface_df["surface"] == "ocean"].iloc[0]

    noise_land_full = noise_df[(noise_df["surface"] == "land") & (noise_df["target"] == "residual_full")].iloc[0]
    noise_ocean_full = noise_df[(noise_df["surface"] == "ocean") & (noise_df["target"] == "residual_full")].iloc[0]

    cond3_pass = bool(placebo_time_degrades and comm_degrades)
    pass_all = bool(h1_pass and h2_pass and cond3_pass)

    placebo_rows = [
        {
            "test": "placebo_time_block_permutation",
            "real_gain": real_gain,
            "null_median_gain": perm_null_median,
            "null_q95_gain": perm_null_q95,
            "perm_p_value": perm_p,
            "pass_degradation": placebo_time_degrades,
        },
        {
            "test": "polynomial_placebo",
            "real_gain": real_gain,
            "placebo_gain": float(poly["oof_gain_frac"]),
            "placebo_perm_p": float(poly["perm_p_value"]),
            "gain_ratio_vs_real": float(poly["oof_gain_frac"] / (real_gain + 1e-12)),
            "pass_degradation": poly_degrades,
        },
        {
            "test": "commutative_control",
            "real_gain": real_gain,
            "comm_gain": comm_gain,
            "gain_ratio_vs_real": float(comm_gain / (real_gain + 1e-12)),
            "comm_var": float(comm_diag["var"]),
            "comm_max_abs": float(comm_diag["max_abs"]),
            "comm_corr_applicable": bool(comm_diag["corr_applicable"]),
            "comm_corr": float(comm_corr),
            "pass_degradation": comm_degrades,
        },
    ]
    placebo_df = pd.DataFrame(placebo_rows)

    summary_out = pd.DataFrame(
        [
            {
                "input_nc": str(input_nc),
                "m_dir": str(m_dir),
                "n_time": int(len(dataset)),
                "n_blocks": int(len(block_df)),
                "h1_real_oof_gain": real_gain,
                "h1_perm_p_value": perm_p,
                "h1_strata_positive_frac": strata_positive_frac,
                "h1_pass": h1_pass,
                "fractal_agreement_spearman": agreement_spearman,
                "fractal_agreement_abs_zdiff_mean": agreement_abs_zdiff_mean,
                "fractal_estimators_agree": fractal_estimators_agree,
                "h2_pass_any_surrogate": h2_pass,
                "placebo_time_degrades": placebo_time_degrades,
                "commutative_control_degrades": comm_degrades,
                "poly_placebo_degrades": poly_degrades,
                "cond3_placebo_comm_pass": cond3_pass,
                "pass_all": pass_all,
                "threshold_min_mae_gain": cfg.min_mae_gain,
                "threshold_max_perm_p": cfg.max_perm_p,
                "threshold_min_strata_positive_frac": cfg.min_strata_positive_frac,
                "threshold_fractal_corr_min": cfg.fractal_corr_min,
                "threshold_fractal_sign_frac_min": cfg.fractal_sign_frac_min,
                "threshold_fractal_oot_corr_min": cfg.fractal_oot_corr_min,
                "threshold_fractal_agreement_spearman_min": cfg.fractal_agreement_spearman_min,
                "threshold_fractal_agreement_abs_zdiff_max": cfg.fractal_agreement_abs_zdiff_max,
            }
        ]
    )

    test_rows = [
        {
            "test": "H1_main_M_detection",
            "metric_1": real_gain,
            "metric_2": perm_p,
            "metric_3": strata_positive_frac,
            "pass": h1_pass,
        },
        {
            "test": "H2_any_surrogate_relation",
            "metric_1": float(np.nanmax(relation_df["corr"].to_numpy(float))),
            "metric_2": float(np.nanmax(relation_df["fold_sign_frac"].to_numpy(float))),
            "metric_3": float(np.nanmax(relation_df["oot_test_corr"].to_numpy(float))),
            "pass": h2_pass,
        },
        {
            "test": "Falsification_placebo_time",
            "metric_1": real_gain,
            "metric_2": perm_null_q95,
            "metric_3": perm_p,
            "pass": placebo_time_degrades,
        },
        {
            "test": "Falsification_commutative_control",
            "metric_1": real_gain,
            "metric_2": comm_gain,
            "metric_3": float(comm_diag["max_abs"]),
            "pass": comm_degrades,
        },
        {
            "test": "LandOcean_split",
            "metric_1": float(land_row["oof_gain_frac"]),
            "metric_2": float(ocean_row["oof_gain_frac"]),
            "metric_3": float(ocean_row["perm_p_value"]),
            "pass": bool(np.isfinite(ocean_row["oof_gain_frac"])),
        },
        {
            "test": "Noise_probe_residual_full",
            "metric_1": float(noise_land_full["oof_gain_frac"]),
            "metric_2": float(noise_ocean_full["oof_gain_frac"]),
            "metric_3": float(noise_ocean_full["perm_p_value"]),
            "pass": bool(np.isfinite(noise_ocean_full["oof_gain_frac"])),
        },
    ]
    test_df = pd.DataFrame(test_rows)

    dataset.to_csv(outdir / "experiment_F5_dataset.csv", index=False)
    block_df.to_csv(outdir / "experiment_F5_block_metrics.csv", index=False)
    relation_df.to_csv(outdir / "experiment_F5_surrogate_metrics.csv", index=False)
    placebo_df.to_csv(outdir / "experiment_F5_placebo_metrics.csv", index=False)
    test_df.to_csv(outdir / "experiment_F5_test_metrics.csv", index=False)
    summary_out.to_csv(outdir / "experiment_F5_summary_metrics.csv", index=False)

    verdict = {
        "pass_all": pass_all,
        "conditions": {
            "condition_1_h1_main_detection": bool(h1_pass),
            "condition_2_h2_multiscale_link": bool(h2_pass),
            "condition_3_placebo_time_and_comm_degrade": bool(cond3_pass),
        },
        "h1": {
            "oof_gain_frac": real_gain,
            "perm_p_value": perm_p,
            "strata_positive_frac": strata_positive_frac,
        },
        "h2": {
            "surrogate_pass_map": surrogate_pass_map,
            "fractal_agreement_spearman": agreement_spearman,
            "fractal_agreement_abs_zdiff_mean": agreement_abs_zdiff_mean,
            "fractal_estimators_agree": fractal_estimators_agree,
        },
        "controls": {
            "placebo_time_degrades": placebo_time_degrades,
            "polynomial_placebo_degrades": poly_degrades,
            "commutative_control_degrades": comm_degrades,
            "comm_corr_applicable": bool(comm_diag["corr_applicable"]),
            "comm_constant_metric": bool(comm_diag["constant_metric"]),
        },
        "inputs": {
            "input_nc": str(input_nc),
            "m_dir": str(m_dir),
            "field_names": field_names,
        },
    }

    with open(outdir / "experiment_F5_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Experiment F5 (ERA5/WPWP): Structural-scale Lambda and Multiscale Surrogates",
        "",
        "## Headline",
        f"- pass_all: {pass_all}",
        f"- H1 (main M threshold): {h1_pass}",
        f"- H2 (at least one surrogate reproducible): {h2_pass}",
        f"- Condition 3 (placebo-time + commutative degradation): {cond3_pass}",
        "",
        "## H1 Main Detection (curated Experiment M)",
        f"- oof_gain_frac = {real_gain:.6f} (threshold >= {cfg.min_mae_gain:.6f})",
        f"- perm_p_value = {perm_p:.6f} (threshold <= {cfg.max_perm_p:.6f})",
        f"- strata_positive_frac = {strata_positive_frac:.6f} (threshold >= {cfg.min_strata_positive_frac:.6f})",
        "",
        "## H2 Multiscale Link",
        f"- estimator agreement Spearman(psd, variogram) = {agreement_spearman:.6f}",
        f"- estimator agreement mean |z_psd-z_vario| = {agreement_abs_zdiff_mean:.6f}",
        f"- estimator agreement pass = {fractal_estimators_agree}",
        "",
        "Per-feature/target checks:",
    ]
    for row in relation_df.itertuples(index=False):
        report_lines.append(
            f"- {row.feature} -> {row.target}: corr={float(row.corr):.6f}, "
            f"fold_sign_frac={float(row.fold_sign_frac):.3f}, oot_corr={float(row.oot_test_corr):.6f}, "
            f"pass={bool(row.pass_relation)}"
        )

    report_lines.extend(
        [
            "",
            "## Placebo/Falsification",
            f"- placebo-time: real_gain={real_gain:.6f}, null_q95={perm_null_q95:.6f}, perm_p={perm_p:.6f}, pass={placebo_time_degrades}",
            f"- polynomial placebo: gain={float(poly['oof_gain_frac']):.6f}, perm_p={float(poly['perm_p_value']):.6f}, pass={poly_degrades}",
            f"- commutative control: gain_comm={comm_gain:.6f}, corr_applicable={bool(comm_diag['corr_applicable'])}, "
            f"Var={float(comm_diag['var']):.3e}, max_abs={float(comm_diag['max_abs']):.3e}, pass={comm_degrades}",
            "",
            "## Land/Ocean + Noise Probe",
            f"- land gain={float(land_row['oof_gain_frac']):.6f}, perm_p={float(land_row['perm_p_value']):.6f}",
            f"- ocean gain={float(ocean_row['oof_gain_frac']):.6f}, perm_p={float(ocean_row['perm_p_value']):.6f}",
            f"- noise residual_full land gain={float(noise_land_full['oof_gain_frac']):.6f}, perm_p={float(noise_land_full['perm_p_value']):.6f}",
            f"- noise residual_full ocean gain={float(noise_ocean_full['oof_gain_frac']):.6f}, perm_p={float(noise_ocean_full['perm_p_value']):.6f}",
            "",
            "## Output Files",
            "- experiment_F5_summary_metrics.csv",
            "- experiment_F5_test_metrics.csv",
            "- experiment_F5_placebo_metrics.csv",
            "- experiment_F5_surrogate_metrics.csv",
            "- experiment_F5_block_metrics.csv",
            "- experiment_F5_dataset.csv",
            "- experiment_F5_verdict.json",
        ]
    )

    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[F5] saved outputs to: {outdir.resolve()}", flush=True)
    return verdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc")
    p.add_argument("--m-dir", default="clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated")
    p.add_argument("--falsification-dir", default="clean_experiments/results/experiment_M_lambda_falsification_tests")
    p.add_argument("--land-ocean-dir", default="clean_experiments/results/experiment_M_land_ocean_split")
    p.add_argument("--noise-probe-dir", default="clean_experiments/results/experiment_M_land_ocean_noise_probe")
    p.add_argument("--out", default="clean_experiments/results/experiment_F5_lambda_struct_fractal_era5")

    p.add_argument("--min-mae-gain", type=float, default=0.002)
    p.add_argument("--max-perm-p", type=float, default=0.05)
    p.add_argument("--min-strata-positive-frac", type=float, default=0.8)

    p.add_argument("--fractal-block", type=int, default=72)
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--oot-frac", type=float, default=0.2)

    p.add_argument("--fractal-corr-min", type=float, default=0.20)
    p.add_argument("--fractal-sign-frac-min", type=float, default=0.80)
    p.add_argument("--fractal-oot-corr-min", type=float, default=0.15)

    p.add_argument("--fractal-agreement-spearman-min", type=float, default=0.60)
    p.add_argument("--fractal-agreement-abs-zdiff-max", type=float, default=1.50)

    p.add_argument("--comm-const-var-tol", type=float, default=1e-20)
    p.add_argument("--comm-const-amp-tol", type=float, default=1e-8)
    p.add_argument("--comm-gain-ratio-max", type=float, default=0.50)
    p.add_argument("--poly-gain-ratio-max", type=float, default=0.50)

    p.add_argument("--perm-seed", type=int, default=20260306)
    args = p.parse_args()

    cfg = F5Config(
        min_mae_gain=float(args.min_mae_gain),
        max_perm_p=float(args.max_perm_p),
        min_strata_positive_frac=float(args.min_strata_positive_frac),
        fractal_block=max(8, int(args.fractal_block)),
        folds=max(2, int(args.folds)),
        oot_frac=float(args.oot_frac),
        fractal_corr_min=float(args.fractal_corr_min),
        fractal_sign_frac_min=float(args.fractal_sign_frac_min),
        fractal_oot_corr_min=float(args.fractal_oot_corr_min),
        fractal_agreement_spearman_min=float(args.fractal_agreement_spearman_min),
        fractal_agreement_abs_zdiff_max=float(args.fractal_agreement_abs_zdiff_max),
        comm_const_var_tol=float(args.comm_const_var_tol),
        comm_const_amp_tol=float(args.comm_const_amp_tol),
        comm_gain_ratio_max=float(args.comm_gain_ratio_max),
        poly_gain_ratio_max=float(args.poly_gain_ratio_max),
        perm_seed=int(args.perm_seed),
    )

    verdict = run_f5(
        input_nc=Path(args.input),
        m_dir=Path(args.m_dir),
        falsification_dir=Path(args.falsification_dir),
        land_ocean_dir=Path(args.land_ocean_dir),
        noise_probe_dir=Path(args.noise_probe_dir),
        outdir=Path(args.out),
        cfg=cfg,
    )

    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
