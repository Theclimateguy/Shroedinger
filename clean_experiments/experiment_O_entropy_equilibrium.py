#!/usr/bin/env python3
"""Experiment O: Clausius-consistent thermodynamic test with Lambda correction."""

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
    from clean_experiments.experiment_M_cosmo_flow import _blocked_splits, _fit_ridge_scaled
except ModuleNotFoundError:
    from experiment_M_cosmo_flow import _blocked_splits, _fit_ridge_scaled  # type: ignore


EPS = 1e-12
CP_D = 1004.64
L_V = 2.5e6
T0_K = 273.15


@dataclass(frozen=True)
class Taxon:
    name: str
    p_bot_hpa: float
    p_top_hpa: float


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


def _parse_taxa(specs: list[str]) -> list[Taxon]:
    out: list[Taxon] = []
    for spec in specs:
        # format: NAME:PBOT-PTOP
        try:
            name, rng = spec.split(":")
            p_bot_s, p_top_s = rng.split("-")
            p_bot = float(p_bot_s)
            p_top = float(p_top_s)
        except Exception as exc:
            raise ValueError(f"Invalid taxa spec '{spec}'. Use NAME:PBOT-PTOP, e.g. FT:850-300") from exc
        if p_bot < p_top:
            raise ValueError(f"Taxon {name}: expected PBOT>=PTOP, got {p_bot}<{p_top}")
        out.append(Taxon(name=name, p_bot_hpa=p_bot, p_top_hpa=p_top))
    return out


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


def _weighted_mean_2d(field_tyx: np.ndarray, area_w: np.ndarray) -> np.ndarray:
    # field_tyx: [t,y,x], area_w: [y,x], pre-normalized for valid region.
    return np.einsum("tyx,yx->t", field_tyx, area_w, optimize=True)


def _build_time_features(time_dt: pd.Series) -> pd.DataFrame:
    t = pd.to_datetime(time_dt)
    frac_day = (t.dt.hour + t.dt.minute / 60.0) / 24.0
    doy = (t.dt.dayofyear - 1).to_numpy(dtype=float) + frac_day.to_numpy(dtype=float)
    phase = 2.0 * np.pi * doy / 365.25
    years = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / (365.25 * 24.0 * 3600.0)
    return pd.DataFrame(
        {
            "sin1": np.sin(phase),
            "cos1": np.cos(phase),
            "sin2": np.sin(2.0 * phase),
            "cos2": np.cos(2.0 * phase),
            "trend": years,
        }
    )


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 8:
        return np.nan
    yv = y[mask]
    pv = p[mask]
    sse = float(np.sum((yv - pv) ** 2))
    sst = float(np.sum((yv - np.mean(yv)) ** 2))
    if sst < EPS:
        return np.nan
    return float(1.0 - sse / (sst + EPS))


def _standardize_with_train(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.nanmean(x[train_idx]))
    sd = float(np.nanstd(x[train_idx]))
    if sd < 1e-15:
        sd = 1.0
    z = (x - mu) / sd
    return z, mu, sd


def _block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    starts = list(range(0, n, block))
    pieces = [x[s : min(s + block, n)] for s in starts]
    rng.shuffle(pieces)
    return np.concatenate(pieces, axis=0)[:n]


def _bootstrap_rel_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = np.arange(0, n, block, dtype=int)
    out: list[np.ndarray] = []
    cur = 0
    while cur < n:
        s = int(rng.choice(starts))
        e = min(s + block, n)
        seg = np.arange(s, e, dtype=int)
        out.append(seg)
        cur += len(seg)
    return np.concatenate(out)[:n]


def _fit_with_alpha(
    *,
    y: np.ndarray,
    x_base: np.ndarray,
    x_full: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, yhat_b = _fit_ridge_scaled(x_base[train_idx], y[train_idx], x_base[eval_idx], alpha)
    _, _, yhat_f = _fit_ridge_scaled(x_full[train_idx], y[train_idx], x_full[eval_idx], alpha)
    return yhat_b, yhat_f


def _cv_alpha_grid(
    *,
    y: np.ndarray,
    x_base: np.ndarray,
    x_full: np.ndarray,
    train_idx: np.ndarray,
    n_folds: int,
    alpha_grid: list[float],
) -> tuple[pd.DataFrame, pd.Series]:
    rel_splits = _blocked_splits(len(train_idx), n_folds=n_folds)
    rows: list[dict[str, float | int]] = []

    for alpha in alpha_grid:
        fold_gains: list[float] = []
        for fid, (tr_rel, va_rel) in enumerate(rel_splits):
            tr = train_idx[tr_rel]
            va = train_idx[va_rel]
            yhat_b, yhat_f = _fit_with_alpha(
                y=y,
                x_base=x_base,
                x_full=x_full,
                train_idx=tr,
                eval_idx=va,
                alpha=alpha,
            )
            r2_b = _r2(y[va], yhat_b)
            r2_f = _r2(y[va], yhat_f)
            gain = float(r2_f - r2_b) if np.isfinite(r2_b) and np.isfinite(r2_f) else np.nan
            fold_gains.append(gain)
            rows.append(
                {
                    "alpha": float(alpha),
                    "fold_id": int(fid),
                    "r2_base": float(r2_b),
                    "r2_full": float(r2_f),
                    "gain_r2": float(gain),
                }
            )

    fold_df = pd.DataFrame(rows)
    stats = (
        fold_df.groupby("alpha", as_index=False)
        .agg(
            cv_gain_mean=("gain_r2", "mean"),
            cv_gain_median=("gain_r2", "median"),
            cv_gain_std=("gain_r2", "std"),
            cv_r2_full_mean=("r2_full", "mean"),
            cv_r2_base_mean=("r2_base", "mean"),
        )
        .sort_values("alpha")
        .reset_index(drop=True)
    )

    best_idx = int(stats["cv_gain_mean"].idxmax())
    best = stats.loc[best_idx]
    se = float(best["cv_gain_std"] / np.sqrt(max(float(n_folds), 1.0)))
    threshold = float(best["cv_gain_mean"] - se)
    eligible = stats[stats["cv_gain_mean"] >= threshold].copy()
    if eligible.empty:
        selected = best
    else:
        eligible = eligible.sort_values(by=["alpha", "cv_gain_mean"], ascending=[False, False]).reset_index(drop=True)
        selected = eligible.iloc[0]

    return fold_df, selected


def _compute_flux_timeseries(
    *,
    input_nc: Path,
    taxa: list[Taxon],
    west_split_lon: float,
    lat_stride: int,
    lon_stride: int,
    batch_size: int,
) -> tuple[pd.DataFrame, float]:
    with Dataset(input_nc, mode="r") as ds:
        time_name = _find_coord_name(ds, ("valid_time", "time", "datetime", "date"), "time")
        lat_name = _find_coord_name(ds, ("latitude", "lat", "y", "rlat"), "latitude")
        lon_name = _find_coord_name(ds, ("longitude", "lon", "x", "rlon"), "longitude")
        level_name = _find_coord_name(ds, ("pressure_level", "level", "plev", "isobaricInhPa"), "pressure level")

        temp_name = _find_var_name(ds, ("temp_pl", "t"), "temp_pl")
        q_name = _find_var_name(ds, ("q_pl", "q"), "q_pl")
        w_name = _find_var_name(ds, ("w_pl", "w", "omega"), "w_pl")

        time_ns = _to_datetime64(ds.variables[time_name])
        levels_raw = np.asarray(ds.variables[level_name][:], dtype=float)
        levels_hpa = levels_raw / 100.0 if np.nanmax(np.abs(levels_raw)) > 3000.0 else levels_raw
        levels_pa = levels_hpa * 100.0
        lvl_w = _level_weights_pa(levels_pa)

        lat = np.asarray(ds.variables[lat_name][:], dtype=float)[::lat_stride]
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)[::lon_stride]
        ny = len(lat)
        nx = len(lon)
        nt = len(time_ns)

        coslat = np.cos(np.deg2rad(lat))
        area_all = np.repeat(coslat[:, None], nx, axis=1)
        split_eff = float(west_split_lon)
        if not (np.any(lon <= split_eff) and np.any(lon > split_eff)):
            split_eff = float(np.median(lon))
            print(
                f"[experiment_O] west_split_lon={west_split_lon} outside lon support "
                f"[{float(np.min(lon)):.2f},{float(np.max(lon)):.2f}], using median split={split_eff:.2f}",
                flush=True,
            )

        west_mask = np.repeat((lon <= split_eff)[None, :], ny, axis=0)
        east_mask = ~west_mask

        def norm_region(mask: np.ndarray) -> np.ndarray:
            aw = np.where(mask, area_all, 0.0)
            s = float(np.sum(aw))
            if s < EPS:
                return aw
            return aw / s

        region_weights = {
            "all": norm_region(np.ones((ny, nx), dtype=bool)),
            "west": norm_region(west_mask),
            "east": norm_region(east_mask),
        }

        temp_var = ds.variables[temp_name]
        q_var = ds.variables[q_name]
        w_var = ds.variables[w_name]

        temp_dims = tuple(temp_var.dimensions)
        q_dims = tuple(q_var.dimensions)
        w_dims = tuple(w_var.dimensions)

        taxa_masks: dict[str, np.ndarray] = {}
        taxa_weights: dict[str, np.ndarray] = {}
        for tx in taxa:
            mask = (levels_hpa <= tx.p_bot_hpa) & (levels_hpa >= tx.p_top_hpa)
            if int(mask.sum()) == 0:
                raise ValueError(
                    f"Taxon {tx.name} has no levels in [{tx.p_top_hpa},{tx.p_bot_hpa}] hPa. Levels: {levels_hpa.tolist()}"
                )
            w = lvl_w[mask]
            w = w / (float(np.sum(w)) + EPS)
            taxa_masks[tx.name] = mask
            taxa_weights[tx.name] = w

        cols: dict[str, np.ndarray] = {}
        for tx in taxa:
            for region in ("all", "west", "east"):
                cols[f"j_anom_{tx.name}_{region}"] = np.full(nt, np.nan, dtype=float)
                cols[f"j_moist_{tx.name}_{region}"] = np.full(nt, np.nan, dtype=float)
                cols[f"j_dry_{tx.name}_{region}"] = np.full(nt, np.nan, dtype=float)
                cols[f"dq_in_{tx.name}_{region}"] = np.full(nt, np.nan, dtype=float)

        for start in range(0, nt, batch_size):
            stop = min(start + batch_size, nt)
            temp_blk = _move_to_tlxy(
                np.asarray(temp_var[start:stop, ...], dtype=float),
                temp_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, :, ::lat_stride, ::lon_stride]
            q_blk = _move_to_tlxy(
                np.asarray(q_var[start:stop, ...], dtype=float),
                q_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, :, ::lat_stride, ::lon_stride]
            w_blk = _move_to_tlxy(
                np.asarray(w_var[start:stop, ...], dtype=float),
                w_dims,
                time_name=time_name,
                level_name=level_name,
                lat_name=lat_name,
                lon_name=lon_name,
            )[:, :, ::lat_stride, ::lon_stride]

            temp_safe = np.clip(temp_blk, 150.0, 350.0)
            s_dry = CP_D * np.log(temp_safe / T0_K)
            s_moist = s_dry + L_V * q_blk / temp_safe
            j_dry = w_blk * s_dry
            j_moist = w_blk * s_moist
            j_anom = j_moist - j_dry
            h_moist = CP_D * temp_safe + L_V * q_blk
            dq_in = -w_blk * h_moist

            for tx in taxa:
                mask = taxa_masks[tx.name]
                wv = taxa_weights[tx.name]
                # shape [t,y,x]
                j_anom_tx = np.einsum("tlyx,l->tyx", j_anom[:, mask, :, :], wv, optimize=True)
                j_moist_tx = np.einsum("tlyx,l->tyx", j_moist[:, mask, :, :], wv, optimize=True)
                j_dry_tx = np.einsum("tlyx,l->tyx", j_dry[:, mask, :, :], wv, optimize=True)
                dq_in_tx = np.einsum("tlyx,l->tyx", dq_in[:, mask, :, :], wv, optimize=True)

                for region, aw in region_weights.items():
                    cols[f"j_anom_{tx.name}_{region}"][start:stop] = _weighted_mean_2d(j_anom_tx, aw)
                    cols[f"j_moist_{tx.name}_{region}"][start:stop] = _weighted_mean_2d(j_moist_tx, aw)
                    cols[f"j_dry_{tx.name}_{region}"][start:stop] = _weighted_mean_2d(j_dry_tx, aw)
                    cols[f"dq_in_{tx.name}_{region}"][start:stop] = _weighted_mean_2d(dq_in_tx, aw)

            print(f"[experiment_O] flux progress: {stop}/{nt}", flush=True)

    out = pd.DataFrame(
        {
            "time_index": np.arange(len(time_ns), dtype=int),
            "time": pd.to_datetime(time_ns).astype(str),
        }
    )
    for k, v in cols.items():
        out[k] = v
    return out, split_eff


def _plot_test_timeseries(
    *,
    time_test: pd.Series,
    y_test: np.ndarray,
    yhat_base: np.ndarray,
    yhat_full: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x = pd.to_datetime(time_test)
    ax.plot(x, y_test, color="#1f77b4", lw=1.2, label="Observed dS_hor proxy")
    ax.plot(x, yhat_base, color="#ff7f0e", lw=1.2, alpha=0.9, label="Clausius baseline")
    ax.plot(x, yhat_full, color="#2ca02c", lw=1.2, alpha=0.9, label="Clausius + Lambda")
    ax.set_title("Experiment O: Clausius fit on FT (test 2019)")
    ax.set_ylabel("dS_hor proxy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_regional_r2(region_df: pd.DataFrame, out_path: Path) -> None:
    d = region_df.copy()
    d = d.sort_values("region")
    x = np.arange(len(d))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(x - w / 2.0, d["r2_base"].to_numpy(dtype=float), width=w, label="Baseline", color="#ff7f0e")
    ax.bar(x + w / 2.0, d["r2_full"].to_numpy(dtype=float), width=w, label="Baseline + Lambda", color="#2ca02c")
    ax.set_xticks(x, d["region"].tolist())
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_ylabel("R^2 (test 2019)")
    ax.set_title("Experiment O: Spatial coherence (FT taxon)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_experiment(
    *,
    input_nc: Path,
    lambda_csv: Path,
    outdir: Path,
    taxa_specs: list[str],
    west_split_lon: float,
    lat_stride: int,
    lon_stride: int,
    batch_size: int,
    train_end_year: int,
    test_year: int,
    n_folds: int,
    alpha_grid: list[float],
    n_perm: int,
    perm_block: int,
    n_boot: int,
    seed: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    taxa = _parse_taxa(taxa_specs)

    print("[experiment_O] Step 1/4: compute entropy-flux anomaly timeseries...", flush=True)
    flux_df, split_eff = _compute_flux_timeseries(
        input_nc=input_nc,
        taxa=taxa,
        west_split_lon=west_split_lon,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        batch_size=batch_size,
    )

    lambda_df = pd.read_csv(lambda_csv, usecols=["time_index", "lambda_struct"]).rename(
        columns={"lambda_struct": "lambda_raw"}
    )
    df = flux_df.merge(lambda_df, on="time_index", how="inner").sort_values("time_index").reset_index(drop=True)
    if len(df) != len(flux_df):
        raise ValueError("Failed to align flux and lambda tables by time_index.")

    df["time_dt"] = pd.to_datetime(df["time"])
    df["year"] = df["time_dt"].dt.year
    df["quarter"] = df["time_dt"].dt.quarter
    df.to_csv(outdir / "entropy_flux_dataset.csv", index=False)

    # Clausius-consistent primary mapping:
    # dS_hor proxy <- entropy-flux anomaly, dQ_in proxy <- moist enthalpy inflow.
    target_col = "j_anom_FT_all"
    dq_col = "dq_in_FT_all"
    for col in (target_col, dq_col):
        if col not in df.columns:
            raise ValueError(f"Required primary column '{col}' missing. Available: {df.columns.tolist()}")

    train_idx = np.where(df["year"].to_numpy(dtype=int) <= train_end_year)[0]
    test_idx = np.where(df["year"].to_numpy(dtype=int) == test_year)[0]
    if len(train_idx) < n_folds * 16:
        raise ValueError("Too few train samples for blocked CV.")
    if len(test_idx) < 100:
        raise ValueError("Too few test samples.")

    lambda_raw = df["lambda_raw"].to_numpy(dtype=float)
    lambda_z, lam_mu, lam_sd = _standardize_with_train(lambda_raw, train_idx)

    def _build_design(dq_series: np.ndarray, tr_idx: np.ndarray, lam_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        dq_z, dq_mu, dq_sd = _standardize_with_train(dq_series, tr_idx)
        x_base_local = dq_z[:, None]
        x_full_local = np.column_stack([dq_z, lam_z])
        return x_base_local, x_full_local, dq_mu, dq_sd

    dq_raw = df[dq_col].to_numpy(dtype=float)
    x_base, x_full, dq_mu, dq_sd = _build_design(dq_raw, train_idx, lambda_z)
    y = df[target_col].to_numpy(dtype=float)

    print("[experiment_O] Step 2/4: blocked CV + one-SE alpha selection...", flush=True)
    cv_folds, selected = _cv_alpha_grid(
        y=y,
        x_base=x_base,
        x_full=x_full,
        train_idx=train_idx,
        n_folds=n_folds,
        alpha_grid=alpha_grid,
    )
    cv_folds.to_csv(outdir / "cv_folds.csv", index=False)
    cv_stats = (
        cv_folds.groupby("alpha", as_index=False)
        .agg(
            cv_gain_mean=("gain_r2", "mean"),
            cv_gain_median=("gain_r2", "median"),
            cv_gain_std=("gain_r2", "std"),
            cv_r2_base_mean=("r2_base", "mean"),
            cv_r2_full_mean=("r2_full", "mean"),
        )
        .sort_values("alpha")
    )
    cv_stats.to_csv(outdir / "cv_stats.csv", index=False)
    selected_alpha = float(selected["alpha"])
    pd.DataFrame([selected]).to_csv(outdir / "selected_config.csv", index=False)
    print(f"[experiment_O] selected alpha={selected_alpha:.2e}", flush=True)

    print("[experiment_O] Step 3/4: out-of-time test + permutation + bootstrap...", flush=True)
    yhat_base_test, yhat_full_test = _fit_with_alpha(
        y=y,
        x_base=x_base,
        x_full=x_full,
        train_idx=train_idx,
        eval_idx=test_idx,
        alpha=selected_alpha,
    )
    r2_base_test = _r2(y[test_idx], yhat_base_test)
    r2_full_test = _r2(y[test_idx], yhat_full_test)
    gain_r2_test = float(r2_full_test - r2_base_test)
    coef_base, intercept_base, _ = _fit_ridge_scaled(x_base[train_idx], y[train_idx], x_base[test_idx], selected_alpha)
    coef_full, intercept_full, _ = _fit_ridge_scaled(x_full[train_idx], y[train_idx], x_full[test_idx], selected_alpha)
    inv_teff_base = float(coef_base[0]) if len(coef_base) >= 1 else np.nan
    teff_base = float(1.0 / inv_teff_base) if abs(inv_teff_base) > 1e-12 else np.nan
    inv_teff_full = float(coef_full[0]) if len(coef_full) >= 1 else np.nan
    lambda_coef_full = float(coef_full[1]) if len(coef_full) >= 2 else np.nan

    rng = np.random.default_rng(seed)
    perm_rows: list[dict[str, float | int]] = []
    ge_count = 0
    for pid in range(n_perm):
        lam_perm = _block_permute(lambda_raw, block=perm_block, rng=rng)
        lam_perm_z, _, _ = _standardize_with_train(lam_perm, train_idx)
        x_full_perm = np.column_stack([x_base[:, 0], lam_perm_z])
        yhat_b_p, yhat_f_p = _fit_with_alpha(
            y=y,
            x_base=x_base,
            x_full=x_full_perm,
            train_idx=train_idx,
            eval_idx=test_idx,
            alpha=selected_alpha,
        )
        r2_b_p = _r2(y[test_idx], yhat_b_p)
        r2_f_p = _r2(y[test_idx], yhat_f_p)
        gain_p = float(r2_f_p - r2_b_p)
        if gain_p >= gain_r2_test:
            ge_count += 1
        perm_rows.append({"perm_id": int(pid), "gain_r2_perm": gain_p, "r2_full_perm": float(r2_f_p)})
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(outdir / "permutation_test.csv", index=False)
    perm_p = float((ge_count + 1) / (n_perm + 1))

    boot_rows: list[dict[str, float | int]] = []
    n_test = len(test_idx)
    for bid in range(n_boot):
        rel = _bootstrap_rel_indices(n=n_test, block=perm_block, rng=rng)
        idx = test_idx[rel]
        r2_b = _r2(y[idx], yhat_base_test[rel])
        r2_f = _r2(y[idx], yhat_full_test[rel])
        gain = float(r2_f - r2_b)
        boot_rows.append({"boot_id": int(bid), "r2_base_boot": r2_b, "r2_full_boot": r2_f, "gain_r2_boot": gain})
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(outdir / "bootstrap_metrics.csv", index=False)

    ci_r2_full = boot_df["r2_full_boot"].quantile([0.025, 0.975]).to_numpy(dtype=float)
    ci_gain = boot_df["gain_r2_boot"].quantile([0.025, 0.975]).to_numpy(dtype=float)

    # Spatial coherence check on FT for all/west/east using the selected alpha.
    region_rows: list[dict[str, float | str]] = []
    for region in ("all", "west", "east"):
        y_reg = df[f"j_anom_FT_{region}"].to_numpy(dtype=float)
        dq_reg = df[f"dq_in_FT_{region}"].to_numpy(dtype=float)
        x_base_reg, x_full_reg, _, _ = _build_design(dq_reg, train_idx, lambda_z)
        yhat_b_reg, yhat_f_reg = _fit_with_alpha(
            y=y_reg,
            x_base=x_base_reg,
            x_full=x_full_reg,
            train_idx=train_idx,
            eval_idx=test_idx,
            alpha=selected_alpha,
        )
        r2_b = _r2(y_reg[test_idx], yhat_b_reg)
        r2_f = _r2(y_reg[test_idx], yhat_f_reg)
        region_rows.append(
            {
                "region": region,
                "r2_base": float(r2_b),
                "r2_full": float(r2_f),
                "gain_r2": float(r2_f - r2_b),
            }
        )
    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(outdir / "regional_metrics.csv", index=False)

    # Taxon summary in all-domain (BL,FT,UT).
    taxon_rows: list[dict[str, float | str]] = []
    for tx in taxa:
        col = f"j_anom_{tx.name}_all"
        dq_tx_col = f"dq_in_{tx.name}_all"
        y_tx = df[col].to_numpy(dtype=float)
        dq_tx = df[dq_tx_col].to_numpy(dtype=float)
        x_base_tx, x_full_tx, _, _ = _build_design(dq_tx, train_idx, lambda_z)
        yhat_b_tx, yhat_f_tx = _fit_with_alpha(
            y=y_tx,
            x_base=x_base_tx,
            x_full=x_full_tx,
            train_idx=train_idx,
            eval_idx=test_idx,
            alpha=selected_alpha,
        )
        r2_b = _r2(y_tx[test_idx], yhat_b_tx)
        r2_f = _r2(y_tx[test_idx], yhat_f_tx)
        taxon_rows.append({"taxon": tx.name, "r2_base": float(r2_b), "r2_full": float(r2_f), "gain_r2": float(r2_f - r2_b)})
    taxon_df = pd.DataFrame(taxon_rows)
    taxon_df.to_csv(outdir / "taxon_metrics.csv", index=False)

    print("[experiment_O] Step 4/4: quarterly rolling-origin diagnostics + report...", flush=True)
    q_rows: list[dict[str, float | int]] = []
    for q in (1, 2, 3, 4):
        mask_test = (df["year"] == test_year) & (df["quarter"] == q)
        if int(mask_test.sum()) == 0:
            continue
        q_start = pd.Timestamp(year=test_year, month=3 * (q - 1) + 1, day=1)
        mask_train = df["time_dt"] < q_start
        tr = np.where(mask_train.to_numpy())[0]
        te = np.where(mask_test.to_numpy())[0]
        if len(tr) < 500 or len(te) < 40:
            continue

        dq_q = dq_raw
        dq_q_z, _, _ = _standardize_with_train(dq_q, tr)
        lam_q_z, _, _ = _standardize_with_train(lambda_raw, tr)
        x_base_q = dq_q_z[:, None]
        x_full_q = np.column_stack([dq_q_z, lam_q_z])
        yb_q, yf_q = _fit_with_alpha(
            y=y,
            x_base=x_base_q,
            x_full=x_full_q,
            train_idx=tr,
            eval_idx=te,
            alpha=selected_alpha,
        )
        r2_b_q = _r2(y[te], yb_q)
        r2_f_q = _r2(y[te], yf_q)
        q_rows.append(
            {
                "quarter": int(q),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "r2_base": float(r2_b_q),
                "r2_full": float(r2_f_q),
                "gain_r2": float(r2_f_q - r2_b_q),
            }
        )
    q_df = pd.DataFrame(q_rows)
    q_df.to_csv(outdir / "quarterly_summary.csv", index=False)

    timeseries = df[["time_index", "time", target_col, dq_col, "lambda_raw"]].copy()
    timeseries = timeseries.rename(columns={target_col: "dS_hor_proxy", dq_col: "dQ_in_proxy"})
    pred_b = np.full(len(df), np.nan, dtype=float)
    pred_f = np.full(len(df), np.nan, dtype=float)
    pred_b[test_idx] = yhat_base_test
    pred_f[test_idx] = yhat_full_test
    timeseries["y_test_pred_base"] = pred_b
    timeseries["y_test_pred_full"] = pred_f
    timeseries.to_csv(outdir / "timeseries_primary.csv", index=False)

    test_row = {
        "target": "dS_hor_proxy=j_anom_FT_all",
        "predictor_clausius": "dQ_in_proxy=dq_in_FT_all",
        "selected_alpha": selected_alpha,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "inv_teff_base": float(inv_teff_base),
        "teff_base": float(teff_base),
        "intercept_base": float(intercept_base),
        "inv_teff_full": float(inv_teff_full),
        "lambda_coef_full": float(lambda_coef_full),
        "intercept_full": float(intercept_full),
        "dq_in_mu_train": float(dq_mu),
        "dq_in_sd_train": float(dq_sd),
        "r2_base_test": float(r2_base_test),
        "r2_full_test": float(r2_full_test),
        "gain_r2_test": float(gain_r2_test),
        "r2_full_ci95_lo": float(ci_r2_full[0]),
        "r2_full_ci95_hi": float(ci_r2_full[1]),
        "gain_r2_ci95_lo": float(ci_gain[0]),
        "gain_r2_ci95_hi": float(ci_gain[1]),
        "perm_p_value": float(perm_p),
        "lambda_mu_train": float(lam_mu),
        "lambda_sd_train": float(lam_sd),
        "train_end_year": int(train_end_year),
        "test_year": int(test_year),
        "lat_stride": int(lat_stride),
        "lon_stride": int(lon_stride),
        "west_split_lon": float(split_eff),
    }
    pd.DataFrame([test_row]).to_csv(outdir / "test_metrics.csv", index=False)

    _plot_test_timeseries(
        time_test=df.loc[test_idx, "time"],
        y_test=y[test_idx],
        yhat_base=yhat_base_test,
        yhat_full=yhat_full_test,
        out_path=outdir / "plot_test_timeseries_ft_all.png",
    )
    _plot_regional_r2(region_df, outdir / "plot_regional_r2_ft.png")

    report_lines = [
        "# Experiment O: Clausius-Consistent Thermodynamic Test",
        "",
        "## Setup",
        f"- Input: `{input_nc}`",
        f"- Lambda source: `{lambda_csv}`",
        f"- Taxa: `{', '.join(taxa_specs)}`",
        f"- Train years <= {train_end_year}, test year = {test_year}",
        "- Baseline equation: dS_hor_proxy ~ (1/T_eff)*dQ_in_proxy + b",
        "- Full equation: dS_hor_proxy ~ (1/T_eff)*dQ_in_proxy + c*Lambda + b",
        "",
        "## Primary Test (FT all-domain)",
        f"- Clausius slope 1/T_eff (baseline): {inv_teff_base:.6f}",
        f"- Clausius T_eff (baseline): {teff_base:.6f}",
        f"- Lambda coefficient (full): {lambda_coef_full:.6f}",
        f"- R2 baseline: {r2_base_test:.6f}",
        f"- R2 full (Clausius + Lambda): {r2_full_test:.6f}",
        f"- Gain (R2_full - R2_base): {gain_r2_test:.6f}",
        f"- CI95 R2_full: [{ci_r2_full[0]:.6f}, {ci_r2_full[1]:.6f}]",
        f"- CI95 gain: [{ci_gain[0]:.6f}, {ci_gain[1]:.6f}]",
        f"- permutation p-value: {perm_p:.6f}",
        "",
        "## Spatial Coherence (FT)",
        *(f"- {row['region']}: gain={row['gain_r2']:.6f}" for _, row in region_df.iterrows()),
        "",
        "## Quarterly Rolling-Origin (2019)",
        (
            f"- mean gain: {float(np.nanmean(q_df['gain_r2'])):.6f}, "
            f"min gain: {float(np.nanmin(q_df['gain_r2'])):.6f}, "
            f"positive quarter frac: {float(np.nanmean(q_df['gain_r2'] > 0.0)):.3f}"
            if not q_df.empty
            else "- quarterly metrics unavailable"
        ),
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("[experiment_O] done.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-nc",
        type=Path,
        default=Path("data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc"),
    )
    parser.add_argument(
        "--lambda-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_O_entropy_equilibrium"),
    )
    parser.add_argument(
        "--taxa",
        nargs="+",
        default=["BL:1000-850", "FT:850-300", "UT:300-100"],
    )
    parser.add_argument("--west-split-lon", type=float, default=140.0)
    parser.add_argument("--lat-stride", type=int, default=1)
    parser.add_argument("--lon-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--n-perm", type=int, default=140)
    parser.add_argument("--perm-block", type=int, default=24)
    parser.add_argument("--n-boot", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        input_nc=args.input_nc,
        lambda_csv=args.lambda_csv,
        outdir=args.outdir,
        taxa_specs=list(args.taxa),
        west_split_lon=args.west_split_lon,
        lat_stride=args.lat_stride,
        lon_stride=args.lon_stride,
        batch_size=args.batch_size,
        train_end_year=args.train_end_year,
        test_year=args.test_year,
        n_folds=args.n_folds,
        alpha_grid=list(args.alpha_grid),
        n_perm=args.n_perm,
        perm_block=args.perm_block,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
