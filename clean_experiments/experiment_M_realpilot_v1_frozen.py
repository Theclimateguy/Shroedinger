#!/usr/bin/env python3
"""Experiment M-realpilot: local structural proxy test on MRMS + GOES ABI/GLM.

Core idea:
- baseline: local MRMS + time context
- full: baseline + local structural multiscale proxy (ABI + GLM + coupling)
- evaluate MAE gain with leave-one-event-out CV
- validate with placebo permutations (time-shuffle and event-shuffle)
"""

from __future__ import annotations

import argparse
import gzip
import math
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import xarray as xr


def _parse_utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _patch_stats(mask: np.ndarray) -> tuple[float, float]:
    if mask.size == 0:
        return 0.0, 0.0
    active = int(mask.sum())
    if active == 0:
        return 0.0, 0.0
    labels, ncomp = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if ncomp <= 0:
        return 0.0, 0.0
    counts = np.bincount(labels.ravel())[1:]
    if counts.size == 0:
        return float(ncomp), 0.0
    largest_frac = float(counts.max() / active)
    return float(ncomp), largest_frac


def _safe_log1p(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(np.log1p(max(0.0, x)))


def _read_mrms_grib_array(grib_gz_path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with gzip.open(grib_gz_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        ds = xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        try:
            if len(ds.data_vars) == 0:
                raise ValueError(f"No data variables in MRMS GRIB: {grib_gz_path}")
            var = next(iter(ds.data_vars))
            arr = np.asarray(ds[var].values, dtype=np.float32)
        finally:
            ds.close()
        return arr
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_mrms_features(grib_gz_path: Path, downsample: int, patch_threshold: float) -> dict[str, float]:
    arr = _read_mrms_grib_array(grib_gz_path)
    valid = np.isfinite(arr) & (arr >= 0.0)
    if int(valid.sum()) == 0:
        return {
            "mrms_mean": np.nan,
            "mrms_std": np.nan,
            "mrms_p90": np.nan,
            "mrms_p95": np.nan,
            "mrms_max": np.nan,
            "mrms_area_gt2": np.nan,
            "mrms_area_gt5": np.nan,
            "mrms_patch_count_gt5": np.nan,
            "mrms_patch_largest_frac_gt5": np.nan,
        }

    vals = arr[valid]
    ds = max(1, int(downsample))
    arr_d = arr[::ds, ::ds]
    valid_d = np.isfinite(arr_d) & (arr_d >= 0.0)
    denom = max(1, int(valid_d.sum()))

    mask_gt2 = valid_d & (arr_d >= 2.0)
    mask_gt5 = valid_d & (arr_d >= patch_threshold)
    patch_count, patch_largest_frac = _patch_stats(mask_gt5)

    return {
        "mrms_mean": float(np.nanmean(vals)),
        "mrms_std": float(np.nanstd(vals)),
        "mrms_p90": float(np.nanpercentile(vals, 90)),
        "mrms_p95": float(np.nanpercentile(vals, 95)),
        "mrms_max": float(np.nanmax(vals)),
        "mrms_area_gt2": float(mask_gt2.sum() / denom),
        "mrms_area_gt5": float(mask_gt5.sum() / denom),
        "mrms_patch_count_gt5": float(patch_count),
        "mrms_patch_largest_frac_gt5": float(patch_largest_frac),
    }


def _extract_abi_features(abi_nc_path: Path, downsample: int, cold_threshold_k: float) -> dict[str, float]:
    ds = xr.open_dataset(abi_nc_path, engine="netcdf4")
    try:
        if "CMI" in ds:
            arr = np.asarray(ds["CMI"].values, dtype=np.float32)
        elif len(ds.data_vars) > 0:
            var = next(iter(ds.data_vars))
            arr = np.asarray(ds[var].values, dtype=np.float32)
        else:
            raise ValueError(f"No data vars found in ABI file: {abi_nc_path}")
    finally:
        ds.close()

    step = max(1, int(downsample))
    a = arr[::step, ::step]
    finite = np.isfinite(a)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {
            "abi_cmi_mean": np.nan,
            "abi_cmi_std": np.nan,
            "abi_cold_frac_235": np.nan,
            "abi_cold_frac_220": np.nan,
            "abi_grad_mean": np.nan,
            "abi_patch_count_235": np.nan,
            "abi_patch_largest_frac_235": np.nan,
        }

    vals = a[finite]
    fill = float(np.nanmedian(vals))
    a_fill = np.where(finite, a, fill)
    gy, gx = np.gradient(a_fill)
    grad = np.hypot(gx, gy)

    mask_cold_235 = finite & (a < cold_threshold_k)
    mask_cold_220 = finite & (a < 220.0)
    patch_count, patch_largest_frac = _patch_stats(mask_cold_235)

    return {
        "abi_cmi_mean": float(np.nanmean(vals)),
        "abi_cmi_std": float(np.nanstd(vals)),
        "abi_cold_frac_235": float(mask_cold_235.sum() / n_finite),
        "abi_cold_frac_220": float(mask_cold_220.sum() / n_finite),
        "abi_grad_mean": float(np.nanmean(grad[finite])),
        "abi_patch_count_235": float(patch_count),
        "abi_patch_largest_frac_235": float(patch_largest_frac),
    }


def _extract_glm_features(glm_nc_path: Path) -> dict[str, float]:
    ds = xr.open_dataset(glm_nc_path, engine="netcdf4")
    try:
        if "flash_area" in ds:
            flash_area = np.asarray(ds["flash_area"].values, dtype=np.float64)
        elif "group_area" in ds:
            flash_area = np.asarray(ds["group_area"].values, dtype=np.float64)
        else:
            flash_area = np.asarray([], dtype=np.float64)

        if "flash_energy" in ds:
            flash_energy = np.asarray(ds["flash_energy"].values, dtype=np.float64)
        elif "group_energy" in ds:
            flash_energy = np.asarray(ds["group_energy"].values, dtype=np.float64)
        else:
            flash_energy = np.asarray([], dtype=np.float64)

        if "group_area" in ds:
            group_area = np.asarray(ds["group_area"].values, dtype=np.float64)
        else:
            group_area = np.asarray([], dtype=np.float64)
    finally:
        ds.close()

    flash_count = int(flash_area.size)
    flash_area_sum = float(np.nansum(flash_area)) if flash_area.size > 0 else 0.0
    flash_area_mean = float(np.nanmean(flash_area)) if flash_area.size > 0 else 0.0
    flash_energy_sum = float(np.nansum(flash_energy)) if flash_energy.size > 0 else 0.0
    group_count = int(group_area.size)

    return {
        "glm_flash_count": float(flash_count),
        "glm_flash_area_sum": float(flash_area_sum),
        "glm_flash_area_mean": float(flash_area_mean),
        "glm_flash_energy_sum": float(flash_energy_sum),
        "glm_group_count": float(group_count),
    }


def _extract_row_features(
    row: pd.Series,
    mrms_downsample: int,
    abi_downsample: int,
    mrms_patch_threshold: float,
    abi_cold_threshold_k: float,
) -> dict[str, float | str]:
    event_id = str(row["event_id"])
    t_mrms = _parse_utc(str(row["mrms_obs_time_utc"]))
    hour = int(t_mrms.hour)
    base = {
        "event_id": event_id,
        "mrms_obs_time_utc": t_mrms.isoformat().replace("+00:00", "Z"),
        "hour_utc": float(hour),
        "hour_sin": float(math.sin(2.0 * math.pi * hour / 24.0)),
        "hour_cos": float(math.cos(2.0 * math.pi * hour / 24.0)),
        "mrms_local_path": str(row["mrms_local_path"]),
        "abi_local_path": str(row["abi_local_path"]),
        "glm_local_path": str(row["glm_local_path"]),
    }

    mrms_f = _extract_mrms_features(
        Path(str(row["mrms_local_path"])),
        downsample=mrms_downsample,
        patch_threshold=mrms_patch_threshold,
    )
    abi_f = _extract_abi_features(
        Path(str(row["abi_local_path"])),
        downsample=abi_downsample,
        cold_threshold_k=abi_cold_threshold_k,
    )
    glm_f = _extract_glm_features(Path(str(row["glm_local_path"])))

    out = {**base, **mrms_f, **abi_f, **glm_f}
    out["glm_flash_count_log"] = _safe_log1p(float(out["glm_flash_count"]))
    out["glm_flash_area_sum_log"] = _safe_log1p(float(out["glm_flash_area_sum"]))
    out["glm_flash_energy_sum_log"] = _safe_log1p(float(out["glm_flash_energy_sum"]))
    out["convective_coupling_index"] = float(out["abi_cold_frac_235"]) * float(out["glm_flash_count_log"])
    out["convective_coupling_area"] = float(out["abi_patch_largest_frac_235"]) * float(out["glm_flash_area_sum_log"])
    return out


def _build_feature_dataset(
    panel_df: pd.DataFrame,
    outdir: Path,
    mrms_downsample: int,
    abi_downsample: int,
    mrms_patch_threshold: float,
    abi_cold_threshold_k: float,
) -> pd.DataFrame:
    rows = []
    n = len(panel_df)
    for i, (_, r) in enumerate(panel_df.iterrows(), start=1):
        print(f"[extract] row {i}/{n} event={r['event_id']} time={r['mrms_obs_time_utc']}", flush=True)
        feat = _extract_row_features(
            r,
            mrms_downsample=mrms_downsample,
            abi_downsample=abi_downsample,
            mrms_patch_threshold=mrms_patch_threshold,
            abi_cold_threshold_k=abi_cold_threshold_k,
        )
        rows.append(feat)

    feat_df = pd.DataFrame(rows).sort_values(["event_id", "mrms_obs_time_utc"]).reset_index(drop=True)
    feat_df.to_csv(outdir / "feature_dataset.csv", index=False)
    return feat_df


def _build_supervised_rows(feat_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    work = feat_df.copy()
    work["mrms_obs_time_utc"] = pd.to_datetime(work["mrms_obs_time_utc"], utc=True)
    work = work.sort_values(["event_id", "mrms_obs_time_utc"]).reset_index(drop=True)

    for event_id, grp in work.groupby("event_id", sort=True):
        g = grp.reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            row = {
                "event_id": str(event_id),
                "step_idx": float(i),
                "mrms_obs_time_utc": cur["mrms_obs_time_utc"].isoformat().replace("+00:00", "Z"),
                "base_prev_p95": float(cur["mrms_p95"]),
                "base_prev_area_gt5": float(cur["mrms_area_gt5"]),
                "base_prev_patch_count_gt5": float(cur["mrms_patch_count_gt5"]),
                "hour_sin": float(cur["hour_sin"]),
                "hour_cos": float(cur["hour_cos"]),
                "abi_cold_frac_235": float(cur["abi_cold_frac_235"]),
                "abi_grad_mean": float(cur["abi_grad_mean"]),
                "abi_cmi_std": float(cur["abi_cmi_std"]),
                "abi_patch_largest_frac_235": float(cur["abi_patch_largest_frac_235"]),
                "glm_flash_count_log": float(cur["glm_flash_count_log"]),
                "glm_flash_area_sum_log": float(cur["glm_flash_area_sum_log"]),
                "glm_flash_energy_sum_log": float(cur["glm_flash_energy_sum_log"]),
                "convective_coupling_index": float(cur["convective_coupling_index"]),
                "convective_coupling_area": float(cur["convective_coupling_area"]),
                "target_next_mrms_p95": float(nxt["mrms_p95"]),
                "target_next_mrms_area_gt5": float(nxt["mrms_area_gt5"]),
                "target_delta_mrms_p95": float(nxt["mrms_p95"] - cur["mrms_p95"]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


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
    return model.predict(x_te_s)


def _evaluate_logo(
    df: pd.DataFrame,
    target_col: str,
    baseline_cols: Sequence[str],
    structural_cols: Sequence[str],
    alpha: float,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    data = df.reset_index(drop=True).copy()
    y = data[target_col].to_numpy(dtype=float)
    groups = data["event_id"].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    pred_base = np.full(len(data), np.nan, dtype=float)
    pred_full = np.full(len(data), np.nan, dtype=float)
    fold_rows: list[dict[str, float | str]] = []

    for fold_id, (tr_idx, te_idx) in enumerate(logo.split(data, y, groups=groups), start=1):
        tr = data.iloc[tr_idx]
        te = data.iloc[te_idx]
        x_base_tr = tr[list(baseline_cols)].to_numpy(dtype=float)
        x_base_te = te[list(baseline_cols)].to_numpy(dtype=float)
        x_full_tr = tr[list(baseline_cols) + list(structural_cols)].to_numpy(dtype=float)
        x_full_te = te[list(baseline_cols) + list(structural_cols)].to_numpy(dtype=float)
        y_tr = tr[target_col].to_numpy(dtype=float)
        y_te = te[target_col].to_numpy(dtype=float)

        pb = _ridge_predict(x_base_tr, y_tr, x_base_te, alpha=alpha)
        pf = _ridge_predict(x_full_tr, y_tr, x_full_te, alpha=alpha)
        pred_base[te_idx] = pb
        pred_full[te_idx] = pf

        mae_b = float(mean_absolute_error(y_te, pb))
        mae_f = float(mean_absolute_error(y_te, pf))
        event_name = str(te["event_id"].iloc[0])
        fold_rows.append(
            {
                "fold_id": float(fold_id),
                "event_id": event_name,
                "n_test": float(len(te_idx)),
                "mae_baseline": mae_b,
                "mae_full": mae_f,
                "mae_gain": mae_b - mae_f,
            }
        )

    oof = data[["event_id", "step_idx", "mrms_obs_time_utc", target_col, "base_prev_p95"]].copy()
    oof["pred_baseline"] = pred_base
    oof["pred_full"] = pred_full
    oof["abs_err_baseline"] = np.abs(oof[target_col] - oof["pred_baseline"])
    oof["abs_err_full"] = np.abs(oof[target_col] - oof["pred_full"])
    oof["pointwise_gain"] = oof["abs_err_baseline"] - oof["abs_err_full"]

    mae_base = float(mean_absolute_error(oof[target_col], oof["pred_baseline"]))
    mae_full = float(mean_absolute_error(oof[target_col], oof["pred_full"]))
    fold_df = pd.DataFrame(fold_rows)
    event_positive_frac = float((fold_df["mae_gain"] > 0.0).mean()) if len(fold_df) > 0 else np.nan
    min_event_gain = float(fold_df["mae_gain"].min()) if len(fold_df) > 0 else np.nan

    summary = {
        "mae_baseline": mae_base,
        "mae_full": mae_full,
        "mean_mae_gain": mae_base - mae_full,
        "min_event_gain": min_event_gain,
        "event_positive_frac": event_positive_frac,
    }
    return summary, oof, fold_df


def _apply_structural_permutation(
    df: pd.DataFrame,
    structural_cols: Sequence[str],
    mode: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if mode == "time_shuffle":
        perm = rng.permutation(len(out))
        for c in structural_cols:
            out[c] = out[c].to_numpy()[perm]
        return out

    if mode == "event_shuffle":
        events = sorted(out["event_id"].astype(str).unique())
        donors = events.copy()
        rng.shuffle(donors)
        mapping = dict(zip(events, donors))

        src = out.copy()
        src["event_id"] = src["event_id"].astype(str)
        src["step_idx"] = src["step_idx"].astype(int)

        for c in structural_cols:
            new_vals = []
            for _, r in out.iterrows():
                e = str(r["event_id"])
                s = int(r["step_idx"])
                donor_event = mapping[e]
                cand = src[(src["event_id"] == donor_event) & (src["step_idx"] == s)]
                if len(cand) == 0:
                    cand = src[src["event_id"] == donor_event]
                if len(cand) == 0:
                    new_vals.append(np.nan)
                else:
                    new_vals.append(float(cand.iloc[0][c]))
            out[c] = np.asarray(new_vals, dtype=float)
        return out

    raise ValueError(f"Unknown permutation mode: {mode}")


def _permutation_test(
    df: pd.DataFrame,
    target_col: str,
    baseline_cols: Sequence[str],
    structural_cols: Sequence[str],
    alpha: float,
    n_perm: int,
    seed: int,
    mode: str,
) -> tuple[float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    gains = []
    for i in range(1, int(n_perm) + 1):
        dp = _apply_structural_permutation(df, structural_cols=structural_cols, mode=mode, rng=rng)
        summary, _, _ = _evaluate_logo(
            dp,
            target_col=target_col,
            baseline_cols=baseline_cols,
            structural_cols=structural_cols,
            alpha=alpha,
        )
        gains.append(float(summary["mean_mae_gain"]))
    perm_df = pd.DataFrame({"perm_id": np.arange(1, n_perm + 1, dtype=int), "mode": mode, "mean_mae_gain": gains})
    return float(np.asarray(gains, dtype=float).mean()), perm_df


def _plot_permutation(real_gain: float, perm_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = perm_df["mean_mae_gain"].to_numpy(dtype=float)
    ax.hist(vals, bins=20, color="#6c8ebf", alpha=0.85, edgecolor="white")
    ax.axvline(real_gain, color="#c0392b", linestyle="--", linewidth=2, label=f"real gain={real_gain:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Mean MAE gain (baseline - full)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_report(
    outdir: Path,
    panel_csv: Path,
    target_col: str,
    summary: dict[str, float | bool],
    baseline_cols: Sequence[str],
    structural_cols: Sequence[str],
) -> None:
    text = "\n".join(
        [
            "# Experiment M-realpilot (Local Structural Proxy)",
            "",
            f"- input panel: `{panel_csv}`",
            f"- target: `{target_col}`",
            f"- baseline features: `{', '.join(baseline_cols)}`",
            f"- structural features: `{', '.join(structural_cols)}`",
            "",
            "## Main metrics",
            f"- n_events: {int(summary['n_events'])}",
            f"- n_model_samples: {int(summary['n_model_samples'])}",
            f"- MAE baseline: {float(summary['mae_baseline']):.6f}",
            f"- MAE full: {float(summary['mae_full']):.6f}",
            f"- mean MAE gain: {float(summary['mean_mae_gain']):.6f}",
            f"- min event gain: {float(summary['min_event_gain']):.6f}",
            f"- event positive frac: {float(summary['event_positive_frac']):.3f}",
            "",
            "## Placebo tests",
            f"- time-shuffle p-value: {float(summary['perm_p_time_shuffle']):.6f}",
            f"- event-shuffle p-value: {float(summary['perm_p_event_shuffle']):.6f}",
            "",
            "## Active vs calm",
            f"- active quantile: {float(summary['active_quantile']):.2f}",
            f"- mean gain active: {float(summary['mean_gain_active']):.6f}",
            f"- mean gain calm: {float(summary['mean_gain_calm']):.6f}",
            f"- active - calm: {float(summary['active_minus_calm']):.6f}",
            "",
            "## Hypotheses",
            f"- H1-local (baseline+structural beats baseline): {bool(summary['H1_pass'])}",
            f"- H2-local (placebo does not reproduce gain): {bool(summary['H2_pass'])}",
            f"- H3-local (effect stronger in active windows): {bool(summary['H3_pass'])}",
            f"- PASS_ALL: {bool(summary['PASS_ALL'])}",
            "",
            "## Notes",
            "- Fixed feature set and fixed ridge alpha were used to avoid pilot-time retuning.",
            "- Event-level CV is leave-one-event-out (strict out-of-event validation).",
        ]
    )
    (outdir / "report.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    panel_df = pd.read_csv(args.panel_csv)
    required_cols = {"event_id", "mrms_obs_time_utc", "mrms_local_path", "abi_local_path", "glm_local_path"}
    missing = required_cols - set(panel_df.columns)
    if missing:
        raise ValueError(f"Panel CSV missing columns: {sorted(missing)}")

    print("[stage] 1/4 feature extraction", flush=True)
    feat_df = _build_feature_dataset(
        panel_df=panel_df,
        outdir=outdir,
        mrms_downsample=args.mrms_downsample,
        abi_downsample=args.abi_downsample,
        mrms_patch_threshold=args.mrms_patch_threshold,
        abi_cold_threshold_k=args.abi_cold_threshold_k,
    )

    print("[stage] 2/4 build supervised rows", flush=True)
    model_df = _build_supervised_rows(feat_df)
    if len(model_df) < 6:
        raise ValueError(f"Too few supervised rows ({len(model_df)}). Need >= 6.")
    model_df.to_csv(outdir / "modeling_dataset.csv", index=False)

    baseline_cols = ["base_prev_p95", "base_prev_area_gt5", "hour_sin", "hour_cos"]
    structural_cols = ["abi_cold_frac_235", "abi_grad_mean", "glm_flash_count_log", "convective_coupling_index"]

    target_col = "target_next_mrms_p95" if args.target == "next_p95" else "target_delta_mrms_p95"

    print("[stage] 3/4 fit/evaluate leave-one-event-out", flush=True)
    summary, oof_df, fold_df = _evaluate_logo(
        model_df,
        target_col=target_col,
        baseline_cols=baseline_cols,
        structural_cols=structural_cols,
        alpha=args.ridge_alpha,
    )
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)
    fold_df.to_csv(outdir / "fold_metrics.csv", index=False)

    print("[stage] 4/4 permutation/placebo tests", flush=True)
    _, perm_time = _permutation_test(
        model_df,
        target_col=target_col,
        baseline_cols=baseline_cols,
        structural_cols=structural_cols,
        alpha=args.ridge_alpha,
        n_perm=args.n_perm,
        seed=args.seed + 11,
        mode="time_shuffle",
    )
    _, perm_event = _permutation_test(
        model_df,
        target_col=target_col,
        baseline_cols=baseline_cols,
        structural_cols=structural_cols,
        alpha=args.ridge_alpha,
        n_perm=args.n_perm,
        seed=args.seed + 23,
        mode="event_shuffle",
    )
    perm_time.to_csv(outdir / "permutation_time_shuffle.csv", index=False)
    perm_event.to_csv(outdir / "permutation_event_shuffle.csv", index=False)

    real_gain = float(summary["mean_mae_gain"])
    p_time = float((1.0 + (perm_time["mean_mae_gain"] >= real_gain).sum()) / (1.0 + len(perm_time)))
    p_event = float((1.0 + (perm_event["mean_mae_gain"] >= real_gain).sum()) / (1.0 + len(perm_event)))
    _plot_permutation(
        real_gain=real_gain,
        perm_df=perm_time,
        out_path=outdir / "plot_permutation_time_shuffle.png",
        title="M-realpilot placebo: time-shuffle",
    )
    _plot_permutation(
        real_gain=real_gain,
        perm_df=perm_event,
        out_path=outdir / "plot_permutation_event_shuffle.png",
        title="M-realpilot placebo: event-shuffle",
    )

    active_thr = float(np.nanquantile(model_df["base_prev_p95"].to_numpy(dtype=float), args.active_quantile))
    active_mask = model_df["base_prev_p95"].to_numpy(dtype=float) >= active_thr
    point_gain = oof_df["pointwise_gain"].to_numpy(dtype=float)
    mean_gain_active = float(np.nanmean(point_gain[active_mask])) if int(active_mask.sum()) > 0 else np.nan
    mean_gain_calm = float(np.nanmean(point_gain[~active_mask])) if int((~active_mask).sum()) > 0 else np.nan
    active_minus_calm = float(mean_gain_active - mean_gain_calm)

    h1_pass = bool(real_gain > 0.0 and float(summary["event_positive_frac"]) >= 0.6)
    h2_pass = bool(p_time <= 0.05 and p_event <= 0.05 and real_gain > 0.0)
    h3_pass = bool(np.isfinite(active_minus_calm) and active_minus_calm > 0.0)
    pass_all = bool(h1_pass and h2_pass and h3_pass)

    summary_row = {
        "target": target_col,
        "n_events": float(model_df["event_id"].nunique()),
        "n_model_samples": float(len(model_df)),
        "mae_baseline": float(summary["mae_baseline"]),
        "mae_full": float(summary["mae_full"]),
        "mean_mae_gain": real_gain,
        "min_event_gain": float(summary["min_event_gain"]),
        "event_positive_frac": float(summary["event_positive_frac"]),
        "perm_p_time_shuffle": p_time,
        "perm_p_event_shuffle": p_event,
        "active_quantile": float(args.active_quantile),
        "active_threshold_p95": active_thr,
        "mean_gain_active": mean_gain_active,
        "mean_gain_calm": mean_gain_calm,
        "active_minus_calm": active_minus_calm,
        "ridge_alpha": float(args.ridge_alpha),
        "n_perm": float(args.n_perm),
        "H1_pass": h1_pass,
        "H2_pass": h2_pass,
        "H3_pass": h3_pass,
        "PASS_ALL": pass_all,
    }
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    _write_report(
        outdir=outdir,
        panel_csv=args.panel_csv,
        target_col=target_col,
        summary={**summary_row},
        baseline_cols=baseline_cols,
        structural_cols=structural_cols,
    )

    print("Experiment M-realpilot complete.", flush=True)
    print(f"Output: {outdir}", flush=True)
    print(f"mean_gain={real_gain:.6f} p_time={p_time:.6f} p_event={p_event:.6f} PASS_ALL={pass_all}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_dataset_panel_v1.csv"),
        help="Unified MRMS+ABI+GLM aligned panel (hour-level rows).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_realpilot_local_structural"),
    )
    p.add_argument("--target", choices=["next_p95", "delta_p95"], default="next_p95")
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--n-perm", type=int, default=499)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--active-quantile", type=float, default=0.67)
    p.add_argument("--mrms-downsample", type=int, default=8)
    p.add_argument("--abi-downsample", type=int, default=4)
    p.add_argument("--mrms-patch-threshold", type=float, default=5.0)
    p.add_argument("--abi-cold-threshold-k", type=float, default=235.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
