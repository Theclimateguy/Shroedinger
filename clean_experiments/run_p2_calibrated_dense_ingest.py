#!/usr/bin/env python3
"""Run dense MRMS+GOES ingest calibrated by P2 theory-bridge ablation.

Pipeline:
1) Score events by calibrated operator activity (|lambda_local| and commutator defect).
2) Select top stable events and widen each event window by context hours.
3) Run stage-1 ABI (manifest-only for budget check, then matched download).
4) Run stage-2 GLM (manifest-only for budget check, then matched download).
5) Build unified panel and write dataset report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _find_aligned_catalog_csv(workdir: Path) -> Path:
    csv_path = workdir / "aligned_catalog.csv"
    if csv_path.exists():
        return csv_path
    parquet_path = workdir / "aligned_catalog.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
        return csv_path
    raise FileNotFoundError(f"No aligned catalog found in {workdir}")


def _score_events(tile_df: pd.DataFrame) -> pd.DataFrame:
    if {"lambda_local", "comm_defect_operator"}.issubset(tile_df.columns):
        lambda_col = "lambda_local"
        comm_col = "comm_defect_operator"
    elif {"lambda_local_raw", "comm_defect_raw"}.issubset(tile_df.columns):
        lambda_col = "lambda_local_raw"
        comm_col = "comm_defect_raw"
    else:
        raise ValueError(
            "Tile dataset must contain either "
            "['lambda_local', 'comm_defect_operator'] or ['lambda_local_raw', 'comm_defect_raw']."
        )

    rows: list[dict[str, float | str]] = []
    for event_id, grp in tile_df.groupby("event_id", sort=False):
        lam = grp[lambda_col].to_numpy(dtype=float)
        com = grp[comm_col].to_numpy(dtype=float)
        lam_abs_p95 = float(np.nanquantile(np.abs(lam), 0.95))
        com_abs_p95 = float(np.nanquantile(np.abs(com), 0.95))
        rows.append(
            {
                "event_id": str(event_id),
                "lambda_abs_p95": lam_abs_p95,
                "comm_abs_p95": com_abs_p95,
                "score": lam_abs_p95 * com_abs_p95,
            }
        )
    scores = pd.DataFrame(rows)
    scores = scores.sort_values(["score", "event_id"], ascending=[False, True]).reset_index(drop=True)
    scores["rank"] = np.arange(1, len(scores) + 1, dtype=int)
    return scores


def _to_utc_z(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _prepare_dense_events(
    *,
    base_events: pd.DataFrame,
    scores: pd.DataFrame,
    top_events: int,
    context_hours: int,
) -> pd.DataFrame:
    required_cols = {"event_id", "start_utc", "end_utc"}
    missing = sorted(required_cols - set(base_events.columns))
    if missing:
        raise ValueError(f"Base events CSV missing columns: {missing}")

    merged = base_events.merge(scores, on="event_id", how="left")
    merged["score"] = merged["score"].fillna(0.0).astype(float)
    merged["rank"] = merged["rank"].fillna(len(merged) + 999).astype(int)
    merged = merged.sort_values(["score", "event_id"], ascending=[False, True]).reset_index(drop=True)

    if top_events > 0:
        merged = merged.head(top_events).copy()

    dt0 = pd.to_datetime(merged["start_utc"], utc=True, errors="coerce")
    dt1 = pd.to_datetime(merged["end_utc"], utc=True, errors="coerce")
    if dt0.isna().any() or dt1.isna().any():
        raise ValueError("Failed to parse some event start/end timestamps.")

    widened_start = dt0 - pd.to_timedelta(context_hours, unit="h")
    widened_end = dt1 + pd.to_timedelta(context_hours, unit="h")

    out = merged.copy()
    out["orig_start_utc"] = merged["start_utc"].astype(str)
    out["orig_end_utc"] = merged["end_utc"].astype(str)
    out["start_utc"] = widened_start.map(_to_utc_z)
    out["end_utc"] = widened_end.map(_to_utc_z)
    out["notes"] = (
        "p2_dense_calibrated rank="
        + out["rank"].astype(str)
        + " score="
        + out["score"].map(lambda x: f"{float(x):.6f}")
        + f" context_h={int(context_hours)}"
    )
    return out


def _run_ultralight(
    *,
    script_path: Path,
    events_csv: Path,
    workdir: Path,
    raw_root: Path,
    stage: str,
    mrms_product: str,
    goes_satellite: str,
    goes_product: str,
    goes_channel: str,
    tolerance_minutes: int,
) -> None:
    cmd = [
        sys.executable,
        str(script_path),
        "--events-csv",
        str(events_csv),
        "--workdir",
        str(workdir),
        "--raw-root",
        str(raw_root),
        "--max-events",
        "0",
        "--stage",
        stage,
        "--mrms-product",
        mrms_product,
        "--goes-satellite",
        goes_satellite,
        "--goes-product",
        goes_product,
        "--goes-channel",
        goes_channel,
        "--tolerance-minutes",
        str(tolerance_minutes),
    ]
    _run(cmd)


def _run_selective_download(
    *,
    script_path: Path,
    stage_workdir: Path,
    download: bool,
) -> None:
    aligned_csv = _find_aligned_catalog_csv(stage_workdir)
    mrms_manifest = stage_workdir / "manifests" / "mrms_manifest.csv"
    goes_manifest = stage_workdir / "manifests" / "goes_manifest.csv"
    report = stage_workdir / "manifests" / "matched_download_report.csv"
    summary = stage_workdir / "manifests" / "matched_download_summary.csv"
    _require_file(mrms_manifest)
    _require_file(goes_manifest)

    cmd = [
        sys.executable,
        str(script_path),
        "--aligned-catalog",
        str(aligned_csv),
        "--mrms-manifest",
        str(mrms_manifest),
        "--goes-manifest",
        str(goes_manifest),
        "--report",
        str(report),
        "--summary",
        str(summary),
    ]
    if download:
        cmd.append("--download")
    _run(cmd)


def _load_summary_row(path: Path) -> dict[str, float | int | bool]:
    _require_file(path)
    df = pd.read_csv(path)
    if df.empty:
        return {
            "n_rows": 0,
            "n_unique_files": 0,
            "bytes_total": 0,
            "gb_total": 0.0,
            "n_errors": 0,
            "n_missing_manifest": 0,
            "n_downloaded": 0,
            "n_exists": 0,
            "download_mode": False,
        }
    row = df.iloc[0].to_dict()
    return {
        "n_rows": int(float(row.get("n_rows", 0) or 0)),
        "n_unique_files": int(float(row.get("n_unique_files", 0) or 0)),
        "bytes_total": int(float(row.get("bytes_total", 0) or 0)),
        "gb_total": float(row.get("gb_total", 0.0) or 0.0),
        "n_errors": int(float(row.get("n_errors", 0) or 0)),
        "n_missing_manifest": int(float(row.get("n_missing_manifest", 0) or 0)),
        "n_downloaded": int(float(row.get("n_downloaded", 0) or 0)),
        "n_exists": int(float(row.get("n_exists", 0) or 0)),
        "download_mode": bool(row.get("download_mode", False)),
    }


def _summary_download_mode(path: Path) -> bool:
    if not path.exists():
        return False
    row = _load_summary_row(path)
    return bool(row.get("download_mode", False))


def _load_report(path: Path) -> pd.DataFrame:
    _require_file(path)
    return pd.read_csv(path)


def _union_unique_bytes(report_dfs: list[pd.DataFrame], statuses: set[str]) -> tuple[int, int]:
    if len(report_dfs) == 0:
        return 0, 0
    all_rows = pd.concat(report_dfs, ignore_index=True)
    if "status" in all_rows.columns:
        all_rows = all_rows[all_rows["status"].astype(str).isin(statuses)].copy()
    if all_rows.empty:
        return 0, 0
    for col in ["local_path", "file_size_bytes"]:
        if col not in all_rows.columns:
            raise ValueError(f"Report missing required column: {col}")
    uniq = all_rows.drop_duplicates(subset=["local_path"]).copy()
    bytes_total = int(uniq["file_size_bytes"].fillna(0).astype(float).sum())
    return bytes_total, int(len(uniq))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--selected-tile-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_theory_bridge_ablation/selected_config_tile_dataset.csv"),
    )
    p.add_argument(
        "--selected-config-csv",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_theory_bridge_ablation/selected_config.csv"),
    )
    p.add_argument(
        "--base-events-csv",
        type=Path,
        default=Path("clean_experiments/pilot_events_real_2024_us_convective_expanded_v1.csv"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_p2dense_calibrated"),
    )

    p.add_argument("--top-events", type=int, default=16)
    p.add_argument("--context-hours", type=int, default=6)
    p.add_argument("--budget-gb", type=float, default=50.0)
    p.add_argument("--tolerance-minutes", type=int, default=10)
    p.add_argument("--skip-download", action="store_true")

    p.add_argument("--mrms-product", default="MultiSensor_QPE_01H_Pass2_00.00")
    p.add_argument("--goes-satellite", default="G16")
    p.add_argument("--abi-product", default="ABI-L2-CMIPF")
    p.add_argument("--abi-channel", default="C13")
    p.add_argument("--glm-product", default="GLM-L2-LCFA")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    ultralight_script = script_dir / "run_ultralight_mrms_goes_pilot.py"
    selective_script = script_dir / "download_matched_windows.py"
    panel_builder = script_dir / "build_realpilot_unified_panel.py"
    _require_file(ultralight_script)
    _require_file(selective_script)
    _require_file(panel_builder)
    _require_file(args.selected_tile_csv)
    _require_file(args.base_events_csv)

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root = out_root / "data_raw_shared"
    stage1_workdir = out_root / "stage1_abi_download"
    stage2_workdir = out_root / "stage2_glm_download"
    s1_summary_path = stage1_workdir / "manifests" / "matched_download_summary.csv"
    s2_summary_path = stage2_workdir / "manifests" / "matched_download_summary.csv"

    tile_df = pd.read_csv(args.selected_tile_csv)
    base_events = pd.read_csv(args.base_events_csv)

    scores = _score_events(tile_df)
    scores_path = out_root / "stable_event_scores.csv"
    scores.to_csv(scores_path, index=False)

    selected_events = _prepare_dense_events(
        base_events=base_events,
        scores=scores,
        top_events=max(0, int(args.top_events)),
        context_hours=max(0, int(args.context_hours)),
    )
    selected_events_path = out_root / "stable_events_dense.csv"
    selected_events.to_csv(selected_events_path, index=False)
    print(f"Selected dense events: {len(selected_events)} (saved: {selected_events_path})", flush=True)

    if s1_summary_path.exists():
        print("[stage] reuse existing ABI manifest summary", flush=True)
    else:
        print("[stage] manifest-only ABI", flush=True)
        _run_ultralight(
            script_path=ultralight_script,
            events_csv=selected_events_path,
            workdir=stage1_workdir,
            raw_root=raw_root,
            stage="manifest_only",
            mrms_product=args.mrms_product,
            goes_satellite=args.goes_satellite,
            goes_product=args.abi_product,
            goes_channel=args.abi_channel,
            tolerance_minutes=args.tolerance_minutes,
        )

    if s2_summary_path.exists():
        print("[stage] reuse existing GLM manifest summary", flush=True)
    else:
        print("[stage] manifest-only GLM", flush=True)
        _run_ultralight(
            script_path=ultralight_script,
            events_csv=selected_events_path,
            workdir=stage2_workdir,
            raw_root=raw_root,
            stage="manifest_only",
            mrms_product=args.mrms_product,
            goes_satellite=args.goes_satellite,
            goes_product=args.glm_product,
            goes_channel=args.abi_channel,
            tolerance_minutes=args.tolerance_minutes,
        )

    s1_manifest_summary = _load_summary_row(s1_summary_path)
    s2_manifest_summary = _load_summary_row(s2_summary_path)

    s1_manifest_report = _load_report(stage1_workdir / "manifests" / "matched_download_report.csv")
    s2_manifest_report = _load_report(stage2_workdir / "manifests" / "matched_download_report.csv")
    manifest_unique_bytes, manifest_unique_files = _union_unique_bytes(
        [s1_manifest_report, s2_manifest_report],
        statuses={"planned", "downloaded", "exists"},
    )
    manifest_unique_gb = manifest_unique_bytes / (1024.0**3)

    print(
        "[budget] manifest unique planned files="
        f"{manifest_unique_files}, unique_gb={manifest_unique_gb:.3f}, budget_gb={args.budget_gb:.3f}",
        flush=True,
    )
    if manifest_unique_gb > float(args.budget_gb):
        raise RuntimeError(
            f"Planned unique volume {manifest_unique_gb:.3f} GB exceeds budget {float(args.budget_gb):.3f} GB."
        )

    if not args.skip_download:
        if _summary_download_mode(s1_summary_path):
            print("[stage] reuse existing ABI download summary", flush=True)
        else:
            print("[stage] selective download ABI", flush=True)
            _run_selective_download(
                script_path=selective_script,
                stage_workdir=stage1_workdir,
                download=True,
            )

        if _summary_download_mode(s2_summary_path):
            print("[stage] reuse existing GLM download summary", flush=True)
        else:
            print("[stage] selective download GLM", flush=True)
            _run_selective_download(
                script_path=selective_script,
                stage_workdir=stage2_workdir,
                download=True,
            )

    s1_download_summary = _load_summary_row(stage1_workdir / "manifests" / "matched_download_summary.csv")
    s2_download_summary = _load_summary_row(stage2_workdir / "manifests" / "matched_download_summary.csv")
    s1_download_report = _load_report(stage1_workdir / "manifests" / "matched_download_report.csv")
    s2_download_report = _load_report(stage2_workdir / "manifests" / "matched_download_report.csv")
    present_unique_bytes, present_unique_files = _union_unique_bytes(
        [s1_download_report, s2_download_report],
        statuses={"downloaded", "exists"},
    )
    present_unique_gb = present_unique_bytes / (1024.0**3)

    print("[stage] build unified panel", flush=True)
    abi_aligned_csv = _find_aligned_catalog_csv(stage1_workdir)
    glm_aligned_csv = _find_aligned_catalog_csv(stage2_workdir)
    out_panel_csv = out_root / "realpilot_2024_dataset_panel_p2dense_calibrated.csv"
    _run(
        [
            sys.executable,
            str(panel_builder),
            "--abi-aligned-csv",
            str(abi_aligned_csv),
            "--glm-aligned-csv",
            str(glm_aligned_csv),
            "--out-csv",
            str(out_panel_csv),
        ]
    )

    panel = pd.read_csv(out_panel_csv)
    stats = (
        panel.groupby("event_id", as_index=False)
        .agg(
            n_rows=("event_id", "size"),
            start_utc=("mrms_obs_time_utc", "min"),
            end_utc=("mrms_obs_time_utc", "max"),
            n_mrms=("mrms_key", "nunique"),
            n_abi=("abi_key", "nunique"),
            n_glm=("glm_key", "nunique"),
        )
        .sort_values("event_id")
        .reset_index(drop=True)
    )
    stats = stats.merge(scores[["event_id", "score", "rank"]], on="event_id", how="left")
    stats_path = out_root / "realpilot_2024_dataset_panel_p2dense_calibrated_event_stats.csv"
    stats.to_csv(stats_path, index=False)

    cfg_text = ""
    if args.selected_config_csv.exists():
        cfg_df = pd.read_csv(args.selected_config_csv)
        if not cfg_df.empty:
            c = cfg_df.iloc[0]
            cfg_text = "\n".join(
                [
                    "## P2 selected config",
                    f"- config_id: `{c.get('config_id', '')}`",
                    f"- decoherence_alpha: `{float(c.get('decoherence_alpha', np.nan))}`",
                    f"- lambda_scale_power: `{float(c.get('lambda_scale_power', np.nan))}`",
                    (
                        "- weights [occ,sq,log,grad]: "
                        f"`[{float(c.get('w_occ', np.nan))}, {float(c.get('w_sq', np.nan))}, "
                        f"{float(c.get('w_log', np.nan))}, {float(c.get('w_grad', np.nan))}]`"
                    ),
                    "",
                ]
            )

    report_lines = [
        "# Real Pilot Dense Dataset (P2-calibrated ingest)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "## Selection policy",
        f"- base events: `{args.base_events_csv}`",
        f"- scored by: `score = P95(|lambda_local|) * P95(|comm_defect_operator|)`",
        f"- selected top events: `{len(selected_events)}`",
        f"- context expansion: `+/- {int(args.context_hours)}h`",
        f"- selected events file: `{selected_events_path}`",
        f"- score table: `{scores_path}`",
        "",
    ]
    if cfg_text:
        report_lines.append(cfg_text)
    report_lines.extend(
        [
            "## Budget check (manifest-only)",
            f"- budget: `{float(args.budget_gb):.3f} GB`",
            f"- planned unique files (ABI+GLM union): `{manifest_unique_files}`",
            f"- planned unique size (ABI+GLM union): `{manifest_unique_gb:.3f} GB`",
            f"- ABI summary gb_total (non-unique): `{float(s1_manifest_summary['gb_total']):.3f} GB`",
            f"- GLM summary gb_total (non-unique): `{float(s2_manifest_summary['gb_total']):.3f} GB`",
            "",
            "## Download result",
            f"- skip_download: `{bool(args.skip_download)}`",
            f"- present unique files (downloaded|exists, ABI+GLM union): `{present_unique_files}`",
            f"- present unique size (downloaded|exists, ABI+GLM union): `{present_unique_gb:.3f} GB`",
            f"- ABI downloaded: `{int(s1_download_summary['n_downloaded'])}` exists: `{int(s1_download_summary['n_exists'])}`",
            f"- GLM downloaded: `{int(s2_download_summary['n_downloaded'])}` exists: `{int(s2_download_summary['n_exists'])}`",
            "",
            "## Artifacts",
            f"- stage1 ABI workdir: `{stage1_workdir}`",
            f"- stage2 GLM workdir: `{stage2_workdir}`",
            f"- unified panel: `{out_panel_csv}`",
            f"- event stats: `{stats_path}`",
            "",
            "## Panel snapshot",
            f"- rows: `{len(panel)}`",
            f"- events: `{panel['event_id'].nunique()}`",
        ]
    )
    report_path = out_root / "DATASET_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Dense ingest complete: {out_root}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
