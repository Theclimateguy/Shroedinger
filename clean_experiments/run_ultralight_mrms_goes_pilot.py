#!/usr/bin/env python3
"""Run ultra-light MRMS+GOES pilot in three-step mode.

Stages:
- manifest_only: list/index only, build aligned catalog, estimate matched volume
- download_matched: same as manifest_only + download only matched files
- full: alias of download_matched
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    start_utc: str
    end_utc: str


def _sanitize_event_id(value: str) -> str:
    text = value.strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text or "event"


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _parse_events(path: Path, max_events: int) -> list[EventWindow]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"event_id", "start_utc", "end_utc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events CSV missing columns: {sorted(missing)}")
    if max_events > 0:
        df = df.head(max_events).copy()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            EventWindow(
                event_id=_sanitize_event_id(str(r["event_id"])),
                start_utc=str(r["start_utc"]),
                end_utc=str(r["end_utc"]),
            )
        )
    return rows


def _hours_between(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[int]:
    if end_utc < start_utc:
        raise ValueError("event end_utc must be >= start_utc")
    cur = start_utc.floor("h")
    end_h = end_utc.floor("h")
    out = set()
    while cur <= end_h:
        out.add(int(cur.hour))
        cur = cur + pd.Timedelta(hours=1)
    return sorted(out)


def _concat_csvs(paths: list[Path], out: Path) -> None:
    dfs = []
    for p in paths:
        if p.exists():
            dfs.append(pd.read_csv(p))
    if len(dfs) == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out, index=False)
        return
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(out, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path, default=Path("clean_experiments/pilot_events_template.csv"))
    parser.add_argument("--workdir", type=Path, default=Path("pilot_ultralight"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Optional root for raw files. Default: <workdir>/data_raw",
    )
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--stage", choices=["manifest_only", "download_matched", "full"], default="manifest_only")
    parser.add_argument("--mrms-product", default="MultiSensor_QPE_01H_Pass2_00.00")
    parser.add_argument("--goes-satellite", default="G19")
    parser.add_argument("--goes-product", default="ABI-L2-CMIPF")
    parser.add_argument("--goes-channel", default="C13")
    parser.add_argument("--tolerance-minutes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    events = _parse_events(args.events_csv, max_events=args.max_events)
    if len(events) == 0:
        raise ValueError("No events loaded")

    event_manifest_dir = args.workdir / "manifests" / "events"
    event_cov_dir = args.workdir / "manifests" / "coverage"
    quicklook_base = args.workdir / "quicklooks"
    metadata_dir = args.workdir / "metadata"
    raw_root = (args.raw_root if args.raw_root is not None else (args.workdir / "data_raw")).resolve()
    raw_mrms = raw_root / "mrms" / "raw"
    raw_goes = raw_root / "goes" / "raw"

    event_manifest_dir.mkdir(parents=True, exist_ok=True)
    event_cov_dir.mkdir(parents=True, exist_ok=True)
    quicklook_base.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    mrms_manifests: list[Path] = []
    goes_manifests: list[Path] = []

    for ev in events:
        s = pd.Timestamp(ev.start_utc)
        e = pd.Timestamp(ev.end_utc)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        else:
            s = s.tz_convert("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        else:
            e = e.tz_convert("UTC")
        if e < s:
            raise ValueError(f"event {ev.event_id}: end < start")

        hours = _hours_between(s, e)
        start_date = s.strftime("%Y-%m-%d")
        end_date = e.strftime("%Y-%m-%d")

        mrms_manifest = event_manifest_dir / f"mrms_{ev.event_id}.csv"
        mrms_cov = event_cov_dir / f"mrms_{ev.event_id}_coverage.csv"
        goes_manifest = event_manifest_dir / f"goes_{ev.event_id}.csv"
        goes_cov = event_cov_dir / f"goes_{ev.event_id}_coverage.csv"

        mrms_manifests.append(mrms_manifest)
        goes_manifests.append(goes_manifest)

        _run(
            [
                sys.executable,
                str(script_dir / "download_mrms.py"),
                "--start-date",
                start_date,
                "--end-date",
                end_date,
                "--start-datetime",
                s.isoformat().replace("+00:00", "Z"),
                "--end-datetime",
                e.isoformat().replace("+00:00", "Z"),
                "--event-id",
                ev.event_id,
                "--products",
                args.mrms_product,
                "--hours",
                *[str(h) for h in hours],
                "--manifest",
                str(mrms_manifest),
                "--raw-dir",
                str(raw_mrms),
                "--coverage",
                str(mrms_cov),
                "--quicklook-dir",
                str(quicklook_base / "mrms" / ev.event_id),
                "--download-log",
                str(metadata_dir / "download_log.json"),
                "--dry-run",
            ]
        )

        _run(
            [
                sys.executable,
                str(script_dir / "download_goes.py"),
                "--start-date",
                start_date,
                "--end-date",
                end_date,
                "--start-datetime",
                s.isoformat().replace("+00:00", "Z"),
                "--end-datetime",
                e.isoformat().replace("+00:00", "Z"),
                "--event-id",
                ev.event_id,
                "--satellites",
                args.goes_satellite,
                "--products",
                args.goes_product,
                "--abi-channels",
                args.goes_channel,
                "--hours",
                *[str(h) for h in hours],
                "--manifest",
                str(goes_manifest),
                "--raw-dir",
                str(raw_goes),
                "--coverage",
                str(goes_cov),
                "--quicklook-dir",
                str(quicklook_base / "goes" / ev.event_id),
                "--download-log",
                str(metadata_dir / "download_log.json"),
                "--dry-run",
            ]
        )

    mrms_manifest_all = args.workdir / "manifests" / "mrms_manifest.csv"
    goes_manifest_all = args.workdir / "manifests" / "goes_manifest.csv"
    _concat_csvs(mrms_manifests, mrms_manifest_all)
    _concat_csvs(goes_manifests, goes_manifest_all)

    aligned_catalog = args.workdir / "aligned_catalog.parquet"
    _run(
        [
            sys.executable,
            str(script_dir / "build_mrms_goes_aligned_catalog.py"),
            "--mrms-manifest",
            str(mrms_manifest_all),
            "--goes-manifest",
            str(goes_manifest_all),
            "--out",
            str(aligned_catalog),
            "--tolerance-minutes",
            str(args.tolerance_minutes),
            "--goes-products",
            args.goes_product,
            "--allowed-status",
            "skipped_dry_run",
        ]
    )

    download_mode = args.stage in {"download_matched", "full"}
    selective_report = args.workdir / "manifests" / "matched_download_report.csv"
    selective_summary = args.workdir / "manifests" / "matched_download_summary.csv"
    cmd = [
        sys.executable,
        str(script_dir / "download_matched_windows.py"),
        "--aligned-catalog",
        str(aligned_catalog.with_suffix(".csv") if not aligned_catalog.exists() else aligned_catalog),
        "--mrms-manifest",
        str(mrms_manifest_all),
        "--goes-manifest",
        str(goes_manifest_all),
        "--report",
        str(selective_report),
        "--summary",
        str(selective_summary),
    ]
    if download_mode:
        cmd.append("--download")
    _run(cmd)

    print("Ultra-light pilot completed.")
    print(f"Workdir: {args.workdir}")
    print(f"Stage: {args.stage}")


if __name__ == "__main__":
    main()
