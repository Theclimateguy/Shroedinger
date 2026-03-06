#!/usr/bin/env python3
"""Download MRMS pilot data from NOAA public S3 over HTTPS.

Design goals (stage-1 pilot):
- idempotent downloads (re-runs only fill gaps)
- no early temporal aggregation
- manifest + coverage + quicklook outputs for downstream event segmentation
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _open_s3_http import download_public_object, iter_public_s3_objects, safe_json_dump


DEFAULT_BUCKET = "noaa-mrms-pds"
DEFAULT_REGION_PREFIX = "CONUS"
DEFAULT_PRODUCTS = (
    "MultiSensor_QPE_01H_Pass2_00.00",
    "MergedReflectivityQCComposite_00.50",
    "Reflectivity_-10C_00.50",
    "MESH_00.50",
    "EchoTop_50_00.50",
)


_MRMS_TS_RE = re.compile(r"_(\d{8})-(\d{6})")


@dataclass(frozen=True)
class DownloadRow:
    source: str
    event_id: str
    bucket: str
    region_prefix: str
    product: str
    key: str
    url: str
    local_path: str
    file_size_bytes: int
    last_modified_utc: str
    obs_time_utc: str
    run_time_utc: str
    status: str
    error: str


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def _parse_dt_utc(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _date_range(d0: date, d1: date) -> Iterator[date]:
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def _parse_obs_time_from_key(key: str) -> datetime | None:
    m = _MRMS_TS_RE.search(key)
    if m is None:
        return None
    ts = m.group(1) + m.group(2)
    return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _daily_quicklooks(manifest: pd.DataFrame, outdir: Path) -> None:
    if manifest.empty:
        return
    work = manifest.copy()
    work = work[work["obs_time_utc"].astype(str).str.len() > 0]
    if work.empty:
        return
    work["obs_dt"] = pd.to_datetime(work["obs_time_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["obs_dt"])
    if work.empty:
        return

    work["day"] = work["obs_dt"].dt.strftime("%Y-%m-%d")
    work["hour"] = work["obs_dt"].dt.hour

    outdir.mkdir(parents=True, exist_ok=True)
    for day, grp in work.groupby("day", sort=True):
        products = sorted(grp["product"].unique())
        hindex = list(range(24))
        matrix = np.zeros((len(products), 24), dtype=float)
        for i, p in enumerate(products):
            s = grp.loc[grp["product"] == p].groupby("hour").size()
            for h, count in s.items():
                matrix[i, int(h)] = float(count)

        fig_h = max(3.0, 0.45 * len(products))
        fig, ax = plt.subplots(figsize=(12, fig_h))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(f"MRMS file count quicklook by hour ({day})")
        ax.set_xlabel("UTC hour")
        ax.set_ylabel("Product")
        ax.set_xticks(range(0, 24, 2))
        ax.set_yticks(range(len(products)))
        ax.set_yticklabels(products)
        fig.colorbar(im, ax=ax, label="file count")
        fig.tight_layout()
        fig.savefig(outdir / f"mrms_quicklook_{day}.png", dpi=140)
        plt.close(fig)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--start-datetime", default="", help="Optional UTC lower bound, e.g. 2026-01-01T00:00:00Z")
    parser.add_argument("--end-datetime", default="", help="Optional UTC upper bound, e.g. 2026-01-01T02:59:59Z")
    parser.add_argument("--event-id", default="", help="Optional event tag propagated into manifest rows.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region-prefix", default=DEFAULT_REGION_PREFIX)
    parser.add_argument("--products", nargs="+", default=list(DEFAULT_PRODUCTS))
    parser.add_argument(
        "--hours",
        nargs="+",
        type=int,
        default=[],
        help="Optional UTC hours filter (0-23). If empty, all hours are included.",
    )
    parser.add_argument(
        "--aoi",
        default="",
        help="Optional AOI bbox as west,south,east,north. Stored in logs for downstream crop stage.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data_raw/mrms/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/mrms_manifest.csv"))
    parser.add_argument("--coverage", type=Path, default=Path("manifests/mrms_coverage.csv"))
    parser.add_argument("--quicklook-dir", type=Path, default=Path("quicklooks/mrms"))
    parser.add_argument("--download-log", type=Path, default=Path("metadata/download_log.json"))
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--list-timeout-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="List/index only, skip file downloads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d0 = _parse_date(args.start_date)
    d1 = _parse_date(args.end_date)
    if d0 > d1:
        raise ValueError("--start-date must be <= --end-date")

    run_utc = datetime.now(timezone.utc)
    run_stamp = run_utc.isoformat()
    allowed_hours = {int(h) for h in args.hours}
    if any((h < 0 or h > 23) for h in allowed_hours):
        raise ValueError("--hours must be within 0..23")
    dt0 = _parse_dt_utc(args.start_datetime) if len(args.start_datetime.strip()) > 0 else None
    dt1 = _parse_dt_utc(args.end_datetime) if len(args.end_datetime.strip()) > 0 else None
    if dt0 is not None and dt1 is not None and dt0 > dt1:
        raise ValueError("--start-datetime must be <= --end-datetime")
    rows: list[DownloadRow] = []

    listed = 0
    downloaded = 0
    exists = 0
    errors = 0

    for product in args.products:
        for day in _date_range(d0, d1):
            day_s = day.strftime("%Y%m%d")
            prefix = f"{args.region_prefix}/{product}/{day_s}/"
            print(f"[MRMS] list prefix: s3://{args.bucket}/{prefix}")
            for obj in iter_public_s3_objects(
                bucket=args.bucket,
                prefix=prefix,
                timeout_seconds=args.list_timeout_seconds,
            ):
                obs_dt = _parse_obs_time_from_key(obj.key)
                if allowed_hours and (obs_dt is not None) and (obs_dt.hour not in allowed_hours):
                    continue
                if dt0 is not None and (obs_dt is not None) and (obs_dt < dt0):
                    continue
                if dt1 is not None and (obs_dt is not None) and (obs_dt > dt1):
                    continue
                listed += 1
                local_path = args.raw_dir / args.region_prefix / product / day_s / Path(obj.key).name
                status = "skipped_dry_run"
                error = ""

                try:
                    if args.dry_run:
                        status = "exists" if local_path.exists() else "skipped_dry_run"
                    else:
                        status = download_public_object(
                            url=obj.url,
                            target_path=local_path,
                            expected_size=obj.size,
                            timeout_seconds=args.timeout_seconds,
                        )
                except Exception as exc:  # noqa: BLE001
                    status = "error"
                    error = str(exc)
                    errors += 1

                if status == "downloaded":
                    downloaded += 1
                elif status == "exists":
                    exists += 1

                rows.append(
                    DownloadRow(
                        source="mrms",
                        event_id=args.event_id,
                        bucket=obj.bucket,
                        region_prefix=args.region_prefix,
                        product=product,
                        key=obj.key,
                        url=obj.url,
                        local_path=str(local_path),
                        file_size_bytes=obj.size,
                        last_modified_utc=(obj.last_modified.isoformat() if obj.last_modified else ""),
                        obs_time_utc=(obs_dt.isoformat() if obs_dt else ""),
                        run_time_utc=run_stamp,
                        status=status,
                        error=error,
                    )
                )

    manifest_cols = list(DownloadRow.__dataclass_fields__.keys())
    manifest = pd.DataFrame([asdict(r) for r in rows], columns=manifest_cols)
    _ensure_parent(args.manifest)
    manifest.to_csv(args.manifest, index=False)

    if manifest.empty:
        coverage = pd.DataFrame(
            columns=[
                "source",
                "event_id",
                "region_prefix",
                "product",
                "day",
                "n_files",
                "n_downloaded",
                "n_exists",
                "n_error",
            ]
        )
    else:
        manifest["day"] = manifest["obs_time_utc"].str.slice(0, 10)
        coverage = (
            manifest.groupby(["source", "event_id", "region_prefix", "product", "day"], dropna=False)
            .agg(
                n_files=("key", "count"),
                n_downloaded=("status", lambda x: int((x == "downloaded").sum())),
                n_exists=("status", lambda x: int((x == "exists").sum())),
                n_error=("status", lambda x: int((x == "error").sum())),
            )
            .reset_index()
        )
    _ensure_parent(args.coverage)
    coverage.to_csv(args.coverage, index=False)

    _daily_quicklooks(manifest, args.quicklook_dir)

    log_payload = {}
    if args.download_log.exists():
        try:
            import json

            with args.download_log.open("r", encoding="utf-8") as f:
                log_payload = json.load(f)
        except Exception:  # noqa: BLE001
            log_payload = {}
    if not isinstance(log_payload, dict):
        log_payload = {}
    runs = log_payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    runs.append(
        {
            "script": "download_mrms.py",
            "run_time_utc": run_stamp,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "start_datetime": args.start_datetime,
            "end_datetime": args.end_datetime,
            "event_id": args.event_id,
            "bucket": args.bucket,
            "region_prefix": args.region_prefix,
            "products": list(args.products),
            "hours": list(args.hours),
            "aoi": args.aoi,
            "listed": listed,
            "downloaded": downloaded,
            "exists": exists,
            "errors": errors,
            "manifest": str(args.manifest),
            "coverage": str(args.coverage),
            "quicklook_dir": str(args.quicklook_dir),
        }
    )
    log_payload["runs"] = runs
    safe_json_dump(args.download_log, log_payload)

    print(f"MRMS manifest written: {args.manifest}")
    print(f"MRMS coverage written: {args.coverage}")
    print(f"MRMS quicklooks dir: {args.quicklook_dir}")
    print(
        "MRMS summary: "
        f"listed={listed}, downloaded={downloaded}, exists={exists}, errors={errors}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
