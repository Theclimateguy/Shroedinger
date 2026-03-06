#!/usr/bin/env python3
"""Download GOES pilot data for MRMS-aligned windows from NOAA public S3.

By default this script uses direct HTTPS listing of S3 open-data buckets.
Optional mode `goes2go` is supported when that package is available.
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


DEFAULT_PRODUCTS = ("ABI-L2-CMIPF", "GLM-L2-LCFA")
DEFAULT_SATELLITES = ("G19", "G18")
DEFAULT_ABI_CHANNELS = ("C02", "C08", "C13")


_START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")
_CHANNEL_RE = re.compile(r"-M\d(C\d{2})_")


@dataclass(frozen=True)
class DownloadRow:
    source: str
    event_id: str
    satellite: str
    bucket: str
    product: str
    abi_channel: str
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


def _satellite_to_bucket(satellite: str) -> str:
    s = satellite.strip().upper()
    if s.startswith("G"):
        num = s[1:]
    else:
        num = s
    if not num.isdigit():
        raise ValueError(f"Invalid satellite tag: {satellite}")
    return f"noaa-goes{int(num)}"


def _parse_obs_time_from_key(key: str) -> datetime | None:
    m = _START_RE.search(key)
    if m is None:
        return None
    year = int(m.group(1))
    doy = int(m.group(2))
    hour = int(m.group(3))
    minute = int(m.group(4))
    second = int(m.group(5))
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy - 1, hours=hour, minutes=minute, seconds=second
    )
    return dt


def _parse_channel_from_key(key: str) -> str:
    m = _CHANNEL_RE.search(key)
    return m.group(1) if m is not None else ""


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
    work["label"] = np.where(
        work["abi_channel"].astype(str).str.len() > 0,
        work["satellite"] + " " + work["product"] + " " + work["abi_channel"],
        work["satellite"] + " " + work["product"],
    )

    outdir.mkdir(parents=True, exist_ok=True)
    for day, grp in work.groupby("day", sort=True):
        labels = sorted(grp["label"].unique())
        matrix = np.zeros((len(labels), 24), dtype=float)
        for i, label in enumerate(labels):
            s = grp.loc[grp["label"] == label].groupby("hour").size()
            for h, count in s.items():
                matrix[i, int(h)] = float(count)

        fig_h = max(3.0, 0.42 * len(labels))
        fig, ax = plt.subplots(figsize=(12, fig_h))
        im = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_title(f"GOES file count quicklook by hour ({day})")
        ax.set_xlabel("UTC hour")
        ax.set_ylabel("Satellite / product / channel")
        ax.set_xticks(range(0, 24, 2))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, label="file count")
        fig.tight_layout()
        fig.savefig(outdir / f"goes_quicklook_{day}.png", dpi=140)
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
    parser.add_argument("--mode", choices=["http", "goes2go"], default="http")
    parser.add_argument("--satellites", nargs="+", default=list(DEFAULT_SATELLITES), help="e.g. G19 G18")
    parser.add_argument("--products", nargs="+", default=list(DEFAULT_PRODUCTS))
    parser.add_argument(
        "--hours",
        nargs="+",
        type=int,
        default=[],
        help="Optional UTC hours filter (0-23). If empty, all hours are included.",
    )
    parser.add_argument("--abi-channels", nargs="+", default=list(DEFAULT_ABI_CHANNELS), help="Used for ABI products")
    parser.add_argument("--all-abi-channels", action="store_true", help="Disable ABI channel filtering")
    parser.add_argument("--raw-dir", type=Path, default=Path("data_raw/goes/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/goes_manifest.csv"))
    parser.add_argument("--coverage", type=Path, default=Path("manifests/goes_coverage.csv"))
    parser.add_argument("--quicklook-dir", type=Path, default=Path("quicklooks/goes"))
    parser.add_argument("--download-log", type=Path, default=Path("metadata/download_log.json"))
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--list-timeout-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="List/index only, skip file downloads")
    return parser.parse_args()


def _download_http(args: argparse.Namespace, run_stamp: str) -> list[DownloadRow]:
    rows: list[DownloadRow] = []
    allow_channels = {c.upper() for c in args.abi_channels}
    allowed_hours = {int(h) for h in args.hours}
    dt0 = _parse_dt_utc(args.start_datetime) if len(args.start_datetime.strip()) > 0 else None
    dt1 = _parse_dt_utc(args.end_datetime) if len(args.end_datetime.strip()) > 0 else None

    for sat in args.satellites:
        sat_norm = sat.upper()
        bucket = _satellite_to_bucket(sat_norm)
        for product in args.products:
            is_abi = product.startswith("ABI-")
            for day in _date_range(_parse_date(args.start_date), _parse_date(args.end_date)):
                year = day.strftime("%Y")
                doy = day.strftime("%j")
                hour_iter = sorted(allowed_hours) if allowed_hours else list(range(24))
                for hour in hour_iter:
                    prefix = f"{product}/{year}/{doy}/{hour:02d}/"
                    print(f"[GOES] list prefix: s3://{bucket}/{prefix}")
                    for obj in iter_public_s3_objects(
                        bucket=bucket,
                        prefix=prefix,
                        timeout_seconds=args.list_timeout_seconds,
                    ):
                        ch = _parse_channel_from_key(obj.key)
                        if is_abi and (not args.all_abi_channels):
                            if len(ch) == 0 or ch.upper() not in allow_channels:
                                continue
                        obs_dt = _parse_obs_time_from_key(obj.key)
                        if dt0 is not None and (obs_dt is not None) and (obs_dt < dt0):
                            continue
                        if dt1 is not None and (obs_dt is not None) and (obs_dt > dt1):
                            continue

                        local_path = (
                            args.raw_dir
                            / sat_norm
                            / product
                            / year
                            / doy
                            / f"{hour:02d}"
                            / Path(obj.key).name
                        )

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

                        rows.append(
                            DownloadRow(
                                source="goes",
                                event_id=args.event_id,
                                satellite=sat_norm,
                                bucket=bucket,
                                product=product,
                                abi_channel=ch,
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
    return rows


def _download_goes2go(args: argparse.Namespace, run_stamp: str) -> list[DownloadRow]:
    try:
        from goes2go import GOES  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("mode=goes2go requires package goes2go. Install with `pip install goes2go`.") from exc

    rows: list[DownloadRow] = []
    start_dt = pd.Timestamp(args.start_date).tz_localize("UTC")
    end_dt = (pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize("UTC")

    allow_channels = {c.upper() for c in args.abi_channels}
    allowed_hours = {int(h) for h in args.hours}
    dt0 = _parse_dt_utc(args.start_datetime) if len(args.start_datetime.strip()) > 0 else None
    dt1 = _parse_dt_utc(args.end_datetime) if len(args.end_datetime.strip()) > 0 else None

    for sat in args.satellites:
        sat_norm = sat.upper()
        sat_num = sat_norm[1:] if sat_norm.startswith("G") else sat_norm
        for product in args.products:
            is_abi = product.startswith("ABI-")
            channel_iter = [""]
            if is_abi and (not args.all_abi_channels):
                channel_iter = sorted(allow_channels)

            for channel in channel_iter:
                kwargs = {
                    "satellite": f"goes{sat_num}",
                    "product": product,
                    "start": start_dt,
                    "end": end_dt,
                    "save_dir": str(args.raw_dir / sat_norm / product),
                    "download": not args.dry_run,
                }
                if is_abi and len(channel) > 0:
                    kwargs["bands"] = [int(channel[1:])]

                print(f"[GOES goes2go] {kwargs['satellite']} {product} channel={channel or 'ALL'}")
                g = GOES(**kwargs)
                files = getattr(g, "files", None)
                if files is None:
                    # Best effort fallback: no explicit file list from this goes2go version.
                    continue

                for fp in files:
                    p = Path(str(fp))
                    obs_dt = _parse_obs_time_from_key(p.name)
                    if allowed_hours and (obs_dt is not None) and (obs_dt.hour not in allowed_hours):
                        continue
                    if dt0 is not None and (obs_dt is not None) and (obs_dt < dt0):
                        continue
                    if dt1 is not None and (obs_dt is not None) and (obs_dt > dt1):
                        continue
                    rows.append(
                        DownloadRow(
                            source="goes",
                            event_id=args.event_id,
                            satellite=sat_norm,
                            bucket=_satellite_to_bucket(sat_norm),
                            product=product,
                            abi_channel=channel,
                            key="",
                            url="",
                            local_path=str(p),
                            file_size_bytes=(p.stat().st_size if p.exists() else 0),
                            last_modified_utc="",
                            obs_time_utc=(obs_dt.isoformat() if obs_dt else ""),
                            run_time_utc=run_stamp,
                            status=("exists" if p.exists() else "missing"),
                            error="",
                        )
                    )

    return rows


def main() -> None:
    args = parse_args()
    d0 = _parse_date(args.start_date)
    d1 = _parse_date(args.end_date)
    if d0 > d1:
        raise ValueError("--start-date must be <= --end-date")
    if any((int(h) < 0 or int(h) > 23) for h in args.hours):
        raise ValueError("--hours must be within 0..23")
    dt0 = _parse_dt_utc(args.start_datetime) if len(args.start_datetime.strip()) > 0 else None
    dt1 = _parse_dt_utc(args.end_datetime) if len(args.end_datetime.strip()) > 0 else None
    if dt0 is not None and dt1 is not None and dt0 > dt1:
        raise ValueError("--start-datetime must be <= --end-datetime")

    run_utc = datetime.now(timezone.utc)
    run_stamp = run_utc.isoformat()

    if args.mode == "http":
        rows = _download_http(args, run_stamp)
    else:
        rows = _download_goes2go(args, run_stamp)

    manifest_cols = list(DownloadRow.__dataclass_fields__.keys())
    manifest = pd.DataFrame([asdict(r) for r in rows], columns=manifest_cols)
    _ensure_parent(args.manifest)
    manifest.to_csv(args.manifest, index=False)

    if manifest.empty:
        coverage = pd.DataFrame(
            columns=[
                "source",
                "event_id",
                "satellite",
                "product",
                "abi_channel",
                "day",
                "n_files",
                "n_downloaded",
                "n_exists",
                "n_error",
            ]
        )
        downloaded = exists = errors = 0
    else:
        manifest["day"] = manifest["obs_time_utc"].str.slice(0, 10)
        coverage = (
            manifest.groupby(["source", "event_id", "satellite", "product", "abi_channel", "day"], dropna=False)
            .agg(
                n_files=("local_path", "count"),
                n_downloaded=("status", lambda x: int((x == "downloaded").sum())),
                n_exists=("status", lambda x: int((x == "exists").sum())),
                n_error=("status", lambda x: int((x == "error").sum())),
            )
            .reset_index()
        )
        downloaded = int((manifest["status"] == "downloaded").sum())
        exists = int((manifest["status"] == "exists").sum())
        errors = int((manifest["status"] == "error").sum())

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
            "script": "download_goes.py",
            "run_time_utc": run_stamp,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "start_datetime": args.start_datetime,
            "end_datetime": args.end_datetime,
            "event_id": args.event_id,
            "mode": args.mode,
            "satellites": list(args.satellites),
            "products": list(args.products),
            "hours": list(args.hours),
            "abi_channels": list(args.abi_channels),
            "all_abi_channels": bool(args.all_abi_channels),
            "dry_run": bool(args.dry_run),
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

    print(f"GOES manifest written: {args.manifest}")
    print(f"GOES coverage written: {args.coverage}")
    print(f"GOES quicklooks dir: {args.quicklook_dir}")
    print(
        "GOES summary: "
        f"rows={len(manifest)}, downloaded={downloaded}, exists={exists}, errors={errors}, "
        f"mode={args.mode}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
