#!/usr/bin/env python3
"""Parallel selective downloader from a matched-download report CSV."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from _open_s3_http import download_public_object


@dataclass(frozen=True)
class DownloadResult:
    source: str
    event_id: str
    key: str
    url: str
    local_path: str
    file_size_bytes: int
    status: str
    error: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-in", type=Path, required=True, help="Input plan/report CSV from download_matched_windows.py")
    p.add_argument("--report-out", type=Path, required=True, help="Output report CSV with download statuses")
    p.add_argument("--summary-out", type=Path, required=True, help="Output summary CSV")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--timeout-seconds", type=int, default=20)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _load_plan(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    needed = {"source", "event_id", "key", "url", "local_path", "file_size_bytes"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Input report missing columns: {missing}")

    df = df.copy()
    df["url"] = df["url"].astype(str).fillna("")
    df["local_path"] = df["local_path"].astype(str).fillna("")
    df["file_size_bytes"] = pd.to_numeric(df["file_size_bytes"], errors="coerce").fillna(0).astype(int)
    df = df[(df["url"].str.len() > 0) & (df["local_path"].str.len() > 0)].copy()
    df = df.drop_duplicates(subset=["local_path"]).reset_index(drop=True)
    return df


def _download_one(
    row: pd.Series,
    *,
    timeout_seconds: int,
    retries: int,
) -> DownloadResult:
    source = str(row.get("source", ""))
    event_id = str(row.get("event_id", ""))
    key = str(row.get("key", ""))
    url = str(row.get("url", ""))
    local_path = str(row.get("local_path", ""))
    size = int(row.get("file_size_bytes", 0) or 0)

    try:
        status = download_public_object(
            url=url,
            target_path=Path(local_path),
            expected_size=(size if size > 0 else None),
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        error = ""
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = str(exc)

    return DownloadResult(
        source=source,
        event_id=event_id,
        key=key,
        url=url,
        local_path=local_path,
        file_size_bytes=size,
        status=status,
        error=error,
    )


def main() -> None:
    args = parse_args()
    plan = _load_plan(args.report_in)
    if args.limit > 0:
        plan = plan.head(int(args.limit)).copy()

    n_total = len(plan)
    print(f"[plan] unique files={n_total}", flush=True)
    if n_total == 0:
        out = pd.DataFrame(columns=list(DownloadResult.__dataclass_fields__.keys()))
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.report_out, index=False)
        summary = pd.DataFrame(
            [
                {
                    "n_rows": 0,
                    "n_unique_files": 0,
                    "bytes_total": 0,
                    "gb_total": 0.0,
                    "n_downloaded": 0,
                    "n_exists": 0,
                    "n_error": 0,
                    "download_mode": True,
                }
            ]
        )
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_out, index=False)
        print("[done] empty plan", flush=True)
        return

    out_rows: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futs = [
            ex.submit(
                _download_one,
                row,
                timeout_seconds=max(1, int(args.timeout_seconds)),
                retries=max(1, int(args.retries)),
            )
            for _, row in plan.iterrows()
        ]
        for i, fut in enumerate(as_completed(futs), start=1):
            out_rows.append(fut.result())
            if i % max(1, int(args.progress_every)) == 0 or i == n_total:
                print(f"[progress] {i}/{n_total}", flush=True)

    out = pd.DataFrame([asdict(x) for x in out_rows])
    out = out.sort_values(["source", "event_id", "local_path"]).reset_index(drop=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.report_out, index=False)

    bytes_total = int(out["file_size_bytes"].fillna(0).astype(int).sum())
    summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(out)),
                "n_unique_files": int(len(out)),
                "bytes_total": bytes_total,
                "gb_total": bytes_total / (1024.0**3),
                "n_downloaded": int((out["status"] == "downloaded").sum()),
                "n_exists": int((out["status"] == "exists").sum()),
                "n_error": int((out["status"] == "error").sum()),
                "download_mode": True,
            }
        ]
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)

    s = summary.iloc[0].to_dict()
    print(
        "[done] "
        f"files={int(s['n_unique_files'])}, "
        f"downloaded={int(s['n_downloaded'])}, "
        f"exists={int(s['n_exists'])}, "
        f"errors={int(s['n_error'])}, "
        f"gb_total={float(s['gb_total']):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

