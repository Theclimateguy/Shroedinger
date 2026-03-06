#!/usr/bin/env python3
"""Selective downloader: fetch only MRMS/GOES files referenced by aligned matched windows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from _open_s3_http import download_public_object


@dataclass(frozen=True)
class SelectiveRow:
    source: str
    event_id: str
    key: str
    url: str
    local_path: str
    file_size_bytes: int
    status: str
    error: str


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-catalog", type=Path, default=Path("aligned_catalog.parquet"))
    parser.add_argument("--mrms-manifest", type=Path, default=Path("manifests/mrms_manifest.csv"))
    parser.add_argument("--goes-manifest", type=Path, default=Path("manifests/goes_manifest.csv"))
    parser.add_argument("--event-ids", nargs="+", default=[])
    parser.add_argument("--source", choices=["both", "mrms", "goes"], default="both")
    parser.add_argument("--download", action="store_true", help="Actually download files (otherwise estimate only)")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--report", type=Path, default=Path("manifests/matched_download_report.csv"))
    parser.add_argument("--summary", type=Path, default=Path("manifests/matched_download_summary.csv"))
    return parser.parse_args()


def _extract_keys(aligned: pd.DataFrame, source: str) -> pd.DataFrame:
    key_col = f"{source}_key"
    event_col = "event_id"
    if key_col not in aligned.columns:
        return pd.DataFrame(columns=["event_id", "key"])
    out = aligned[[event_col, key_col]].copy() if event_col in aligned.columns else aligned[[key_col]].copy()
    if event_col not in out.columns:
        out[event_col] = ""
    out = out.rename(columns={key_col: "key"})
    out = out[out["key"].astype(str).str.len() > 0]
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def _prepare_manifest(manifest: pd.DataFrame, source: str) -> pd.DataFrame:
    cols = ["event_id", "key", "url", "local_path", "file_size_bytes", "status"]
    for c in cols:
        if c not in manifest.columns:
            if c == "event_id":
                manifest[c] = ""
            elif c == "file_size_bytes":
                manifest[c] = 0
            else:
                manifest[c] = ""
    manifest = manifest[cols].copy()
    manifest["source"] = source
    manifest = manifest.drop_duplicates(subset=["key", "local_path"]).reset_index(drop=True)
    return manifest


def main() -> None:
    args = parse_args()

    aligned = _load_table(args.aligned_catalog)
    if "match_within_tolerance" not in aligned.columns:
        raise ValueError("aligned catalog missing match_within_tolerance column")
    aligned = aligned[aligned["match_within_tolerance"].fillna(False)].copy()

    if args.event_ids:
        if "event_id" not in aligned.columns:
            raise ValueError("--event-ids was passed but aligned catalog has no event_id column")
        aligned = aligned[aligned["event_id"].astype(str).isin(set(args.event_ids))].copy()

    mrms_manifest = _prepare_manifest(_load_table(args.mrms_manifest), source="mrms")
    goes_manifest = _prepare_manifest(_load_table(args.goes_manifest), source="goes")

    want_rows = []
    if args.source in {"both", "mrms"}:
        mrms_keys = _extract_keys(aligned, "mrms")
        if not mrms_keys.empty:
            want_rows.append(mrms_keys.merge(mrms_manifest, on=["event_id", "key"], how="left"))
    if args.source in {"both", "goes"}:
        goes_keys = _extract_keys(aligned, "goes")
        if not goes_keys.empty:
            want_rows.append(goes_keys.merge(goes_manifest, on=["event_id", "key"], how="left"))

    if len(want_rows) == 0:
        report = pd.DataFrame(columns=["source", "event_id", "key", "url", "local_path", "file_size_bytes", "status", "error"])
    else:
        plan = pd.concat(want_rows, ignore_index=True)
        plan = plan.drop_duplicates(subset=["source", "event_id", "key", "local_path"]).reset_index(drop=True)

        out_rows: list[SelectiveRow] = []
        for _, r in plan.iterrows():
            source = str(r.get("source", ""))
            event_id = str(r.get("event_id", ""))
            key = str(r.get("key", ""))
            url = str(r.get("url", ""))
            local_path = str(r.get("local_path", ""))
            size_raw = r.get("file_size_bytes", 0)
            if pd.isna(size_raw):
                size = 0
            else:
                try:
                    size = int(float(size_raw))
                except Exception:  # noqa: BLE001
                    size = 0

            if len(local_path) == 0:
                out_rows.append(
                    SelectiveRow(source, event_id, key, url, local_path, size, "missing_manifest", "local_path not found in manifest")
                )
                continue
            if len(url) == 0:
                out_rows.append(
                    SelectiveRow(source, event_id, key, url, local_path, size, "missing_manifest", "url not found in manifest")
                )
                continue

            status = "planned"
            error = ""
            if args.download:
                try:
                    status = download_public_object(
                        url=url,
                        target_path=Path(local_path),
                        expected_size=(size if size > 0 else None),
                        timeout_seconds=args.timeout_seconds,
                    )
                except Exception as exc:  # noqa: BLE001
                    status = "error"
                    error = str(exc)

            out_rows.append(
                SelectiveRow(
                    source=source,
                    event_id=event_id,
                    key=key,
                    url=url,
                    local_path=local_path,
                    file_size_bytes=size,
                    status=status,
                    error=error,
                )
            )

        report = pd.DataFrame([asdict(x) for x in out_rows])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)

    if report.empty:
        summary = pd.DataFrame(
            [{"n_rows": 0, "n_unique_files": 0, "bytes_total": 0, "gb_total": 0.0, "n_errors": 0, "n_missing_manifest": 0}]
        )
    else:
        unique_files = report.drop_duplicates(subset=["source", "local_path"])
        bytes_total = int(unique_files["file_size_bytes"].fillna(0).astype(float).sum())
        summary = pd.DataFrame(
            [
                {
                    "n_rows": int(len(report)),
                    "n_unique_files": int(len(unique_files)),
                    "bytes_total": bytes_total,
                    "gb_total": bytes_total / (1024.0**3),
                    "n_errors": int((report["status"] == "error").sum()),
                    "n_missing_manifest": int((report["status"] == "missing_manifest").sum()),
                    "n_downloaded": int((report["status"] == "downloaded").sum()),
                    "n_exists": int((report["status"] == "exists").sum()),
                    "download_mode": bool(args.download),
                }
            ]
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    s = summary.iloc[0].to_dict()
    print(f"Selective report written: {args.report}")
    print(f"Selective summary written: {args.summary}")
    print(
        "Selective summary: "
        f"files={int(s['n_unique_files'])}, gb_total={float(s['gb_total']):.3f}, "
        f"download_mode={bool(s.get('download_mode', False))}, errors={int(s['n_errors'])}"
    )


if __name__ == "__main__":
    main()
