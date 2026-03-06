#!/usr/bin/env python3
"""Utilities for reading public S3 buckets over plain HTTPS list/download APIs.

This module avoids hard dependency on awscli/boto3 and works against NOAA open-data
buckets via `https://<bucket>.s3.amazonaws.com/?list-type=2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import quote
import time
import xml.etree.ElementTree as ET

import requests


_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass(frozen=True)
class S3Object:
    """Metadata row for a public S3 object key."""

    bucket: str
    key: str
    size: int
    etag: str | None
    last_modified: datetime | None
    storage_class: str | None

    @property
    def url(self) -> str:
        encoded = quote(self.key, safe="/")
        return f"https://{self.bucket}.s3.amazonaws.com/{encoded}"


def _parse_iso8601_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) == 0:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def iter_public_s3_objects(
    *,
    bucket: str,
    prefix: str,
    max_keys: int = 1000,
    timeout_seconds: int = 60,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> Iterator[S3Object]:
    """Yield S3 objects under `prefix` using public list API (V2)."""
    if max_keys <= 0:
        raise ValueError("max_keys must be > 0")
    if retries <= 0:
        raise ValueError("retries must be > 0")

    endpoint = f"https://{bucket}.s3.amazonaws.com/"
    continuation: str | None = None

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": str(max_keys),
        }
        if continuation is not None:
            params["continuation-token"] = continuation

        response_text = ""
        attempt = 0
        while True:
            attempt += 1
            try:
                response = requests.get(endpoint, params=params, timeout=timeout_seconds)
                response.raise_for_status()
                response_text = response.text
                break
            except Exception:
                if attempt >= retries:
                    raise
                time.sleep(backoff_seconds * attempt)

        root = ET.fromstring(response_text)
        for node in root.findall("s3:Contents", _S3_NS):
            key = node.findtext("s3:Key", default="", namespaces=_S3_NS)
            if len(key) == 0:
                continue
            size_text = node.findtext("s3:Size", default="0", namespaces=_S3_NS)
            etag_text = node.findtext("s3:ETag", default="", namespaces=_S3_NS)
            storage_class = node.findtext("s3:StorageClass", default="", namespaces=_S3_NS)
            yield S3Object(
                bucket=bucket,
                key=key,
                size=int(size_text),
                etag=etag_text.strip('"') if len(etag_text) > 0 else None,
                last_modified=_parse_iso8601_utc(
                    node.findtext("s3:LastModified", default="", namespaces=_S3_NS)
                ),
                storage_class=storage_class or None,
            )

        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS)
        if truncated.lower() != "true":
            break
        continuation = root.findtext(
            "s3:NextContinuationToken", default=None, namespaces=_S3_NS
        )
        if continuation is None:
            # Defensive fallback for non-standard responses.
            break


def download_public_object(
    *,
    url: str,
    target_path: Path,
    expected_size: int | None = None,
    chunk_bytes: int = 1024 * 1024,
    timeout_seconds: int = 120,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    """Download URL to target path with idempotence and atomic rename.

    Returns one of: "exists", "downloaded".
    """
    if target_path.exists():
        if expected_size is None:
            return "exists"
        if target_path.stat().st_size == expected_size:
            return "exists"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    attempt = 0
    while True:
        attempt += 1
        try:
            with requests.get(url, stream=True, timeout=timeout_seconds) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if not chunk:
                            continue
                        out.write(chunk)
            if expected_size is not None and tmp_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch for {url}: got {tmp_path.stat().st_size}, "
                    f"expected {expected_size}"
                )
            tmp_path.replace(target_path)
            return "downloaded"
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt >= retries:
                raise
            time.sleep(backoff_seconds * attempt)


def safe_json_dump(path: Path, payload: dict) -> None:
    """Write JSON payload atomically."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    tmp.replace(path)
