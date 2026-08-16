#!/usr/bin/env python3
"""Phase-13 data: CMIP6 HighResMIP free-running (no data assimilation) winds.

Protocol: docs/PROTOCOL_PHASE13_FREE_RUNNING.md.

highresSST-present is an AMIP-type integration: prescribed SST, no
assimilation of any atmospheric observation. If the regional geography of A
survives there, the observing-network confound the manuscript declares
unresolved is closed; if it collapses, A describes where data assimilation
has nothing to fill the mesoscale with.

Files are whole monthly globals on the DKRZ ESGF node (no OPeNDAP), fetched
with resumable range requests.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SEARCH = "https://esgf-data.dkrz.de/esg-search/search/"
MODELS = {
    "ECMWF-IFS-HR": {"variant": "r1i1p1f1", "res": "25 km"},
    "CMCC-CM2-VHR4": {"variant": "r1i1p1f1", "res": "25 km"},
}
MONTHS = {"W15_2014JFM": ("2014", ["01", "02", "03"]),
          "W16_2014JAS": ("2014", ["07", "08", "09"])}
VARIABLES = ("ua", "va")


def catalogue(model: str, variable: str) -> dict[str, str]:
    params = {
        "project": "CMIP6", "activity_id": "HighResMIP",
        "experiment_id": "highresSST-present", "variable": variable,
        "frequency": "6hr", "source_id": model,
        "variant_label": MODELS[model]["variant"], "type": "File",
        "format": "application/solr+json", "limit": "3000",
        "fields": "title,url,data_node",
    }
    r = requests.get(SEARCH, params=params, timeout=180)
    r.raise_for_status()
    out: dict[str, str] = {}
    for doc in r.json()["response"]["docs"]:
        if doc.get("data_node") != "esgf3.dkrz.de":
            continue          # only node that serves us at a usable rate
        for u in doc["url"]:
            if "HTTPServer" in u:
                out.setdefault(doc["title"], u.split("|")[0])
    return out


def wanted(titles: dict[str, str]) -> dict[str, str]:
    keep = {}
    for window, (year, months) in MONTHS.items():
        for m in months:
            stamp = f"{year}{m}"
            for t, u in titles.items():
                if f"_{stamp}01" in t:
                    keep[t] = u
    return keep


def fetch(title: str, url: str, out_dir: Path) -> str:
    target = out_dir / title
    if target.exists() and target.stat().st_size > 10_000_000:
        return f"SKIP {title} ({target.stat().st_size/1e9:.2f} GB)"
    part = target.with_suffix(".part")
    for attempt in range(6):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            total = int(requests.head(url, timeout=120).headers.get("content-length", 0))
            if total and part.stat().st_size < total:
                raise RuntimeError(f"short: {part.stat().st_size}/{total}")
            part.replace(target)
            return f"OK {title} ({target.stat().st_size/1e9:.2f} GB)"
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                return f"FAIL {title}: {exc}"
            time.sleep(10 * (attempt + 1))
    return f"FAIL {title}: exhausted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), default="ECMWF-IFS-HR")
    ap.add_argument("--out-dir", type=Path, default=Path("data/b13hrmip"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    out_dir = args.out_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: dict[str, str] = {}
    for var in VARIABLES:
        jobs.update(wanted(catalogue(args.model, var)))
    print(f"{args.model}: {len(jobs)} monthly files -> {out_dir}", flush=True)
    for t in sorted(jobs):
        print("   ", t, flush=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(fetch, t, u, out_dir) for t, u in sorted(jobs.items())]
        for fut in as_completed(futs):
            msg = fut.result()
            print(msg, flush=True)
            fail += msg.startswith("FAIL")
            ok += not msg.startswith("FAIL")
    print(f"SUMMARY {args.model}: ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
