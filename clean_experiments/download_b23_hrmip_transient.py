#!/usr/bin/env python3
"""Phase-23 downloader: ECMWF-IFS-HR highresSST-present day-table ua/va,
epoch years (1979-1986, 2007-2014), sliced to 850 hPa on arrival (the
full multi-level yearly file is deleted after slicing).

Usage: python3 download_b23_hrmip_transient.py [--member r1i1p1f1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

SEARCH = "https://esgf-data.dkrz.de/esg-search/search/"
OUT = Path("data/b23hrmip")
EPOCH_YEARS = list(range(1979, 1987)) + list(range(2007, 2015))
VARIABLES = ("ua", "va")


def catalogue(member: str, var: str) -> dict[str, list[str]]:
    params = {
        "project": "CMIP6", "experiment_id": "highresSST-present",
        "source_id": "ECMWF-IFS-HR", "variant_label": member,
        "table_id": "day", "variable": var, "type": "File",
        "format": "application/solr+json", "limit": 400,
        "fields": "title,url,data_node",
    }
    r = requests.get(SEARCH, params=params, timeout=120)
    r.raise_for_status()
    out: dict[str, list[str]] = {}
    for doc in r.json()["response"]["docs"]:
        title = doc["title"]
        year = int(title.split("_gr_")[1][:4])
        if year not in EPOCH_YEARS:
            continue
        for u in doc["url"]:
            if "HTTPServer" in u:
                out.setdefault(title, []).append(u.split("|")[0])
    # prefer the DKRZ replica (CEDA has been timing out)
    for title in out:
        out[title].sort(key=lambda u: 0 if "dkrz" in u else 1)
    return out


def slice_850(full: Path, target: Path) -> None:
    import xarray as xr
    ds = xr.open_dataset(full)
    da = ds[[v for v in ("ua", "va") if v in ds][0]].sel(plev=85000.0)
    da.to_netcdf(target, encoding={da.name: {"dtype": "float32",
                                             "zlib": True, "complevel": 1}})
    ds.close()


def fetch(title: str, urls: list[str], max_retries: int = 5) -> str:
    var = title.split("_")[0]
    year = title.split("_gr_")[1][:4]
    target = OUT / f"{var}850_{year}.nc"
    if target.exists() and target.stat().st_size > 50_000_000:
        return f"SKIP {title}"
    tmp = OUT / (title + ".part")
    full = OUT / title
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 22):
                            f.write(chunk)
                tmp.rename(full)
                slice_850(full, target)
                full.unlink()
                return f"DONE {target.name} ({target.stat().st_size/1e6:.0f} MB)"
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"RETRY {title} a{attempt}: {exc}", flush=True)
                time.sleep(min(60 * attempt, 300))
    return f"FAIL {title}: {last}"


DKRZ_ROOT = ("https://esgf3.dkrz.de/thredds/fileServer/cmip6/HighResMIP/"
             "ECMWF/ECMWF-IFS-HR/highresSST-present/{member}/day/{var}/gr/"
             "{version}/{title}")
CEDA_ROOT = ("https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/"
             "HighResMIP/ECMWF/ECMWF-IFS-HR/highresSST-present/{member}/day/"
             "{var}/gr/{version}/{title}")
VERSION = {"r1i1p1f1": "v20170915"}


def pattern_jobs(member: str) -> dict[str, list[str]]:
    """Direct THREDDS URLs (the ESGF search index has been flaky);
    DKRZ replica first, CEDA fallback."""
    ver = VERSION.get(member, "v20181119")
    jobs: dict[str, list[str]] = {}
    for var in VARIABLES:
        for y in EPOCH_YEARS:
            title = (f"{var}_day_ECMWF-IFS-HR_highresSST-present_"
                     f"{member}_gr_{y}0101-{y}1231.nc")
            jobs[title] = [
                DKRZ_ROOT.format(member=member, var=var, version=ver,
                                 title=title),
                CEDA_ROOT.format(member=member, var=var, version=ver,
                                 title=title)]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--member", default="r1i1p1f1")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = pattern_jobs(args.member)
    print(f"{len(jobs)} yearly files for {args.member}", flush=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, t, u) for t, u in sorted(jobs.items())]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
