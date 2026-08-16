#!/usr/bin/env python3
"""Phase-9 data: NOAA GEFS v12 ensemble 850 hPa wind, regional subsets.

Protocol: docs/PROTOCOL_PHASE9_PREDICTABILITY.md (frozen 2026-08-15).

Pulls only the UGRD/VGRD 850 hPa GRIB messages via HTTP byte-range requests
against the published .idx files on the AWS Open Data mirror (anonymous),
extracts all 12 program regions from each global field, and writes one
compressed npz per (init date, lead) holding every member and region.

Usage:
    python clean_experiments/download_b9_gefs.py --sample primary
    python clean_experiments/download_b9_gefs.py --sample oos
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import clean_experiments.download_b1_era5_wind as dl_b1
    from clean_experiments.download_b2_era5_wind import REGIONS_B2
    from clean_experiments.download_b2b_era5_wind import REGIONS_B2B
except ImportError:  # pragma: no cover
    import download_b1_era5_wind as dl_b1  # type: ignore
    from download_b2_era5_wind import REGIONS_B2  # type: ignore
    from download_b2b_era5_wind import REGIONS_B2B  # type: ignore

REGIONS_ALL = {**dl_b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
# 6 members (control + 5 perturbed). Fixed and identical for every region and
# window, so ensemble size is a constant of the design, not a covariate.
# Files downloaded before this was fixed hold 11 members; the experiment slices
# every stack to the first N_MEMBERS in the same fixed order.
N_MEMBERS = 6
MEMBERS = (["gec00"] + [f"gep{i:02d}" for i in range(1, 11)])[:N_MEMBERS]
LEADS = [12, 24, 48, 72, 96, 120, 144, 168]
INIT_STRIDE_DAYS = 5

WINDOWS = {
    "W5_2021JFM": ("2021-01-01", "2021-03-31"),
    "W6_2021JAS": ("2021-07-01", "2021-09-30"),
    "W7_2022JFM": ("2022-01-01", "2022-03-31"),
    "W8_2022JAS": ("2022-07-01", "2022-09-30"),
    "W9_2023JFM": ("2023-01-01", "2023-03-31"),
    "W10_2023JAS": ("2023-07-01", "2023-09-30"),
    "W11_2024JFM": ("2024-01-01", "2024-03-31"),
    "W12_2024JAS": ("2024-07-01", "2024-09-30"),
}
SAMPLES = {
    "primary": ["W9_2023JFM", "W10_2023JAS", "W11_2024JFM", "W12_2024JAS"],
    "oos": ["W5_2021JFM", "W6_2021JAS", "W7_2022JFM", "W8_2022JAS"],
    # Phase 10 (docs/PROTOCOL_PHASE10_ANALYSIS_IRRECOVERABILITY.md)
    "b10": ["W11_2024JFM", "W12_2024JAS"],
}

# GEFS 0.5 deg global grid
NLAT, NLON = 361, 720


def region_indices(area: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """area = [north, west, south, east] in degrees -> (lat_idx, lon_idx)."""
    north, west, south, east = area
    lats = np.arange(north, south - 1e-9, -0.5)
    lat_idx = np.rint((90.0 - lats) * 2).astype(int)
    lons = np.arange(west, east + 1e-9, 0.5)
    lon_idx = (np.rint(lons * 2).astype(int)) % NLON
    return lat_idx, lon_idx


REGION_IDX = {r: region_indices(a) for r, a in REGIONS_ALL.items()}


def use_domains(which: str) -> None:
    """Swap the extracted domain set (programme regions vs Phase-10 lattice)."""
    global REGION_IDX
    if which == "program":
        regions = REGIONS_ALL
    else:
        try:
            from clean_experiments.regions_b10 import REGIONS_B10
        except ImportError:  # pragma: no cover
            from regions_b10 import REGIONS_B10  # type: ignore
        regions = REGIONS_B10 if which == "b10" else {**REGIONS_ALL, **REGIONS_B10}
    REGION_IDX = {r: region_indices(a) for r, a in regions.items()}


REGION_COORDS = {
    r: (
        np.arange(a[0], a[2] - 1e-9, -0.5),
        np.arange(a[1], a[3] + 1e-9, 0.5),
    )
    for r, a in REGIONS_ALL.items()
}


def init_dates(window: str) -> list[dt.date]:
    d0, d1 = (dt.date.fromisoformat(s) for s in WINDOWS[window])
    out, d = [], d0
    while d <= d1:
        out.append(d)
        d += dt.timedelta(days=INIT_STRIDE_DAYS)
    return out


def _get(url: str, headers: dict | None = None, tries: int = 4) -> bytes:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            if r.status_code in (200, 206):
                return r.content
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(1.5 * (k + 1))
    raise last  # type: ignore[misc]


def member_fields(date: dt.date, member: str, lead: int, cycle: str = "00") -> np.ndarray:
    """Return global (2, NLAT, NLON) array [u, v] at 850 hPa."""
    url = (
        f"{BASE}/gefs.{date:%Y%m%d}/{cycle}/atmos/pgrb2ap5/"
        f"{member}.t{cycle}z.pgrb2a.0p50.f{lead:03d}"
    )
    idx_txt = _get(url + ".idx").decode("utf-8", "replace")
    lines = [ln for ln in idx_txt.strip().split("\n") if ln]
    starts = [int(ln.split(":")[1]) for ln in lines]
    want = {}
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        if parts[3] in ("UGRD", "VGRD") and parts[4] == "850 mb":
            end = starts[i + 1] - 1 if i + 1 < len(starts) else ""
            want[parts[3]] = (starts[i], end)
    if len(want) != 2:
        raise RuntimeError(f"850 hPa wind not found in idx: {url}")

    import xarray as xr

    out = np.empty((2, NLAT, NLON), dtype=np.float32)
    for j, var in enumerate(("UGRD", "VGRD")):
        s, e = want[var]
        blob = _get(url, headers={"Range": f"bytes={s}-{e}"})
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as fh:
            fh.write(blob)
            tmp = fh.name
        try:
            ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})
            name = list(ds.data_vars)[0]
            arr = np.asarray(ds[name].values, dtype=np.float32)
            ds.close()
        finally:
            for suffix in ("", ".923a8.idx"):
                try:
                    os.unlink(tmp + suffix)
                except OSError:
                    pass
        if arr.shape != (NLAT, NLON):
            raise RuntimeError(f"unexpected grid {arr.shape} for {url}")
        out[j] = arr
    return out


def job(date: dt.date, lead: int, out_dir: Path, inner: int) -> str:
    target = out_dir / f"gefs_{date:%Y%m%d}_f{lead:03d}.npz"
    if target.exists():
        return f"cached {target.name}"
    try:
        with ThreadPoolExecutor(max_workers=inner) as pool:
            futs = {pool.submit(member_fields, date, m, lead): m for m in MEMBERS}
            glob_fields = {}
            for fut in as_completed(futs):
                glob_fields[futs[fut]] = fut.result()
        payload = {}
        for region, (li, lo) in REGION_IDX.items():
            stack = np.empty((len(MEMBERS), 2, len(li), len(lo)), dtype=np.float32)
            for k, m in enumerate(MEMBERS):
                stack[k] = glob_fields[m][:, li[:, None], lo[None, :]]
            payload[region] = stack
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(target)
        return f"ok {target.name}"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {date:%Y%m%d} f{lead:03d}: {exc}"


def analysis_job(date: dt.date, cycle: str, out_dir: Path) -> str:
    """GEFS control analysis (gec00 f000) = the system's own verifying analysis."""
    target = out_dir / f"anl_{date:%Y%m%d}_t{cycle}z.npz"
    if target.exists():
        return f"cached {target.name}"
    try:
        g = member_fields(date, "gec00", 0, cycle=cycle)
        payload = {}
        for region, (li, lo) in REGION_IDX.items():
            payload[region] = g[:, li[:, None], lo[None, :]].astype(np.float32)
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(target)
        return f"ok {target.name}"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL anl {date:%Y%m%d} {cycle}z: {exc}"


def analysis_slots_daily(window: str) -> list[tuple[dt.date, str]]:
    """Every 00 and 12 UTC analysis inside the window (Phase-10 D statistic)."""
    d0, d1 = (dt.date.fromisoformat(s) for s in WINDOWS[window])
    out, d = [], d0
    while d <= d1:
        out.append((d, "00"))
        out.append((d, "12"))
        d += dt.timedelta(days=1)
    return out


def analysis_slots(window: str) -> list[tuple[dt.date, str]]:
    """Valid times needed to verify every (init, lead) pair of the window."""
    d0, d1 = (dt.date.fromisoformat(s) for s in WINDOWS[window])
    slots: set[tuple[dt.date, str]] = set()
    for d in init_dates(window):
        for lead in LEADS:
            v = dt.datetime(d.year, d.month, d.day) + dt.timedelta(hours=lead)
            if d0 <= v.date() <= d1:
                slots.add((v.date(), f"{v.hour:02d}"))
    return sorted(slots)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", choices=sorted(SAMPLES), default="primary")
    ap.add_argument("--out-dir", type=Path, default=Path("data/b9gefs"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inner", type=int, default=8)
    ap.add_argument("--analysis", action="store_true",
                    help="download the GEFS control analysis instead of forecasts")
    ap.add_argument("--analysis-mode", choices=["slots", "daily"], default="slots")
    ap.add_argument("--domains", choices=["program", "b10", "both"],
                    default="program")
    ap.add_argument("--leads", type=int, nargs="*", default=None,
                    help="override the lead set (hours)")
    args = ap.parse_args()

    use_domains(args.domains)
    if args.leads:
        global LEADS
        LEADS = list(args.leads)

    out_dir = args.out_dir / args.sample
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.analysis:
        picker = analysis_slots_daily if args.analysis_mode == "daily" else analysis_slots
        slots = sorted({s for w in SAMPLES[args.sample] for s in picker(w)})
        print(f"{len(slots)} analysis slots -> {out_dir}", flush=True)
        done = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers * args.inner) as pool:
            futs = [pool.submit(analysis_job, d, c, out_dir) for d, c in slots]
            for i, fut in enumerate(as_completed(futs), 1):
                msg = fut.result()
                if msg.startswith("FAIL"):
                    fail += 1
                    print(msg, flush=True)
                else:
                    done += 1
                if i % 50 == 0:
                    print(f"  {i}/{len(slots)} ok={done} fail={fail}", flush=True)
        print(f"SUMMARY: ok={done} fail={fail}", flush=True)
        if fail:
            raise SystemExit(1)
        return

    jobs = []
    for w in SAMPLES[args.sample]:
        for d in init_dates(w):
            for lead in LEADS:
                jobs.append((d, lead))
    print(f"{len(jobs)} (init, lead) jobs -> {out_dir}", flush=True)

    done = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(job, d, lead, out_dir, args.inner) for d, lead in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if msg.startswith("FAIL"):
                fail += 1
                print(msg, flush=True)
            else:
                done += 1
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} ok={done} fail={fail}", flush=True)
    print(f"SUMMARY: ok={done} fail={fail}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
