#!/usr/bin/env python3
"""Phase-8 download: MERRA-2 850 hPa U,V subsets for the 12 program regions.

Protocol: docs/PROTOCOL_PHASE8_CROSS_REANALYSIS.md (frozen 2026-08-13).
Requires NASA Earthdata credentials in ~/.netrc
(machine urs.earthdata.nasa.gov login ... password ...).

Strategy: one pass over daily M2I6NPANA granules; each granule is opened
lazily over HTTPS and the 850 hPa U,V slabs for all 12 regions are extracted
at once; per-region-window NetCDF files are assembled incrementally and are
resumable at day granularity via a JSON progress ledger.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

import download_b1_era5_wind as b1
from download_b2_era5_wind import REGIONS_B2
from download_b2b_era5_wind import REGIONS_B2B

REGIONS_ALL = {**b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}
WINDOWS = {
    "W9_2023JFM": (date(2023, 1, 1), date(2023, 3, 31)),
    "W10_2023JAS": (date(2023, 7, 1), date(2023, 9, 30)),
    "W11_2024JFM": (date(2024, 1, 1), date(2024, 3, 31)),
    "W12_2024JAS": (date(2024, 7, 1), date(2024, 9, 30)),
}
SHORT_NAME = "M2I6NPANA"


def _days(w: str):
    d0, d1 = WINDOWS[w]
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def _region_slice(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    n, w, s, e = area
    out = ds.sel(lat=slice(s, n))
    lon = out["lon"].values
    if w <= e:
        m = (lon >= w) & (lon <= e)
    else:
        m = (lon >= w) | (lon <= e)
    return out.isel(lon=np.where(m)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/b8merra2"))
    parser.add_argument("--windows", type=str, default=",".join(WINDOWS))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import earthaccess
    auth = None
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
            if auth and auth.authenticated:
                break
        except Exception:
            continue
    if not auth or not auth.authenticated:
        raise SystemExit("Earthdata login failed: set EARTHDATA_TOKEN or ~/.netrc")

    for window in args.windows.split(","):
        ledger_path = args.out_dir / f"ledger_{window}.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        store: dict[str, list[xr.Dataset]] = {r: [] for r in REGIONS_ALL}
        part_dir = args.out_dir / f"parts_{window}"
        part_dir.mkdir(exist_ok=True)

        for d in _days(window):
            key = d.isoformat()
            if ledger.get(key) == "done" or ledger.get(key) == "missing":
                continue
            ok = False
            for attempt in range(1, 5):
                try:
                    results = earthaccess.search_data(
                        short_name=SHORT_NAME, temporal=(key, key))
                    if not results:
                        print(f"[{window}] {key}: no granule", flush=True)
                        ledger[key] = "missing"
                        ok = True
                        break
                    files = earthaccess.open(results)
                    ds = xr.open_dataset(files[0], engine="h5netcdf")
                    slab = ds[["U", "V"]].sel(lev=850.0).load()
                    ds.close()
                    for region, area in REGIONS_ALL.items():
                        sub = _region_slice(slab, area)
                        sub.to_netcdf(part_dir / f"{region}__{key}.nc")
                    ledger[key] = "done"
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001 - network retry
                    print(f"[{window}] {key}: attempt {attempt} failed "
                          f"({type(exc).__name__})", flush=True)
                    import time as _t
                    _t.sleep(60 * attempt)
            if not ok:
                ledger[key] = "error"
                print(f"[{window}] {key}: ERROR after retries", flush=True)
            ledger_path.write_text(json.dumps(ledger))
            if ok and ledger[key] == "done":
                print(f"[{window}] {key}: done", flush=True)

        # assemble per-region files
        for region in REGIONS_ALL:
            target = args.out_dir / f"merra2_wind850_{region}__{window}.nc"
            if target.exists():
                continue
            parts = sorted(part_dir.glob(f"{region}__*.nc"))
            if not parts:
                continue
            merged = xr.concat([xr.open_dataset(p) for p in parts], dim="time")
            merged = merged.rename({"U": "u", "V": "v"})
            merged.to_netcdf(target)
            print(f"[{window}] assembled {region} ({len(parts)} days)", flush=True)

    print("SUMMARY: merra2 download pass complete", flush=True)


if __name__ == "__main__":
    main()
