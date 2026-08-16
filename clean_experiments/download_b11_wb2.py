#!/usr/bin/env python3
"""Phase-11 data: WeatherBench 2 forecast systems, 850 hPa wind, domain subsets.

Protocol: docs/PROTOCOL_PHASE11_LEARNED_REFINEMENT.md (frozen 2026-08-15).

Reads zarr v2 chunks straight over anonymous HTTPS (no zarr/gcsfs needed):
each chunk spans the full level/lat/lon extent, so one GET yields a global
field, from which every programme domain is extracted before the global array
is discarded.

Usage:
    python clean_experiments/download_b11_wb2.py --system graphcast
    python clean_experiments/download_b11_wb2.py --system era5
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numcodecs
import numpy as np
import requests

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.download_b9_gefs import REGIONS_ALL
    from clean_experiments.regions_b10 import REGIONS_B10
except ImportError:  # pragma: no cover
    from download_b9_gefs import REGIONS_ALL  # type: ignore
    from regions_b10 import REGIONS_B10  # type: ignore

DOMAINS = {**REGIONS_ALL, **REGIONS_B10}
BASE = "https://storage.googleapis.com/weatherbench2/datasets/"

SYSTEMS = {
    "graphcast": {"path": "graphcast_v2/2020-1440x721.zarr", "kind": "forecast"},
    "pangu": {"path": "pangu/2018-2022_0012_0p25.zarr", "kind": "forecast"},
    "gencast_mean": {"path": "gencast/2020-1440x721_mean.zarr", "kind": "forecast"},
    "gencast_member": {"path": "gencast/2020-1440x721.zarr", "kind": "forecast",
                       "sample": 0},
    "hres": {"path": "hres/2016-2022-0012-1440x721.zarr", "kind": "forecast"},
    "era5": {"path": "era5/1959-2023_01_10-wb13-6h-1440x721.zarr", "kind": "truth"},
}

VARS = ("u_component_of_wind", "v_component_of_wind")
LEVEL_HPA = 850
LEADS = (24, 120)
INIT_STRIDE_DAYS = 10
WINDOWS = {"W13_2020JFM": ("2020-01-01", "2020-03-31"),
           "W14_2020JAS": ("2020-07-01", "2020-09-30")}
GENCAST_MEMBER_MAX_INITS = 6      # per window; fixed in the protocol


def init_dates() -> list[dt.datetime]:
    out = []
    for _w, (a, b) in WINDOWS.items():
        d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        d = d0
        while d <= d1:
            out.append(dt.datetime(d.year, d.month, d.day))
            d += dt.timedelta(days=INIT_STRIDE_DAYS)
    return out


class Store:
    """Minimal read-only zarr v2 reader over HTTPS."""

    def __init__(self, path: str):
        self.url = BASE + path.rstrip("/")
        self.meta = requests.get(self.url + "/.zmetadata", timeout=60).json()["metadata"]
        self.lat, self.lat_name = self._coord_any(("lat", "latitude"))
        self.lon, self.lon_name = self._coord_any(("lon", "longitude"))
        self.level = self._coord("level")
        self.time, self.time_units = self._coord("time"), \
            self.meta["time/.zattrs"].get("units", "")
        self.lead = self._coord("prediction_timedelta") \
            if "prediction_timedelta/.zarray" in self.meta else None

    def _raw(self, name: str, key: str) -> bytes:
        for attempt in range(4):
            try:
                r = requests.get(f"{self.url}/{name}/{key}", timeout=300)
                if r.status_code == 200:
                    return r.content
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"chunk {name}/{key} unavailable")

    def _coord(self, name: str) -> np.ndarray:
        """Read a 1-D coordinate, concatenating every chunk (they are chunked)."""
        za = self.meta[name + "/.zarray"]
        n, csize = int(za["shape"][0]), int(za["chunks"][0])
        pieces = []
        for k in range((n + csize - 1) // csize):
            raw = self._raw(name, str(k))
            if za.get("compressor"):
                raw = numcodecs.get_codec(za["compressor"]).decode(raw)
            pieces.append(np.frombuffer(raw, dtype=np.dtype(za["dtype"])))
        return np.concatenate(pieces)[:n].copy()

    def _coord_any(self, names) -> tuple[np.ndarray, str]:
        for n in names:
            if n + "/.zarray" in self.meta:
                return self._coord(n), n
        raise KeyError(names)

    def dims(self, var: str) -> list[str]:
        return list(self.meta[var + "/.zattrs"]["_ARRAY_DIMENSIONS"])

    def time_index(self, when: dt.datetime) -> int:
        base = self.time_units.split("since")[-1].strip()
        origin = dt.datetime.fromisoformat(base.replace(" ", "T")[:19])
        hours = (when - origin).total_seconds() / 3600.0
        idx = int(np.argmin(np.abs(self.time.astype(float) - hours)))
        if abs(float(self.time[idx]) - hours) > 1e-6:
            raise KeyError(f"{when} not in {self.url}")
        return idx

    def field(self, var: str, when: dt.datetime, lead_h: int | None,
              sample: int | None = None) -> np.ndarray:
        """Global (nlat, nlon) field at LEVEL_HPA."""
        za = self.meta[var + "/.zarray"]
        dims, chunks, shape = self.dims(var), za["chunks"], za["shape"]
        pos = {}
        pos["time"] = self.time_index(when)
        if lead_h is not None:
            pos["prediction_timedelta"] = int(np.where(self.lead == lead_h)[0][0])
        if sample is not None:
            pos["sample"] = sample
        key = []
        for d, c, s in zip(dims, chunks, shape):
            idx = pos.get(d, 0)
            key.append(str(idx // c) if c == 1 else "0")
        raw = self._raw(var, ".".join(key))
        if za.get("compressor"):
            raw = numcodecs.get_codec(za["compressor"]).decode(raw)
        arr = np.frombuffer(raw, dtype=np.dtype(za["dtype"])).reshape(chunks)
        lev = int(np.where(self.level == LEVEL_HPA)[0][0])
        sel = []
        for d, c in zip(dims, chunks):
            if d == "level":
                sel.append(lev)
            elif d in (self.lat_name, self.lon_name):
                sel.append(slice(None))
            elif d == "sample" and c > 1:
                sel.append(sample or 0)
            else:
                sel.append(0)
        return np.asarray(arr[tuple(sel)], dtype=np.float32)


def domain_indices(lat: np.ndarray, lon: np.ndarray) -> dict:
    out = {}
    step = float(abs(lat[1] - lat[0]))
    for name, (n, w, s, e) in DOMAINS.items():
        want_lat = np.arange(n, s - 1e-9, -step)
        li = np.array([int(np.argmin(np.abs(lat - v))) for v in want_lat])
        want_lon = (np.arange(w, e + 1e-9, step) + 360.0) % 360.0
        lo = np.array([int(np.argmin(np.abs(((lon + 360) % 360) - v))) for v in want_lon])
        out[name] = (li, lo)
    return out


def run(system: str, out_dir: Path, workers: int) -> None:
    cfg = SYSTEMS[system]
    store = Store(cfg["path"])
    idx = domain_indices(store.lat, store.lon)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    inits = init_dates()
    if system == "gencast_member":
        per_window = {}
        for d in inits:
            w = "W13_2020JFM" if d.month < 7 else "W14_2020JAS"
            per_window.setdefault(w, []).append(d)
        inits = [d for w, ds in per_window.items()
                 for d in ds[:GENCAST_MEMBER_MAX_INITS]]
    if cfg["kind"] == "truth":
        for d in inits:
            for lead in LEADS:
                jobs.append((d + dt.timedelta(hours=lead), None))
        jobs = sorted(set(jobs))
    else:
        jobs = [(d, lead) for d in inits for lead in LEADS]

    print(f"{system}: {len(jobs)} chunks x {len(VARS)} vars -> {out_dir}", flush=True)

    def one(when: dt.datetime, lead: int | None) -> str:
        stamp = f"{when:%Y%m%d%H}" + ("" if lead is None else f"_f{lead:03d}")
        target = out_dir / f"{system}_{stamp}.npz"
        if target.exists():
            return f"cached {target.name}"
        try:
            payload = {}
            fields = [store.field(v, when, lead,
                                  sample=cfg.get("sample")) for v in VARS]
            for name, (li, lo) in idx.items():
                payload[name] = np.stack(
                    [f[li[:, None], lo[None, :]] for f in fields]).astype(np.float32)
            tmp = target.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **payload)
            tmp.replace(target)
            return f"ok {target.name}"
        except Exception as exc:  # noqa: BLE001
            return f"FAIL {stamp}: {exc}"

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, w, l) for w, l in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if msg.startswith("FAIL"):
                fail += 1
                print(msg, flush=True)
            else:
                ok += 1
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} ok={ok} fail={fail}", flush=True)
    print(f"SUMMARY {system}: ok={ok} fail={fail}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/b11wb2"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    run(args.system, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
