#!/usr/bin/env python3
"""Phase-33 Arm C data: WeatherBench-2 forecasts for 2022, 850 hPa wind, 39 domains.

Protocol: docs/PROTOCOL_PHASE33_MODEL_FIDELITY.md, section "Arm C" (frozen 2026-10-15).

Reuses the Phase-11 zarr reader (`download_b11_wb2.Store`) but generalises chunk
indexing: a dimension whose chunk holds several entries (Aurora: 5 leads per chunk;
IFS ENS: 50 members per chunk) is addressed by chunk key idx // c and in-chunk
index idx % c.  The Phase-11 reader assumed one entry per chunk on those axes.

Usage:
    python clean_experiments/download_b33c_wb2_2022.py --system graphcast
    python clean_experiments/download_b33c_wb2_2022.py --system ifs_ens_member
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numcodecs
import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import download_b11_wb2 as dl  # noqa: E402

SYSTEMS = {
    "graphcast": {"path": "graphcast_v2/2022-1440x721.zarr", "kind": "forecast"},
    "pangu": {"path": "pangu/2018-2022_0012_0p25.zarr", "kind": "forecast"},
    "hres": {"path": "hres/2016-2022-0012-1440x721.zarr", "kind": "forecast"},
    "aurora": {"path": "aurora/2022-1440x721.zarr", "kind": "forecast"},
    "ifs_ens_mean": {"path": "ifs_ens/2018-2022-1440x721_mean.zarr", "kind": "forecast"},
    "ifs_ens_member": {"path": "ifs_ens/2018-2022-1440x721.zarr", "kind": "forecast",
                       "member_dim": "number", "member_index": 0},
    "era5": {"path": "era5/1959-2023_01_10-wb13-6h-1440x721.zarr", "kind": "truth"},
}
WINDOWS = {"W17_2022JFM": ("2022-01-01", "2022-03-31"),
           "W18_2022AMJ": ("2022-04-01", "2022-06-30"),
           "W19_2022JAS": ("2022-07-01", "2022-09-30"),
           "W20_2022OND": ("2022-10-01", "2022-12-31")}
STRIDE_DAYS = 5
MEMBER_MAX_INITS = 6          # per window, IFS ENS member only
LEADS = dl.LEADS              # (24, 120)


def init_dates(member: bool = False) -> list[dt.datetime]:
    out = []
    for _w, (a, b) in WINDOWS.items():
        d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        ds, d = [], d0
        while d <= d1:
            ds.append(dt.datetime(d.year, d.month, d.day))
            d += dt.timedelta(days=STRIDE_DAYS)
        out += ds[:MEMBER_MAX_INITS] if member else ds
    return out


class Store(dl.Store):
    def field(self, var: str, when: dt.datetime, lead_h: int | None,
              member_dim: str | None = None, member_index: int = 0) -> np.ndarray:
        za = self.meta[var + "/.zarray"]
        dims, chunks = self.dims(var), za["chunks"]
        pos = {"time": self.time_index(when)}
        if lead_h is not None:
            hits = np.where(self.lead.astype(float) == float(lead_h))[0]
            if hits.size == 0:
                raise KeyError(f"lead {lead_h} h not in {self.url}")
            pos["prediction_timedelta"] = int(hits[0])
        if member_dim is not None:
            pos[member_dim] = member_index
        key, sel = [], []
        lev = int(np.where(self.level == dl.LEVEL_HPA)[0][0])
        for d, c in zip(dims, chunks):
            if d in (self.lat_name, self.lon_name):
                key.append("0"); sel.append(slice(None))
            elif d == "level":
                key.append(str(lev // c)); sel.append(lev % c)
            else:
                idx = pos.get(d, 0)
                key.append(str(idx // c)); sel.append(idx % c)
        raw = self._raw(var, ".".join(key))
        if za.get("compressor"):
            raw = numcodecs.get_codec(za["compressor"]).decode(raw)
        arr = np.frombuffer(raw, dtype=np.dtype(za["dtype"])).reshape(chunks)
        return np.asarray(arr[tuple(sel)], dtype=np.float32)


def run(system: str, out_dir: Path, workers: int) -> None:
    cfg = SYSTEMS[system]
    store = Store(cfg["path"])
    idx = dl.domain_indices(store.lat, store.lon)
    lat_ok = all(abs(abs(store.lat[1] - store.lat[0]) - 0.25) < 1e-6 for _ in [0])
    if not lat_ok:
        raise SystemExit(f"{system}: unexpected latitude step")
    out_dir.mkdir(parents=True, exist_ok=True)
    member = "member_dim" in cfg
    inits = init_dates(member=member)
    if cfg["kind"] == "truth":
        jobs = sorted({(d + dt.timedelta(hours=lead), None) for d in init_dates() for lead in LEADS})
    else:
        jobs = [(d, lead) for d in inits for lead in LEADS]
    print(f"{system}: {len(jobs)} chunks x {len(dl.VARS)} vars -> {out_dir}", flush=True)

    def one(when: dt.datetime, lead: int | None) -> str:
        stamp = f"{when:%Y%m%d%H}" + ("" if lead is None else f"_f{lead:03d}")
        target = out_dir / f"{system}_{stamp}.npz"
        if target.exists():
            return f"cached {target.name}"
        try:
            fields = [store.field(v, when, lead, cfg.get("member_dim"), cfg.get("member_index", 0))
                      for v in dl.VARS]
            payload = {name: np.stack([f[li[:, None], lo[None, :]] for f in fields]).astype(np.float32)
                       for name, (li, lo) in idx.items()}
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
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} ok={ok} fail={fail}", flush=True)
    print(f"SUMMARY {system}: ok={ok} fail={fail}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/b33c_wb2"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(args.system, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
