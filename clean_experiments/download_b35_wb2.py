#!/usr/bin/env python3
"""Phase-35 data: ensemble members, lead curve and a second truth (WeatherBench 2, 2022).

Protocol: docs/PROTOCOL_PHASE35_COLOCATION_PREDICTABILITY.md (frozen 2026-10-15).

Jobs:
    --job members --system ifs_ens|fgn      all members, 24 inits (6 per window), leads 24/120 (fgn: 120)
    --job leads --system graphcast|ifs_ens_mean|fgn_mean|hres   leads 48, 72, 168, 240 (fgn_mean: all six)
    --job truth --system era5               ERA5 at the new valid times of the lead curve
    --job truth --system hres_t0            IFS analysis at the Arm-C valid times (75 inits, 24/120 h)
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
import download_b33c_wb2_2022 as c  # noqa: E402

SYSTEMS = {
    **c.SYSTEMS,
    "ifs_ens": {"path": "ifs_ens/2018-2022-1440x721.zarr", "member_dim": "number"},
    "fgn": {"path": "fgn/2022-1440x721.zarr", "member_dim": "sample"},
    "fgn_mean": {"path": "fgn/2022-1440x721_mean.zarr"},
    "hres_t0": {"path": "hres_t0/2016-2022-6h-1440x721.zarr", "kind": "truth"},
}
CURVE_LEADS = (48, 72, 168, 240)
ALL_LEADS = (24, 48, 72, 120, 168, 240)


def member_inits() -> list[dt.datetime]:
    return c.init_dates(member=True)          # first 6 per window, 24 inits


class Store(c.Store):
    def field_members(self, var: str, when: dt.datetime, lead_h: int, member_dim: str) -> np.ndarray:
        """(M, nlat, nlon) at 850 hPa: the whole member axis of one chunk."""
        za = self.meta[var + "/.zarray"]
        dims, chunks, shape = self.dims(var), za["chunks"], za["shape"]
        pos = {"time": self.time_index(when)}
        hits = np.where(self.lead.astype(float) == float(lead_h))[0]
        pos["prediction_timedelta"] = int(hits[0])
        lev = int(np.where(self.level == dl.LEVEL_HPA)[0][0])
        key, sel = [], []
        for d, ch, n in zip(dims, chunks, shape):
            if d in (self.lat_name, self.lon_name):
                key.append("0"); sel.append(slice(None))
            elif d == member_dim:
                if ch != n:
                    raise RuntimeError(f"member axis split over chunks in {self.url}")
                key.append("0"); sel.append(slice(None))
            elif d == "level":
                key.append(str(lev // ch)); sel.append(lev % ch)
            else:
                idx = pos.get(d, 0)
                key.append(str(idx // ch)); sel.append(idx % ch)
        raw = self._raw(var, ".".join(key))
        if za.get("compressor"):
            raw = numcodecs.get_codec(za["compressor"]).decode(raw)
        arr = np.frombuffer(raw, dtype=np.dtype(za["dtype"])).reshape(chunks)
        return np.asarray(arr[tuple(sel)], dtype=np.float32)


def jobs_for(job: str, system: str) -> list[tuple[dt.datetime, int | None]]:
    if job == "members":
        leads = (120,) if system == "fgn" else (24, 120)
        return [(d, l) for d in member_inits() for l in leads]
    if job == "leads":
        leads = ALL_LEADS if system == "fgn_mean" else CURVE_LEADS
        return [(d, l) for d in member_inits() for l in leads]
    if system == "era5":
        return sorted({(d + dt.timedelta(hours=l), None) for d in member_inits() for l in CURVE_LEADS})
    return sorted({(d + dt.timedelta(hours=l), None) for d in c.init_dates() for l in (24, 120)})


def run(job: str, system: str, out_dir: Path, workers: int) -> None:
    cfg = SYSTEMS[system]
    store = Store(cfg["path"])
    idx = dl.domain_indices(store.lat, store.lon)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(job, system)
    print(f"{job}/{system}: {len(jobs)} chunks x {len(dl.VARS)} vars -> {out_dir}", flush=True)

    def one(when: dt.datetime, lead: int | None) -> str:
        stamp = f"{when:%Y%m%d%H}" + ("" if lead is None else f"_f{lead:03d}")
        name = f"{system}_members" if job == "members" else system
        target = out_dir / f"{name}_{stamp}.npz"
        if target.exists():
            return f"cached {target.name}"
        try:
            if job == "members":
                fields = [store.field_members(v, when, lead, cfg["member_dim"]) for v in dl.VARS]
                payload = {n: np.stack([f[:, li[:, None], lo[None, :]] for f in fields], axis=1).astype(np.float32)
                           for n, (li, lo) in idx.items()}                 # (M, 2, ny, nx)
            else:
                fields = [store.field(v, when, lead) for v in dl.VARS]
                payload = {n: np.stack([f[li[:, None], lo[None, :]] for f in fields]).astype(np.float32)
                           for n, (li, lo) in idx.items()}
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
    print(f"SUMMARY {job}/{system}: ok={ok} fail={fail}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", choices=["members", "leads", "truth"], required=True)
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/b35_wb2"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.job, a.system, a.out_dir, a.workers)


if __name__ == "__main__":
    main()
