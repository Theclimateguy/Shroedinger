#!/usr/bin/env python3
"""Phase 34: what places the activity of adjacent scales together (bands 200-1600 km).

Frozen protocol: docs/PROTOCOL_PHASE34_COLOCATION_GEOGRAPHY.md (2026-10-15).

Stages:
    windows --source era5 --year 2023 --season JFM    per-window covariance matrices and profile
    windows --source ifshr                             free-running IFS-HR 2014 (JFM and JAS)
    null --year 2024 --season JFM                      mirror-surrogate null for the compensation test
    tests                                              hypotheses, verdict, maps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_HERE.parent)

PROTOCOL = Path("docs/PROTOCOL_PHASE34_COLOCATION_GEOGRAPHY.md")
OUT = Path("clean_experiments/results/experiment_B34_colocation_geography")
GLOB = Path("data/b20global")
IFSHR = Path("data/b13hrmip/ECMWF-IFS-HR")
ORO = Path("data/b32/era5_orography.nc")
SDOR = Path("data/xp_physics/era5_static_subgrid_orography.nc")
PH31 = Path("clean_experiments/results/experiment_B31_boundary_template/tile_fields.npz")

SEED = 20261015
EARTH_KM = 6371.0
EARTH_M = 6.371e6
Z_850_M = 1457.0
ELLS = (200.0, 400.0, 800.0, 1600.0)
EPS = 1e-12
LAT_COMPUTE = 70.0
LAT_STAT = 60.0
BLOCK_DAYS = 5
CHUNK = 24
MONTHS = {"JFM": ("01", "02", "03"), "JAS": ("07", "08", "09")}
PAIRS = {"B1|B2": (0, 1), "B2|B3": (1, 2), "B1|B3": (0, 2)}
MARGIN_KM = 1100.0
NULL_STRIDE = 4
N_PERM = 999


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def windows() -> list[dict]:
    out = []
    belts = [(-60, -40, 40), (-40, -20, 30), (-20, 0, 30), (0, 20, 30), (20, 40, 30), (40, 60, 40)]
    for b, (la0, la1, width) in enumerate(belts):
        for lo0 in range(0, 360, width):
            out.append({"wid": f"W{len(out):02d}", "belt": b, "lat0": la0, "lat1": la1,
                        "lon0": float(lo0), "lon1": float(lo0 + width),
                        "lat_c": 0.5 * (la0 + la1), "lon_c": lo0 + 0.5 * width})
    assert len(out) == 66
    return out


def window_index(lat: np.ndarray, lon: np.ndarray, w: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = np.where((lat >= w["lat0"]) & (lat < w["lat1"]))[0]
    if w["lat1"] == LAT_STAT:
        rows = np.where((lat >= w["lat0"]) & (lat <= w["lat1"]))[0]
    cols = np.where((lon >= w["lon0"]) & (lon < w["lon1"]))[0]
    return rows, cols


# ---------------------------------------------------------------------------
# fields on the sphere (periodic in longitude)
# ---------------------------------------------------------------------------

def vorticity_sphere(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """V12 relative vorticity, centred differences, periodic in longitude. lat ascending."""
    phi = np.deg2rad(lat)[None, :, None]
    dlam = np.deg2rad(float(lon[1] - lon[0]))
    dv_dlam = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2.0 * dlam)
    du_dphi = np.gradient(u, np.deg2rad(lat), axis=-2)
    zeta = dv_dlam / (EARTH_M * np.cos(phi)) - du_dphi / EARTH_M + u * np.tan(phi) / EARTH_M
    return zeta.astype(np.float32)


class Smoother:
    """Normalised Gaussian convolution, km-based, periodic in longitude."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, valid: np.ndarray):
        self.dy = EARTH_KM * np.deg2rad(float(lat[1] - lat[0]))
        self.dx = EARTH_KM * np.cos(np.deg2rad(lat)) * np.deg2rad(float(lon[1] - lon[0]))
        self.w = valid.astype(np.float32)
        self.den = {s: self._g(self.w[None], s)[0] for s in ELLS}

    def _g(self, f: np.ndarray, ell: float) -> np.ndarray:
        sig = ell / np.sqrt(12.0)
        out = gaussian_filter1d(f, sigma=sig / self.dy, axis=-2, mode="nearest")
        keys = np.round(sig / self.dx, 2)
        for k in np.unique(keys):
            rows = np.where(keys == k)[0]
            out[:, rows, :] = gaussian_filter1d(out[:, rows, :], sigma=float(k), axis=-1, mode="wrap")
        return out

    def bar(self, f: np.ndarray, ell: float) -> np.ndarray:
        num = self._g(np.nan_to_num(f) * self.w[None], ell)
        den = self.den[ell]
        return np.where(den[None] > 1e-3, num / np.maximum(den[None], 1e-6), np.nan).astype(np.float32)


def envelopes(om: np.ndarray, sm: Smoother) -> list[np.ndarray]:
    """Log-envelopes of B1 = 200-400, B2 = 400-800, B3 = 800-1600 km (V12 convention)."""
    w = np.where(sm.w[None] > 0, om, np.nan)
    bars = [sm.bar(w, ell) for ell in ELLS]
    envs = []
    for k in range(3):
        d = np.abs(np.nan_to_num(bars[k] - bars[k + 1]))
        e = np.log(sm.bar(d, ELLS[k + 1]) + EPS)
        envs.append(np.where(sm.w[None] > 0, e, np.nan).astype(np.float32))
    return envs


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_era5(year: int, season: str):
    import xarray as xr
    parts = [xr.open_dataset(GLOB / f"era5_wind850_global_{year}{m}.nc") for m in MONTHS[season]]
    ds = xr.concat(parts, dim="valid_time")
    lat = np.asarray(ds["latitude"].values, float)
    lon = np.asarray(ds["longitude"].values, float)
    u = np.asarray(ds["u"].squeeze().values, np.float32)
    v = np.asarray(ds["v"].squeeze().values, np.float32)
    t = ds["valid_time"].values.astype("datetime64[D]")
    for p in parts:
        p.close()
    if lat[0] > lat[-1]:
        lat, u, v = lat[::-1], u[:, ::-1], v[:, ::-1]
    keep = np.abs(lat) <= LAT_COMPUTE + 1e-9
    lat, u, v = lat[keep], np.ascontiguousarray(u[:, keep]), np.ascontiguousarray(v[:, keep])
    import experiment_B32_vorticity_mask as vm
    below = vm.below_surface_mask(lat, lon)
    return lat, lon, u, v, t, ~below


def load_ifshr(season: str):
    import xarray as xr
    us, vs, ts = [], [], []
    for m in MONTHS[season]:
        fu = sorted(IFSHR.glob(f"ua_6hrPlevPt_*_2014{m}*.nc"))[0]
        fv = sorted(IFSHR.glob(f"va_6hrPlevPt_*_2014{m}*.nc"))[0]
        du, dv = xr.open_dataset(fu), xr.open_dataset(fv)
        k = int(np.argmin(np.abs(du["plev"].values - 85000.0)))
        us.append(np.asarray(du["ua"].isel(plev=k).values, np.float32))
        vs.append(np.asarray(dv["va"].isel(plev=k).values, np.float32))
        ts.append(du["time"].values)
        lat = np.asarray(du["lat"].values, float)
        lon = np.asarray(du["lon"].values, float)
        du.close(); dv.close()
    u, v = np.concatenate(us), np.concatenate(vs)
    t = np.concatenate(ts)
    try:
        t = t.astype("datetime64[D]")
    except Exception:  # cftime objects
        t = np.array([np.datetime64(f"{x.year:04d}-{x.month:02d}-{x.day:02d}") for x in t])
    if lat[0] > lat[-1]:
        lat, u, v = lat[::-1], u[:, ::-1], v[:, ::-1]
    keep = np.abs(lat) <= LAT_COMPUTE + 1e-9
    lat, u, v = lat[keep], u[:, keep], v[:, keep]
    valid = np.all(np.isfinite(u) & np.isfinite(v) & (np.abs(u) < 1e10), axis=0)
    if valid.all():                     # Deviation D1: IFS-HR extrapolates below ground -> ERA5 mask
        import experiment_B32_vorticity_mask as vm
        valid = ~vm.below_surface_mask(lat, lon)
    u = np.where(valid[None], u, 0.0).astype(np.float32)
    v = np.where(valid[None], v, 0.0).astype(np.float32)
    return lat, lon, np.ascontiguousarray(u), np.ascontiguousarray(v), t, valid


# ---------------------------------------------------------------------------
# stage: windows
# ---------------------------------------------------------------------------

def wcov(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted covariance along the last axis; a, b (..., n), w (n,)."""
    sw = w.sum()
    ma = (a * w).sum(-1, keepdims=True) / sw
    mb = (b * w).sum(-1, keepdims=True) / sw
    return ((a - ma) * (b - mb) * w).sum(-1) / sw


def stage_windows(source: str, year: int, season: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{source}_{year}_{season}"
    target = OUT / f"windows_{tag}.json"
    if target.exists():
        print("cached", target.name, flush=True)
        return
    if source == "era5":
        lat, lon, u, v, t, valid = load_era5(year, season)
    else:
        lat, lon, u, v, t, valid = load_ifshr(season)
    sm = Smoother(lat, lon, valid)
    block = (((t - t[0]).astype(int) // BLOCK_DAYS) % 2).astype(int)
    W = windows()
    idx = [window_index(lat, lon, w) for w in W]
    coslat = np.cos(np.deg2rad(lat)).astype(np.float64)
    ny, nx = len(lat), len(lon)
    sums = np.zeros((2, 3, ny, nx)); cnts = np.zeros((2, 3, ny, nx))
    tot = np.zeros((len(W), 3, 3)); nstep = np.zeros(len(W))
    p4 = [[] for _ in W]; p5 = [[] for _ in W]
    nt = u.shape[0]
    for c0 in range(0, nt, CHUNK):
        sl = slice(c0, min(nt, c0 + CHUNK))
        om = vorticity_sphere(u[sl], v[sl], lat, lon)
        E = envelopes(om, sm)
        for b in (0, 1):
            m = block[sl] == b
            if m.any():
                for k in range(3):
                    x = E[k][m]
                    sums[b, k] += np.nansum(x, axis=0)
                    cnts[b, k] += np.isfinite(x).sum(axis=0)
        for wi, (rows, cols) in enumerate(idx):
            sub = np.stack([e[:, rows][:, :, cols] for e in E])            # (3, nt_c, nr, nc)
            fin = np.all(np.isfinite(sub), axis=(0, 1))
            wt = (coslat[rows][:, None] * np.ones(len(cols))[None, :]) * fin
            sel = wt > 0
            x = sub[:, :, sel].astype(np.float64)                           # (3, nt_c, n)
            ww = wt[sel]
            for i in range(3):
                for j in range(i, 3):
                    cij = wcov(x[i], x[j], ww)                              # (nt_c,)
                    tot[wi, i, j] += cij.sum(); tot[wi, j, i] = tot[wi, i, j]
            c11, c22, c33 = (wcov(x[i], x[i], ww) for i in range(3))
            p4[wi] += list(wcov(x[0], x[1], ww) / np.sqrt(c11 * c22))
            p5[wi] += list(wcov(x[1], x[2], ww) / np.sqrt(c22 * c33))
            nstep[wi] += x.shape[1]
        print(f"[{tag}] steps {sl.stop}/{nt}", flush=True)
    means = sums / np.maximum(cnts, 1)
    means[cnts == 0] = np.nan
    res = []
    for wi, (w, (rows, cols)) in enumerate(zip(W, idx)):
        mA = means[0][:, rows][:, :, cols]; mB = means[1][:, rows][:, :, cols]
        fin = np.all(np.isfinite(mA), axis=0) & np.all(np.isfinite(mB), axis=0)
        wt = (coslat[rows][:, None] * np.ones(len(cols))[None, :]) * fin
        sel = wt > 0
        a = mA[:, sel]; b = mB[:, sel]; ww = wt[sel]
        tpl = np.array([[0.5 * (wcov(a[i], b[j], ww) + wcov(b[i], a[j], ww)) for j in range(3)] for i in range(3)])
        vf = float((valid[rows][:, cols] * coslat[rows][:, None]).sum() / (coslat[rows].sum() * len(cols)))
        res.append({**w, "tot": (tot[wi] / max(nstep[wi], 1)).tolist(), "tpl": tpl.tolist(),
                    "P4": float(np.median(p4[wi])), "P5": float(np.median(p5[wi])),
                    "n_steps": int(nstep[wi]), "valid_frac": vf})
    target.write_text(json.dumps({"source": source, "year": year, "season": season,
                                  "n_steps": int(nt), "windows": res}), encoding="utf-8")
    print(f"B34_WINDOWS_{tag}_DONE", flush=True)


# ---------------------------------------------------------------------------
# stage: null (azimuthal-equidistant plane, mirror surrogates)
# ---------------------------------------------------------------------------

_G: dict = {}


def aeqd_grid(w: dict, step_km: float):
    """Azimuthal-equidistant grid centred on the window; returns lat, lon of each node and the window mask."""
    half_y = (w["lat1"] - w["lat0"]) / 2 * np.pi / 180 * EARTH_KM + MARGIN_KM
    half_x = (w["lon1"] - w["lon0"]) / 2 * np.pi / 180 * EARTH_KM * np.cos(np.deg2rad(w["lat_c"])) + MARGIN_KM
    ys = np.arange(-half_y, half_y + 1e-9, step_km)
    xs = np.arange(-half_x, half_x + 1e-9, step_km)
    X, Y = np.meshgrid(xs, ys)
    rho = np.hypot(X, Y) / EARTH_KM
    az = np.arctan2(X, Y)
    p0, l0 = np.deg2rad(w["lat_c"]), np.deg2rad(w["lon_c"])
    sp = np.sin(p0) * np.cos(rho) + np.cos(p0) * np.sin(rho) * np.cos(az)
    plat = np.arcsin(np.clip(sp, -1, 1))
    plon = l0 + np.arctan2(np.sin(az) * np.sin(rho) * np.cos(p0), np.cos(rho) - np.sin(p0) * sp)
    lat_d, lon_d = np.rad2deg(plat), np.mod(np.rad2deg(plon), 360.0)
    inside = (lat_d >= w["lat0"]) & (lat_d < w["lat1"]) & (lon_d >= w["lon0"]) & (lon_d < w["lon1"])
    return lat_d, lon_d, inside


def _null_init(om, valid, lat, lon):
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(om=om, valid=valid, lat=lat, lon=lon)


def _plane_stats(f: np.ndarray, vmask: np.ndarray, inside: np.ndarray, step_km: float) -> np.ndarray:
    """Time-mean weighted covariance matrix of the three log-envelopes on a uniform km plane."""
    wv = vmask.astype(np.float32)

    def g(x, ell):
        return gaussian_filter(x, sigma=(0, ell / np.sqrt(12.0) / step_km, ell / np.sqrt(12.0) / step_km), mode="nearest")

    def bar(x, ell):
        num = g(np.nan_to_num(x) * wv[None], ell)
        den = g(wv[None], ell)
        return np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan)
    bars = [bar(f, ell) for ell in ELLS]
    E = [np.log(bar(np.abs(np.nan_to_num(bars[k] - bars[k + 1])), ELLS[k + 1]) + EPS) for k in range(3)]
    sel = inside & vmask & np.all([np.all(np.isfinite(e), axis=0) for e in E], axis=0)
    x = np.stack([e[:, sel] for e in E]).astype(np.float64)
    ww = np.ones(x.shape[-1])
    return np.array([[wcov(x[i], x[j], ww).mean() for j in range(3)] for i in range(3)])


def _null_worker(args):
    wi, w, seed = args
    om, valid, lat, lon = (_G[k] for k in ("om", "valid", "lat", "lon"))
    step = EARTH_KM * np.deg2rad(float(lat[1] - lat[0]))
    plat, plon, inside = aeqd_grid(w, step)
    fy = np.interp(plat, lat, np.arange(len(lat)))
    fx = plon / float(lon[1] - lon[0])
    vm = map_coordinates(valid.astype(np.float32), [fy, fx], order=0, mode="grid-wrap") > 0.5
    rng = np.random.default_rng(seed)
    real, sur = [], []
    for t in range(0, om.shape[0], NULL_STRIDE):
        f = map_coordinates(np.nan_to_num(om[t]), [fy, fx], order=1, mode="grid-wrap").astype(np.float64)
        f = np.where(vm, f - f[vm].mean(), 0.0)
        real.append(f)
        ext = np.block([[f, f[:, ::-1]], [f[::-1, :], f[::-1, ::-1]]])
        F = np.fft.rfft2(ext)
        ph = np.exp(1j * np.angle(np.fft.rfft2(rng.standard_normal(ext.shape))))
        ph[0, 0] = 1.0
        s = np.fft.irfft2(F * ph, s=ext.shape)[: f.shape[0], : f.shape[1]]
        sur.append(np.where(vm, s, np.nan))
    real = np.where(vm[None], np.stack(real), np.nan)
    Cr = _plane_stats(real, vm, inside, step)
    Cs = _plane_stats(np.stack(sur), vm, inside, step)
    r = lambda C, i, j: float(C[i, j] / np.sqrt(C[i, i] * C[j, j]))  # noqa: E731
    return {"wid": w["wid"], "r_tot_reg": {p: r(Cr, i, j) for p, (i, j) in PAIRS.items()},
            "r_tot_null": {p: r(Cs, i, j) for p, (i, j) in PAIRS.items()}, "n_steps": len(real)}


def stage_null(year: int, season: str, workers: int) -> None:
    target = OUT / f"null_era5_{year}_{season}.json"
    if target.exists():
        print("cached", target.name, flush=True)
        return
    lat, lon, u, v, t, valid = load_era5(year, season)
    om = np.concatenate([vorticity_sphere(u[i:i + CHUNK], v[i:i + CHUNK], lat, lon)
                         for i in range(0, u.shape[0], CHUNK)])
    om = np.where(valid[None], om, np.nan).astype(np.float32)
    del u, v
    W = windows()
    seeds = np.random.SeedSequence(SEED + year + (0 if season == "JFM" else 7)).generate_state(len(W))
    with mp.get_context("fork").Pool(workers, _null_init, (om, valid, lat, lon)) as pool:
        res = pool.map(_null_worker, [(i, w, int(s)) for i, (w, s) in enumerate(zip(W, seeds))])
    target.write_text(json.dumps({"year": year, "season": season, "windows": res}), encoding="utf-8")
    print(f"B34_NULL_{year}_{season}_DONE", flush=True)


# ---------------------------------------------------------------------------
# stage: tests
# ---------------------------------------------------------------------------

def _corr(C, i, j):
    return C[..., i, j] / np.sqrt(np.maximum(C[..., i, i], 1e-30) * np.maximum(C[..., j, j], 1e-30))


def load_year(source: str, year: int) -> dict:
    per = []
    for s in MONTHS:
        d = json.loads((OUT / f"windows_{source}_{year}_{s}.json").read_text())
        per.append({w["wid"]: w for w in d["windows"]})
    W = windows()
    tot = np.array([[p[w["wid"]]["tot"] for w in W] for p in per])      # (2, 66, 3, 3)
    tpl = np.array([[p[w["wid"]]["tpl"] for w in W] for p in per])
    out = {"s_season": np.array([np.mean([tp[:, i, i] / to[:, i, i] for i in range(3)], axis=0)
                                 for to, tp in zip(tot, tpl)])}
    out["s"] = out["s_season"].mean(0)
    for pn, (i, j) in PAIRS.items():
        out[f"r_tot_{pn}"] = np.mean([_corr(to, i, j) for to in tot], axis=0)
        out[f"r_mov_{pn}"] = np.mean([_corr(to - tp, i, j) for to, tp in zip(tot, tpl)], axis=0)
        rt = []
        for to, tp in zip(tot, tpl):
            ok = (tp[:, i, i] > 0.02 * to[:, i, i]) & (tp[:, j, j] > 0.02 * to[:, j, j])
            rt.append(np.where(ok, _corr(tp, i, j), np.nan))
        out[f"r_tpl_{pn}"] = np.nanmean(rt, axis=0)
    out["P4"] = np.mean([[p[w["wid"]]["P4"] for w in W] for p in per], axis=0)
    out["P5"] = np.mean([[p[w["wid"]]["P5"] for w in W] for p in per], axis=0)
    out["P45"] = 0.5 * (out["P4"] + out["P5"])
    return out


def window_covariates() -> dict:
    import xarray as xr
    from global_land_mask import globe
    W = windows()
    land, sdor = [], []
    d = xr.open_dataset(SDOR)
    sd = d["sdor"].squeeze().values; la = d["latitude"].values; lo = d["longitude"].values
    d.close()
    for w in W:
        gl = np.arange(w["lat0"] + 0.125, w["lat1"], 0.25)
        go = np.arange(w["lon0"] + 0.125, w["lon1"], 0.25)
        LA, LO = np.meshgrid(gl, np.where(go > 180, go - 360, go), indexing="ij")
        wt = np.cos(np.deg2rad(LA))
        land.append(float((globe.is_land(LA, LO) * wt).sum() / wt.sum()))
        a = (la >= w["lat0"]) & (la <= w["lat1"]); b = (lo >= w["lon0"]) & (lo < w["lon1"])
        sw = np.cos(np.deg2rad(la[a]))[:, None] * np.ones(b.sum())[None]
        sdor.append(float(np.nansum(sd[np.ix_(a, b)] * sw) / sw.sum()))
    return {"land": np.array(land), "sdor": np.array(sdor)}


def belt_perm(x: np.ndarray, belts: np.ndarray, rng, cyclic: bool) -> np.ndarray:
    y = x.copy()
    for b in np.unique(belts):
        m = np.where(belts == b)[0]
        y[m] = np.roll(x[m], rng.integers(1, len(m))) if cyclic else x[m][rng.permutation(len(m))]
    return y


def ward_labels(Z: np.ndarray, k: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    return fcluster(linkage(Z, method="ward"), t=k, criterion="maxclust")


def stage_tests() -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score as ari
    rng = np.random.default_rng(SEED)
    W = windows()
    belts = np.array([w["belt"] for w in W])
    cov = window_covariates()
    Y = {y: load_year("era5", y) for y in (2022, 2023, 2024) if (OUT / f"windows_era5_{y}_JAS.json").exists()}
    M = load_year("ifshr", 2014) if (OUT / "windows_ifshr_2014_JAS.json").exists() else None
    sp = lambda a, b: float(spearmanr(a, b).statistic)  # noqa: E731
    res = {"protocol": str(PROTOCOL), "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
           "n_windows": len(W), "n_land": int((cov["land"] >= 0.5).sum()), "n_ocean": int((cov["land"] <= 0.1).sum())}
    s23, s24 = Y[2023]["s"], Y[2024]["s"]
    s_mean = 0.5 * (s23 + s24)

    # H34-1
    rho = sp(s23, s24)
    null = np.array([sp(s23, belt_perm(s24, belts, rng, cyclic=True)) for _ in range(N_PERM)])
    p = float((np.sum(null >= rho) + 1) / (N_PERM + 1))
    h1 = {"rho_2023_2024": rho, "p_belt_shift": p, "median_s": {str(y): float(np.median(Y[y]["s"])) for y in Y},
          "pass_two_years": bool(rho >= 0.70 and p < 0.01)}
    if 2022 in Y:
        h1["rho_2022_vs_mean"] = sp(Y[2022]["s"], s_mean)
        h1["pass_fresh"] = bool(h1["rho_2022_vs_mean"] >= 0.70)
    h1["pass"] = bool(h1["pass_two_years"] and h1.get("pass_fresh", False))
    h1["season_agreement"] = {str(y): sp(Y[y]["s_season"][0], Y[y]["s_season"][1]) for y in Y}
    res["H34_1"] = h1

    # H34-2
    landw, oceanw = cov["land"] >= 0.5, cov["land"] <= 0.1
    ratio = float(np.median(s_mean[landw]) / np.median(s_mean[oceanw]))
    rs = sp(s_mean, cov["sdor"])
    res["H34_2"] = {"rho_s_sdor": rs, "median_s_land": float(np.median(s_mean[landw])),
                    "median_s_ocean": float(np.median(s_mean[oceanw])), "ratio": ratio,
                    "rho_s_landfrac": sp(s_mean, cov["land"]), "rho_s_abslat": sp(s_mean, np.abs([w["lat_c"] for w in W])),
                    "pass": bool(rs >= 0.30 and ratio >= 1.5)}

    # H34-3
    rt, rm = Y[2024]["r_tot_B1|B3"], Y[2024]["r_mov_B1|B3"]
    a = abs(sp(rt, s23)) <= 0.20
    b = sp(rm, s23) <= -0.30
    h3 = {"rho_rtot_s": sp(rt, s23), "rho_rmov_s": sp(rm, s23), "a": bool(a), "b": bool(b)}
    nulls = [OUT / f"null_era5_2024_{s}.json" for s in MONTHS]
    if all(f.exists() for f in nulls):
        dd = [{x["wid"]: x for x in json.loads(f.read_text())["windows"]} for f in nulls]
        reg = np.mean([[d[w["wid"]]["r_tot_reg"]["B1|B3"] for w in W] for d in dd], axis=0)
        nul = np.mean([[d[w["wid"]]["r_tot_null"]["B1|B3"] for w in W] for d in dd], axis=0)
        a_tot = reg - nul
        c = bool(np.median(a_tot) >= 0.05 and abs(sp(a_tot, s23)) <= 0.20)
        h3.update({"median_r_tot_reg": float(np.median(reg)), "median_r_tot_null": float(np.median(nul)),
                   "median_a_tot": float(np.median(a_tot)), "rho_atot_s": sp(a_tot, s23),
                   "rho_rtot_global_vs_plane": sp(rt, reg), "c": c})
        h3["pass"] = bool(a and b and c)
    else:
        h3["pass"] = None
    res["H34_3"] = h3

    # H34-4
    med = {pn: float(np.nanmedian(Y[2024][f"r_tpl_{pn}"])) for pn in PAIRS}
    res["H34_4"] = {"median_r_tpl": med, "n_defined": {pn: int(np.isfinite(Y[2024][f"r_tpl_{pn}"]).sum()) for pn in PAIRS},
                    "pass": bool(med["B1|B2"] >= 0.70 and med["B2|B3"] >= 0.70)}

    # H34-5
    if M is not None:
        p45_mean = 0.5 * (Y[2023]["P45"] + Y[2024]["P45"])
        r_s, r_p = sp(M["s"], s_mean), sp(M["P45"], p45_mean)
        res["H34_5"] = {"rho_s_model": r_s, "rho_P45_model": r_p,
                        "ceiling_rho_P45_2023_2024": sp(Y[2023]["P45"], Y[2024]["P45"]),
                        "pass": bool(r_s >= 0.60 and r_p >= 0.60)}

    # H34-6
    def feats(y):
        return np.column_stack([y["P4"], y["P5"], y["s"], y["r_mov_B1|B3"]])

    def zs(X, mu, sd):
        return (X - mu) / sd
    h6 = {"k": {}}
    for k in (3, 4, 5):
        X23, X24 = feats(Y[2023]), feats(Y[2024])
        l23 = ward_labels(zs(X23, X23.mean(0), X23.std(0)), k)
        l24 = ward_labels(zs(X24, X24.mean(0), X24.std(0)), k)
        a0 = float(ari(l23, l24))
        nul = np.array([ari(l23, belt_perm(l24, belts, rng, cyclic=False)) for _ in range(N_PERM)])
        rec = {"ari_2023_2024": a0, "p_belt_perm": float((np.sum(nul >= a0) + 1) / (N_PERM + 1))}
        Xp = 0.5 * (X23 + X24)
        mu, sd = Xp.mean(0), Xp.std(0)
        lp = ward_labels(zs(Xp, mu, sd), k)
        if 2022 in Y:
            cen = np.array([zs(Xp, mu, sd)[lp == c].mean(0) for c in np.unique(lp)])
            X22 = zs(feats(Y[2022]), mu, sd)
            l22 = np.unique(lp)[np.argmin(((X22[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)]
            a2 = float(ari(lp, l22))
            nul2 = np.array([ari(lp, belt_perm(l22, belts, rng, cyclic=False)) for _ in range(N_PERM)])
            rec.update({"ari_2022_vs_pooled": a2, "p_2022_belt_perm": float((np.sum(nul2 >= a2) + 1) / (N_PERM + 1))})
        rec["labels_pooled"] = lp.tolist()
        rec["cluster_composition"] = {int(c): {"n": int((lp == c).sum()),
                                               "land_frac_mean": float(cov["land"][lp == c].mean()),
                                               "abs_lat_mean": float(np.mean(np.abs([w["lat_c"] for w, l in zip(W, lp) if l == c]))),
                                               "P45": float((0.5 * (Y[2023]["P45"] + Y[2024]["P45"]))[lp == c].mean()),
                                               "s": float(s_mean[lp == c].mean())} for c in np.unique(lp)}
        h6["k"][str(k)] = rec
    r4 = h6["k"]["4"]
    h6["pass_two_years"] = bool(r4["ari_2023_2024"] >= 0.40 and r4["p_belt_perm"] < 0.05)
    h6["pass_fresh"] = bool(r4.get("ari_2022_vs_pooled", -1) >= 0.40 and r4.get("p_2022_belt_perm", 1) < 0.05)
    h6["pass"] = bool(h6["pass_two_years"] and h6["pass_fresh"])
    res["H34_6"] = h6

    # descriptive: Phase-31 fine-band share inside the windows
    if PH31.exists():
        z = np.load(PH31)
        fine = []
        for w in W:
            m = (z["lat"] >= w["lat0"]) & (z["lat"] < w["lat1"]) & (np.mod(z["lon"], 360) >= w["lon0"]) & (np.mod(z["lon"], 360) < w["lon1"])
            fine.append(float(np.mean(0.5 * (z["share_2023"][m] + z["share_2024"][m]))) if m.any() else np.nan)
        fine = np.array(fine); ok = np.isfinite(fine)
        res["descr_phase31_fine_share"] = {"rho_with_s": sp(fine[ok], s_mean[ok]), "n": int(ok.sum())}

    v = ("NEGATIVE" if not h1["pass"] else
         "REGIONALISATION_CONFIRMED" if h6["pass"] else "TEMPLATE_INVARIANT_ONLY")
    res["VERDICT"] = v
    res["template_word"] = "boundary template" if res["H34_2"]["pass"] else "stationary template (surface + climatological circulation)"
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(OUT / "window_fields.npz", wid=np.array([w["wid"] for w in W]), lat_c=[w["lat_c"] for w in W],
             lon_c=[w["lon_c"] for w in W], land=cov["land"], sdor=cov["sdor"],
             **{f"{k}_{y}": Y[y][k] for y in Y for k in ("s", "P4", "P5", "P45", "r_tot_B1|B3", "r_mov_B1|B3")},
             **({f"{k}_ifshr": M[k] for k in ("s", "P45")} if M is not None else {}))
    print(json.dumps({k: res[k] for k in ("VERDICT", "template_word")}, indent=1), flush=True)
    return res


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["windows", "null", "tests"])
    ap.add_argument("--source", choices=["era5", "ifshr"], default="era5")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--season", choices=list(MONTHS), default="JFM")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.stage == "windows":
        if a.source == "ifshr":
            for s in MONTHS:
                stage_windows("ifshr", 2014, s)
        else:
            stage_windows("era5", a.year, a.season)
    elif a.stage == "null":
        stage_null(a.year, a.season, a.workers)
    else:
        stage_tests()


if __name__ == "__main__":
    main()
