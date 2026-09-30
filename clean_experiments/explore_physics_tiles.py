#!/usr/bin/env python3
"""EXPLORATORY (not frozen) physics probes of anchored fine-P.

Register of hypotheses, written before computation:
docs/EXPLORATION_PHYSICS_2026-09-29.md

Reuses the frozen Phase-20 Arm-B tile machinery (same tile grid, band
ladder, envelope estimator, phase surrogates) and adds:
  - per-UTC-hour bookkeeping of the real and surrogate coupling (T1);
  - transient variant: season x UTC-hour composite removed (T2);
  - divergence-field variant and vorticity-divergence envelope coupling (T3);
  - any pressure level of the on-disk free-running IFS-HR run (T4);
  - second/third-order features of the real field: rotational/divergent
    band variances, cross-validated stationary and diurnal variance
    fractions, cyclonic skewness, cyclonic/anticyclonic conditional
    coupling (T2, T3, T5).

Surrogate count is reduced (default 24, frozen value 99); the control is
reproduction of the committed Arm-B map.

Usage:
  python clean_experiments/explore_physics_tiles.py --source era5 --season JFM
  ... --transient 1 | --field div | --source model --level 500
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B18_equal_km_regions import envelope_rho_profile_ells  # noqa: E402
from experiment_B1_true_flux_baselines import (  # noqa: E402
    _grid_geometry,
    _interior_mask,
    phase_randomize,
)
from experiment_B20_armB_global_map import tile_grid  # noqa: E402
from experiment_B20_p_geography import ELLS_FINE, N_FINE, RUNMEAN_STEPS  # noqa: E402
from experiment_B2_scale_irreversibility import (  # noqa: E402
    compute_vorticity,
    gaussian_bar_grouped,
)

SEED_SUR = 20260811
EPS = 1e-12
OUT = _HERE / "results" / "explore_physics"
GLOB = Path("data/b20global")
MODEL_DIR = Path("data/b13hrmip/ECMWF-IFS-HR")
ERA5_MONTHS = {"JFM": ["202301", "202302", "202303"],
               "JAS": ["202307", "202308", "202309"]}
MODEL_MONTHS = {"JFM": ["201401", "201402", "201403"],
                "JAS": ["201407", "201408", "201409"]}
BLOCK_DAYS = 5                      # alternating blocks for cross-validated composites

_G: dict[str, object] = {}


def compute_divergence(u, v, lat, lon) -> np.ndarray:
    """delta = du/dx + dv/dy in 1/s, same discretisation as compute_vorticity."""
    dx_km, dy_km = _grid_geometry(lat, lon)
    lat_sign = 1.0 if lat[1] > lat[0] else -1.0
    uk = np.nan_to_num(u).astype(np.float32) / 1000.0
    vk = np.nan_to_num(v).astype(np.float32) / 1000.0
    dudx = np.gradient(uk, axis=2) / dx_km[None, :, None]
    dvdy = np.gradient(vk, axis=1) / dy_km * lat_sign
    return (dudx + dvdy).astype(np.float32)


def band_fields(w, dx_km, dy_km):
    """The four fine band-pass fields: <50, 50-100, 100-200, 200-400 km."""
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km)
            for ell in ELLS_FINE]
    out = [w - bars[0]]
    for i in range(N_FINE):
        out.append(bars[i] - bars[i + 1])
    return out, bars[-1]


def log_envelopes(bands, dx_km, dy_km):
    ells = [ELLS_FINE[0]] + list(ELLS_FINE[1:])
    return [np.log(gaussian_bar_grouped(np.abs(b), ell / np.sqrt(12.0),
                                        dx_km, dy_km) + EPS)
            for b, ell in zip(bands, ells)]


def _rank_rows(m):
    return np.argsort(np.argsort(m, axis=1), axis=1).astype(np.float32)


def _corr_rows(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    den = np.where(den < 1e-12, 1.0, den)
    return (a * b).sum(axis=1) / den


def same_band_coupling(env_a, env_b, mask):
    """Per-step spatial Spearman between same-band envelopes of two fields."""
    out = np.empty((env_a[0].shape[0], len(env_a)))
    for b in range(len(env_a)):
        out[:, b] = _corr_rows(_rank_rows(env_a[b][:, mask]),
                               _rank_rows(env_b[b][:, mask]))
    return out


def by_hour(x, hour_idx):
    """Median over time within each UTC-hour class; shape (4, ...)."""
    return np.stack([np.median(x[hour_idx == h], axis=0) for h in range(4)])


def cv_fractions(bands, mask, hour_idx, block):
    """Cross-validated stationary and diurnal variance fractions per band.

    Composites are formed on two interleaved halves (alternating 5-day
    blocks); the spatial covariance between the halves is an unbiased
    estimate of the phase-locked variance."""
    s_stat, s_diur, v_tot = [], [], []
    for d in bands:
        m = d[:, mask].astype(np.float64)
        m = m - m.mean(axis=1, keepdims=True)
        vt = float(np.mean(np.var(m, axis=1)))
        comp = {}
        for half in (0, 1):
            ch = np.stack([m[(hour_idx == h) & (block == half)].mean(axis=0)
                           for h in range(4)])
            comp[half] = ch
        ma, mb = comp[0].mean(axis=0), comp[1].mean(axis=0)
        stat = float(np.mean((ma - ma.mean()) * (mb - mb.mean())))
        da, db = comp[0] - ma, comp[1] - mb
        diur = float(np.mean([np.mean((da[h] - da[h].mean())
                                      * (db[h] - db[h].mean()))
                              for h in range(4)]))
        s_stat.append(stat / vt if vt > 0 else np.nan)
        s_diur.append(diur / vt if vt > 0 else np.nan)
        v_tot.append(vt)
    return s_stat, s_diur, v_tot


def conditional_coupling(envs, large, mask, sign_lat):
    """Adjacent-band envelope Spearman inside cyclonic / anticyclonic parts
    of the tile (sign of the >400 km vorticity), median over steps."""
    from scipy.stats import rankdata
    nt = envs[0].shape[0]
    cyc = (sign_lat * large[:, mask]) > 0
    e = [x[:, mask] for x in envs]
    npix = e[0].shape[1]
    out_c = np.full((nt, N_FINE), np.nan)
    out_a = np.full((nt, N_FINE), np.nan)
    for t in range(nt):
        for sel, out in ((cyc[t], out_c), (~cyc[t], out_a)):
            if sel.sum() < max(20, 0.25 * npix):
                continue
            r = [rankdata(x[t, sel]) for x in e]
            for i in range(N_FINE):
                out[t, i] = np.corrcoef(r[i], r[i + 1])[0, 1]
    frac = float(np.mean(cyc))
    return (np.nanmedian(out_c, axis=0), np.nanmedian(out_a, axis=0), frac,
            int(np.sum(np.isfinite(out_c[:, 0]))),
            int(np.sum(np.isfinite(out_a[:, 0]))))


def phase_randomize_mirror(u, v, rng):
    """Phase surrogate on the even (mirror) extension of the tile.

    The programme's surrogate takes the FFT of the bare tile; the jump
    between opposite edges then leaks broadband power, which phase
    randomisation spreads over the interior as small-scale noise. The mirror
    extension is continuous across the edges, so the leak is removed."""
    nt, ny, nx = u.shape

    def ext(a):
        a2 = np.concatenate([a, a[:, ::-1, :]], axis=1)
        return np.concatenate([a2, a2[:, :, ::-1]], axis=2)

    us, vs = phase_randomize(ext(u), ext(v), rng)
    return (np.ascontiguousarray(us[:, :ny, :nx]),
            np.ascontiguousarray(vs[:, :ny, :nx]))


def make_surrogate(u, v, rng):
    if _G.get("surrogate") == "mirror":
        return phase_randomize_mirror(u, v, rng)
    return phase_randomize(u, v, rng)


def env_decomp(w, dx_km, dy_km, mask):
    """Envelope-level split of the coupling: standard, transient-envelope
    (time-mean log-envelope removed per pixel) and stationary-envelope
    (coupling of the time-mean log-envelopes); plus the stationary share of
    the envelope variance."""
    from scipy.stats import spearmanr
    bands, _ = band_fields(np.nan_to_num(w), dx_km, dy_km)
    envs = [e[:, mask].astype(np.float64)
            for e in log_envelopes(bands, dx_km, dy_km)]
    means = [e.mean(axis=0) for e in envs]
    nt = envs[0].shape[0]
    std = np.empty((nt, N_FINE))
    tr = np.empty((nt, N_FINE))
    for i in range(N_FINE):
        std[:, i] = _corr_rows(_rank_rows(envs[i]), _rank_rows(envs[i + 1]))
        tr[:, i] = _corr_rows(_rank_rows(envs[i] - means[i][None]),
                              _rank_rows(envs[i + 1] - means[i + 1][None]))
    stat = [float(spearmanr(means[i], means[i + 1]).statistic)
            for i in range(N_FINE)]
    share = [float(np.var(means[i]) / max(np.mean(np.var(envs[i], axis=1)), 1e-30))
             for i in range(N_FINE + 1)]
    return np.median(std, axis=0), np.median(tr, axis=0), np.array(stat), np.array(share)


def _tile_worker(t: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    from scipy.ndimage import uniform_filter1d
    from scipy.stats import skew, kurtosis
    u, v = _G["u"], _G["v"]
    lat, lon = _G["lat"], _G["lon"]
    hour_idx, block = _G["hour_idx"], _G["block"]
    field, n_sur, salt = _G["field"], _G["n_sur"], _G["salt"]
    features = _G["features"]
    dlon_g = float(lon[1] - lon[0])
    iy = np.where((lat >= t["lat0"] - 1e-9) & (lat <= t["lat1"] + 1e-9))[0]
    nx = len(lon)
    i0 = int(np.searchsorted(lon, t["lon0"] - 1e-9))
    n_cols = max(4, int(round((t["lon1"] - t["lon0"]) / dlon_g)))
    ix = (np.arange(i0, i0 + n_cols)) % nx
    la = lat[iy]
    lo = t["lon0"] + dlon_g * np.arange(n_cols)
    uu = u[:, iy][:, :, ix].astype(np.float32)
    vv = v[:, iy][:, :, ix].astype(np.float32)
    if _G.get("isogrid") or _G.get("xfactor"):
        # control for the zonal oversampling of the lat-lon grid: resample
        # the tile to a zonal step that equals the meridional one in km
        # (isogrid), or to an arbitrary multiple of the native step (xfactor)
        step = (dlon_g / np.cos(np.deg2rad(t["lat_c"])) if _G.get("isogrid")
                else dlon_g * float(_G["xfactor"]))
        xo = np.arange(n_cols) * dlon_g
        xn = np.arange(int(np.floor((n_cols - 1) * dlon_g / step + 1e-9)) + 1) * step
        i0_ = np.clip(np.searchsorted(xo, xn, side="right") - 1, 0, n_cols - 2)
        wg = ((xn - xo[i0_]) / dlon_g).astype(np.float32)
        uu = uu[:, :, i0_] * (1 - wg) + uu[:, :, i0_ + 1] * wg
        vv = vv[:, :, i0_] * (1 - wg) + vv[:, :, i0_ + 1] * wg
        lo = t["lon0"] + xn
    dx_km, dy_km = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1], dx_km, dy_km)
    sign_lat = 1.0 if t["lat_c"] >= 0 else -1.0

    def fields(a, b):
        om = compute_vorticity(a, b, la, lo)
        if field == "vort":
            return om, None
        return compute_divergence(a, b, la, lo), om

    prim, other = fields(uu, vv)
    if _G.get("envdecomp"):
        tcode = zlib.crc32(f"{salt}_{t['tid']}".encode()) % 100000
        seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(n_sur)
        r = env_decomp(prim, dx_km, dy_km, mask)
        sur = [[], [], [], []]
        for s in seeds:
            rng = np.random.default_rng(int(s))
            us, vs = make_surrogate(uu, vv, rng)
            p_s, _ = fields(us, vs)
            for k_, v_ in enumerate(env_decomp(p_s, dx_km, dy_km, mask)):
                sur[k_].append(v_)
        names = ("std", "trans", "stat", "share")
        out = {"tid": t["tid"]}
        for k_, nme in enumerate(names):
            out[f"E_{nme}_real"] = np.asarray(r[k_]).tolist()
            out[f"E_{nme}_sur"] = np.median(np.asarray(sur[k_]), axis=0).tolist()
        return out
    rho = envelope_rho_profile_ells(prim, dx_km, dy_km, mask, ELLS_FINE)
    out = {"tid": t["tid"],
           "P_real_all": np.median(rho, axis=0).tolist(),
           "P_real_h": by_hour(rho, hour_idx).tolist()}
    if field == "div":
        b_d, _ = band_fields(np.nan_to_num(prim), dx_km, dy_km)
        b_v, _ = band_fields(np.nan_to_num(other), dx_km, dy_km)
        x_real = same_band_coupling(log_envelopes(b_v, dx_km, dy_km),
                                    log_envelopes(b_d, dx_km, dy_km), mask)
        out["X_real_all"] = np.median(x_real, axis=0).tolist()
        del b_d, b_v

    tcode = zlib.crc32(f"{salt}_{t['tid']}".encode()) % 100000
    seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(n_sur)
    sur_all = np.empty((n_sur, N_FINE))
    sur_h = np.empty((n_sur, 4, N_FINE))
    sur_x = np.empty((n_sur, N_FINE + 1))
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        us, vs = make_surrogate(uu, vv, rng)
        p_s, o_s = fields(us, vs)
        r_s = envelope_rho_profile_ells(p_s, dx_km, dy_km, mask, ELLS_FINE)
        sur_all[j] = np.median(r_s, axis=0)
        sur_h[j] = by_hour(r_s, hour_idx)
        if field == "div":
            b_d, _ = band_fields(np.nan_to_num(p_s), dx_km, dy_km)
            b_v, _ = band_fields(np.nan_to_num(o_s), dx_km, dy_km)
            sur_x[j] = np.median(same_band_coupling(
                log_envelopes(b_v, dx_km, dy_km),
                log_envelopes(b_d, dx_km, dy_km), mask), axis=0)
    out["P_sur_all"] = np.median(sur_all, axis=0).tolist()
    out["P_sur_h"] = np.median(sur_h, axis=0).tolist()
    out["P_sur_sd"] = np.std(sur_all, axis=0).tolist()
    out["P_anch_mean"] = float(np.mean(np.median(rho, axis=0)
                                       - np.median(sur_all, axis=0)))
    if field == "div":
        out["X_sur_all"] = np.median(sur_x, axis=0).tolist()

    if features:
        om = np.nan_to_num(prim if field == "vort" else other)
        dv = np.nan_to_num(compute_divergence(uu, vv, la, lo))
        b_v, large = band_fields(om, dx_km, dy_km)
        b_d, _ = band_fields(dv, dx_km, dy_km)
        ss, sd, vt = cv_fractions(b_v, mask, hour_idx, block)
        out["vort_S_stat"], out["vort_S_diur"], out["vort_var"] = ss, sd, vt
        ss, sd, vt = cv_fractions(b_d, mask, hour_idx, block)
        out["div_S_stat"], out["div_S_diur"], out["div_var"] = ss, sd, vt
        out["vort_skew_cyc"] = [float(sign_lat * skew(b[:, mask].ravel()))
                                for b in b_v]
        out["vort_kurt"] = [float(kurtosis(b[:, mask].ravel())) for b in b_v]
        out["div_skew"] = [float(skew(b[:, mask].ravel())) for b in b_d]
        envs = log_envelopes(b_v, dx_km, dy_km)
        pc, pa, fc, nc, na = conditional_coupling(envs, large, mask, sign_lat)
        out["P_cyc"], out["P_anti"] = pc.tolist(), pa.tolist()
        out["cyc_frac"], out["n_cyc"], out["n_anti"] = fc, nc, na
        out["env_var"] = [float(np.mean(np.var(e[:, mask], axis=1)))
                          for e in envs]
        up = uu - uniform_filter1d(uu, RUNMEAN_STEPS, axis=0, mode="nearest")
        vp = vv - uniform_filter1d(vv, RUNMEAN_STEPS, axis=0, mode="nearest")
        out["eke_syn"] = float(np.mean(
            0.5 * (np.var(up, axis=0) + np.var(vp, axis=0))[mask]))
        out["speed_mean"] = float(np.mean(np.sqrt(uu ** 2 + vv ** 2)[:, mask]))
    return out


def load_cube(source: str, season: str, level: int):
    import xarray as xr
    if source == "era5":
        parts = [xr.open_dataset(GLOB / f"era5_wind850_global_{ym}.nc")
                 for ym in ERA5_MONTHS[season]]
        ds = xr.concat(parts, dim="valid_time")
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
        time = ds["valid_time"].values
        u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
        v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
        for p in parts:
            p.close()
    else:
        pu, pv = [], []
        for ym in MODEL_MONTHS[season]:
            fu = next(MODEL_DIR.glob(f"ua_*_{ym}01*-*.nc"))
            fv = next(MODEL_DIR.glob(f"va_*_{ym}01*-*.nc"))
            pu.append(xr.open_dataset(fu)["ua"].sel(plev=float(level) * 100.0))
            pv.append(xr.open_dataset(fv)["va"].sel(plev=float(level) * 100.0))
        ua = xr.concat(pu, dim="time")
        va = xr.concat(pv, dim="time")
        lat = np.asarray(ua["lat"].values, dtype=float)
        lon = np.asarray(ua["lon"].values, dtype=float)
        time = ua["time"].values
        u = np.asarray(ua.values, dtype=np.float32)
        v = np.asarray(va.values, dtype=np.float32)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]
    t = time.astype("datetime64[h]")
    hours = (t - t.astype("datetime64[D]")).astype(int)
    hour_idx = (hours // 6).astype(int)
    day = (t.astype("datetime64[D]") - t.astype("datetime64[D]")[0]).astype(int)
    block = ((day // BLOCK_DAYS) % 2).astype(int)
    return np.ascontiguousarray(u), np.ascontiguousarray(v), lat, lon, hour_idx, block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["era5", "model"], default="era5")
    ap.add_argument("--season", choices=["JFM", "JAS"], default="JFM")
    ap.add_argument("--level", type=int, default=850)
    ap.add_argument("--field", choices=["vort", "div"], default="vort")
    ap.add_argument("--transient", type=int, default=0)
    ap.add_argument("--features", type=int, default=1)
    ap.add_argument("--nsur", type=int, default=24)
    ap.add_argument("--envdecomp", type=int, default=0)
    ap.add_argument("--isogrid", type=int, default=0)
    ap.add_argument("--xfactor", type=float, default=0.0,
                    help="resample the zonal step to this multiple of the native one")
    ap.add_argument("--surrogate", choices=["fft", "mirror"], default="fft")
    ap.add_argument("--latmin", type=float, default=0.0)
    ap.add_argument("--latmax", type=float, default=90.0)
    ap.add_argument("--every", type=int, default=1, help="use every n-th tile")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 3))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N tiles")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.source}{args.level}_{args.field}_"
           f"{'trans' if args.transient else 'full'}"
           f"{'_envdecomp' if args.envdecomp else ''}"
           f"{'_iso' if args.isogrid else ''}"
           f"{f'_xf{args.xfactor:g}' if args.xfactor else ''}"
           f"{'_mirror' if args.surrogate == 'mirror' else ''}"
           f"{f'_lat{args.latmin:g}-{args.latmax:g}' if (args.latmin > 0 or args.latmax < 90) else ''}"
           f"_{args.season}")
    cached = OUT / f"tiles_{tag}.json"
    if cached.exists() and not args.limit:
        print("cached", tag, flush=True)
        return
    u, v, lat, lon, hour_idx, block = load_cube(args.source, args.season,
                                                args.level)
    print(f"{tag}: cube {u.shape}", flush=True)
    if args.transient:
        for h in range(4):
            sel = hour_idx == h
            u[sel] -= u[sel].mean(axis=0, keepdims=True)
            v[sel] -= v[sel].mean(axis=0, keepdims=True)
        print("composite removed", flush=True)
    _G.update(u=u, v=v, lat=lat, lon=lon, hour_idx=hour_idx, block=block,
              field=args.field, n_sur=args.nsur, salt=f"xp_{tag}",
              features=bool(args.features), envdecomp=bool(args.envdecomp),
              isogrid=bool(args.isogrid), xfactor=args.xfactor,
              surrogate=args.surrogate)
    tiles = [t for t in tile_grid()
             if args.latmin <= abs(t["lat_c"]) <= args.latmax]
    if args.every > 1:
        tiles = tiles[:: args.every]
    if args.limit:
        tiles = tiles[:: max(1, len(tiles) // args.limit)][: args.limit]
    ctx = mp.get_context("fork")
    res = []
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_tile_worker, tiles,
                                                  chunksize=4)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"[{tag}: {i+1}/{len(tiles)}]", flush=True)
    if args.limit:
        print(json.dumps(res[:2], indent=1)[:3000])
        return
    cached.write_text(json.dumps({"tag": tag, "args": vars(args),
                                  "tiles": res}), encoding="utf-8")
    print(f"{tag}: {len(res)} tiles done", flush=True)


if __name__ == "__main__":
    main()
