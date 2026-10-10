#!/usr/bin/env python3
"""Method figures of article 1, revised version v8 (definition V12 of Phase 32).

fig1_method_v8.png   two-part figure: (а) from the vorticity field to the activity
                     map of one scale band (Europe, R11, one snapshot); (б) from two
                     activity maps to the coupling: ranks, phase surrogate, the
                     time series of rho over the window and the corrected profile.
                     The below-surface mask (850 hPa under the terrain) is hatched.
fig5_congo_amazon_v8.png  Congo vs Amazon: activity maps of the 800-1600 and 400-800 km
                     bands at one snapshot and the joint rank distribution over all
                     snapshots of the window (JFM 2023).

Pipeline = V12 exactly as in experiment_B32_vorticity_mask / experiment_B33:
spherical curl, below-surface points missing (normalised convolution), rank
correlation over interior & above-surface nodes.

Stages: --stage data | figures   (data: ~3 min with 8 workers)
Output: manuscript_v2/article1/figures/v8/
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

import experiment_B32_vorticity_mask as vm  # noqa: E402
from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask, phase_randomize  # noqa: E402

CACHE = _HERE / "results" / "article1_method_v8"
FIG = ROOT / "manuscript_v2" / "article1" / "figures" / "v8"
ELLS = vm.ELLS
EPS = vm.EPS
N_SUR = 199
SEED_SUR = 20261008
STEP = 3                    # pair 200-400 | 400-800 km (step 4)
_G: dict = {}

METHOD_REGION, METHOD_WINDOW = "R11_EURO", "W11_2024JFM"
CA_WINDOW = "W9_2023JFM"


# ---------------------------------------------------------------- data

def load(tag):
    import xarray as xr
    ds = xr.open_dataset(ROOT / "data" / "b3" / f"era5_wind850_{tag}.nc")
    lat = np.asarray(ds["latitude"].values, float)
    lon = np.asarray(ds["longitude"].values, float)
    u = np.asarray(ds["u"].squeeze().values, np.float32)
    v = np.asarray(ds["v"].squeeze().values, np.float32)
    tm = [str(x)[:13] for x in ds["valid_time"].values]
    ds.close()
    if lat[0] > lat[-1]:
        lat, u, v = lat[::-1], u[:, ::-1], v[:, ::-1]
    return lat, lon, u, v, tm


def geometry(lat, lon, ny, nx):
    dx, dy = _grid_geometry(lat, lon)
    interior = _interior_mask(ny, nx, ELLS[-1], dx, dy)
    below = vm.below_surface_mask(lat, lon)
    return dx, dy, interior, below


def coast(lat, lon):
    import xarray as xr
    ds = xr.open_dataset(ROOT / "data" / "b4cov" / "era5_invariants_global.nc")
    lsm = ds["lsm"].squeeze()
    sel = lsm.sel(latitude=xr.DataArray(lat, dims="y"), longitude=xr.DataArray(np.mod(lon, 360.0), dims="x"),
                  method="nearest")
    ds.close()
    return np.asarray(sel.values, float)


def levels_v12(w, dx, dy, valid):
    """Smoothed fields, octave levels and envelopes of one snapshot under V12 (NaN below surface)."""
    w = np.where(valid, w, np.nan).astype(np.float32)[None]
    bars = [vm.bar_nan(w, ell / np.sqrt(12.0), dx, dy, valid)[0] for ell in ELLS]
    D = [np.nan_to_num(w[0]) - np.nan_to_num(bars[0])] + [bars[i] - bars[i + 1] for i in range(len(ELLS) - 1)]
    E = [vm.bar_nan(np.abs(np.nan_to_num(D[0]))[None], ELLS[0] / np.sqrt(12.0), dx, dy, valid)[0]]
    E += [vm.bar_nan(np.abs(np.nan_to_num(D[i + 1]))[None], ELLS[i + 1] / np.sqrt(12.0), dx, dy, valid)[0]
          for i in range(len(ELLS) - 1)]
    return bars, D, E


def _sur_rho(seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    us, vs = phase_randomize(_G["u"], _G["v"], np.random.default_rng(int(seed)))
    om = vm.vorticity(us, vs, _G["la"], _G["lo"], spherical=True)
    return vm.rho_profile(om, _G["dx"], _G["dy"], _G["corr"], _G["valid"], ELLS).astype(np.float32)


def stage_data(workers):
    CACHE.mkdir(parents=True, exist_ok=True)
    # ---- method window
    tag = f"{METHOD_REGION}__{METHOD_WINDOW}"
    la, lo, u, v, tm = load(tag)
    dx, dy, interior, below = geometry(la, lo, u.shape[1], u.shape[2])
    valid, corr = ~below, interior & ~below
    om = vm.vorticity(u, v, la, lo, spherical=True)
    rho = vm.rho_profile(om, dx, dy, corr, valid, ELLS)
    seeds = np.random.SeedSequence(SEED_SUR + zlib.crc32(tag.encode()) % 100000).generate_state(N_SUR)
    _G.update(u=u, v=v, la=la, lo=lo, dx=dx, dy=dy, corr=corr, valid=valid)
    with mp.get_context("fork").Pool(workers) as pool:
        rho_sur = np.stack(pool.map(_sur_rho, seeds))               # (n_sur, nt, 5)
    med = np.median(rho[:, STEP])
    cand = np.where(np.abs(rho[:, STEP] - med) < 0.01)[0]
    t0 = int(cand[np.argmax(np.nan_to_num(om[cand])[:, corr].std(axis=1))])
    us, vs = phase_randomize(u[t0:t0 + 1], v[t0:t0 + 1], np.random.default_rng(int(seeds[0])))
    om_s = vm.vorticity(us, vs, la, lo, spherical=True)
    rho_s0 = vm.rho_profile(om_s, dx, dy, corr, valid, ELLS)[0]
    out = {}
    for name, w in (("real", om[t0]), ("sur", om_s[0])):
        bars, D, E = levels_v12(w, dx, dy, valid)
        out.update({f"{name}_om": np.where(valid, w, np.nan), f"{name}_bar200": bars[2], f"{name}_bar400": bars[3],
                    f"{name}_D3": D[3], f"{name}_E3": E[3], f"{name}_E4": E[4]})
    np.savez_compressed(CACHE / f"method_{tag}.npz", la=la, lo=lo, dx=dx, dy=dy, interior=interior, below=below,
                        corr=corr, lsm=coast(la, lo), rho=rho, rho_sur=rho_sur, t0=t0, time=tm[t0], rho_s0=rho_s0, **out)
    print("method:", tag, "t0", t0, tm[t0], "P", np.round(np.median(rho, 0), 3),
          "Psur", np.round(np.median(np.median(rho_sur, 1), 0), 3), "below frac", round(float(below[interior].mean()), 3))
    # ---- Congo / Amazon
    for reg in ("R7_CONGO", "R3_AMAZ"):
        tag = f"{reg}__{CA_WINDOW}"
        la, lo, u, v, tm = load(tag)
        dx, dy, interior, below = geometry(la, lo, u.shape[1], u.shape[2])
        valid, corr = ~below, interior & ~below
        om = vm.vorticity(u, v, la, lo, spherical=True)
        rho = vm.rho_profile(om, dx, dy, corr, valid, ELLS)
        # envelopes of bands 400-800 (E4) and 800-1600 (E5) for all snapshots: ranks pooled
        w = np.where(valid[None], om, np.nan).astype(np.float32)
        bars = [vm.bar_nan(w, ell / np.sqrt(12.0), dx, dy, valid) for ell in ELLS]
        E4 = vm.bar_nan(np.abs(np.nan_to_num(bars[3] - bars[4])), ELLS[4] / np.sqrt(12.0), dx, dy, valid)
        E5 = vm.bar_nan(np.abs(np.nan_to_num(bars[4] - bars[5])), ELLS[5] / np.sqrt(12.0), dx, dy, valid)
        del bars
        r4 = np.argsort(np.argsort(E4[:, corr], axis=1), axis=1) / (corr.sum() - 1.0)
        r5 = np.argsort(np.argsort(E5[:, corr], axis=1), axis=1) / (corr.sum() - 1.0)
        H, _, _ = np.histogram2d(r5.ravel(), r4.ravel(), bins=25, range=[[0, 1], [0, 1]])
        med5 = np.median(rho[:, 4])
        cand = np.where(np.abs(rho[:, 4] - med5) < 0.02)[0]
        t0 = int(cand[np.argmax(np.nan_to_num(om[cand])[:, corr].std(axis=1))])
        np.savez_compressed(CACHE / f"ca_{tag}.npz", la=la, lo=lo, interior=interior, below=below, corr=corr,
                            lsm=coast(la, lo), rho=rho, t0=t0, time=tm[t0], E4=E4[t0], E5=E5[t0], H=H,
                            rho5_t0=float(rho[t0, 4]))
        print("congo/amazon:", tag, "t0", t0, tm[t0], "P", np.round(np.median(rho, 0), 3), "rho5(t0)", round(float(rho[t0, 4]), 3))


# ---------------------------------------------------------------- figures
W_IN = 17.5 / 2.54
DPI = 400
C_REAL, C_SUR, C_DIFF, INK, GRID = "#1c5cab", "#8a8a86", "#eb6834", "#1a1a19", "#d9d8d3"
PAIRS = ["50–100 |\n100–200", "100–200 |\n200–400", "200–400 |\n400–800", "400–800 |\n800–1600"]
STEPS = [1, 2, 3, 4]


def T(s):
    return s.replace("ё", "е")


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "hatch.linewidth": 0.4})
    return plt


def _show(ax, f, la, lo, **kw):
    dlo, dla = lo[1] - lo[0], la[1] - la[0]
    ext = [lo[0] - dlo / 2, lo[-1] + dlo / 2, la[0] - dla / 2, la[-1] + dla / 2]
    return ax.imshow(f, origin="lower", extent=ext, aspect=dlo / dla, interpolation="bilinear", **kw)


def _overlay(ax, d, la, lo, box=True, coastcol="#4d4d4d", maskcol=INK):
    from matplotlib.patches import Rectangle
    ax.contour(lo, la, d["lsm"], levels=[0.5], colors=coastcol, linewidths=0.45)
    if d["below"].any():
        import matplotlib as _mpl
        with _mpl.rc_context({"hatch.color": maskcol}):
            ax.contourf(lo, la, d["below"].astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.contour(lo, la, d["below"].astype(float), levels=[0.5], colors=maskcol, linewidths=0.4)
    if box:
        m = d["interior"]
        rows, cols = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
        ax.add_patch(Rectangle((lo[cols[0]], la[rows[0]]), lo[cols[-1]] - lo[cols[0]], la[rows[-1]] - la[rows[0]],
                               fill=False, ec="white", lw=0.9, ls=(0, (3, 2))))


def _ticks(ax, la, lo, xlab, ylab, xs=None, ys=None):
    xs = xs if xs is not None else [x for x in range(int(np.ceil(lo[0] / 10) * 10), int(lo[-1]) + 1, 10)]
    ys = ys if ys is not None else [y for y in range(int(np.ceil(la[0] / 10) * 10), int(la[-1]) + 1, 10)]
    ax.set_xticks(xs); ax.set_yticks(ys)

    def fx(x):
        return f"{abs(x)}°" if x != xs[-1] else f"{abs(x)}° {'в.д.' if x > 0 else 'з.д.'}"

    def fy(y):
        return "0°" if y == 0 else f"{abs(y)}° {'с.ш.' if y > 0 else 'ю.ш.'}"
    ax.set_xticklabels([fx(x) for x in xs] if xlab else [])
    ax.set_yticklabels([fy(y) for y in ys] if ylab else [])
    ax.tick_params(length=2, pad=1.5)


def _ranks(a):
    return np.argsort(np.argsort(a)) / (a.size - 1.0)


def fig1(plt):
    from matplotlib.gridspec import GridSpec
    d = np.load(CACHE / f"method_{METHOD_REGION}__{METHOD_WINDOW}.npz")
    la, lo = d["la"], d["lo"]
    rho, rho_sur, t0 = d["rho"], d["rho_sur"], int(d["t0"])
    P, Ps = np.median(rho, 0), np.median(np.median(rho_sur, 1), 0)
    sc = 1e5
    fig = plt.figure(figsize=(W_IN, 7.6 / 2.54))
    ga = GridSpec(2, 3, figure=fig, left=0.065, right=0.93, bottom=0.10, top=0.93, hspace=0.34, wspace=0.14)
    panels = [("1  вихрь ω", d["real_om"] * sc, "RdGy_r", True),
              ("2  сглаживание, ℓ = 200 км", d["real_bar200"] * sc, "RdGy_r", True),
              ("3  сглаживание, ℓ = 400 км", d["real_bar400"] * sc, "RdGy_r", True),
              ("4  полоса 200–400 км: (2) − (3)", d["real_D3"] * sc, "RdGy_r", True),
              ("5  модуль полосы |(4)|", np.abs(d["real_D3"]) * sc, "Greys", False),
              ("6  карта активности", d["real_E3"] * sc, "Greys", False)]
    km_per_deg = float(np.mean(d["dx"])) / (lo[1] - lo[0])
    for k, (title, f, cmap, signed) in enumerate(panels):
        ax = fig.add_subplot(ga[k // 3, k % 3])
        vmax = float(np.nanquantile(np.abs(f), 0.995))
        im = _show(ax, f, la, lo, cmap=cmap, vmin=-vmax if signed else 0.0, vmax=vmax)
        _overlay(ax, d, la, lo, box=(k == 5))
        _ticks(ax, la, lo, xlab=(k >= 3), ylab=(k % 3 == 0), xs=[0, 10, 20], ys=[40, 50])
        ax.set_title(T(title), loc="left", pad=2.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.4)
        if k % 3 == 2:
            cb.set_label("10⁻⁵ с⁻¹", fontsize=6.5)
        if k in (1, 2):
            ell = 200.0 if k == 1 else 400.0
            x0, y0 = lo[4], la[5]
            ax.plot([x0, x0 + ell / km_per_deg], [y0, y0], color=INK, lw=2, solid_capstyle="butt")
            ax.text(x0, y0 + 0.6, f"{int(ell)} км", fontsize=6, va="bottom", ha="left", color=INK)
    out = FIG / "fig1_method_v8.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out, dict(time=str(d["time"]), t0=t0, P=P, Ps=Ps, rho_t0=float(rho[t0, STEP]), rho_s0=float(d["rho_s0"][STEP]),
                     below=float(d["below"][d["interior"]].mean()))


def fig5(plt):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(W_IN, 9.0 / 2.54))
    gs = GridSpec(2, 3, figure=fig, left=0.095, right=0.985, bottom=0.10, top=0.93, hspace=0.55, wspace=0.20,
                  width_ratios=[1.6, 1.6, 0.85])
    info = {}
    for r, (reg, name) in enumerate((("R7_CONGO", "бассейн Конго"), ("R3_AMAZ", "Амазония"))):
        d = np.load(CACHE / f"ca_{reg}__{CA_WINDOW}.npz")
        la, lo = d["la"], d["lo"]
        xs_ = [10, 20, 30, 40] if reg == "R7_CONGO" else [-70, -60, -50, -40]
        e5, e4 = d["E5"], d["E4"]
        f5 = (e5 - np.nanmin(e5)) / (np.nanmax(e5) - np.nanmin(e5))
        f4 = (e4 - np.nanmin(e4)) / (np.nanmax(e4) - np.nanmin(e4))
        ax = fig.add_subplot(gs[r, 0])
        _show(ax, f5, la, lo, cmap="magma", vmin=0, vmax=1)
        ax.contour(lo, la, f5, levels=[0.3, 0.55, 0.8], colors="white", linewidths=0.6)
        _overlay(ax, d, la, lo, box=True, coastcol="#9fd3ff", maskcol="white")
        _ticks(ax, la, lo, xlab=True, ylab=True, xs=xs_, ys=[-10, 0])
        ax.set_title(T(f"{1 + 3 * r}  {name}: 800–1600 км, один срок"), loc="left", pad=2.5)
        ax = fig.add_subplot(gs[r, 1])
        _show(ax, f4, la, lo, cmap="magma", vmin=0, vmax=1)
        ax.contour(lo, la, f5, levels=[0.3, 0.55, 0.8], colors="white", linewidths=0.6)
        _overlay(ax, d, la, lo, box=True, coastcol="#9fd3ff", maskcol="white")
        _ticks(ax, la, lo, xlab=True, ylab=False, xs=xs_, ys=[-10, 0])
        ax.set_title(T(f"{2 + 3 * r}  400–800 км, тот же срок"), loc="left", pad=2.5)
        ax = fig.add_subplot(gs[r, 2])
        H = d["H"]
        H = H / H.sum() * H.size
        ax.imshow(H, origin="lower", extent=[0, 1, 0, 1], cmap="magma", vmin=0, vmax=float(np.quantile(H, 0.99)),
                  interpolation="nearest", aspect="equal")
        ax.plot([0, 1], [0, 1], color="white", lw=0.8, ls=(0, (4, 2)))
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"] if r else []); ax.set_yticklabels(["0", "0.5", "1"])
        if r:
            ax.set_xlabel(T("ранг, 800–1600 км"), labelpad=1.5)
        ax.set_ylabel(T("ранг, 400–800 км"), labelpad=1.5)
        ax.set_title(T(f"{3 + 3 * r}  все сроки окна"), loc="left", pad=2.5)
        info[reg] = dict(time=str(d["time"]), P=np.median(d["rho"], 0), rho5_t0=float(d["rho5_t0"]),
                         below=float(d["below"][d["interior"]].mean()))
    out = FIG / "fig5_congo_amazon_v8.png"
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out, info


def stage_figures():
    plt = _setup()
    FIG.mkdir(parents=True, exist_ok=True)
    out1, i1 = fig1(plt)
    out5, i5 = fig5(plt)
    lines = ["fig1_method_v8.png: " + str(i1), "fig5_congo_amazon_v8.png: " + str(i5)]
    (FIG / "method_figures_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    from PIL import Image
    for out in (out1, out5):
        im = Image.open(out)
        print(out.name, im.size, "dpi", im.info.get("dpi"), f"{im.size[0] / DPI * 2.54:.1f} x {im.size[1] / DPI * 2.54:.1f} cm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["data", "figures"], default="figures")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "data":
        stage_data(a.workers)
    else:
        stage_figures()
