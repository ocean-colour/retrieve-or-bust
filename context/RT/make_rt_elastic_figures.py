"""
Figures for context/rt_elastic_model.md — the elastic radiative-transfer (RT)
forward-model synthesis for retrieve-or-bust.

All figures are generated from the Loisel+2023 (L23) *elastic* Hydrolight dataset
(scenario X=1: no inelastic processes) and the wavelength-dependent Gordon
coefficient tables fit in the BING repo. Nothing here is hand-typed from a log:
the rRMS ladder is recomputed from L23, and the G0(λ)/Gb(λ) curves are read from
the shipped BING coefficient CSVs.

Data:
  L23 elastic  : $OS_COLOR_DATA/Loisel2023/Hydrolight100.nc
                 (found on this laptop at /Users/xavier/data/Color/Loisel2023/)
  BING coeffs  : <bing>/bing/data/RT/gordon_coefficients_with_{G0,Gb}.csv

Convention (matches bing.rt.rrs): subsurface rrs = Rrs / (A + B*Rrs),
A = 0.52, B = 1.7 (Lee et al. 2002); u = bb / (a + bb).

Run in the `ocean14` conda environment:
    python context/RT/make_rt_elastic_figures.py

Outputs (written next to this script, in context/RT/):
  fig_rrs_vs_u.png      — Rrs is not univocal in u (the central physical point)
  fig_rrms_ladder.png   — per-wavelength rRMS: standard vs +G0 vs +Gb (recomputed)
  fig_G_lambda.png      — the fitted G0(λ) and Gb(λ) wavelength structure

Author: Claude (Fable) for J. Xavier Prochaska, 2026-07-19.
"""
from __future__ import annotations

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------- paths -----
HERE = os.path.dirname(os.path.abspath(__file__))

A_RRS, B_RRS = 0.52, 1.7          # Rrs <-> rrs (Lee 2002), matches bing.rt.rrs
STD_G1, STD_G2 = 0.0949, 0.0794   # canonical Gordon (1988) coefficients


def find_l23_elastic() -> str:
    """Locate the L23 elastic file Hydrolight100.nc (X=1, no inelastic)."""
    candidates = []
    for base in (os.getenv("OS_COLOR_DATA"), os.getenv("OS_COLOR"),
                 "/Users/xavier/data/Color", os.path.expanduser("~/data/Color")):
        if base:
            candidates.append(os.path.join(base, "Loisel2023", "Hydrolight100.nc"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    hits = glob.glob("/Users/xavier/**/Loisel2023/Hydrolight100.nc", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("Could not find Hydrolight100.nc (L23 elastic).")


def find_bing_coeff(name: str) -> str:
    for base in ("/Users/xavier/Oceanography/python/bing/bing/data/RT",):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(name)


def Rrs_to_rrs(Rrs, A=A_RRS, B=B_RRS):
    return Rrs / (A + B * Rrs)


# ---------------------------------------------------------------- load ------
def load_l23():
    """Return (wave, u, rrs, bbp, bbw, a, bb) from the L23 elastic set."""
    ds = xr.open_dataset(find_l23_elastic())
    wave = ds["Lambda"].values.astype(float)           # (nwave,)
    Rrs = ds["Rrs"].values                             # (nscene, nwave)
    a = ds["a"].values
    bb = ds["bb"].values
    bbp = ds["bbnw"].values                            # non-water backscatter
    bbw = bb - bbp
    u = bb / (a + bb)
    rrs = Rrs_to_rrs(Rrs)
    return dict(wave=wave, u=u, rrs=rrs, bbp=bbp, bbw=bbw, a=a, bb=bb, Rrs=Rrs)


# --------------------------------------------------- fits (recomputed) ------
def fit_std_rRMS(u_col, rrs_col):
    pred = STD_G1 * u_col + STD_G2 * u_col**2
    return rrms(rrs_col, pred)


def fit_quad_rRMS(u_col, rrs_col):
    def m(u, g1, g2):
        return g1 * u + g2 * u**2
    p, _ = curve_fit(m, u_col, rrs_col, p0=[STD_G1, STD_G2], maxfev=20000)
    return rrms(rrs_col, m(u_col, *p))


def fit_const_rRMS(u_col, rrs_col):
    def m(u, g0, g1, g2):
        return g0 + g1 * u + g2 * u**2
    p, _ = curve_fit(m, u_col, rrs_col, p0=[0.0, STD_G1, STD_G2], maxfev=20000)
    return rrms(rrs_col, m(u_col, *p))


def fit_bbp_rRMS(u_col, rrs_col, bbp_col):
    def m(X, g1, g2, gb):
        u, bp = X
        return g1 * u + g2 * u**2 + gb * bp
    p, _ = curve_fit(m, (u_col, bbp_col), rrs_col,
                     p0=[STD_G1, STD_G2, 0.0], maxfev=20000)
    return rrms(rrs_col, m((u_col, bbp_col), *p))


def rrms(truth, pred):
    """Relative RMS in rrs-space (%), matching the BING definition."""
    resid = (pred - truth) / truth
    return 100.0 * np.sqrt(np.mean(resid**2))


# ---------------------------------------------------------------- fig 1 -----
def fig_rrs_vs_u(D, panels=(440.0, 550.0, 665.0)):
    wave = D["wave"]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.2), sharey=False)
    for ax, lam in zip(axes, panels):
        j = int(np.argmin(np.abs(wave - lam)))
        u = D["u"][:, j]
        rrs = D["rrs"][:, j]
        bbp = D["bbp"][:, j]
        order = np.argsort(bbp)
        sc = ax.scatter(u[order], rrs[order], c=bbp[order], s=7,
                        cmap="viridis", norm=LogNorm(), alpha=0.8, lw=0)
        ug = np.linspace(u.min(), u.max(), 200)
        ax.plot(ug, STD_G1 * ug + STD_G2 * ug**2, "r-", lw=2,
                label="standard Gordon\n$0.0949\\,u+0.0794\\,u^2$")
        ax.set_title(f"{wave[j]:.0f} nm", fontsize=12)
        ax.set_xlabel(r"$u = b_b/(a+b_b)$")
        if ax is axes[0]:
            ax.set_ylabel(r"subsurface $r_{rs}$  (sr$^{-1}$)")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$b_{bp}$  (m$^{-1}$)", fontsize=9)
    fig.suptitle("At fixed $u$, $r_{rs}$ still varies with $b_{bp}$ — "
                 "$r_{rs}$ is not univocal in $u$  (L23 elastic, 3320 scenes)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "fig_rrs_vs_u.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- fig 2 -----
def fig_rrms_ladder(D, lams=(400, 450, 500, 550, 600, 650, 700)):
    wave = D["wave"]
    rows = []
    for lam in lams:
        j = int(np.argmin(np.abs(wave - lam)))
        u = D["u"][:, j]
        rrs = D["rrs"][:, j]
        bbp = D["bbp"][:, j]
        good = np.isfinite(u) & np.isfinite(rrs) & (rrs > 0)
        rows.append(dict(
            lam=wave[j],
            std=fit_std_rRMS(u[good], rrs[good]),
            quad=fit_quad_rRMS(u[good], rrs[good]),
            const=fit_const_rRMS(u[good], rrs[good]),
            bbp=fit_bbp_rRMS(u[good], rrs[good], bbp[good]),
        ))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.plot(df.lam, df["std"], "o-", color="0.4", label="standard Gordon (fixed $G_1,G_2$)")
    ax.plot(df.lam, df["quad"], "s-", color="C0", label=r"per-$\lambda$ 2-param $(G_1,G_2)(\lambda)$")
    ax.plot(df.lam, df["const"], "^-", color="C3", label=r"per-$\lambda$ 3-param $+G_0(\lambda)$")
    ax.plot(df.lam, df["bbp"], "v-", color="C2", label=r"per-$\lambda$ 3-param $+G_b\,b_{bp}$")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("rRMS vs Hydrolight  (%,  $r_{rs}$-space)")
    ax.set_title("Recovering the residual $(a,b_b)$ structure collapses the red-$\\lambda$ error\n"
                 "(recomputed from L23 elastic, unregularized per-$\\lambda$ fits)", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_rrms_ladder.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    df.to_csv(os.path.join(HERE, "fig_rrms_ladder.csv"), index=False)
    print("wrote", out)
    print(df.round(2).to_string(index=False))


# ---------------------------------------------------------------- fig 3 -----
def fig_G_lambda():
    g0 = pd.read_csv(find_bing_coeff("gordon_coefficients_with_G0.csv"), comment="#")
    gb = pd.read_csv(find_bing_coeff("gordon_coefficients_with_Gb.csv"), comment="#")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.4))

    ax0.axhline(0, color="0.7", lw=0.8)
    ax0.plot(g0.wavelength, g0.G0, "o-", color="C3", ms=3)
    ax0.set_xlabel("wavelength (nm)")
    ax0.set_ylabel(r"$G_0(\lambda)$  (sr$^{-1}$, offset term)")
    ax0.set_title(r"$G_0(\lambda)$: the constant offset (3-param $+G_0$ fit)", fontsize=11)
    ax0.grid(alpha=0.3)
    ax0.annotate("sign change near 510 nm", xy=(510, 0), xytext=(560, -4e-4),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="0.4"))

    ax1.axhline(0, color="0.7", lw=0.8)
    ax1.plot(gb.wavelength, gb.Gb, "o-", color="C2", ms=3)
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel(r"$G_b(\lambda)$  (slope on $b_{bp}$)")
    ax1.set_title(r"$G_b(\lambda)$: the $b_{bp}$-linear term (3-param $+G_b$ fit)", fontsize=11)
    ax1.grid(alpha=0.3)

    fig.suptitle("BING's wavelength-dependent enrichment terms — the empirical "
                 "signature that $r_{rs}\\neq f(u)$ alone", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "fig_G_lambda.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    warnings.filterwarnings("ignore")
    mpl.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    D = load_l23()
    print(f"L23 elastic: {D['rrs'].shape[0]} scenes x {D['wave'].size} wavelengths "
          f"({D['wave'].min():.0f}-{D['wave'].max():.0f} nm)")
    fig_rrs_vs_u(D)
    fig_rrms_ladder(D)
    fig_G_lambda()


if __name__ == "__main__":
    main()
