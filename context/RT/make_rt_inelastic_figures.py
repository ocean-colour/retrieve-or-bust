"""
Figures for context/rt_inelastic_bing_summary.md
=================================================

Quantifies BING's inelastic-RT approximations against the Loisel et al.
(2023, "L23") synthetic database, which provides paired HydroLight runs:

- X = 1 : elastic only
- X = 2 : + Raman scattering by water
- X = 4 : + Raman and chlorophyll-a fluorescence (phi_C = 0.02)

at three solar zenith angles (00, 30, 60 deg), 3320 IOP scenes each,
350-750 nm at 5 nm. Scenario differences isolate each inelastic process:

- Raman truth       : Rrs(X2) / Rrs(X1)   (BING models this as a
                      multiplicative correction factor)
- fluorescence truth: Rrs(X4) - Rrs(X2)   (BING models this as an
                      additive Rrs term)

L23 used HydroLight *defaults* (Mobley 2012 Raman settings; phi_C = 0.02),
which are also BING's defaults, so the comparisons below isolate the error
of BING's *formulation* (Sathyendranath & Platt 1998 two-flow, single
3400 cm^-1 shift, flat-Ed assumption, fixed mean cosines) with the physical
constants held equal.

Outputs (written next to this script, context/RT/):
- fig_l23_inelastic_impact.png   : size of each process in the L23 truth
- fig_raman_bing_vs_l23.png      : BING Raman correction vs L23, 3 variants
- fig_fluor_bing_vs_l23.png      : BING fluorescence term vs L23
- fig_raman_redistribution.png   : single-shift vs Walrafen redistribution
- rt_inelastic_metrics.csv       : summary numbers quoted in the report

Run with:  conda run -n ocean14 python make_rt_inelastic_figures.py
"""

import os

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

from bing.rt import raman
from bing.rt import rrs as bing_rrs

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# L23 location: $OS_COLOR_DATA/Loisel2023 if set, else the known path
L23_PATH = os.path.join(
    os.environ.get('OS_COLOR_DATA', '/mnt/tank/Oceanography/data/Color'),
    'Loisel2023')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ZENITHS = ['00', '30', '60']

# Okabe-Ito (colorblind-safe); truth is always black
C_TRUTH = 'k'
C_V1 = '#D55E00'   # BING default (flat Ed, mu_d = 0.9)
C_V2 = '#0072B2'   # + true Ed(lambda') / Ed(lambda)
C_V3 = '#009E73'   # + zenith-dependent mu_d
# Sequential blues for the three zeniths (light -> dark = 0 -> 60 deg)
C_ZEN = {'00': '#9ecae1', '30': '#4292c6', '60': '#08519c'}

FIG_KW = dict(dpi=200, bbox_inches='tight')


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_scenario(x: int, zen: str):
    """Load one L23 surface file.

    Parameters
    ----------
    x : int
        Inelastic scenario (1, 2 or 4).
    zen : str
        Solar zenith angle string ('00', '30', '60').

    Returns
    -------
    dict with keys 'wave' (81,), 'Rrs', 'a', 'bb', 'aph', 'Ed'
        (each (3320, 81) except wave; Ed is Ed(0+)).
    """
    fname = os.path.join(L23_PATH, f'Hydrolight{x}{zen}.nc')
    ds = xr.open_dataset(fname)
    out = dict(
        wave=ds['Lambda'].values.astype(float),
        Rrs=ds['Rrs'].values.astype(float),
        a=ds['a'].values.astype(float),
        bb=ds['bb'].values.astype(float),
        aph=ds['aph'].values.astype(float),
        Ed=ds['Ed_0+'].values.astype(float),
    )
    ds.close()
    return out


def interp_scenes(wave, arr, wave_new):
    """Interpolate a (n_scene, n_wave) array onto wave_new (linear)."""
    f = interp1d(wave, arr, axis=-1, kind='linear',
                 bounds_error=True)
    return f(wave_new)


# ----------------------------------------------------------------------
# Raman comparison
# ----------------------------------------------------------------------

def mu_d_of_zenith(zen_deg: float) -> float:
    """In-water mean cosine of the refracted solar beam (Snell, n=1.34)."""
    theta_w = np.arcsin(np.sin(np.radians(zen_deg)) / 1.34)
    return float(np.cos(theta_w))


def raman_comparison():
    """BING Raman correction factor vs the L23 X2/X1 ratio.

    Three model variants isolate individual approximations:
    - v1: BING defaults (Ed ratio = 1, mu_d = 0.9)
    - v2: + true Ed(lambda')/Ed(lambda) from the L23 solar spectrum
    - v3: + mu_d = cos(theta_w) per zenith (on top of v2)

    Returns
    -------
    results : dict keyed by zenith with wave_em, truth and model ratios
    """
    results = {}
    for zen in ZENITHS:
        d1 = load_scenario(1, zen)
        d2 = load_scenario(2, zen)
        wave = d1['wave']

        # Input IOPs are identical across X scenarios by construction
        assert np.allclose(d1['a'], d2['a'], rtol=1e-5)

        # Emission grid: excitation must stay inside the 350 nm edge
        # of the L23 grid -> lambda_em >= 400 nm
        keep = wave >= 400.
        wave_em = wave[keep]
        wave_ex = raman.emission_to_excitation_wavelength(wave_em)

        a_em, bb_em = d1['a'][:, keep], d1['bb'][:, keep]
        a_ex = interp_scenes(wave, d1['a'], wave_ex)
        bb_ex = interp_scenes(wave, d1['bb'], wave_ex)
        bb_R = raman.raman_backscattering_coeff(wave_ex)

        # Ed(0+) is a sky property: identical across scenes. Collapse.
        Ed = d1['Ed'].mean(axis=0)
        assert (d1['Ed'].std(axis=0) / Ed).max() < 1e-3
        f_Ed = interp1d(wave, Ed, kind='linear')
        Ed_ratio = f_Ed(wave_ex) / f_Ed(wave_em)

        # Truth: above-surface ratio, exactly how BING applies the factor
        truth = d2['Rrs'][:, keep] / d1['Rrs'][:, keep]

        v1 = bing_rrs.calc_raman_correction_factor(
            a_em, bb_em, a_ex, bb_ex, bb_R)
        v2 = bing_rrs.calc_raman_correction_factor(
            a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=Ed_ratio)
        v3 = bing_rrs.calc_raman_correction_factor(
            a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=Ed_ratio,
            mu_d=mu_d_of_zenith(float(zen)))

        results[zen] = dict(wave_em=wave_em, truth=truth,
                            v1=v1, v2=v2, v3=v3)
    return results


# ----------------------------------------------------------------------
# Fluorescence comparison
# ----------------------------------------------------------------------

def fluorescence_comparison():
    """BING additive fluorescence Rrs vs the L23 X4 - X2 difference.

    Uses BING defaults (phi_C = 0.02 -- the same value HydroLight used
    for L23) with the true L23 solar spectrum, for both the single- and
    double-Gaussian emission shapes.
    """
    results = {}
    for zen in ZENITHS:
        d2 = load_scenario(2, zen)
        d4 = load_scenario(4, zen)
        wave = d2['wave']

        # Emission grid: the red band; excitation grid: BING's 370-690
        em = (wave >= 650.) & (wave <= 750.)
        ex = (wave >= 370.) & (wave <= 690.)
        wave_em, wave_ex = wave[em], wave[ex]

        Ed = d2['Ed'].mean(axis=0)          # sky property (see above)
        Ed_ex, Ed_em = Ed[ex], Ed[em]

        common = dict(
            wavelength=wave_em,
            a_em=d2['a'][:, em], bb_em=d2['bb'][:, em],
            a_ex=d2['a'][:, ex], bb_ex=d2['bb'][:, ex],
            aph_ex=d2['aph'][:, ex],
            wavelength_ex=wave_ex, Ed_ex=Ed_ex, Ed_em=Ed_em,
            phi_C=0.02)

        single = bing_rrs.calc_Rrs_fluorescence(
            double_gaussian=False, **common)
        double = bing_rrs.calc_Rrs_fluorescence(
            double_gaussian=True, **common)

        truth = d4['Rrs'][:, em] - d2['Rrs'][:, em]

        results[zen] = dict(wave_em=wave_em, truth=truth,
                            single=single, double=double,
                            Rrs4_em=d4['Rrs'][:, em],
                            aph440=d2['aph'][:, np.argmin(
                                np.abs(wave - 440.))])
    return results


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def med_band(ax, x, y2d, color, label, ls='-'):
    """Plot the scene median with a 16-84 percentile band."""
    lo, med, hi = np.nanpercentile(y2d, [16, 50, 84], axis=0)
    ax.plot(x, med, color=color, ls=ls, lw=2, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)


def fig_impact(ram, flu):
    """Fig 1: how large each inelastic process is in the L23 truth."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for zen in ZENITHS:
        r = ram[zen]
        med_band(ax, r['wave_em'], 100. * (r['truth'] - 1.),
                 C_ZEN[zen], f'{int(zen)}°')
    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel(r'Raman: $100\,(R_{rs}^{X2}/R_{rs}^{X1} - 1)$  [%]')
    ax.set_title('Raman contribution (L23 truth)')
    ax.legend(title='solar zenith', frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for zen in ZENITHS:
        f = flu[zen]
        med_band(ax, f['wave_em'], 100. * f['truth'] / f['Rrs4_em'],
                 C_ZEN[zen], f'{int(zen)}°')
    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel(r'fluorescence: $100\,\Delta R_{rs} / R_{rs}^{X4}$  [%]')
    ax.set_title('Chl-a fluorescence contribution (L23 truth)')
    ax.legend(title='solar zenith', frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_l23_inelastic_impact.png'),
                **FIG_KW)
    plt.close(fig)


def fig_raman(ram):
    """Fig 2: BING Raman correction vs L23 truth, per variant/zenith."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)

    for j, zen in enumerate(ZENITHS):
        r = ram[zen]
        w = r['wave_em']

        ax = axes[0, j]
        med_band(ax, w, r['truth'], C_TRUTH, 'L23 truth (X2/X1)')
        med_band(ax, w, r['v1'], C_V1,
                 'BING default (flat Ed, $\\mu_d$=0.9)', ls='--')
        med_band(ax, w, r['v2'], C_V2, '+ true Ed ratio', ls='-.')
        med_band(ax, w, r['v3'], C_V3,
                 '+ $\\mu_d(\\theta_w)$', ls=':')
        ax.set_title(f'solar zenith {int(zen)}°')
        ax.set_ylabel('Raman correction factor' if j == 0 else '')
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(frameon=False, fontsize=8)

        # Error on the Raman *increment* (factor - 1), median over scenes
        ax = axes[1, j]
        ax.axhline(0., color='k', lw=0.8)
        for key, c, ls, lab in [('v1', C_V1, '--', 'BING default'),
                                ('v2', C_V2, '-.', '+ true Ed'),
                                ('v3', C_V3, ':',
                                 '+ $\\mu_d(\\theta_w)$')]:
            err = 100. * (r[key] - 1.) / (r['truth'] - 1.) - 100.
            med_band(ax, w, err, c, lab, ls=ls)
        ax.set_ylim(-100., 100.)
        ax.set_xlabel('wavelength [nm]')
        ax.set_ylabel('increment error [%]' if j == 0 else '')
        ax.grid(alpha=0.25)

    fig.suptitle('BING Raman correction factor vs L23/HydroLight truth',
                 y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_raman_bing_vs_l23.png'),
                **FIG_KW)
    plt.close(fig)


def fig_fluor(flu):
    """Fig 3: BING fluorescence Rrs vs L23 truth."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) median spectra at 30 deg
    zen = '30'
    f = flu[zen]
    ax = axes[0]
    med_band(ax, f['wave_em'], 1e4 * f['truth'], C_TRUTH,
             'L23 truth (X4$-$X2)')
    med_band(ax, f['wave_em'], 1e4 * f['single'], C_V1,
             'BING single Gaussian', ls='--')
    med_band(ax, f['wave_em'], 1e4 * f['double'], C_V2,
             'BING double Gaussian', ls='-.')
    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel(r'$\Delta R_{rs}$  [$10^{-4}$ sr$^{-1}$]')
    ax.set_title(f'fluorescence spectrum (zenith {int(zen)}°)')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # (b) scatter at 685 nm, single Gaussian, all zeniths
    ax = axes[1]
    for zen in ZENITHS:
        f = flu[zen]
        i685 = np.argmin(np.abs(f['wave_em'] - 685.))
        ax.scatter(1e4 * f['truth'][:, i685],
                   1e4 * f['single'][:, i685],
                   s=3, alpha=0.25, color=C_ZEN[zen],
                   label=f'{int(zen)}°', rasterized=True)
    lim = ax.get_xlim()
    ax.plot(lim, lim, 'k-', lw=0.8)
    ax.set_xlabel(r'truth $\Delta R_{rs}(685)$ [$10^{-4}$ sr$^{-1}$]')
    ax.set_ylabel(r'BING $\Delta R_{rs}(685)$ [$10^{-4}$ sr$^{-1}$]')
    ax.set_title('685 nm, single Gaussian')
    leg = ax.legend(title='solar zenith', frameon=False, fontsize=8)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
        h.set_sizes([20])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(alpha=0.25)

    # (c) ratio model/truth at 685 vs aph(440) (trophic proxy)
    ax = axes[2]
    for zen in ZENITHS:
        f = flu[zen]
        i685 = np.argmin(np.abs(f['wave_em'] - 685.))
        ratio = f['single'][:, i685] / f['truth'][:, i685]
        ax.scatter(f['aph440'], ratio, s=3, alpha=0.25,
                   color=C_ZEN[zen], label=f'{int(zen)}°',
                   rasterized=True)
    ax.axhline(1., color='k', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r'$a_{ph}(440)$ [m$^{-1}$]')
    ax.set_ylabel('BING / truth at 685 nm')
    ax.set_title('amplitude ratio vs trophic state')
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_fluor_bing_vs_l23.png'),
                **FIG_KW)
    plt.close(fig)


def fig_redistribution():
    """Fig 4: the single-shift approximation vs Walrafen redistribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Raman scattering coefficient vs excitation wavelength
    ax = axes[0]
    lam = np.linspace(350., 700., 200)
    ax.plot(lam, 1e4 * raman.raman_scattering_coeff(lam), color=C_V2,
            lw=2)
    ax.axhline(1e4 * raman.B_RAMAN_488_HYDROLIGHT, color='0.5', lw=0.8,
               ls=':')
    ax.axvline(488., color='0.5', lw=0.8, ls=':')
    ax.annotate('$b_R(488) = 2.6\\times10^{-4}$ m$^{-1}$',
                xy=(492., 2.7), fontsize=9, color='0.35')
    ax.set_xlabel("excitation wavelength $\\lambda'$ [nm]")
    ax.set_ylabel("$b_R(\\lambda')$  [$10^{-4}$ m$^{-1}$]")
    ax.set_title("Raman scattering coefficient "
                 "($\\propto \\lambda'^{-5.5}$, energy units)")
    ax.grid(alpha=0.25)

    # (b) emission redistribution for 488 nm excitation
    ax = axes[1]
    lam_em = np.linspace(540., 640., 500)
    f_R = raman.wavelength_redistribution(488., lam_em)
    ax.plot(lam_em, f_R, color=C_V2, lw=2,
            label='Walrafen (1967) $f_R(488, \\lambda)$')
    lam_c = raman.excitation_to_emission_wavelength(488.)
    ax.axvline(lam_c, color=C_V1, lw=2, ls='--',
               label=f'single shift (3400 cm$^{{-1}}$): '
                     f'{lam_c:.0f} nm')
    ax.set_xlabel('emission wavelength $\\lambda$ [nm]')
    ax.set_ylabel('$f_R$  [nm$^{-1}$]')
    ax.set_title("redistribution for $\\lambda'$ = 488 nm")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_raman_redistribution.png'),
                **FIG_KW)
    plt.close(fig)


# ----------------------------------------------------------------------
# Metrics table
# ----------------------------------------------------------------------

def write_metrics(ram, flu):
    """Summary numbers quoted in the report -> rt_inelastic_metrics.csv."""
    rows = []
    for zen in ZENITHS:
        r = ram[zen]
        w = r['wave_em']
        for band in (490., 550., 660.):
            i = np.argmin(np.abs(w - band))
            t = np.median(r['truth'][:, i] - 1.)
            rows.append(dict(
                process='Raman', zenith=int(zen), wavelength=band,
                truth_pct=100. * t,
                err_default_pct=100. * (np.median(r['v1'][:, i] - 1.)
                                        / t - 1.),
                err_trueEd_pct=100. * (np.median(r['v2'][:, i] - 1.)
                                       / t - 1.),
                err_trueEd_mud_pct=100. * (np.median(r['v3'][:, i] - 1.)
                                           / t - 1.)))
        f = flu[zen]
        i685 = np.argmin(np.abs(f['wave_em'] - 685.))
        t685 = np.median(f['truth'][:, i685])
        rows.append(dict(
            process='ChlFl', zenith=int(zen), wavelength=685.,
            truth_pct=np.nan,
            err_default_pct=100. * (np.median(f['single'][:, i685])
                                    / t685 - 1.),
            err_trueEd_pct=100. * (np.median(f['double'][:, i685])
                                   / t685 - 1.),
            err_trueEd_mud_pct=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'rt_inelastic_metrics.csv'),
              index=False, float_format='%.3f')
    print(df.to_string(index=False))


# ----------------------------------------------------------------------

if __name__ == '__main__':
    print(f'L23 path: {L23_PATH}')
    ram = raman_comparison()
    flu = fluorescence_comparison()
    fig_impact(ram, flu)
    fig_raman(ram)
    fig_fluor(flu)
    fig_redistribution()
    write_metrics(ram, flu)
    print('Done. Figures written to', OUT_DIR)
