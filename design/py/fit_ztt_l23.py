#!/usr/bin/env python
"""
Refit ZTT's µ∞ (Eq. 8) and µd (Eq. 14/17) to L23 -- M7 task 3, ``rt_fit_ztt.md``.

Three fits/diagnostics, all deterministic (linear least squares, no optimizer):

(a) **µ∞, Equation (8)** -- sixteen coefficients m1..m16, fitted to µ∞ from the
    asymptotic RTE eigenproblem (:mod:`robust.rt.asymptotic`; Q13 answer (a))
    over every train-split (scene, λ). All points carry equal weight: the
    eigen µ∞ is defined everywhere, so the plan's down-weighted "extrapolated"
    points no longer exist. Residuals are relative (each row divided by its
    target). The (X, L) = (log10 bb/a, log10 η_bb) box of the training points
    is recorded for clamping (Q3).
(b) **µd** -- Equation (17)'s eight M*_d coefficients fitted *jointly* with the
    atmospheric correction c(θs') = 1 + c1 u + c2 u², u = 1 − cos θs' (Q10b),
    to µd(0⁻) from the L23 profiles at all three zeniths. Exactly linear: with
    t = µd / M⁺_d(θ) and g_θ = 1/c(θ), the residual ``M*_d(x)/t − g_θ`` is
    linear in (m*, g_30, g_60), with g_0 = 1 fixed by c(0) = 1.
(c) **Ψ_KLu diagnostic** (Q11, no refit) -- KLu(0⁻) µ∞ / a at ψ = 180°, ~158°,
    ~140° against TT's 1 + F(ψ), with the L23-fit and the TT2017 µ∞.

Held-out (scene-split) errors are evaluated **through the functions task 4
ships** (``ztt.mu_infinity``, ``ztt.mu_d`` with ``box``/``md_plus_corr``), and
cross-checked against this script's own design matrices -- the task-3 gate.

Writes ``design/ztt_l23/``: ``coefficients.json``, ``metrics.md``,
``probes.json`` (points the test suite re-evaluates through ``ztt``), and
``fig_mu_inf_residuals.png`` / ``fig_mu_d_residuals.png``. The eigen µ∞ is
cached next to the data (``$OS_COLOR/Loisel2023/derived/``; ~4 min to build).

    python design/py/fit_ztt_l23.py              # fit, write
    python design/py/fit_ztt_l23.py --dry-run    # fit, report, write nothing
    python design/py/fit_ztt_l23.py --check      # refit, compare with files on disk
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # fits and metrics in float64

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robust.rt import asymptotic as A  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import l23 as L  # noqa: E402

OUT = ROOT / "design" / "ztt_l23"

#: Bumped when the eigen-µ∞ definition changes, so a stale cache is refused.
EIGEN_VERSION = 2  # v2: δ-truncation 5° → 2° (task 3 convergence study)

#: The three L23 zeniths are above-water sun angles (θs').
ZENITHS = np.asarray(L.ZENITHS, dtype=float)


# ---------------------------------------------------------------- inputs --
def load_iops():
    """Per-(scene, λ) IOPs from the zenith-0 surface file (identical across
    zeniths -- record §3.2.1), float64."""
    from ocpy.hydrolight import loisel23

    ds = loisel23.load_ds(L.ELASTIC_X, 0)
    get = {
        v: np.asarray(ds[v].data, dtype=float)
        for v in ("a", "b", "bb", "bnw", "bbnw", "bph", "bd")
    }
    ds.close()
    bb_w = get["bb"] - get["bbnw"]
    return {
        "a": get["a"],
        "bb": get["bb"],
        "x": get["bb"] / get["a"],
        "eta": bb_w / get["bb"],
        "b_comp": np.stack([get["b"] - get["bnw"], get["bph"], get["bd"]], -1),
    }


def eigen_cache_path() -> Path:
    """Where the eigen µ∞ is cached: next to the profile product."""
    return L.default_profile_product_path().parent / (
        f"l23_mu_inf_eigen_v{EIGEN_VERSION}.npz"
    )


def eigen_mu_inf(iops, *, recompute=False) -> np.ndarray:
    """µ∞ for every (scene, λ) from the asymptotic eigenproblem, cached."""
    path = eigen_cache_path()
    settings = {
        "version": EIGEN_VERSION,
        "n_streams": A.N_STREAMS,
        "truncation_deg": A.TRUNCATION_DEG,
        "n_phi": A.N_PHI,
        "water_depolarization": Z.WATER_DEPOLARIZATION,
        "ff_B": list(L.L23_FF_B),
        "ff_convention": "Mobley et al. 2002, n = 1.01 + 0.1542 (mu - 3)",
    }
    if path.is_file() and not recompute:
        with np.load(path) as data:
            if json.loads(str(data["settings"])) == settings:
                return np.asarray(data["mu_inf"])
        print(f"eigen cache {path} has other settings; recomputing")
    print("computing eigen µ∞ for every (scene, λ) -- a few minutes")
    mu, weight, kernels = A.l23_kernels()
    mu_inf = A.mu_infinity_eigen(iops["a"], iops["b_comp"], mu, weight, kernels)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, mu_inf=mu_inf, settings=np.asarray(json.dumps(settings)))
    os.replace(tmp, path)
    return mu_inf


def scene_split():
    """Train/test scene masks from the record's seeded split (SPLIT_SEED)."""
    batch = L.load_batch(validate=False)
    test_scenes = L.make_splits(batch).test_scenes
    test = np.zeros(L.N_SCENES, dtype=bool)
    test[test_scenes] = True
    return ~test, test


# ------------------------------------------------------- design matrices --
def mu_inf_design(X, Lg):
    """Equation (8)'s 16 columns in ``ztt.mu_infinity``'s coefficient order:
    m[4j + i] multiplies L^(3−i) X^(3−j)."""
    return np.stack(
        [Lg ** (3 - i) * X ** (3 - j) for j in range(4) for i in range(4)], axis=-1
    )


def md_star_design(X, Lg):
    """Equation (17)'s 8 columns in ``ztt.Md_star``'s order."""
    one = np.ones_like(X)
    return np.stack([Lg * X**3, X**3, Lg * X**2, X**2, Lg * X, X, Lg, one], axis=-1)


def fit_mu_inf(X, Lg, target):
    """Relative-residual LSQ for m1..m16."""
    D = mu_inf_design(X, Lg) / target[:, None]
    coeffs, *_ = np.linalg.lstsq(D, np.ones_like(target), rcond=None)
    return coeffs, float(np.linalg.cond(D))


def fit_mu_d(X, Lg, theta, t):
    """Joint linear LSQ for M*_d (8) and g_θ = 1/c(θ) at the non-zero zeniths.

    Rows: ``M*_d(x)/t − g_θ = 0`` (θ ≠ 0) and ``M*_d(x)/t = 1`` (θ = 0).
    """
    D = md_star_design(X, Lg) / t[:, None]
    others = [z for z in ZENITHS if z != 0.0]
    G = np.stack([-(theta == z).astype(float) for z in others], axis=-1)
    rhs = (theta == 0.0).astype(float)
    sol, *_ = np.linalg.lstsq(np.concatenate([D, G], axis=1), rhs, rcond=None)
    md_star, g = sol[:8], sol[8:]
    c_nodes = 1.0 / g
    u = 1.0 - np.cos(np.deg2rad(np.asarray(others)))
    c1, c2 = np.linalg.solve(np.stack([u, u**2], axis=-1), c_nodes - 1.0)
    return md_star, (float(c1), float(c2)), dict(zip(others, c_nodes, strict=True))


# ----------------------------------------------------------------- stats --
def output_range(values, *, cosine=False):
    """A smooth output bound for a fitted polynomial: the training values'
    range padded by five ``OUT_CLAMP_WIDTH`` (so the guard is invisible on the
    data, < 1e-5 relative). An average cosine (``cosine=True``) is also capped
    below 1; M*_d is only a factor of one and is not."""
    pad = 5 * Z.OUT_CLAMP_WIDTH
    lo = float(np.min(values)) - pad
    hi = float(np.max(values)) + pad
    if cosine:
        hi = min(hi, 1.0 - Z.OUT_CLAMP_WIDTH)
    return [lo, hi]


def pct(values, qs=(1, 16, 50, 84, 99)):
    """Percentiles of a flat array, as a list of floats."""
    return [float(v) for v in np.nanpercentile(np.ravel(values), qs)]


def describe(rel):
    """``(median |err|, signed 16/50/84/99% of err)`` for relative errors."""
    return {
        "mape": float(np.nanmedian(np.abs(rel))),
        "mean_abs": float(np.nanmean(np.abs(rel))),
        "signed_pct_1_16_50_84_99": pct(rel),
    }


def run(recompute_eigen=False):
    """Do everything; return (coefficients, metrics, probes, arrays for figures)."""
    iops = load_iops()
    train, test = scene_split()
    X, Lg = np.log10(iops["x"]), np.log10(iops["eta"])
    mu_eig = eigen_mu_inf(iops, recompute=recompute_eigen)
    product = L.load_profile_product(L.default_profile_product_path())
    tr = np.broadcast_to(train[:, None], X.shape)
    te = np.broadcast_to(test[:, None], X.shape)

    # ---------------------------------------------------------------- (a) µ∞
    m16, cond = fit_mu_inf(X[tr], Lg[tr], mu_eig[tr])
    box = [
        float(X[tr].min()),
        float(X[tr].max()),
        float(Lg[tr].min()),
        float(Lg[tr].max()),
    ]
    mu_range = output_range(mu_eig[tr], cosine=True)
    mu_fit = np.asarray(
        Z.mu_infinity(iops["x"], iops["eta"], tuple(m16), box=box, out_range=mu_range)
    )
    mu_tt = np.asarray(Z.mu_infinity_tt2017(iops["x"], iops["eta"]))
    # Gate: the shipped evaluator reproduces the script's own design matrix
    # wherever neither guard acts -- inside the input box by 10 widths, and the
    # bare polynomial inside its output range by 10 widths.
    bare = mu_inf_design(X, Lg) @ m16
    out_w = 10 * Z.OUT_CLAMP_WIDTH
    inside = (
        (X > box[0] + 10 * Z.CLAMP_WIDTH)
        & (X < box[1] - 10 * Z.CLAMP_WIDTH)
        & (Lg > box[2] + 10 * Z.CLAMP_WIDTH)
        & (Lg < box[3] - 10 * Z.CLAMP_WIDTH)
    )
    unguarded = inside & (bare > mu_range[0] + out_w) & (bare < mu_range[1] - out_w)
    agree_mu = float(np.max(np.abs(mu_fit[unguarded] / bare[unguarded] - 1)))
    guard_effect = {
        "fraction_heldout_changed_gt_1e-4": float(
            np.mean(np.abs(mu_fit / bare - 1)[te] > 1e-4)
        ),
        "heldout_mape_bare": float(np.median(np.abs(bare[te] / mu_eig[te] - 1))),
        "heldout_mape_guarded": float(np.median(np.abs(mu_fit[te] / mu_eig[te] - 1))),
    }
    asym = product.flag == L.FLAG_ASYMPTOTIC

    # ---------------------------------------------------------------- (b) µd
    md0 = product.mu_d0.astype(float)  # (zenith, scene, wave)
    theta = np.broadcast_to(ZENITHS[:, None, None], md0.shape)
    plus = np.asarray(Z.Md_plus(theta))
    t = md0 / plus
    tr3 = np.broadcast_to(tr, md0.shape)
    X3, L3 = np.broadcast_to(X, md0.shape), np.broadcast_to(Lg, md0.shape)
    md8, corr, c_nodes = fit_mu_d(X3[tr3], L3[tr3], theta[tr3], t[tr3])
    md_star_range = output_range(md_star_design(X[tr], Lg[tr]) @ md8)
    mud_fit = np.asarray(
        Z.mu_d(
            theta,
            iops["x"],
            iops["eta"],
            md_star_coeffs=tuple(md8),
            md_plus_corr=corr,
            box=box,
            md_star_range=md_star_range,
        )
    )
    mud_tt = np.asarray(Z.mu_d(theta, iops["x"], iops["eta"]))
    own_d = (
        (md_star_design(X3[:, unguarded], L3[:, unguarded]) @ md8)
        * plus[:, unguarded]
        * np.asarray(Z.md_plus_correction(theta[:, unguarded], corr))
    )
    agree_mud = float(np.max(np.abs(mud_fit[:, unguarded] / own_d - 1)))

    # --------------------------------------------------------- (c) Ψ_KLu check
    theta_w = np.asarray(Z.in_water_zenith(ZENITHS))
    psi = 180.0 - theta_w
    psi_tt = np.asarray(Z.psi_KLu(psi))
    psi_rows = []
    for k, z in enumerate(ZENITHS):
        klu = product.KLu0[k].astype(float)
        row = {"theta_s": float(z), "psi": float(psi[k]), "TT_1+F": float(psi_tt[k])}
        for name, mu_inf in (("L23", mu_fit), ("TT2017", mu_tt), ("eigen", mu_eig)):
            obs = klu * mu_inf / iops["a"]
            row[f"measured_{name}_pct_16_50_84"] = pct(obs[te], (16, 50, 84))
        psi_rows.append(row)

    # ------------------------------------------------------------ assemble
    coefficients = {
        "about": "ZTT refit to L23 (M7 task 3, rt_fit_ztt.md). Report as "
        "'ZTT (L23 mu_inf)'. mu_inf target: asymptotic RTE eigenproblem "
        "on L23 IOPs, FF Mobley-2002 convention; reads +1.4% vs "
        "HydroLight profiles where those are asymptotic.",
        "mu_inf_coeffs_m1_m16": [float(v) for v in m16],
        "md_star_coeffs_m1_m8": [float(v) for v in md8],
        "md_plus_corr_c1_c2": list(corr),
        "clamp_box_X_lo_hi_L_lo_hi": box,
        "mu_inf_out_range": mu_range,
        "md_star_out_range": md_star_range,
        "fit": {
            "split": (
                f"scenes, SPLIT_SEED={L.SPLIT_SEED}, test_fraction={L.TEST_FRACTION}"
            ),
            "n_train_scenes": int(train.sum()),
            "n_test_scenes": int(test.sum()),
            "mu_inf_design_condition": cond,
            "c_at_nodes": {str(int(k)): float(v) for k, v in c_nodes.items()},
        },
    }
    metrics = {
        "mu_inf_heldout_vs_eigen": {
            "L23_fit": describe(mu_fit[te] / mu_eig[te] - 1),
            "TT2017": describe(mu_tt[te] / mu_eig[te] - 1),
        },
        "mu_inf_train_vs_eigen_L23_fit": describe(mu_fit[tr] / mu_eig[tr] - 1),
        "mu_inf_vs_profiles_asymptotic": {
            "n": int(asym.sum()),
            "eigen": describe(mu_eig[asym] / product.mu_inf[asym] - 1),
            "L23_fit": describe(mu_fit[asym] / product.mu_inf[asym] - 1),
            "TT2017": describe(mu_tt[asym] / product.mu_inf[asym] - 1),
        },
        "mu_d_heldout": {
            f"{int(z)}": {
                "L23_fit": describe(mud_fit[k][te] / md0[k][te] - 1),
                "TT": describe(mud_tt[k][te] / md0[k][te] - 1),
            }
            for k, z in enumerate(ZENITHS)
        },
        "psi_KLu": psi_rows,
        "gate_evaluator_agreement": {"mu_inf": agree_mu, "mu_d": agree_mud},
        "mu_inf_guard_effect": guard_effect,
        "fraction_heldout_outside_box": float(
            np.mean(((X < box[0]) | (X > box[1]) | (Lg < box[2]) | (Lg > box[3]))[te])
        ),
    }
    # Probes: a fixed spread of points, re-evaluated through ztt by the tests.
    rng = np.random.default_rng(L.SPLIT_SEED)
    pick = rng.choice(X.size, 12, replace=False)
    s_idx, w_idx = np.unravel_index(pick, X.shape)
    probes = [
        {
            "scene": int(s),
            "wave_index": int(w),
            "bb_over_a": float(iops["x"][s, w]),
            "eta_bb": float(iops["eta"][s, w]),
            "mu_inf_L23": float(mu_fit[s, w]),
            "mu_d_L23": {
                str(int(z)): float(mud_fit[k, s, w]) for k, z in enumerate(ZENITHS)
            },
        }
        for s, w in zip(s_idx, w_idx, strict=True)
    ]
    figs = dict(
        X=X,
        Lg=Lg,
        te=te,
        mu_eig=mu_eig,
        mu_fit=mu_fit,
        mu_tt=mu_tt,
        md0=md0,
        mud_fit=mud_fit,
        mud_tt=mud_tt,
        box=box,
    )
    return coefficients, metrics, probes, figs


# ---------------------------------------------------------------- report --
def fmt(d):
    """One table cell: median |err| and the signed 16–84% band, in percent."""
    q = d["signed_pct_1_16_50_84_99"]
    return (
        f"{100 * d['mape']:.2f} | {100 * q[1]:+.2f} / {100 * q[2]:+.2f} / "
        f"{100 * q[3]:+.2f} | {100 * q[4]:+.2f}"
    )


def report(coefficients, metrics) -> str:
    m = metrics
    lines = [
        "# ZTT refit to L23 — µ∞ and µd (M7 task 3)",
        "",
        "*Generated by `design/py/fit_ztt_l23.py`. Do not edit by hand.*",
        "",
        "Errors are relative, in %: the median of |error|, the signed 16 / 50 / 84%",
        "percentiles, and the 99th percentile. Held-out = the 20% scene split",
        f"({coefficients['fit']['n_test_scenes']} scenes × 81 λ).",
        "",
        "## µ∞ (Equation 8)",
        "",
        "Target: the asymptotic eigenproblem (`robust.rt.asymptotic`).",
        "",
        "| model vs eigen µ∞, held out | median abs | 16 / 50 / 84 | 99 |",
        "|---|---|---|---|",
        "| **ZTT (L23 µ∞)**, Eq. (8) refit | "
        + fmt(m["mu_inf_heldout_vs_eigen"]["L23_fit"])
        + " |",
        "| ZTT (TT2017 µ∞) | " + fmt(m["mu_inf_heldout_vs_eigen"]["TT2017"]) + " |",
        "",
        "TT quote 0.19–3.5% (mean absolute, by η_bb) for their own Eq. (8) fit.",
        f"L23 fit mean |err| on train: "
        f"{100 * m['mu_inf_train_vs_eigen_L23_fit']['mean_abs']:.2f}%; "
        f"held out: {100 * m['mu_inf_heldout_vs_eigen']['L23_fit']['mean_abs']:.2f}%.",
        "",
        f"Against the {m['mu_inf_vs_profiles_asymptotic']['n']} points where the "
        "HydroLight profiles are asymptotic (all splits):",
        "",
        "| vs profile µ∞ | median abs | 16 / 50 / 84 | 99 |",
        "|---|---|---|---|",
        *[
            f"| {k} | {fmt(v)} |"
            for k, v in m["mu_inf_vs_profiles_asymptotic"].items()
            if k != "n"
        ],
        "",
        "## µd (Equations 14, 17) — held out, vs L23 µd(0⁻)",
        "",
        "| θs′ | model | median abs | 16 / 50 / 84 | 99 |",
        "|---|---|---|---|---|",
    ]
    for z, d in m["mu_d_heldout"].items():
        lines.append(f"| {z}° | **L23 refit** | {fmt(d['L23_fit'])} |")
        lines.append(f"| {z}° | TT (Table A2) | {fmt(d['TT'])} |")
    c1, c2 = coefficients["md_plus_corr_c1_c2"]
    nodes = coefficients["fit"]["c_at_nodes"]
    lines += [
        "",
        f"Atmospheric correction c(θs′) = 1 + {c1:.5f} u + {c2:.5f} u², "
        f"u = 1 − cos θs′; c(30°) = {nodes['30']:.4f}, c(60°) = {nodes['60']:.4f}. "
        "TT quote < 1% for the full µd expression.",
        "",
        "## Ψ_KLu diagnostic (no refit) — KLu(0⁻) µ∞ / a vs TT's 1 + F(ψ), held out",
        "",
        "| θs′ | ψ | TT 1 + F(ψ) | with L23 µ∞: 16 / 50 / 84 "
        "| with TT2017 µ∞ | with eigen µ∞ |",
        "|---|---|---|---|---|---|",
    ]
    for r in m["psi_KLu"]:

        def band(q):
            return " / ".join(f"{v:.3f}" for v in q)

        lines.append(
            f"| {r['theta_s']:.0f}° | {r['psi']:.1f}° | {r['TT_1+F']:.3f} | "
            f"{band(r['measured_L23_pct_16_50_84'])} | "
            f"{band(r['measured_TT2017_pct_16_50_84'])} | "
            f"{band(r['measured_eigen_pct_16_50_84'])} |"
        )
    g = m["gate_evaluator_agreement"]
    lines += [
        "",
        "## Gate and housekeeping",
        "",
        "- `ztt` evaluators vs this script's design matrices (where no guard "
        "acts): "
        f"max relative difference µ∞ {g['mu_inf']:.1e}, µd {g['mu_d']:.1e}.",
        "- Guards (input box + output range) change held-out µ∞ by > 1e-4 on "
        f"{100 * m['mu_inf_guard_effect']['fraction_heldout_changed_gt_1e-4']:.1f}% "
        "of points; held-out median |err| "
        f"{100 * m['mu_inf_guard_effect']['heldout_mape_bare']:.3f}% bare, "
        f"{100 * m['mu_inf_guard_effect']['heldout_mape_guarded']:.3f}% guarded.",
        "- Clamp box (X, L) = "
        + ", ".join(f"{v:.4f}" for v in coefficients["clamp_box_X_lo_hi_L_lo_hi"])
        + f"; {100 * m['fraction_heldout_outside_box']:.3f}% of held-out points "
        "fall outside it.",
        f"- Eq. (8) relative design-matrix condition number: "
        f"{coefficients['fit']['mu_inf_design_condition']:.3g}.",
        "",
    ]
    return "\n".join(lines)


def figures(figs, out: Path) -> None:
    """Median signed relative error in (X, L) bins, held out: µ∞ and µd at 60°."""
    from run_validation import apply_house_style, save_figure_atomically

    plt = apply_house_style()
    te = figs["te"]
    x, y = figs["X"][te], figs["Lg"][te]
    bins = [np.linspace(-4.0, 0.0, 33), np.linspace(-1.8, 0.0, 28)]

    def median_map(err):
        sums = np.full((len(bins[0]) - 1, len(bins[1]) - 1), np.nan)
        ix = np.clip(np.digitize(x, bins[0]) - 1, 0, sums.shape[0] - 1)
        iy = np.clip(np.digitize(y, bins[1]) - 1, 0, sums.shape[1] - 1)
        for i in range(sums.shape[0]):
            for j in range(sums.shape[1]):
                sel = (ix == i) & (iy == j)
                if sel.sum() >= 5:
                    sums[i, j] = np.median(err[sel])
        return sums

    panels = {
        "fig_mu_inf_residuals.png": (
            ("ZTT (L23 µ∞) − eigen", figs["mu_fit"][te] / figs["mu_eig"][te] - 1),
            ("ZTT (TT2017 µ∞) − eigen", figs["mu_tt"][te] / figs["mu_eig"][te] - 1),
            "µ∞ relative error, held out (%)",
        ),
        "fig_mu_d_residuals.png": (
            (
                "µd L23 refit − L23, θs′ = 60°",
                figs["mud_fit"][2][te] / figs["md0"][2][te] - 1,
            ),
            (
                "µd TT (Table A2) − L23, θs′ = 60°",
                figs["mud_tt"][2][te] / figs["md0"][2][te] - 1,
            ),
            "µd relative error, held out (%)",
        ),
    }
    for name, (left, right, label) in panels.items():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
        maps = [median_map(e) for _, e in (left, right)]
        lim = max(1.0, float(np.nanpercentile(np.abs(100 * np.stack(maps)), 98)))
        for ax, (title, _), mp in zip(axes, (left, right), maps, strict=True):
            ax.grid(False)
            im = ax.pcolormesh(
                bins[0],
                bins[1],
                100 * mp.T,
                cmap="RdBu_r",
                vmin=-lim,
                vmax=lim,
                shading="flat",
            )
            ax.set_title(title, loc="left", fontsize=10.5)
            ax.set_xlabel("log10(bb / a)")
            box = figs["box"]
            ax.plot(
                [box[0], box[1], box[1], box[0], box[0]],
                [box[2], box[2], box[3], box[3], box[2]],
                color="#5c5c5c",
                lw=0.8,
                ls="--",
            )
        axes[0].set_ylabel("log10(η_bb)")
        fig.colorbar(im, ax=axes, label=label)
        save_figure_atomically(fig, out / name)
        plt.close(fig)


def outputs(coefficients, metrics, probes) -> dict[str, str]:
    """The text files, exactly as written (so --check can compare bytes)."""
    dump = {"indent": 2, "sort_keys": True}
    return {
        "coefficients.json": json.dumps(coefficients, **dump) + "\n",
        "metrics.json": json.dumps(metrics, **dump) + "\n",
        "probes.json": json.dumps(probes, **dump) + "\n",
        "metrics.md": report(coefficients, metrics),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="write nothing")
    parser.add_argument(
        "--check",
        action="store_true",
        help="refit and compare byte-for-byte with design/ztt_l23/",
    )
    parser.add_argument("--recompute-eigen", action="store_true")
    args = parser.parse_args(argv)

    coefficients, metrics, probes, figs = run(args.recompute_eigen)
    files = outputs(coefficients, metrics, probes)
    print(files["metrics.md"])
    if args.check:
        stale = [n for n, text in files.items() if (OUT / n).read_text() != text]
        print("check:", "identical" if not stale else f"DIFFERS: {stale}")
        return 1 if stale else 0
    if args.dry_run:
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT / name).write_text(text)
    figures(figs, OUT)
    print(f"wrote {sorted(files)} and figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
