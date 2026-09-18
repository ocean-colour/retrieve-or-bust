"""
The CDOM-fluorescence v1 gate, items 3–5 — CDOM design §5, one line per test.

The M5 companion to ``test_inelastic_validation.py``: this file is where the
CDOM design's truth-less acceptance is *stated and stays stated*, one numbered
test per remaining §5 line (items 1–2 — the off-state bit-identity and the
implementation-correctness pins — were earned in ``test_inelastic_types.py``
and ``test_cdom_fl.py`` at tasks 3–5 and are not re-spelled here). Every
metric goes through :mod:`robust.rt.validation`'s protocol functions,
including the new :func:`~robust.rt.validation.cdom_gradient_report` — a
separate report with its own ``CDOM_FD_STEPS``, so the shipped M4 gate's
``INELASTIC_FD_STEPS`` refusal rule stays untouched.

**What these gates are, honestly**: implementation and plausibility checks,
not validation. No CDOM-fl truth exists (L23 omits it by design; BING never
implemented it), so §5.3's band is "reported and gated loosely" — the design's
own words — and the quantitative M6 gate stays deferred until the HydroLight
"X4 vs X4 + CDOM-fl" runs land. Until then the term is "Hawes-consistent and
plausible", never "validated" (design §8).

Skips: the full-release lines need ``$OS_COLOR``; the gradient and speed
lines need the committed M3 heads (the composed forward carries them).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import cdom_fl as CF
from robust.rt import hybrid as H
from robust.rt import inelastic_corr as IC
from robust.rt import validation as V
from robust.rt.data import l23 as L
from robust.rt.types import CDOMFl, Geometry, Inelastic, IOPs, PhaseParams
from robust.tests.conftest import needs_l23_inelastic, needs_weights

#: The blue-green band the §5.3 fraction is averaged over, nm. A band mean
#: rather than a single wavelength, so the statistic is not hostage to one
#: grid point; 440–500 nm is where the design's "a few % in the blue-green"
#: language points and where the Hawes emission (peaking ~430–570 nm for
#: 350–490 nm excitation) overlaps the elastic blue-green.
PLAUSIBILITY_BAND = (440.0, 500.0)

#: §5.3's loose bands, stated as what they are — a *loose characterization*
#: of "a few % of Rrs for CDOM-rich scenes, ≲ 1 % oligotrophic", not a tight
#: physics claim (there is no truth to tighten them against). Measured on the
#: full release (task 7): top decile mean 4.16 %, bottom 0.30 %.
CDOM_RICH_BAND = (0.003, 0.15)
OLIGOTROPHIC_MAX = 0.012

#: The CDOM speed gate's bound — **machine-anchored, by decision** (Q&A CQ3,
#: 2026-08-30: "Go ahead and rescope the budget and make note that it is
#: machine-anchored"). The design's original bar was the shared 2× elastic
#: budget (``validation.INELASTIC_GATE_SPEED`` — deliberately NOT reused here:
#: that constant gates the *shipped* M4 Raman+Chl-fl record and must not
#: move), but on JXP's Mac the everything-on forward reproducibly measures
#: 2.26–2.34× (quiet; ~2.45× under concurrent load) — mostly baseline drift:
#: the shipped R+F model itself measures ~1.9× here vs its M4-recorded 1.59×,
#: and the CDOM marginal is ~0.4× elastic. Like the strict SHA-256 pins, this
#: number characterizes *this machine's* measured behavior, not a portable
#: physical requirement — a different machine (e.g. the tank server that
#: anchored the M4 speed record) may reproduce a different, possibly tighter,
#: ratio. 2.6 = the measured quiet-machine medians (2.30–2.34 on 2026-08-30)
#: plus headroom for load; design §5 item 5 carries the same rescoped bar.
CDOM_GATE_SPEED_MACHINE_ANCHORED = 2.6


@pytest.fixture(scope="module")
def cdom_release():
    """The full L23 inelastic batch, its elastic hybrid Rrs, and unit K_cdom.

    Module-scoped like ``test_inelastic_validation.full_release``: the netCDF
    load and the full-batch passes are shared by the plausibility and speed
    lines. ``a_cdom`` is present by construction — task 2 wired the ``ag``
    extraction into :func:`robust.rt.data.l23.load_inelastic_batch`.
    """
    batch = L.load_inelastic_batch()
    rrs_elastic = np.asarray(
        H.forward(
            batch.iops,
            batch.phase_params,
            batch.geometry,
            batch.wave,
            "hybrid",
            check_domain=False,
        )
    )
    k_cdom = np.asarray(CF.cdom_kernel(batch.iops, batch.geometry, batch.wave))
    return batch, rrs_elastic, k_cdom


@needs_l23_inelastic
def test_cdom_gate_3_plausibility_band(cdom_release):
    """**CDOM §5 item 3**: the literature-plausibility band, loosely gated.

    ``K_cdom`` (unit ``scale``) as a fraction of the elastic hybrid ``Rrs``,
    averaged over :data:`PLAUSIBILITY_BAND`, stratified into a_cdom(440)
    deciles via :func:`validation.quantile_bin_labels` (the a_ph(440)-decile
    idiom of ``run_validation.py``). Gated: the CDOM-rich top decile lands in
    the loose "a few percent" band, the bottom (oligotrophic) decile stays
    ≲ 1 %, and the per-decile mean fraction is **strictly increasing across
    all ten deciles** — the full monotone sequence, which held on the release
    and is stronger than a top-vs-bottom comparison.

    Measured table (full release, 9960 scenes, mean 440–500 nm fraction per
    a_cdom(440) decile, 2026-08-30) — reported per the design's "reported and
    gated loosely", with the loose gates *around* these values, not at them::

        decile   a_cdom440 med   mean frac   median frac
             0          0.0016     0.00297       0.00298
             1          0.0027     0.00474       0.00461
             2          0.0037     0.00632       0.00622
             3          0.0048     0.00781       0.00771
             4          0.0059     0.00950       0.00944
             5          0.0074     0.01145       0.01131
             6          0.0095     0.01401       0.01414
             7          0.0125     0.01737       0.01721
             8          0.0183     0.02304       0.02246
             9          0.0381     0.04162       0.03649

    (Top/bottom split is zenith-stable: 0°/30°/60° give top-decile means
    4.20/4.16/4.12 %. At 460 nm alone: bottom 0.23 %, top 3.60 %.)
    """
    batch, rrs_elastic, k_cdom = cdom_release
    wave = np.asarray(batch.wave)
    band = (wave >= PLAUSIBILITY_BAND[0]) & (wave <= PLAUSIBILITY_BAND[1])
    fraction = (k_cdom[:, band] / rrs_elastic[:, band]).mean(axis=1)

    i440 = int(np.abs(wave - 440.0).argmin())
    labels, _ = V.quantile_bin_labels(np.asarray(batch.iops.a_cdom)[:, i440], n_bins=10)
    decile_means = np.array([fraction[labels == d].mean() for d in range(10)])

    rich, oligo = decile_means[-1], decile_means[0]
    assert CDOM_RICH_BAND[0] <= rich <= CDOM_RICH_BAND[1], (
        f"CDOM-rich top decile: mean blue-green fraction {rich:.4f} outside "
        f"the loose 'a few percent' band {CDOM_RICH_BAND}"
    )
    assert oligo <= OLIGOTROPHIC_MAX, (
        f"oligotrophic bottom decile: mean fraction {oligo:.4f} exceeds the "
        f"loose <~1% characterization ({OLIGOTROPHIC_MAX})"
    )
    assert np.all(np.diff(decile_means) > 0.0), (
        "the mean fraction is not strictly increasing across a_cdom(440) "
        f"deciles: {np.array2string(decile_means, precision=5)}"
    )


@needs_weights
def test_cdom_gate_4_gradients_all_inputs(jax_x64, l23_small_inelastic_batch):
    """**CDOM §5 item 4**: FD agreement for all inputs incl. scale and a_cdom.

    :func:`validation.cdom_gradient_report` — the M2/M3 protocol (float64,
    per-variable steps, θ_s = 35°, off the piecewise-linear Ed anchors where
    the derivative is one-sided) — through the composed forward with **all
    three** inelastic processes on and the trained Raman/fl heads carried,
    all eight variables ≤ the elastic tolerance.

    Measured (2026-08-30): every variable agrees to ≤ 2.2e-8 relative (worst
    B_p at 2.1e-8; a_cdom 1.4e-8, scale 9.8e-10) — all well under the 1e-6
    gate.
    """
    batch = l23_small_inelastic_batch
    heads = IC.load_default()
    rows = np.where(batch.zenith == 30.0)[0][:3]
    f64 = lambda x: jnp.asarray(np.asarray(x)[rows], dtype=jnp.float64)  # noqa: E731

    iops = IOPs(
        a=f64(batch.iops.a),
        bb_w=f64(batch.iops.bb_w),
        bb_p=f64(batch.iops.bb_p),
        a_ph=f64(batch.iops.a_ph),
        a_cdom=f64(batch.iops.a_cdom),
    )
    phase = PhaseParams(B_p=f64(batch.phase_params.B_p))
    geometry = Geometry.nadir(jnp.full((len(rows),), 35.0, dtype=jnp.float64))
    wave = jnp.asarray(np.asarray(batch.wave), dtype=jnp.float64)

    def model(i, p, g, w, phi, scale):
        return H.forward(
            i,
            p,
            g,
            w,
            "hybrid",
            inelastic=Inelastic(phi_C=phi, cdom_fl=CDOMFl(scale=scale)),
            corrections=heads,
            check_domain=False,
        )

    report = V.cdom_gradient_report(
        model,
        iops,
        phase,
        geometry,
        wave,
        phi_C=jnp.asarray(0.02, jnp.float64),
        scale=jnp.asarray(1.0, jnp.float64),
    )
    assert set(report) == set(V.CDOM_FD_STEPS)
    for name, value in report.items():
        assert value <= V.GRADIENT_TOL, f"d/d{name}: {value:.3e}"


@needs_weights
@needs_l23_inelastic
def test_cdom_gate_5_speed_within_rescoped_budget(cdom_release):
    """**CDOM §5 item 5**: full-batch forward with CDOM-fl on, within budget.

    ``test_gate_6_speed_within_twice_elastic``'s protocol exactly — median
    of three :func:`validation.speed_ratio` trials, alternating order, both
    callables jit-wrapped once here, loaders resolved eagerly first — with
    the "on" side now carrying **all three** inelastic terms
    (``Inelastic(cdom_fl=CDOMFl())``; Raman and Chl-fl stay on by default),
    matching how gate 6 already measures the total inelastic forward against
    the pure elastic hybrid. The *marginal* cost of the CDOM term alone
    (CDOM-only forward vs elastic) is measured and reported alongside, not
    gated — the design's budget is on the composed model.

    **The bound is machine-anchored, not the original design 2×** — the CQ3
    history, in short: at M5 task 7 (2026-08-30, JXP's Mac, quiet machine)
    the median measured **2.26–2.34× reproducibly** (2.45× under concurrent
    load), vs R+F-only at 1.94× and CDOM-only at ~1.4× the same day. The
    CDOM marginal (~0.3–0.4× elastic) is modest; most of the overage is
    baseline drift on this machine — the shipped R+F model measured 1.59×
    at its M4 acceptance and ~1.9× here/now (on the M4-era baseline the
    total would sit ~1.9×, under the original gate). JXP's call (Q&A CQ3):
    rescope the budget and note it as machine-anchored. Hence
    :data:`CDOM_GATE_SPEED_MACHINE_ANCHORED` (see its docstring for what
    "machine-anchored" means here — the strict-hash-pin sense: a
    characterization of this Mac, not a portable claim); design §5 item 5
    records the same rescoped bar. The M4 record's shared
    ``INELASTIC_GATE_SPEED = 2.0`` is untouched.
    """
    batch, _, _ = cdom_release
    from robust.rt import emulator as E

    em = E.load_default()
    heads = IC.load_default()
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)

    @jax.jit
    def everything_on(i, p, g, w):
        return H.forward(
            i,
            p,
            g,
            w,
            "hybrid",
            inelastic=Inelastic(cdom_fl=CDOMFl()),
            corrections=heads,
            emulator=em,
            check_domain=False,
        )

    @jax.jit
    def cdom_only(i, p, g, w):
        return H.forward(
            i,
            p,
            g,
            w,
            "hybrid",
            inelastic=Inelastic(raman=False, fluorescence=False, cdom_fl=CDOMFl()),
            emulator=em,
            check_domain=False,
        )

    @jax.jit
    def elastic(i, p, g, w):
        return H.forward(i, p, g, w, "hybrid", emulator=em, check_domain=False)

    ratios = [
        V.speed_ratio(everything_on, elastic, *args, repeats=5, reverse=bool(t % 2))[0]
        for t in range(3)
    ]
    median = float(np.median(ratios))
    marginal = V.speed_ratio(cdom_only, elastic, *args, repeats=5)[0]
    print(
        f"\nCDOM speed: everything-on/elastic median {median:.2f}x "
        f"(trials {[f'{r:.2f}' for r in ratios]}); "
        f"CDOM-only/elastic {marginal:.2f}x (reported, not gated)"
    )
    assert median <= CDOM_GATE_SPEED_MACHINE_ANCHORED, (
        f"median speed ratio {median:.2f}x over {len(ratios)} trials "
        f"(all: {[f'{r:.2f}' for r in ratios]}) exceeds the machine-anchored "
        f"{CDOM_GATE_SPEED_MACHINE_ANCHORED}x gate (Q&A CQ3)"
    )
