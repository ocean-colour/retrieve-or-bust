"""
Tests for :mod:`robust.rt.validation`'s protocol functions (M4 task 2).

``rrms`` itself is tested in ``test_baselines.py``, against the independently
produced ladder in ``context/RT/fig_rrms_ladder.csv``. What lands here is the
machinery M4 added around it: the per-λ, per-zenith and per-``B_p``-bin
breakdowns, the throughput timer, and the finite-difference gradient report — plus
the out-of-domain **fallback policy** on :func:`robust.rt.hybrid.rrs_forward`,
which is the code JXP asked for in prompt 5's Q7.

Two of these deserve their own note, because they are the ones that could pass
while being wrong:

**The gradient report has two special cases that are agreement, not failure.** A
model that genuinely ignores a variable (O25 and ``B_p``) has an exactly-zero
derivative on both sides; a ratio would report that as infinitely wrong. And a
step that leaves the physical domain returns NaN, which is a bad *step* rather
than a bad gradient. Both are pinned.

**The fallback must survive ``jit``.** It is implemented on the traceable mask
rather than the host-side warning check precisely so that compiling the model
cannot silently switch the policy off — a model that changes its answer when you
``jit`` it is worse than one with no policy at all. The test compares jitted
against eager for the *same* function, since the float32 fusion XLA does makes
bitwise equality the wrong instrument.
"""

from __future__ import annotations

import csv
import dataclasses
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import baselines as B
from robust.rt import conventions as C
from robust.rt import emulator as E
from robust.rt import hybrid as H
from robust.rt import validation as V
from robust.rt import ztt as Z
from robust.rt.types import Geometry, IOPs, PhaseParams


def synthetic(n_sample: int = 12, n_wave: int = 5):
    """A small batch with a known, smooth structure — no reference data needed."""
    wave = jnp.linspace(400.0, 700.0, n_wave)
    a = jnp.linspace(0.05, 0.5, n_sample)[:, None] * jnp.linspace(1.0, 1.4, n_wave)
    bb_p = jnp.linspace(0.002, 0.02, n_sample)[:, None] * jnp.ones(n_wave)
    bb_w = jnp.broadcast_to(C.bb_w(wave), a.shape)
    iops = IOPs(a=a, bb_w=bb_w, bb_p=bb_p)
    phase = PhaseParams(B_p=jnp.asarray(0.0126))
    geometry = Geometry.nadir(jnp.asarray(np.tile([0.0, 30.0, 60.0], n_sample // 3)))
    return iops, phase, geometry, wave


# ------------------------------------------------------------- the breakdowns --


def test_rrms_per_wavelength_is_the_axis_zero_reduction():
    """The named ladder is exactly ``rrms(axis=0)`` — one definition, not two.

    Worth pinning because the whole point of the shared metric is that the number
    in a log, in the record and in a figure is the same quantity; a second
    definition that drifted by a factor would be invisible.
    """
    rng = np.random.default_rng(0)
    truth = jnp.asarray(rng.uniform(1e-3, 2e-2, (20, 6)))
    pred = truth * jnp.asarray(rng.uniform(0.9, 1.1, (20, 6)))

    np.testing.assert_array_equal(
        np.asarray(V.rrms_per_wavelength(truth, pred)),
        np.asarray(V.rrms(truth, pred, axis=0)),
    )


def test_group_rrms_matches_scoring_each_group_by_hand():
    """Grouping is a slice-and-score, and the labels are host-side metadata.

    The failure this guards against is a grouping that silently averages the
    per-group numbers instead of recomputing on the subset — which differs
    whenever the groups are unbalanced.
    """
    rng = np.random.default_rng(1)
    truth = jnp.asarray(rng.uniform(1e-3, 2e-2, (9, 4)))
    pred = truth * jnp.asarray(rng.uniform(0.8, 1.2, (9, 4)))
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2])  # deliberately unbalanced

    got = V.group_rrms(truth, pred, labels)

    assert set(got) == {0.0, 1.0, 2.0}
    for value in (0, 1, 2):
        mask = labels == value
        assert got[float(value)] == pytest.approx(
            float(V.rrms(truth[mask], pred[mask])), rel=1e-12
        )


def test_bp_bins_are_equal_count_and_cover_everything():
    """Quantile bins, because L23's ``B_p`` spans only a factor ~1.7.

    Equal-width bins over so narrow a range would put nearly every sample in the
    middle two and report the outer two on a handful of scenes. Equal counts make
    each row's error comparable, which is the only reason the cut is informative
    at all — and every sample must land in exactly one bin.
    """
    rng = np.random.default_rng(2)
    B_p = jnp.asarray(rng.uniform(0.0103, 0.0180, (40, 3)))

    labels, edges = V.bp_bin_labels(B_p, n_bins=4)

    assert labels.shape == (40,)
    assert edges.shape == (5,)
    assert np.all(np.diff(edges) > 0)
    counts = np.bincount(labels, minlength=4)
    assert counts.sum() == 40
    # Equal-count to within one sample, which is all quantiles can promise for a
    # count not divisible by the bin number.
    assert counts.max() - counts.min() <= 1


def test_bp_bins_accept_a_per_sample_vector_too():
    """``B_p`` may be a spectrum or a per-sample scalar; both bin the same way."""
    rng = np.random.default_rng(3)
    per_sample = rng.uniform(0.0103, 0.0180, 24)
    spectrum = np.repeat(per_sample[:, None], 5, axis=1)

    labels_v, edges_v = V.bp_bin_labels(jnp.asarray(per_sample))
    labels_s, edges_s = V.bp_bin_labels(jnp.asarray(spectrum))

    np.testing.assert_array_equal(labels_v, labels_s)
    np.testing.assert_allclose(edges_v, edges_s, rtol=1e-6)


# ------------------------------------------------------------ speed and grads --


def test_throughput_reports_a_consistent_rate():
    """Seconds and rate describe the same measurement.

    Timing on a shared machine is not reproducible to better than ~20%, so the
    only thing worth asserting is internal consistency — ``rate == size /
    seconds`` — and that the call was actually made. The *ratio* between models is
    the number the report quotes; see the record.
    """
    iops, phase, geometry, wave = synthetic()

    seconds, rate = V.throughput(
        lambda i, p, g, w: B.rrs_gordon(i, p, g, w),
        iops,
        phase,
        geometry,
        wave,
        repeats=2,
    )

    assert seconds > 0.0
    assert rate == pytest.approx(iops.a.size / seconds, rel=1e-9)


@pytest.mark.parametrize("model", ["ztt", "o25"])
def test_gradient_report_passes_the_gate_for_the_analytic_models(jax_x64, model):
    """Every variable agrees with central differences inside ``GRADIENT_TOL``.

    The differentiability axis of design §6, as a function so the gate and the
    report call the same code. Evaluated at 45°, deliberately **between** O25's
    coefficient-table nodes: a piecewise-linear lookup has a kink at each node, so
    autodiff takes one one-sided slope while the central difference averages both
    and they disagree by O(1). L23's own angles are the nodes, which is exactly
    the trap. Measured here: everything at or below 3e-9.
    """
    iops, phase, geometry, wave = synthetic(n_sample=3)
    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731
    iops = IOPs(a=f64(iops.a), bb_w=f64(iops.bb_w), bb_p=f64(iops.bb_p))
    phase = PhaseParams(B_p=f64(phase.B_p))
    geometry = Geometry.nadir(jnp.full((3,), 45.0, dtype=jnp.float64))
    fn = (
        (lambda i, p, g, w: Z.rrs_ZTT(i, p, g, w))
        if model == "ztt"
        else (lambda i, p, g, w: B.rrs_o25(i, p, g, w))
    )

    report = V.gradient_report(fn, iops, phase, geometry, f64(wave))

    assert set(report) == set(V.FD_STEPS)
    for name, value in report.items():
        assert value < V.GRADIENT_TOL, f"d/d{name} disagrees by {value:.2e}"


def test_gradient_report_calls_an_ignored_variable_agreement_not_failure(jax_x64):
    """O25 ignores ``B_p``: both derivatives are exactly 0, which is agreement.

    A ratio-based comparison reports 0/0 as infinitely wrong, and an earlier
    version of this function did exactly that — turning a model's *documented*
    blind spot into a spurious gradient-gate failure.
    """
    iops, phase, geometry, wave = synthetic(n_sample=3)
    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731

    report = V.gradient_report(
        lambda i, p, g, w: B.rrs_o25(i, p, g, w),
        IOPs(a=f64(iops.a), bb_w=f64(iops.bb_w), bb_p=f64(iops.bb_p)),
        PhaseParams(B_p=f64(phase.B_p)),
        Geometry.nadir(jnp.full((3,), 45.0, dtype=jnp.float64)),
        f64(wave),
    )

    assert report["B_p"] == 0.0


def test_gradient_report_flags_a_step_that_leaves_the_domain(jax_x64):
    """A step that drives ``bb_p`` negative reports ``inf``, not a number.

    ZTT returns NaN outside the physical domain. That is a bad *step*, not a bad
    gradient, and silently comparing against NaN would have produced a passing
    ``False`` in a boolean gate. M2 hit exactly this when an ``argmin`` picked a
    NaN as the best step.
    """
    iops, phase, geometry, wave = synthetic(n_sample=3)
    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731

    report = V.gradient_report(
        lambda i, p, g, w: Z.rrs_ZTT(i, p, g, w),
        IOPs(a=f64(iops.a), bb_w=f64(iops.bb_w), bb_p=f64(iops.bb_p)),
        PhaseParams(B_p=f64(phase.B_p)),
        Geometry.nadir(jnp.full((3,), 45.0, dtype=jnp.float64)),
        f64(wave),
        steps={
            "a": 1e-6,
            "bb_p": 1.0,
            "B_p": 1e-8,
            "theta_s": 1e-3,
        },  # bb_p -> negative
    )

    assert not np.isfinite(report["bb_p"])


# ------------------------------------------------------------------ reporting --


def test_score_models_slices_one_evaluation_per_model():
    """Every model is scored on literally the same rows.

    The point of passing predictions rather than callables: "identical data" is a
    slice of one evaluation, not a claim about two code paths agreeing.
    """
    rng = np.random.default_rng(4)
    truth = jnp.asarray(rng.uniform(1e-3, 2e-2, (10, 3)))
    # The error must DIFFER between the halves, or the test cannot tell slicing from
    # not slicing: with a uniform relative error every subset scores identically, and
    # a `score_models` that ignored `masks` entirely used to pass this.
    factor = np.where(np.arange(10) < 5, 1.01, 1.05)[:, None]
    models = {"a": truth * jnp.asarray(factor), "b": truth * 0.98}
    masks = {"all": np.ones(10, dtype=bool), "first half": np.arange(10) < 5}

    table = V.score_models(models, truth, masks)

    assert set(table) == {"a", "b"}
    assert table["a"]["first half"] == pytest.approx(1.0, rel=1e-4)  # +1% there
    assert table["a"]["all"] == pytest.approx(
        100 * np.sqrt(np.mean((factor - 1.0) ** 2)), rel=1e-4
    )  # ... and +1%/+5% pooled, which differs from either half
    assert table["a"]["all"] > 1.5 * table["a"]["first half"]
    assert table["b"]["all"] == pytest.approx(2.0, rel=1e-4)


def test_markdown_table_formats_numbers_and_headers():
    """The table renders as GitHub markdown with two decimals on floats."""
    text = V.markdown_table([["model", 1.234, 5.0]], ["name", "x", "y"])

    lines = text.splitlines()
    assert lines[0] == "| name | x | y |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| model | 1.23 | 5.00 |"


# ------------------------------------------- the out-of-domain policy (Q7) ----


def test_supported_zenith_envelope_is_zero_to_sixty():
    """The sanctioned span is 0–60°, L23's coverage and JXP's call (Q7).

    Deliberately *not* whatever angles a given emulator happened to see: a fit
    trained on 0/30 is allowed to be asked for 60° without complaint. Pinning the
    constant keeps that a decision rather than an accident.
    """
    assert E.SUPPORTED_THETA_S == (0.0, 60.0)


def test_a_narrow_fit_is_not_flagged_inside_the_supported_envelope(l23_small_batch):
    """An emulator trained on 0/30 may be used at 60° without a warning.

    This is the Q7 correction: before it, the domain check compared against the
    *trained* angles and warned at 60° for such a fit. The other six features are
    still judged against the trained range, where "outside what I learned" does
    mean unreliable.
    """
    from robust.rt.data import l23

    batch = l23_small_batch
    splits = l23.make_splits(batch)
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    narrow, _ = E.fit(
        *args,
        C.Rrs_to_rrs(batch.Rrs),
        train=splits.zenith_train,
        config=E.EmulatorConfig(steps=50, eval_every=50),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H.rrs_forward(*args, "hybrid", emulator=narrow)
    assert [w for w in caught if issubclass(w.category, H.DomainWarning)] == []

    # Judged against the trained range instead, it *is* extrapolating -- the
    # question the option exists to answer.
    breaches = narrow.out_of_domain(*args, theta_s_limits=None)
    assert "cos_theta_s" in breaches


def test_fallback_returns_the_backbone_beyond_the_envelope(l23_small_batch):
    """``on_out_of_domain="ztt"`` degrades to the backbone past 60°, not before.

    The other half of JXP's Q6 instruction — "we won't use the emulator at larger
    angles *or* will warn". Inside the envelope the two policies must agree
    exactly, or the option would be silently changing sanctioned results.
    """
    batch = l23_small_batch
    emulator = E.load_default()
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)

    inside_warn = H.rrs_forward(*args, "hybrid", emulator=emulator)
    inside_ztt = H.rrs_forward(
        *args, "hybrid", emulator=emulator, on_out_of_domain="ztt"
    )
    np.testing.assert_array_equal(np.asarray(inside_warn), np.asarray(inside_ztt))

    low_sun = (
        batch.iops,
        batch.phase_params,
        Geometry.nadir(jnp.full_like(batch.geometry.theta_s, 75.0)),
        batch.wave,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", H.DomainWarning)
        outside = H.rrs_forward(
            *low_sun, "hybrid", emulator=emulator, on_out_of_domain="ztt"
        )
        applied = H.rrs_forward(*low_sun, "hybrid", emulator=emulator)
    np.testing.assert_array_equal(np.asarray(outside), np.asarray(Z.rrs_ZTT(*low_sun)))
    assert not np.array_equal(np.asarray(applied), np.asarray(outside))


def test_fallback_survives_jit(l23_small_batch):
    """The policy holds under ``jit`` — the reason it is built on a traceable mask.

    A policy implemented with the host-side warning check would lapse silently the
    moment anyone compiled the model, which is the hot path. Compared jitted
    against eager for the *same* function rather than bitwise against a separately
    computed backbone: XLA fuses ``rrs_ztt + 0.0`` differently, which moves the
    float32 result by ~5e-7 — while a policy that had lapsed would differ by
    ~2e-1, five orders of magnitude larger.
    """
    batch = l23_small_batch
    emulator = E.load_default()
    low_sun = (
        batch.iops,
        batch.phase_params,
        Geometry.nadir(jnp.full_like(batch.geometry.theta_s, 75.0)),
        batch.wave,
    )

    def model(i, p, g, w):
        return H.rrs_forward(
            i,
            p,
            g,
            w,
            "hybrid",
            emulator=emulator,
            check_domain=False,
            on_out_of_domain="ztt",
        )

    eager = np.asarray(model(*low_sun))
    jitted = np.asarray(jax.jit(model)(*low_sun))

    np.testing.assert_allclose(jitted, eager, rtol=1e-5)
    lapsed = np.asarray(
        jax.jit(
            lambda i, p, g, w: H.rrs_forward(
                i, p, g, w, "hybrid", emulator=emulator, check_domain=False
            )
        )(*low_sun)
    )
    assert np.abs(lapsed / eager - 1.0).max() > 1e-2  # what a lapsed policy looks like


def test_unknown_out_of_domain_policy_raises():
    """A typo'd policy name raises rather than silently meaning "warn".

    Same reasoning as the ``mode`` check: a mis-spelled policy that quietly did
    nothing would corrupt a comparison table with no visible symptom.
    """
    iops, phase, geometry, wave = synthetic()

    with pytest.raises(ValueError, match="on_out_of_domain"):
        H.rrs_forward(iops, phase, geometry, wave, "hybrid", on_out_of_domain="ZTT")
    with pytest.raises(ValueError, match="on_out_of_domain"):
        H.forward(iops, phase, geometry, wave, "hybrid", on_out_of_domain="fallback")


# ================================ THE M4 ACCEPTANCE GATE (task 3) =============
#
# The coding plan's wording is "hybrid beats standard Gordon on BOTH held-out
# splits, and passes the gradient-correctness gate". Two of JXP's decisions
# reshape it, and both are recorded here rather than in a commit message:
#
#   Q9 -- gate on beating **O25** on the scene split. Gordon is now the weakest
#   thing in the table: the hybrid beats it by 24x while beating O25, the
#   strongest available benchmark, by 2.3x. Gating on the weak one would have
#   let the milestone pass on a claim nobody should care about.
#
#   Q7 + Q6 -- the *zenith* half is reported, not gated. M3 measured the
#   hybrid's unseen-60 error as seed-dependent (4.74-12.24% across five seeds
#   against Gordon's 9.01%), and JXP called 60-degree extrapolation a stretch
#   goal. Note the interaction the two answers produce: because the sanctioned
#   envelope reaches 60 degrees, the out-of-domain fallback deliberately does
#   NOT fire there, so it cannot rescue that half of the gate -- verified, the
#   fallback triggers on 0 of 9960 samples. Gating it would have been gating a
#   coin flip.
#
# The committed gate therefore runs on the fixture with a toy fit, and asserts
# what is reproducible. The full-data numbers live in design/validation/.


#: Steps for the committed gate fit. 400 was not enough: measured across seeds
#: {23, 1, 7, 101, 2024} the hybrid then scored 0.454-0.575% against O25's 0.578%, so
#: the worst seed passed by **0.6%** — the gate was very nearly seed luck. At 800 the
#: range is 0.376-0.419%, a 27% margin, and every seed clears `0.9 * O25`.
GATE_STEPS = 800


@pytest.fixture(scope="module")
def gate_fit(l23_small_batch):
    """One toy fit on the fixture's train split, shared by the gate tests.

    Module-scoped for speed, which makes it sensitive to something subtle: JAX's x64
    flag is global and the ``jax_x64`` fixture is function-scoped, so whichever test
    instantiates this first decides the dtype it trains in — measured 0.5468% in
    float32 against 0.5628% in float64, which is half of the gate's margin. The
    gradient test therefore trains its **own** emulator rather than sharing this one,
    so no test order can change what the gate is measuring.
    """
    from robust.rt.data import l23

    splits = l23.make_splits(l23_small_batch)
    emulator, _ = E.fit_l23(
        l23_small_batch,
        splits,
        config=E.EmulatorConfig(steps=GATE_STEPS, eval_every=200),
    )
    return emulator, splits


@pytest.fixture(scope="module")
def gate_scores(gate_fit, l23_small_batch):
    """Every model's rRMS on the held-out scenes, from one evaluation each."""
    emulator, splits = gate_fit
    batch = l23_small_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    truth = C.Rrs_to_rrs(batch.Rrs)
    o25 = B.fit_o25(batch.iops, batch.Rrs, batch.geometry, train=splits.scene_train)
    models = {
        "gordon": B.rrs_gordon(*args),
        "ztt": Z.rrs_ZTT(*args),
        "o25": B.rrs_o25(*args, coeffs=o25),
        "hybrid": H.rrs_forward(*args, "hybrid", emulator=emulator, check_domain=False),
    }
    return V.score_models(models, truth, {"held_out": splits.scene_test})


def test_gate_hybrid_beats_o25_on_the_held_out_scenes(gate_scores):
    """**THE GATE (Q9).** The hybrid beats the strongest benchmark, not the weakest.

    O25 refit on our own training split is a far harder target than standard
    Gordon: on the full batch it reaches 0.69% against Gordon's 7.21%, so a
    milestone gated on Gordon would pass while losing to the actual state of the
    art.

    **The assertion demands a margin, not a hair.** On the fixture O25 scores 0.578%
    and the hybrid 0.376-0.419% across five seeds, so 10% of headroom is comfortable
    at every seed — where an earlier 400-step version left the worst seed passing by
    0.6%, i.e. by luck. A 3% shrink of the learned correction used to pass; the
    margin now makes that fail.
    """
    scores = {name: row["held_out"] for name, row in gate_scores.items()}
    assert scores["hybrid"] < 0.9 * scores["o25"], (
        f"hybrid {scores['hybrid']:.3f}% did not beat the O25 refit "
        f"{scores['o25']:.3f}% by the required 10% margin on the held-out scenes"
    )


def test_gate_hybrid_beats_gordon_and_the_backbone(gate_scores):
    """The plan's original floor, kept: the hybrid also beats Gordon and ZTT.

    Weaker than the O25 gate above and retained only because it is the coding
    plan's own wording — ZTT alone already beats Gordon, so an emulator returning
    exactly zero would satisfy it.
    """
    scores = {name: row["held_out"] for name, row in gate_scores.items()}
    assert scores["hybrid"] < scores["gordon"]
    assert scores["hybrid"] < scores["ztt"]


def test_gate_gradient_correctness(jax_x64, l23_small_batch):
    """**THE GATE, second half.** ``jax.grad`` of the full hybrid matches central
    differences on every input.

    The property the eventual inversion depends on, and the one thing in M4 that
    is a hard pass/fail rather than a comparison. Evaluated at 45 deg, between
    O25's table nodes, so the same harness can score every model without landing
    on a lookup kink.
    """
    from robust.rt.data import l23

    batch = l23_small_batch
    # Its own emulator, not the module-scoped ``gate_fit``: this test enables x64, and
    # sharing the fixture would mean the gate's numbers depended on test order. A
    # short fit is plenty -- what is under test is the gradient, not the accuracy.
    emulator, _ = E.fit_l23(
        batch,
        l23.make_splits(batch),
        config=E.EmulatorConfig(steps=100, eval_every=100),
    )
    rows = np.where(batch.zenith == 30.0)[0][:3]
    f64 = lambda x: jnp.asarray(np.asarray(x)[rows], dtype=jnp.float64)  # noqa: E731

    report = V.gradient_report(
        lambda i, p, g, w: H.rrs_forward(
            i, p, g, w, "hybrid", emulator=emulator, check_domain=False
        ),
        IOPs(a=f64(batch.iops.a), bb_w=f64(batch.iops.bb_w), bb_p=f64(batch.iops.bb_p)),
        PhaseParams(B_p=f64(batch.phase_params.B_p)),
        Geometry.nadir(jnp.full((len(rows),), 45.0, dtype=jnp.float64)),
        jnp.asarray(np.asarray(batch.wave), dtype=jnp.float64),
    )

    for name, value in report.items():
        assert value < V.GRADIENT_TOL, f"d/d{name} disagrees by {value:.2e}"

    # And every variable must actually be exercised. `gradient_report` reports 0.0
    # when both derivatives are exactly zero, which is right for a model that ignores
    # a variable (O25 and `B_p`) — but it would also hide a perturbation that was
    # never applied, letting a wiring bug read as perfect agreement. The hybrid
    # depends on all four, so all four must be non-zero here.
    for name, value in report.items():
        assert value > 0.0, (
            f"d/d{name} reported exactly 0.0 for a model that depends on it — the "
            "perturbation was probably never applied"
        )


def test_the_zenith_half_is_reported_not_gated(l23_small_batch):
    """The unseen-60 split is measured and *not* asserted — deliberately.

    M3 found the hybrid's error there spans 4.74–12.24% across seeds while Gordon
    is 9.01%, so an assertion would pass or fail on the seed. JXP's Q6 call was
    report-and-defer. This test pins the *reporting*: the numbers must be
    computable and finite, and the fallback must be inert here, because 60 deg is
    inside the sanctioned envelope (:data:`robust.rt.emulator.SUPPORTED_THETA_S`)
    and so the policy cannot rescue this half of the gate. That last point is the
    non-obvious interaction between two of JXP's answers, and it is why the gate
    above stops at the scene split.
    """
    from robust.rt.data import l23

    batch = l23_small_batch
    splits = l23.make_splits(batch)
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    truth = C.Rrs_to_rrs(batch.Rrs)
    rrs_ztt = Z.rrs_ZTT(*args)
    emulator, _ = E.fit(
        *args,
        truth,
        train=splits.zenith_train,
        rrs_ztt=rrs_ztt,
        config=E.EmulatorConfig(steps=200, eval_every=100),
    )
    unseen = splits.zenith_test

    plain = H.rrs_forward(*args, "hybrid", emulator=emulator, check_domain=False)
    fallback = H.rrs_forward(
        *args, "hybrid", emulator=emulator, check_domain=False, on_out_of_domain="ztt"
    )

    assert np.isfinite(float(V.rrms(truth[unseen], plain[unseen])))
    # The fallback is inert at 60 deg: identical output, and nothing flagged.
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(fallback))
    assert int(np.asarray(emulator.out_of_domain_mask(*args)).sum()) == 0


# ------------------------------------------- the committed artefacts (task 3) --


ARTEFACTS = Path(__file__).resolve().parents[2] / "design" / "validation"


@pytest.mark.parametrize(
    ("name", "columns"),
    [
        ("metrics.csv", ["model", "split", "rrms_percent"]),
        ("rrms_per_wavelength.csv", None),
    ],
)
def test_committed_csvs_parse_to_their_promised_columns(name, columns):
    """The artefacts must survive ``csv.DictReader`` with the header they advertise.

    Found by review, and worth a permanent test because the failure was **silent**.
    The first version of ``run_validation.py`` joined fields with commas by hand, and
    the model names contain commas — "O25 form, refit on L23", "hybrid, MLP". So
    ``metrics.csv`` carried four fields under a three-field header, and the ladder's
    header expanded seven model names into ten columns: a consumer would have
    mis-labelled every column without anything raising. Writing through :mod:`csv`
    fixes it; this pins that nobody hand-joins one again.
    """
    path = ARTEFACTS / name
    if not path.is_file():
        pytest.skip(f"artefact not generated: {path}")

    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    header, body = rows[0], rows[1:]
    assert body, "artefact has a header but no data"
    if columns is not None:
        assert header == columns
    # Every row must have exactly as many fields as the header promises. This is the
    # assertion the hand-joined version failed.
    for i, row in enumerate(body, start=2):
        assert len(row) == len(header), (
            f"{name} line {i}: {len(row)} fields against a {len(header)}-field header"
        )


def test_the_per_wavelength_ladder_aggregates_to_the_scalar_table():
    """Each ladder column must RMS back to its scalar in ``metrics.csv``.

    **This is the test that catches a stale artefact**, and it needs no trust in the
    model code at all — it is a pure internal-consistency check between two files
    written by the same run: since rRMS is a root-mean-square over samples and
    wavelengths, RMS-ing a per-λ column reproduces the pooled scalar exactly.

    Worth having because it already would have paid: a ladder CSV sat committed whose
    Gordon column aggregated to 9.57% against the table's 7.21% — a stale run that no
    other check noticed, and that made the committed figure overstate Gordon's blue-end
    error by ~2x.
    """
    metrics, ladder = ARTEFACTS / "metrics.csv", ARTEFACTS / "rrms_per_wavelength.csv"
    if not (metrics.is_file() and ladder.is_file()):
        pytest.skip("artefacts not generated")

    with metrics.open(newline="") as handle:
        scalars = {
            row["model"]: float(row["rrms_percent"])
            for row in csv.DictReader(handle)
            if row["split"] == "held-out scenes"
        }
    with ladder.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for model, scalar in scalars.items():
        column = np.array([float(row[model]) for row in rows])
        pooled = float(np.sqrt((column**2).mean()))
        # rtol 1e-3: both sides are stored to four decimals, so their own rounding
        # dominates; a stale column is out by tens of percent, not tenths.
        assert pooled == pytest.approx(scalar, rel=1e-3), (
            f"{model}: ladder aggregates to {pooled:.4f}% but metrics.csv says "
            f"{scalar:.4f}% — one of the two artefacts is stale"
        )


def test_committed_metrics_agree_between_the_two_csvs():
    """The model names in `metrics.csv` are exactly the ladder's data columns.

    A cheap consistency check across two artefacts written by the same run: if they
    ever disagree, one of them was regenerated and the other was not, and a reader
    comparing them would be comparing different fits.
    """
    metrics, ladder = ARTEFACTS / "metrics.csv", ARTEFACTS / "rrms_per_wavelength.csv"
    if not (metrics.is_file() and ladder.is_file()):
        pytest.skip("artefacts not generated")

    with metrics.open(newline="") as handle:
        models = {row["model"] for row in csv.DictReader(handle)}
    with ladder.open(newline="") as handle:
        columns = set(next(csv.reader(handle))[1:])

    assert models == columns, (
        f"only in metrics: {models - columns}; only in ladder: {columns - models}"
    )


# ------------------------------------ regressions found in the M4 review pass --


def test_a_slightly_off_nadir_view_is_flagged(l23_small_batch):
    """**Regression.** A view angle the emulator never saw must be reported.

    Found reviewing the M4 diff, and the worst kind of bug: silent. ``cos_theta_v``
    is constant in L23, so its trained span is zero — and the domain check used to
    scale the excursion by the feature's own *value*, making 5° of sensor zenith a
    0.4% excursion and therefore "in domain". Meanwhile the standardisation divides
    the same excursion by ``_STD_FLOOR``, so the network saw **-3.8e5**, every
    ``tanh`` saturated, and the correction collapsed from a spectrum spanning
    [-0.10, +0.27] to a flat **+0.046 at all 81 wavelengths** — with no warning and
    no fallback. Both now divide by the same number, so the check measures the
    excursion in the units the network actually sees.
    """
    batch = l23_small_batch
    emulator = E.load_default()
    zeros = jnp.zeros_like(batch.geometry.theta_s)

    def at_view(theta_v):
        geometry = Geometry(
            theta_s=batch.geometry.theta_s,
            theta_v=jnp.full_like(batch.geometry.theta_s, theta_v),
            dphi=zeros,
        )
        return (batch.iops, batch.phase_params, geometry, batch.wave)

    # Nadir is what L23 contains, and must stay clean.
    assert emulator.out_of_domain(*at_view(0.0)) == {}
    assert int(np.asarray(emulator.out_of_domain_mask(*at_view(0.0))).sum()) == 0

    # Half a degree off is already outside anything it was trained on.
    for theta_v in (0.5, 5.0):
        breaches = emulator.out_of_domain(*at_view(theta_v))
        assert "cos_theta_v" in breaches, f"theta_v={theta_v} went unreported"
        assert int(
            np.asarray(emulator.out_of_domain_mask(*at_view(theta_v))).sum()
        ) == (batch.n_sample)


def test_the_two_domain_predicates_agree_on_non_finite_input(l23_small_batch):
    """**Regression.** A NaN input must be out of domain for *both* implementations.

    ``out_of_domain`` and ``out_of_domain_mask`` are two implementations of one
    predicate, and NaN made them disagree: ``excess > tol`` is False for NaN, so the
    traceable mask — the one the fallback policy acts on — answered "in domain" while
    the host check answered "out". A single NaN in ``a`` is exactly what an inversion
    overshoot produces, so this is the path that mattered. The mask now negates the
    in-range test instead, which sends NaN to ``True``.
    """
    batch = l23_small_batch
    emulator = E.load_default()
    a = np.asarray(batch.iops.a).copy()
    a[0, 0] = np.nan
    args = (
        IOPs(a=jnp.asarray(a), bb_w=batch.iops.bb_w, bb_p=batch.iops.bb_p),
        batch.phase_params,
        batch.geometry,
        batch.wave,
    )

    breaches = emulator.out_of_domain(*args)
    assert "log10_u" in breaches
    assert not np.isfinite(breaches["log10_u"].excess)  # reported, not "nan% of span"
    assert int(np.asarray(emulator.out_of_domain_mask(*args)).sum()) == 1


def test_gradient_report_keeps_the_callers_geometry(jax_x64):
    """**Regression.** Only ``theta_s`` is perturbed; the rest of the geometry stands.

    An earlier version rebuilt the geometry with ``Geometry.nadir``, silently
    discarding ``theta_v`` and ``dphi``. Harmless on nadir-only L23 — and exactly the
    kind of thing that would certify a gradient at the wrong geometry the moment M5
    goes off-nadir. A spy model records what it was actually called with.
    """
    iops, phase, _, wave = synthetic(n_sample=3)
    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731
    seen = []

    def spy(i, p, g, w):
        seen.append((float(np.asarray(g.theta_v)[0]), float(np.asarray(g.dphi)[0])))
        return B.rrs_gordon(i, p, g, w)

    V.gradient_report(
        spy,
        IOPs(a=f64(iops.a), bb_w=f64(iops.bb_w), bb_p=f64(iops.bb_p)),
        PhaseParams(B_p=f64(phase.B_p)),
        Geometry(
            theta_s=jnp.full((3,), 30.0, dtype=jnp.float64),
            theta_v=jnp.full((3,), 30.0, dtype=jnp.float64),
            dphi=jnp.full((3,), 90.0, dtype=jnp.float64),
        ),
        f64(wave),
    )

    assert seen, "the model was never called"
    assert set(seen) == {(30.0, 90.0)}, f"geometry was altered: {set(seen)}"


@pytest.mark.parametrize(
    "steps, why",
    [
        ({"a": 1e-6, "theta_S": 1e-3}, "a typo in a variable name"),
        ({"chlorophyll": 1e-3}, "a variable that is not a field at all"),
        ({"wind": 1e-3}, "a field that is None on these inputs"),
    ],
)
def test_gradient_report_rejects_a_variable_it_cannot_perturb(jax_x64, steps, why):
    """**Regression, restated for M5.** An unknown variable raises.

    The guard this replaces demanded *exactly* M2's four variables, so no gate
    could check ``theta_v``, ``dphi`` or a new ``PhaseParams`` field — the reason
    M5 task 6 exists. What it was really protecting against was an extra key
    reporting **0.0**, "perfect agreement", for a variable that was never
    perturbed; that is now structural rather than enforced, since every name is
    resolved to a real field and perturbed through ``dataclasses.replace``.

    So subsets and extra *known* variables are now legal (see the tests below),
    and what must still raise is a name the inputs cannot offer — including
    ``wind``, which is a real field but ``None`` here and would otherwise be
    ``None + 1e-3``.
    """
    iops, phase, geometry, wave = synthetic(n_sample=3)
    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731

    with pytest.raises(ValueError, match="cannot perturb"):
        V.gradient_report(
            lambda i, p, g, w: B.rrs_gordon(i, p, g, w),
            IOPs(a=f64(iops.a), bb_w=f64(iops.bb_w), bb_p=f64(iops.bb_p)),
            PhaseParams(B_p=f64(phase.B_p)),
            geometry,
            f64(wave),
            steps=steps,
        )


def test_gradient_report_rejects_an_empty_steps_dict(jax_x64):
    """Checking nothing must not read as checking everything."""
    iops, phase, geometry, wave = synthetic(n_sample=3)
    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731

    with pytest.raises(ValueError, match="names no variables"):
        V.gradient_report(
            lambda i, p, g, w: B.rrs_gordon(i, p, g, w),
            IOPs(a=f64(iops.a), bb_w=f64(iops.bb_w), bb_p=f64(iops.bb_p)),
            PhaseParams(B_p=f64(phase.B_p)),
            geometry,
            f64(wave),
            steps={},
        )


def test_throughput_rejects_zero_repeats():
    """``repeats=0`` raises instead of dividing by zero."""
    iops, phase, geometry, wave = synthetic()

    with pytest.raises(ValueError, match="repeats must be >= 1"):
        V.throughput(
            lambda i, p, g, w: B.rrs_gordon(i, p, g, w),
            iops,
            phase,
            geometry,
            wave,
            repeats=0,
        )


# ------------------------------------------------- M5 task 6: the toolkit ----
# Three limits, each a reasonable choice while L23 was the only dataset, each
# blocking a gate M5 needs. One regression test apiece, and each one fails
# against the pre-task-6 code: the first two call keyword arguments that did not
# exist, and the third perturbs a field the old closure silently discarded.


def test_rrms_mask_excludes_bad_bands_instead_of_whole_spectra():
    """**Regression.** ``rrms`` divides by truth; PB24 has exactly-zero ``rrs``.

    Without a mask the only choices were an ``inf`` or dropping the whole
    spectrum -- eleven good bands to exclude one bad value (`pb24.load_batch`'s
    ``drop_zero_rrs``). Now the bad band alone can go.
    """
    truth = jnp.array([[1.0, 2.0, 0.0], [1.0, 1.0, 1.0]])
    pred = jnp.array([[1.1, 2.0, 5.0], [1.0, 1.0, 1.0]])
    keep = jnp.array([[True, True, False], [True, True, True]])

    assert not np.isfinite(float(V.rrms(truth, pred)))

    masked = float(V.rrms(truth, pred, where=keep))
    by_hand = 100.0 * np.sqrt(np.mean([0.1**2, 0.0, 0.0, 0.0, 0.0]))

    assert masked == pytest.approx(by_hand, rel=1e-5)


def test_rrms_mask_keeps_the_gradient_finite():
    """The double-``where``: masking after the division would train to NaN.

    ``rrms`` doubles as a training loss (M3), so a mask that hides a NaN in the
    forward pass while leaking it into the backward pass would be worse than no
    mask at all -- the loss would look healthy and the gradient would be NaN.
    """
    truth = jnp.array([[1.0, 2.0, 0.0]])
    pred = jnp.array([[1.1, 2.0, 5.0]])
    keep = jnp.array([[True, True, False]])

    masked = jax.grad(lambda p: V.rrms(truth, p, where=keep))(pred)
    unmasked = jax.grad(lambda p: V.rrms(truth, p))(pred)

    assert bool(np.isfinite(np.asarray(masked)).all())
    assert not bool(np.isfinite(np.asarray(unmasked)).all())


def test_rrms_reports_nan_for_a_wholly_masked_group():
    """An empty selection must announce itself, not arrive as a zero."""
    truth = jnp.ones((2, 3))
    pred = jnp.ones((2, 3)) * 1.5

    assert np.isnan(float(V.rrms(truth, pred, where=jnp.zeros((2, 3), bool))))


def test_group_rrms_keeps_a_column_the_data_does_not_have():
    """**Regression.** A missing group used to shift every column to its left.

    ``np.unique`` can only produce non-empty groups, so this function could not
    return a short dict -- which is why ``run_validation.py`` zipped its
    ``.values()`` against hard-coded headers. On PB24, with eight zeniths and
    eight view angles, a split may legitimately omit one.
    """
    truth = jnp.ones((4, 2))
    pred = jnp.ones((4, 2)) * 1.1
    labels = np.array([0.0, 0.0, 30.0, 30.0])

    without = V.group_rrms(truth, pred, labels)
    with_expected = V.group_rrms(truth, pred, labels, expected=[0.0, 30.0, 60.0])

    assert list(without) == [0.0, 30.0]
    assert list(with_expected) == [0.0, 30.0, 60.0]
    assert np.isnan(with_expected[60.0])
    assert with_expected[0.0] == pytest.approx(without[0.0])


def test_group_rrms_passes_the_mask_through_per_group():
    """A per-band mask has to be sliced with the samples, not applied to all."""
    truth = jnp.array([[1.0, 0.0], [1.0, 1.0]])
    pred = jnp.array([[1.1, 9.0], [1.2, 1.0]])
    labels = np.array([0.0, 1.0])
    keep = jnp.array([[True, False], [True, True]])

    got = V.group_rrms(truth, pred, labels, where=keep)

    assert got[0.0] == pytest.approx(10.0, rel=1e-4)  # only the good band
    assert np.isfinite(got[1.0])


@dataclasses.dataclass(frozen=True)
class _RichPhase:
    """A ``PhaseParams`` with a field this module has never heard of.

    Stands in for what M5 task 14 will add when ``PhaseParams`` is promoted to
    the ZTT backward-VSF parameterization. Deliberately *not* the real class: the
    point is to prove ``gradient_report`` carries fields it does not know about.
    """

    B_p: object
    gamma: object


def test_gradient_report_carries_a_phase_field_it_has_never_seen(jax_x64):
    """**Regression.** The closure used to rebuild ``PhaseParams(B_p=...)``.

    That construction silently dropped every other field, so a task-14 model
    would have been certified at the wrong phase function with no symptom.
    Rebuilding with ``dataclasses.replace`` keeps whatever the caller passed --
    and this test proves the new field both survives *and* is perturbed, since a
    model that reads it gets a non-zero derivative.
    """
    iops, phase, geometry, wave = synthetic(n_sample=3)
    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731
    rich = _RichPhase(B_p=f64(phase.B_p), gamma=f64(np.full(3, 2.0)))

    seen = []

    def model(i, p, g, w):
        seen.append(type(p).__name__)
        # depends on gamma, so a dropped field shows up as a zero derivative
        return B.rrs_gordon(i, PhaseParams(B_p=p.B_p), g, w) * p.gamma[:, None]

    report = V.gradient_report(
        model,
        IOPs(a=f64(iops.a), bb_w=f64(iops.bb_w), bb_p=f64(iops.bb_p)),
        rich,
        geometry,
        f64(wave),
        steps={"gamma": 1e-6, "B_p": 1e-8},
    )

    assert set(seen) == {"_RichPhase"}, "the container type was not preserved"
    # 0.0 would mean the model ignores gamma -- i.e. the perturbation never
    # arrived. A real agreement is a small relative difference, not an identity.
    assert report["gamma"] < V.GRADIENT_TOL
    assert report["gamma"] != 0.0


def test_gradient_report_can_check_the_view_angles(jax_x64):
    """**Regression.** ``theta_v`` and ``dphi`` were inexpressible before task 6.

    Task 11 trains the emulator with the view angles live, and its gate is a
    gradient check; that check could not have been written against the old
    ``steps`` guard.
    """
    iops, phase, geometry, wave = synthetic(n_sample=3)
    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731
    off_nadir = Geometry(
        theta_s=f64(np.full(3, 30.0)),
        theta_v=f64(np.full(3, 25.0)),
        dphi=f64(np.full(3, 90.0)),
    )

    def model(i, p, g, w):
        # A stand-in with a genuine, smooth dependence on both view angles.
        tilt = jnp.cos(jnp.deg2rad(g.theta_v)) + 0.1 * jnp.cos(jnp.deg2rad(g.dphi))
        return B.rrs_gordon(i, p, g, w) * tilt[:, None]

    report = V.gradient_report(
        model,
        IOPs(a=f64(iops.a), bb_w=f64(iops.bb_w), bb_p=f64(iops.bb_p)),
        PhaseParams(B_p=f64(phase.B_p)),
        off_nadir,
        f64(wave),
        steps=V.default_steps(["theta_v", "dphi"]),
    )

    assert set(report) == {"theta_v", "dphi"}
    for name, value in report.items():
        assert value < V.GRADIENT_TOL, f"{name}: {value}"
        assert value != 0.0, f"{name} was never perturbed"


def test_default_steps_refuses_to_invent_one():
    """A guessed step turns a gradient gate into a statement about step size."""
    assert V.default_steps(["theta_v"]) == {"theta_v": V.FD_STEPS_EXTRA["theta_v"]}
    assert V.default_steps(["a"]) == {"a": V.FD_STEPS["a"]}

    with pytest.raises(KeyError, match="no measured step"):
        V.default_steps(["gamma"])
