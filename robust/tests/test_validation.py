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

import warnings

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
    models = {"a": truth * 1.01, "b": truth * 0.98}
    masks = {"all": np.ones(10, dtype=bool), "half": np.arange(10) < 5}

    table = V.score_models(models, truth, masks)

    assert set(table) == {"a", "b"}
    assert table["a"]["all"] == pytest.approx(1.0, rel=1e-4)  # a uniform +1% error
    assert table["b"]["half"] == pytest.approx(2.0, rel=1e-4)


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


@pytest.fixture(scope="module")
def gate_fit(l23_small_batch):
    """One toy fit on the fixture's train split, shared by the gate tests."""
    from robust.rt.data import l23

    splits = l23.make_splits(l23_small_batch)
    emulator, _ = E.fit_l23(
        l23_small_batch, splits, config=E.EmulatorConfig(steps=400, eval_every=100)
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
    art. Measured on the fixture with a 400-step toy fit: hybrid ~0.53% against
    O25's ~0.9%.
    """
    scores = {name: row["held_out"] for name, row in gate_scores.items()}
    assert scores["hybrid"] < scores["o25"], (
        f"hybrid {scores['hybrid']:.3f}% did not beat the O25 refit "
        f"{scores['o25']:.3f}% on the held-out scenes"
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


def test_gate_gradient_correctness(jax_x64, l23_small_batch, gate_fit):
    """**THE GATE, second half.** ``jax.grad`` of the full hybrid matches central
    differences on every input.

    The property the eventual inversion depends on, and the one thing in M4 that
    is a hard pass/fail rather than a comparison. Evaluated at 45 deg, between
    O25's table nodes, so the same harness can score every model without landing
    on a lookup kink.
    """
    emulator, _ = gate_fit
    batch = l23_small_batch
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
