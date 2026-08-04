"""
Tests for :mod:`robust.rt.hybrid` (M3 task 2) — the milestone's acceptance gate.

Everything here runs on the committed 50-scene fixture (no ``$OS_COLOR``), with one
toy emulator fit shared module-wide. Six groups:

**The gate.** The hybrid must beat ``mode="ztt"`` on the train split at each of the
three solar zeniths separately — the load-bearing criterion from Q5, since ZTT alone
already beats Gordon everywhere — and the emulator's contribution is printed as a
number, not implied. Beating Gordon is kept as the weaker plan-level floor.

**The mode flag cannot change the physics.** ``mode="ztt"`` is bitwise
``ztt.rrs_ZTT`` and touches no emulator at all, so the analytic path needs neither
the ML stack nor the weights file.

**Additivity lives in ``rrs`` space.** ``hybrid == ztt + emulator`` exactly below
the surface, and fails above it by a physically meaningful ~1.3e-4 sr⁻¹ — the
air-water interface is non-linear, not a rounding effect.

**The gradient gate.** ``jax.grad`` of the full hybrid ``forward`` (emulator in the
path) against central finite differences, per-variable steps, float64 — the same
protocol as M2's gate in ``test_ztt.py``.

**Throughput has not collapsed.** Jitted hybrid vs jitted ZTT, with only a weak
ratio asserted; the reference numbers live in the implementation record.

**The extrapolation warning is operational.** ``DomainWarning`` fires outside the
trained range, stays quiet inside it, obeys ``check_domain=False``, and is skipped
under ``jit`` (the check needs concrete values — documented, not accidental).

The full-data numbers (packaged hybrid 0.30% etc.) are *not* asserted here; the
packaged-weights test is explicitly labelled as in-sample.
"""

from __future__ import annotations

import time
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

#: The three solar zeniths in the fixture, degrees — the gate is per-zenith.
ZENITHS = (0.0, 30.0, 60.0)


def take_rows(tree, index):
    """Index every leaf of a per-sample pytree (IOPs, PhaseParams) by row."""
    return jax.tree_util.tree_map(lambda leaf: leaf[index], tree)


def batch_args(batch):
    """The positional arguments of :func:`robust.rt.hybrid.forward` for a batch."""
    return batch.iops, batch.phase_params, batch.geometry, batch.wave


def tiny_args():
    """Minimal synthetic inputs for tests that never evaluate the model."""
    wave = jnp.asarray([440.0, 550.0])
    iops = IOPs(
        a=jnp.asarray([0.15, 0.12]),
        bb_w=C.bb_w(wave),
        bb_p=jnp.asarray([0.003, 0.003]),
    )
    return (
        iops,
        PhaseParams(B_p=jnp.asarray(0.0126)),
        Geometry.nadir(jnp.asarray(30.0)),
        wave,
    )


@pytest.fixture(scope="module")
def toy_fit(l23_small_batch):
    """One 400-step ``fit_l23`` on the fixture's ``scene_train`` split, run once.

    Toy-sized (the real 3000-step training run lives in
    ``design/py/train_emulator.py``), but a real fit on real L23 numbers: ~3.5 s,
    deterministic from the default seed, and enough for the hybrid to reach
    ~0.53% train rRMS against ZTT's 3.9-7.4%.
    """
    from robust.rt.data import l23

    splits = l23.make_splits(l23_small_batch)
    config = E.EmulatorConfig(steps=400, eval_every=100)
    emulator, _history = E.fit_l23(l23_small_batch, splits, config=config)
    return emulator, splits


@pytest.fixture(scope="module")
def scores(toy_fit, l23_small_batch):
    """Per-zenith train-split rRMS for ztt / hybrid / gordon, computed once.

    Scored with ``validation.rrms`` in ``rrs`` space via ``rrs_forward`` (design
    §6). ``check_domain=False`` on the hybrid call because the *full* fixture
    batch includes the held-out scenes, whose features fall slightly outside this
    toy fit's training range — the warning would be correct, but it is scored on
    the train mask below and the warning has its own tests.
    """
    emulator, splits = toy_fit
    batch = l23_small_batch
    truth = C.Rrs_to_rrs(batch.Rrs)
    outputs = {
        "ztt": H.rrs_forward(*batch_args(batch), "ztt"),
        "hybrid": H.rrs_forward(
            *batch_args(batch), "hybrid", emulator=emulator, check_domain=False
        ),
        "gordon": B.rrs_gordon(*batch_args(batch)),
    }
    table = {}
    for z in ZENITHS:
        mask = splits.scene_train & (batch.zenith == z)
        table[z] = {
            name: float(V.rrms(truth[mask], out[mask])) for name, out in outputs.items()
        }
    return table


# ----------------------------------------------------------------- the gate --


def test_gate_hybrid_beats_ztt_at_every_zenith(scores):
    """**THE GATE.** Train-split hybrid rRMS beats ``mode="ztt"`` at 0/30/60°.

    This is the strengthened criterion from Q5: ZTT alone already beats Gordon on
    every split, so beating the *backbone* — at each solar zenith separately, not
    just pooled — is what proves the learned half earns its place. Measured with
    the 400-step toy fit: ztt 4.15/3.90/7.41% vs hybrid 0.53/0.53/0.54%.
    """
    for z in ZENITHS:
        assert scores[z]["hybrid"] < scores[z]["ztt"], (
            f"zenith {z:.0f}: hybrid {scores[z]['hybrid']:.3f}% did not beat "
            f"ztt {scores[z]['ztt']:.3f}%"
        )


def test_gate_hybrid_beats_gordon_at_every_zenith(scores):
    """The plan-level floor: the hybrid also beats standard Gordon at 0/30/60°.

    Deliberately the *weaker* statement — an emulator outputting exactly zero
    would pass this one (ZTT already beats Gordon), so it is kept only as the
    coding plan's original wording, not as evidence about the learned half.
    """
    for z in ZENITHS:
        assert scores[z]["hybrid"] < scores[z]["gordon"], (
            f"zenith {z:.0f}: hybrid {scores[z]['hybrid']:.3f}% did not beat "
            f"Gordon {scores[z]['gordon']:.3f}%"
        )


def test_emulator_contribution_is_reported_as_a_number(scores):
    """The rRMS reduction over the backbone is printed per zenith, and positive.

    M3 requires the emulator's contribution *reported as a number* (run with
    ``-s`` to see it) — a silently passing gate would not satisfy that. The
    assertion is the same per-zenith claim as the gate, restated as a reduction.
    """
    print()
    print("emulator contribution, fixture train split (rRMS, %):")
    print(f"{'zenith':>8} {'ztt':>8} {'hybrid':>8} {'reduction':>10}")
    for z in ZENITHS:
        reduction = scores[z]["ztt"] - scores[z]["hybrid"]
        print(
            f"{z:8.0f} {scores[z]['ztt']:8.3f} {scores[z]['hybrid']:8.3f} "
            f"{reduction:10.3f}"
        )
        assert reduction > 0.0, f"no contribution at zenith {z:.0f}"


# ------------------------------------- mode="ztt" is exactly the backbone ----


def test_mode_ztt_is_bitwise_the_backbone_and_needs_no_emulator(l23_small_batch):
    """``mode="ztt"`` returns ``ztt.rrs_ZTT`` bitwise and never touches an emulator.

    Bitwise (``jnp.array_equal``), not approximately: the mode flag must be unable
    to silently change the physics. And the analytic path must not depend on the
    ML stack or the packaged weights, so a deliberately absurd ``object()`` passed
    as the emulator must be ignored entirely — ``mode="ztt"`` returns before the
    emulator (or the default-weights loader) is ever resolved.
    """
    args = batch_args(l23_small_batch)
    reference = Z.rrs_ZTT(*args)

    assert jnp.array_equal(H.rrs_forward(*args, "ztt"), reference)
    assert jnp.array_equal(H.rrs_forward(*args, "ztt", emulator=object()), reference)
    assert jnp.array_equal(
        H.forward(*args, "ztt", emulator=object()), C.rrs_to_Rrs(reference)
    )


# ------------------------------------- additivity: rrs yes, Rrs no -----------


def test_additivity_holds_exactly_in_rrs_space(toy_fit, l23_small_batch):
    """``rrs_forward(hybrid) == rrs_forward(ztt) + rrs_forward(emulator)``, bitwise.

    The module docstring claims the decomposition is *exact* below the surface —
    the hybrid literally computes ``rrs_ztt + delta_rrs`` — and the three modes
    recompute the same deterministic terms, so the sums agree bitwise (measured:
    max relative difference 0.0), not merely to tolerance.
    """
    emulator, _ = toy_fit
    args = batch_args(l23_small_batch)

    ztt_part = H.rrs_forward(*args, "ztt")
    emu_part = H.rrs_forward(*args, "emulator", emulator=emulator, check_domain=False)
    hybrid = H.rrs_forward(*args, "hybrid", emulator=emulator, check_domain=False)

    assert jnp.array_equal(hybrid, ztt_part + emu_part)


def test_additivity_fails_in_Rrs_space_by_a_physical_amount(toy_fit, l23_small_batch):
    """``forward(hybrid) != forward(ztt) + forward(emulator)``: the interface is real.

    Lee's ``A·rrs/(1 − B·rrs)`` is non-linear, so the same decomposition above the
    surface must fail by far more than rounding. Measured on this exact
    (deterministic) fit: max discrepancy 1.28e-4 sr⁻¹ — a couple of percent of a
    typical ocean ``Rrs`` — so 1e-6 sr⁻¹ is asserted with two decades of headroom
    while staying far above any float32 rounding (~1e-9 sr⁻¹ here).
    """
    emulator, _ = toy_fit
    args = batch_args(l23_small_batch)

    ztt_part = H.forward(*args, "ztt")
    emu_part = H.forward(*args, "emulator", emulator=emulator, check_domain=False)
    hybrid = H.forward(*args, "hybrid", emulator=emulator, check_domain=False)

    discrepancy = float(jnp.max(jnp.abs(hybrid - (ztt_part + emu_part))))
    assert discrepancy > 1e-6, f"interface non-linearity vanished: {discrepancy:.3e}"


# ------------------------------------------------------- the gradient gate ---


@pytest.mark.parametrize(
    ("name", "step"),
    [("a", 1e-6), ("bb_p", 1e-9), ("B_p", 1e-8), ("theta_s", 1e-3)],
)
def test_gradient_matches_finite_differences_through_the_emulator(
    jax_x64, toy_fit, l23_small_batch, name, step
):
    """**The hard gate.** ``jax.grad`` of the hybrid ``forward`` agrees with FD.

    The emulator is in the differentiation path — this is the property the future
    inversion depends on, now for the full model rather than the backbone alone.
    Same protocol as ``test_ztt.py``: float64 with the dtype pinned on the
    *arrays* (a Python-float perturbation would silently be float64 anyway and
    prove nothing), and a per-variable step, since ``theta_s`` is O(30) and wants
    h ~ 1e-3 while the IOP-like variables want ~1e-7-1e-9 — no single step clears
    the tolerance for all four. Measured agreement with these steps: a 2.7e-9,
    bb_p 6.8e-11, B_p 1.7e-9, theta_s 5.4e-10.

    Three real fixture samples at the interior zenith (30°), so the ``theta_s``
    step cannot leave the physical domain; the steps for the IOP variables are
    far below the fixture minima (bb_p ≥ 3.2e-4), so no perturbed input goes
    negative — a larger step would, and ZTT returns NaN there, which is why the
    finite difference is asserted finite before it is compared.

    ``check_domain=False`` because the domain check is not what this test gates —
    the perturbed IOPs do leave the fixture's trained range, and a warning per
    finite-difference evaluation would drown the run. That it *works* with the
    check left on is pinned separately, by
    :func:`test_grad_of_one_input_does_not_break_the_domain_check`.
    """
    emulator, _ = toy_fit
    batch = l23_small_batch
    dtype = jnp.float64
    rows = np.where(batch.zenith == 30.0)[0][:3]

    a0 = jnp.asarray(np.asarray(batch.iops.a)[rows], dtype=dtype)
    bb_w0 = jnp.asarray(np.asarray(batch.iops.bb_w)[rows], dtype=dtype)
    bb_p0 = jnp.asarray(np.asarray(batch.iops.bb_p)[rows], dtype=dtype)
    B_p0 = jnp.asarray(np.asarray(batch.phase_params.B_p)[rows], dtype=dtype)
    theta0 = jnp.asarray(np.asarray(batch.geometry.theta_s)[rows], dtype=dtype)
    wave = jnp.asarray(np.asarray(batch.wave), dtype=dtype)

    def scalar(shift):
        """Mean hybrid Rrs with one variable shifted by the scalar ``shift``."""
        offsets = dict.fromkeys(("a", "bb_p", "B_p", "theta_s"), 0.0)
        offsets[name] = shift
        iops = IOPs(a=a0 + offsets["a"], bb_w=bb_w0, bb_p=bb_p0 + offsets["bb_p"])
        phase = PhaseParams(B_p=B_p0 + offsets["B_p"])
        geometry = Geometry.nadir(theta0 + offsets["theta_s"])
        return jnp.mean(
            H.forward(
                iops,
                phase,
                geometry,
                wave,
                "hybrid",
                emulator=emulator,
                check_domain=False,
            )
        )

    analytic = float(jax.grad(scalar)(jnp.asarray(0.0, dtype=dtype)))
    h = jnp.asarray(step, dtype=dtype)
    numeric = float((scalar(h) - scalar(-h)) / (2.0 * h))

    # A step that left the physical domain shows up as NaN; that is a wrong
    # *step*, not a wrong gradient, so it must fail with this message rather
    # than leak into the comparison.
    assert np.isfinite(numeric), f"d/d{name}: step {step:g} left the domain"
    assert analytic == pytest.approx(numeric, rel=1e-6), (
        f"d/d{name}: autodiff {analytic:.10e} vs finite difference {numeric:.10e}"
    )


@pytest.mark.parametrize("name", ["a", "bb_p", "B_p", "theta_s"])
def test_grad_of_one_input_does_not_break_the_domain_check(
    toy_fit, l23_small_batch, name
):
    """Regression: ``jax.grad`` w.r.t. **one** input must not trip the domain check.

    ``jax.grad`` traces only the variable it differentiates, so the other inputs
    stay concrete. A guard that sampled just a couple of leaves therefore reported
    "not traced" while differentiating w.r.t. ``bb_p`` or ``B_p``, and
    ``out_of_domain``'s ``np.asarray`` died with a
    ``TracerArrayConversionError`` — with ``check_domain`` at its default, i.e. on
    the path any caller would take, and specifically for backscattering, which is
    what an inversion retrieves. The guard now inspects every leaf; this pins each
    input separately so a partial guard cannot come back.
    """
    emulator, _ = toy_fit
    batch = l23_small_batch
    rows = np.where(batch.zenith == 30.0)[0][:2]
    parts = {
        "a": jnp.asarray(np.asarray(batch.iops.a)[rows]),
        "bb_w": jnp.asarray(np.asarray(batch.iops.bb_w)[rows]),
        "bb_p": jnp.asarray(np.asarray(batch.iops.bb_p)[rows]),
        "B_p": jnp.asarray(np.asarray(batch.phase_params.B_p)[rows]),
        "theta_s": jnp.asarray(np.asarray(batch.geometry.theta_s)[rows]),
    }

    def scalar(value):
        """Mean hybrid ``Rrs`` with ``name`` supplied as the traced argument."""
        p = dict(parts, **{name: value})
        return jnp.mean(
            H.forward(
                IOPs(a=p["a"], bb_w=p["bb_w"], bb_p=p["bb_p"]),
                PhaseParams(B_p=p["B_p"]),
                Geometry.nadir(p["theta_s"]),
                batch.wave,
                "hybrid",
                emulator=emulator,
            )
        )

    # Default check_domain=True: the whole point is that the guard skips cleanly.
    grad = jax.grad(scalar)(parts[name])
    assert np.all(np.isfinite(np.asarray(grad)))


# ------------------------------------------------------------- throughput ----


def test_hybrid_throughput_has_not_collapsed(toy_fit, l23_small_batch):
    """Jitted hybrid ``rrs_forward`` costs less than 25× jitted ZTT.

    A weak, robust bound only: hard timing numbers are not reproducible
    assertions on shared CI. The reference measurement is in the implementation
    record — ZTT 3.1 ms on the full 9960×81 batch and the hybrid ~4.8× that (the
    same ratio measured here on the 150-sample fixture) — so 25× only guards
    against a collapse (an accidental un-jittable path, a recompile per call),
    not against drift. Printed for the log; run with ``-s`` to see it.
    """
    emulator, _ = toy_fit
    args = batch_args(l23_small_batch)

    ztt_fn = jax.jit(lambda i, p, g, w: H.rrs_forward(i, p, g, w, "ztt"))
    hybrid_fn = jax.jit(
        lambda i, p, g, w: H.rrs_forward(i, p, g, w, "hybrid", emulator=emulator)
    )

    def best_of(fn, repeats=5):
        """Min wall time over ``repeats`` after a compile/warm-up call."""
        fn(*args).block_until_ready()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn(*args).block_until_ready()
            times.append(time.perf_counter() - start)
        return min(times)

    t_ztt = best_of(ztt_fn)
    t_hybrid = best_of(hybrid_fn)
    ratio = t_hybrid / t_ztt

    print()
    print(
        f"throughput, jitted, {l23_small_batch.n_sample}x"
        f"{l23_small_batch.n_wave} fixture: ztt {t_ztt * 1e3:.3f} ms, "
        f"hybrid {t_hybrid * 1e3:.3f} ms, ratio {ratio:.2f}x"
    )
    assert ratio < 25.0, f"hybrid is {ratio:.1f}x ZTT — throughput collapsed"


# ----------------------------------------------------------- DomainWarning ---


def out_of_range_inputs(batch, zenith=75.0, n=4):
    """A few real fixture spectra pushed to an untrained solar zenith.

    ``cos_theta_s`` *decreases* with zenith, so a low sun (75° against the
    trained 0/30/60°, i.e. cos 0.26 against [0.5, 1.0]) breaches the domain's
    *lower* bound.
    """
    rows = slice(0, n)
    return (
        take_rows(batch.iops, rows),
        take_rows(batch.phase_params, rows),
        Geometry.nadir(jnp.full((n,), zenith)),
        batch.wave,
    )


def in_range_inputs(batch, splits):
    """The training rows themselves — inside the domain by construction."""
    rows = np.asarray(splits.scene_train)
    return (
        take_rows(batch.iops, rows),
        take_rows(batch.phase_params, rows),
        Geometry.nadir(batch.geometry.theta_s[rows]),
        batch.wave,
    )


def test_out_of_range_zenith_warns(toy_fit, l23_small_batch):
    """A solar zenith the training never saw raises ``DomainWarning``.

    75° against a 0/30/60° training set — the exact situation M3 measured as
    seed-dependent and occasionally worse than the backbone, and the one JXP
    ruled must not pass silently.
    """
    emulator, _ = toy_fit

    with pytest.warns(H.DomainWarning, match="outside its training range"):
        H.rrs_forward(
            *out_of_range_inputs(l23_small_batch), "hybrid", emulator=emulator
        )


def test_in_range_inputs_do_not_warn(toy_fit, l23_small_batch):
    """Inputs inside the trained domain stay silent — the warning is not noise.

    A warning that also fires in range would be filtered out by every caller
    within a week, so silence here is as load-bearing as the warning above.
    """
    emulator, splits = toy_fit

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H.rrs_forward(
            *in_range_inputs(l23_small_batch, splits), "hybrid", emulator=emulator
        )

    domain_warnings = [w for w in caught if issubclass(w.category, H.DomainWarning)]
    assert domain_warnings == []


def test_check_domain_false_silences_the_warning(toy_fit, l23_small_batch):
    """``check_domain=False`` is the documented opt-out for a study that means it."""
    emulator, _ = toy_fit

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H.rrs_forward(
            *out_of_range_inputs(l23_small_batch),
            "hybrid",
            emulator=emulator,
            check_domain=False,
        )

    domain_warnings = [w for w in caught if issubclass(w.category, H.DomainWarning)]
    assert domain_warnings == []


def test_domain_check_is_skipped_under_jit(toy_fit, l23_small_batch):
    """A jitted call at 75° does not warn: the check needs concrete values.

    Deliberate and documented on ``forward`` — under ``jit`` the inputs are
    tracers, there is nothing concrete to compare against the domain, and a check
    that cannot run must not pretend to. This pins that the jitted hot path
    neither warns nor crashes.
    """
    emulator, _ = toy_fit
    fn = jax.jit(
        lambda i, p, g, w: H.rrs_forward(i, p, g, w, "hybrid", emulator=emulator)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*out_of_range_inputs(l23_small_batch)).block_until_ready()

    domain_warnings = [w for w in caught if issubclass(w.category, H.DomainWarning)]
    assert domain_warnings == []
    assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------- mode validation --


@pytest.mark.parametrize("fn", [H.rrs_forward, H.forward])
def test_unknown_mode_raises_and_lists_the_valid_modes(fn):
    """A typo'd mode raises ``ValueError`` naming the offender and all of MODES.

    Silently defaulting would corrupt a comparison table with no visible symptom
    (the docstring's own argument), and the message must let the caller fix it
    without opening the source — checked on both public entry points.
    """
    with pytest.raises(ValueError, match="mode must be one of") as excinfo:
        fn(*tiny_args(), "gordon")

    message = str(excinfo.value)
    assert "'gordon'" in message
    for mode in H.MODES:
        assert mode in message


# --------------------------------------------------------- packaged weights --


def test_packaged_weights_load_and_improve_on_ztt(l23_small_batch):
    """``load_default()`` returns a trained emulator that improves on ZTT here.

    **Largely an in-sample check, and honestly so:** the packaged weights were
    trained on the FULL L23 ``scene_train`` split, which overlaps most of these
    50 fixture scenes, so this pins that the shipped file loads, carries its
    training domain, and is wired into ``forward`` correctly — it is NOT evidence
    of generalisation. (The measured fixture numbers: ZTT 5.30%, packaged hybrid
    0.26%, scored in ``rrs`` per design §6.)
    """
    emulator = E.load_default()
    assert emulator.domain is not None

    batch = l23_small_batch
    truth = C.Rrs_to_rrs(batch.Rrs)
    rrs_ztt = H.rrs_forward(*batch_args(batch), "ztt")
    rrs_hybrid = H.rrs_forward(*batch_args(batch), "hybrid", emulator=emulator)

    assert float(V.rrms(truth, rrs_hybrid)) < float(V.rrms(truth, rrs_ztt))
    # And the public Rrs-space entry point runs with it end to end.
    Rrs = H.forward(*batch_args(batch), "hybrid", emulator=emulator)
    assert np.all(np.isfinite(np.asarray(Rrs)))
