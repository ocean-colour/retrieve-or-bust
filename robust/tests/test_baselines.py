"""
Tests for :mod:`robust.rt.baselines` and :func:`robust.rt.validation.rrms` (M2 task 1).

The load-bearing test here is the **exact reproduction of the published rRMS
ladder** in ``context/RT/fig_rrms_ladder.csv``. That table was produced
independently, in NumPy, by the synthesis figure script; if the JAX path agrees with
it to rounding then the metric definition, the ``Rrs -> rrs`` conversion, ``u``, and
the Gordon coefficients are all consistent between the two. Every relative accuracy
claim from M3 onward is measured against this baseline, so it is worth pinning hard
now rather than discovering a discrepancy at M4.

The other theme is Gordon's **structural blindness**: it accepts ``phase_params``
and ``geometry`` and ignores both. Tests assert that rather than leaving it as a
docstring claim, because it is the reason the hybrid has something to beat.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import baselines as B
from robust.rt import conventions as C
from robust.rt import validation as V
from robust.rt.types import Geometry, IOPs, PhaseParams
from robust.tests.conftest import needs_l23

N = C.N_WAVE

#: The synthesis figure script's output, treated as the reference.
LADDER_CSV = Path(__file__).resolve().parents[2] / "context/RT/fig_rrms_ladder.csv"


def simple_iops(a=0.15, bb_p=0.003, shape=(N,)) -> IOPs:
    """IOPs with a flat spectrum, for algebraic checks."""
    return IOPs(
        a=jnp.full(shape, a),
        bb_w=jnp.broadcast_to(jnp.asarray(C.BB_W_L23), shape),
        bb_p=jnp.full(shape, bb_p),
    )


# ---------------------------------------------------------------- rrms ------


def test_rrms_known_answer():
    """A uniform 10% overprediction is 10% rRMS."""
    truth = jnp.asarray([1.0, 2.0, 4.0])

    assert float(V.rrms(truth, truth * 1.1)) == pytest.approx(10.0, rel=1e-5)


def test_rrms_is_zero_for_a_perfect_model():
    truth = jnp.asarray([1e-3, 5e-3, 2e-2])

    assert float(V.rrms(truth, truth)) == pytest.approx(0.0, abs=1e-10)


def test_rrms_is_relative_not_absolute():
    """The property that makes it usable across a decade of Rrs.

    Two errors of equal *relative* size score equally even when their absolute
    sizes differ by 1000x -- which is the whole reason the design specifies a
    relative metric: an absolute RMS over 350-750 nm would be a statement about
    the blue and nearly blind past 600 nm.
    """
    blue = jnp.asarray([2e-2])
    red = jnp.asarray([2e-5])

    assert float(V.rrms(blue, blue * 1.05)) == pytest.approx(
        float(V.rrms(red, red * 1.05)), rel=1e-6
    )


def test_rrms_reduces_along_an_axis():
    """``axis=0`` gives the per-wavelength ladder the design asks for."""
    truth = jnp.ones((7, 3))
    pred = truth * jnp.asarray([1.1, 1.2, 1.3])

    per_wave = V.rrms(truth, pred, axis=0)

    assert per_wave.shape == (3,)
    np.testing.assert_allclose(np.asarray(per_wave), [10.0, 20.0, 30.0], rtol=1e-4)


def test_rrms_is_jittable_and_differentiable():
    """It doubles as M3's training loss, so it must survive both."""
    truth = jnp.asarray([1e-3, 5e-3])

    assert float(jax.jit(V.rrms)(truth, truth * 1.01)) > 0.0

    grad = jax.grad(lambda p: V.rrms(truth, p))(truth * 1.01)

    assert np.all(np.isfinite(np.asarray(grad)))


# -------------------------------------------------------------- Gordon ------


def test_gordon_matches_the_algebra():
    """``rrs = G1 u + G2 u^2``, spelled out independently."""
    iops = simple_iops()
    u = float(iops.u[0])

    expected = B.G1_GORDON * u + B.G2_GORDON * u**2

    assert float(B.rrs_gordon(iops)[0]) == pytest.approx(expected, rel=1e-6)


def test_gordon_coefficients_match_the_synthesis_script():
    """The two numbers that make our rRMS comparable with the published ladder."""
    assert (B.G1_GORDON, B.G2_GORDON) == (0.0949, 0.0794)


def test_Rrs_gordon_is_rrs_gordon_through_the_interface():
    iops = simple_iops()

    np.testing.assert_allclose(
        np.asarray(B.Rrs_gordon(iops)),
        np.asarray(C.rrs_to_Rrs(B.rrs_gordon(iops))),
        rtol=1e-6,
    )


def test_gordon_ignores_geometry_and_phase_params():
    """Its defining limitation, asserted rather than merely documented.

    Standard Gordon has no solar-zenith term and no phase-function input, so a
    60 deg sun and a 0 deg sun get the same answer. L23 says the truth differs by
    a median 5.1%, which is precisely the gap M2's backbone and M3's emulator exist
    to close.
    """
    iops = simple_iops()
    at_nadir = B.rrs_gordon(
        iops, PhaseParams(B_p=jnp.asarray(0.010)), Geometry.nadir(jnp.asarray(0.0))
    )
    at_sixty = B.rrs_gordon(
        iops, PhaseParams(B_p=jnp.asarray(0.018)), Geometry.nadir(jnp.asarray(60.0))
    )

    np.testing.assert_array_equal(np.asarray(at_nadir), np.asarray(at_sixty))


def test_gordon_takes_the_forward_signature_positionally():
    """Interchangeable with ``forward`` / ``Rrs_ZTT``, so M4 can score in one loop."""
    iops = simple_iops()

    out = B.Rrs_gordon(
        iops,
        PhaseParams(B_p=jnp.asarray(0.0126)),
        Geometry.nadir(jnp.asarray(30.0)),
        C.canonical_wave(),
    )

    assert out.shape == (N,)


def test_gordon_is_jittable_and_differentiable():
    """It sits alongside the model on the same tooling."""
    iops = simple_iops()

    assert float(jnp.sum(jax.jit(B.Rrs_gordon)(iops))) > 0.0

    grads = jax.grad(lambda x: jnp.sum(B.rrs_gordon(x)))(iops)

    assert isinstance(grads, IOPs)
    # More absorption lowers rrs; more particulate backscatter raises it.
    assert float(grads.a[0]) < 0.0
    assert float(grads.bb_p[0]) > 0.0


def test_gordon_is_batched():
    iops = simple_iops(shape=(11, N))

    assert B.rrs_gordon(iops).shape == (11, N)


# ------------------------------------- against the reference data (fixture) --


def test_gordon_runs_on_the_cached_fixture(l23_small_batch):
    """Real numbers in CI: finite, positive, and the right shape."""
    rrs = B.rrs_gordon(l23_small_batch.iops)

    assert rrs.shape == l23_small_batch.Rrs.shape
    assert np.all(np.isfinite(np.asarray(rrs)))
    assert float(jnp.min(rrs)) > 0.0


def test_gordon_prediction_is_identical_across_zeniths(l23_small_batch):
    """The zenith blindness, on real data.

    M1 established that the IOP fields are identical across the three zenith
    files, so a zenith-blind model must return bit-identical predictions for all
    three -- while the reference ``Rrs`` does not. This is the cleanest possible
    statement of what Gordon cannot do.
    """
    rrs = np.asarray(B.rrs_gordon(l23_small_batch.iops))
    zenith = l23_small_batch.zenith

    np.testing.assert_array_equal(rrs[zenith == 0.0], rrs[zenith == 30.0])
    np.testing.assert_array_equal(rrs[zenith == 0.0], rrs[zenith == 60.0])

    truth = np.asarray(C.Rrs_to_rrs(l23_small_batch.Rrs))
    assert not np.array_equal(truth[zenith == 0.0], truth[zenith == 60.0])


def test_gordon_rrms_on_the_fixture_is_pinned(l23_small_batch):
    """Regression pin, runnable without the dataset.

    Values measured on the committed 50-scene fixture on 2026-08-01. The
    full-release numbers live in the test below; these exist so a change in the
    metric or the model fails in CI too.
    """
    truth = C.Rrs_to_rrs(l23_small_batch.Rrs)
    pred = B.rrs_gordon(l23_small_batch.iops)
    zenith = l23_small_batch.zenith

    for angle, expected in ((0.0, 4.9603), (30.0, 5.2267), (60.0, 8.4024)):
        mask = zenith == angle
        got = float(V.rrms(truth[mask], pred[mask]))
        assert got == pytest.approx(expected, abs=1e-3), f"{angle} deg: {got:.4f}"


# --------------------------------------- against the reference data (full) --


def read_published_ladder() -> dict[float, float]:
    """The ``std`` column of the synthesis rRMS ladder, keyed by wavelength."""
    with open(LADDER_CSV, newline="") as handle:
        return {float(row["lam"]): float(row["std"]) for row in csv.DictReader(handle)}


def test_published_ladder_is_readable():
    """Guards the reference itself: seven wavelengths, plausible magnitudes."""
    ladder = read_published_ladder()

    assert len(ladder) == 7
    assert min(ladder) == 400.0 and max(ladder) == 700.0
    assert all(1.0 < value < 20.0 for value in ladder.values())


@needs_l23
def test_gordon_reproduces_the_published_rrms_ladder(l23_batch):
    """**The task-1 gate.** Our JAX Gordon equals the synthesis NumPy Gordon.

    Reproduces ``context/RT/fig_rrms_ladder.csv`` (``std`` column) at Y = 0 -- the
    same 3320 scenes, the same fixed coefficients, the same relative-rRMS
    definition. Agreement to 1e-3 percentage points means the metric, the
    ``Rrs -> rrs`` conversion, ``u``, and the coefficients all match the
    independently-written reference implementation.
    """
    ladder = read_published_ladder()
    wave = np.asarray(l23_batch.wave)
    at_nadir = l23_batch.zenith == 0.0

    truth = C.Rrs_to_rrs(l23_batch.Rrs)[at_nadir]
    pred = B.rrs_gordon(l23_batch.iops)[at_nadir]

    for lam, published in ladder.items():
        j = int(np.argmin(np.abs(wave - lam)))
        got = float(V.rrms(truth[:, j], pred[:, j]))
        assert got == pytest.approx(published, abs=1e-3), (
            f"{lam:.0f} nm: got {got:.5f}%, published {published:.5f}%"
        )


@needs_l23
def test_gordon_is_worst_at_the_held_out_zenith(l23_batch):
    """Gordon's error is largest at 60 deg -- at every wavelength.

    This is the benchmark asymmetry the M3/M4 comparison rests on, and it cuts
    both ways: a hybrid win on the 60 deg hold-out is partly a win against a model
    evaluated outside its best geometry. Worth stating rather than banking.

    Note 30 deg is *not* uniformly worse than nadir (see the log): the fixed
    coefficients happen to suit ~30 deg better in the blue. So the honest claim is
    "60 deg is the clear loser", not "error grows with zenith".
    """
    wave = np.asarray(l23_batch.wave)
    truth = C.Rrs_to_rrs(l23_batch.Rrs)
    pred = B.rrs_gordon(l23_batch.iops)

    for lam in (400.0, 550.0, 700.0):
        j = int(np.argmin(np.abs(wave - lam)))
        scores = {
            angle: float(
                V.rrms(
                    truth[l23_batch.zenith == angle][:, j],
                    pred[l23_batch.zenith == angle][:, j],
                )
            )
            for angle in (0.0, 30.0, 60.0)
        }
        assert scores[60.0] > scores[0.0], f"{lam:.0f} nm: {scores}"
        assert scores[60.0] > scores[30.0], f"{lam:.0f} nm: {scores}"


@needs_l23
def test_gordon_rrms_per_wavelength_is_reported_shape(l23_batch):
    """The per-λ ladder comes out of one vectorised call, not a Python loop."""
    at_nadir = l23_batch.zenith == 0.0
    truth = C.Rrs_to_rrs(l23_batch.Rrs)[at_nadir]
    pred = B.rrs_gordon(l23_batch.iops)[at_nadir]

    ladder = V.rrms(truth, pred, axis=0)

    assert ladder.shape == (C.N_WAVE,)
    assert np.all(np.asarray(ladder) > 0.0)
