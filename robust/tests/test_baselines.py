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

The O25 half (M4 task 1) repeats both themes for the stronger comparison model.
Its load-bearing test is the **provenance gate**: refitting on the ``scene_train``
split must reproduce the embedded :data:`robust.rt.baselines.O25_L23_REFIT`, so the
committed constants cannot drift from the code that made them. Around it sit the
same kinds of pins as Gordon's -- hand-checked algebra, deliberate blind spots
(``wave`` and ``phase_params`` ignored *by the model's design*), the gradient gate,
and the fairness of the fitting objective (weighted vs the paper's unweighted).
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
from robust.rt.data import l23
from robust.rt.types import Geometry, IOPs, PhaseParams
from robust.tests.conftest import needs_l23, needs_pb24

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


# ------------------------------------------- O25: the table and the algebra --


def albedo_iops(w_bw=0.10, w_bp=0.05, shape=(N,)) -> IOPs:
    """IOPs engineered so O25's two albedos come out to these round numbers.

    With ``a + bb = 1`` the albedos equal the backscattering values themselves:
    ``w_bw = bb_w / (a + bb) = bb_w`` and likewise for ``w_bp``.
    """
    return IOPs(
        a=jnp.full(shape, 1.0 - w_bw - w_bp),
        bb_w=jnp.full(shape, w_bw),
        bb_p=jnp.full(shape, w_bp),
    )


def o25_table() -> np.ndarray:
    """:data:`O25_L23_REFIT` as the float32 array the implementation works in."""
    return np.asarray(jnp.asarray(B.O25_L23_REFIT))


def test_o25_matches_the_algebra():
    """``Rrs = (Gw0 + Gw1 w_bw) w_bw + (Gp0 + Gp1 w_bp) w_bp``, spelled out by hand.

    The expected value is assembled in the test from the first table row and the
    engineered albedos (0.10 and 0.05), never copied from the implementation's
    output -- so a transposed coefficient or swapped albedo cannot hide.
    """
    iops = albedo_iops(w_bw=0.10, w_bp=0.05)
    _, Gw0, Gw1, Gp0, Gp1 = B.O25_L23_REFIT[0]

    expected = (Gw0 + Gw1 * 0.10) * 0.10 + (Gp0 + Gp1 * 0.05) * 0.05
    got = float(B.Rrs_o25(iops, geometry=Geometry.nadir(jnp.asarray(0.0)))[0])

    # rel 1e-6: the model evaluates in float32 (eps ~1.2e-7) against a float64
    # hand computation; measured agreement is 4.3e-9, so this is pure headroom.
    assert got == pytest.approx(expected, rel=1e-6)


def test_o25_coefficients_return_the_table_rows_exactly():
    """At each tabulated angle the interpolation returns that row, bit for bit.

    The same guard as the TT2017 node test in ``test_ztt.py``: it pins the table
    transcription and the lookup in one shot. ``jnp.interp`` at a node adds a
    zero-length lerp step, so exact equality is the correct expectation.
    """
    table = o25_table()

    for i, angle in enumerate((0.0, 30.0, 60.0)):
        got = np.asarray([float(g) for g in B.o25_coefficients(jnp.asarray(angle))])
        np.testing.assert_array_equal(got, table[i, 1:], err_msg=f"{angle} deg")


def test_o25_coefficients_interpolate_linearly_between_rows():
    """Halfway between two rows is their mean -- how the published model is used too."""
    table = o25_table()

    got = np.asarray([float(g) for g in B.o25_coefficients(jnp.asarray(15.0))])

    # rtol 1e-6: measured bit-exact; the slack only forgives a re-associated lerp.
    np.testing.assert_allclose(got, 0.5 * (table[0, 1:] + table[1, 1:]), rtol=1e-6)


def test_o25_coefficients_clamp_flat_outside_the_table():
    """Beyond [0, 60] deg the end row is held rather than extrapolated as a ramp.

    The conservative choice the docstring promises: a linear ramp on ``Gp1`` past
    60 deg grows without bound, while a held value at least stays inside the
    fitted family. So 90 deg must equal the 60 deg row and -10 deg the 0 deg row,
    bitwise -- any difference means someone swapped ``jnp.interp`` for a scheme
    that extrapolates.
    """
    table = o25_table()

    hi = np.asarray([float(g) for g in B.o25_coefficients(jnp.asarray(90.0))])
    lo = np.asarray([float(g) for g in B.o25_coefficients(jnp.asarray(-10.0))])

    np.testing.assert_array_equal(hi, table[2, 1:])
    np.testing.assert_array_equal(lo, table[0, 1:])


def test_o25_coefficients_are_differentiable_in_theta_s():
    """``jax.grad`` through the lookup gives the segment slope -- zero when clamped.

    Differentiability in the geometry is part of the shared-harness contract; the
    slope is computed here from the table, so the assertion is independent of the
    implementation. rel 1e-5: both sides are float32 (eps ~1.2e-7); measured
    agreement is exact to the printed 7 digits.
    """
    table = o25_table()
    grad_gp1 = jax.grad(lambda t: B.o25_coefficients(t)[3])

    slope = (table[1, 4] - table[0, 4]) / 30.0
    assert float(grad_gp1(jnp.asarray(15.0))) == pytest.approx(slope, rel=1e-5)
    assert float(grad_gp1(jnp.asarray(90.0))) == 0.0  # flat clamp, flat gradient


# --------------------------- O25: blind spots, and the contrast with Gordon --


def test_o25_ignores_wave():
    """Two different wavelength grids give bitwise-identical output.

    This is O25's design, not a shortcut in our implementation: its coefficients
    are wavelength-independent *by construction* -- λ enters only through the
    IOPs -- so the blind spot must stay deliberate, and a test is how it does.
    """
    iops = simple_iops()
    geometry = Geometry.nadir(jnp.asarray(30.0))

    on_canonical = B.Rrs_o25(iops, None, geometry, C.canonical_wave())
    on_shifted = B.Rrs_o25(iops, None, geometry, C.canonical_wave() + 100.0)

    np.testing.assert_array_equal(np.asarray(on_canonical), np.asarray(on_shifted))


def test_o25_ignores_phase_params():
    """Two very different ``B_p`` values give bitwise-identical output.

    Again the model's design rather than our laziness: O25 has no phase-function
    input at all -- its calibration set had *prescribed* Fournier-Forand phase
    functions, so the shape is baked into the fitted coefficients. That gap is
    what the ZTT backbone exists to address, and this test keeps it visible.
    """
    iops = simple_iops()
    geometry = Geometry.nadir(jnp.asarray(30.0))

    low = B.Rrs_o25(iops, PhaseParams(B_p=jnp.asarray(0.010)), geometry)
    high = B.Rrs_o25(iops, PhaseParams(B_p=jnp.asarray(0.018)), geometry)

    np.testing.assert_array_equal(np.asarray(low), np.asarray(high))


def test_o25_requires_a_geometry_where_gordon_does_not():
    """``geometry=None`` raises for O25 and is legitimate for Gordon.

    The contrast is the point: O25's coefficients are indexed by solar zenith, so
    silently substituting a default angle would produce plausible numbers from
    the wrong coefficients; Gordon genuinely has nothing to miss.
    """
    iops = simple_iops()

    with pytest.raises(ValueError, match="solar zenith"):
        B.Rrs_o25(iops)
    with pytest.raises(ValueError, match="solar zenith"):
        B.rrs_o25(iops)

    assert np.all(np.isfinite(np.asarray(B.rrs_gordon(iops))))  # geometry=None


def test_o25_takes_the_forward_signature_and_broadcasts_per_sample():
    """Interchangeable with ``forward``, and coefficients broadcast per sample.

    The mixed-zenith batch is where a broadcasting mistake in the per-sample
    coefficients against the per-wavelength albedos would bite, so each batched
    row must equal the corresponding single-sample call bit for bit.
    """
    out = B.Rrs_o25(
        simple_iops(),
        PhaseParams(B_p=jnp.asarray(0.0126)),
        Geometry.nadir(jnp.asarray(30.0)),
        C.canonical_wave(),
    )
    assert out.shape == (N,)

    angles = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)
    batched = np.asarray(
        B.Rrs_o25(simple_iops(shape=(6, N)), None, Geometry.nadir(jnp.asarray(angles)))
    )
    assert batched.shape == (6, N)

    for i, angle in enumerate(angles):
        single = B.Rrs_o25(simple_iops(), None, Geometry.nadir(jnp.asarray(angle)))
        np.testing.assert_array_equal(
            batched[i], np.asarray(single), err_msg=f"{angle} deg"
        )


def test_rrs_o25_is_Rrs_o25_through_the_interface():
    """The pair is the reverse of Gordon's: here ``Rrs`` is the primitive.

    O25 is defined (and fitted) in ``Rrs``; ``rrs_o25`` converts for scoring.
    """
    iops = simple_iops()
    geometry = Geometry.nadir(jnp.asarray(30.0))

    np.testing.assert_allclose(
        np.asarray(B.rrs_o25(iops, None, geometry)),
        np.asarray(C.Rrs_to_rrs(B.Rrs_o25(iops, None, geometry))),
        rtol=1e-6,
    )


# --------------------------------------------------- O25: the gradient gate --


def fd_gradient(fn, x, h):
    """Central finite difference of a scalar ``fn`` at ``x`` with step ``h``."""
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


@pytest.mark.parametrize(
    ("name", "step"),
    [("a", 1e-6), ("bb_p", 1e-9), ("theta_s", 1e-3)],
)
def test_o25_gradient_matches_finite_differences(jax_x64, name, step):
    """The same hard gate ZTT passes: one harness scores every model.

    Run under ``jax_x64`` with the dtype pinned on the arrays (a Python-float
    perturbation would silently compute in float64 and prove nothing). Steps are
    per variable, as in ``test_ztt.py``: ``a`` is O(0.1), ``bb_p`` O(1e-3), and
    ``theta_s`` O(15) wants the largest step -- sweeping it shows h=1e-3 gives
    1.1e-9 relative agreement while 1e-8 degrades to 2.4e-4 as cancellation takes
    over. Evaluated at 15 deg, strictly *inside* a table segment: at a node
    (0/30/60) the piecewise-linear lookup has a kink where one-sided ``jax.grad``
    and a central difference legitimately disagree. Tolerance 1e-6 against
    measured agreement of 4.4e-11 (``a``), 4.7e-11 (``bb_p``), 1.1e-9
    (``theta_s``).
    """
    dtype = jnp.float64
    base = {
        "a": jnp.asarray(0.15, dtype=dtype),
        "bb_p": jnp.asarray(0.003, dtype=dtype),
        "theta_s": jnp.asarray(15.0, dtype=dtype),
    }
    wave = jnp.asarray([440.0, 550.0, 660.0], dtype=dtype)
    bb_w = jnp.asarray(C.bb_w(wave), dtype=dtype)

    def scalar(value):
        p = dict(base, **{name: value})
        iops = IOPs(
            a=jnp.full(wave.shape, p["a"], dtype=dtype),
            bb_w=bb_w,
            bb_p=jnp.full(wave.shape, p["bb_p"], dtype=dtype),
        )
        return jnp.sum(B.rrs_o25(iops, None, Geometry.nadir(p["theta_s"]), wave))

    analytic = float(jax.grad(scalar)(base[name]))
    numeric = float(fd_gradient(scalar, base[name], jnp.asarray(step, dtype=dtype)))

    assert analytic == pytest.approx(numeric, rel=1e-6), (
        f"d/d{name}: autodiff {analytic:.10e} vs finite difference {numeric:.10e}"
    )


# ---------------------------------- O25 against the reference data (fixture) --


def test_o25_runs_on_the_cached_fixture(l23_small_batch):
    """Real numbers in CI: finite, positive, and the right shape."""
    rrs = B.rrs_o25(
        l23_small_batch.iops,
        l23_small_batch.phase_params,
        l23_small_batch.geometry,
        l23_small_batch.wave,
    )

    assert rrs.shape == l23_small_batch.Rrs.shape
    assert np.all(np.isfinite(np.asarray(rrs)))
    assert float(jnp.min(rrs)) > 0.0


def test_o25_rrms_on_the_fixture_is_pinned(l23_small_batch):
    """Regression pin for the model *and* the embedded table together.

    Values measured on the committed 50-scene fixture on 2026-08-06 (abs 1e-3,
    matching the Gordon pin above). Note the contrast with Gordon's 4.96 / 5.23 /
    8.40: O25's zenith-indexed refit is not only ~8x better, its error is nearly
    *uniform* across zeniths -- which is exactly what indexing the geometry buys.
    """
    truth = C.Rrs_to_rrs(l23_small_batch.Rrs)
    pred = B.rrs_o25(l23_small_batch.iops, None, l23_small_batch.geometry)
    zenith = l23_small_batch.zenith

    for angle, expected in ((0.0, 0.6330), (30.0, 0.6394), (60.0, 0.6398)):
        mask = zenith == angle
        got = float(V.rrms(truth[mask], pred[mask]))
        assert got == pytest.approx(expected, abs=1e-3), f"{angle} deg: {got:.4f}"


def test_the_fixture_respects_the_o25_validity_ceiling(l23_small_batch):
    """The fixture's brightest scene sits far below the fitted ceiling."""
    assert float(jnp.max(l23_small_batch.Rrs)) < B.O25_RRS_CEILING


# ------------------------------------------------------------- fitting O25 --


def test_fit_o25_is_deterministic(l23_small_batch):
    """Two identical calls agree bit for bit.

    Closed-form ``lstsq`` -- no seed, no learning rate, no stopping rule -- so
    the equality is exact by construction and asserted exactly: nondeterminism
    creeping in (say a nondeterministic accelerator reduction) should fail loudly.
    """
    args = (l23_small_batch.iops, l23_small_batch.Rrs, l23_small_batch.geometry)
    train = np.ones(l23_small_batch.n_sample, dtype=bool)

    assert B.fit_o25(*args, train=train) == B.fit_o25(*args, train=train)


def test_fit_o25_honours_the_train_mask(l23_small_batch):
    """Fitting on a subset gives measurably different coefficients.

    The guard that matters most here: fitting a *comparison* model on test data
    would flatter the model it is compared against, and it is the one direction
    of bias nobody thinks to check. Measured on the fixture: restricting to the
    first 25 scenes moves every zenith's coefficients by 4.4-6.3% (max relative
    per row), so the 1% threshold sits well below the effect while failing hard
    if the mask were ignored (the difference would be exactly zero).
    """
    args = (l23_small_batch.iops, l23_small_batch.Rrs, l23_small_batch.geometry)
    everything = np.ones(l23_small_batch.n_sample, dtype=bool)

    on_all = np.asarray(B.fit_o25(*args, train=everything))
    on_half = np.asarray(B.fit_o25(*args, train=l23_small_batch.scene < 25))

    per_row = np.max(np.abs(on_half[:, 1:] / on_all[:, 1:] - 1.0), axis=1)
    assert np.all(per_row > 0.01), f"per-zenith max relative shift: {per_row}"


def test_fit_o25_rejects_a_mask_with_an_empty_zenith(l23_small_batch):
    """An empty group raises; a named-but-absent zenith raises.

    **Restated for M5 (task 8).** ``zeniths`` used to default to
    ``(0.0, 30.0, 60.0)``, which is right for L23 and silently wrong elsewhere:
    on PB24 it fits 3 of the 8 in-window zeniths and interpolates across the
    other 5 without a word. The default is now to derive the list from the
    training data, so what must raise has shifted -- see the companion test for
    the coverage check that replaces the old blanket guard.
    """
    args = (l23_small_batch.iops, l23_small_batch.Rrs, l23_small_batch.geometry)

    with pytest.raises(ValueError, match="no samples at all"):
        B.fit_o25(*args, train=np.zeros(l23_small_batch.n_sample, dtype=bool))

    with pytest.raises(ValueError, match="no samples at theta_s = 60"):
        B.fit_o25(
            *args, train=l23_small_batch.zenith != 60.0, zeniths=(0.0, 30.0, 60.0)
        )


def test_fit_o25_refuses_to_ignore_a_zenith_the_data_contains(l23_small_batch):
    """**Regression (M5 task 8).** The silent failure PB24 would have hit.

    Naming a subset of the angles present in the training data used to succeed,
    fitting some and interpolating across the rest. On L23 that is invisible
    (its three zeniths *are* the old default); on PB24 it would have quietly
    discarded five of eight.
    """
    args = (l23_small_batch.iops, l23_small_batch.Rrs, l23_small_batch.geometry)
    everything = np.ones(l23_small_batch.n_sample, dtype=bool)

    with pytest.raises(ValueError, match="does not cover"):
        B.fit_o25(*args, train=everything, zeniths=(0.0, 30.0))


def test_fit_o25_derives_the_zeniths_from_the_training_data(l23_small_batch):
    """The new default, and what a held-out zenith now produces.

    Holding a zenith out of the training mask yields a table without that row --
    correct, and visible in the row count, which is what makes the clamp at
    evaluation time a decision rather than an accident.
    """
    args = (l23_small_batch.iops, l23_small_batch.Rrs, l23_small_batch.geometry)
    everything = np.ones(l23_small_batch.n_sample, dtype=bool)

    full = B.fit_o25(*args, train=everything)
    held = B.fit_o25(*args, train=l23_small_batch.zenith != 60.0)

    assert [row[0] for row in full] == [0.0, 30.0, 60.0]
    assert [row[0] for row in held] == [0.0, 30.0]
    # and the derived default reproduces the old explicit one exactly
    explicit = B.fit_o25(*args, train=everything, zeniths=(0.0, 30.0, 60.0))
    np.testing.assert_array_equal(np.asarray(full), np.asarray(explicit))


def test_fit_o25_weighted_default_beats_the_papers_unweighted_fit(l23_small_batch):
    """The fairness claim behind the default, pinned as an ordering.

    An unweighted objective in ``Rrs`` optimises the bright blue and abandons the
    dark red, so under the project's *relative* metric it must lose. Measured on
    the fixture: weighted 0.62% vs unweighted 2.19% rRMS (full L23: 0.70% vs
    2.62%). Only the ordering is asserted -- with a factor-two margin against a
    measured factor 3.5 -- because the exact numbers belong to the data, not the
    claim. Why it matters: fitting a *rival* model with the wrong objective would
    have made our own hybrid look ~4x better than a fair comparison allows.
    """
    args = (l23_small_batch.iops, l23_small_batch.Rrs, l23_small_batch.geometry)
    train = np.ones(l23_small_batch.n_sample, dtype=bool)
    truth = C.Rrs_to_rrs(l23_small_batch.Rrs)

    scores = {}
    for weighted in (True, False):
        rows = B.fit_o25(*args, train=train, weighted=weighted)
        pred = B.rrs_o25(
            l23_small_batch.iops, None, l23_small_batch.geometry, coeffs=rows
        )
        scores[weighted] = float(V.rrms(truth, pred))

    assert scores[True] * 2.0 < scores[False], f"weighted/unweighted rRMS: {scores}"


# ------------------------------------ O25 against the reference data (full) --


@needs_l23
def test_fit_o25_reproduces_the_embedded_table(l23_batch):
    """**The provenance gate.** Refitting on ``scene_train`` returns the constants.

    Same data, same seeded split (:data:`robust.rt.data.l23.SPLIT_SEED`), same
    weighted objective -- so :data:`O25_L23_REFIT` cannot drift from the code
    that made it. rtol 1e-6: the table stores 8 significant figures, so its own
    rounding contributes up to ~1e-7 (measured max difference 1.24e-7); anything
    beyond that means the fit, the split, or the loader changed.
    """
    splits = l23.make_splits(l23_batch)

    refit = B.fit_o25(
        l23_batch.iops, l23_batch.Rrs, l23_batch.geometry, train=splits.scene_train
    )

    np.testing.assert_allclose(
        np.asarray(refit), np.asarray(B.O25_L23_REFIT), rtol=1e-6
    )


@needs_l23
def test_full_l23_respects_the_o25_validity_ceiling(l23_batch):
    """L23's brightest ``Rrs`` (0.0248) is below O25's fitted ceiling (0.06).

    Cheap, but the assumption is load-bearing: the quadratic was fitted for
    ``Rrs`` up to :data:`O25_RRS_CEILING`, and a future dataset that exceeds it
    would be extrapolating in brightness. This is the test that notices.
    """
    assert float(jnp.max(l23_batch.Rrs)) < B.O25_RRS_CEILING


# --------------------------------------- O25 over the full geometry (task 8) -
# The benchmark M5's gate is measured against. If it is crippled, beating it
# means nothing -- so these tests are about O25 being given its best shot.


def _toy_o25_table():
    """A 2 x 2 x 2 table with a deliberate view-angle dependence."""
    G = np.zeros((2, 2, 2, 4))
    G[..., 0] = 0.06  # Gw0
    G[..., 1] = 0.03  # Gw1
    G[..., 2] = np.array([0.04, 0.02])[None, :, None]  # Gp0 falls with theta_v
    G[..., 3] = 0.15  # Gp1
    return B.O25Table(
        theta_s=np.array([0.0, 60.0]),
        theta_v=np.array([0.0, 60.0]),
        dphi=np.array([0.0, 180.0]),
        G=G,
        provenance="toy",
    )


def test_a_lifted_zenith_table_reproduces_the_one_d_path_exactly(l23_small_batch):
    """**Gate.** The L23 path is untouched by the new machinery.

    ``O25Table.from_rows`` lifts the shipped zenith-only table into the 3-D form
    with one view-angle and one azimuth node. It must then evaluate to exactly
    what ``o25_coefficients`` gives, or M4's numbers would have moved.
    """
    batch = l23_small_batch
    lifted = B.O25Table.from_rows(B.O25_L23_REFIT)

    one_d = B.Rrs_o25(batch.iops, None, batch.geometry, batch.wave)
    three_d = B.Rrs_o25(batch.iops, None, batch.geometry, batch.wave, coeffs=lifted)

    np.testing.assert_allclose(
        np.asarray(three_d), np.asarray(one_d), rtol=1e-6, atol=0.0
    )


def test_the_table_interpolates_between_nodes_and_clamps_outside():
    table = _toy_o25_table()
    geometry = Geometry(
        theta_s=jnp.asarray([0.0, 0.0, 0.0]),
        theta_v=jnp.asarray([0.0, 30.0, 89.0]),
        dphi=jnp.asarray([0.0, 0.0, 0.0]),
    )

    _, _, Gp0, _ = table.coefficients(geometry)

    assert float(Gp0[0]) == pytest.approx(0.04)  # at a node
    assert float(Gp0[1]) == pytest.approx(0.03)  # halfway
    assert float(Gp0[2]) == pytest.approx(0.02)  # clamped past the last node


def test_the_table_is_jittable_and_differentiable_between_nodes():
    """A lookup table has kinks at its nodes, so check at 25 deg (M4 gotcha 4)."""
    table = _toy_o25_table()

    def f(theta_v):
        geometry = Geometry(
            theta_s=jnp.asarray([30.0]), theta_v=theta_v, dphi=jnp.asarray([90.0])
        )
        return table.coefficients(geometry)[2][0]

    assert float(jax.jit(f)(jnp.asarray([25.0]))) > 0.0
    slope = jax.grad(lambda t: f(jnp.asarray([t])))(25.0)
    assert float(slope) < 0.0  # Gp0 falls with view angle in the toy table


def test_fit_o25_table_recovers_a_known_answer():
    """A round trip: build data from a table, fit it back."""
    rng = np.random.default_rng(3)
    n = 800
    theta_s = rng.choice([0.0, 60.0], n)
    theta_v = rng.choice([0.0, 60.0], n)
    dphi = rng.choice([0.0, 180.0], n)
    geometry = Geometry(
        theta_s=jnp.asarray(theta_s),
        theta_v=jnp.asarray(theta_v),
        dphi=jnp.asarray(dphi),
    )
    iops = IOPs(
        a=jnp.asarray(rng.uniform(0.02, 0.5, (n, 5))),
        bb_w=jnp.asarray(rng.uniform(1e-4, 3e-3, (n, 5))),
        bb_p=jnp.asarray(rng.uniform(1e-3, 2e-2, (n, 5))),
    )
    truth = _toy_o25_table()
    Rrs = B.Rrs_o25(iops, None, geometry, coeffs=truth)

    fitted = B.fit_o25_table(
        iops, Rrs, geometry, train=np.ones(n, dtype=bool), min_samples=10
    )

    assert fitted.shape == (2, 2, 2)
    np.testing.assert_allclose(fitted.G, truth.G, rtol=1e-4, atol=1e-6)


def test_fit_o25_table_refuses_a_thin_or_empty_cell():
    """Four coefficients from a handful of points is noise, not a benchmark."""
    rng = np.random.default_rng(4)
    n = 60
    geometry = Geometry(
        theta_s=jnp.zeros(n),
        theta_v=jnp.asarray(np.repeat([0.0, 60.0], n // 2)),
        dphi=jnp.zeros(n),
    )
    iops = IOPs(
        a=jnp.asarray(rng.uniform(0.02, 0.5, (n, 3))),
        bb_w=jnp.asarray(rng.uniform(1e-4, 3e-3, (n, 3))),
        bb_p=jnp.asarray(rng.uniform(1e-3, 2e-2, (n, 3))),
    )
    Rrs = B.Rrs_o25(iops, None, geometry, coeffs=B.O25Table.from_rows())

    with pytest.raises(ValueError, match="fewer than"):
        B.fit_o25_table(
            iops, Rrs, geometry, train=np.ones(n, dtype=bool), min_samples=10_000
        )


def test_rrs_o25_can_use_the_geometry_aware_transfer():
    """The one scoring path the nadir transfer contaminates."""
    geometry = Geometry(
        theta_s=jnp.asarray([30.0]),
        theta_v=jnp.asarray([55.0]),
        dphi=jnp.asarray([90.0]),
    )
    iops = IOPs(
        a=jnp.asarray([[0.08, 0.05]]),
        bb_w=jnp.asarray([[2e-3, 1.5e-3]]),
        bb_p=jnp.asarray([[5e-3, 4e-3]]),
    )

    nadir = B.rrs_o25(iops, None, geometry, coeffs=B.O25_L23_REFIT)
    aware = B.rrs_o25(
        iops, None, geometry, coeffs=B.O25_L23_REFIT, transfer=C.default_transfer()
    )

    # At 55 deg the fitted A is well below 0.52, so the same Rrs implies a
    # noticeably larger subsurface rrs.
    assert float(aware[0, 0]) > float(nadir[0, 0]) * 1.1


@needs_pb24
def test_the_three_d_table_beats_the_zenith_only_refit_on_held_out_pb24():
    """**The gate.** The extra axes must earn their place, or we say they do not.

    Measured: 1.67x better overall on held-out realisations, and 1.58-1.86x at
    every view angle -- the gain is roughly uniform because the zenith-only fit's
    error comes from pooling *all* view geometries into one fit per solar zenith,
    which hurts at nadir as much as anywhere.
    """
    from robust.rt.data import pb24 as P

    batch = P.load_batch(realisations=40, angles="window")
    splits = P.make_splits(batch, kinds=("realisation",))
    train, test = splits.train("realisation"), splits.test("realisation")

    rows = B.fit_o25(batch.iops, batch.Rrs, batch.geometry, train=train)
    table = B.fit_o25_table(batch.iops, batch.Rrs, batch.geometry, train=train)
    transfer = C.default_transfer()

    assert len(rows) == 8  # derived from the data, not the old (0, 30, 60)
    assert table.shape == (8, 8, 13)

    def score(coeffs):
        pred = B.rrs_o25(
            batch.iops,
            None,
            batch.geometry,
            batch.wave,
            coeffs=coeffs,
            transfer=transfer,
        )
        return float(V.rrms(batch.rrs[test], pred[test]))

    one_d, three_d = score(rows), score(table)

    assert three_d < one_d / 1.4, (
        f"the extra axes are not earning their place: {one_d:.2f}% -> {three_d:.2f}%"
    )


@needs_pb24
def test_the_nadir_transfer_would_have_hidden_the_difference():
    """Why task 7 had to precede task 8, as a measurement rather than a claim.

    Scored through the *nadir* transfer, the interface error swamps both models
    and the 3-D table's advantage all but disappears -- so a benchmark built in
    the other order would have concluded the extra axes did not matter.
    """
    from robust.rt.data import pb24 as P

    batch = P.load_batch(realisations=40, angles="window")
    splits = P.make_splits(batch, kinds=("realisation",))
    train, test = splits.train("realisation"), splits.test("realisation")

    rows = B.fit_o25(batch.iops, batch.Rrs, batch.geometry, train=train)
    table = B.fit_o25_table(batch.iops, batch.Rrs, batch.geometry, train=train)

    def score(coeffs, transfer):
        pred = B.rrs_o25(
            batch.iops,
            None,
            batch.geometry,
            batch.wave,
            coeffs=coeffs,
            transfer=transfer,
        )
        return float(V.rrms(batch.rrs[test], pred[test]))

    aware_gain = score(rows, C.default_transfer()) / score(table, C.default_transfer())
    nadir_gain = score(rows, None) / score(table, None)

    assert aware_gain > 1.4
    assert nadir_gain < 1.2
    assert aware_gain > nadir_gain * 1.3
