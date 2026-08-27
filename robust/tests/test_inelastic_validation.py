"""
The inelastic M4 acceptance gate — design §6, one line per test (M4 task 2).

This file is the gate *spelled in one place*: six numbered tests, one per §6
line, every metric computed through :mod:`robust.rt.validation`'s protocol
functions — the same definitions ``test_inelastic_corr.py``'s standing gates
delegate to and ``design/py/run_validation.py --inelastic`` reports, so the
promise here, the M3 pins, and the committed metrics table are one quantity
each. Where a line duplicates a standing test (the per-process deltas, the
hash pins), that is deliberate: the standing tests are where the numbers were
earned; this file is where the design's acceptance is stated and stays
stated.

**The gate band is 400–700 nm** (prompt 5 Q&A Q1, JXP: "do not gate on the
rms outside the 400-700nm range"). Below 400 nm the Raman excitation leaves
the L23 grid and clamps and the heads never trained — the model's stated
domain since M3; above 700 nm the domain answer excludes the far-red tail
too. The full 350–750 nm number is *reported* by the validation script
(candor), never gated.

Skips: the full-release lines need ``$OS_COLOR`` (CI skips them); everything
weight-dependent skips with a regenerate message if the committed heads are
absent; the bit-identity line's strict tier is machine-anchored and skips
under CI exactly as ``test_inelastic_types.py``'s does (the two-tier rule,
record §2.8).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import hybrid as H
from robust.rt import inelastic as I
from robust.rt import inelastic_corr as IC
from robust.rt import validation as V
from robust.rt.data import l23 as L
from robust.rt.types import Geometry, Inelastic, IOPs, PhaseParams
from robust.tests import test_inelastic_types as hash_pins
from robust.tests.conftest import needs_l23_inelastic

#: Gate line (1): the wavelength band the total rRMS is scored over — JXP's
#: prompt 5 Q&A Q1 answer, defined once in :mod:`robust.rt.validation` so the
#: gate here and the committed metrics table cannot score different bands.
GATE_BAND = V.INELASTIC_GATE_BAND

#: Gate line (1): held-out total rRMS vs Rrs_X4, percent, at each zenith.
TOTAL_GATE = 0.5

#: Gate lines (2)-(3): per-process median |error|, fractional (5 %).
DELTA_GATE = 0.05

#: Gate line (6): full-batch forward over elastic-hybrid runtime.
SPEED_GATE = 2.0

#: The committed weights; the corrected model does not exist without them.
needs_weights = pytest.mark.skipif(
    not (IC.DEFAULT_RAMAN_WEIGHTS.exists() and IC.DEFAULT_FL_WEIGHTS.exists()),
    reason="committed correction weights missing — run "
    "design/py/train_inelastic_corr.py",
)


@pytest.fixture(scope="module")
def full_release():
    """The full L23 inelastic batch, splits, heads, and both model channels.

    Module-scoped, as ``test_inelastic_corr.py``'s: the netCDF load plus the
    full-batch forward passes cost ~20 s and four gate lines share them.
    Predictions are computed once so every line scores literally identical
    data.
    """
    batch = L.load_inelastic_batch()
    splits = L.make_splits(batch)
    heads = IC.load_default()
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    rrs_corrected = np.asarray(
        H.rrs_forward(
            *args,
            "hybrid",
            inelastic=Inelastic(),
            corrections=heads,
            check_domain=False,
        )
    )
    return batch, splits, heads, rrs_corrected


@needs_weights
@needs_l23_inelastic
def test_gate_1_total_rrms_vs_x4(full_release):
    """**§6 line 1**: held-out total rRMS vs ``Rrs_X4`` ≤ 0.5 % at each zenith.

    All processes on, φ_C = 0.02, committed weights, ``rrs`` space, the
    elastic scene split — scored over :data:`GATE_BAND` (Q&A Q1). Measured at
    task 2: ~0.34 % per zenith; asserted at the gate bar, not the measured
    values — the gate is the promise, the metrics table is the achievement.
    """
    batch, splits, _, rrs_corrected = full_release
    wave = np.asarray(batch.wave)
    band = (wave >= GATE_BAND[0]) & (wave <= GATE_BAND[1])
    held = splits.scene_test
    truth = np.asarray(C.Rrs_to_rrs(batch.Rrs_x4))

    per_zenith = V.group_rrms(
        jnp.asarray(truth[held][:, band]),
        jnp.asarray(rrs_corrected[held][:, band]),
        batch.zenith[held],
    )
    assert set(per_zenith) == {0.0, 30.0, 60.0}
    for zenith, value in per_zenith.items():
        assert value <= TOTAL_GATE, (
            f"zenith {zenith:.0f}: held-out rRMS vs X4 {value:.3f} % over "
            f"{GATE_BAND} nm exceeds the {TOTAL_GATE} % gate"
        )


@needs_weights
@needs_l23_inelastic
def test_gate_2_raman_delta(full_release):
    """**§6 line 2**: median |Raman increment error| ≤ 5 % per zenith incl. 0°.

    Through :func:`validation.median_increment_error` — the definition the M3
    gate (``test_inelastic_corr.py``) delegates to — over 550–700 nm, with the
    490 nm line riding at the same bar. 0° is named in the design because the
    analytic backbone fails it by −38.6 %; this line is what δ_R earned.
    """
    batch, splits, heads, _ = full_release
    wave = np.asarray(batch.wave)
    held = splits.scene_test
    f_phys = np.asarray(I.raman_factor(batch.iops, batch.geometry, batch.wave))
    delta = np.asarray(heads.raman.delta(batch.iops, batch.geometry, batch.wave))
    f_corr = np.asarray(IC.corrected_raman_factor(delta, f_phys))
    truth = np.asarray(batch.truth_raman_factor)

    for label, band in (
        ("550-700 nm", (wave >= 550.0) & (wave <= 700.0)),
        ("490 nm", np.abs(wave - 490.0) < 1e-6),
    ):
        errors = V.median_increment_error(
            f_corr[held], truth[held], batch.zenith[held], band
        )
        assert set(errors) == {0.0, 30.0, 60.0}
        for zenith, err in errors.items():
            assert abs(err) <= DELTA_GATE, f"zenith {zenith:.0f}, {label}: {err:+.4f}"


@needs_weights
@needs_l23_inelastic
def test_gate_3_fluorescence_delta(full_release):
    """**§6 line 3**: median |685 nm fluorescence error| ≤ 5 % per zenith.

    Through :func:`validation.peak_ratio_error` against ``Rrs_X4 − Rrs_X2``
    at the truth's φ_C = 0.02. The analytic backbone sits at −13.7 % at 60°;
    the corrected term measures ~+0.1 %.
    """
    batch, splits, heads, _ = full_release
    wave = np.asarray(batch.wave)
    held = splits.scene_test
    i685 = int(np.abs(wave - 685.0).argmin())
    k_fl = np.asarray(I.fluorescence_kernel(batch.iops, batch.geometry, batch.wave))
    delta = np.asarray(heads.fl.delta(batch.iops, batch.geometry, batch.wave))
    model = L.PHI_C_L23 * k_fl * (1.0 + delta)
    truth = np.asarray(batch.truth_fluorescence)

    errors = V.peak_ratio_error(model[held], truth[held], batch.zenith[held], i685)
    assert set(errors) == {0.0, 30.0, 60.0}
    for zenith, err in errors.items():
        assert abs(err) <= DELTA_GATE, f"zenith {zenith:.0f}: 685 nm {err:+.4f}"


def test_gate_4_elastic_bit_identity(l23_small_batch):
    """**§6 line 4**: ``inelastic=None`` bit-identical to the elastic hybrid.

    Two assertions, both bitwise: omitting ``inelastic`` and passing ``None``
    are the same arrays, and turning every process off
    (``Inelastic(raman=False, fluorescence=False)``) returns them too — the
    design §1 guarantee that the elastic path is a no-op *by construction*
    (the ``None`` branch returns the same object), not by cancelling
    arithmetic. The pre-change anchoring — the SHA-256 pins and the committed
    reference arrays — is the standing two-tier regression in
    ``test_inelastic_types.py``; :func:`test_gate_4_pre_change_pins` re-runs
    its strict tier under this gate's name.
    """
    batch = l23_small_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    omitted = np.asarray(H.forward(*args, check_domain=False))
    explicit_none = np.asarray(H.forward(*args, inelastic=None, check_domain=False))
    all_off = np.asarray(
        H.forward(
            *args,
            inelastic=Inelastic(raman=False, fluorescence=False),
            check_domain=False,
        )
    )
    np.testing.assert_array_equal(omitted, explicit_none)
    np.testing.assert_array_equal(omitted, all_off)


@hash_pins.strict_bits_are_local
def test_gate_4_pre_change_pins(l23_small_batch):
    """**§6 line 4, the anchor**: the pre-extension hashes still pin the bytes.

    ``test_inelastic_types.py``'s strict tier invoked through the gate file —
    same helper, same pins, no second definition — so the acceptance record
    names the bit-identity line explicitly. Machine-anchored; CI runs the
    closeness tier in the standing module instead.
    """
    Rrs, rrs = hash_pins.elastic_outputs(l23_small_batch)
    assert hash_pins.sha256_of(Rrs) == hash_pins.PRE_CHANGE_SHA256_RRS_ABOVE
    assert hash_pins.sha256_of(rrs) == hash_pins.PRE_CHANGE_SHA256_RRS_BELOW


@needs_weights
def test_gate_5_gradients_all_inputs(jax_x64, l23_small_inelastic_batch):
    """**§6 line 5**: FD gradient agreement for all inputs, φ_C included.

    :func:`validation.inelastic_gradient_report` — the M2/M3 protocol
    (float64, per-variable steps, θ_s = 35°, off the piecewise-linear Ed
    anchors where the derivative is one-sided, record §4.4) — through the
    composed *corrected* forward, all six variables ≤ the elastic tolerance.
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
    )
    phase = PhaseParams(B_p=f64(batch.phase_params.B_p))
    geometry = Geometry.nadir(jnp.full((len(rows),), 35.0, dtype=jnp.float64))
    wave = jnp.asarray(np.asarray(batch.wave), dtype=jnp.float64)

    def model(i, p, g, w, phi):
        return H.forward(
            i,
            p,
            g,
            w,
            "hybrid",
            inelastic=Inelastic(phi_C=phi),
            corrections=heads,
            check_domain=False,
        )

    report = V.inelastic_gradient_report(
        model, iops, phase, geometry, wave, phi_C=jnp.asarray(0.02, jnp.float64)
    )
    assert set(report) == set(V.INELASTIC_FD_STEPS)
    for name, value in report.items():
        assert value <= V.GRADIENT_TOL, f"d/d{name}: {value:.3e}"


@needs_weights
@needs_l23_inelastic
def test_gate_6_speed_within_twice_elastic(full_release):
    """**§6 line 6**: full-batch corrected forward ≤ 2× the elastic hybrid.

    :func:`validation.speed_ratio` on the full release (9960 × 81), jitted,
    identical arguments; the **median of three trials** is asserted because
    single wall-clock ratios wander ~±5 % on a shared machine (task 1
    measured 1.96–2.11 on the *same* code from trial noise alone). Measured
    after the task-1 fallback: ~1.6× (it entered M4 at 6.3×). Both loaders
    are resolved eagerly first — a memoised loader first touched inside a
    ``jit`` trace caches tracers (the task-1 lesson).
    """
    batch, _, heads, _ = full_release
    from robust.rt import emulator as E

    em = E.load_default()
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)

    def corrected(i, p, g, w):
        return H.forward(
            i,
            p,
            g,
            w,
            "hybrid",
            inelastic=Inelastic(),
            corrections=heads,
            emulator=em,
            check_domain=False,
        )

    def elastic(i, p, g, w):
        return H.forward(i, p, g, w, "hybrid", emulator=em, check_domain=False)

    ratios = [V.speed_ratio(corrected, elastic, *args, repeats=5)[0] for _ in range(3)]
    median = float(np.median(ratios))
    assert median <= SPEED_GATE, (
        f"median speed ratio {median:.2f}x over {len(ratios)} trials "
        f"(all: {[f'{r:.2f}' for r in ratios]}) exceeds the {SPEED_GATE}x gate"
    )
