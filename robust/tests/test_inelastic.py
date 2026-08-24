"""
Tests for :mod:`robust.rt.inelastic` — M2 task 1, the Raman factor.

Three layers, mirroring the M2 gate structure (the formal live-BING
cross-check module and the fluorescence half arrive with tasks 2-3; the
Raman-side versions of both start here):

- **Port correctness**: constants equal BING's; a spot cross-check against
  ``bing.rt.rrs.calc_raman_correction_factor`` on fixture rows at all three
  zeniths (``importorskip`` — mandatory-green on this machine, skipped on CI).
- **Characterization, not perfection**: the analytic backbone's *measured*
  errors against the fixture truth are pinned as bands — on this 50-scene
  fixture the median increment error (550-700 nm) is **+1.6 % / -4.0 % at
  30/60 deg and -38.6 % at 0 deg**, matching the assessment's full-release
  table (+1/-4/-39). The 0-deg failure is delta_R's job (M3); a test that
  demanded zero error here would be testing the wrong promise.
- **Composition wiring**: ``forward(..., inelastic=Inelastic(
  fluorescence=False))`` multiplies the elastic result by ``f_phys`` in
  ``Rrs`` space (design §2), the ``None`` and all-off routes stay bitwise
  elastic, fluorescence still raises until task 2, and ``mode='emulator'``
  refuses the composition.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import ed as E
from robust.rt import hybrid as H
from robust.rt import inelastic as I
from robust.rt.types import Inelastic

#: The Raman-only configuration every wiring test uses.
RAMAN_ONLY = Inelastic(fluorescence=False)

#: Emission band of the characterization gate (assessment / coding plan M2).
BAND = (550.0, 700.0)

#: Median increment-error bands, (lo, hi) fraction, measured on THIS fixture
#: at task 1 (+1.6 / -4.0 / -38.6 %): generous enough for float noise, tight
#: enough that a broken term or a flat-Ed regression leaves them.
INCREMENT_ERROR_BANDS = {0.0: (-0.45, -0.32), 30.0: (-0.05, 0.08), 60.0: (-0.11, 0.02)}


def factor_and_truth(batch):
    """f_phys and the truth factor, as NumPy."""
    ours = np.asarray(I.raman_factor(batch.iops, batch.geometry, batch.wave))
    return ours, np.asarray(batch.truth_raman_factor)


# ------------------------------------------------------------ constants/port ----


def test_constants_equal_bings():
    """The port's constants ARE bing's — same physics by construction."""
    raman = pytest.importorskip("bing.rt.raman")
    assert I.B_RAMAN_488 == raman.B_RAMAN_488_HYDROLIGHT
    assert I.RAMAN_EXPONENT == raman.EXPONENT_ENERGY_EXCITATION
    assert I.MU_D == raman.MU_D_DEFAULT
    assert I.MU_U == raman.MU_U_DEFAULT
    assert I.MU_R == raman.MU_R_DEFAULT


def test_raman_bb_matches_bing():
    """``raman_bb`` equals bing's backscattering coefficient over the band.

    rtol 5e-7, not tighter: bing computes its b_b/b ratio by *numerical
    quadrature* of the phase function (≈ 0.5 to ~2e-7), where this port uses
    the analytic 1/2 that bing's own derivation comment arrives at. The
    difference is bing's integrator noise, inside the M2 rtol 1e-6 contract.
    """
    raman = pytest.importorskip("bing.rt.raman")
    wave_ex = np.linspace(350.0, 700.0, 15)
    # float64 without the fixture would be *silently truncated* to float32
    # (with a warning) — the recurring dtype lesson; hence NumPy on our side.
    ours = np.asarray(I.raman_bb(wave_ex))
    np.testing.assert_allclose(
        ours, raman.raman_backscattering_coeff(wave_ex), rtol=5e-7
    )


def test_factor_spot_check_against_bing(l23_small_inelastic_batch):
    """f_phys equals bing's calc_raman_correction_factor on fixture rows.

    One row per zenith, same inputs fed to both (bing gets NumPy interp for
    the excitation IOPs and our own Ed ratio, per its signature). rtol 5e-7:
    measured agreement is ~1.6e-7 on float32 inputs; the M2 gate is 1e-6.
    The exhaustive version is task 3's xcheck module.
    """
    bing_rrs = pytest.importorskip("bing.rt.rrs")
    raman = pytest.importorskip("bing.rt.raman")
    batch = l23_small_inelastic_batch
    wave = np.asarray(batch.wave)
    # NumPy float64 for the reference inputs (jnp.float64 outside the x64
    # fixture silently truncates — the recurring dtype lesson).
    lam_ex = 1.0 / (1.0 / wave + C.RAMAN_SHIFT * 1e-7)
    ours, _ = factor_and_truth(batch)

    for row in (0, 75, 149):  # one per zenith in the 150-sample fixture
        a = np.asarray(batch.iops.a)[row]
        bb = np.asarray(batch.iops.bb)[row]
        reference = bing_rrs.calc_raman_correction_factor(
            a,
            bb,
            np.interp(lam_ex, wave, a),
            np.interp(lam_ex, wave, bb),
            raman.raman_backscattering_coeff(lam_ex),
            np.asarray(
                E.ratio(float(batch.zenith[row]), jnp.asarray(lam_ex), batch.wave)
            ),
        )
        np.testing.assert_allclose(ours[row], reference, rtol=5e-7)


# ------------------------------------------------------------- physicality ----


def test_factor_is_physical(l23_small_inelastic_batch):
    """f_phys ≥ 1 (Raman adds photons), finite, and larger in the red."""
    ours, _ = factor_and_truth(l23_small_inelastic_batch)
    wave = np.asarray(l23_small_inelastic_batch.wave)
    assert np.all(np.isfinite(ours))
    assert ours.min() >= 1.0
    blue = np.median(ours[:, np.abs(wave - 450.0).argmin()])
    red = np.median(ours[:, np.abs(wave - 650.0).argmin()])
    assert red > blue > 1.0


def test_characterization_bands(l23_small_inelastic_batch):
    """The backbone's measured error structure, pinned per zenith.

    Median of (f_phys - 1)/(truth - 1) - 1 over 550-700 nm. The -39 %-at-0°
    failure is *expected* (S&P98's fixed mean cosines break at high sun —
    design §4.3); it is what M3's delta_R must fix, and M3's gate will
    demand |error| ≤ 5 % where this test accepts -45..-32 %.
    """
    batch = l23_small_inelastic_batch
    ours, truth = factor_and_truth(batch)
    wave = np.asarray(batch.wave)
    band = (wave >= BAND[0]) & (wave <= BAND[1])

    for zenith, (lo, hi) in INCREMENT_ERROR_BANDS.items():
        rows = batch.zenith == zenith
        err = (
            np.median((ours[rows][:, band] - 1.0) / (truth[rows][:, band] - 1.0)) - 1.0
        )
        assert lo < err < hi, f"zenith {zenith}: median increment error {err:.3f}"


def test_ed_override_changes_the_factor(l23_small_inelastic_batch):
    """The geometry.Ed seam reaches the physics end to end.

    A flat override kills the Ed-ratio structure, so the factor must move —
    and by a lot in the red, where the true ratio is ~1.5.
    """
    batch = l23_small_inelastic_batch
    flat_sky = (jnp.asarray([350.0, 750.0]), jnp.asarray([1.0, 1.0]))
    geometry = dataclasses.replace(batch.geometry, Ed=flat_sky)

    default = np.asarray(I.raman_factor(batch.iops, batch.geometry, batch.wave))
    flat = np.asarray(I.raman_factor(batch.iops, geometry, batch.wave))

    wave = np.asarray(batch.wave)
    red = np.abs(wave - 650.0).argmin()
    assert not np.allclose(default[:, red], flat[:, red], rtol=1e-3)
    # Flat Ed *underestimates* the red increment (true ratio > 1 there).
    assert np.median(flat[:, red] - 1.0) < np.median(default[:, red] - 1.0)


# ------------------------------------------------------------ JAX behaviour ----


def test_factor_jit_and_vmap(l23_small_inelastic_batch):
    """Compiled and mapped evaluation agree with eager."""
    batch = l23_small_inelastic_batch
    eager = np.asarray(I.raman_factor(batch.iops, batch.geometry, batch.wave))
    jitted = np.asarray(jax.jit(I.raman_factor)(batch.iops, batch.geometry, batch.wave))
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)

    mapped = np.asarray(
        jax.vmap(lambda io, g: I.raman_factor(io, g, batch.wave))(
            batch.iops, batch.geometry
        )
    )
    np.testing.assert_allclose(mapped, eager, rtol=1e-6)


def test_factor_gradients_are_finite(l23_small_inelastic_batch):
    """grad of a factor summary w.r.t. a, bb_p, theta_s is finite and sane.

    The full FD gate through the composed forward (incl. phi_C) is task 3;
    this pins that nothing in the port blocks differentiation.
    """
    batch = l23_small_inelastic_batch
    row = jax.tree_util.tree_map(lambda x: x[75], batch.iops)
    geom = jax.tree_util.tree_map(lambda x: x[75], batch.geometry)

    g_iops = jax.grad(lambda io: I.raman_factor(io, geom, batch.wave).sum())(row)
    assert np.all(np.isfinite(np.asarray(g_iops.a)))
    assert np.all(np.isfinite(np.asarray(g_iops.bb_p)))
    g_theta = jax.grad(
        lambda t: I.raman_factor(
            row, dataclasses.replace(geom, theta_s=t), batch.wave
        ).sum()
    )(geom.theta_s)
    assert np.isfinite(float(g_theta))


# ----------------------------------------------------------------- wiring ----


def test_forward_composes_raman_in_Rrs_space(l23_small_inelastic_batch):
    """forward(raman-only) == rrs_to_Rrs(elastic rrs) × f_phys (design §2)."""
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)

    composed = np.asarray(H.forward(*args, inelastic=RAMAN_ONLY, check_domain=False))
    elastic = np.asarray(H.forward(*args, check_domain=False))
    factor = np.asarray(I.raman_factor(batch.iops, batch.geometry, batch.wave))

    # Algebraically exact; the rrs round trip inside costs ~1 ULP (float32).
    np.testing.assert_allclose(composed, elastic * factor, rtol=5e-6)
    assert np.all(composed >= elastic * 0.999999)  # Raman only adds


def test_forward_all_processes_off_is_bitwise_elastic(l23_small_inelastic_batch):
    """raman=False, fluorescence=False composes nothing — bitwise elastic."""
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    off = Inelastic(raman=False, fluorescence=False)
    a = np.asarray(H.forward(*args, check_domain=False))
    b = np.asarray(H.forward(*args, inelastic=off, check_domain=False))
    np.testing.assert_array_equal(a, b)


def test_fluorescence_still_raises_until_task_2(l23_small_inelastic_batch):
    """The default configuration asks for fluorescence — loud error, no array."""
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    with pytest.raises(NotImplementedError, match="M2 task 2"):
        H.forward(*args, inelastic=Inelastic(), check_domain=False)


def test_emulator_mode_refuses_inelastic(l23_small_inelastic_batch):
    """mode='emulator' is a term, not a model; composing on it is an error."""
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    with pytest.raises(ValueError, match="emulator"):
        H.rrs_forward(*args, "emulator", inelastic=RAMAN_ONLY, check_domain=False)


def test_composed_forward_gradient_finite(l23_small_inelastic_batch):
    """grad through the full composed forward w.r.t. the IOPs is finite."""
    batch = l23_small_inelastic_batch
    row_iops = jax.tree_util.tree_map(lambda x: x[75:76], batch.iops)
    row_pp = jax.tree_util.tree_map(lambda x: x[75:76], batch.phase_params)
    row_geom = jax.tree_util.tree_map(lambda x: x[75:76], batch.geometry)

    grad = jax.grad(
        lambda io: H.rrs_forward(
            io,
            row_pp,
            row_geom,
            batch.wave,
            inelastic=RAMAN_ONLY,
            check_domain=False,
        ).sum()
    )(row_iops)
    assert np.all(np.isfinite(np.asarray(grad.a)))
    assert np.all(np.isfinite(np.asarray(grad.bb_p)))
