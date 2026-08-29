"""
Tests for :mod:`robust.rt.inelastic` — M2 tasks 1-2, the analytic terms.

Three layers, mirroring the M2 gate structure (the formal live-BING
cross-check module arrives with task 3; spot versions of it start here):

- **Port correctness**: constants equal BING's; spot cross-checks against
  ``bing.rt.rrs.calc_raman_correction_factor`` and
  ``bing.rt.rrs.calc_Rrs_fluorescence`` on fixture rows at all three zeniths
  (``importorskip`` — mandatory-green on this machine, skipped on CI).
- **Characterization, not perfection**: the analytic backbone's *measured*
  errors against the fixture truth are pinned as bands. Raman: on this
  50-scene fixture the median increment error (550-700 nm) is **+1.6 % /
  -4.0 % at 30/60 deg and -38.6 % at 0 deg**, matching the assessment's
  full-release table (+1/-4/-39). Fluorescence: the median 685 nm
  model/truth ratio is **0.99 / 0.94 / 0.85 at 0/30/60 deg**, the
  assessment's 1.00/0.95/0.86 reproduced on our fixture. Both residual
  structures are the M3 heads' job (delta_R, delta_F); a test that demanded
  zero error here would be testing the wrong promise.
- **Composition wiring**: ``forward(..., inelastic=...)`` composes in ``Rrs``
  space (design §2) — Raman multiplies by ``f_phys``, fluorescence adds
  ``phi_C * K_fl`` (linear in ``phi_C`` by construction) — the ``None`` and
  all-off routes stay bitwise elastic, fluorescence without ``a_ph`` is a
  loud ``ValueError``, and ``mode='emulator'`` refuses the composition.

Every ``forward`` call here passes ``corrections=False`` (deliberate, M3
task 1): this module pins the **analytic** terms, and must keep doing so
bit-for-bit after the trained δ heads land and become the default path.
The corrected path has its own module, ``test_inelastic_corr.py``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import cdom_fl
from robust.rt import conventions as C
from robust.rt import ed as E
from robust.rt import hybrid as H
from robust.rt import inelastic as I
from robust.rt.data.l23 import PHI_C_L23
from robust.rt.types import CDOMFl, Geometry, Inelastic, IOPs, PhaseParams

#: The Raman-only configuration the task-1 wiring tests use.
RAMAN_ONLY = Inelastic(fluorescence=False)

#: Emission band of the characterization gate (assessment / coding plan M2).
BAND = (550.0, 700.0)

#: Median increment-error bands, (lo, hi) fraction, measured on THIS fixture
#: at task 1 (+1.6 / -4.0 / -38.6 %): generous enough for float noise, tight
#: enough that a broken term or a flat-Ed regression leaves them.
INCREMENT_ERROR_BANDS = {0.0: (-0.45, -0.32), 30.0: (-0.05, 0.08), 60.0: (-0.11, 0.02)}

#: The 490 nm row of the assessment's Raman table ("+30 % at 490 nm,
#: 30-60 deg"), measured on THIS fixture at task 3: +29.7 % / +30.4 % at
#: 30/60 deg and -3.0 % at 0 deg (the assessment did not quote 0 deg here).
INCREMENT_ERROR_BANDS_490 = {
    0.0: (-0.10, 0.04),
    30.0: (0.23, 0.36),
    60.0: (0.24, 0.37),
}

#: Median 685 nm model/truth bands for ``phi_C * K_fl`` vs the X4-X2 truth,
#: measured on THIS fixture at task 2 (0.991 / 0.937 / 0.853 at 0/30/60 deg —
#: the assessment's 1.00/0.95/0.86 on our narrower scene sample). Tight
#: enough that a pi-normalization regression (x3) or a flat-Ed regression
#: leaves them instantly; the zenith drift itself is delta_F's target (M3).
FL_685_RATIO_BANDS = {0.0: (0.96, 1.03), 30.0: (0.90, 0.97), 60.0: (0.82, 0.89)}


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


def test_characterization_band_490(l23_small_inelastic_batch):
    """The blue-side row of the assessment's error table, pinned per zenith.

    Median of (f_phys - 1)/(truth - 1) - 1 at 490 nm: **+29.7 % / +30.4 %**
    at 30/60 deg on this fixture — the assessment's "+30 % at 490 nm
    (30-60 deg)" — and -3.0 % at 0 deg (unquoted there). With the 550-700 nm
    bands this completes the M2 error table; the structured blue residual is
    delta_R's other target (M3).
    """
    batch = l23_small_inelastic_batch
    ours, truth = factor_and_truth(batch)
    i490 = int(np.abs(np.asarray(batch.wave) - 490.0).argmin())
    for zenith, (lo, hi) in INCREMENT_ERROR_BANDS_490.items():
        rows = batch.zenith == zenith
        err = np.median((ours[rows, i490] - 1.0) / (truth[rows, i490] - 1.0)) - 1.0
        assert lo < err < hi, f"zenith {zenith}: 490 nm increment error {err:.3f}"


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

    composed = np.asarray(
        H.forward(*args, inelastic=RAMAN_ONLY, corrections=False, check_domain=False)
    )
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
    b = np.asarray(
        H.forward(*args, inelastic=off, corrections=False, check_domain=False)
    )
    np.testing.assert_array_equal(a, b)


def test_fluorescence_without_aph_is_a_loud_error(l23_small_inelastic_batch):
    """Fluorescence without its source term — clear error, no array.

    Successor of the "raises until task 2" pin, updated deliberately: the
    kernel exists now, so the guarded promise is the *physical* requirement
    (``b_F = phi_C * a_ph``), at both entry points — ``forward``'s fast
    pre-check and the kernel's own.
    """
    batch = l23_small_inelastic_batch
    iops = dataclasses.replace(batch.iops, a_ph=None)
    with pytest.raises(ValueError, match="a_ph"):
        H.forward(
            iops,
            batch.phase_params,
            batch.geometry,
            batch.wave,
            inelastic=Inelastic(),
            corrections=False,
            check_domain=False,
        )
    with pytest.raises(ValueError, match="a_ph"):
        I.fluorescence_kernel(iops, batch.geometry, batch.wave)


def test_emulator_mode_refuses_inelastic(l23_small_inelastic_batch):
    """mode='emulator' is a term, not a model; composing on it is an error."""
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    with pytest.raises(ValueError, match="emulator"):
        H.rrs_forward(
            *args,
            "emulator",
            inelastic=RAMAN_ONLY,
            corrections=False,
            check_domain=False,
        )


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
            corrections=False,
            check_domain=False,
        ).sum()
    )(row_iops)
    assert np.all(np.isfinite(np.asarray(grad.a)))
    assert np.all(np.isfinite(np.asarray(grad.bb_p)))


# ------------------------------------------------- fluorescence kernel (task 2) ----


def kernel_and_truth(batch, **kwargs):
    """phi_C_L23 * K_fl and the X4-X2 truth channel, as NumPy."""
    k_fl = np.asarray(
        I.fluorescence_kernel(batch.iops, batch.geometry, batch.wave, **kwargs)
    )
    return PHI_C_L23 * k_fl, np.asarray(batch.truth_fluorescence)


def index_685(batch) -> int:
    return int(np.abs(np.asarray(batch.wave) - I.LAMBDA_FL).argmin())


def test_fl_reference_phi_is_the_truths():
    """The kernel's internal reference yield IS the L23 truth's phi_C.

    ``K_fl = Rrs_fl(PHI_C_REF)/PHI_C_REF``, so ``phi_C * K_fl`` equals fixed
    BING exactly at the truth's yield and is phi_C-linear by construction
    elsewhere (design §4.4). If these ever diverged, the cross-check and the
    truth channels would silently score different quantities.
    """
    assert I.PHI_C_REF == PHI_C_L23


def test_fl_constants_equal_bings():
    """The port's fluorescence constants ARE bing's."""
    chl_fl = pytest.importorskip("bing.rt.chl_fl")
    assert I.PHI_C_REF == chl_fl.PHI_FL_DEFAULT
    assert I.LAMBDA_FL == chl_fl.LAMBDA_FL_PRIMARY
    assert I.SIGMA_FL == chl_fl.SIGMA_FL_PRIMARY
    assert I.LAMBDA_FL_SECONDARY == chl_fl.LAMBDA_FL_SECONDARY
    assert I.SIGMA_FL_SECONDARY == chl_fl.SIGMA_FL_SECONDARY
    assert I.FL_WEIGHT_PRIMARY == chl_fl.WEIGHT_PRIMARY
    assert I.FL_EX_MIN == chl_fl.LAMBDA_EX_MIN
    assert I.FL_EX_MAX == chl_fl.LAMBDA_EX_MAX
    assert I.MU_F == 0.5  # bing hardcodes the isotropic default inline


def test_emission_line_matches_bing():
    """Both emission shapes equal bing's, and 'single' integrates to ~1.

    rtol 1e-5: the reference is float64 NumPy where ours runs float32 by
    default, and a Gaussian tail's relative error grows as |x| * eps.
    """
    chl_fl = pytest.importorskip("bing.rt.chl_fl")
    wave = np.linspace(640.0, 750.0, 23)
    np.testing.assert_allclose(
        np.asarray(I.emission_line(wave, "single")),
        chl_fl.emission_line_single_gaussian(wave),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(I.emission_line(wave, "double")),
        chl_fl.emission_line_double_gaussian(wave),
        rtol=1e-5,
    )
    fine = np.linspace(600.0, 800.0, 2001)
    integral = np.trapezoid(np.asarray(I.emission_line(fine, "single")), fine)
    assert integral == pytest.approx(1.0, rel=1e-4)
    with pytest.raises(ValueError, match="single"):
        I.emission_line(wave, "triple")


def test_kernel_spot_check_against_bing(l23_small_inelastic_batch):
    """phi_C * K_fl equals bing's calc_Rrs_fluorescence on fixture rows.

    One row per zenith, both models fed the identical excitation grid, IOPs,
    and Ed — so any disagreement is the port, not the inputs. rtol 1e-5:
    measured float32 agreement is ~3e-6 (the 65-node trapezoid accumulates;
    in float64 the port is exact to ~7e-16 — task 3's xcheck pins that under
    the x64 fixture at the 1e-6 gate). atol clips the h_C tail, which
    underflows float32 where the float64 reference keeps ~1e-200 values.
    The 'double' shape rides along on one row — same integral, bing's
    ``double_gaussian=True``.
    """
    bing_rrs = pytest.importorskip("bing.rt.rrs")
    batch = l23_small_inelastic_batch
    wave = np.asarray(batch.wave)
    wave_ex = np.asarray(I.fl_excitation_grid())
    ours, _ = kernel_and_truth(batch)
    ours_double = PHI_C_L23 * np.asarray(
        I.fluorescence_kernel(
            batch.iops, batch.geometry, batch.wave, emission_shape="double"
        )
    )

    for row in (0, 75, 149):  # one per zenith in the 150-sample fixture
        a = np.asarray(batch.iops.a)[row]
        bb = np.asarray(batch.iops.bb)[row]
        aph = np.asarray(batch.iops.a_ph)[row]
        zenith = float(batch.zenith[row])
        bing_args = (
            wave,
            a,
            bb,
            np.interp(wave_ex, wave, a),
            np.interp(wave_ex, wave, bb),
            np.interp(wave_ex, wave, aph),
            wave_ex,
            np.asarray(E.Ed(zenith, jnp.asarray(wave_ex))),
            np.asarray(E.Ed(zenith, batch.wave)),
        )
        reference = bing_rrs.calc_Rrs_fluorescence(
            *bing_args, phi_C=PHI_C_L23, double_gaussian=False
        )
        np.testing.assert_allclose(ours[row], reference, rtol=1e-5, atol=1e-12)
        if row == 75:
            reference = bing_rrs.calc_Rrs_fluorescence(
                *bing_args, phi_C=PHI_C_L23, double_gaussian=True
            )
            np.testing.assert_allclose(
                ours_double[row], reference, rtol=1e-5, atol=1e-12
            )


def test_kernel_is_physical(l23_small_inelastic_batch):
    """K_fl >= 0, finite, and peaked at the 685 nm emission line."""
    batch = l23_small_inelastic_batch
    ours, _ = kernel_and_truth(batch)
    wave = np.asarray(batch.wave)
    assert np.all(np.isfinite(ours))
    assert ours.min() >= 0.0
    assert np.all(ours[:, index_685(batch)] > 0.0)
    assert wave[np.argmax(np.median(ours, axis=0))] == I.LAMBDA_FL


def test_kernel_characterization_685(l23_small_inelastic_batch):
    """The backbone's measured 685 nm amplitude, pinned per zenith.

    Median of (phi_C * K_fl) / (Rrs_X4 - Rrs_X2) at 685 nm. The ~1.00 -> 0.85
    drift with zenith is *expected* (fixed two-flow mean cosines — design
    §4.4); it is what M3's delta_F must fix, and its gate will demand
    |error| <= 5 % where these bands accept up to -18 %.
    """
    batch = l23_small_inelastic_batch
    ours, truth = kernel_and_truth(batch)
    i685 = index_685(batch)
    for zenith, (lo, hi) in FL_685_RATIO_BANDS.items():
        rows = batch.zenith == zenith
        ratio = np.median(ours[rows, i685] / truth[rows, i685])
        assert lo < ratio < hi, f"zenith {zenith}: 685 nm model/truth {ratio:.3f}"


def test_kernel_ed_override_reaches_the_quadrature(l23_small_inelastic_batch):
    """The geometry.Ed seam feeds the excitation integral end to end.

    A flat sky replaces both Ed(lambda') and Ed(lambda_em) — the kernel must
    move at the peak, and stay finite and positive.
    """
    batch = l23_small_inelastic_batch
    flat_sky = (jnp.asarray([350.0, 750.0]), jnp.asarray([1.0, 1.0]))
    geometry = dataclasses.replace(batch.geometry, Ed=flat_sky)

    default = np.asarray(I.fluorescence_kernel(batch.iops, batch.geometry, batch.wave))
    flat = np.asarray(I.fluorescence_kernel(batch.iops, geometry, batch.wave))

    i685 = index_685(batch)
    assert np.all(np.isfinite(flat)) and flat.min() >= 0.0
    assert not np.allclose(default[:, i685], flat[:, i685], rtol=1e-3)


def test_double_emission_adds_the_730_shoulder(l23_small_inelastic_batch):
    """'double' puts real energy at 730 nm where 'single' has only a tail.

    Reported, never gated against L23 (unvalidatable — design §4.4): the
    test pins that the switch does what it says, not that it is right.
    """
    batch = l23_small_inelastic_batch
    single, _ = kernel_and_truth(batch)
    double, _ = kernel_and_truth(batch, emission_shape="double")
    i730 = int(np.abs(np.asarray(batch.wave) - I.LAMBDA_FL_SECONDARY).argmin())
    assert np.all(double[:, i730] > 10.0 * single[:, i730])


def test_kernel_jit_and_vmap(l23_small_inelastic_batch):
    """Compiled and mapped evaluation agree with eager."""
    batch = l23_small_inelastic_batch
    eager = np.asarray(I.fluorescence_kernel(batch.iops, batch.geometry, batch.wave))
    jitted = np.asarray(
        jax.jit(I.fluorescence_kernel, static_argnames="emission_shape")(
            batch.iops, batch.geometry, batch.wave
        )
    )
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)

    mapped = np.asarray(
        jax.vmap(lambda io, g: I.fluorescence_kernel(io, g, batch.wave))(
            batch.iops, batch.geometry
        )
    )
    np.testing.assert_allclose(mapped, eager, rtol=1e-6)


# ------------------------------------------------- fluorescence wiring (task 2) ----


def test_forward_composes_fluorescence_additively(l23_small_inelastic_batch):
    """forward(default) == forward(raman-only) + phi_C * K_fl (design §2).

    Algebraically exact in Rrs space — the composition law itself; the rrs
    round trip inside costs ~1 ULP (float32). Also pins that the composed
    delta is strictly positive at 685 nm on every sample, as the truth is.
    """
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)

    full = np.asarray(
        H.forward(*args, inelastic=Inelastic(), corrections=False, check_domain=False)
    )
    raman = np.asarray(
        H.forward(*args, inelastic=RAMAN_ONLY, corrections=False, check_domain=False)
    )
    ours, _ = kernel_and_truth(batch)

    np.testing.assert_allclose(full, raman + ours, rtol=5e-6, atol=1e-10)
    assert np.all((full - raman)[:, index_685(batch)] > 0.0)


def test_forward_passes_emission_shape_through(l23_small_inelastic_batch):
    """Inelastic.emission_shape reaches the kernel — visible at 730 nm."""
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    single = np.asarray(
        H.forward(*args, inelastic=Inelastic(), corrections=False, check_domain=False)
    )
    double = np.asarray(
        H.forward(
            *args,
            inelastic=Inelastic(emission_shape="double"),
            corrections=False,
            check_domain=False,
        )
    )
    i730 = int(np.abs(np.asarray(batch.wave) - I.LAMBDA_FL_SECONDARY).argmin())
    assert np.all(double[:, i730] > single[:, i730])


def test_phi_C_is_a_linear_handle(l23_small_inelastic_batch):
    """Doubling phi_C doubles the fluorescence term, and dRrs/dphi_C == K_fl.

    phi_C-linearity by construction (design §4.4, DQ4): the composed Rrs is
    affine in phi_C, so the gradient — the new physiology handle — is the
    kernel itself, independent of phi_C.
    """
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    raman = np.asarray(
        H.forward(*args, inelastic=RAMAN_ONLY, corrections=False, check_domain=False)
    )

    def full(phi):
        return H.forward(
            *args, inelastic=Inelastic(phi_C=phi), corrections=False, check_domain=False
        )

    delta_1x = np.asarray(full(jnp.asarray(0.02))) - raman
    delta_2x = np.asarray(full(jnp.asarray(0.04))) - raman
    i685 = index_685(batch)
    np.testing.assert_allclose(delta_2x[:, i685], 2.0 * delta_1x[:, i685], rtol=1e-4)

    grad = np.asarray(jax.grad(lambda phi: full(phi)[:, i685].sum())(jnp.asarray(0.02)))
    k_685 = np.asarray(I.fluorescence_kernel(batch.iops, batch.geometry, batch.wave))[
        :, i685
    ].sum()
    np.testing.assert_allclose(grad, k_685, rtol=1e-4)


def test_composed_forward_gradient_finite_incl_aph(l23_small_inelastic_batch):
    """grad through the full default forward w.r.t. the IOPs (a_ph too) is finite."""
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
            inelastic=Inelastic(),
            corrections=False,
            check_domain=False,
        ).sum()
    )(row_iops)
    for leaf in (grad.a, grad.bb_p, grad.a_ph):
        assert np.all(np.isfinite(np.asarray(leaf)))
    # a_ph only *adds* photons at the emission peak: positive sensitivity.
    assert float(np.asarray(grad.a_ph)[0, index_685(batch)]) > 0.0


# --------------------------------------------------- the gradient gate (task 3) ----


@pytest.mark.parametrize(
    ("name", "step"),
    [
        ("a", 1e-6),
        ("bb_p", 1e-9),
        ("a_ph", 1e-8),
        ("phi_C", 1e-6),
        ("theta_s", 1e-3),
    ],
)
def test_gradient_matches_finite_differences_composed(
    jax_x64, l23_small_inelastic_batch, name, step
):
    """**The M2 gradient gate.** autodiff == central FD through the full model.

    The complete composed forward — ZTT + packaged emulator, x f_phys,
    + phi_C * K_fl — differentiated w.r.t. each input the future inversion
    will need, ``phi_C`` and ``a_ph`` now among them (design §4.6). Same
    protocol as the elastic gate in ``test_hybrid.py``: float64 with the
    dtype pinned on the arrays, per-variable steps (theta_s is O(30) and
    wants h ~ 1e-3; the IOP-likes want 1e-6..1e-9, all far below the fixture
    minima so no perturbed input goes negative), the FD asserted finite
    before it is compared. ``check_domain=False`` for the same reason as the
    elastic gate — perturbed inputs leaving the trained range is the check
    working, not the property under test.

    **Evaluated at theta_s = 35 deg, not 30** — a real difference from the
    elastic gate. The inelastic terms consume ``ed.Ed``, which is
    piecewise-*linear* in theta_s with anchors at exactly 0/30/60 deg: at an
    anchor the theta-derivative has a kink (model structure, not a bug), so
    autodiff takes one side while a central difference straddling the knot
    averages both — they disagreed at the 7th digit at 30 deg sharp. Inside
    a segment the function is smooth; measured agreement there: a 2.5e-9,
    bb_p 1.5e-10, a_ph 3.8e-8, phi_C 3.9e-9, theta_s 1.7e-8 relative.
    """
    batch = l23_small_inelastic_batch
    dtype = jnp.float64
    rows = np.where(batch.zenith == 30.0)[0][:3]

    a0 = jnp.asarray(np.asarray(batch.iops.a)[rows], dtype=dtype)
    bb_w0 = jnp.asarray(np.asarray(batch.iops.bb_w)[rows], dtype=dtype)
    bb_p0 = jnp.asarray(np.asarray(batch.iops.bb_p)[rows], dtype=dtype)
    a_ph0 = jnp.asarray(np.asarray(batch.iops.a_ph)[rows], dtype=dtype)
    B_p0 = jnp.asarray(np.asarray(batch.phase_params.B_p)[rows], dtype=dtype)
    # +5 deg off the 30-deg scenes: inside the smooth 30-60 Ed segment
    # (see the docstring), still far from both anchors at step 1e-3.
    theta0 = jnp.asarray(np.asarray(batch.geometry.theta_s)[rows] + 5.0, dtype=dtype)
    wave = jnp.asarray(np.asarray(batch.wave), dtype=dtype)
    phi0 = jnp.asarray(PHI_C_L23, dtype=dtype)

    def scalar(shift):
        """Mean composed Rrs with one variable shifted by the scalar ``shift``."""
        offsets = dict.fromkeys(("a", "bb_p", "a_ph", "phi_C", "theta_s"), 0.0)
        offsets[name] = shift
        iops = IOPs(
            a=a0 + offsets["a"],
            bb_w=bb_w0,
            bb_p=bb_p0 + offsets["bb_p"],
            a_ph=a_ph0 + offsets["a_ph"],
        )
        return jnp.mean(
            H.forward(
                iops,
                PhaseParams(B_p=B_p0),
                Geometry.nadir(theta0 + offsets["theta_s"]),
                wave,
                "hybrid",
                inelastic=Inelastic(phi_C=phi0 + offsets["phi_C"]),
                corrections=False,
                check_domain=False,
            )
        )

    analytic = float(jax.grad(scalar)(jnp.asarray(0.0, dtype=dtype)))
    h = jnp.asarray(step, dtype=dtype)
    numeric = float((scalar(h) - scalar(-h)) / (2.0 * h))

    assert np.isfinite(numeric), f"d/d{name}: step {step:g} left the domain"
    assert analytic == pytest.approx(numeric, rel=1e-6), (
        f"d/d{name}: autodiff {analytic:.10e} vs finite difference {numeric:.10e}"
    )


# --------------------------------------- CDOM-fluorescence wiring (M5 task 5) ----


def test_forward_composes_cdom_fluorescence_additively(l23_small_inelastic_batch):
    """forward(default + cdom_fl) == forward(default) + scale * K_cdom.

    The M5 task-5 gate's additive claim, spot-checked at the kernel's value —
    the :func:`test_forward_composes_fluorescence_additively` twin for the
    third term (CDOM design §2). Algebraically exact in **Rrs space** (the
    composition law's space — ``K_cdom`` ends in ``rrs_to_Rrs``, so the term
    adds above the surface; a difference of ``rrs_forward`` outputs would
    pick up Lee's non-linear conversion instead); the rrs round trip inside
    costs ~1 ULP (float32). ``scale=2.0`` — deliberately not the default
    1.0, so a wiring that dropped or ignored the amplitude cannot pass. The
    same identity is asserted a second time from ``rrs_forward`` outputs
    explicitly converted up, pinning that the term is additive at exactly
    the composed layer and not by accident of ``forward``'s final
    conversion.
    """
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    scale = 2.0

    with_cdom = np.asarray(
        H.forward(
            *args,
            inelastic=Inelastic(cdom_fl=CDOMFl(scale=scale)),
            corrections=False,
            check_domain=False,
        )
    )
    without = np.asarray(
        H.forward(*args, inelastic=Inelastic(), corrections=False, check_domain=False)
    )
    k_cdom = np.asarray(cdom_fl.cdom_kernel(batch.iops, batch.geometry, batch.wave))

    np.testing.assert_allclose(
        with_cdom, without + scale * k_cdom, rtol=5e-6, atol=1e-10
    )
    # The term only adds, and visibly so in the blue-green (the Hawes band).
    i460 = int(np.abs(np.asarray(batch.wave) - 460.0).argmin())
    assert np.all((with_cdom - without)[:, i460] > 0.0)

    # The rrs_forward form of the same identity, converted up explicitly.
    rrs_with = H.rrs_forward(
        *args,
        inelastic=Inelastic(cdom_fl=CDOMFl(scale=scale)),
        corrections=False,
        check_domain=False,
    )
    rrs_without = H.rrs_forward(
        *args, inelastic=Inelastic(), corrections=False, check_domain=False
    )
    np.testing.assert_allclose(
        np.asarray(C.rrs_to_Rrs(rrs_with)),
        np.asarray(C.rrs_to_Rrs(rrs_without)) + scale * k_cdom,
        rtol=5e-6,
        atol=1e-10,
    )


def test_cdom_fl_alone_composes(l23_small_inelastic_batch):
    """cdom_fl set with raman/fluorescence off still composes its term.

    The regression for the M5 task-5 guard fix: ``_apply_inelastic``'s
    early return used to test only ``raman or fluorescence``, so this exact
    configuration would have silently returned the untouched elastic ``rrs``
    — a plausible-looking array with the requested physics missing, the
    failure mode this module's error-guard tests exist to prevent. Asserted
    both ways: the output is *not* the elastic one bitwise, and it *is*
    elastic + K_cdom (scale = 1, the default) to composition-law tolerance.
    """
    batch = l23_small_inelastic_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    cdom_only = Inelastic(raman=False, fluorescence=False, cdom_fl=CDOMFl())

    composed = np.asarray(
        H.forward(*args, inelastic=cdom_only, corrections=False, check_domain=False)
    )
    elastic = np.asarray(H.forward(*args, check_domain=False))
    k_cdom = np.asarray(cdom_fl.cdom_kernel(batch.iops, batch.geometry, batch.wave))

    assert not np.array_equal(composed, elastic)  # the pre-fix silent no-op
    np.testing.assert_allclose(composed, elastic + k_cdom, rtol=5e-6, atol=1e-10)
