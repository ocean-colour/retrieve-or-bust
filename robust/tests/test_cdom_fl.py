"""
Tests for the analytic CDOM-fluorescence kernel (CDOM design M5, task 3).

The design §5.2 correctness pins, honestly labeled: with no truth channel (L23
omits CDOM-fl by design) and no BING reference (BING never implemented it),
these tests verify the *implementation* — the Hawes η_Y wiring against its
tabulated inputs, the structural physics (non-negativity, red-shifted peak,
the 350 nm clamp), and quadrature convergence — **not** the physics accuracy.
Until M6's HydroLight truth runs, the term is "Hawes-consistent and
plausible", never "validated" (design §8).

Constants provenance (mirrored from ``robust/rt/cdom_fl.py``): the functional
form is peer-reviewed-verified (Zhai et al. 2017, Eqs. 5–8, read directly);
the FA7 numeric constants are from Mobley's Ocean Optics Web Book, retrieved
2026-08-29, **not independently cross-checked** — the pins below reproduce
exactly those tabulated numbers, so re-pinning is trivial if JXP corrects the
source (Q&A CQ2).

Everything here runs on the committed fixtures (no ``$OS_COLOR``).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import cdom_fl as C
from robust.rt import conventions, ed

# ------------------------------------------------------------ the Hawes η_Y ----


def test_a0_table_reproduced_at_the_gaussian_peak():
    """η_Y at each tabulated λ_e, evaluated at its own emission peak, is A0(λ_e).

    At the peak the Gaussian argument vanishes (1/λ = A1/λ_e + B1), so
    η_Y = A0 exactly. **What this really tests**: that the A0 table lookup and
    ``jnp.interp`` wiring are correct — A0 is a directly tabulated *input*, so
    this pins the plumbing, not the physics. It also documents exactly which
    numbers are pinned (the flagged Ocean Optics Web Book FA7 values), so a
    corrected primary source (CQ2) means re-pinning here, deliberately.
    """
    for wave_e, a0 in zip(C.HAWES_A0_WAVE, C.HAWES_A0, strict=True):
        wave_peak = 1.0 / (C.HAWES_A1 / wave_e + C.HAWES_B1)
        eta = float(C.eta_hawes(jnp.asarray(wave_peak), jnp.asarray(wave_e)))
        assert eta == pytest.approx(a0, rel=1e-6), f"lambda_e = {wave_e}"


def test_eta_is_nonnegative_and_gated():
    """Quantum bookkeeping, structurally: η_Y ≥ 0 everywhere (A0 > 0 and
    exp > 0), and ``g_Y`` zeroes excitation outside [310, 490] nm (Eq. 8)."""
    wave_em = jnp.linspace(320.0, 750.0, 87)[:, None]
    wave_ex = jnp.linspace(250.0, 600.0, 71)[None, :]
    eta = np.asarray(C.eta_hawes(wave_em, wave_ex))
    assert np.all(np.isfinite(eta))
    assert eta.min() >= 0.0

    ex = np.linspace(250.0, 600.0, 71)
    inside = (ex >= C.GY_EX_MIN) & (ex <= C.GY_EX_MAX)
    assert np.all(eta[:, ~inside] == 0.0)
    assert np.all(eta[:, inside].max(axis=0) > 0.0)


def test_emission_peak_is_red_shifted():
    """The η_Y *peak* is red of the excitation for every admissible λ_e.

    Analytically: peak wavenumber ``A1/λ_e + B1 < 1/λ_e`` whenever
    ``λ_e < (1−A1)/B1 ≈ 656 nm`` — true across g_Y's whole support. Checked
    numerically on a dense emission grid, per excitation node.

    Stated plainly rather than oversold: the Gaussian-in-wavenumber form does
    **not** enforce a strict Stokes shift — the blue tail is real (η_Y at
    λ = λ_e is ~6 % of the peak at λ_e = 350 nm, ~22 % at 490 nm), so only
    the peak's red shift is asserted, never that blue emission is zero.
    """
    wave_em = np.linspace(300.0, 750.0, 4501)
    for wave_e in np.linspace(C.CDOM_EX_MIN, C.CDOM_EX_MAX, 15):
        eta = np.asarray(C.eta_hawes(jnp.asarray(wave_em), jnp.asarray(wave_e)))
        peak = wave_em[np.argmax(eta)]
        analytic = 1.0 / (C.HAWES_A1 / wave_e + C.HAWES_B1)
        assert peak > wave_e
        assert peak == pytest.approx(analytic, abs=0.5)
        # The blue tail exists but is subdominant — documented, not asserted away.
        at_excitation = float(C.eta_hawes(jnp.asarray(wave_e), jnp.asarray(wave_e)))
        assert 0.0 < at_excitation < 0.3 * eta.max()


# ----------------------------------------------------------- excitation grid ----


def test_excitation_grid_is_the_hard_clamp():
    """The grid *starts* at 350 nm (the clamp is the grid, not arithmetic),
    ends at g_Y's 490 nm cutoff, and its 29 nodes land exactly on canonical
    grid points, so the excitation IOPs interpolate losslessly."""
    grid = np.asarray(C.cdom_excitation_grid())
    assert grid.shape == (29,)
    assert grid[0] == C.CDOM_EX_MIN == 350.0
    assert grid[-1] == C.CDOM_EX_MAX == 490.0
    np.testing.assert_array_equal(np.diff(grid), C.CDOM_EX_STEP)
    canonical = np.asarray(conventions.canonical_wave())
    assert np.all(np.isin(grid, canonical))


def test_kernel_never_reads_iops_or_ed_below_350(
    l23_small_inelastic_batch, monkeypatch
):
    """The clamp, proved at the seams: every wavelength the kernel hands to
    ``interp_spectrum`` (IOPs) or ``ed.Ed`` (the sky) is ≥ 350 nm.

    Spies wrap the real functions, so the kernel still computes — the test
    fails if any read (excitation or emission side) dips below the clamp.
    The truncated-*fraction* diagnostic is task 4; this pins only that the
    kernel's support is provably clamped.
    """
    batch = l23_small_inelastic_batch
    seen: list[float] = []

    real_interp = conventions.interp_spectrum
    real_ed = ed.Ed

    def spy_interp(wave_new, grid, spectra):
        seen.append(float(np.asarray(wave_new).min()))
        return real_interp(wave_new, grid, spectra)

    def spy_ed(theta_s, wave=None, **kwargs):
        if wave is not None:
            seen.append(float(np.asarray(wave).min()))
        return real_ed(theta_s, wave, **kwargs)

    monkeypatch.setattr(conventions, "interp_spectrum", spy_interp)
    monkeypatch.setattr(ed, "Ed", spy_ed)

    kernel = np.asarray(C.cdom_kernel(batch.iops, batch.geometry, batch.wave))
    assert np.all(np.isfinite(kernel))
    assert len(seen) >= 5  # 3 IOP interps + 2 Ed calls
    assert min(seen) >= C.CDOM_EX_MIN


# ------------------------------------ truncated-fraction diagnostic (task 4) ----

#: The measured truncated fractions (0.25 nm quadrature, canonical grid),
#: banded generously (±0.03 absolute) — the diagnostic *characterizes* the
#: clamp's cost (design §2), it does not gate it to zero. The implementation
#: record §8.1 carries the same numbers as the documented caveat.
TRUNCATED_FRACTION_BANDS = {
    350.0: 0.846,
    400.0: 0.566,
    450.0: 0.297,
    500.0: 0.142,
    550.0: 0.083,
    600.0: 0.070,
    650.0: 0.078,
    700.0: 0.103,
    750.0: 0.146,
}


def test_truncated_fraction_pinned_at_representative_wavelengths():
    """The clamp's measured cost, pinned as bands per emission wavelength.

    ``fraction(λ) = ∫₃₁₀³⁵⁰ η_Y dλ_e / ∫₃₁₀⁴⁹⁰ η_Y dλ_e`` — a property of the
    Hawes FA7 function alone (no IOPs, no Ed). The headline caveat: **57 % of
    the nominal 310–490 nm-excited emission at λ = 400 nm is excluded by the
    production 350 nm clamp** (30 % at 450 nm, ~7 % at the 605 nm minimum,
    rising again to ~15 % at 750 nm through the sub-350 Gaussians' red
    tails). Large blue-band values were design §8's flagged risk — recorded
    here and in the implementation record §8.1, not asserted away. Re-pin
    deliberately if the FA7 constants are corrected (Q&A CQ2).
    """
    wave = conventions.canonical_wave()
    fraction = np.asarray(C.truncated_excitation_fraction())
    w = np.asarray(wave)
    assert fraction.shape == w.shape
    assert np.all(np.isfinite(fraction))
    assert np.all((fraction >= 0.0) & (fraction <= 1.0))
    for lam, expected in TRUNCATED_FRACTION_BANDS.items():
        i = int(np.abs(w - lam).argmin())
        assert abs(fraction[i] - expected) < 0.03, (
            f"lambda_em = {lam:.0f} nm: measured {fraction[i]:.4f}, "
            f"pinned band {expected:.3f} +/- 0.03"
        )


def test_truncated_fraction_quadrature_converged():
    """The default 0.25 nm sub-grid agrees with a 2× refinement (measured max
    3.5e-6 relative on the canonical grid) — the reported numbers are
    quadrature-converged, not grid artifacts."""
    native = np.asarray(C.truncated_excitation_fraction())
    refined = np.asarray(C.truncated_excitation_fraction(quad_step=0.125))
    np.testing.assert_allclose(native, refined, rtol=1e-5, atol=0.0)


def test_truncated_fraction_guard_and_domain():
    """The 0/0 guard: far outside the Hawes band both integrals underflow to
    exactly 0.0 and the fraction is *defined* as 0 (no emission → nothing
    truncated), never a silent NaN. On the canonical grid the denominator is
    strictly positive, so the guard is provably inert there (min measured
    fraction ~0.070 at 605 nm)."""
    pathological = np.asarray(
        C.truncated_excitation_fraction(wave=jnp.asarray([100.0]))
    )
    assert pathological[0] == 0.0  # both integrals underflow — guarded, not NaN
    on_grid = np.asarray(C.truncated_excitation_fraction())
    assert np.all(on_grid > 0.0)  # the guard never fires on the canonical grid


# The task-4 gate's other half — "the production kernel provably never reads
# IOPs or Ed below 350 nm" — is fully covered by the task-3 spy test
# ``test_kernel_never_reads_iops_or_ed_below_350`` above (it wraps the real
# ``interp_spectrum``/``ed.Ed`` seams and asserts every wavelength the kernel
# reads is >= 350 nm) together with ``test_excitation_grid_is_the_hard_clamp``
# (the grid *is* the clamp). Not duplicated here; ``eta_hawes`` itself being
# evaluated below 350 nm by ``truncated_excitation_fraction`` is deliberate
# and touches no IOPs or Ed (see its docstring).


# ------------------------------------------------------------------ the kernel ----


def test_kernel_requires_a_cdom(l23_small_inelastic_batch):
    """No a_cdom, no source term — a loud error naming the field."""
    batch = l23_small_inelastic_batch
    stripped = dataclasses.replace(batch.iops, a_cdom=None)
    with pytest.raises(ValueError, match="a_cdom"):
        C.cdom_kernel(stripped, batch.geometry, batch.wave)


def test_kernel_is_physical(l23_small_inelastic_batch):
    """K_cdom ≥ 0, finite, and broad — energy in the blue-green through the
    yellow (the Hawes emission band), no 685 nm-style line."""
    batch = l23_small_inelastic_batch
    kernel = np.asarray(C.cdom_kernel(batch.iops, batch.geometry, batch.wave))
    wave = np.asarray(batch.wave)
    assert kernel.shape == (batch.n_sample, batch.n_wave)
    assert np.all(np.isfinite(kernel))
    assert kernel.min() >= 0.0
    # The median spectrum peaks inside the Hawes emission band (~430-570 nm
    # for 350-490 nm excitation), not at an emission line.
    peak_wave = wave[np.argmax(np.median(kernel, axis=0))]
    assert 400.0 <= peak_wave <= 600.0


def test_quadrature_convergence_under_grid_refinement(l23_small_inelastic_batch):
    """Design §5.2: the kernel is stable between the native 5 nm excitation
    grid and a 2× (2.5 nm) refinement — rtol 1e-2, measured max relative
    difference 5.6e-3 on the 150-sample fixture (the refinement re-samples
    the piecewise-linear A0 table, Ed, and the excitation IOPs between the
    native nodes, so exact equality is not expected)."""
    batch = l23_small_inelastic_batch
    native = np.asarray(C.cdom_kernel(batch.iops, batch.geometry, batch.wave))
    refined = np.asarray(
        C.cdom_kernel(
            batch.iops,
            batch.geometry,
            batch.wave,
            wave_ex=C.cdom_excitation_grid(step=C.CDOM_EX_STEP / 2.0),
        )
    )
    np.testing.assert_allclose(refined, native, rtol=1e-2)


def test_kernel_ed_override_reaches_the_quadrature(l23_small_inelastic_batch):
    """The geometry.Ed seam feeds the excitation integral end to end."""
    batch = l23_small_inelastic_batch
    flat_sky = (jnp.asarray([350.0, 750.0]), jnp.asarray([1.0, 1.0]))
    geometry = dataclasses.replace(batch.geometry, Ed=flat_sky)

    default = np.asarray(C.cdom_kernel(batch.iops, batch.geometry, batch.wave))
    flat = np.asarray(C.cdom_kernel(batch.iops, geometry, batch.wave))

    assert np.all(np.isfinite(flat)) and flat.min() >= 0.0
    i460 = int(np.abs(np.asarray(batch.wave) - 460.0).argmin())
    assert not np.allclose(default[:, i460], flat[:, i460], rtol=1e-3)


def test_kernel_jit_and_vmap(l23_small_inelastic_batch):
    """Compiled and mapped evaluation agree with eager."""
    batch = l23_small_inelastic_batch
    eager = np.asarray(C.cdom_kernel(batch.iops, batch.geometry, batch.wave))
    jitted = np.asarray(jax.jit(C.cdom_kernel)(batch.iops, batch.geometry, batch.wave))
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)

    mapped = np.asarray(
        jax.vmap(lambda io, g: C.cdom_kernel(io, g, batch.wave))(
            batch.iops, batch.geometry
        )
    )
    np.testing.assert_allclose(mapped, eager, rtol=1e-6)


def test_kernel_gradients_are_finite(l23_small_inelastic_batch):
    """Smoke gradient through the kernel (a_cdom included) — finite, and the
    a_cdom gradient is nonzero (the source term really is differentiable).
    The full FD-vs-grad gate, `scale` included, is task 7."""
    batch = l23_small_inelastic_batch
    row = jax.tree_util.tree_map(lambda leaf: leaf[0], batch.iops)
    geometry = dataclasses.replace(
        jax.tree_util.tree_map(lambda leaf: leaf[0], batch.geometry), Ed=None
    )

    grad = jax.grad(lambda io: C.cdom_kernel(io, geometry, batch.wave).sum())(row)
    for name in ("a", "bb_p", "a_cdom"):
        leaf = np.asarray(getattr(grad, name))
        assert np.all(np.isfinite(leaf)), name
    assert np.any(np.asarray(grad.a_cdom) != 0.0)
