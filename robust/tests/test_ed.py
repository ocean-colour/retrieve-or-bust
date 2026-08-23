"""
Tests for :mod:`robust.rt.ed` (inelastic coding plan, M1 task 1).

Split like every data-touching module: the packaged ``ed_l23.npz`` ships with
the repo, so its properties and the interpolation logic run everywhere — CI
included — while the golden comparison against the raw X=2 netCDFs carries
``needs_l23_inelastic`` (the elastic ``needs_l23`` covers only the X=1 files;
a machine holding those three but not the inelastic six must skip here, not
fail).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import ed as E
from robust.rt.types import Geometry

from .conftest import needs_l23_inelastic

WAVE_TABLE, ED_TABLE = E.load_table()


# ---------------------------------------------------------- packaged table ----


def test_table_shape_grid_and_positivity():
    """Three spectra on the canonical grid, all finite and positive."""
    assert ED_TABLE.shape == (3, C.N_WAVE)
    np.testing.assert_array_equal(WAVE_TABLE, C.WAVE)
    assert np.all(np.isfinite(ED_TABLE)) and np.all(ED_TABLE > 0.0)


def test_table_falls_with_zenith_everywhere():
    """A lower sun delivers less irradiance at every wavelength.

    This is the ordering the generator asserts before packaging; re-asserted
    here so a scrambled or re-sorted file fails in CI, not just at generation.
    """
    assert np.all(np.diff(ED_TABLE, axis=0) < 0.0)


def test_table_magnitudes_are_solar():
    """Peak Ed(0 deg) is O(1) W m^-2 nm^-1 — the scale sunlight actually has.

    A unit slip (mW, per-m, per-micron) would move this by orders of magnitude
    while leaving every shape-based test green.
    """
    peak = ED_TABLE[0].max()
    assert 0.5 < peak < 5.0


@needs_l23_inelastic
def test_table_matches_raw_netcdf():
    """The packaged spectra reproduce the raw X=2 files at rtol 1e-5."""
    from ocpy.hydrolight import loisel23

    for row, zenith in enumerate(E.ZENITH_ANCHORS):
        ds = loisel23.load_ds(2, int(zenith))
        raw = ds["Ed_0+"].values.mean(axis=0, dtype=np.float64)
        np.testing.assert_allclose(ED_TABLE[row], raw, rtol=1e-5)
        ds.close()


# ------------------------------------------------------- zenith interpolation ----


@pytest.mark.parametrize("row,theta", [(0, 0.0), (1, 30.0), (2, 60.0)])
def test_anchors_are_exact(row, theta):
    """At the three packaged zeniths the interpolation returns the row itself."""
    np.testing.assert_allclose(
        np.asarray(E.Ed(theta)), ED_TABLE[row], rtol=1e-6, atol=0.0
    )


def test_midpoint_is_the_mean_of_its_anchors():
    """Linear in theta_s: Ed(15 deg) is exactly the 0/30 average."""
    np.testing.assert_allclose(
        np.asarray(E.Ed(15.0)), 0.5 * (ED_TABLE[0] + ED_TABLE[1]), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(E.Ed(45.0)), 0.5 * (ED_TABLE[1] + ED_TABLE[2]), rtol=1e-6
    )


def test_clamped_outside_the_anchor_range():
    """Beyond 0-60 deg the nearest anchor is returned, never an extrapolation."""
    np.testing.assert_array_equal(np.asarray(E.Ed(-5.0)), np.asarray(E.Ed(0.0)))
    np.testing.assert_array_equal(np.asarray(E.Ed(75.0)), np.asarray(E.Ed(60.0)))


def test_batched_theta_s():
    """A batch of zeniths yields a stacked (batch, wave) result, row for row."""
    batched = np.asarray(E.Ed(jnp.asarray([0.0, 15.0, 60.0])))
    assert batched.shape == (3, C.N_WAVE)
    np.testing.assert_allclose(batched[0], np.asarray(E.Ed(0.0)), rtol=1e-6)
    np.testing.assert_allclose(batched[1], np.asarray(E.Ed(15.0)), rtol=1e-6)
    np.testing.assert_allclose(batched[2], np.asarray(E.Ed(60.0)), rtol=1e-6)


# --------------------------------------------------- wavelength interpolation ----


def test_wavelength_interpolation_matches_numpy_interp():
    """Off-grid wavelengths agree with ``numpy.interp`` on the same table."""
    wave = jnp.asarray([352.5, 487.5, 683.0, 712.5])
    ours = np.asarray(E.Ed(30.0, wave))
    reference = np.interp(np.asarray(wave), WAVE_TABLE, ED_TABLE[1])
    np.testing.assert_allclose(ours, reference, rtol=1e-6)


def test_wavelength_clamped_at_grid_ends():
    """Outside 350-750 nm the end value is returned (the ``bb_w`` precedent)."""
    out = np.asarray(E.Ed(30.0, jnp.asarray([300.0, 350.0, 750.0, 800.0])))
    assert out[0] == out[1]
    assert out[2] == out[3]


# ------------------------------------------------------------------ override ----


def test_override_replaces_the_packaged_sky_and_ignores_theta():
    """An override is one sky: it is interpolated, and theta_s does nothing."""
    wave_ed = jnp.asarray([350.0, 550.0, 750.0])
    flat = jnp.asarray([2.0, 1.0, 2.0])
    wave = jnp.asarray([450.0, 550.0, 650.0])

    a = np.asarray(E.Ed(0.0, wave, override=(wave_ed, flat)))
    b = np.asarray(E.Ed(60.0, wave, override=(wave_ed, flat)))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, [1.5, 1.0, 1.5], rtol=1e-6)


def test_geometry_ed_pair_plumbs_through():
    """The :attr:`Geometry.Ed` pair is accepted verbatim as the override."""
    geom = dataclasses.replace(
        Geometry.nadir(jnp.asarray(30.0)),
        Ed=(jnp.linspace(350.0, 750.0, 5), jnp.full(5, 1.2)),
    )
    out = np.asarray(E.Ed(geom.theta_s, override=geom.Ed))
    np.testing.assert_allclose(out, np.full(C.N_WAVE, 1.2), rtol=1e-6)


# --------------------------------------------------------------------- ratio ----


def test_ratio_matches_hand_computation():
    """``ratio`` is Ed(num)/Ed(den), each via the same interpolation."""
    lam = np.asarray([450.0, 550.0, 650.0])
    lam_prime = np.asarray([417.0, 500.5, 581.0])
    ours = np.asarray(E.ratio(30.0, jnp.asarray(lam_prime), jnp.asarray(lam)))
    by_hand = np.interp(lam_prime, WAVE_TABLE, ED_TABLE[1]) / np.interp(
        lam, WAVE_TABLE, ED_TABLE[1]
    )
    np.testing.assert_allclose(ours, by_hand, rtol=1e-6)


def test_ratio_is_not_flat():
    """The Raman-relevant ratio really does vary spectrally.

    The assessment's flat-Ed error (+60 %/−50 %) exists because this ratio is
    far from constant; if it ever came out flat, the packaged data would be
    broken in a way the golden tests might miss.
    """
    lam = C.WAVE[C.WAVE >= 400.0]
    lam_prime = 1.0 / (1.0 / lam + 3400e-7)  # the Raman excitation map, nm
    r = np.asarray(E.ratio(30.0, jnp.asarray(lam_prime), jnp.asarray(lam)))
    assert r.max() / r.min() > 1.5


# ------------------------------------------------------------- JAX behaviour ----


def test_jit_and_vmap_traverse():
    """Compiled and mapped evaluation agree with the eager path."""
    thetas = jnp.asarray([0.0, 22.0, 47.0])
    eager = np.asarray(E.Ed(thetas))
    jitted = np.asarray(jax.jit(E.Ed)(thetas))
    mapped = np.asarray(jax.vmap(E.Ed)(thetas))
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)
    np.testing.assert_allclose(mapped, eager, rtol=1e-6)


def test_grad_wrt_theta_matches_finite_differences(jax_x64):
    """d(Ed)/d(theta_s) from autodiff agrees with central differences.

    Float64 via the fixture, dtypes pinned on the arrays (the elastic record's
    §2 lesson); evaluated at 22 deg, inside the 0-30 segment, where the
    piecewise-linear interpolation is smooth.
    """
    theta = jnp.asarray(22.0, dtype=jnp.float64)
    h = 1e-3

    grad = np.asarray(jax.grad(lambda t: E.Ed(t).sum())(theta))
    fd = (np.asarray(E.Ed(theta + h)).sum() - np.asarray(E.Ed(theta - h)).sum()) / (
        2.0 * h
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


def test_grad_wrt_override_values(jax_x64):
    """The override values are differentiable — the seam future skies need."""
    wave_ed = jnp.linspace(350.0, 750.0, 9, dtype=jnp.float64)
    values = jnp.full(9, 1.3, dtype=jnp.float64)

    grad = np.asarray(
        jax.grad(lambda v: E.Ed(30.0, override=(wave_ed, v)).sum())(values)
    )
    assert np.all(np.isfinite(grad))
    assert grad.sum() > 0.0
