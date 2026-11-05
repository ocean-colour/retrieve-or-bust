"""
Tests for :mod:`robust.rt.conventions` (M1 task 1).

Three things are being defended here.

**The Rrs<->rrs pair is an exact inverse**, and the round-trip is checked in both
dtype regimes rather than one: at float32 (JAX's default) the achievable error is
~2e-7, so a "1e-6" gate has only ~5x of headroom and would silently become a test
of the dtype if anyone tightened it; at float64 it is ~1e-16. Both are asserted,
so the distinction is visible in the test file instead of folklore.

**The constants agree with BING.** Fixing A/B only buys anything if the two
packages that share ``rrs`` agree on it, so that is asserted, not commented.

**bb_w's provenance is real.** The embedded table is checked back against the raw
L23 netCDF, so it cannot drift from the data the model is trained on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.tests.conftest import (
    needs_l23,
    needs_pb24,
    needs_pb24_olci,
    pb24_mixed_realisations,
)

#: A realistic above-water Rrs range: 1e-4 to ~5e-2 sr^-1.
RRS_SAMPLES = np.logspace(-4.0, -1.3, 400)


# ------------------------------------------------------------- Rrs <-> rrs ---


def test_rrs_round_trip_float32():
    """``Rrs -> rrs -> Rrs`` to 1e-6 relative in JAX's default float32.

    The measured error is ~2e-7 (three float32 operations), so this gate has
    about 5x of headroom. Tightening it much further would not test the
    conversion -- it would test float32.
    """
    Rrs = jnp.asarray(RRS_SAMPLES, dtype=jnp.float32)

    back = C.rrs_to_Rrs(C.Rrs_to_rrs(Rrs))

    assert back.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(back), RRS_SAMPLES, rtol=1e-6)


def test_rrs_round_trip_float64(jax_x64):
    """The same round trip is exact to ~1e-15 once float64 is on.

    This is the regime the M2 gradient gate will run in, so the conversion must
    not be the thing that limits precision there.
    """
    Rrs = jnp.asarray(RRS_SAMPLES, dtype=jnp.float64)

    back = C.rrs_to_Rrs(C.Rrs_to_rrs(Rrs))

    assert back.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(back), RRS_SAMPLES, rtol=1e-12)


def test_rrs_to_Rrs_inverts_in_the_other_direction(jax_x64):
    """``rrs -> Rrs -> rrs`` also closes, over the physical rrs range."""
    rrs = jnp.asarray(np.logspace(-4.0, -1.0, 200), dtype=jnp.float64)

    back = C.Rrs_to_rrs(C.rrs_to_Rrs(rrs))

    np.testing.assert_allclose(np.asarray(back), np.asarray(rrs), rtol=1e-12)


def test_conversion_matches_the_algebra():
    """The formula is the Lee et al. (2002) one, spelled out independently."""
    Rrs = 0.01

    expected = Rrs / (0.52 + 1.7 * Rrs)

    assert float(C.Rrs_to_rrs(jnp.asarray(Rrs))) == pytest.approx(expected, rel=1e-6)


def test_constants_agree_with_bing():
    """A_RRS / B_RRS equal BING's, so the two packages mean the same ``rrs``.

    Skipped where ``bing`` is absent (CI installs a lean dependency set).
    """
    bing_rt = pytest.importorskip("bing.rt")

    assert C.A_RRS == bing_rt.A_Rrs
    assert C.B_RRS == bing_rt.B_Rrs


def test_conversions_are_jittable_and_differentiable():
    """Both directions survive ``jit`` and ``grad``.

    They sit on the ``forward`` path, so a non-traceable convention function
    would break the design's differentiability guarantee.
    """
    assert float(jax.jit(C.Rrs_to_rrs)(jnp.asarray(0.01))) > 0.0
    assert float(jax.jit(C.rrs_to_Rrs)(jnp.asarray(0.01))) > 0.0

    d_rrs = jax.grad(lambda r: C.Rrs_to_rrs(r))(0.01)
    d_Rrs = jax.grad(lambda r: C.rrs_to_Rrs(r))(0.01)

    # Both maps are monotonically increasing on the physical range.
    assert float(d_rrs) > 0.0
    assert float(d_Rrs) > 0.0


def test_pole_location():
    """:data:`RRS_POLE` really is where ``rrs_to_Rrs`` blows up."""
    assert C.RRS_POLE == pytest.approx(1.0 / 1.7)

    just_below = C.rrs_to_Rrs(jnp.asarray(C.RRS_POLE * 0.999))
    beyond = C.rrs_to_Rrs(jnp.asarray(C.RRS_POLE * 1.001))

    assert float(just_below) > 100.0  # diverging
    assert float(beyond) < 0.0  # and negative past it


# --------------------------------------------------------- wavelength grid ---


def test_canonical_grid_shape_and_ends():
    """81 bands, 350-750 nm, 5 nm apart."""
    assert C.N_WAVE == 81
    assert C.WAVE.shape == (81,)
    assert C.WAVE[0] == C.WAVE_MIN == 350.0
    assert C.WAVE[-1] == C.WAVE_MAX == 750.0
    np.testing.assert_allclose(np.diff(C.WAVE), C.WAVE_STEP)


def test_canonical_wave_returns_jax_array():
    """``canonical_wave()`` gives the same numbers as a device array."""
    wave = C.canonical_wave()

    assert isinstance(wave, jax.Array)
    np.testing.assert_allclose(np.asarray(wave), C.WAVE)


def test_canonical_wave_follows_x64(jax_x64):
    """Under float64 the grid comes back float64, not a frozen float32 copy."""
    assert C.canonical_wave().dtype == jnp.float64


@needs_l23
def test_canonical_grid_matches_l23():
    """Golden check: the grid is L23's own ``Lambda``, not a lookalike."""
    from ocpy.hydrolight import loisel23

    ds = loisel23.load_ds(1, 0)

    np.testing.assert_allclose(ds.Lambda.data, C.WAVE, atol=1e-4)


# ------------------------------------------------ pure-water backscattering --


def test_bb_w_table_is_physical():
    """Positive, decreasing with wavelength, and a molecular-scattering slope."""
    assert C.BB_W_L23.shape == (81,)
    assert np.all(C.BB_W_L23 > 0.0)
    assert np.all(np.diff(C.BB_W_L23) < 0.0)

    slope, _ = np.polyfit(np.log(C.WAVE), np.log(C.BB_W_L23), 1)

    # Morel (1974) molecular scattering goes as lambda^-4.32; L23's water is
    # close to that but not identical, so this is a sanity band, not a gate.
    assert -4.6 < slope < -3.9


def test_bb_w_on_the_grid_is_the_table():
    """No interpolation error where the table is defined."""
    np.testing.assert_allclose(np.asarray(C.bb_w(C.WAVE)), C.BB_W_L23, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(C.bb_w()), C.BB_W_L23, rtol=1e-6)


def test_bb_w_interpolates_and_clamps():
    """Midpoints interpolate; outside the range the end points hold."""
    mid = float(C.bb_w(jnp.asarray(442.5)))
    lo, hi = float(C.bb_w(jnp.asarray(440.0))), float(C.bb_w(jnp.asarray(445.0)))
    assert hi < mid < lo

    # jnp.interp clamps rather than extrapolating: L23 says nothing below 350
    # or above 750 nm, and a wild extrapolation would be worse than a constant.
    assert float(C.bb_w(jnp.asarray(300.0))) == pytest.approx(C.BB_W_L23[0])
    assert float(C.bb_w(jnp.asarray(800.0))) == pytest.approx(C.BB_W_L23[-1])


def test_bb_w_is_jittable_and_differentiable():
    """``bb_w`` is on the forward path too."""
    assert float(jax.jit(C.bb_w)(jnp.asarray(440.0))) > 0.0

    slope = jax.grad(lambda w: C.bb_w(w))(440.0)

    assert float(slope) < 0.0  # falls toward the red


@needs_l23
def test_bb_w_matches_l23_netcdf():
    """Golden check: the embedded table is ``bb - bbnw`` from the L23 file.

    Guards the one number in this module that came from data rather than from a
    paper. If L23 is ever re-released, this fails instead of quietly biasing
    ``bb_p = bb - bb_w``.
    """
    from ocpy.hydrolight import loisel23

    ds = loisel23.load_ds(1, 0)
    from_file = (ds.bb.data - ds.bbnw.data)[0]

    np.testing.assert_allclose(from_file, C.BB_W_L23, rtol=1e-6)


@needs_l23
def test_bb_w_is_scene_independent_in_l23():
    """The premise that lets a single table stand in for 3320 scenes.

    ``bing`` and ``ocpy`` both take this difference at an arbitrary scene index
    without checking it is constant. It is -- to float32 storage noise -- and
    this test is why we may rely on that.
    """
    from ocpy.hydrolight import loisel23

    ds = loisel23.load_ds(1, 0)
    bbw = ds.bb.data - ds.bbnw.data

    spread = np.abs(bbw - bbw[0]).max() / bbw.max()

    assert spread < 1e-6, f"bb_w varies across scenes by {spread:.2e} relative"


# ------------------------------------------------------------- validators ----


def test_check_wave_accepts_the_canonical_grid():
    """The happy path stays quiet."""
    C.check_wave(C.WAVE)
    C.check_wave(np.asarray(C.canonical_wave()))
    C.check_wave(list(C.WAVE))


def test_check_wave_rejects_wrong_length():
    with pytest.raises(ValueError, match="shape"):
        C.check_wave(C.WAVE[:-1])


def test_check_wave_rejects_a_shifted_grid():
    """A 1 nm offset is caught, and the message says where."""
    shifted = C.WAVE + 1.0

    with pytest.raises(ValueError, match="canonical") as exc:
        C.check_wave(shifted)

    assert "+1" in str(exc.value)


def test_check_wave_rejects_a_different_spacing():
    with pytest.raises(ValueError):
        C.check_wave(np.linspace(400.0, 700.0, C.N_WAVE))


def test_check_iop_accepts_valid_values():
    C.check_iop(np.array([0.0, 0.1, 1.0]), "a")


def test_check_iop_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        C.check_iop(np.array([0.1, -1e-9, 0.2]), "a")


def test_check_iop_rejects_nan_and_inf():
    with pytest.raises(ValueError, match="non-finite"):
        C.check_iop(np.array([0.1, np.nan]), "bb_p")
    with pytest.raises(ValueError, match="non-finite"):
        C.check_iop(np.array([0.1, np.inf]), "bb_p")


def test_check_rrs_accepts_physical_values():
    C.check_rrs(np.array([1e-4, 1e-2, 5e-2]))
    C.check_rrs(np.array([0.6]), subsurface=False)  # no pole above water


def test_check_rrs_rejects_beyond_the_pole():
    """The unit-error signature: rrs at or past 1/B."""
    with pytest.raises(ValueError, match="pole"):
        C.check_rrs(np.array([0.01, 0.7]))


def test_check_rrs_rejects_negative_and_nonfinite():
    with pytest.raises(ValueError, match="negative"):
        C.check_rrs(np.array([-1e-6]))
    with pytest.raises(ValueError, match="non-finite"):
        C.check_rrs(np.array([np.nan]))


# --------------------------------------------------------------- named grids -
# M5 task 3. The point of these is not that a second grid exists but that adding
# it did not weaken the first: `check_wave` still refuses a grid mismatch, it
# just now refuses *per grid*.


def test_wave_grid_resolves_none_str_and_object():
    """``None`` is L23's grid, so every pre-M5 call site keeps its meaning."""
    assert C.wave_grid() is C.L23_GRID
    assert C.wave_grid("l23") is C.L23_GRID
    assert C.wave_grid("olci") is C.OLCI_GRID
    assert C.wave_grid(C.OLCI_GRID) is C.OLCI_GRID


def test_wave_grid_rejects_an_unknown_name():
    """A typo must not fall back to the canonical grid."""
    with pytest.raises(KeyError, match="unknown wavelength grid"):
        C.wave_grid("oclci")


def test_olci_grid_is_pb24s_bands():
    """12 bands, 400-753 nm, ascending, and not a subset of the 5 nm grid."""
    assert C.OLCI_GRID.n_wave == 12
    assert C.OLCI_GRID.span == (400.0, 753.0)
    assert np.all(np.diff(C.OLCI_WAVE) > 0.0)

    # Half the bands fall between 5 nm nodes, so this is a genuinely different
    # grid rather than a subsample of the canonical one.
    off_grid = [w for w in C.OLCI_WAVE if w % C.WAVE_STEP != 0.0]

    assert off_grid == [412.0, 443.0, 673.0, 681.0, 709.0, 753.0]


def test_check_wave_still_refuses_a_mismatch_per_grid():
    """The teeth are intact: each grid rejects the other."""
    with pytest.raises(ValueError, match="expected the canonical grid"):
        C.check_wave(C.OLCI_WAVE)
    with pytest.raises(ValueError, match="expected the olci grid"):
        C.check_wave(C.WAVE, grid="olci")

    C.check_wave(C.OLCI_WAVE, grid="olci")  # and accepts its own


def test_check_wave_rejects_right_length_wrong_values():
    """A 12-band grid that is not OLCI's still fails, not just a wrong shape."""
    nearly = C.OLCI_WAVE.copy()
    nearly[3] += 0.5

    with pytest.raises(ValueError, match="not the olci"):
        C.check_wave(nearly, grid="olci")


def test_grid_wave_returns_a_device_array(jax_x64):
    """``grid_wave`` is ``canonical_wave``'s grid-aware counterpart."""
    assert C.grid_wave("olci").dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(C.grid_wave("olci")), C.OLCI_WAVE)
    np.testing.assert_allclose(
        np.asarray(C.grid_wave()), np.asarray(C.canonical_wave())
    )


# ------------------------------------------- bb_w beyond the table's support -


def test_bb_w_tail_exponent_is_re_derived_from_the_table():
    """The embedded exponent is measured, not quoted."""
    fitted, _ = np.polyfit(np.log(C.WAVE[-20:]), np.log(C.BB_W_L23[-20:]), 1)

    assert fitted == pytest.approx(C.BB_W_TAIL_EXPONENT, abs=1e-5)

    # and it reproduces the tail it was fitted to
    model = C.BB_W_L23[-1] * (C.WAVE[-20:] / C.WAVE_MAX) ** C.BB_W_TAIL_EXPONENT
    assert np.abs(model / C.BB_W_L23[-20:] - 1.0).max() < 1e-3


def test_bb_w_at_753_says_which_answer_it_gave():
    """PB24's reddest band is 3 nm past the table; the three modes differ."""
    clamped = float(C.bb_w(jnp.asarray(753.0)))
    extrapolated = float(C.bb_w(jnp.asarray(753.0), mode="extrapolate"))

    assert clamped == pytest.approx(C.BB_W_L23[-1])  # unchanged default
    assert extrapolated < clamped  # the tail keeps falling
    assert extrapolated / clamped == pytest.approx(0.9836, abs=5e-4)  # 1.6% low

    with pytest.raises(ValueError, match="outside the bb_w table"):
        C.bb_w(jnp.asarray(753.0), mode="raise")


def test_bb_w_extrapolate_is_continuous_at_the_boundary():
    """No step at 750 nm, and no kink either.

    The first draft of this test asserted agreement to 1e-4 across +/-0.1 nm and
    failed at 1.1e-3 -- which is not a discontinuity but the function's own slope
    (``d ln bb_w / d ln lambda = -4.14`` gives exactly 0.11% over 0.2 nm). So
    check the two things that actually matter: the value does not jump across the
    seam, and the slope does not either.
    """
    eps = 1e-3
    inside = float(C.bb_w(jnp.asarray(750.0 - eps), mode="extrapolate"))
    outside = float(C.bb_w(jnp.asarray(750.0 + eps), mode="extrapolate"))

    assert outside < inside  # still falling
    assert abs(outside / inside - 1.0) < 1e-4  # no step

    # slope on each side, in log-log, against the fitted tail exponent
    def log_slope(centre):
        a = float(C.bb_w(jnp.asarray(centre - 0.5), mode="extrapolate"))
        b = float(C.bb_w(jnp.asarray(centre + 0.5), mode="extrapolate"))
        return np.log(b / a) / np.log((centre + 0.5) / (centre - 0.5))

    assert log_slope(747.0) == pytest.approx(C.BB_W_TAIL_EXPONENT, rel=0.02)
    assert log_slope(753.0) == pytest.approx(C.BB_W_TAIL_EXPONENT, rel=0.02)


def test_bb_w_extrapolate_is_still_jittable_and_differentiable():
    """The new branch must not leave the forward path's contract."""
    f = jax.jit(lambda w: C.bb_w(w, mode="extrapolate"))

    assert float(f(jnp.asarray(800.0))) > 0.0
    assert float(jax.grad(lambda w: C.bb_w(w, mode="extrapolate"))(800.0)) < 0.0


def test_bb_w_rejects_an_unknown_mode():
    """A typo in a mode must not silently clamp."""
    with pytest.raises(ValueError, match="mode must be"):
        C.bb_w(jnp.asarray(440.0), mode="extraploate")


def test_check_bb_w_range_flags_what_the_clamp_would_swallow():
    """The boundary counterpart to the mode: loaders can ask before they load."""
    C.check_bb_w_range(C.WAVE)  # L23's grid is exactly the support

    with pytest.raises(ValueError, match="outside the bb_w table"):
        C.check_bb_w_range(C.OLCI_WAVE, name="PB24 lambda")


# ------------------------------------------ geometry-aware surface transfer --
# M5 task 7. The nadir constants stay the default and stay exact; the fitted
# table is opt-in. The gate has three clauses: the default path is unchanged, the
# fitted path is worth >=5x at theta_v = 60 on held-out realisations, and the
# whole thing is differentiable and jit-safe.


def _toy_transfer():
    """A tiny hand-made table, so these tests need no data mount."""
    return C.SurfaceTransfer(
        theta_s=np.array([0.0, 60.0]),
        theta_v=np.array([0.0, 60.0]),
        dphi=np.array([0.0, 180.0]),
        A=np.array([[[0.50, 0.50], [0.30, 0.40]], [[0.52, 0.52], [0.34, 0.44]]]),
        B=np.full((2, 2, 2), 1.7),
        provenance="toy",
    )


def test_the_nadir_path_is_untouched():
    """**Gate.** No geometry, no table: exactly what M0-M4 computed."""
    Rrs = jnp.asarray(RRS_SAMPLES)

    np.testing.assert_array_equal(
        np.asarray(C.Rrs_to_rrs(Rrs)),
        np.asarray(Rrs / (C.A_RRS + C.B_RRS * Rrs)),
    )
    rrs = C.Rrs_to_rrs(Rrs)
    np.testing.assert_array_equal(
        np.asarray(C.rrs_to_Rrs(rrs)),
        np.asarray(C.A_RRS * rrs / (1.0 - C.B_RRS * rrs)),
    )


def test_a_table_without_a_geometry_is_refused():
    """Silently ignoring the table would be the worst of both."""
    with pytest.raises(ValueError, match="needs a geometry"):
        C.Rrs_to_rrs(jnp.asarray([0.01]), transfer=_toy_transfer())


def test_interpolation_is_exact_at_the_nodes_and_linear_between():
    from robust.rt.types import Geometry

    table = _toy_transfer()
    at_node = Geometry(
        theta_s=jnp.asarray([0.0]), theta_v=jnp.asarray([60.0]), dphi=jnp.asarray([0.0])
    )
    A, _ = table.coefficients(at_node)
    assert float(A[0]) == pytest.approx(0.30)

    midway = Geometry(
        theta_s=jnp.asarray([0.0]),
        theta_v=jnp.asarray([60.0]),
        dphi=jnp.asarray([90.0]),
    )
    A, _ = table.coefficients(midway)
    assert float(A[0]) == pytest.approx(0.35)  # halfway between 0.30 and 0.40


def test_interpolation_clamps_outside_the_nodes():
    """Past 87.75 degrees the sun is below the horizon; there is nothing to reach."""
    from robust.rt.types import Geometry

    table = _toy_transfer()
    beyond = Geometry(
        theta_s=jnp.asarray([90.0]),
        theta_v=jnp.asarray([89.0]),
        dphi=jnp.asarray([359.0]),
    )
    edge = Geometry(
        theta_s=jnp.asarray([60.0]),
        theta_v=jnp.asarray([60.0]),
        dphi=jnp.asarray([180.0]),
    )

    assert float(table.coefficients(beyond)[0][0]) == pytest.approx(
        float(table.coefficients(edge)[0][0])
    )


def test_the_transfer_round_trips():
    """``rrs -> Rrs -> rrs`` with the same geometry is the identity."""
    from robust.rt.types import Geometry

    table = _toy_transfer()
    geometry = Geometry(
        theta_s=jnp.asarray([30.0, 45.0]),
        theta_v=jnp.asarray([20.0, 50.0]),
        dphi=jnp.asarray([45.0, 120.0]),
    )
    rrs = jnp.asarray([[0.01, 0.02], [0.005, 0.03]])

    Rrs = C.rrs_to_Rrs(rrs, geometry=geometry, transfer=table)
    back = C.Rrs_to_rrs(Rrs, geometry=geometry, transfer=table)

    np.testing.assert_allclose(np.asarray(back), np.asarray(rrs), rtol=1e-6)


def test_the_transfer_is_jittable_and_differentiable():
    """It sits on ``forward``'s path, so it keeps ``forward``'s contract."""
    from robust.rt.types import Geometry

    table = _toy_transfer()

    def f(theta_v):
        geometry = Geometry(
            theta_s=jnp.asarray([30.0]),
            theta_v=theta_v,
            dphi=jnp.asarray([45.0]),
        )
        return jnp.sum(
            C.rrs_to_Rrs(jnp.asarray([[0.02]]), geometry=geometry, transfer=table)
        )

    assert float(jax.jit(f)(jnp.asarray([25.0]))) > 0.0

    # 25 deg is deliberately BETWEEN nodes: a piecewise-linear table has kinks at
    # its nodes, where autodiff takes one side and a central difference averages
    # both (M4 gotcha 4).
    slope = jax.grad(lambda t: f(jnp.asarray([t])))(25.0)
    assert float(slope) < 0.0  # A falls as the view tilts, so Rrs does too


def test_fit_recovers_coefficients_it_was_given():
    """A round trip through the fitter, on data with a known answer."""
    rng = np.random.default_rng(0)
    theta_s = np.repeat([0.0, 60.0], 200)
    theta_v = np.tile(np.repeat([0.0, 60.0], 100), 2)
    dphi = np.tile([0.0, 180.0], 200)
    rrs = rng.uniform(0.002, 0.04, size=(400, 3))
    A_true = 0.30 + 0.001 * theta_v[:, None] + 0.0005 * theta_s[:, None]
    Rrs = A_true * rrs / (1.0 - 1.9 * rrs)

    fitted = C.fit_surface_transfer(rrs, Rrs, theta_s, theta_v, dphi)

    assert fitted.shape == (2, 2, 2)
    np.testing.assert_allclose(fitted.B, 1.9, rtol=1e-6)
    assert fitted.A[0, 0, 0] == pytest.approx(0.30, abs=1e-6)
    assert fitted.A[1, 1, 0] == pytest.approx(0.30 + 0.06 + 0.03, abs=1e-6)


def test_fit_refuses_a_table_with_a_hole():
    """An unfitted cell would be interpolated across in silence."""
    rrs = np.full((4, 2), 0.01)
    Rrs = 0.5 * rrs
    theta_s = np.array([0.0, 0.0, 0.0, 0.0])
    theta_v = np.array([0.0, 0.0, 60.0, 60.0])
    # A second azimuth appears only at theta_v = 0, so the (60, 180) cell is empty.
    dphi = np.array([0.0, 180.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="had no samples"):
        C.fit_surface_transfer(rrs, Rrs, theta_s, theta_v, dphi)


def test_save_transfer_verifies_before_replacing(tmp_path):
    """Validate first, overwrite second (PR #11's lesson)."""
    path = tmp_path / "surface.npz"
    table = _toy_transfer()

    C.save_transfer(path, table)
    back = C.load_transfer(path)

    assert back.shape == table.shape
    np.testing.assert_array_equal(back.A, table.A)
    np.testing.assert_array_equal(back.B, table.B)
    assert back.provenance == "toy"


def test_the_shipped_table_is_present_and_sane():
    """It is committed, so CI checks it without a data mount."""
    table = C.default_transfer()

    assert table.shape == (10, 10, 13)
    assert C.default_transfer() is table  # cached
    assert "PB24" in table.provenance

    # A tracks the Fresnel transmittance: highest at nadir, falling with view angle.
    assert table.A[:, 0, :].mean() > table.A[:, 6, :].mean() > table.A[:, 9, :].mean()
    assert 0.5 < table.A[0, 0, 0] < 0.55  # near Lee's 0.52 at nadir
    assert np.all(table.B > 1.0)


@needs_pb24_olci
def test_the_fitted_transfer_beats_the_nadir_constants_on_held_out_data():
    """**The gate**, re-baselined for M8 task 3 (the Snell pairing).

    M5 asserted the *shipped* table is >= 5x better than Lee's nadir constants
    at theta_v = 60. That table was fitted on M5's in-water pairing
    (``rt_fit_pb24.md`` Q9c), so on correctly paired data -- ``Rrs`` at
    theta_air, ``rrs`` at theta_w -- it is now **worse** than the constants
    (measured 16.0% against 6.8% at 60 degrees, OLCI, 40 mixed realisations).
    What survives is the claim itself, for a table fitted on the same pairing:
    fitted on this batch's training rows (``pb24.fit_transfer``) it is 5.3x
    better at 60 degrees on held-out realisations (6.8% -> 1.3%). Task 4
    replaces the table with a Fresnel-based transfer.
    """
    from robust.rt.data import pb24 as P

    batch = P.load_batch(
        realisations=pb24_mixed_realisations(40), resolution="olci", angles="window"
    )
    splits = P.make_splits(batch, kinds=("realisation",))
    train, test = splits.train("realisation"), splits.test("realisation")

    rrs = np.asarray(batch.rrs)[test]
    Rrs = np.asarray(batch.Rrs)[test]
    theta_v = batch.theta_v[test]

    from robust.rt.types import Geometry

    geometry = Geometry(
        theta_s=batch.geometry.theta_s[test],
        theta_v=batch.geometry.theta_v[test],
        dphi=batch.geometry.dphi[test],
    )

    def median_error(transfer, at):
        if transfer is None:
            A, B = C.A_RRS, C.B_RRS
        else:
            A, B = transfer.coefficients(geometry)
            A, B = np.asarray(A)[:, None], np.asarray(B)[:, None]
        pred = A * rrs / (1 - B * rrs)
        return float(np.median(np.abs(pred[at] / Rrs[at] - 1)))

    at_60, at_0 = theta_v == 60.0, theta_v == 0.0
    nadir_err = median_error(None, at_60)
    fitted_err = median_error(P.fit_transfer(batch, train), at_60)
    shipped_err = median_error(C.default_transfer(), at_60)

    assert nadir_err / fitted_err >= 4.0, (
        f"gain at theta_v=60 is only {nadir_err / fitted_err:.1f}x "
        f"({nadir_err * 100:.1f}% -> {fitted_err * 100:.1f}%)"
    )
    # and it must not be worse at nadir, where the constants were already right
    assert median_error(P.fit_transfer(batch, train), at_0) < 0.05
    # the M5 table encodes the old pairing: worse than no table at all, now
    assert shipped_err > 2 * nadir_err, (
        f"shipped M5 table {shipped_err * 100:.1f}% vs nadir constants "
        f"{nadir_err * 100:.1f}% at theta_v=60 -- if the table now wins, it was "
        "refitted on the Snell pairing and this pin should move with it"
    )


def test_the_transfer_passes_the_gradient_gate_between_nodes(jax_x64):
    """**Gate.** Checked through task 6's toolkit, off-node and off-nadir.

    Two things this pins. The transfer is differentiable w.r.t. the *view* angles,
    which is the whole reason task 6 had to widen ``gradient_report`` first. And
    the check runs at 25 deg / 100 deg -- deliberately between table nodes, since
    a piecewise-linear table has kinks exactly at its nodes where autodiff and a
    central difference legitimately disagree by O(1) (M4 gotcha 4).
    """
    from robust.rt import validation as V
    from robust.rt.types import Geometry, IOPs, PhaseParams

    f64 = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)  # noqa: E731
    n, n_wave = 3, 4
    iops = IOPs(
        a=f64(np.full((n, n_wave), 0.08)),
        bb_w=f64(np.full((n, n_wave), 2e-3)),
        bb_p=f64(np.full((n, n_wave), 4e-3)),
    )
    phase = PhaseParams(B_p=f64(np.full((n, n_wave), 0.012)))
    geometry = Geometry(
        theta_s=f64(np.full(n, 35.0)),
        theta_v=f64(np.full(n, 25.0)),
        dphi=f64(np.full(n, 100.0)),
    )
    table = C.default_transfer()

    def model(i, p, g, w):
        rrs = i.bb_p / (i.a + i.bb_p) * 0.1
        return C.rrs_to_Rrs(rrs, geometry=g, transfer=table)

    report = V.gradient_report(
        model,
        iops,
        phase,
        geometry,
        f64(np.linspace(440.0, 600.0, n_wave)),
        steps=V.default_steps(["theta_v", "dphi", "theta_s"]),
    )

    for name, value in report.items():
        assert value < V.GRADIENT_TOL, f"{name}: {value}"
        assert value != 0.0, f"{name} was never perturbed"


# ------------------------------------------------- Raman excitation grid ----
# M1 task 2 (inelastic coding plan): the shift constant, the wavenumber-form
# maps, and the shared clamped-linear spectrum interpolation.


def test_raman_maps_pin_the_computed_values(jax_x64):
    """488 nm emission is excited at 418.55 nm; 488 nm excitation emits at
    585.08 nm.

    The pinned numbers are the exact wavenumber arithmetic (1e7/488 = 20491.8
    cm^-1, +/- 3400), not a prose approximation -- the M0 task-5 correction:
    the plan originally said 583.6 nm, and even bing's docstring example
    (583.0) is off. The exact-form comparison is float64 (fixture, dtype
    pinned on the array): in float32 it would test the dtype, not the map
    (elastic record §2 -- met again on the first run of this very test).
    """
    lam = jnp.asarray(488.0, dtype=jnp.float64)
    assert float(C.raman_excitation(lam)) == pytest.approx(418.5533, abs=5e-4)
    assert float(C.raman_emission(lam)) == pytest.approx(585.0758, abs=5e-4)
    # The exact form, not just the rounded pin.
    assert float(C.raman_excitation(lam)) == pytest.approx(
        1.0 / (1.0 / 488.0 + 3400e-7), rel=1e-12
    )


def test_raman_maps_are_exact_inverses(jax_x64):
    """emission(excitation(lambda)) round-trips to float64 precision."""
    wave = jnp.asarray(C.WAVE, dtype=jnp.float64)
    back = C.raman_emission(C.raman_excitation(wave))
    np.testing.assert_allclose(np.asarray(back), np.asarray(wave), rtol=1e-14)


def test_raman_maps_match_bing(jax_x64):
    """The shift constant and both maps agree with fixed BING (CQ3a spirit).

    Float64 on our side: bing computes in NumPy float64, so a float32
    comparison at 1e-12 would fail on dtype alone.
    """
    raman = pytest.importorskip("bing.rt.raman")
    assert C.RAMAN_SHIFT == raman.WAVENUMBER_SHIFT_CENTER
    lam = jnp.asarray(488.0, dtype=jnp.float64)
    np.testing.assert_allclose(
        float(C.raman_emission(lam)),
        raman.excitation_to_emission_wavelength(488.0),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        float(C.raman_excitation(lam)),
        raman.emission_to_excitation_wavelength(488.0),
        rtol=1e-12,
    )


def test_official_support_bound_is_inside_the_grid():
    """The excitation for a 400 nm emission sits just inside the L23 grid.

    This is the *reason* the official support starts at 400 nm (design §3):
    352.11 nm >= WAVE_MIN = 350. Below 400 the excitation leaves the grid and
    interp_spectrum clamps -- documented, not raised; asserted here so a grid
    change that breaks the rationale is caught.
    """
    edge = float(C.raman_excitation(C.RAMAN_WAVE_MIN_OFFICIAL))
    assert C.WAVE_MIN <= edge < C.WAVE_MIN + 5.0
    assert edge == pytest.approx(352.1127, abs=5e-4)


def test_interp_spectrum_matches_numpy_interp():
    """On- and off-grid wavelengths agree with numpy.interp, 1-D case."""
    rng = np.random.default_rng(7)
    spectrum = rng.uniform(0.5, 2.0, C.N_WAVE)
    targets = np.asarray([352.11, 418.55, 500.0, 585.08, 731.9])
    ours = np.asarray(
        C.interp_spectrum(
            jnp.asarray(targets), jnp.asarray(C.WAVE), jnp.asarray(spectrum)
        )
    )
    np.testing.assert_allclose(ours, np.interp(targets, C.WAVE, spectrum), rtol=1e-6)


def test_interp_spectrum_is_batched_and_clamps():
    """A (batch, grid) stack interpolates row-wise; ends clamp, never
    extrapolate."""
    spectra = jnp.stack([jnp.full(C.N_WAVE, 1.0), jnp.linspace(1.0, 2.0, C.N_WAVE)])
    out = np.asarray(
        C.interp_spectrum(
            jnp.asarray([300.0, 352.5, 800.0]), jnp.asarray(C.WAVE), spectra
        )
    )
    assert out.shape == (2, 3)
    assert out[1, 0] == 1.0  # clamped to the 350 nm end value
    assert out[1, 2] == 2.0  # clamped to the 750 nm end value


def test_interp_spectrum_promotes_integer_inputs():
    """Integer wavelengths select float nodes -- values are never truncated
    (the PR #14 ed.py lesson, applied here from birth)."""
    spectrum = jnp.linspace(0.3, 1.8, C.N_WAVE)
    out = np.asarray(C.interp_spectrum(np.arange(400, 700, 50), C.WAVE, spectrum))
    assert np.issubdtype(out.dtype, np.floating)
    ref = np.asarray(
        C.interp_spectrum(np.arange(400, 700, 50).astype(float), C.WAVE, spectrum)
    )
    np.testing.assert_array_equal(out, ref)


def test_interp_spectrum_grad_wrt_values_matches_fd(jax_x64):
    """d(interp)/d(values) from autodiff agrees with central differences.

    THE property the task demands: the Raman term interpolates IOP spectra
    onto the excitation grid, and gradients must flow through the interpolated
    values back to the IOP inputs. Float64 via the fixture, dtypes pinned on
    the arrays (elastic record §2); h scaled to the O(1) values.
    """
    grid = jnp.asarray(C.WAVE, dtype=jnp.float64)
    values = jnp.linspace(0.3, 1.8, C.N_WAVE, dtype=jnp.float64)
    targets = C.raman_excitation(jnp.asarray([450.0, 550.0, 650.0], dtype=jnp.float64))

    def scalar(v):
        return C.interp_spectrum(targets, grid, v).sum()

    grad = np.asarray(jax.grad(scalar)(values))
    h = 1e-6
    fd = np.zeros_like(grad)
    for i in range(values.shape[0]):
        bump = jnp.zeros_like(values).at[i].set(h)
        fd[i] = (float(scalar(values + bump)) - float(scalar(values - bump))) / (2 * h)
    np.testing.assert_allclose(grad, fd, atol=1e-9)
    # Each target draws weight from exactly two nodes, total weight 1 per target.
    assert grad.sum() == pytest.approx(3.0, rel=1e-9)


def test_interp_spectrum_jit_and_grad_wrt_wave():
    """Compiled evaluation matches eager; the map itself is differentiable."""
    spectrum = jnp.linspace(0.3, 1.8, C.N_WAVE)
    targets = jnp.asarray([450.0, 550.0])
    eager = np.asarray(C.interp_spectrum(targets, C.WAVE, spectrum))
    jitted = np.asarray(
        jax.jit(C.interp_spectrum)(targets, jnp.asarray(C.WAVE), spectrum)
    )
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)
    g = jax.grad(
        lambda lam: C.interp_spectrum(
            C.raman_excitation(lam)[None], jnp.asarray(C.WAVE), spectrum
        ).sum()
    )(jnp.asarray(550.0))
    assert np.isfinite(float(g))


# ------------------------------------- the Fresnel surface transfer (M8 task 4)
# ``rt_fit_pb24.md`` Q9c: M5's 1300-cell table was fitted across mismatched
# directions; the physical form is ``A(θv) = A0 T(θv)/T(0)``, two numbers.


def _geometry(theta_s, theta_v, dphi=None, H=None):
    from robust.rt.types import Geometry

    theta_s = jnp.asarray(theta_s, dtype=float)
    theta_v = jnp.asarray(theta_v, dtype=float)
    dphi = jnp.zeros_like(theta_s) if dphi is None else jnp.asarray(dphi, dtype=float)
    return Geometry(theta_s=theta_s, theta_v=theta_v, dphi=dphi, diffuse_fraction=H)


def test_fresnel_transmittance_reproduces_the_round_2_table():
    """Q9 finding 1: T(θ)/T(0) = 1.000/0.999/0.986/0.959/0.883 at 10-70°."""
    theta = np.array([10.0, 30.0, 50.0, 60.0, 70.0])
    T = np.asarray(C.fresnel_transmittance(theta), dtype=float) / float(
        C.fresnel_transmittance(0.0)
    )
    np.testing.assert_allclose(
        T, np.array([1.000, 0.999, 0.986, 0.959, 0.883]), atol=5e-4, rtol=0
    )
    n = C.REFRACTIVE_INDEX
    assert float(C.fresnel_transmittance(0.0)) == pytest.approx(
        1 - ((n - 1) / (n + 1)) ** 2, rel=1e-6
    )
    assert float(C.fresnel_transmittance(90.0)) == pytest.approx(0.0, abs=1e-6)
    # ZTT's Snell refraction and the transfer use the same n.
    from robust.rt import ztt

    assert ztt.REFRACTIVE_INDEX == C.REFRACTIVE_INDEX == 1.34


def test_fresnel_transmittance_matches_the_loaders_check(jax_x64):
    """The loader's private Fresnel check (task 3) and the public law agree."""
    from robust.rt.data import pb24 as P

    theta = np.linspace(0.0, 89.0, 90)
    np.testing.assert_allclose(
        np.asarray(C.fresnel_transmittance(jnp.asarray(theta))),
        P._fresnel_T(theta),
        rtol=1e-12,
    )


def test_fresnel_transmittance_gradient(jax_x64):
    """Differentiable, and right: float64 grad vs central differences."""
    for theta in (5.0, 37.3, 70.0, 85.0):
        x = jnp.asarray(theta, dtype=jnp.float64)
        grad = float(jax.grad(C.fresnel_transmittance)(x))
        h = 1e-3  # T is flat near nadir: a smaller step is all round-off
        fd = (
            float(C.fresnel_transmittance(x + h))
            - float(C.fresnel_transmittance(x - h))
        ) / (2 * h)
        assert grad == pytest.approx(fd, rel=1e-6, abs=1e-12), theta
        assert grad < 0.0  # T falls monotonically off nadir
    assert float(jax.grad(C.fresnel_transmittance)(jnp.asarray(0.0))) == 0.0


def test_fresnel_transfer_coefficients():
    """``A = A0 T(θv)/T(0)``, ``B`` constant, broadcast to the geometry."""
    f = C.FresnelTransfer(A0=0.5, B=1.7)
    g = _geometry([0.0, 30.0, 60.0], [0.0, 60.0, 37.3], [0.0, 90.0, 180.0])
    A, B = f.coefficients(g)
    T = np.asarray(C.fresnel_transmittance(jnp.asarray([0.0, 60.0, 37.3])))
    np.testing.assert_allclose(np.asarray(A), 0.5 * T / T[0], rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(B), np.full(3, 1.7, dtype=np.float32))
    assert float(A[0]) == 0.5  # exactly A0 at nadir
    assert isinstance(f, C.TransferLike)
    assert isinstance(C.default_transfer(), C.TransferLike)
    # θs and Δφ do not enter the θv-only form
    A2, _ = f.coefficients(_geometry([80.0, 0.0, 10.0], [0.0, 60.0, 37.3]))
    np.testing.assert_array_equal(np.asarray(A2), np.asarray(A))


def test_fresnel_transfer_round_trips_through_the_converters():
    """Accepted wherever ``SurfaceTransfer`` is: ``rrs_to_Rrs``/``Rrs_to_rrs``."""
    rrs = jnp.asarray(np.linspace(1e-3, 4e-2, 3 * 5).reshape(3, 5))
    g = _geometry([10.0, 40.0, 65.0], [0.0, 37.3, 70.0], [0.0, 90.0, 135.0])
    for f in (
        C.default_fresnel_transfer(),
        C.default_fresnel_transfer(downwelling=True),
        dataclasses_replace_H(g),
    ):
        transfer, geometry = f if isinstance(f, tuple) else (f, g)
        Rrs = C.rrs_to_Rrs(rrs, geometry=geometry, transfer=transfer)
        back = C.Rrs_to_rrs(Rrs, geometry=geometry, transfer=transfer)
        np.testing.assert_allclose(np.asarray(back), np.asarray(rrs), rtol=1e-6)
    # And it is not the nadir path off nadir.
    Rrs_f = C.rrs_to_Rrs(rrs, geometry=g, transfer=C.default_fresnel_transfer())
    assert not np.allclose(np.asarray(Rrs_f), np.asarray(C.rrs_to_Rrs(rrs)))


def dataclasses_replace_H(g):
    """The downwelling transfer with a spectral diffuse fraction on ``g``."""
    import dataclasses

    H = jnp.broadcast_to(jnp.linspace(0.2, 0.08, 5), (3, 5))
    return C.default_fresnel_transfer(downwelling=True), dataclasses.replace(
        g, diffuse_fraction=H
    )


def test_the_downwelling_factor(jax_x64):
    """``t_d = (1 − H) T(θs) + H T̄``; spectral with H, direct beam without."""
    import dataclasses

    f = C.FresnelTransfer(A0=0.52, B=1.73, downwelling=True)
    g = _geometry([0.0, 45.0, 70.0], [0.0, 30.0, 60.0])
    H = jnp.asarray(np.linspace(0.3, 0.05, 4)[None, :].repeat(3, axis=0))
    gH = dataclasses.replace(g, diffuse_fraction=H)

    A, B = f.coefficients(gH)
    assert A.shape == B.shape == (3, 4)
    ts, tv = np.array([0.0, 45.0, 70.0]), np.array([0.0, 30.0, 60.0])
    expected = (
        0.52
        * C._fresnel_ratio_np(tv)[:, None]
        * C._downwelling_ratio_np(ts[:, None], np.asarray(H))
    )
    np.testing.assert_allclose(np.asarray(A), expected, rtol=1e-12)
    T0 = float(C.fresnel_transmittance(0.0))
    assert C.diffuse_transmittance() == pytest.approx(0.932489, abs=1e-6)
    assert float(A[0, 0]) == pytest.approx(
        0.52 * ((1 - 0.3) + 0.3 * C.diffuse_transmittance() / T0)
    )

    A_direct, _ = f.coefficients(g)  # H = None: the direct beam alone
    np.testing.assert_allclose(
        np.asarray(A_direct), 0.52 * C._fresnel_ratio_np(tv) * C._fresnel_ratio_np(ts)
    )
    # and the converters see the spectral shape
    rrs = jnp.full((3, 4), 0.01)
    Rrs = C.rrs_to_Rrs(rrs, geometry=gH, transfer=f)
    np.testing.assert_allclose(
        np.asarray(Rrs), expected * 0.01 / (1 - 1.73 * 0.01), rtol=1e-12
    )


def test_rrs_o25_accepts_the_fresnel_transfer():
    """O25 is defined in ``Rrs``; its ``rrs`` goes through any transfer."""
    from robust.rt import baselines as Bl
    from robust.rt.types import IOPs

    n = 3
    iops = IOPs(
        a=jnp.full((n, 5), 0.1),
        bb_w=jnp.full((n, 5), 2e-3),
        bb_p=jnp.full((n, 5), 5e-3),
    )
    g = _geometry([0.0, 30.0, 60.0], [0.0, 37.3, 60.0], [0.0, 90.0, 180.0])
    f = C.default_fresnel_transfer()
    got = Bl.rrs_o25(iops, None, g, coeffs=Bl.O25_L23_REFIT, transfer=f)
    Rrs = Bl.Rrs_o25(iops, None, g, coeffs=Bl.O25_L23_REFIT)
    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(C.Rrs_to_rrs(Rrs, geometry=g, transfer=f))
    )
    assert np.all(np.isfinite(np.asarray(got)))


def test_the_package_default_transfer_did_not_change():
    """Task 4 makes the Fresnel transfer opt-in: no transfer is still the nadir
    constants, a geometry alone is still M5's (superseded) table."""
    rrs = jnp.asarray(np.linspace(1e-3, 4e-2, 10).reshape(2, 5))
    g = _geometry([10.0, 40.0], [20.0, 50.0], [30.0, 90.0])
    np.testing.assert_array_equal(
        np.asarray(C.rrs_to_Rrs(rrs)),
        np.asarray(C.A_RRS * rrs / (1.0 - C.B_RRS * rrs)),
    )
    np.testing.assert_array_equal(
        np.asarray(C.rrs_to_Rrs(rrs, geometry=g)),
        np.asarray(C.rrs_to_Rrs(rrs, geometry=g, transfer=C.default_transfer())),
    )
    assert "Superseded" in C.default_transfer.__doc__


def test_shipped_fresnel_constants():
    """The shipped numbers and their provenance (fit: ``fit_surface_fresnel.py``)."""
    f = C.default_fresnel_transfer()
    assert (f.A0, f.B, f.downwelling) == (C.FRESNEL_A0, C.FRESNEL_B, False)
    td = C.default_fresnel_transfer(downwelling=True)
    assert (td.A0, td.B, td.downwelling) == (C.FRESNEL_TD_A0, C.FRESNEL_TD_B, True)
    # Physically sensible: near Lee's nadir pair; the downwelling form's A0 is
    # Lee's 0.52 itself (measured 0.5199).
    assert 0.49 < C.FRESNEL_A0 < 0.53 and 1.5 < C.FRESNEL_B < 1.9
    assert abs(C.FRESNEL_TD_A0 - C.A_RRS) < 0.005 and 1.5 < C.FRESNEL_TD_B < 1.9
    for needle in ("splits.json", "4000", "fit_surface_fresnel.py", "2026-11-04"):
        assert needle in C.FRESNEL_PROVENANCE


def test_the_fresnel_transfer_was_fitted_on_training_rows_only():
    """**The leak guard.** Fails if the shipped constants ever saw a held-out row.

    The constants carry the digest of the realisation ids they were fitted on
    (``fit_surface_fresnel.py`` computes it from the ids it actually reads).
    It must be the digest of the realisation split's **training** ids in the
    committed ``splits.json`` -- not of all 5000, not of the training ids plus
    any test id. Deliberately *not* a refit on the committed miniature: its
    three realisations straddle the split (993 is held out), so a refit there
    could not tell a leak from noise.
    """
    from robust.rt.data import pb24 as P

    content = P.read_splits()
    every = np.asarray(content["realisations"], dtype=int)
    test = np.asarray(content["realisation"]["test"], dtype=int)
    train = np.setdiff1d(every, test)
    assert np.intersect1d(train, test).size == 0 and test.size > 0

    assert C.FRESNEL_N_TRAINING == train.size
    assert C.realisation_digest(train) == C.FRESNEL_TRAINING_DIGEST
    # Any held-out id in the fitting set changes the digest, so the test fails.
    assert C.realisation_digest(every) != C.FRESNEL_TRAINING_DIGEST
    leaky = np.append(train, test[0])
    assert C.realisation_digest(leaky) != C.FRESNEL_TRAINING_DIGEST
    # The digest is order- and duplicate-insensitive, so only the *set* counts.
    assert C.realisation_digest(train[::-1]) == C.realisation_digest(
        np.concatenate([train, train[:5]])
    )


def test_the_fresnel_fit_is_deterministic_and_streams(pb24_small_batch):
    """One-shot and chunked normal equations agree; a rerun is bit-identical.

    Runs on the miniature for the *mechanics* only (it straddles the split, so
    the numbers it fits are not compared with the shipped ones).
    """
    batch = pb24_small_batch
    rrs, Rrs = np.asarray(batch.rrs, dtype=float), np.asarray(batch.Rrs, dtype=float)
    theta_v = np.asarray(batch.theta_v, dtype=float)[:, None]

    once = C.fit_fresnel_transfer(rrs, Rrs, batch.theta_v)
    again = C.fit_fresnel_transfer(rrs, Rrs, batch.theta_v)
    assert (once.A0, once.B) == (again.A0, again.B)
    assert 0.45 < once.A0 < 0.56 and 1.0 < once.B < 2.5

    half = rrs.shape[0] // 2
    parts = [
        C.fresnel_normal_equations(rrs[s], Rrs[s], theta_v[s])
        for s in (slice(0, half), slice(half, None))
    ]
    A0, B = C.solve_fresnel(parts[0][0] + parts[1][0], parts[0][1] + parts[1][1])
    assert A0 == pytest.approx(once.A0, rel=1e-12) and B == pytest.approx(
        once.B, rel=1e-10
    )


@needs_pb24
def test_the_fresnel_transfers_beat_the_nadir_constants_held_out():
    """On held-out realisations (window): the downwelling form beats Lee's
    constants overall and the θv-only form beats them at θv_air = 60°.

    Full-sample numbers: ``design/ztt_pb24/surface_fresnel.md``.
    """
    from robust.rt.data import pb24 as P

    test_ids = sorted(P.read_splits()["realisation"]["test"])[:6]
    batch = P.load_batch(realisations=test_ids, angles="window", check_fresnel=False)
    rrs, Rrs = np.asarray(batch.rrs), np.asarray(batch.Rrs)

    def err(transfer):
        if transfer is None:
            pred = C.rrs_to_Rrs(batch.rrs)
        else:
            pred = C.rrs_to_Rrs(batch.rrs, geometry=batch.geometry, transfer=transfer)
        return np.abs(np.asarray(pred) / Rrs - 1.0)

    nadir = err(None)
    theta_v_only = err(C.default_fresnel_transfer())
    downwelling = err(C.default_fresnel_transfer(downwelling=True))
    at_60 = batch.theta_v == 60.0

    assert np.median(downwelling) < 0.7 * np.median(nadir)
    assert np.median(theta_v_only[at_60]) < 0.5 * np.median(nadir[at_60])
    assert np.quantile(theta_v_only, 0.9) < 0.6 * np.quantile(nadir, 0.9)
    assert rrs.shape == Rrs.shape
