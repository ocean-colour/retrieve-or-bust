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
from robust.tests.conftest import needs_l23, needs_pb24

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


@needs_pb24
def test_the_fitted_transfer_beats_the_nadir_constants_on_held_out_data():
    """**The gate**: >=5x better at theta_v = 60, on realisations never fitted."""
    from robust.rt.data import pb24 as P

    batch = P.load_batch(realisations=40, angles="window")
    splits = P.make_splits(batch, kinds=("realisation",))
    test = splits.test("realisation")

    rrs = np.asarray(batch.rrs)[test]
    Rrs = np.asarray(batch.Rrs)[test]
    theta_v = batch.theta_v[test]

    from robust.rt.types import Geometry

    geometry = Geometry(
        theta_s=batch.geometry.theta_s[test],
        theta_v=batch.geometry.theta_v[test],
        dphi=batch.geometry.dphi[test],
    )
    A, B = C.default_transfer().coefficients(geometry)
    A = np.asarray(A)[:, None]
    B = np.asarray(B)[:, None]

    at_60 = theta_v == 60.0
    nadir_err = np.median(
        np.abs((C.A_RRS * rrs / (1 - C.B_RRS * rrs))[at_60] / Rrs[at_60] - 1)
    )
    fitted_err = np.median(np.abs((A * rrs / (1 - B * rrs))[at_60] / Rrs[at_60] - 1))

    assert nadir_err / fitted_err >= 5.0, (
        f"gain at theta_v=60 is only {nadir_err / fitted_err:.1f}x "
        f"({nadir_err * 100:.1f}% -> {fitted_err * 100:.1f}%)"
    )
    # and it must not be worse at nadir, where the constants were already right
    at_0 = theta_v == 0.0
    fitted_0 = np.median(np.abs((A * rrs / (1 - B * rrs))[at_0] / Rrs[at_0] - 1))
    assert fitted_0 < 0.05


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
