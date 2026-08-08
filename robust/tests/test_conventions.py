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
from robust.tests.conftest import needs_l23

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
