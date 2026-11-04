"""
Tests for :mod:`robust.rt.data.pb24` (M5 task 4; rebuilt for M8 task 3).

What is being defended, now that a sample is an **above-water** geometry
(Q9 of ``rt_fit_pb24.md``):

**The loader runs on real numbers without the dataset.** The committed cache
miniature (``pb24_cache_small.zip``: realisations 1, 993, 2500; 4-D fields at
every 10th band) is unpacked per session and fed to the real loader through
``cache_dir=``, so ``load_batch`` genuinely executes in CI.

**The pairing is Snell's.** ``Rrs`` is read at its above-water node and ``rrs``
/ ``Q`` interpolated linearly in theta to ``theta_w``; the interpolation is
recomputed here by hand from the two bracketing in-water nodes, and the Fresnel
check (``Rrs(theta_air) / rrs(theta_w)`` normalised at nadir against
``T(theta_air) / T(0)``) holds within 5% for ``theta_air <= 70`` on every
realisation loaded.

**Golden values against raw reads** through ``ocpy.hydrolight.pb24`` (needs
``$OS_COLOR``), for both resolutions.

**Nothing is dropped silently.** Window, shell and in-water-only counts are
pinned in air angles; the zero-``rrs`` filter has work to do only on the
in-water-only set (the 87.5-degree node), and its count is asserted there.

**The splits are saved, not drawn.** ``design/ztt_pb24/splits.json`` is
disjoint and non-empty on each side, identical on reload, rebuilt
byte-identically from the cache, and a batch with a realisation the file does
not cover raises.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt.data import pb24 as P
from robust.rt.data import pb24_cache as PC
from robust.tests.conftest import (
    PB24_CACHE_FIXTURE_BANDS,
    PB24_FIXTURE_REALISATIONS,
    needs_pb24,
    needs_pb24_olci,
)

#: Geometry counts on PB24's grid: 10 x 10 x 13, in either medium.
N_GEOMETRY = 1300

#: In the window theta_s and theta_v_air each keep 8 of 10 nodes (0-70).
N_WINDOW = 8 * 8 * 13

#: The shell: the complement in air.
N_SHELL = N_GEOMETRY - N_WINDOW

#: In-water-only: every theta_s, the five in-water nodes 50-87.5.
N_IN_WATER_ONLY = 10 * 5 * 13

#: Bands in the cache miniature's 4-D fields (every 10th, 350-800 nm).
N_FIXTURE_BANDS = len(PB24_CACHE_FIXTURE_BANDS)

#: The realisation whose rrs underflows to zero at the 87.5-degree in-water node
#: in the fixture (theta_s = 80; 730, 750 and 800 nm).
ZERO_REALISATION = 993


def _fresnel_T(theta_air):
    """Independent copy of the flat-surface transmittance (n = 1.34), for tests."""
    n = 1.34
    ti = np.radians(np.asarray(theta_air, dtype=float))
    tt = np.arcsin(np.sin(ti) / n)
    rs = (np.cos(ti) - n * np.cos(tt)) / (np.cos(ti) + n * np.cos(tt))
    rp = (n * np.cos(ti) - np.cos(tt)) / (n * np.cos(ti) + np.cos(tt))
    return 1.0 - 0.5 * (rs**2 + rp**2)


@pytest.fixture(scope="module")
def cache(pb24_cache_dir):
    """The unpacked cache miniature, opened."""
    return PC.open_cache(pb24_cache_dir)


@pytest.fixture(scope="module")
def all_batch(pb24_cache_dir):
    """All 1300 air geometries of the three fixture realisations, unfiltered."""
    return P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS,
        angles="all",
        drop_zero_rrs=False,
        extras=P.EXTRA_FIELDS,
        cache_dir=pb24_cache_dir,
    )


@pytest.fixture(scope="module")
def in_water_batch(pb24_cache_dir):
    """The in-water-only set, unfiltered -- where the zero ``rrs`` values live."""
    return P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS,
        angles="in_water_only",
        drop_zero_rrs=False,
        cache_dir=pb24_cache_dir,
    )


def _flat(theta_s_i: int, theta_i: int, phi_i: int) -> int:
    """Flattened geometry index, C-order (theta_s, theta, phi), of one realisation."""
    return (theta_s_i * 10 + theta_i) * 13 + phi_i


# ------------------------------------------------------------- geometry ----


def test_theta_air_to_water_is_snell():
    """sin(theta_air) = 1.34 sin(theta_w); the window edge is 44.53 degrees."""
    air = np.array([0.0, 30.0, 70.0, 87.5])
    water = P.theta_air_to_water(air)

    np.testing.assert_allclose(
        np.sin(np.radians(air)), 1.34 * np.sin(np.radians(water))
    )
    assert water[0] == 0.0
    assert water[2] == pytest.approx(44.528, abs=1e-3)
    assert P.CRITICAL_ANGLE == pytest.approx(48.27, abs=1e-2)
    assert water.max() < P.CRITICAL_ANGLE < P.IN_WATER_ONLY_MIN


def test_interp_in_water_is_linear_and_exact_at_nodes():
    """The function the loader uses, on a field whose answer is known."""
    nodes = np.array([0.0, 10.0, 20.0, 30.0])
    field = np.zeros((2, 4, 3, 5))
    field[:, :, :, :] = (nodes**2)[None, :, None, None]  # quadratic in theta

    got = P.interp_in_water(field, nodes, [0.0, 10.0, 12.5, 30.0])

    assert got.shape == (2, 4, 3, 5)
    np.testing.assert_array_equal(got[:, 0], 0.0)
    np.testing.assert_array_equal(got[:, 1], 100.0)
    # linear between 100 (10 deg) and 400 (20 deg), a quarter of the way
    np.testing.assert_allclose(got[:, 2], 175.0)
    np.testing.assert_array_equal(got[:, 3], 900.0)

    with pytest.raises(ValueError, match="within"):
        P.interp_in_water(field, nodes, [31.0])
    with pytest.raises(ValueError, match="ascending"):
        P.interp_in_water(field, nodes[::-1], [5.0])


# ---------------------------------------------------------------- counts ----


def test_window_load_has_the_expected_shape(pb24_small_batch):
    """3 realisations x 832 above-water geometries, 46 fixture bands."""
    batch = pb24_small_batch

    assert batch.n_wave == N_FIXTURE_BANDS
    assert batch.n_sample == len(PB24_FIXTURE_REALISATIONS) * N_WINDOW
    assert batch.iops.a.shape == (batch.n_sample, N_FIXTURE_BANDS)
    assert batch.rrs.shape == batch.Rrs.shape == batch.iops.a.shape
    np.testing.assert_array_equal(
        np.asarray(batch.wave), P.HYP_WAVE[list(PB24_CACHE_FIXTURE_BANDS)]
    )
    assert batch.report.resolution == "hyp"


def test_window_shell_and_in_water_only_counts(
    pb24_small_batch, all_batch, in_water_batch, pb24_cache_dir
):
    """**Gate.** The counts, in air angles: 832 + 468 = 1300, and 650 in water."""
    shell = P.load_batch(realisations=(1,), angles="shell", cache_dir=pb24_cache_dir)

    assert pb24_small_batch.report.n_geometry == N_WINDOW == 832
    assert shell.report.n_geometry == N_SHELL == 468
    assert all_batch.report.n_geometry == N_GEOMETRY
    assert all_batch.report.n_dropped_angle == 0
    assert in_water_batch.report.n_geometry == N_IN_WATER_ONLY == 650
    assert in_water_batch.report.n_dropped_angle == N_GEOMETRY - N_IN_WATER_ONLY
    for report in (pb24_small_batch.report, shell.report, in_water_batch.report):
        assert report.n_geometry + report.n_dropped_angle == report.n_geometry_available


def test_the_window_is_in_air_angles(pb24_small_batch):
    """Both zeniths reach 70 degrees in air; the in-water view stops at 44.53."""
    batch = pb24_small_batch

    assert batch.theta_s.max() == P.ANGLE_WINDOW_MAX == 70.0
    assert batch.theta_v.max() == P.ANGLE_WINDOW_MAX
    assert sorted(set(batch.theta_v.tolist())) == [0, 10, 20, 30, 40, 50, 60, 70]
    assert len(set(batch.dphi.tolist())) == 13
    np.testing.assert_allclose(batch.theta_w, P.theta_air_to_water(batch.theta_v))
    assert batch.theta_w.max() == pytest.approx(44.528, abs=1e-3)


def test_the_shell_is_the_extrapolation_set(all_batch):
    """Every shell sample has an air angle past 70; its theta_w stays below 48.3."""
    shell = (all_batch.theta_s > P.ANGLE_WINDOW_MAX) | (
        all_batch.theta_v > P.ANGLE_WINDOW_MAX
    )

    assert int(shell.sum()) == len(PB24_FIXTURE_REALISATIONS) * N_SHELL
    assert all_batch.theta_s.max() == pytest.approx(87.75)
    assert all_batch.theta_v.max() == pytest.approx(87.5)
    assert all_batch.theta_w.max() < P.CRITICAL_ANGLE


def test_in_water_only_holds_in_water_angles_and_no_Rrs(in_water_batch):
    """theta_v is the in-water node there, and Rrs is absent (all NaN)."""
    batch = in_water_batch

    assert sorted(set(batch.theta_v.tolist())) == [50.0, 60.0, 70.0, 80.0, 87.5]
    np.testing.assert_array_equal(batch.theta_w, batch.theta_v)
    assert batch.in_water_only
    assert bool(np.isnan(np.asarray(batch.Rrs)).all())
    assert len(set(batch.theta_s.tolist())) == 10


# ------------------------------------------------- nothing dropped silently -


def test_the_stride_reports_what_it_dropped(pb24_cache_dir):
    """A subsample is visible at the call site, not implied."""
    batch = P.load_batch(realisations=(1,), geometry_stride=3, cache_dir=pb24_cache_dir)
    report = batch.report

    kept = len(range(0, N_WINDOW, 3))
    assert report.n_geometry == kept
    assert report.n_dropped_stride == N_WINDOW - kept
    assert batch.n_sample == kept
    assert "stride=3" in report.summary()


def test_the_air_sets_hold_no_zero_rrs(all_batch):
    """**Re-measured (M8).** No above-water sample interpolates from 87.5 in water."""
    assert all_batch.report.n_zero_bands == 0
    assert not bool((np.asarray(all_batch.rrs) <= 0.0).any())


def test_zero_rrs_lives_at_the_87_5_in_water_node(in_water_batch, pb24_cache_dir):
    """The filter's real work is on the in-water-only set; the count is asserted."""
    batch = in_water_batch
    zero = np.asarray(batch.rrs) <= 0.0
    rows = zero.any(axis=-1)

    assert batch.report.n_zero_bands == 3
    assert batch.report.n_dropped_zero_rrs == 0  # not asked to drop
    assert set(batch.realisation[rows].tolist()) == {ZERO_REALISATION}
    assert set(batch.theta_v[rows].tolist()) == {87.5}
    assert set(batch.theta_s[rows].tolist()) == {80.0}
    np.testing.assert_array_equal(
        np.asarray(batch.wave)[zero.any(axis=0)], [730.0, 750.0, 800.0]
    )

    filtered = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS,
        angles="in_water_only",
        cache_dir=pb24_cache_dir,
    )
    assert filtered.report.n_dropped_zero_rrs == int(rows.sum())
    assert filtered.n_sample == batch.n_sample - int(rows.sum())
    assert not bool((np.asarray(filtered.rrs) <= 0.0).any())
    # the cost of dropping whole spectra is visible: good bands go with bad ones
    lost_good = filtered.report.n_dropped_zero_rrs * N_FIXTURE_BANDS - 3
    assert lost_good > 0


# ------------------------------------------------------------ stride aliasing -


def test_a_stride_that_aliases_the_azimuth_axis_warns(pb24_cache_dir):
    """13 azimuths innermost: stride 13 keeps one azimuth and no BRDF at all."""
    with pytest.warns(UserWarning, match="loses whole angle values"):
        batch = P.load_batch(
            realisations=(1,), geometry_stride=13, cache_dir=pb24_cache_dir
        )

    assert batch.report.aliased_axes == ("dphi",)
    assert len(set(batch.dphi.tolist())) == 1
    assert "ALIASED" in batch.report.summary()


def test_a_coprime_stride_keeps_every_angle(pb24_cache_dir):
    """The healthy case, so the warning above is not merely always-on."""
    batch = P.load_batch(realisations=(1,), geometry_stride=7, cache_dir=pb24_cache_dir)

    assert batch.report.aliased_axes == ()
    assert batch.report.coverage == {
        "theta_s": (8, 8),
        "theta_v": (8, 8),
        "dphi": (13, 13),
    }


# ------------------------------------------- the pairing, against the cache -


def test_Rrs_is_read_at_its_above_water_node(all_batch, cache):
    """``Rrs`` in the batch is the cache's float32 value at the air node, exactly."""
    for index in PB24_FIXTURE_REALISATIONS:
        rows = np.flatnonzero(all_batch.realisation == index)
        truth = cache.four_d("Rrs", index)  # (theta_s, theta, phi, band)
        for i_ts, i_t, i_ph in ((0, 0, 0), (6, 4, 9), (9, 9, 12)):
            got = np.asarray(all_batch.Rrs[rows[_flat(i_ts, i_t, i_ph)]])
            np.testing.assert_array_equal(got, truth[i_ts, i_t, i_ph])
            assert all_batch.theta_v[rows[_flat(i_ts, i_t, i_ph)]] == float(
                cache.grids["theta"][i_t]
            )


def test_interpolated_rrs_recomputed_by_hand(all_batch, cache):
    """**Gate.** ``rrs`` and ``Q`` at theta_w, from the two bracketing in-water nodes.

    Realisation 2500, theta_s = 60, theta_v_air = 60 (theta_w = 40.262, between
    the in-water nodes 40 and 50), phi = 135, 560 nm.
    """
    index, i_ts, i_t, i_ph = 2500, 6, 6, 9
    band = int(np.flatnonzero(np.asarray(all_batch.wave) == 560.0)[0])
    theta_w = float(np.degrees(np.arcsin(np.sin(np.radians(60.0)) / 1.34)))
    t = (theta_w - 40.0) / (50.0 - 40.0)
    row = np.flatnonzero(all_batch.realisation == index)[_flat(i_ts, i_t, i_ph)]

    assert all_batch.theta_w[row] == pytest.approx(theta_w)
    assert theta_w == pytest.approx(40.262, abs=1e-3)
    for name, got in (("rrs", all_batch.rrs), ("Q", all_batch.aops["Q"])):
        lo = float(cache.four_d_at(name, index, 60.0, 40.0, 135.0, 560.0))
        hi = float(cache.four_d_at(name, index, 60.0, 50.0, 135.0, 560.0))
        by_hand = (1.0 - t) * lo + t * hi

        assert float(got[row, band]) == pytest.approx(by_hand, rel=1e-6), name
        assert min(lo, hi) <= float(got[row, band]) <= max(lo, hi)


def test_nadir_rrs_is_the_node_value_exactly(all_batch, cache):
    """At theta_v = 0 there is nothing to interpolate."""
    rows = np.flatnonzero(all_batch.realisation == 1)
    for i_ts, i_ph in ((0, 0), (5, 7)):
        got = np.asarray(all_batch.rrs[rows[_flat(i_ts, 0, i_ph)]])
        np.testing.assert_array_equal(got, cache.four_d("rrs", 1)[i_ts, 0, i_ph])


def test_in_water_only_is_not_interpolated(in_water_batch, cache):
    """The in-water nodes are read as tabulated."""
    rows = np.flatnonzero(in_water_batch.realisation == 2500)
    # in-water-only geometry list is (theta_s, theta in 50..87.5, phi)
    k = (3 * 5 + 2) * 13 + 4  # theta_s index 3, node 70 (third of five), phi index 4
    assert in_water_batch.theta_v[rows[k]] == 70.0
    np.testing.assert_array_equal(
        np.asarray(in_water_batch.rrs[rows[k]]), cache.four_d("rrs", 2500)[3, 7, 4]
    )


# --------------------------------------------------------- the Fresnel check -


def test_the_fresnel_check_holds_in_the_window(all_batch):
    """**Gate.** Within 5% of T(theta)/T(0) for theta_air <= 70, every realisation.

    Measured on the fixture (3 realisations, 46 bands, every theta_s and dphi):
    the median deviation falls from +0.2% at 10 degrees to -3.4% at 70, the
    worst is -4.60% at 70. At 80 and 87.5 the flat-surface Fresnel law fails
    (+5.5% to +202%): the 5 m/s surface is not flat there and theta_w sits
    between the 40 and 50 in-water nodes.
    """
    fresnel = all_batch.report.fresnel

    assert fresnel.n_realisation == len(PB24_FIXTURE_REALISATIONS)
    assert fresnel.theta_v_air == (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 87.5)
    assert fresnel.n_excluded == 0
    assert set(fresnel.n_value) == {3 * 10 * 13 * N_FIXTURE_BANDS}
    assert fresnel.within(), fresnel.table()
    assert fresnel.n_beyond == 0 and fresnel.realisations_beyond == ()
    assert fresnel.worst() == pytest.approx(0.0460, abs=5e-4)
    # every sun angle, not just the window's
    assert max(fresnel.max_abs_by_theta_s.values()) < P.FRESNEL_TOLERANCE
    # and beyond the window the law genuinely fails -- the check is not vacuous
    q80 = fresnel.deviation[fresnel.theta_v_air.index(80.0)]
    assert q80[0] > P.FRESNEL_TOLERANCE
    np.testing.assert_allclose(
        fresnel.expected[6], 0.8833, atol=1e-4
    )  # Round 2's table at 70 degrees


def test_the_fresnel_report_agrees_with_a_recomputation(all_batch):
    """The streamed extremes equal a direct NumPy computation on the batch."""
    batch = all_batch
    rrs, Rrs = np.asarray(batch.rrs, float), np.asarray(batch.Rrs, float)
    key = np.stack([batch.realisation, batch.theta_s, batch.dphi], axis=1)
    nadir = batch.theta_v == 0.0
    lookup = {
        tuple(k): i for i, k in zip(np.flatnonzero(nadir), key[nadir], strict=True)
    }
    ref = np.array([lookup[tuple(k)] for k in key])
    norm = (Rrs / rrs) / (Rrs[ref] / rrs[ref])
    dev = norm / (_fresnel_T(batch.theta_v) / _fresnel_T(0.0))[:, None] - 1.0

    report = batch.report.fresnel
    for t, mx in zip(report.theta_v_air, report.max_abs, strict=True):
        at = batch.theta_v == t
        assert np.abs(dev[at]).max() == pytest.approx(mx, abs=1e-6), t


def test_the_fresnel_check_can_be_skipped(pb24_cache_dir):
    batch = P.load_batch(
        realisations=(1,),
        geometry_stride=7,
        check_fresnel=False,
        cache_dir=pb24_cache_dir,
    )
    assert batch.report.fresnel is None


def test_interpolation_error_is_measured(pb24_cache_dir):
    """Linear against cubic, per view node; nadir is excluded (both exact)."""
    report = P.interpolation_error(PB24_FIXTURE_REALISATIONS, cache_dir=pb24_cache_dir)

    assert report.fields == ("rrs", "Q")
    assert report.theta_v_air[0] == 10.0 and report.theta_v_air[-1] == 87.5
    rrs = dict(zip(report.theta_v_air, report.error["rrs"], strict=True))
    # measured on the fixture: worst 2.56% at theta_v_air = 70 (theta_w 44.5,
    # between the 40 and 50 nodes), median under 0.4% everywhere
    assert all(row[2] < 0.004 for row in rrs.values())
    assert 0.01 < rrs[70.0][-1] < 0.05
    assert "theta_w" in report.table("rrs")


# ------------------------------------------------------------------ contents --


def test_iops_are_component_sums_and_carry_a_ph(all_batch, cache):
    """``a``, ``bb_p`` from components; ``bb_w`` is the file's; ``a_ph`` = ``aph``."""
    batch = all_batch
    bands = list(PB24_CACHE_FIXTURE_BANDS)
    row = np.flatnonzero(batch.realisation == 993)[0]
    t = {n: cache.table(n, 993)[bands].astype(float) for n in P._IOPS}

    np.testing.assert_allclose(
        np.asarray(batch.iops.a[row]),
        t["aw"] + t["aph"] + t["ag"] + t["aNAP"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(batch.iops.bb_p[row]), t["bbph"] + t["bbNAP"], rtol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(batch.iops.bb_w[row]), cache.table("bbw", 993)[bands]
    )
    np.testing.assert_array_equal(
        np.asarray(batch.iops.a_ph[row]), cache.table("aph", 993)[bands]
    )
    ours = np.asarray(C.bb_w(batch.wave))
    assert not np.allclose(ours, np.asarray(batch.iops.bb_w[0]), rtol=1e-3)


def test_B_p_varies_with_wavelength_and_equals_the_ff_mixture(all_batch):
    """Bulk B_p is the scattering-weighted mean of the two FF components."""
    phase = all_batch.phase_params
    B_p = np.asarray(phase.B_p, float)
    ff_B = np.asarray(phase.ff_B, float)
    ff_w = np.asarray(phase.ff_weight, float)

    assert ff_B.shape == ff_w.shape == B_p.shape + (2,)
    np.testing.assert_allclose(ff_w.sum(axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose((ff_w * ff_B).sum(axis=-1), B_p, rtol=2e-6)
    within = B_p.max(axis=-1) / B_p.min(axis=-1)
    assert within.max() > 1.001  # not flat in wavelength, unlike each component
    # each component ratio *is* flat within a realisation (PB24 design)
    assert (ff_B.max(axis=1) / ff_B.min(axis=1)).max() < 1.01
    # PB24's B_NAP ~ U(0.01, 0.02)
    assert 0.0099 < ff_B[..., 1].min() and ff_B[..., 1].max() < 0.0201


def test_diffuse_fraction_is_one_minus_esdir_at_each_samples_sun(all_batch):
    """``Geometry.diffuse_fraction = 1 - Esdir_Es_ratio`` gathered by theta_s."""
    batch = all_batch
    df = np.asarray(batch.geometry.diffuse_fraction, float)
    esdir = np.asarray(batch.aops["Esdir_Es_ratio"], float)

    assert df.shape == batch.rrs.shape
    np.testing.assert_allclose(df, 1.0 - esdir, atol=1e-6)
    first = batch.realisation == 1
    assert np.unique(np.round(df[first][:, 0], 7)).size == 10  # one per sun angle
    # the diffuse share grows as the sun goes down, and is bluer (Rayleigh)
    at_0 = df[first & (batch.theta_s == 0.0)][0]
    at_80 = df[first & (batch.theta_s == 80.0)][0]
    assert at_80.mean() > at_0.mean()
    assert at_0[0] > at_0[-1]


def test_extras_are_gathered_at_each_samples_solar_zenith(pb24_small_batch):
    """``mu_d`` is tabulated per zenith; every sample must get its own."""
    batch = pb24_small_batch

    assert set(batch.aops) == {"Q", "mu_d", "mu_u", "mu_tot"}
    for name, values in batch.aops.items():
        assert values.shape == batch.rrs.shape, name

    mu_d = np.asarray(batch.aops["mu_d"])
    first = batch.realisation == PB24_FIXTURE_REALISATIONS[0]
    distinct = np.unique(np.round(mu_d[first][:, 0], 6))
    assert distinct.size == 8

    at_0 = mu_d[first & (batch.theta_s == 0.0)][:, 0]
    at_70 = mu_d[first & (batch.theta_s == 70.0)][:, 0]
    assert at_0.mean() > at_70.mean()


def test_gershun_holds_in_the_surface_aops(all_batch):
    """``a / Knet = mu_tot``: a data property M5 found (median 0.9999), kept."""
    a = np.asarray(all_batch.iops.a, float)
    ratio = (
        a
        / np.asarray(all_batch.aops["Knet"], float)
        / np.asarray(all_batch.aops["mu_tot"], float)
    )

    assert np.median(ratio) == pytest.approx(1.0, abs=1e-3)
    assert np.quantile(np.abs(ratio - 1.0), 0.99) < 0.01


def test_extras_can_be_requested_and_are_validated(pb24_cache_dir):
    """Opt-in K's work; an unknown name fails rather than being ignored."""
    batch = P.load_batch(
        realisations=(1,),
        extras=("mu_d", "Kd"),
        geometry_stride=7,
        cache_dir=pb24_cache_dir,
    )

    assert set(batch.aops) == {"Q", "mu_d", "Kd"}

    with pytest.raises(ValueError, match="unknown extras"):
        P.load_batch(realisations=(1,), extras=("mu_x",), cache_dir=pb24_cache_dir)


def test_water_classes_are_opt_in(pb24_cache_dir):
    """The default attaches none, so a fixture load is environment-independent."""
    default = P.load_batch(
        realisations=(1,), geometry_stride=7, cache_dir=pb24_cache_dir
    )
    explicit = P.load_batch(
        realisations=(1,),
        geometry_stride=7,
        water_classes=np.full(P.N_REALISATION, 7),
        cache_dir=pb24_cache_dir,
    )

    assert default.water_class is None
    assert set(explicit.water_class.tolist()) == {7}

    with pytest.raises(ValueError, match="must be an array, None or 'auto'"):
        P.load_batch(realisations=(1,), water_classes="all", cache_dir=pb24_cache_dir)


# -------------------------------------------------------------- the API edges -


def test_select_subsets_every_leaf(all_batch):
    """Including the FF mixture, diffuse_fraction and theta_w."""
    batch = all_batch
    mask = batch.theta_v == 30.0

    subset = P.select(batch, mask)

    assert subset.n_sample == int(mask.sum())
    assert subset.iops.a_ph.shape[0] == subset.n_sample
    assert subset.phase_params.ff_B.shape == (subset.n_sample, batch.n_wave, 2)
    assert subset.geometry.diffuse_fraction.shape == (subset.n_sample, batch.n_wave)
    assert subset.theta_w.shape == (subset.n_sample,)
    for name, values in subset.aops.items():
        assert values.shape[0] == subset.n_sample, name
    assert bool((np.asarray(subset.theta_v) == 30.0).all())
    subset.validate()


def test_select_rejects_a_wrong_mask(pb24_small_batch):
    with pytest.raises(ValueError, match="boolean mask"):
        P.select(pb24_small_batch, np.ones(3, dtype=bool))


def test_load_batch_rejects_nonsense_arguments(pb24_cache_dir):
    """Every knob fails loudly rather than guessing."""
    kw = {"cache_dir": pb24_cache_dir}
    with pytest.raises(ValueError, match="angles must be one of"):
        P.load_batch(realisations=(1,), angles="everything", **kw)
    with pytest.raises(ValueError, match="geometry_stride must be"):
        P.load_batch(realisations=(1,), geometry_stride=0, **kw)
    with pytest.raises(ValueError, match="selects nothing"):
        P.load_batch(realisations=(), **kw)
    with pytest.raises(ValueError, match="not in the source"):
        P.load_batch(realisations=(7,), **kw)
    with pytest.raises(ValueError, match="unknown resolution"):
        P.load_batch(realisations=(1,), resolution="hyperspectral", **kw)
    with pytest.raises(ValueError, match="hyp-only"):
        P.load_batch(realisations=(1,), resolution="olci", **kw)


def test_file_path_rejects_an_out_of_range_index():
    with pytest.raises(ValueError, match=r"outside \[1, 5000\]"):
        P.file_path(0)
    with pytest.raises(ValueError, match="unknown resolution"):
        P.file_path(1, "hyperspectral")


def test_data_dir_explains_itself_without_os_color(monkeypatch):
    """A missing mount must say so, not raise a KeyError three frames down."""
    monkeypatch.delenv("OS_COLOR", raising=False)

    with pytest.raises(RuntimeError, match=r"\$OS_COLOR is not set"):
        P.data_dir()


def _dataset_from_cache(cache, index):
    """One realisation of the cache as an ``xarray.Dataset`` in the files' layout."""
    import xarray as xr

    g = cache.grids
    wave = g["lambda"][cache.four_d_bands]
    coords = {
        "lambda": wave,
        "theta_s": g["theta_s"],
        "theta": g["theta"],
        "phi": g["phi"],
    }
    data = {}
    for name in P._SCALARS:
        data[name] = (("scalar",), np.atleast_1d(cache.table(name, index)))
    for name in P._IOPS:
        data[name] = (("lambda",), cache.table(name, index)[cache.four_d_bands])
    for name in P._AOPS:
        values = cache.table(name, index)[:, cache.four_d_bands]
        data[name] = (("lambda", "theta_s"), values.T)
    for name in P._FOUR_D:
        values = cache.four_d(name, index)  # (theta_s, theta, phi, lambda)
        data[name] = (
            ("lambda", "phi", "theta", "theta_s"),
            values.transpose(3, 2, 1, 0),
        )
    return xr.Dataset(data, coords=coords)


def test_the_per_file_path_matches_the_cache_path(cache, pb24_cache_dir):
    """The OLCI path's reader seam, without the data: same numbers, bit for bit.

    A reader that serves the cache's own values in the netCDF layout (dims
    ``(lambda, phi, theta, theta_s)``) must produce exactly the batch the cache
    path does -- so the transposes in the per-file path are right.
    """

    def reader(index, resolution):
        return _dataset_from_cache(cache, index)

    via_cache = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS, angles="all", cache_dir=pb24_cache_dir
    )
    via_files = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS, angles="all", reader=reader
    )

    for name in ("rrs", "Rrs"):
        np.testing.assert_array_equal(
            np.asarray(getattr(via_cache, name)), np.asarray(getattr(via_files, name))
        )
    np.testing.assert_array_equal(
        np.asarray(via_cache.geometry.diffuse_fraction),
        np.asarray(via_files.geometry.diffuse_fraction),
    )
    np.testing.assert_array_equal(
        np.asarray(via_cache.phase_params.ff_B), np.asarray(via_files.phase_params.ff_B)
    )
    np.testing.assert_array_equal(
        np.asarray(via_cache.aops["Q"]), np.asarray(via_files.aops["Q"])
    )


def test_loader_stops_on_a_mismatched_grid(cache):
    """A batch spanning two angle grids would be silently mislabelled."""

    def bad(index, resolution):
        ds = _dataset_from_cache(cache, index)
        if index != PB24_FIXTURE_REALISATIONS[0]:
            ds = ds.assign_coords(phi=ds["phi"].values + 1.0)
        return ds

    with pytest.raises(ValueError, match="different 'phi' grid"):
        P.load_batch(realisations=PB24_FIXTURE_REALISATIONS, reader=bad)


# ------------------------------------------------------- against the raw data -


@needs_pb24
def test_golden_values_against_the_ocpy_reader():
    """**Gate.** Named samples against raw reads through ``ocpy.hydrolight.pb24``.

    An ``Rrs`` at a node; an interpolated ``rrs`` recomputed by hand from the two
    bracketing in-water nodes; ``aph``; ``diffuse_fraction = 1 - Esdir_Es_ratio``;
    ``ff_B``. Realisation 2500, theta_s = 60, theta_v_air = 40 (theta_w = 28.665,
    between the in-water nodes 20 and 30), phi = 135, 560 nm.
    """
    from ocpy.hydrolight import pb24 as O

    ds = O.load_ds(2500)
    batch = P.load_batch(realisations=(2500,), angles="all")
    row = _flat(6, 4, 9)
    band = int(np.flatnonzero(np.asarray(batch.wave) == 560.0)[0])
    sel = {"lambda": band, "theta_s": 6, "phi": 9}

    assert (batch.theta_s[row], batch.theta_v[row], batch.dphi[row]) == (
        60.0,
        40.0,
        135.0,
    )
    # Rrs at its node: the float32 value as stored (0.0447451025 is task 1's golden
    # value at theta = 40 -- so this sample's Rrs is its air node, not the in-water one)
    Rrs_truth = float(ds["Rrs"].isel(theta=4, **sel))
    assert float(batch.Rrs[row, band]) == Rrs_truth
    assert Rrs_truth == pytest.approx(0.0447451025, rel=1e-8)

    theta_w = float(np.degrees(np.arcsin(np.sin(np.radians(40.0)) / 1.34)))
    t = (theta_w - 20.0) / 10.0
    lo = float(ds["rrs"].isel(theta=2, **sel))
    hi = float(ds["rrs"].isel(theta=3, **sel))
    assert float(batch.rrs[row, band]) == pytest.approx((1 - t) * lo + t * hi, rel=1e-6)
    assert batch.theta_w[row] == pytest.approx(28.665, abs=1e-3)

    assert float(batch.iops.a_ph[row, band]) == float(ds["aph"].isel({"lambda": band}))
    esdir = float(ds["Esdir_Es_ratio"].isel({"lambda": band, "theta_s": 6}))
    assert float(batch.geometry.diffuse_fraction[row, band]) == pytest.approx(
        1.0 - esdir, rel=1e-6
    )
    i = {"lambda": band}
    B_ph = float(ds["bbph"].isel(i)) / float(ds["bph"].isel(i))
    B_nap = float(ds["bbNAP"].isel(i)) / float(ds["bNAP"].isel(i))
    np.testing.assert_allclose(
        np.asarray(batch.phase_params.ff_B[row, band]), [B_ph, B_nap], rtol=1e-6
    )


@needs_pb24_olci
def test_golden_values_on_the_olci_path():
    """The OLCI set through the same sample definition (per-file reads)."""
    from ocpy.hydrolight import pb24 as O

    ds = O.load_ds(4321, resolution="olci")
    batch = P.load_batch(realisations=(4321,), angles="all", resolution="olci")
    row = _flat(6, 6, 9)  # theta_s 60, theta_v_air 60, phi 135
    band = 5  # 560 nm
    sel = {"lambda": band, "theta_s": 6, "phi": 9}

    assert batch.n_wave == 12 and float(batch.wave[band]) == 560.0
    assert float(batch.Rrs[row, band]) == float(ds["Rrs"].isel(theta=6, **sel))
    t = (float(P.theta_air_to_water(60.0)) - 40.0) / 10.0
    lo = float(ds["rrs"].isel(theta=4, **sel))
    hi = float(ds["rrs"].isel(theta=5, **sel))
    assert float(batch.rrs[row, band]) == pytest.approx((1 - t) * lo + t * hi, rel=1e-6)
    assert batch.report.resolution == "olci"
    assert batch.report.fresnel.within()


@needs_pb24
def test_the_fresnel_check_on_a_spread_of_realisations():
    """**Gate.** 250 realisations across the release, every geometry and band.

    The check runs on every realisation *read*, whatever the selection, so a
    one-geometry stride keeps the batch small while the check sees all 1300.
    The full 5000 are in the task-3 log (``design/py/check_pb24_geometry.py``).
    """
    batch = P.load_batch(
        realisations=tuple(range(1, 5001, 20)),
        angles="window",
        geometry_stride=(8, 8, 13),
        extras=(),
    )
    fresnel = batch.report.fresnel

    assert fresnel.n_realisation == 250
    assert fresnel.n_excluded == 0
    assert fresnel.within(), fresnel.table()
    medians = dict(
        zip(fresnel.theta_v_air, (d[2] for d in fresnel.deviation), strict=True)
    )
    assert -0.045 < medians[70.0] < -0.025  # Round 2's slice read 0.856-0.864 of 0.883


@needs_pb24
def test_the_one_realisation_beyond_the_fresnel_tolerance():
    """**Measured, not hidden.** Over all 5000 the check holds on 4999.

    Realisation 4492 exceeds 5% at 5 of its values (2.05e9 compared in all):
    theta_v_air = 70, 356-376 nm, theta_s 70 and 80, nadir ``rrs`` ~3e-6 sr^-1
    -- the darkest UV water in the set -- worst -5.30%. The loader warns and the
    report names it; ``design/ztt_pb24/geometry_check.md`` has the full table.
    """
    with pytest.warns(UserWarning, match="Fresnel check leaves"):
        batch = P.load_batch(
            realisations=(4492,), geometry_stride=(8, 8, 13), extras=()
        )
    fresnel = batch.report.fresnel

    assert not fresnel.within()
    assert fresnel.n_beyond == 5
    assert fresnel.realisations_beyond == (4492,)
    assert fresnel.worst() == pytest.approx(0.0530, abs=5e-4)
    assert fresnel.worst(theta_max=60.0) < P.FRESNEL_TOLERANCE


@needs_pb24
def test_the_angle_grid_is_identical_across_files():
    """Asserted, not assumed -- the loader relies on it to label samples."""
    from ocpy.hydrolight import pb24 as O

    reference = None
    for index in (1, 993, 2500, 5000):
        ds = O.load_ds(index, verify_grids=False)
        grids = tuple(
            np.asarray(ds[name].values)
            for name in ("lambda", "theta_s", "theta", "phi")
        )
        if reference is None:
            reference = grids
        for got, want in zip(grids, reference, strict=True):
            assert np.array_equal(got, want), index


@needs_pb24
def test_zero_rrs_only_at_the_87_5_in_water_node_at_scale():
    """The zero-``rrs`` location, on the first 1000 realisations of the cache."""
    cache = PC.open_cache()
    rrs = cache.array("rrs")[:1000]
    zero = ~(np.asarray(rrs) > 0.0)
    where = np.argwhere(zero)

    assert where.size > 0
    assert set(cache.grids["theta"][where[:, 2]].tolist()) == {87.5}
    assert cache.grids["theta_s"][where[:, 1]].min() >= 40.0


@needs_pb24_olci
def test_read_classes_matches_the_sidecar():
    """12 unbalanced classes, one per realisation."""
    classes = P.read_classes()

    assert classes.shape == (P.N_REALISATION,)
    assert set(np.unique(classes).tolist()) == set(range(1, P.N_CLASS + 1))
    counts = np.bincount(classes)[1:]
    assert counts.min() < counts.max() / 5


# ------------------------------------------------------------------- splits --


def test_the_committed_splits_file_is_disjoint_and_non_empty():
    """**Gate.** Every kind has both sides, over all 5000, by id."""
    content = P.read_splits()
    ids = set(content["realisations"])

    assert ids == set(range(1, P.N_REALISATION + 1))
    assert content["seed"] == P.SPLIT_SEED
    assert content["test_fraction"] == P.TEST_FRACTION
    assert tuple(content["bp_band_quantiles"]) == P.BP_BAND_QUANTILES
    for kind in ("realisation", "bp_band"):
        held = set(content[kind]["test"])
        assert held < ids, kind
        assert len(held) == 1000, kind  # 20% each
        assert len(content[kind]["test"]) == len(held), kind
    assert "theta_v_air >= 80 or theta_s >= 80" in content["geometry"]["rule"]
    assert content["geometry"]["held_min_deg"] == P.GEOMETRY_HOLDOUT_MIN


def test_the_committed_splits_file_is_identical_on_reload():
    a, b = P.read_splits(), P.read_splits()
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_bp_band_is_interior_and_cleanly_separated():
    """The band is held out whole; no training realisation's mean B_p is inside."""
    content = P.read_splits()
    bp = dict(zip(content["realisations"], content["B_p_mean"], strict=True))
    lo, hi = content["bp_band"]["lo"], content["bp_band"]["hi"]
    held = set(content["bp_band"]["test"])
    train = np.array([v for i, v in bp.items() if i not in held])

    assert not ((train >= lo) & (train <= hi)).any()
    assert train.min() < lo and train.max() > hi
    assert max(bp.values()) / min(bp.values()) > 5.0  # B_p spans ~30x


def test_the_bp_confound_is_recorded_in_the_file():
    """corr(log B_p, log C) over all 5000, the number the split must disclose."""
    conf = P.read_splits()["confound"]

    assert -0.75 < conf["corr_log_B_p_log_C"] < -0.3
    assert set(conf["bp_band"]) >= {"C", "N", "Y", "a_mean", "bb_p_mean", "B_p_mean"}
    for name, (median, lo, hi) in conf["random_reference"].items():
        assert lo <= median <= hi, name


@needs_pb24
def test_the_splits_file_rebuilds_byte_identically_from_the_cache(tmp_path):
    """The committed file is what the builder produces from the cache today."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[2] / "design" / "py" / "make_pb24_splits.py"
    )
    spec = importlib.util.spec_from_file_location("make_pb24_splits", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ids, bp, labels, source = module.per_realisation()
    out = P.write_splits(
        tmp_path / "splits.json", P.build_splits(ids, bp, labels, source=source)
    )

    assert out.read_text() == P.DEFAULT_SPLITS_FILE.read_text()


def test_make_splits_maps_the_file_onto_a_batch(pb24_small_batch):
    """Realisation 993 is held out by the saved random split; 1 and 2500 train."""
    splits = P.make_splits(pb24_small_batch, kinds=("realisation",))
    train, test = splits.train("realisation"), splits.test("realisation")

    assert set(pb24_small_batch.realisation[test].tolist()) == {993}
    assert set(pb24_small_batch.realisation[train].tolist()) == {1, 2500}
    assert not bool((train & test).any()) and bool((train | test).all())
    assert splits.seed == P.SPLIT_SEED
    assert splits.path == str(P.DEFAULT_SPLITS_FILE)

    again = P.make_splits(pb24_small_batch, kinds=("realisation",))
    np.testing.assert_array_equal(again.test("realisation"), test)


def test_make_splits_refuses_a_side_the_batch_does_not_reach(pb24_small_batch):
    """None of 1, 993, 2500 is in the saved B_p band: an empty side raises."""
    with pytest.raises(ValueError, match="leaves its test side empty"):
        P.make_splits(pb24_small_batch, kinds=("bp_band",))


def test_make_splits_raises_for_a_realisation_not_in_the_file(
    pb24_small_batch, tmp_path
):
    """**Gate.** A batch whose realisations are not all in the file raises."""
    path = P.write_splits(
        tmp_path / "two.json",
        P.build_splits(
            [1, 993],
            [0.01, 0.02],
            {"C": [1.0, 2.0]},
            test_fraction=0.5,
            bp_quantiles=(0.0, 0.5),
        ),
    )

    with pytest.raises(ValueError, match=r"realisations \[2500\] are not in"):
        P.make_splits(pb24_small_batch, kinds=("realisation",), path=path)


@pytest.fixture(scope="module")
def fixture_splits_file(tmp_path_factory, pb24_small_batch):
    """A splits file over the three fixture realisations, built like the real one."""
    ids, bp = P._per_realisation_B_p(pb24_small_batch)
    labels = {
        name: [
            pb24_small_batch.labels[name][pb24_small_batch.realisation == i][0]
            for i in ids
        ]
        for name in ("C", "N", "Y")
    }
    content = P.build_splits(
        ids, bp, labels, bp_quantiles=(0.4, 0.6), test_fraction=0.34
    )
    return P.write_splits(tmp_path_factory.mktemp("splits") / "fixture.json", content)


def test_every_split_is_disjoint_and_exhaustive(all_batch, fixture_splits_file):
    """Train and test partition the sample axis, for every kind."""
    splits = P.make_splits(all_batch, kinds=P.SPLIT_KINDS, path=fixture_splits_file)

    for kind in P.SPLIT_KINDS:
        train, test = splits.train(kind), splits.test(kind)

        assert train.shape == test.shape == (all_batch.n_sample,)
        assert not bool((train & test).any()), kind
        assert bool((train | test).all()), kind
        assert splits.reports[kind].n_test > 0 and splits.reports[kind].n_train > 0


def test_bp_band_holds_out_an_interior_band(pb24_small_batch, fixture_splits_file):
    """Train is both tails, so the split tests interpolation, not extrapolation."""
    report = P.make_splits(
        pb24_small_batch, kinds=("bp_band",), path=fixture_splits_file
    ).reports["bp_band"]
    detail = report.detail

    assert detail["B_p_train_lo"] < detail["B_p_band_lo"]
    assert detail["B_p_train_hi"] > detail["B_p_band_hi"]
    assert detail["n_train_inside_band"] == 0.0
    assert set(report.confound) >= {"C", "N", "Y", "a_mean", "bb_p_mean", "B_p_mean"}
    assert any(abs(v - 1.0) > 0.05 for v in report.confound.values())
    assert "side-effect" in report.summary()


def test_confound_reference_is_a_band_not_a_point(pb24_small_batch):
    """A random hold-out already moves heavy-tailed labels; that is the yardstick."""
    reference = P.confound_reference(
        pb24_small_batch, test_fraction=0.34, seeds=range(5)
    )

    assert set(reference) >= {"C", "B_p_mean"}
    for name, (median, lo, hi) in reference.items():
        assert lo <= median <= hi, name


def test_geometry_split_refuses_a_window_only_batch(pb24_small_batch):
    """The interaction that would have made this split score nothing at all."""
    with pytest.raises(ValueError, match="angles='all'"):
        P.make_splits(pb24_small_batch, kinds=("geometry",))


def test_geometry_split_is_the_rule_in_air_angles(all_batch):
    """Train stops at 70 degrees in air; test starts at 80 -- 468 per realisation."""
    splits = P.make_splits(all_batch, kinds=("geometry",))
    train, test = splits.train("geometry"), splits.test("geometry")
    worst = np.maximum(all_batch.theta_s, all_batch.theta_v)
    report = splits.reports["geometry"]

    assert worst[train].max() == P.ANGLE_WINDOW_MAX == 70.0
    assert worst[test].min() == P.GEOMETRY_HOLDOUT_MIN == 80.0
    assert report.n_test == len(PB24_FIXTURE_REALISATIONS) * N_SHELL
    assert report.detail["worst_angle_max_test"] == pytest.approx(87.75)
    # every realisation on both sides: it divides angles, not water bodies
    assert report.n_train_realisation == report.n_test_realisation == 3
    assert all(v == pytest.approx(1.0) for v in report.confound.values())


def test_geometry_split_refuses_in_water_angles(in_water_batch):
    with pytest.raises(ValueError, match="above-water"):
        P.make_splits(in_water_batch, kinds=("geometry",))


def test_make_splits_rejects_nonsense_arguments(pb24_small_batch):
    with pytest.raises(ValueError, match="unknown split kind"):
        P.make_splits(pb24_small_batch, kinds=("realization",))
    with pytest.raises(ValueError, match="selects nothing"):
        P.make_splits(pb24_small_batch, kinds=())
    with pytest.raises(ValueError, match="test_fraction"):
        P.build_splits([1, 2], [0.01, 0.02], test_fraction=0.0)
    with pytest.raises(ValueError, match="bp_quantiles"):
        P.build_splits([1, 2], [0.01, 0.02], bp_quantiles=(0.6, 0.4))


# ------------------------------------ the per-consumer surface transfer ------
# `fit_transfer` (M5, design/m5_report.md §6) must be built from the training
# rows alone, say which those were, and refuse a mask that empties a grid cell.


def test_fit_transfer_sees_only_the_training_rows(pb24_small_batch):
    """**The gate.** Held-out realisations must not reach the fitted table."""
    batch = pb24_small_batch
    train = batch.realisation != 993

    transfer = P.fit_transfer(batch, train)

    sub = P.select(batch, train)
    direct = C.fit_surface_transfer(
        np.asarray(sub.rrs),
        np.asarray(sub.Rrs),
        sub.theta_s,
        sub.theta_v,
        sub.dphi,
    )

    np.testing.assert_array_equal(transfer.A, direct.A)
    np.testing.assert_array_equal(transfer.B, direct.B)


def test_fit_transfer_records_which_realisations_it_used(pb24_small_batch):
    batch = pb24_small_batch
    train = batch.realisation != 993

    transfer = P.fit_transfer(batch, train)

    kept = np.unique(batch.realisation[train])
    assert f"{kept.size} training realisations" in transfer.provenance
    assert f"({kept.min()}-{kept.max()})" in transfer.provenance
    assert "above-water geometry" in transfer.provenance


def test_fit_transfer_refuses_a_mask_that_empties_a_grid_cell(pb24_small_batch):
    """A hole in the table interpolates across itself silently; raise instead."""
    batch = pb24_small_batch
    theta_v = batch.theta_v
    train = np.ones(batch.n_sample, dtype=bool)
    train[(theta_v == np.unique(theta_v)[1]) & (batch.theta_s == batch.theta_s[0])] = (
        False
    )

    with pytest.raises(ValueError, match="had no samples"):
        P.fit_transfer(batch, train)


def test_fit_transfer_refuses_an_in_water_only_batch(in_water_batch):
    with pytest.raises(ValueError, match="no Rrs"):
        P.fit_transfer(in_water_batch, np.ones(in_water_batch.n_sample, bool))
