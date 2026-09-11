"""
Tests for :mod:`robust.solar`.

No fixture, no marker, no ``$OS_COLOR``, no JAX: this module is pure NumPy
arithmetic over the calendar, so everything here runs everywhere, CI included.
That is deliberate and is itself asserted -- :func:`test_import_does_not_pull_jax`
fails if someone ever reaches for ``jax.numpy`` in a module that data-prep code
imports before it has decided whether it wants a GPU.

Four kinds of check, in increasing order of independence from the code under
test.

**Against a real ephemeris.** :data:`REFERENCE` holds ten geocentric-apparent
solar positions computed with ``astropy`` 8.0.1 / ERFA -- a completely separate
implementation, built on the JPL-grade IAU models rather than on Meeus's
truncated series. The generating snippet is in the table's comment so anyone can
reproduce it; ``astropy`` is *not* imported here, because CI installs only what
the suite needs and a reference value has to be a constant to be a reference.

**Against the printed literature.** Meeus, *Astronomical Algorithms*, works two
examples for this exact date, and both are asserted: Example 25.b (apparent
declination) and Example 28.b (equation of time). These are numbers in a book,
independent of every piece of software involved.

**Against physics.** The Sun is overhead at the subsolar point; declination
vanishes at the equinoxes and reaches the obliquity at the solstices; the zenith
is minimised at local solar noon. None of these needs an external number at all.

**Against the interface contract.** Every documented input form gives the same
answer, longitudes wrap, broadcasting follows NumPy, and polar night is reported
rather than clamped.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys

import numpy as np
import pytest

from robust import solar

#: Independent reference positions, ``(iso_utc, lat, lon, zenith, azimuth)`` in
#: degrees, geometric (unrefracted), Sun's centre. Generated with astropy 8.0.1
#: (ERFA / IAU models, ``get_sun`` apparent place) by::
#:
#:     from astropy.coordinates import AltAz, EarthLocation, get_sun
#:     from astropy.time import Time
#:     import astropy.units as u
#:     t = Time(iso, scale="utc")
#:     loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)
#:     sun = get_sun(t).transform_to(AltAz(obstime=t, location=loc,
#:                                         pressure=0*u.hPa))  # 0 hPa = no refraction
#:     zenith, azimuth = 90.0 - sun.alt.deg, sun.az.deg
#:
#: The cases are chosen to exercise the corners the pipeline will actually meet:
#: both hemispheres, the date line, polar day and polar night, a pole, and the
#: two ends of the 1900--2100 window the module claims.
REFERENCE = (
    # (iso_utc, lat, lon, zenith, azimuth)
    # Santa Cruz, June solstice, near local solar noon
    ("2024-06-20T20:10:00", 36.9741, -122.0308, 13.53655, 180.0951),
    # Bermuda / the BATS time-series station, spring afternoon
    ("2024-04-15T17:30:00", 31.6667, -64.1667, 27.44406, 222.3080),
    # Equator on the date line, at the March equinox
    ("2023-03-20T21:24:00", 0.0, 180.0, 40.85330, 90.0005),
    # Southern Ocean south of Tasmania, austral winter
    ("2021-07-05T03:15:00", -55.0, 145.0, 78.51768, 348.1519),
    # Utqiagvik (Barrow), Alaska -- polar day
    ("2019-06-21T22:00:00", 71.2906, -156.7886, 48.03846, 171.0441),
    # Utqiagvik again -- polar night, zenith past 90
    ("2019-12-21T22:00:00", 71.2906, -156.7886, 94.83158, 174.1844),
    # Amundsen-Scott, the South Pole itself (azimuth undefined there)
    ("2020-01-15T06:00:00", -90.0, 0.0, 68.78986, 92.2829),
    # Bay of Bengal, monsoon morning
    ("1995-08-10T04:45:00", 15.0, 88.0, 21.32414, 85.1660),
    # North Sea, low sun -- the far end of the 1900-2100 window
    ("2100-11-05T15:00:00", 56.0, 3.0, 84.00703, 229.7765),
    # Mid-Atlantic -- the near end of that window
    ("1901-09-23T12:00:00", 20.0, -40.0, 42.28930, 113.4168),
)

#: Gate for the comparison above. The measured worst case over the table is
#: 0.010 deg, and 0.016 deg over a 4000-point random sweep of 1900--2100 and the
#: whole globe, so this leaves ~3x of headroom -- tight enough that a real
#: regression (a dropped series term, a sign, a wrong epoch) cannot hide inside
#: it, loose enough that it is not a test of ERFA's aberration model. Note that
#: the residual is dominated by what the NOAA truncation omits, *not* by the
#: ~0.006 deg of aberration and ~0.002 deg of parallax that separate astropy's
#: apparent topocentric place from a geometric one.
TOL_DEG = 0.05

#: Utqiagvik (Barrow), Alaska: the canonical polar-night site.
UTQIAGVIK = (71.2906, -156.7886)


# --------------------------------------------------- against a real ephemeris -


@pytest.mark.parametrize(("iso", "lat", "lon", "zenith", "azimuth"), REFERENCE)
def test_zenith_matches_astropy(iso, lat, lon, zenith, azimuth):
    """Every reference zenith is reproduced to :data:`TOL_DEG`."""
    del azimuth  # checked separately, where the pole has to be excluded
    assert solar.solar_zenith(iso, lat, lon) == pytest.approx(zenith, abs=TOL_DEG)


@pytest.mark.parametrize(("iso", "lat", "lon", "zenith", "azimuth"), REFERENCE)
def test_azimuth_matches_astropy(iso, lat, lon, zenith, azimuth):
    """The azimuths match too, except at the pole where azimuth has no meaning.

    At ``lat = -90`` every direction is north and the arctan2 arguments both
    collapse to zero, so the returned bearing is arbitrary; astropy's is
    arbitrary in a different way. Excluding that row is the honest thing to do,
    and the module docstring says so.
    """
    del zenith
    if abs(lat) > 89.0:
        pytest.skip("azimuth is undefined at the geographic poles")
    _, computed = solar.solar_position(iso, lat, lon)
    # Compare on the circle: 359.99 and 0.01 are 0.02 deg apart, not 359.98.
    assert (computed - azimuth + 180.0) % 360.0 - 180.0 == pytest.approx(
        0.0, abs=TOL_DEG
    )


def test_reference_set_spans_the_claimed_window():
    """The table really does bracket 1900--2100, so the era claim is tested.

    Cheap insurance against someone pruning the 1901 and 2100 rows -- the two
    that catch an epoch or Julian-century error -- while the suite stays green.
    """
    years = sorted(int(row[0][:4]) for row in REFERENCE)
    assert years[0] < 1910 and years[-1] > 2090


# ------------------------------------------------ against the printed literature


def test_declination_matches_meeus_example_25b():
    """Meeus (1998) Example 25.b: 1992 Oct 13.0 TD, apparent decl -7 deg 47' 01.74".

    The example is stated in TD (= TT), and this function takes UTC; in 1992
    those differ by ~59 s, over which the declination moves 0.0003 deg. That is
    an order of magnitude below the gate, so the distinction is noted rather
    than corrected for -- exactly as the NOAA calculator itself does.
    """
    expected = -(7.0 + 47.0 / 60.0 + 1.74 / 3600.0)  # -7.783817 deg

    computed = solar.solar_declination("1992-10-13T00:00:00")

    assert computed == pytest.approx(expected, abs=0.01)


def test_equation_of_time_matches_meeus_example_28b():
    """Meeus (1998) Example 28.b: the same instant, E = +13m 42.6s."""
    expected = 13.0 + 42.6 / 60.0  # 13.710 minutes

    computed = solar.equation_of_time("1992-10-13T00:00:00")

    assert computed == pytest.approx(expected, abs=0.02)


# ------------------------------------------------------------ against physics -


def test_sun_is_overhead_at_the_subsolar_point():
    """Zenith is 0 at (declination, subsolar longitude), and |lat - decl| off it.

    The strongest check in the file, because it needs no external number: the
    subsolar longitude is where true solar time is noon, which the module's own
    equation of time locates, and the Sun is by definition overhead there. The
    offset case then pins the whole spherical-trigonometry step, since on the
    noon meridian the zenith must be exactly the latitude difference.
    """
    iso = "2024-05-14T19:37:02"
    minutes = 19 * 60 + 37 + 2 / 60.0
    declination = float(solar.solar_declination(iso))
    # true solar time = 720 min  =>  4 * lon = 720 - utc_minutes - eot
    subsolar_lon = (720.0 - minutes - float(solar.equation_of_time(iso))) / 4.0

    assert solar.solar_zenith(iso, declination, subsolar_lon) == pytest.approx(
        0.0, abs=1e-6
    )
    for offset in (-40.0, -17.0, 5.0, 33.0):
        assert solar.solar_zenith(
            iso, declination + offset, subsolar_lon
        ) == pytest.approx(abs(offset), abs=1e-6)


@pytest.mark.parametrize(
    ("iso", "expected"),
    [
        # Equinox and solstice instants, USNO / IERS published times (UTC).
        ("2023-03-20T21:24:00", 0.0),
        ("2023-09-23T06:50:00", 0.0),
        ("2024-03-20T03:06:00", 0.0),
        ("2023-06-21T14:58:00", 23.4365),  # obliquity of the ecliptic, 2023
        ("2023-12-22T03:27:00", -23.4365),
    ],
)
def test_declination_at_equinoxes_and_solstices(iso, expected):
    """Declination vanishes at the equinoxes and equals the obliquity at solstice.

    That is what those four instants *are*, so this tests the declination series
    against the definition rather than against another program. The 0.01 deg
    gate accommodates the nutation and aberration terms (up to ~0.003 deg) that
    separate the apparent declination returned here from the mean obliquity
    quoted for 2023.
    """
    assert solar.solar_declination(iso) == pytest.approx(expected, abs=0.01)


def test_zenith_is_minimised_at_local_solar_noon():
    """Sweeping a day at one place, the minimum zenith lands on solar noon.

    Solar noon is predicted independently of the sweep -- from the equation of
    time and the longitude -- so agreement means the hour-angle convention and
    the equation-of-time sign are both right. A sign error on the equation of
    time would shift the minimum by up to ~30 minutes and is exactly what this
    catches.
    """
    lat, lon = 36.9741, -122.0308  # Santa Cruz
    midday = np.datetime64("2024-09-12T12:00")
    # Solar noon in minutes after 00:00 UTC: mean noon, shifted west by the
    # longitude (4 min/deg) and by the equation of time.
    predicted_noon = 720.0 - 4.0 * lon - float(solar.equation_of_time(midday))

    # One full solar day, centred on the prediction, sampled every minute.
    offsets = np.arange(-720, 721)
    times = (
        np.datetime64("2024-09-12T00:00")
        + np.timedelta64(int(round(predicted_noon)), "m")
        + offsets.astype("timedelta64[m]")
    )
    zenith = solar.solar_zenith(times, lat, lon)
    argmin = int(np.argmin(zenith))

    assert abs(int(offsets[argmin])) <= 1
    # ... and the profile is a single bowl about it. The check stops 2 h short
    # of solar midnight at either end, where the curve is flat to within the
    # declination's drift over one minute and "monotone" stops being meaningful.
    lo, hi = 120, 1321  # offsets -600 .. +600 minutes
    assert np.all(np.diff(zenith[lo : argmin + 1]) < 0.0)
    assert np.all(np.diff(zenith[argmin:hi]) > 0.0)


def test_zenith_stays_within_the_geometric_range():
    """0 <= zenith <= 180 over a coarse global, decade-long sweep."""
    times = np.arange(
        np.datetime64("2015-01-01T00:00"),
        np.datetime64("2025-01-01T00:00"),
        np.timedelta64(3607, "m"),
    )
    lats = np.linspace(-90.0, 90.0, times.size)
    lons = np.linspace(-180.0, 180.0, times.size)

    zenith = solar.solar_zenith(times, lats, lons)

    assert np.all(np.isfinite(zenith))
    assert zenith.min() >= 0.0 and zenith.max() <= 180.0


# ---------------------------------------------------------- polar night and day


def test_polar_night_is_reported_not_clamped():
    """Utqiagvik in mid-December: the Sun is below the horizon at every hour.

    The point is the *absence* of a clamp -- the values must be genuinely above
    90 deg, and must still vary with hour, because "how far below the horizon"
    is information a clamp would throw away.
    """
    hours = np.datetime64("2019-12-21T00:00") + np.arange(24).astype("timedelta64[h]")

    zenith = solar.solar_zenith(hours, *UTQIAGVIK)

    assert zenith.shape == (24,)
    assert np.all(zenith > 90.0)
    assert zenith.max() > 100.0  # local midnight: well down
    assert np.ptp(zenith) > 20.0  # still a real diurnal cycle


def test_polar_day_never_sets():
    """The same place at the June solstice: above the horizon at every hour."""
    hours = np.datetime64("2019-06-21T00:00") + np.arange(24).astype("timedelta64[h]")

    zenith = solar.solar_zenith(hours, *UTQIAGVIK)

    assert np.all(zenith < 90.0)
    assert zenith.min() < 50.0


# ------------------------------------------------------- longitude wraparound -


def test_equivalent_longitudes_agree_exactly():
    """-170, 190 and 550 are one meridian, and give one answer.

    A single 360 deg shift is *bit-identical*: the longitude enters as
    ``mod(... + 4 * lon, 1440)``, and 360 deg is exactly 1440 minutes, so the
    addition is exact in binary floating point. Two or more turns push the
    intermediate into a coarser binade and cost a few ulp, which is what the
    1e-9 deg gate on the far branches allows -- 4 microdegrees of arc, or about
    0.4 mm on the ground. A pipeline that normalises its longitudes differently
    upstream must not see a different sun.
    """
    iso = "2020-05-05T12:34:56"

    reference = solar.solar_zenith(iso, 12.0, -170.0)

    for one_turn in (190.0, -530.0):
        assert solar.solar_zenith(iso, 12.0, one_turn) == reference
    for many_turns in (550.0, -890.0, 1990.0):
        assert solar.solar_zenith(iso, 12.0, many_turns) == pytest.approx(
            reference, abs=1e-9
        )


def test_date_line_is_not_a_discontinuity():
    """Crossing the date line changes nothing; 0.2 deg of longitude changes little.

    ``+179.9`` and ``-180.1`` are the *same* meridian (they differ by exactly
    360), so they must agree exactly. ``+179.9`` and ``-179.9`` are two points
    0.2 deg apart, which can move the zenith by at most 0.2 deg -- and must not
    move it by the ~90 deg a mishandled wrap would produce.
    """
    iso = "2021-09-09T03:00:00"

    east = solar.solar_zenith(iso, 10.0, 179.9)
    same_meridian = solar.solar_zenith(iso, 10.0, -180.1)
    west = solar.solar_zenith(iso, 10.0, -179.9)

    assert east == same_meridian
    assert abs(east - west) < 0.25


# ------------------------------------------------------------- input handling -


def test_all_scalar_time_formats_agree_exactly():
    """datetime, ISO string, date-only string and datetime64 are one instant.

    Exact equality, not ``approx``: these are four spellings of the same
    microsecond, and any difference means a parse -- not the arithmetic -- went
    wrong.
    """
    lat, lon = -12.5, 43.25
    expected = solar.solar_zenith(dt.datetime(2022, 11, 3, 8, 45, 30), lat, lon)

    for spelling in (
        "2022-11-03T08:45:30",
        "2022-11-03 08:45:30",
        np.datetime64("2022-11-03T08:45:30"),
        np.datetime64("2022-11-03T08:45:30.000000", "us"),
        np.datetime64(1667465130, "s"),
    ):
        assert solar.solar_zenith(spelling, lat, lon) == expected


def test_array_time_formats_agree_exactly():
    """The same four spellings agree when passed as arrays, elementwise.

    Sequences of ``datetime`` and of ``str`` arrive as object / unicode arrays
    and go down the element-wise path, while a ``datetime64`` array is cast
    wholesale; this is the test that those two paths cannot diverge.
    """
    lat, lon = np.array([0.0, 45.0, -60.0]), np.array([10.0, -75.0, 150.0])
    stamps = ["2022-11-03T08:45:30", "2023-02-14T23:01:00", "2024-08-01T11:11:11"]

    from_strings = solar.solar_zenith(stamps, lat, lon)
    from_datetime64 = solar.solar_zenith(
        np.array(stamps, dtype="datetime64[s]"), lat, lon
    )
    from_datetimes = solar.solar_zenith(
        [dt.datetime.fromisoformat(s) for s in stamps], lat, lon
    )

    np.testing.assert_array_equal(from_datetime64, from_strings)
    np.testing.assert_array_equal(from_datetimes, from_strings)


def test_date_only_input_is_midnight_utc():
    """A ``datetime.date`` and a bare date string both mean 00:00 UTC."""
    expected = solar.solar_zenith("2020-01-01T00:00:00", 0.0, 0.0)

    assert solar.solar_zenith(dt.date(2020, 1, 1), 0.0, 0.0) == expected
    assert solar.solar_zenith("2020-01-01", 0.0, 0.0) == expected


def test_timezone_aware_input_is_converted_to_utc():
    """An aware datetime equals its UTC twin -- and is not read as naive.

    Both halves matter. The first assert would pass on an implementation that
    silently dropped the offset only if the offset were zero; the second pins
    that the offset was actually *applied*, by showing the aware value differs
    from the same wall-clock time read as UTC.
    """
    eastern = dt.timezone(dt.timedelta(hours=-5))
    aware = dt.datetime(2020, 1, 1, 12, 0, 0, tzinfo=eastern)

    assert solar.solar_zenith(aware, 30.0, -75.0) == solar.solar_zenith(
        dt.datetime(2020, 1, 1, 17, 0, 0), 30.0, -75.0
    )
    assert solar.solar_zenith(aware, 30.0, -75.0) != solar.solar_zenith(
        dt.datetime(2020, 1, 1, 12, 0, 0), 30.0, -75.0
    )


def test_iso_string_offsets_and_z_suffix_are_honoured():
    """``...Z`` and ``...-04:00`` mean what ISO-8601 says they mean.

    NumPy's own parser deprecated timezone-aware strings, so these go through
    :meth:`datetime.datetime.fromisoformat`; this test is what stops that
    dispatch from quietly regressing to "offset ignored".
    """
    utc = solar.solar_zenith("2024-05-14T19:37:02", 31.2, -64.1)

    assert solar.solar_zenith("2024-05-14T19:37:02Z", 31.2, -64.1) == utc
    assert solar.solar_zenith("2024-05-14T15:37:02-04:00", 31.2, -64.1) == utc


def test_nat_propagates_as_nan():
    """A missing timestamp yields NaN rather than a year-292277026596 answer."""
    times = np.array(["2020-01-01T00:00", "NaT", "2020-07-01T00:00"], "datetime64[m]")

    zenith = solar.solar_zenith(times, 0.0, 0.0)

    assert np.isnan(zenith[1])
    assert np.all(np.isfinite(zenith[[0, 2]]))


def test_numeric_time_is_rejected():
    """A bare number is ambiguous ("seconds since when?") and must not be guessed."""
    with pytest.raises(TypeError, match="dtype"):
        solar.solar_zenith(1_600_000_000.0, 0.0, 0.0)


def test_unparseable_object_is_rejected():
    """So is anything else that is not a recognisable instant."""
    with pytest.raises(TypeError, match="cannot interpret"):
        solar.solar_zenith(object(), 0.0, 0.0)


# ------------------------------------------------- shapes and the return type -


def test_all_scalar_input_returns_a_numpy_scalar():
    """Scalars in, ``numpy.float64`` out -- the convention NumPy's ufuncs use."""
    zenith = solar.solar_zenith("2020-06-01T12:00:00", 10.0, 20.0)

    assert isinstance(zenith, np.float64)
    assert np.ndim(zenith) == 0


@pytest.mark.parametrize(
    ("n_time", "lat_shape", "lon_shape", "expected"),
    [
        (7, (), (), (7,)),
        (1, (5,), (), (5,)),
        (4, (4,), (4,), (4,)),
        (3, (2, 3), (3,), (2, 3)),
    ],
)
def test_broadcasting_follows_numpy(n_time, lat_shape, lon_shape, expected):
    """Times, latitudes and longitudes broadcast by the ordinary rules."""
    times = np.datetime64("2024-03-01T00:00") + np.arange(n_time).astype(
        "timedelta64[h]"
    )
    if n_time == 1:
        times = times[0]
    lat = np.full(lat_shape, 42.0) if lat_shape else 42.0
    lon = np.full(lon_shape, -8.0) if lon_shape else -8.0

    zenith = solar.solar_zenith(times, lat, lon)

    assert zenith.shape == expected
    assert zenith.dtype == np.float64


def test_scalar_time_over_a_latitude_grid():
    """One instant, many latitudes: the shape and the trend are both checked.

    At 12:00 UTC on the Greenwich meridian near the June solstice the Sun is
    close to overhead at +23 deg, so the zenith must rise monotonically as the
    grid runs south from there.
    """
    lats = np.linspace(23.0, -80.0, 30)

    zenith = solar.solar_zenith("2024-06-21T12:00:00", lats, 0.0)

    assert zenith.shape == (30,)
    assert np.all(np.diff(zenith) > 0.0)


def test_solar_position_returns_both_and_agrees_with_solar_zenith():
    """``solar_position`` is ``solar_zenith`` plus an azimuth, not a second model."""
    times = np.array(["2024-01-05T09:00", "2024-07-05T21:00"], "datetime64[m]")

    zenith, azimuth = solar.solar_position(times, 45.0, -120.0)

    np.testing.assert_array_equal(zenith, solar.solar_zenith(times, 45.0, -120.0))
    assert azimuth.shape == zenith.shape
    assert np.all((azimuth >= 0.0) & (azimuth < 360.0))


# ------------------------------------------------------------ the JAX promise -


def test_import_does_not_pull_jax():
    """``import robust.solar`` must not import JAX.

    The module's reason to live outside ``robust.rt`` is that data preparation
    should not pay for -- or be constrained by -- the JAX stack. A subprocess,
    because by the time this file runs the rest of the suite has long since
    imported jax into this interpreter.
    """
    code = "import robust.solar, sys; print('jax' in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
