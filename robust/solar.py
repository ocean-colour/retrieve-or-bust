"""
Geometric solar zenith (and azimuth) from UTC observation time and position.

**Why this lives at the package top level and not in** ``robust.rt``. Everything
under ``robust/rt/`` is radiative-transfer *physics* -- in-water absorption,
backscattering, the inelastic terms -- written in JAX because it has to be
differentiable. This module is neither: it is *observation geometry*, the
arithmetic that turns "this PACE pixel was measured at 2024-05-14T19:37:02Z at
(31.2 N, 64.1 W)" into the ``theta_s`` that
:class:`Geometry <robust.rt.types.Geometry>` wants. It runs once at data-prep
time, over at most a few million records, and nothing ever takes a gradient
through it. So it is plain NumPy: importing it must not touch JAX, and it adds
no dependency beyond NumPy itself.

**The algorithm** is the one NOAA's Global Monitoring Laboratory publishes as
its "Solar Calculation Details" and implements in the NOAA Solar Calculator
(https://gml.noaa.gov/grad/solcalc/calcdetails.html) -- the Julian-century
series for the Sun's geometric mean longitude, mean anomaly, equation of centre,
apparent longitude and corrected obliquity, from which the declination and the
equation of time follow. The series are Meeus, *Astronomical Algorithms*
(2nd ed., 1998), chapters 25 and 28, truncated as NOAA truncates them. NOAA
quotes the result as good to about a minute of time for sunrise/sunset over
1800--2100; in angle it agrees with a full ephemeris to **~0.01 deg over
1900--2100**, and the test suite measures that against ``astropy`` rather than
asserting it from this comment. That is an order of magnitude better than the
0.1--0.2 deg the retrieval pipeline needs, and far better than the simpler
"fractional year" form in NOAA's ``solareqns.PDF``, whose fixed day-of-year
phase drifts by up to ~0.3 deg across the leap cycle.

**Three deliberate choices**, each of which a caller can be bitten by:

* **No atmospheric refraction.** What comes back is the *geometric* zenith of
  the Sun's centre. This is what an in-water RT geometry wants -- the Snell
  refraction at the sea surface is the forward model's business, and applying
  an atmospheric-refraction correction here would double-count the bending of
  the ray near the horizon. Near sunrise/sunset the apparent (refracted) Sun
  sits up to ~0.6 deg higher than the value returned here.
* **No clamping at the horizon.** Polar night simply returns a zenith greater
  than 90 deg, and so does any night-time record anywhere. That is information
  -- the Sun is below the horizon, by this much -- and clamping it to 90 deg
  would destroy it silently. Callers who need a day/night mask should write
  ``zenith < 90.0`` and mean it.
* **UTC, always.** ``time`` is interpreted as UTC. A naive
  :class:`datetime.datetime` is *assumed* to be UTC; a timezone-aware one is
  converted. There is no local-time entry point on purpose: a local time
  without its zone is the single most common way to be an hour and 15 deg
  wrong.

Examples
--------
At the June 2020 solstice the Sun stands over the Tropic of Cancer:

>>> from robust import solar
>>> round(float(solar.solar_declination("2020-06-20T21:43:00")), 3)
23.437
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

__all__ = [  # noqa: RUF022  - grouped by role, not alphabetical
    # The public entry points
    "solar_zenith",
    "solar_position",
    # The two intermediates, useful on their own
    "solar_declination",
    "equation_of_time",
    # Provenance
    "ALGORITHM",
]

#: Human-readable provenance for the series below, so a figure caption or a
#: written-up table can name the algorithm without anyone re-deriving it from
#: the source.
ALGORITHM = (
    "NOAA Global Monitoring Laboratory Solar Calculator "
    "(https://gml.noaa.gov/grad/solcalc/calcdetails.html), after Meeus, "
    "Astronomical Algorithms, 2nd ed., ch. 25 & 28; geometric (unrefracted)."
)

#: Days from the Unix epoch (1970-01-01T00:00Z) to J2000.0
#: (2000-01-01T12:00 TT, JD 2451545.0). The Julian centuries the series need
#: are formed from this offset rather than from an absolute Julian Day, so the
#: float64 arithmetic never has to carry 2.45e6 worth of leading digits.
_DAYS_1970_TO_J2000 = 10957.5

_US_PER_DAY = 86_400_000_000
_US_PER_MINUTE = 60_000_000
_MINUTES_PER_DAY = 1440.0

#: Degrees of hour angle per minute of true solar time (360 / 1440).
_DEG_PER_MINUTE = 0.25


# ------------------------------------------------------------ time handling --


def _scalar_to_datetime64(value: Any) -> np.datetime64:
    """Normalise one time-like object to a UTC ``datetime64[us]``.

    Parameters
    ----------
    value : datetime.datetime or datetime.date or str or numpy.datetime64
        A single instant. A naive ``datetime`` is taken to be UTC; an aware one
        is converted to UTC and its offset dropped. Strings go through
        :meth:`datetime.datetime.fromisoformat` first -- which, unlike NumPy,
        understands a trailing ``Z`` and ``+HH:MM`` offsets without a
        deprecation warning -- and fall back to NumPy's parser for the partial
        forms it alone accepts (``"2020"``, ``"2020-06"``).

    Returns
    -------
    numpy.datetime64
        In microseconds, timezone-naive, meaning UTC.

    Raises
    ------
    TypeError
        If ``value`` is not one of the accepted types. Numbers are rejected
        loudly rather than guessed at: "seconds since when?" has no good
        default.
    """
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]")
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC).replace(tzinfo=None)
        return np.datetime64(value, "us")
    if isinstance(value, dt.date):
        return np.datetime64(value.isoformat(), "us")
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value)
        except ValueError:
            return np.datetime64(value, "us")
        return _scalar_to_datetime64(parsed)
    raise TypeError(
        f"cannot interpret {value!r} (type {type(value).__name__}) as a UTC "
        "time; pass a datetime, an ISO-8601 string, or a numpy.datetime64"
    )


def _as_utc_datetime64(time: Any) -> np.ndarray:
    """Normalise any accepted ``time`` input to a ``datetime64[us]`` array.

    Parameters
    ----------
    time : datetime-like or str or numpy.datetime64 or array_like
        Scalar or array/sequence of any mixture of those.

    Returns
    -------
    numpy.ndarray
        ``datetime64[us]``, with the shape of the input (0-d for a scalar).

    Raises
    ------
    TypeError
        If the input is numeric, or holds an element that
        :func:`_scalar_to_datetime64` cannot interpret.
    """
    arr = np.asarray(time)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[us]")
    if arr.dtype.kind in "OSU":
        flat = [_scalar_to_datetime64(v) for v in arr.reshape(-1).tolist()]
        return np.array(flat, dtype="datetime64[us]").reshape(arr.shape)
    raise TypeError(
        f"time has dtype {arr.dtype!r}; expected datetimes, ISO-8601 strings, "
        "or numpy.datetime64 (a bare number is ambiguous and is not accepted)"
    )


def _julian_century_and_utc_minutes(time: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split a UTC time into the two quantities the algorithm needs.

    Parameters
    ----------
    time : datetime-like or str or numpy.datetime64 or array_like
        See :func:`solar_zenith`.

    Returns
    -------
    jc : numpy.ndarray
        Julian centuries since J2000.0. Drives the ephemeris series; a 69-second
        error here (the UTC-vs-TT offset the NOAA algorithm deliberately
        ignores) moves the Sun by 0.0008 deg of ecliptic longitude, which is why
        ignoring it is safe.
    minutes : numpy.ndarray
        Minutes elapsed since 00:00 UTC on the same day, ``[0, 1440)``. Drives
        the hour angle, where the clock *is* the answer and no time-scale
        conversion belongs.

    Notes
    -----
    ``NaT`` propagates as ``nan`` through both outputs rather than becoming a
    large negative integer.
    """
    stamps = _as_utc_datetime64(time)
    invalid = np.isnat(stamps)
    micros = stamps.astype("int64")

    days = np.where(invalid, np.nan, micros / _US_PER_DAY)
    jc = (days - _DAYS_1970_TO_J2000) / 36525.0
    # Python/NumPy floor-modulo, so pre-1970 instants land in [0, 1440) too.
    minutes = np.where(invalid, np.nan, (micros % _US_PER_DAY) / _US_PER_MINUTE)
    return jc, minutes


def _out(values: Any) -> Any:
    """Return float64, as a 0-d-collapsing NumPy scalar for all-scalar input.

    Indexing with an empty tuple is the idiomatic collapse: it turns a 0-d
    array into a :class:`numpy.float64` and leaves any other array untouched,
    which is exactly the convention a ufunc like :func:`numpy.sin` follows.
    """
    return np.asarray(values, dtype=np.float64)[()]


# ------------------------------------------------------------- the ephemeris -


def _sun_declination_and_eot(jc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Declination (deg) and equation of time (minutes) from Julian centuries.

    A transcription of the NOAA Solar Calculator's spreadsheet columns, in
    their order, with the spreadsheet's own names in the comments so the two
    can be read side by side.

    Parameters
    ----------
    jc : numpy.ndarray
        Julian centuries since J2000.0.

    Returns
    -------
    declination : numpy.ndarray
        Apparent solar declination, degrees, positive north.
    eot : numpy.ndarray
        Equation of time, minutes: apparent solar time minus mean solar time,
        so it is *added* to the clock to get true solar time. Ranges over
        roughly -14 to +16 minutes across the year.
    """
    # Geometric mean longitude of the Sun (deg), wrapped for the tan/sin below.
    mean_long = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    # Geometric mean anomaly (deg) -- deliberately not wrapped; sin/cos of it
    # are all that is used.
    mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    # Eccentricity of Earth's orbit (dimensionless).
    eccent = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    anom = np.deg2rad(mean_anom)
    # Equation of the centre (deg): true anomaly minus mean anomaly.
    centre = (
        np.sin(anom) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(2.0 * anom) * (0.019993 - 0.000101 * jc)
        + np.sin(3.0 * anom) * 0.000289
    )
    true_long = mean_long + centre

    # Longitude of the Moon's ascending node (deg): the nutation/aberration
    # term that turns the true longitude into the apparent one.
    omega = np.deg2rad(125.04 - 1934.136 * jc)
    app_long = true_long - 0.00569 - 0.00478 * np.sin(omega)

    # Mean obliquity of the ecliptic (deg), written as NOAA writes it --
    # 23 deg 26' 21.448" minus the secular terms -- then nutation-corrected.
    mean_obliq = (
        23.0
        + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0)
        / 60.0
    )
    obliq = np.deg2rad(mean_obliq + 0.00256 * np.cos(omega))

    declination = np.rad2deg(np.arcsin(np.sin(obliq) * np.sin(np.deg2rad(app_long))))

    # Equation of time, Meeus eq. 28.3: y = tan^2(eps/2).
    y = np.tan(obliq / 2.0) ** 2
    lam = np.deg2rad(mean_long)
    eot = 4.0 * np.rad2deg(
        y * np.sin(2.0 * lam)
        - 2.0 * eccent * np.sin(anom)
        + 4.0 * eccent * y * np.sin(anom) * np.cos(2.0 * lam)
        - 0.5 * y * y * np.sin(4.0 * lam)
        - 1.25 * eccent * eccent * np.sin(2.0 * anom)
    )
    return declination, eot


# ---------------------------------------------------------------- public API -


def solar_declination(time: Any) -> Any:
    """Apparent solar declination in degrees, positive north.

    Parameters
    ----------
    time : datetime-like or str or numpy.datetime64 or array_like
        UTC instant(s); see :func:`solar_zenith` for the accepted forms.

    Returns
    -------
    numpy.ndarray or numpy.float64
        Declination in degrees, in about ``[-23.45, +23.45]``. Scalar for
        scalar input.
    """
    jc, _ = _julian_century_and_utc_minutes(time)
    declination, _ = _sun_declination_and_eot(jc)
    return _out(declination)


def equation_of_time(time: Any) -> Any:
    """Equation of time in minutes: apparent solar time minus mean solar time.

    Parameters
    ----------
    time : datetime-like or str or numpy.datetime64 or array_like
        UTC instant(s); see :func:`solar_zenith` for the accepted forms.

    Returns
    -------
    numpy.ndarray or numpy.float64
        Minutes, in about ``[-14.3, +16.4]`` over the year. Add it to the
        UTC clock (and to ``4 * longitude``) to get true solar time. Scalar for
        scalar input.
    """
    jc, _ = _julian_century_and_utc_minutes(time)
    _, eot = _sun_declination_and_eot(jc)
    return _out(eot)


def solar_position(time: Any, lat: Any, lon: Any) -> tuple[Any, Any]:
    """Geometric solar zenith and azimuth, in degrees.

    Parameters
    ----------
    time : datetime-like or str or numpy.datetime64 or array_like
        UTC instant(s). Accepted forms, scalar or array/sequence: a
        :class:`datetime.datetime` (naive is read as UTC, aware is converted),
        an ISO-8601 string, a :class:`numpy.datetime64`, or a
        :class:`datetime.date` (midnight UTC). ``NaT`` propagates as ``nan``.
    lat : array_like
        Geodetic latitude in degrees, positive north, in ``[-90, 90]``.
    lon : array_like
        Longitude in degrees, positive east. Any branch works -- ``-170``,
        ``190`` and ``550`` describe the same meridian and give the same
        answer, because the longitude enters only through a true-solar-time
        that is reduced modulo a day.
    time, lat, lon
        Broadcast against one another by the usual NumPy rules.

    Returns
    -------
    zenith : numpy.ndarray or numpy.float64
        Angle from the local zenith to the centre of the Sun, degrees, in
        ``[0, 180]``. Greater than 90 means the Sun is below the horizon; that
        is **not** clamped (see the module docstring).
    azimuth : numpy.ndarray or numpy.float64
        Bearing of the Sun measured clockwise from true north, degrees, in
        ``[0, 360)``. Undefined, and returned as an arbitrary value, exactly at
        the geographic poles.

    Notes
    -----
    Both are geometric: no atmospheric refraction, no correction for the solar
    disc's finite radius, and the Sun's centre rather than its upper limb.
    """
    jc, minutes = _julian_century_and_utc_minutes(time)
    declination, eot = _sun_declination_and_eot(jc)

    latitude = np.asarray(lat, dtype=np.float64)
    longitude = np.asarray(lon, dtype=np.float64)

    # True solar time in minutes since local solar midnight. The modulo is what
    # makes the longitude branch irrelevant: 4 minutes per degree means a 360
    # deg shift is exactly one 1440-minute day.
    true_solar_time = np.mod(minutes + eot + 4.0 * longitude, _MINUTES_PER_DAY)
    hour_angle = np.deg2rad(true_solar_time * _DEG_PER_MINUTE - 180.0)

    phi = np.deg2rad(latitude)
    dec = np.deg2rad(declination)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)

    cos_zenith = sin_phi * sin_dec + cos_phi * cos_dec * np.cos(hour_angle)
    # The clip guards arccos against |cos| = 1 + 1e-16 at the exact subsolar
    # point; it is float64 hygiene, not a horizon clamp.
    zenith = np.rad2deg(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))

    # Azimuth clockwise from north. Both arctan2 arguments carry a factor
    # cos(phi) >= 0, which cancels in the ratio but keeps the expression finite
    # at the poles (where azimuth is meaningless anyway).
    azimuth = np.mod(
        np.rad2deg(
            np.arctan2(
                -cos_dec * np.sin(hour_angle) * cos_phi,
                sin_dec - sin_phi * cos_zenith,
            )
        ),
        360.0,
    )
    return _out(zenith), _out(azimuth)


def solar_zenith(time: Any, lat: Any, lon: Any) -> Any:
    """Geometric (unrefracted) solar zenith angle in degrees.

    The one function the retrieval pipeline needs: it turns an observation's
    UTC timestamp and position into the ``theta_s`` of
    :class:`Geometry <robust.rt.types.Geometry>`.

    Parameters
    ----------
    time : datetime-like or str or numpy.datetime64 or array_like
        UTC instant(s). Accepted forms, scalar or array/sequence: a
        :class:`datetime.datetime` (naive is read as UTC, aware is converted to
        UTC), an ISO-8601 string such as ``"2024-05-14T19:37:02"`` or
        ``"2024-05-14T15:37:02-04:00"``, a :class:`numpy.datetime64`, or a
        :class:`datetime.date` (taken at midnight UTC). ``NaT`` gives ``nan``.
    lat : array_like
        Geodetic latitude in degrees, positive north.
    lon : array_like
        Longitude in degrees, positive east; any branch (``[-180, 180)``,
        ``[0, 360)``, or beyond) gives the same answer.
    time, lat, lon
        Broadcast against one another by the usual NumPy rules, so an ``(N,)``
        track of times with scalar coordinates, or one instant over an ``(M,)``
        grid of latitudes, both work.

    Returns
    -------
    numpy.ndarray or numpy.float64
        Solar zenith angle in degrees, float64, with the broadcast shape of the
        inputs -- collapsed to a :class:`numpy.float64` scalar when every input
        is a scalar, following the convention of NumPy's own ufuncs. Values
        exceed 90 deg whenever the Sun is below the horizon and are **never**
        clamped: mid-December at Utqiagvik returns ~95--100 deg, which is the
        correct statement that there is no sun.

    See Also
    --------
    solar_position : the same computation, also returning the azimuth.

    Notes
    -----
    Accuracy is ~0.01 deg against a full ephemeris over 1900--2100 (see the
    module docstring for the algorithm and its provenance). No atmospheric
    refraction is applied, so near the horizon this differs from the *apparent*
    position of the Sun by up to ~0.6 deg -- which is the right choice here,
    because the surface and in-water refraction the RT model applies must not
    be double-counted.

    Examples
    --------
    Santa Cruz, California at local solar noon on the June 2024 solstice:

    >>> import numpy as np
    >>> from robust import solar
    >>> round(float(solar.solar_zenith("2024-06-20T20:10:00", 36.9741, -122.0308)), 2)
    13.54

    Times broadcast against scalar coordinates:

    >>> solar.solar_zenith(
    ...     np.array(["2024-06-20T12:00:00", "2024-06-20T18:00:00"], "datetime64[s]"),
    ...     36.9741,
    ...     -122.0308,
    ... ).shape
    (2,)
    """
    zenith, _ = solar_position(time, lat, lon)
    return zenith
