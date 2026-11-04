"""
Pitarch & Brando (2024) multi-angular reference batches -- "PB24".

The dataset M5 was built on, rebuilt for M8 (task 3, ``rt_fit_pb24.md``) on
the consolidation caches and in **above-water geometry**.

**What it is.** 5000 HydroLight realisations (Pitarch & Brando 2025; Zenodo
10.5281/zenodo.11637178) at ``$OS_COLOR/SD/v5``, each a netCDF in two spectral
resolutions -- ``SD_hyp_no_R_NNNN.nc`` (451 bands, 350-800 nm at 1 nm) and
``SD_OLCI_no_R_NNNN.nc`` (12 OLCI bands, 400-753 nm; separate HydroLight runs on
the same realisations, not a band subset) -- on a shared ``theta_s`` (10) x
``theta`` (10) x ``phi`` (13) grid. Each file carries IOP components, ``rrs``,
``Rrs`` and ``Q`` on the 4-D grid, eleven surface AOPs and ``Esdir_Es_ratio``
on ``(lambda, theta_s)``. A ``.mat`` sidecar labels every realisation with one
of 12 optical water classes.

**Where the numbers come from.** ``resolution="hyp"`` (the default) is served
from the memory-mapped cache of :mod:`robust.rt.data.pb24_cache`
(``cache_dir=`` overrides its location; the committed miniature
``robust/tests/files/pb24_cache_small.zip`` feeds the real loader in CI).
``resolution="olci"`` reads one file per realisation through
:func:`ocpy.hydrolight.pb24.load_ds` (the OLCI set is not cached). Both go
through the same assembly code, so a sample means the same thing in both.
``reader=`` replaces the per-file reader (and forces the per-file path), which
is how a test feeds the OLCI path without the data.

**Directions (Q9 of ``rt_fit_pb24.md``).** PB24 tabulates ``Rrs(theta)`` at the
**above-water** zenith and ``rrs(theta)`` and ``Q(theta)`` at the **in-water**
zenith (the ``rrs`` variable's ``comment`` says so), on the same nominal nodes
0, 10, ..., 80, 87.5. ``theta`` and ``phi`` name the radiance *propagation*
direction; ``phi = 0`` is towards the sun. The two are paired by Snell's law,
``theta_w = asin(sin(theta_air) / 1.34)``:

- **A sample is an above-water geometry** ``(realisation, theta_s, theta_v_air,
  dphi)``. ``Rrs`` is read at its node. ``rrs`` and ``Q`` are interpolated
  **linearly in theta** between the in-water nodes to ``theta_w``
  (:func:`interp_in_water`); every ``theta_w`` lies in [0, 48.2] degrees, so
  only the nodes 0-50 are used. ``Geometry.theta_v`` is the above-water sensor
  zenith; :attr:`PB24Batch.theta_w` carries the in-water one.
- The interpolation error is **measured**, not assumed: linear against a cubic
  spline (:func:`interpolation_error`), and by the **Fresnel check**
  (:class:`FresnelReport`, attached to every load): ``Rrs(theta_air) /
  rrs(theta_w)``, normalised at nadir, against the flat-surface Fresnel
  transmittance ``T(theta_air) / T(0)``. Within 5% for ``theta_air <= 70``.
- The in-water nodes beyond the critical angle (50-87.5 degrees) reach no
  above-water direction. They form the ``"in_water_only"`` angle set, for ZTT
  physics only: un-interpolated, ``Rrs`` absent (all NaN), and there
  ``Geometry.theta_v`` **holds the in-water angle**.

*Dated note (2026-11-03, M8 task 3).* M5's loader paired ``rrs`` and ``Rrs`` at
the same nominal ``theta`` and called it the sensor zenith. That pairing was
wrong (record §7.10's "``rrs x Q = R``" test cannot distinguish directions,
because ``Q`` is tabulated per in-water direction too) and is superseded by the
Snell pairing above. Every M5 number computed on PB24 samples is on the old
pairing.

**The angle window, in air.** JXP's Q14 answer (M5) set the sanctioned envelope
at 0-70 degrees; under Q9d it is ``theta_s <= 70`` and ``theta_v_air <= 70``
(``theta_w <= 44.5``): :data:`ANGLE_WINDOW_MAX` keeps its value and gains the
air-angle meaning. ``"window"`` is 8 x 8 x 13 = 832 geometries per realisation,
``"shell"`` its complement in air (468), ``"all"`` the 1300, ``"in_water_only"``
10 x 5 x 13 = 650 in-water nodes.

**What else a batch carries.**

- ``IOPs``: ``a = aw + aph + ag + aNAP``, ``bb_w = bbw`` (the file's own water),
  ``bb_p = bbph + bbNAP``, ``a_ph = aph``.
- ``PhaseParams``: ``B_p = (bbph + bbNAP) / (bph + bNAP)`` as a spectrum, and
  PB24's own two-component Fournier-Forand mixture (Pitarch & Brando 2025
  §1.1): ``ff_B = (bbph/bph, bbNAP/bNAP)`` and ``ff_weight = (bph, bNAP) /
  (bph + bNAP)``, component axis last, shape ``(n_sample, n_wave, 2)`` as L23's.
  16 realisations have ``bNAP = 0`` at some or all bands (15 everywhere); their
  NAP weight is 0 and their NAP ``ff_B`` is the realisation's own value at its
  other bands, or :data:`FF_B_NAP_FILL` if it has none.
- ``Geometry.diffuse_fraction = 1 - Esdir_Es_ratio`` at the sample's
  ``theta_s``, ``(n_sample, n_wave)``. Nothing consumes it until task 4.
- ``aops``: ``Q`` always, plus any of :data:`EXTRA_FIELDS` gathered at each
  sample's ``theta_s``.

**Measured properties** (M5 task 0 on the OLCI set; M8 task 3 on the cache):

- The particle backscatter ratio varies per realisation: ``bbph/bph`` is flat
  in wavelength within a file (max/min <= 1.006) and spans 0.0010-0.050 across
  the 5000 hyperspectral files; ``bbNAP/bNAP`` spans 0.0100-0.0200. Bulk
  ``B_p`` spans 0.00115-0.0430 (per-realisation means 0.00144-0.0430, 30x).
- Bulk ``B_p`` is **not** flat in wavelength, because the phytoplankton/NAP mix
  shifts across the spectrum. Carried as a spectrum, like L23's.
- ``C``, ``N``, ``Y`` are **labels, not generators** (record §7.2).
- ``rrs`` is **exactly 0** only at the in-water node ``theta = 87.5`` (36,680
  of the hyperspectral cache's values, 1350 realisations, ``theta_s >= 40``,
  float32 underflow). No above-water sample interpolates from that node, so the
  air sets hold **no** zero ``rrs``; only ``"in_water_only"`` does. See
  ``drop_zero_rrs``.

**Layout: one flat sample axis**, as in :mod:`robust.rt.data.l23`: a sample is
one geometry of one realisation carrying an ``n_wave`` spectrum, so every leaf
shares the batch shape and ``jax.vmap(f, in_axes=0)`` works. Per-realisation
fields are duplicated across geometries; subsampling is an explicit argument.

**Nothing here is a hidden sample.** Every load returns a :class:`LoadReport`
saying how many realisations and geometries were kept out of how many, how
many samples the zero-``rrs`` filter removed, and the Fresnel check on every
realisation read.

**Splits** are fixed once over all 5000 realisations by
``design/py/make_pb24_splits.py`` and saved to ``design/ztt_pb24/splits.json``
by realisation id (Q14); :func:`make_splits` loads that file and maps it onto
a batch.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .. import conventions
from ..types import Geometry, IOPs, PhaseParams
from ..ztt import REFRACTIVE_INDEX

__all__ = [  # noqa: RUF022  - grouped by role
    # Dataset constants
    "SUBDIR",
    "N_REALISATION",
    "RESOLUTIONS",
    "DEFAULT_RESOLUTION",
    "HYP_WAVE",
    "CLASSES_FILE",
    "N_CLASS",
    "ANGLE_WINDOW_MAX",
    "ANGLE_MODES",
    "AIR_ANGLE_MODES",
    "CRITICAL_ANGLE",
    "IN_WATER_ONLY_MIN",
    "B_P_EXPECTED",
    "FF_B_NAP_FILL",
    "FRESNEL_TOLERANCE",
    "FRESNEL_QUANTILES",
    # Split configuration
    "SPLIT_SEED",
    "TEST_FRACTION",
    "BP_BAND_QUANTILES",
    "GEOMETRY_HOLDOUT_MIN",
    "SPLIT_KINDS",
    "DEFAULT_SPLIT_KINDS",
    "CONFOUND_SEEDS",
    "SPLITS_VERSION",
    "DEFAULT_SPLITS_FILE",
    # Containers
    "PB24Batch",
    "LoadReport",
    "FresnelReport",
    "InterpolationReport",
    "Splits",
    "SplitReport",
    # Geometry helpers
    "theta_air_to_water",
    "interp_in_water",
    # Functions
    "load_batch",
    "interpolation_error",
    "make_splits",
    "build_splits",
    "write_splits",
    "read_splits",
    "confound_reference",
    "select",
    "fit_transfer",
    "data_dir",
    "file_path",
    "read_classes",
    # Field lists
    "EXTRA_FIELDS",
    "DEFAULT_EXTRAS",
]

#: Location of the release under ``$OS_COLOR``.
SUBDIR = ("SD", "v5")

#: Realisations in the release, indexed 1..5000 in the file names.
N_REALISATION = 5000

#: Spectral resolutions, mapped to their file-name prefix (as in
#: :data:`ocpy.hydrolight.pb24.RESOLUTIONS`).
RESOLUTIONS = {
    "hyp": "SD_hyp_no_R_",
    "olci": "SD_OLCI_no_R_",
}

#: The default resolution: hyperspectral, from the cache (JXP's Q1/Q5).
DEFAULT_RESOLUTION = "hyp"

#: The hyperspectral grid, 350-800 nm at 1 nm.
HYP_WAVE = np.arange(350.0, 801.0, 1.0)

#: The optical-water-class sidecar, and how many classes it defines.
CLASSES_FILE = "classes_Rrs_OLCI_v5_20240214.mat"
N_CLASS = 12

#: float: The sanctioned window, in **above-water** angles (Q14, Q9d):
#: ``theta_s <= 70`` and ``theta_v_air <= 70``.
ANGLE_WINDOW_MAX = 70.0

#: Angle-selection modes for :func:`load_batch`.
ANGLE_MODES = ("window", "shell", "all", "in_water_only")

#: The modes whose samples are above-water geometries.
AIR_ANGLE_MODES = ("window", "shell", "all")

#: float: The critical angle for n = 1.34, degrees (48.27): in-water directions
#: beyond it reach no above-water direction.
CRITICAL_ANGLE = float(np.degrees(np.arcsin(1.0 / REFRACTIVE_INDEX)))

#: float: The smallest in-water node of ``"in_water_only"`` -- the first node
#: beyond :data:`CRITICAL_ANGLE`.
IN_WATER_ONLY_MIN = 50.0

#: The advisory range for ``B_p`` on **this** dataset. Widened at M8 task 3 from
#: M5's (0.0009, 0.04) to cover the hyperspectral set, whose bulk ``B_p``
#: reaches 0.0430. The loader warns outside it and never clips.
B_P_EXPECTED = (0.0009, 0.045)

#: ``ff_B`` for the NAP component where ``bNAP = 0`` and the realisation has no
#: other band to take it from: the midpoint of PB24's design range
#: U(0.01, 0.02). Its weight there is 0, so the value is inert.
FF_B_NAP_FILL = 0.015

#: float: The Fresnel check's tolerance (task 3 gate): ``|ratio - 1| <= 5%`` for
#: ``theta_v_air <= ANGLE_WINDOW_MAX``.
FRESNEL_TOLERANCE = 0.05

#: Quantiles :class:`FresnelReport` and :class:`InterpolationReport` record.
FRESNEL_QUANTILES = (0.0, 0.01, 0.5, 0.99, 1.0)

#: Seed for the random split. Arbitrary but fixed forever.
SPLIT_SEED = 23

#: Fraction of *realisations* held out by the random split.
TEST_FRACTION = 0.2

#: Quantile band of per-realisation mean ``B_p`` held out by ``bp_band``: an
#: **interior** band (interpolation across phase functions, not extrapolation).
BP_BAND_QUANTILES = (0.4, 0.6)

#: float: The geometry split holds out ``theta_v_air >= 80`` or
#: ``theta_s >= 80`` (Q14) -- a rule, not a list.
GEOMETRY_HOLDOUT_MIN = 80.0

#: The splits M5 defined and M8 keeps.
SPLIT_KINDS = ("realisation", "bp_band", "geometry")

#: Seeds :func:`confound_reference` averages over by default.
CONFOUND_SEEDS = tuple(range(12))

#: What :func:`make_splits` builds unless told otherwise -- the two a default
#: (window) load can support.
DEFAULT_SPLIT_KINDS = ("realisation", "bp_band")

#: Version of the splits file layout.
SPLITS_VERSION = 1

#: The committed splits file (``design/make_pb24_splits.py`` writes it).
DEFAULT_SPLITS_FILE = (
    Path(__file__).resolve().parents[3] / "design" / "ztt_pb24" / "splits.json"
)

#: Fields that are per ``(theta_s, wave)`` and become per-sample spectra by
#: indexing each sample's solar zenith.
EXTRA_FIELDS = (
    "mu_d",
    "mu_u",
    "mu_tot",
    "Kd",
    "Ku",
    "Ko",
    "Kod",
    "Kou",
    "Knet",
    "KLu",
    "R",
    "Esdir_Es_ratio",
)

#: What :func:`load_batch` materialises by default; ``Q`` is always carried.
DEFAULT_EXTRAS = ("mu_d", "mu_u", "mu_tot")

#: Per-realisation scalars, IOP spectra and (theta_s, wave) AOPs the loader reads.
_SCALARS = ("C", "N", "Y")
_IOPS = ("aw", "aph", "ag", "aNAP", "bw", "bph", "bNAP", "bbw", "bbph", "bbNAP")
_AOPS = EXTRA_FIELDS
_FOUR_D = ("rrs", "Rrs", "Q")

#: Realisations per cache read in :func:`load_batch`.
_CHUNK = 32


# ------------------------------------------------------------ geometry ------


def theta_air_to_water(theta_air):
    """In-water zenith of an above-water direction, by Snell's law.

    Parameters
    ----------
    theta_air : array_like
        Above-water zenith, degrees, in [0, 90].

    Returns
    -------
    numpy.ndarray
        ``asin(sin(theta_air) / 1.34)`` in degrees (float64).
    """
    theta_air = np.asarray(theta_air, dtype=float)
    return np.degrees(np.arcsin(np.sin(np.radians(theta_air)) / REFRACTIVE_INDEX))


def interp_in_water(field4d, theta_nodes, theta_w, *, axis: int = -3) -> np.ndarray:
    """Interpolate an in-water field **linearly in theta** to in-water angles.

    The interpolation the loader applies to ``rrs`` and ``Q`` (Q9b), exposed so
    a test can recompute it by hand: for each ``theta_w`` the bracketing nodes
    ``theta_nodes[i] <= theta_w <= theta_nodes[i + 1]`` and
    ``t = (theta_w - theta_nodes[i]) / (theta_nodes[i + 1] - theta_nodes[i])``
    give ``(1 - t) * field[i] + t * field[i + 1]``. At a node, ``t = 0`` and the
    node's value is returned exactly.

    Parameters
    ----------
    field4d : array_like
        Field with the in-water zenith on ``axis`` (the cache layout
        ``(..., theta_s, theta, phi, wave)`` has it at -3, the default).
    theta_nodes : array_like
        The in-water zenith nodes, ascending, degrees.
    theta_w : array_like
        1-D in-water zeniths to interpolate to, within the nodes' span.
    axis : int, optional
        The zenith axis of ``field4d``.

    Returns
    -------
    numpy.ndarray
        float64, ``field4d``'s shape with ``axis`` of length ``len(theta_w)``.

    Raises
    ------
    ValueError
        If the nodes are not ascending, do not match the axis, or a
        ``theta_w`` lies outside them (no extrapolation).
    """
    field4d = np.asarray(field4d, dtype=float)
    nodes = np.asarray(theta_nodes, dtype=float)
    theta_w = np.atleast_1d(np.asarray(theta_w, dtype=float))
    axis = axis % field4d.ndim
    if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0):
        raise ValueError("interp_in_water: theta_nodes must be 1-D and ascending")
    if field4d.shape[axis] != nodes.size:
        raise ValueError(
            f"interp_in_water: axis {axis} has {field4d.shape[axis]} values, "
            f"theta_nodes {nodes.size}"
        )
    if theta_w.ndim != 1 or np.any(theta_w < nodes[0]) or np.any(theta_w > nodes[-1]):
        raise ValueError(
            f"interp_in_water: theta_w must be 1-D within [{nodes[0]}, {nodes[-1]}]"
        )
    lo = np.clip(np.searchsorted(nodes, theta_w, side="right") - 1, 0, nodes.size - 2)
    t = (theta_w - nodes[lo]) / (nodes[lo + 1] - nodes[lo])
    shape = [1] * field4d.ndim
    shape[axis] = theta_w.size
    t = t.reshape(shape)
    below = np.take(field4d, lo, axis=axis)
    above = np.take(field4d, lo + 1, axis=axis)
    return below * (1.0 - t) + above * t


def _fresnel_T(theta_air):
    """Unpolarised flat-surface Fresnel transmittance, air to water, n = 1.34.

    Private on purpose: task 4 adds the public
    ``conventions.fresnel_transmittance``.
    """
    ti = np.radians(np.asarray(theta_air, dtype=float))
    tt = np.arcsin(np.sin(ti) / REFRACTIVE_INDEX)
    n = REFRACTIVE_INDEX
    rs = (np.cos(ti) - n * np.cos(tt)) / (np.cos(ti) + n * np.cos(tt))
    rp = (n * np.cos(ti) - np.cos(tt)) / (n * np.cos(ti) + np.cos(tt))
    return 1.0 - 0.5 * (rs**2 + rp**2)


class _Histogram:
    """Fixed-bin histograms per group, with exact min/max -- streaming quantiles.

    Quantiles are bin centres, so they are exact to half a bin width; the ends
    (q = 0 and 1) are the exact extremes.
    """

    def __init__(self, n_group: int, lo: float, hi: float, width: float):
        self.lo, self.width = lo, width
        self.n_bin = int(round((hi - lo) / width))
        self.counts = np.zeros((n_group, self.n_bin), dtype=np.int64)
        self.mins = np.full(n_group, np.inf)
        self.maxs = np.full(n_group, -np.inf)

    def add(self, group: int, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        if not values.size:
            return
        idx = np.clip(((values - self.lo) / self.width).astype(np.int64), 0, None)
        idx = np.minimum(idx, self.n_bin - 1)
        self.counts[group] += np.bincount(idx, minlength=self.n_bin)
        self.mins[group] = min(self.mins[group], float(values.min()))
        self.maxs[group] = max(self.maxs[group], float(values.max()))

    def quantiles(self, groups, qs) -> tuple[float, ...]:
        groups = np.atleast_1d(groups)
        counts = self.counts[groups].sum(axis=0)
        total = int(counts.sum())
        if total == 0:
            return tuple(float("nan") for _ in qs)
        cdf = np.cumsum(counts)
        out = []
        for q in qs:
            if q <= 0.0:
                out.append(float(self.mins[groups].min()))
            elif q >= 1.0:
                out.append(float(self.maxs[groups].max()))
            else:
                i = int(np.searchsorted(cdf, q * total))
                out.append(self.lo + (i + 0.5) * self.width)
        return tuple(out)

    def n(self, groups) -> int:
        return int(self.counts[np.atleast_1d(groups)].sum())


@dataclass(frozen=True)
class FresnelReport:
    """The Fresnel check of Q9: is the Snell pairing surface physics?

    For every realisation read, every ``theta_s``, ``dphi`` and band, the ratio
    ``Rrs(theta_air) / rrs(theta_w)`` (``rrs`` interpolated by
    :func:`interp_in_water`), normalised by the same ratio at nadir, is compared
    with the flat-surface Fresnel transmittance ``T(theta_air) / T(0)``
    (unpolarised, n = 1.34). ``deviation = (normalised ratio) / (T/T0) - 1``.

    Attributes
    ----------
    theta_v_air : tuple of float
        The above-water view nodes (nadir excluded: it is 0 by construction).
    expected : tuple of float
        ``T(theta_air) / T(0)`` at each node.
    n_value : tuple of int
        Values compared at each node.
    deviation : tuple of tuple of float
        Per node, the :data:`FRESNEL_QUANTILES` of ``deviation`` (q = 0 and 1
        exact; the rest to 5e-5, the half-width of the histogram bins).
    max_abs : tuple of float
        Per node, ``max |deviation|`` (exact).
    max_abs_by_theta_s : dict
        ``{theta_s: max |deviation| over theta_v_air <= ANGLE_WINDOW_MAX}``.
    n_realisation : int
        Realisations the check ran on.
    n_excluded : int
        Values skipped because a ratio was not finite (a zero ``rrs``).
    n_beyond : int
        Values with ``|deviation| > tolerance`` at ``theta_v_air <=
        ANGLE_WINDOW_MAX``.
    realisations_beyond : tuple of int
        The realisations those values belong to. On the full release: 5 values
        of 2.05e9, all realisation 4492 (``theta_v_air = 70``, 356-376 nm,
        nadir ``rrs`` ~3e-6 sr^-1), worst -5.30%.
    tolerance : float
        :data:`FRESNEL_TOLERANCE`.
    """

    theta_v_air: tuple[float, ...]
    expected: tuple[float, ...]
    n_value: tuple[int, ...]
    deviation: tuple[tuple[float, ...], ...]
    max_abs: tuple[float, ...]
    max_abs_by_theta_s: dict[float, float]
    n_realisation: int
    n_excluded: int
    n_beyond: int = 0
    realisations_beyond: tuple[int, ...] = ()
    tolerance: float = FRESNEL_TOLERANCE
    quantiles: tuple[float, ...] = FRESNEL_QUANTILES

    def worst(self, theta_max: float = ANGLE_WINDOW_MAX) -> float:
        """``max |deviation|`` over ``theta_v_air <= theta_max``."""
        return max(
            (
                m
                for t, m in zip(self.theta_v_air, self.max_abs, strict=True)
                if t <= theta_max
            ),
            default=float("nan"),
        )

    def within(self, theta_max: float = ANGLE_WINDOW_MAX) -> bool:
        """Whether every deviation at ``theta_v_air <= theta_max`` is within
        :attr:`tolerance`."""
        return bool(self.worst(theta_max) <= self.tolerance)

    def table(self) -> str:
        """A markdown table, one row per view node."""
        head = " | ".join(f"q{q:g}" for q in self.quantiles)
        lines = [
            f"| theta_v_air | T/T0 | n | {head} | max abs |",
            "|---" * (4 + len(self.quantiles)) + "|",
        ]
        for t, e, n, d, m in zip(
            self.theta_v_air,
            self.expected,
            self.n_value,
            self.deviation,
            self.max_abs,
            strict=True,
        ):
            vals = " | ".join(f"{v * 100:+.2f}%" for v in d)
            lines.append(f"| {t:g} | {e:.4f} | {n} | {vals} | {m * 100:.2f}% |")
        return "\n".join(lines)


class _FresnelAccumulator:
    """Streams the Fresnel check over chunks of realisations."""

    def __init__(self, theta_s: np.ndarray, theta: np.ndarray):
        self.theta_s = np.asarray(theta_s, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.hist = _Histogram(self.theta_s.size * self.theta.size, -1.0, 3.0, 1e-4)
        self.expected = _fresnel_T(self.theta) / _fresnel_T(0.0)
        self.n_realisation = 0
        self.n_excluded = 0
        self.n_beyond = 0
        self.beyond: set[int] = set()

    def add(self, Rrs4: np.ndarray, rrs_air4: np.ndarray, ids) -> None:
        """``Rrs4`` and ``rrs_air4``: ``(n, theta_s, theta_air, phi, wave)``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = Rrs4 / rrs_air4
            norm = ratio / ratio[:, :, :1]
            dev = norm / self.expected[None, None, :, None, None] - 1.0
        self.n_excluded += int((~np.isfinite(dev[:, :, 1:])).sum())
        window = self.theta <= ANGLE_WINDOW_MAX
        with np.errstate(invalid="ignore"):
            over = np.abs(dev[:, :, window]) > FRESNEL_TOLERANCE
        if over.any():
            self.n_beyond += int(over.sum())
            hit = over.reshape(over.shape[0], -1).any(axis=1)
            self.beyond.update(int(i) for i in np.asarray(ids)[hit])
        for i_s in range(self.theta_s.size):
            for i_t in range(1, self.theta.size):
                self.hist.add(i_s * self.theta.size + i_t, dev[:, i_s, i_t].ravel())
        self.n_realisation += Rrs4.shape[0]

    def report(self) -> FresnelReport:
        n_t = self.theta.size
        nodes = range(1, n_t)
        per_node = [
            [i_s * n_t + i_t for i_s in range(self.theta_s.size)] for i_t in nodes
        ]
        max_abs = tuple(
            float(max(abs(self.hist.mins[g].min()), abs(self.hist.maxs[g].max())))
            for g in per_node
        )
        window_t = [i_t for i_t in nodes if self.theta[i_t] <= ANGLE_WINDOW_MAX]
        by_s = {}
        for i_s, ts in enumerate(self.theta_s):
            groups = [i_s * n_t + i_t for i_t in window_t]
            by_s[float(ts)] = float(
                max(
                    abs(self.hist.mins[groups].min()), abs(self.hist.maxs[groups].max())
                )
            )
        return FresnelReport(
            theta_v_air=tuple(float(self.theta[i]) for i in nodes),
            expected=tuple(float(self.expected[i]) for i in nodes),
            n_value=tuple(self.hist.n(g) for g in per_node),
            deviation=tuple(
                self.hist.quantiles(g, FRESNEL_QUANTILES) for g in per_node
            ),
            max_abs=max_abs,
            max_abs_by_theta_s=by_s,
            n_realisation=self.n_realisation,
            n_excluded=self.n_excluded,
            n_beyond=self.n_beyond,
            realisations_beyond=tuple(sorted(self.beyond)),
        )


# ------------------------------------------------------------- reports ------


@dataclass(frozen=True)
class LoadReport:
    """What a load kept, and what it dropped.

    Returned with every batch so a subsample is visible at the call site
    instead of implied.

    Attributes
    ----------
    n_realisation_available, n_realisation : int
        Realisations in the source and in the batch.
    n_geometry_available, n_geometry : int
        Geometries per realisation before and after the angle mode and stride
        (1300 available in either grid: air for the air modes, in-water for
        ``"in_water_only"``).
    angles : str
        Which angle mode was applied (:data:`ANGLE_MODES`).
    geometry_stride : int or tuple of int
        Stride applied within the selected geometries; 1 means none.
    n_dropped_angle : int
        Geometries removed by the angle mode alone.
    n_dropped_stride : int
        Geometries removed by the stride alone.
    n_dropped_zero_rrs : int
        Samples removed because their spectrum contained a zero ``rrs``.
    n_zero_bands : int
        ``(sample, band)`` values of ``rrs`` that were zero, before any drop.
    coverage : dict
        ``{axis: (kept, available)}`` distinct angle values per axis, after the
        stride (a flat stride can alias against the 13 azimuths).
    resolution : str
        ``"hyp"`` or ``"olci"``.
    fresnel : FresnelReport or None
        The Fresnel check over every realisation read (all geometries, not just
        the selected ones); ``None`` if ``check_fresnel=False``.
    """

    n_realisation_available: int
    n_realisation: int
    n_geometry_available: int
    n_geometry: int
    angles: str
    geometry_stride: int
    n_dropped_angle: int
    n_dropped_stride: int
    n_dropped_zero_rrs: int
    n_zero_bands: int
    coverage: dict[str, tuple[int, int]] = field(default_factory=dict)
    resolution: str = DEFAULT_RESOLUTION
    fresnel: FresnelReport | None = None

    @property
    def aliased_axes(self) -> tuple[str, ...]:
        """Angle axes the stride lost values from -- empty is the healthy case."""
        return tuple(
            axis for axis, (kept, avail) in self.coverage.items() if kept < avail
        )

    def summary(self) -> str:
        """One line, for a log or a notebook cell."""
        aliased = self.aliased_axes
        tail = f"; ALIASED on {list(aliased)}" if aliased else ""
        if self.fresnel is not None:
            tail += (
                f"; Fresnel check max |dev| {self.fresnel.worst() * 100:.2f}% at "
                f"theta_v_air <= {ANGLE_WINDOW_MAX:g}"
            )
        return (
            f"PB24 ({self.resolution}): {self.n_realisation}/"
            f"{self.n_realisation_available} realisations x "
            f"{self.n_geometry}/{self.n_geometry_available} geometries "
            f"(angles={self.angles}, stride={self.geometry_stride}); dropped "
            f"{self.n_dropped_angle} by angle, {self.n_dropped_stride} by stride, "
            f"{self.n_dropped_zero_rrs} samples for zero rrs "
            f"({self.n_zero_bands} zero bands){tail}"
        )


@dataclass(frozen=True)
class PB24Batch:
    """A stacked PB24 batch: model inputs, both reflectances, and labels.

    Not a registered pytree -- an analysis container; ``realisation``,
    ``theta_w`` and ``water_class`` are host-side metadata. The model inputs
    it holds are pytrees.

    Attributes
    ----------
    iops : IOPs
        ``(n_sample, n_wave)`` per field, ``a_ph`` included. ``bb_w`` is the
        file's own ``bbw``, not :func:`robust.rt.conventions.bb_w`.
    phase_params : PhaseParams
        ``B_p`` as a spectrum; ``ff_B``/``ff_weight`` ``(n_sample, n_wave, 2)``,
        components (phytoplankton, NAP).
    geometry : Geometry
        Per-sample ``theta_s``, ``theta_v`` (above-water, except in an
        ``"in_water_only"`` batch, where it is the in-water angle), ``dphi`` in
        degrees, and ``diffuse_fraction`` ``(n_sample, n_wave)``.
    rrs, Rrs : Array
        ``rrs`` at ``theta_w`` (interpolated), ``Rrs`` at ``theta_v`` (a node),
        sr^-1. ``Rrs`` is all NaN in an ``"in_water_only"`` batch.
    wave : Array
        Wavelengths (nm).
    realisation : numpy.ndarray
        1-based realisation index per sample, matching the file names.
    labels : dict
        ``C``, ``N``, ``Y`` per sample, as NumPy (labels, not generators).
    water_class : numpy.ndarray or None
        Optical water class in ``[1, N_CLASS]`` per sample, if requested.
    aops : dict
        ``Q`` (at ``theta_w``, interpolated) always, plus the requested extras.
    report : LoadReport or None
        What this load kept and dropped.
    theta_w : numpy.ndarray or None
        Per-sample in-water view zenith, degrees: ``theta_air_to_water(theta_v)``
        for the air sets, the node itself for ``"in_water_only"``.
    """

    iops: IOPs
    phase_params: PhaseParams
    geometry: Geometry
    rrs: Float[Array, "sample wave"]
    Rrs: Float[Array, "sample wave"]
    wave: Float[Array, " wave"]
    realisation: np.ndarray
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    water_class: np.ndarray | None = None
    aops: dict[str, Float[Array, "sample wave"]] = field(default_factory=dict)
    report: LoadReport | None = None
    theta_w: np.ndarray | None = None

    @property
    def n_sample(self) -> int:
        """Number of samples (realisations x kept geometries)."""
        return int(self.rrs.shape[0])

    @property
    def n_wave(self) -> int:
        """Number of wavelength bands."""
        return int(self.rrs.shape[-1])

    @property
    def theta_s(self) -> np.ndarray:
        """Per-sample solar zenith, degrees, as NumPy."""
        return np.asarray(self.geometry.theta_s)

    @property
    def theta_v(self) -> np.ndarray:
        """Per-sample sensor zenith, degrees, as NumPy (see :attr:`geometry`)."""
        return np.asarray(self.geometry.theta_v)

    @property
    def dphi(self) -> np.ndarray:
        """Per-sample relative azimuth, degrees, as NumPy."""
        return np.asarray(self.geometry.dphi)

    @property
    def in_water_only(self) -> bool:
        """Whether this is an ``"in_water_only"`` batch (no ``Rrs``)."""
        return self.report is not None and self.report.angles == "in_water_only"

    def validate(self, *, grid=None) -> None:
        """Raise ``ValueError`` unless the batch is self-consistent and physical.

        Boundary check; not for use under ``jit``.

        Parameters
        ----------
        grid : None, str, or WaveGrid, optional
            Wavelength grid the batch claims to be on. ``None`` (default): the
            grid of the resolution it was loaded at (the OLCI grid, or the
            hyperspectral bands it holds).

        Raises
        ------
        ValueError
            On a shape mismatch, a bad wavelength grid, a non-physical IOP, an
            out-of-range angle, a negative reflectance, or an ``Rrs`` that is
            partly NaN.
        """
        if grid is None:
            resolution = self.report.resolution if self.report else DEFAULT_RESOLUTION
            grid = _wave_grid(resolution, np.asarray(self.wave))
        conventions.check_wave(self.wave, name="PB24Batch.wave", grid=grid)
        self.iops.validate(wave=self.wave, grid=grid)
        self.phase_params.validate()
        self.geometry.validate()
        conventions.check_rrs(self.rrs, name="PB24Batch.rrs", subsurface=True)
        Rrs = np.asarray(self.Rrs)
        if not np.isnan(Rrs).all():
            conventions.check_rrs(self.Rrs, name="PB24Batch.Rrs", subsurface=False)

        if self.rrs.shape != self.iops.a.shape:
            raise ValueError(
                f"PB24Batch: rrs {self.rrs.shape} does not match "
                f"IOPs.a {self.iops.a.shape}"
            )
        if self.Rrs.shape != self.rrs.shape:
            raise ValueError(
                f"PB24Batch: Rrs {self.Rrs.shape} does not match rrs {self.rrs.shape}"
            )
        df = self.geometry.diffuse_fraction
        if df is not None and np.shape(df) != self.rrs.shape:
            raise ValueError(
                f"PB24Batch: diffuse_fraction {np.shape(df)} does not match "
                f"rrs {self.rrs.shape}"
            )
        if self.realisation.shape != (self.n_sample,):
            raise ValueError(
                f"PB24Batch: realisation {self.realisation.shape} does not label "
                f"{self.n_sample} samples"
            )
        if self.theta_w is not None and self.theta_w.shape != (self.n_sample,):
            raise ValueError(
                f"PB24Batch: theta_w {self.theta_w.shape} does not label "
                f"{self.n_sample} samples"
            )
        for name, values in self.labels.items():
            if values.shape != (self.n_sample,):
                raise ValueError(
                    f"PB24Batch: labels[{name!r}] {values.shape} does not label "
                    f"{self.n_sample} samples"
                )
        if self.water_class is not None and self.water_class.shape != (self.n_sample,):
            raise ValueError(
                f"PB24Batch: water_class {self.water_class.shape} does not label "
                f"{self.n_sample} samples"
            )
        for name, values in self.aops.items():
            if values.shape != self.rrs.shape:
                raise ValueError(
                    f"PB24Batch: aops[{name!r}] {values.shape} does not match "
                    f"rrs {self.rrs.shape}"
                )


def _wave_grid(resolution: str, wave: np.ndarray) -> conventions.WaveGrid:
    """The :class:`~robust.rt.conventions.WaveGrid` a batch's bands must match.

    OLCI is the registered grid. The hyperspectral bands are checked to be a
    subset of :data:`HYP_WAVE` (the committed fixture keeps every 10th band)
    and returned as a local grid -- ``conventions`` is not touched (freeze).
    """
    if resolution == "olci":
        return conventions.OLCI_GRID
    wave = np.asarray(wave, dtype=float)
    on_grid = np.isin(np.round(wave, 6), HYP_WAVE)
    if wave.ndim != 1 or not on_grid.all() or np.any(np.diff(wave) <= 0):
        raise ValueError(
            "PB24: hyperspectral bands must be an ascending subset of 350-800 nm "
            "at 1 nm"
        )
    return conventions.WaveGrid(
        "pb24_hyp", HYP_WAVE[np.isin(HYP_WAVE, np.round(wave, 6))], "PB24 hyp bands"
    )


# --------------------------------------------------------------- paths ------


def data_dir() -> Path:
    """The release directory, from ``$OS_COLOR``.

    Raises
    ------
    RuntimeError
        If ``$OS_COLOR`` is unset, or the directory is missing.
    """
    root = os.environ.get("OS_COLOR")
    if not root:
        raise RuntimeError(
            "pb24: $OS_COLOR is not set, so the PB24 release cannot be located; "
            "pass cache_dir= (e.g. the unpacked test fixture) instead"
        )
    path = Path(root).joinpath(*SUBDIR)
    if not path.is_dir():
        raise RuntimeError(f"pb24: {path} is not a directory (from $OS_COLOR)")
    return path


def file_path(index: int, resolution: str = DEFAULT_RESOLUTION) -> Path:
    """Path of one realisation's netCDF.

    Parameters
    ----------
    index : int
        1-based realisation index, as in the file names.
    resolution : str, optional
        A key of :data:`RESOLUTIONS`.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(
            f"pb24: unknown resolution {resolution!r}; known: {sorted(RESOLUTIONS)}"
        )
    if not 1 <= int(index) <= N_REALISATION:
        raise ValueError(
            f"pb24: realisation index {index} outside [1, {N_REALISATION}]"
        )
    return data_dir() / f"{RESOLUTIONS[resolution]}{int(index):04d}.nc"


def read_classes() -> np.ndarray:
    """Optical water class per realisation, from the ``.mat`` sidecar.

    Returns
    -------
    numpy.ndarray
        Shape ``(N_REALISATION,)``, values in ``[1, N_CLASS]`` (very
        unbalanced: 84 to 1042 of 5000).
    """
    from scipy.io import loadmat

    raw = loadmat(data_dir() / CLASSES_FILE)["i_classes"]
    return np.asarray(raw, dtype=int).reshape(-1)


# ------------------------------------------------------------- sources ------


class _CacheSource:
    """Hyperspectral realisations from a :mod:`pb24_cache` directory."""

    resolution = "hyp"

    def __init__(self, cache_dir=None):
        from . import pb24_cache

        self.cache = pb24_cache.open_cache(cache_dir)
        if self.cache.sidecar.get("resolution", "hyp") != "hyp":
            raise ValueError("pb24: the cache is not hyperspectral")
        grids = self.cache.grids
        self.bands = self.cache.four_d_bands
        self.grids = {
            "wave": grids["lambda"].astype(float)[self.bands],
            "theta_s": grids["theta_s"].astype(float),
            "theta": grids["theta"].astype(float),
            "phi": grids["phi"].astype(float),
        }
        self.ids = self.cache.realisations

    def read(self, ids) -> dict[str, np.ndarray]:
        ids = np.asarray(ids, dtype=int)
        out: dict[str, np.ndarray] = {}
        for name in _SCALARS:
            out[name] = np.asarray(self.cache.table(name, ids), dtype=float)
        for name in _IOPS:
            out[name] = np.asarray(self.cache.table(name, ids)[:, self.bands], float)
        for name in _AOPS:
            out[name] = np.asarray(self.cache.table(name, ids)[..., self.bands], float)
        for name in _FOUR_D:
            out[name] = np.asarray(self.cache.four_d(name, ids), dtype=float)
        return out


class _FileSource:
    """Realisations read one netCDF at a time (the OLCI path, or any ``reader``).

    ``reader(index, resolution) -> xarray.Dataset`` with the files' variable
    names and dims; :func:`ocpy.hydrolight.pb24.load_ds` by default.
    """

    def __init__(self, resolution: str, reader=None, first: int = 1):
        if reader is None:
            from ocpy.hydrolight import pb24 as _ocpy_pb24

            reader = _ocpy_pb24.load_ds
        self.resolution = resolution
        self.reader = reader
        self.ids = np.arange(1, N_REALISATION + 1)
        self._first = int(first)
        self._grids = None

    @property
    def grids(self) -> dict[str, np.ndarray]:
        if self._grids is None:
            self._grids = self._grids_of(self.reader(self._first, self.resolution))
        return self._grids

    @staticmethod
    def _grids_of(ds) -> dict[str, np.ndarray]:
        return {
            "wave": np.asarray(ds["lambda"].values, dtype=float),
            "theta_s": np.asarray(ds["theta_s"].values, dtype=float),
            "theta": np.asarray(ds["theta"].values, dtype=float),
            "phi": np.asarray(ds["phi"].values, dtype=float),
        }

    def read(self, ids) -> dict[str, np.ndarray]:
        parts: dict[str, list[np.ndarray]] = {}
        for index in np.asarray(ids, dtype=int):
            ds = self.reader(int(index), self.resolution)
            grids = self._grids_of(ds)
            for name, values in grids.items():
                if not np.array_equal(values, self.grids[name]):
                    raise ValueError(
                        f"load_batch: realisation {index} has a different "
                        f"{name!r} grid than realisation {self._first}"
                    )
            one = {
                name: np.asarray(ds[name].values, float).reshape(())
                for name in _SCALARS
            }
            for name in _IOPS:
                one[name] = np.asarray(ds[name].transpose("lambda").values, float)
            for name in _AOPS:
                one[name] = np.asarray(
                    ds[name].transpose("theta_s", "lambda").values, float
                )
            for name in _FOUR_D:
                one[name] = np.asarray(
                    ds[name].transpose("theta_s", "theta", "phi", "lambda").values,
                    float,
                )
            for name, values in one.items():
                parts.setdefault(name, []).append(values)
        return {name: np.stack(values) for name, values in parts.items()}


def _open_source(resolution: str, cache_dir, reader, first: int):
    if resolution not in RESOLUTIONS:
        raise ValueError(
            f"load_batch: unknown resolution {resolution!r}; known: "
            f"{sorted(RESOLUTIONS)}"
        )
    if reader is not None:
        if cache_dir is not None:
            raise ValueError("load_batch: pass cache_dir or reader, not both")
        return _FileSource(resolution, reader, first)
    if resolution == "hyp":
        return _CacheSource(cache_dir)
    if cache_dir is not None:
        raise ValueError(
            "load_batch: the OLCI set is not cached; cache_dir is hyp-only"
        )
    return _FileSource(resolution, None, first)


# ------------------------------------------------------------- loading ------


def _geometry_table(grids: dict, angles: str) -> dict[str, np.ndarray]:
    """Flatten the angle grid to one geometry axis, C-order ``(theta_s, theta, phi)``.

    For the air modes ``theta`` is the above-water view node and ``theta_w`` its
    Snell image; for ``"in_water_only"`` both are the in-water node. The
    indices ``i_ts``, ``i_t``, ``i_ph`` address the 4-D fields.
    """
    ts, th, ph = grids["theta_s"], grids["theta"], grids["phi"]
    i_ts, i_t, i_ph = np.meshgrid(
        np.arange(ts.size), np.arange(th.size), np.arange(ph.size), indexing="ij"
    )
    i_ts, i_t, i_ph = i_ts.reshape(-1), i_t.reshape(-1), i_ph.reshape(-1)
    theta_v = th[i_t]
    theta_w = theta_v if angles == "in_water_only" else theta_air_to_water(theta_v)
    return {
        "theta_s": ts[i_ts],
        "theta_v": theta_v,
        "dphi": ph[i_ph],
        "theta_w": theta_w,
        "i_ts": i_ts,
        "i_t": i_t,
        "i_ph": i_ph,
    }


def _angle_mask(theta_s: np.ndarray, theta_v: np.ndarray, angles: str) -> np.ndarray:
    """Boolean mask over flattened geometries for an angle mode.

    For the air modes ``theta_v`` is the above-water angle; for
    ``"in_water_only"`` it is the in-water node.
    """
    if angles not in ANGLE_MODES:
        raise ValueError(f"pb24: angles must be one of {ANGLE_MODES}; got {angles!r}")
    if angles == "in_water_only":
        return theta_v >= IN_WATER_ONLY_MIN
    inside = (theta_s <= ANGLE_WINDOW_MAX) & (theta_v <= ANGLE_WINDOW_MAX)
    if angles == "window":
        return inside
    if angles == "shell":
        return ~inside
    return np.ones_like(inside, dtype=bool)


def _apply_stride(selected: np.ndarray, stride, angles) -> np.ndarray:
    """Thin the selected geometries, either flat or **per axis**.

    An ``int`` strides the flattened list (and can alias against the 13
    azimuths); a 3-tuple ``(s_theta_s, s_theta_v, s_dphi)`` strides each angle
    axis and keeps the full product of what survives, which a gridded fit on
    the same batch needs.
    """
    if isinstance(stride, (int, np.integer)):
        return selected[:: int(stride)]

    if len(stride) != 3:
        raise ValueError(
            f"load_batch: a per-axis geometry_stride needs three values "
            f"(theta_s, theta_v, dphi); got {stride!r}"
        )
    keep = np.ones(selected.size, dtype=bool)
    for step, values in zip(stride, angles, strict=True):
        step = int(step)
        if step < 1:
            raise ValueError(
                f"load_batch: geometry_stride entries must be >= 1; {stride!r}"
            )
        present = np.unique(values[selected])
        wanted = set(present[::step].tolist())
        keep &= np.isin(values[selected], list(wanted))
    return selected[keep]


def _ff_mixture(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``ff_B`` and ``ff_weight``, ``(n, wave, 2)``, from the IOP components."""
    bph, bnap = raw["bph"], raw["bNAP"]
    B_ph = raw["bbph"] / bph
    with np.errstate(divide="ignore", invalid="ignore"):
        B_nap = np.where(bnap > 0.0, raw["bbNAP"] / bnap, np.nan)
    for row in range(B_nap.shape[0]):
        bad = ~np.isfinite(B_nap[row])
        if bad.any():
            good = B_nap[row][~bad]
            B_nap[row, bad] = float(np.median(good)) if good.size else FF_B_NAP_FILL
    b_p = bph + bnap
    ff_B = np.stack([B_ph, B_nap], axis=-1)
    ff_weight = np.stack([bph / b_p, bnap / b_p], axis=-1)
    return ff_B, ff_weight


def load_batch(
    *,
    realisations=None,
    angles: str = "window",
    geometry_stride: int = 1,
    extras: tuple[str, ...] = DEFAULT_EXTRAS,
    drop_zero_rrs: bool = True,
    water_classes=None,
    validate: bool = True,
    resolution: str = DEFAULT_RESOLUTION,
    cache_dir=None,
    reader=None,
    check_fresnel: bool = True,
) -> PB24Batch:
    """Load PB24 as one stacked batch of JAX arrays, in above-water geometry.

    Parameters
    ----------
    realisations : int or sequence of int, optional
        An ``int`` takes the first *n* realisations of the source; a sequence
        names 1-based ids; ``None`` takes every realisation the source holds
        (5000, or the fixture's three). **There is no default subsample.**
    angles : str, optional
        ``"window"`` (default): ``theta_s`` and ``theta_v_air`` both <=
        :data:`ANGLE_WINDOW_MAX`; ``"shell"``: its complement in air; ``"all"``:
        the 1300 air geometries; ``"in_water_only"``: the in-water nodes >=
        :data:`IN_WATER_ONLY_MIN`, un-interpolated, ``Rrs`` all NaN, and
        ``theta_v`` holding the **in-water** angle.
    geometry_stride : int or tuple of int, optional
        Thin the selected geometries; an ``int`` strides the flattened list, a
        3-tuple ``(s_theta_s, s_theta_v, s_dphi)`` strides each axis and keeps
        the product. Reported in :class:`LoadReport`.
    extras : tuple of str, optional
        Which of :data:`EXTRA_FIELDS` to materialise as per-sample spectra.
    drop_zero_rrs : bool, optional
        Drop samples whose spectrum contains a zero ``rrs`` (default True). Only
        ``"in_water_only"`` has any (at the 87.5-degree node).
    water_classes : array_like or str, optional
        Per-realisation classes, 1-based; ``"auto"`` reads the ``.mat`` sidecar.
    validate : bool, optional
        Run :meth:`PB24Batch.validate` before returning (default True).
    resolution : str, optional
        ``"hyp"`` (default; from the cache) or ``"olci"`` (per file).
    cache_dir : str or pathlib.Path, optional
        The hyperspectral cache; default
        :func:`robust.rt.data.pb24_cache.default_cache_dir`.
    reader : callable, optional
        ``(index, resolution) -> xarray.Dataset`` in the files' layout. Forces
        the per-file path; default :func:`ocpy.hydrolight.pb24.load_ds` for
        ``"olci"``.
    check_fresnel : bool, optional
        Run the Fresnel check over every realisation read (default True) and
        warn if it leaves :data:`FRESNEL_TOLERANCE` inside the window.

    Returns
    -------
    PB24Batch
        Samples ordered realisation-major, then geometry in C-order
        ``(theta_s, theta_v, phi)``.

    Warns
    -----
    UserWarning
        If ``B_p`` falls outside :data:`B_P_EXPECTED` (never clipped), or the
        Fresnel check fails inside the window.
    """
    if isinstance(realisations, (int, np.integer)):
        first = 1
    elif realisations is None:
        first = 1
    else:
        listed = tuple(int(i) for i in realisations)
        first = listed[0] if listed else 1
    source = _open_source(resolution, cache_dir, reader, first)

    if realisations is None:
        wanted = tuple(int(i) for i in source.ids)
    elif isinstance(realisations, (int, np.integer)):
        wanted = tuple(int(i) for i in source.ids[: int(realisations)])
        if len(wanted) < int(realisations):
            raise ValueError(
                f"load_batch: asked for {realisations} realisations; the source "
                f"holds {len(source.ids)}"
            )
    else:
        wanted = tuple(int(i) for i in realisations)
    if not wanted:
        raise ValueError("load_batch: `realisations` selects nothing")
    missing = sorted(set(wanted) - {int(i) for i in source.ids})
    if missing:
        raise ValueError(
            f"load_batch: realisations {missing[:10]} are not in the source "
            f"({len(source.ids)} realisations)"
        )
    if angles not in ANGLE_MODES:
        raise ValueError(f"pb24: angles must be one of {ANGLE_MODES}; got {angles!r}")
    if isinstance(geometry_stride, (int, np.integer)) and geometry_stride < 1:
        raise ValueError(
            f"load_batch: geometry_stride must be >= 1; got {geometry_stride}"
        )
    unknown = set(extras) - set(EXTRA_FIELDS)
    if unknown:
        raise ValueError(
            f"load_batch: unknown extras {sorted(unknown)}; known: {list(EXTRA_FIELDS)}"
        )

    if isinstance(water_classes, str):
        if water_classes != "auto":
            raise ValueError(
                f"load_batch: water_classes must be an array, None or 'auto'; "
                f"got {water_classes!r}"
            )
        try:
            water_classes = read_classes()
        except Exception as exc:  # noqa: BLE001 - a missing sidecar is not fatal
            warnings.warn(
                f"pb24: water_classes='auto' could not read the sidecar ({exc}); "
                "the batch will carry no classes.",
                UserWarning,
                stacklevel=2,
            )
            water_classes = None
    elif water_classes is not None:
        water_classes = np.asarray(water_classes, dtype=int).reshape(-1)

    grids = source.grids
    if resolution == "olci":
        conventions.check_wave(grids["wave"], name="PB24 lambda", grid="olci")
    else:
        _wave_grid("hyp", grids["wave"])
    in_air = angles in AIR_ANGLE_MODES

    table = _geometry_table(grids, angles)
    n_geometry_available = table["theta_s"].size
    mask = _angle_mask(table["theta_s"], table["theta_v"], angles)
    n_dropped_angle = int((~mask).sum())
    selected = np.flatnonzero(mask)
    keep = _apply_stride(
        selected,
        geometry_stride,
        (table["theta_s"], table["theta_v"], table["dphi"]),
    )
    n_dropped_stride = int(selected.size - keep.size)
    if keep.size == 0:
        raise ValueError(
            f"load_batch: angles={angles!r} with stride {geometry_stride} "
            "selects no geometries"
        )
    coverage = {
        axis: (
            int(np.unique(table[key][keep]).size),
            int(np.unique(table[key][selected]).size),
        )
        for axis, key in (
            ("theta_s", "theta_s"),
            ("theta_v", "theta_v"),
            ("dphi", "dphi"),
        )
    }
    if isinstance(geometry_stride, (int, np.integer)):
        _warn_if_stride_aliases(coverage, geometry_stride)
    i_ts, i_t, i_ph = table["i_ts"][keep], table["i_t"][keep], table["i_ph"][keep]
    n_geometry = keep.size

    theta_nodes = grids["theta"]
    theta_w_air = theta_air_to_water(theta_nodes)
    fresnel = (
        _FresnelAccumulator(grids["theta_s"], theta_nodes) if check_fresnel else None
    )

    names = ("a", "bb_w", "bb_p", "a_ph", "B_p", "ff_B", "ff_weight", "df")
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in names + ("rrs", "Rrs", "Q") + tuple(extras)
    }
    label_parts: dict[str, list[np.ndarray]] = {"C": [], "N": [], "Y": []}
    realisation_parts, class_parts = [], []

    def per_sample(values: np.ndarray) -> np.ndarray:
        """``(n, ...)`` per realisation -> ``(n * n_geometry, ...)``."""
        n = values.shape[0]
        out = np.broadcast_to(values[:, None], (n, n_geometry) + values.shape[1:])
        return out.reshape((n * n_geometry,) + values.shape[1:])

    chunk = _CHUNK if isinstance(source, _CacheSource) else 8
    for start in range(0, len(wanted), chunk):
        ids = wanted[start : start + chunk]
        raw = source.read(ids)
        n = len(ids)

        if in_air or fresnel is not None:
            rrs_air = interp_in_water(raw["rrs"], theta_nodes, theta_w_air)
            if fresnel is not None:
                fresnel.add(raw["Rrs"], rrs_air, ids)
        if in_air:
            Q_air = interp_in_water(raw["Q"], theta_nodes, theta_w_air)
            rrs = rrs_air[:, i_ts, i_t, i_ph]
            Q = Q_air[:, i_ts, i_t, i_ph]
            Rrs = raw["Rrs"][:, i_ts, i_t, i_ph]
        else:
            rrs = raw["rrs"][:, i_ts, i_t, i_ph]
            Q = raw["Q"][:, i_ts, i_t, i_ph]
            Rrs = np.full_like(rrs, np.nan)
        n_wave = rrs.shape[-1]
        for name, values in (("rrs", rrs), ("Rrs", Rrs), ("Q", Q)):
            parts[name].append(values.reshape(n * n_geometry, n_wave))

        bb_p = raw["bbph"] + raw["bbNAP"]
        ff_B, ff_weight = _ff_mixture(raw)
        parts["a"].append(per_sample(raw["aw"] + raw["aph"] + raw["ag"] + raw["aNAP"]))
        parts["bb_w"].append(per_sample(raw["bbw"]))
        parts["bb_p"].append(per_sample(bb_p))
        parts["a_ph"].append(per_sample(raw["aph"]))
        parts["B_p"].append(per_sample(bb_p / (raw["bph"] + raw["bNAP"])))
        parts["ff_B"].append(per_sample(ff_B))
        parts["ff_weight"].append(per_sample(ff_weight))
        parts["df"].append(
            (1.0 - raw["Esdir_Es_ratio"][:, i_ts]).reshape(n * n_geometry, n_wave)
        )
        for name in extras:
            parts[name].append(raw[name][:, i_ts].reshape(n * n_geometry, n_wave))

        realisation_parts.append(np.repeat(np.asarray(ids, dtype=int), n_geometry))
        for name in label_parts:
            label_parts[name].append(np.repeat(raw[name].reshape(n), n_geometry))
        if water_classes is not None:
            class_parts.append(
                np.repeat(water_classes[np.asarray(ids, dtype=int) - 1], n_geometry)
            )

    stacked = {name: np.concatenate(values) for name, values in parts.items()}
    _warn_if_B_p_unexpected(stacked["B_p"])

    realisation = np.concatenate(realisation_parts)
    labels = {name: np.concatenate(values) for name, values in label_parts.items()}
    water_class = np.concatenate(class_parts) if class_parts else None
    n_repeat = len(wanted)
    geometry = {
        "theta_s": np.tile(table["theta_s"][keep], n_repeat),
        "theta_v": np.tile(table["theta_v"][keep], n_repeat),
        "dphi": np.tile(table["dphi"][keep], n_repeat),
    }
    theta_w = np.tile(table["theta_w"][keep], n_repeat)

    zero_bands = int((stacked["rrs"] <= 0.0).sum())
    n_dropped_zero = 0
    if drop_zero_rrs and zero_bands:
        good = ~np.any(stacked["rrs"] <= 0.0, axis=-1)
        n_dropped_zero = int((~good).sum())
        stacked = {name: values[good] for name, values in stacked.items()}
        geometry = {name: values[good] for name, values in geometry.items()}
        theta_w = theta_w[good]
        realisation = realisation[good]
        labels = {name: values[good] for name, values in labels.items()}
        if water_class is not None:
            water_class = water_class[good]

    fresnel_report = fresnel.report() if fresnel is not None else None
    if fresnel_report is not None and not fresnel_report.within():
        warnings.warn(
            f"pb24: the Fresnel check leaves {FRESNEL_TOLERANCE:.0%} inside the "
            f"window: {fresnel_report.n_beyond} values, realisations "
            f"{list(fresnel_report.realisations_beyond)[:10]}, max |dev| "
            f"{fresnel_report.worst() * 100:.2f}%. On the full release only "
            "realisation 4492 does (5 UV values at theta_v_air = 70, -5.30%).",
            UserWarning,
            stacklevel=2,
        )

    report = LoadReport(
        n_realisation_available=int(len(source.ids)),
        n_realisation=len(wanted),
        n_geometry_available=n_geometry_available,
        n_geometry=n_geometry,
        angles=angles,
        geometry_stride=geometry_stride,
        n_dropped_angle=n_dropped_angle,
        n_dropped_stride=n_dropped_stride,
        n_dropped_zero_rrs=n_dropped_zero,
        n_zero_bands=zero_bands,
        coverage=coverage,
        resolution=resolution,
        fresnel=fresnel_report,
    )

    batch = PB24Batch(
        iops=IOPs(
            a=jnp.asarray(stacked["a"]),
            bb_w=jnp.asarray(stacked["bb_w"]),
            bb_p=jnp.asarray(stacked["bb_p"]),
            a_ph=jnp.asarray(stacked["a_ph"]),
        ),
        phase_params=PhaseParams(
            B_p=jnp.asarray(stacked["B_p"]),
            ff_B=jnp.asarray(stacked["ff_B"]),
            ff_weight=jnp.asarray(stacked["ff_weight"]),
        ),
        geometry=Geometry(
            theta_s=jnp.asarray(geometry["theta_s"]),
            theta_v=jnp.asarray(geometry["theta_v"]),
            dphi=jnp.asarray(geometry["dphi"]),
            diffuse_fraction=jnp.asarray(stacked["df"]),
        ),
        rrs=jnp.asarray(stacked["rrs"]),
        Rrs=jnp.asarray(stacked["Rrs"]),
        wave=jnp.asarray(grids["wave"]),
        realisation=realisation,
        labels=labels,
        water_class=water_class,
        aops={"Q": jnp.asarray(stacked["Q"])}
        | {name: jnp.asarray(stacked[name]) for name in extras},
        report=report,
        theta_w=theta_w,
    )

    if validate:
        batch.validate()
    return batch


def _warn_if_stride_aliases(coverage, stride: int) -> None:
    """Warn if a flat stride deleted whole angle values rather than thinning them."""
    lost = {
        axis: (kept, avail) for axis, (kept, avail) in coverage.items() if kept < avail
    }
    if lost:
        detail = ", ".join(
            f"{axis}: {kept} of {avail} values" for axis, (kept, avail) in lost.items()
        )
        warnings.warn(
            f"geometry_stride={stride} loses whole angle values -- {detail}. "
            "Either the stride shares a factor with an axis length (the azimuth "
            "axis has 13 values, so stride 13 keeps one azimuth and no BRDF at "
            "all) or it is simply too coarse to reach every value. A stride "
            "coprime with 13 and 10, and well below the geometry count, avoids "
            "both.",
            UserWarning,
            stacklevel=3,
        )


def _warn_if_B_p_unexpected(B_p: np.ndarray) -> None:
    """Warn if ``B_p`` leaves :data:`B_P_EXPECTED`; never modify it."""
    lo, hi = B_P_EXPECTED
    outside = (B_p < lo) | (B_p > hi)
    if outside.any():
        warnings.warn(
            f"B_p leaves the expected range {B_P_EXPECTED}: "
            f"{int(outside.sum())} of {B_p.size} values, observed range "
            f"[{B_p.min():.5g}, {B_p.max():.5g}]. Reported, not clipped.",
            UserWarning,
            stacklevel=3,
        )


# --------------------------------------------------- interpolation error ----


@dataclass(frozen=True)
class InterpolationReport:
    """Linear against cubic interpolation of the in-water fields to ``theta_w``.

    ``error = |linear / cubic - 1|`` per ``(realisation, theta_s, theta_v_air,
    dphi, band)``; the cubic is :class:`scipy.interpolate.CubicSpline`
    (not-a-knot) through the in-water nodes 0-80 -- the grazing node 87.5 is
    left out because ``Q`` is ``inf`` there wherever ``rrs`` underflows to 0,
    and no ``theta_w`` (<= 48.2) is near it. Neither is the truth: the
    spread is the size of the interpolation question, and the Fresnel check is
    the physical test.

    Attributes
    ----------
    fields : tuple of str
        Which fields were measured (``rrs``, ``Q``).
    theta_v_air, theta_w : tuple of float
        The above-water nodes and their in-water images (nadir excluded).
    error : dict
        ``{field: per-node tuple of FRESNEL_QUANTILES of the error}``; q = 0
        and 1 exact, the rest to 5e-6.
    n_realisation : int
    window_only_theta_s : bool
        Whether only ``theta_s <= ANGLE_WINDOW_MAX`` was included.
    """

    fields: tuple[str, ...]
    theta_v_air: tuple[float, ...]
    theta_w: tuple[float, ...]
    error: dict[str, tuple[tuple[float, ...], ...]]
    n_realisation: int
    window_only_theta_s: bool = True
    quantiles: tuple[float, ...] = FRESNEL_QUANTILES

    def table(self, name: str = "rrs") -> str:
        """A markdown table of one field's error, one row per view node."""
        head = " | ".join(f"q{q:g}" for q in self.quantiles)
        lines = [
            f"| theta_v_air | theta_w | {head} |",
            "|---" * (2 + len(self.quantiles)) + "|",
        ]
        for t, w, row in zip(
            self.theta_v_air, self.theta_w, self.error[name], strict=True
        ):
            vals = " | ".join(f"{v * 100:.3f}%" for v in row)
            lines.append(f"| {t:g} | {w:.2f} | {vals} |")
        return "\n".join(lines)


def interpolation_error(
    realisations,
    *,
    fields: tuple[str, ...] = ("rrs", "Q"),
    resolution: str = DEFAULT_RESOLUTION,
    cache_dir=None,
    reader=None,
    window_only: bool = True,
) -> InterpolationReport:
    """Measure linear against cubic interpolation of ``rrs``/``Q`` to ``theta_w``.

    Parameters
    ----------
    realisations : sequence of int
        Realisation ids to measure on (a sample; this reads the full 4-D grid).
    fields : tuple of str, optional
        Any of ``"rrs"``, ``"Q"``.
    resolution, cache_dir, reader
        As :func:`load_batch`.
    window_only : bool, optional
        Restrict to ``theta_s <= ANGLE_WINDOW_MAX`` (default True); every view
        node is reported either way.

    Returns
    -------
    InterpolationReport
    """
    from scipy.interpolate import CubicSpline

    ids = tuple(int(i) for i in realisations)
    if not ids:
        raise ValueError("interpolation_error: no realisations")
    source = _open_source(resolution, cache_dir, reader, ids[0])
    grids = source.grids
    nodes = grids["theta"]
    theta_w = theta_air_to_water(nodes)
    s_keep = grids["theta_s"] <= ANGLE_WINDOW_MAX if window_only else slice(None)
    hists = {name: _Histogram(nodes.size, 0.0, 0.5, 1e-5) for name in fields}
    chunk = _CHUNK if isinstance(source, _CacheSource) else 8
    for start in range(0, len(ids), chunk):
        raw = source.read(ids[start : start + chunk])
        for name in fields:
            f = raw[name][:, s_keep]
            lin = interp_in_water(f, nodes, theta_w)
            cub = CubicSpline(nodes[:-1], f[:, :, :-1], axis=2)(theta_w)
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.abs(lin / cub - 1.0)
            for i in range(1, nodes.size):
                hists[name].add(i, err[:, :, i].ravel())
    rows = range(1, nodes.size)
    return InterpolationReport(
        fields=tuple(fields),
        theta_v_air=tuple(float(nodes[i]) for i in rows),
        theta_w=tuple(float(theta_w[i]) for i in rows),
        error={
            name: tuple(hists[name].quantiles(i, FRESNEL_QUANTILES) for i in rows)
            for name in fields
        },
        n_realisation=len(ids),
        window_only_theta_s=window_only,
    )


# -------------------------------------------------------------- select ------


def select(batch: PB24Batch, mask: np.ndarray) -> PB24Batch:
    """Subset a batch along its sample axis.

    Every leaf of ``iops``, ``phase_params`` and ``geometry`` is indexed (the
    FF mixture and ``diffuse_fraction`` included); ``wave`` and ``report`` are
    carried through unchanged, the latter because it describes the *load*.

    Raises
    ------
    ValueError
        If ``mask`` is not a boolean array of the right length.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (batch.n_sample,):
        raise ValueError(
            f"select: expected a boolean mask of shape ({batch.n_sample},); got "
            f"dtype {mask.dtype}, shape {mask.shape}"
        )

    keep = jnp.asarray(np.flatnonzero(mask))

    def take(tree):
        return jax.tree_util.tree_map(lambda x: x[keep], tree)

    return PB24Batch(
        iops=take(batch.iops),
        phase_params=take(batch.phase_params),
        geometry=take(batch.geometry),
        rrs=batch.rrs[keep],
        Rrs=batch.Rrs[keep],
        wave=batch.wave,
        realisation=batch.realisation[mask],
        labels={name: values[mask] for name, values in batch.labels.items()},
        water_class=None if batch.water_class is None else batch.water_class[mask],
        aops={name: values[keep] for name, values in batch.aops.items()},
        report=batch.report,
        theta_w=None if batch.theta_w is None else batch.theta_w[mask],
    )


# -------------------------------------------------------------- splits ------


@dataclass(frozen=True)
class SplitReport:
    """What one split holds out, and what else it holds out by accident.

    A split on ``B_p`` is not a split on ``B_p`` alone: ``B_p`` correlates with
    chlorophyll across PB24 (corr(log B_p, log C) = -0.47 over all 5000
    realisations, hyperspectral; see ``design/ztt_pb24/splits.md``), so holding
    out a band of ``B_p`` also shifts the water type. The confound is measured
    on the batch the split was mapped onto and carried with it. Read the ratios
    against :func:`confound_reference`, not against 1.0.

    Attributes
    ----------
    kind : str
        One of :data:`SPLIT_KINDS`.
    n_train, n_test : int
        Samples each side.
    n_train_realisation, n_test_realisation : int
        Distinct realisations each side (equal for ``"geometry"``).
    detail : dict
        Kind-specific numbers -- band edges, angle ranges, held-out counts.
    confound : dict
        ``test/train`` ratio of the median of each label and IOP summary.
    """

    kind: str
    n_train: int
    n_test: int
    n_train_realisation: int
    n_test_realisation: int
    detail: dict[str, float] = field(default_factory=dict)
    confound: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """One line, for a log, a table, or a notebook cell."""
        worst = ""
        if self.confound:
            name, value = max(
                self.confound.items(), key=lambda kv: abs(np.log(max(kv[1], 1e-12)))
            )
            worst = f"; largest side-effect {name} x{value:.2f}"
        return (
            f"{self.kind}: {self.n_train} train / {self.n_test} test samples "
            f"({self.n_train_realisation} / {self.n_test_realisation} "
            f"realisations){worst}"
        )


@dataclass(frozen=True)
class Splits:
    """Boolean masks over a batch's sample axis, one pair per split kind.

    Attributes
    ----------
    masks : dict
        ``{kind: (train, test)}``, each a boolean array of length
        ``batch.n_sample``.
    reports : dict
        One :class:`SplitReport` per kind.
    seed : int
        The seed the saved random split was drawn with.
    path : str
        The splits file the masks came from ("" if built in memory).
    """

    masks: dict[str, tuple[np.ndarray, np.ndarray]]
    reports: dict[str, SplitReport]
    seed: int
    path: str = ""

    def train(self, kind: str) -> np.ndarray:
        """Training mask for ``kind``."""
        return self.masks[kind][0]

    def test(self, kind: str) -> np.ndarray:
        """Held-out mask for ``kind``."""
        return self.masks[kind][1]

    def summary(self) -> str:
        """One line per split."""
        return "\n".join(self.reports[kind].summary() for kind in self.masks)


def _random_held(ids: np.ndarray, seed: int, test_fraction: float) -> np.ndarray:
    """The random split's held-out ids: a ``seed``-permutation's first
    ``round(test_fraction * n)``, sorted."""
    n_test = max(1, int(round(test_fraction * ids.size)))
    drawn = np.random.default_rng(seed).permutation(np.sort(ids))
    return np.sort(drawn[:n_test])


def _median_ratio(values: np.ndarray, test: np.ndarray) -> float:
    lo = float(np.median(values[~test]))
    hi = float(np.median(values[test]))
    return hi / lo if lo > 0.0 else float("nan")


def build_splits(
    ids,
    B_p_mean,
    labels: dict[str, np.ndarray] | None = None,
    *,
    seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
    bp_quantiles: tuple[float, float] = BP_BAND_QUANTILES,
    confound_seeds=CONFOUND_SEEDS,
    source: str = "",
) -> dict:
    """The content of a splits file, from per-realisation quantities.

    Parameters
    ----------
    ids : array_like of int
        Every realisation the file covers.
    B_p_mean : array_like
        Per-realisation mean bulk ``B_p`` (same order as ``ids``).
    labels : dict, optional
        Per-realisation summaries for the confound (``C``, ``N``, ``Y``,
        ``a_mean``, ...), same order as ``ids``.
    seed, test_fraction, bp_quantiles
        The split definition (:data:`SPLIT_SEED`, :data:`TEST_FRACTION`,
        :data:`BP_BAND_QUANTILES`).
    confound_seeds : sequence of int, optional
        Seeds for the random-split confound reference.
    source : str, optional
        Provenance text stored in the file.

    Returns
    -------
    dict
        JSON-ready: the ids, the random and B_p-band held-out ids, the band
        edges, the per-realisation ``B_p`` means, the geometry **rule**, and
        the confound (with a random-split reference band).

    Raises
    ------
    ValueError
        On bad fractions or quantiles, mismatched lengths, or an empty side.
    """
    ids = np.asarray(ids, dtype=int)
    B_p_mean = np.asarray(B_p_mean, dtype=float)
    labels = dict(labels or {})
    if ids.ndim != 1 or B_p_mean.shape != ids.shape:
        raise ValueError("build_splits: ids and B_p_mean must be 1-D, same length")
    if np.unique(ids).size != ids.size:
        raise ValueError("build_splits: ids must be unique")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"build_splits: test_fraction must lie in (0, 1); got {test_fraction}"
        )
    lo_q, hi_q = bp_quantiles
    if not 0.0 <= lo_q < hi_q <= 1.0:
        raise ValueError(
            f"build_splits: bp_quantiles must satisfy 0 <= lo < hi <= 1; "
            f"got {bp_quantiles}"
        )
    order = np.argsort(ids)
    ids, B_p_mean = ids[order], B_p_mean[order]
    labels = {k: np.asarray(v, dtype=float)[order] for k, v in labels.items()}
    columns = labels | {"B_p_mean": B_p_mean}

    held_random = _random_held(ids, seed, test_fraction)
    band_lo, band_hi = (float(v) for v in np.quantile(B_p_mean, [lo_q, hi_q]))
    held_band = ids[(B_p_mean >= band_lo) & (B_p_mean <= band_hi)]
    for name, held in (("realisation", held_random), ("bp_band", held_band)):
        if held.size in (0, ids.size):
            raise ValueError(f"build_splits: the {name!r} split has an empty side")

    def confound(held):
        test = np.isin(ids, held)
        return {k: _median_ratio(v, test) for k, v in columns.items()}

    reference: dict[str, list[float]] = {}
    for s in confound_seeds:
        for k, v in confound(_random_held(ids, int(s), test_fraction)).items():
            reference.setdefault(k, []).append(v)
    corr = (
        float(np.corrcoef(np.log(B_p_mean), np.log(labels["C"]))[0, 1])
        if "C" in labels
        else float("nan")
    )
    return {
        "version": SPLITS_VERSION,
        "source": source,
        "realisations": ids.tolist(),
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "bp_band_quantiles": [float(lo_q), float(hi_q)],
        "B_p_mean": B_p_mean.tolist(),
        "realisation": {"test": held_random.tolist()},
        "bp_band": {"lo": band_lo, "hi": band_hi, "test": held_band.tolist()},
        "geometry": {
            "rule": (
                "held out: theta_v_air >= 80 or theta_s >= 80 (degrees, "
                "above-water angles); everything else trains"
            ),
            "held_min_deg": GEOMETRY_HOLDOUT_MIN,
        },
        "confound": {
            "corr_log_B_p_log_C": corr,
            "realisation": confound(held_random),
            "bp_band": confound(held_band),
            "random_reference": {
                k: [float(np.median(v)), float(np.min(v)), float(np.max(v))]
                for k, v in reference.items()
            },
            "random_reference_seeds": [int(s) for s in confound_seeds],
        },
    }


def write_splits(path, content: dict) -> Path:
    """Write a :func:`build_splits` dict as deterministic JSON (sorted keys).

    Returns
    -------
    pathlib.Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(content, indent=1, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def read_splits(path=None) -> dict:
    """Read a splits file (default :data:`DEFAULT_SPLITS_FILE`).

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    ValueError
        On a version mismatch.
    """
    path = Path(DEFAULT_SPLITS_FILE if path is None else path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no PB24 splits file (build it with design/py/make_pb24_splits.py)"
        )
    content = json.loads(path.read_text())
    if content.get("version") != SPLITS_VERSION:
        raise ValueError(
            f"{path}: splits version {content.get('version')}, expected "
            f"{SPLITS_VERSION}"
        )
    return content


def _per_realisation_B_p(batch: PB24Batch) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``B_p`` per realisation in the batch, and the ids it belongs to."""
    B_p = np.asarray(batch.phase_params.B_p).mean(axis=-1)
    realisations = np.unique(batch.realisation)
    means = np.array([B_p[batch.realisation == index].mean() for index in realisations])
    return realisations, means


def _confound(
    batch: PB24Batch, train: np.ndarray, test: np.ndarray
) -> dict[str, float]:
    """``test/train`` ratio of median labels and IOP summaries across a split."""
    out: dict[str, float] = {}
    columns = dict(batch.labels)
    a = np.asarray(batch.iops.a).mean(axis=-1)
    bb_p = np.asarray(batch.iops.bb_p).mean(axis=-1)
    B_p = np.asarray(batch.phase_params.B_p).mean(axis=-1)
    columns.update({"a_mean": a, "bb_p_mean": bb_p, "B_p_mean": B_p})

    for name, values in columns.items():
        lo = float(np.median(values[train]))
        hi = float(np.median(values[test]))
        out[name] = hi / lo if lo > 0.0 else float("nan")
    return out


def make_splits(
    batch: PB24Batch,
    *,
    kinds: tuple[str, ...] = DEFAULT_SPLIT_KINDS,
    path=None,
) -> Splits:
    """Map the saved PB24 splits onto a batch.

    The partitions are fixed once over all 5000 realisations, by id, in
    :data:`DEFAULT_SPLITS_FILE` (Q14; M5 gotcha: a seed names a procedure,
    not a partition, once the realisation set changes). This loads that file:

    ``"realisation"``
        The saved random ``TEST_FRACTION`` of realisations, held out **whole**.
    ``"bp_band"``
        The saved interior band of per-realisation mean ``B_p``
        (:data:`BP_BAND_QUANTILES` over all 5000), held out whole. Read its
        :attr:`SplitReport.confound` before quoting the result.
    ``"geometry"``
        The rule ``theta_v_air >= 80`` or ``theta_s >= 80`` held out, the window
        trains. Needs an air batch loaded with ``angles="all"`` (or the shell).

    Parameters
    ----------
    batch : PB24Batch
    kinds : tuple of str, optional
        Defaults to :data:`DEFAULT_SPLIT_KINDS`.
    path : str or pathlib.Path, optional
        The splits file; default :data:`DEFAULT_SPLITS_FILE`.

    Returns
    -------
    Splits

    Raises
    ------
    ValueError
        On an unknown kind; a batch with a realisation the file does not
        cover; a geometry split on an ``"in_water_only"`` batch; or **any split
        whose test or train side would be empty** on this batch (an empty side
        scores nothing and reports success).
    """
    unknown = tuple(k for k in kinds if k not in SPLIT_KINDS)
    if unknown:
        raise ValueError(
            f"make_splits: unknown split kind(s) {list(unknown)}; "
            f"known: {list(SPLIT_KINDS)}"
        )
    if not kinds:
        raise ValueError("make_splits: `kinds` selects nothing")

    content = read_splits(path)
    file_ids = np.asarray(content["realisations"], dtype=int)
    realisations = np.unique(batch.realisation)
    outside = np.setdiff1d(realisations, file_ids)
    if outside.size:
        raise ValueError(
            f"make_splits: realisations {outside[:10].tolist()} are not in the "
            f"splits file {path or DEFAULT_SPLITS_FILE}"
        )
    bp_file = dict(zip(file_ids.tolist(), content["B_p_mean"], strict=True))

    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reports: dict[str, SplitReport] = {}
    for kind in kinds:
        if kind == "realisation":
            held = np.asarray(content["realisation"]["test"], dtype=int)
            test = np.isin(batch.realisation, held)
            n_held = int(np.isin(realisations, held).sum())
            detail = {
                "n_realisation_held": float(n_held),
                "test_fraction_file": float(content["test_fraction"]),
                "test_fraction_batch": float(n_held / realisations.size),
            }
        elif kind == "bp_band":
            held = np.asarray(content["bp_band"]["test"], dtype=int)
            lo, hi = content["bp_band"]["lo"], content["bp_band"]["hi"]
            test = np.isin(batch.realisation, held)
            bp_mean = np.array([bp_file[int(i)] for i in realisations])
            trained_on = bp_mean[~np.isin(realisations, held)]
            detail = {
                "B_p_band_lo": float(lo),
                "B_p_band_hi": float(hi),
                "B_p_train_lo": float(trained_on.min())
                if trained_on.size
                else float("nan"),
                "B_p_train_hi": float(trained_on.max())
                if trained_on.size
                else float("nan"),
                "n_train_inside_band": float(
                    int(((trained_on >= lo) & (trained_on <= hi)).sum())
                ),
                "n_realisation_held": float(int(np.isin(realisations, held).sum())),
                "B_p_span_all": float(bp_mean.max() / bp_mean.min()),
            }
        else:  # "geometry"
            if batch.in_water_only:
                raise ValueError(
                    "make_splits: the geometry split is defined in above-water "
                    "angles; an 'in_water_only' batch's theta_v is in-water"
                )
            cut = float(content["geometry"]["held_min_deg"])
            test = (batch.theta_s >= cut) | (batch.theta_v >= cut)
            worst = np.maximum(batch.theta_s, batch.theta_v)
            detail = {
                "held_min_deg": cut,
                "worst_angle_max_train": float(worst[~test].max())
                if (~test).any()
                else float("nan"),
                "worst_angle_min_test": float(worst[test].min())
                if test.any()
                else float("nan"),
                "worst_angle_max_test": float(worst[test].max())
                if test.any()
                else float("nan"),
            }

        train = ~test
        if not test.any() or not train.any():
            side = "test" if not test.any() else "train"
            extra = (
                " -- the batch was loaded with angles='window', which holds no "
                "shell samples; load with angles='all' for this split"
                if kind == "geometry"
                else " -- this batch's realisations all fall on one side of the "
                "saved split; load realisations from both sides"
            )
            raise ValueError(
                f"make_splits: the {kind!r} split leaves its {side} side empty"
                f"{extra}. An empty side scores nothing and reports it as success."
            )

        masks[kind] = (train, test)
        reports[kind] = SplitReport(
            kind=kind,
            n_train=int(train.sum()),
            n_test=int(test.sum()),
            n_train_realisation=int(np.unique(batch.realisation[train]).size),
            n_test_realisation=int(np.unique(batch.realisation[test]).size),
            detail=detail,
            confound=_confound(batch, train, test),
        )

    return Splits(
        masks=masks,
        reports=reports,
        seed=int(content["seed"]),
        path=str(path or DEFAULT_SPLITS_FILE),
    )


def confound_reference(
    batch: PB24Batch,
    *,
    test_fraction: float = TEST_FRACTION,
    seeds=CONFOUND_SEEDS,
) -> dict[str, tuple[float, float, float]]:
    """What a *random* hold-out of the same size does to the same labels.

    Drawn over the batch's own realisations, in memory (not from the splits
    file): the yardstick :attr:`SplitReport.confound` needs, since PB24's label
    distributions are heavy-tailed and a random 20% already moves medians a
    long way. ``B_p_mean`` is the useful entry: a random draw holds it near 1.

    Returns
    -------
    dict
        ``{label: (median, min, max)}`` of the ``test/train`` ratio across seeds.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"confound_reference: test_fraction must lie in (0, 1); got {test_fraction}"
        )
    realisations = np.unique(batch.realisation)
    collected: dict[str, list[float]] = {}
    for seed in seeds:
        held = _random_held(realisations, int(seed), test_fraction)
        test = np.isin(batch.realisation, held)
        if not test.any() or test.all():
            continue
        for name, value in _confound(batch, ~test, test).items():
            collected.setdefault(name, []).append(value)
    return {
        name: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
        for name, v in collected.items()
    }


def fit_transfer(
    batch: PB24Batch, train: np.ndarray, *, provenance: str = ""
) -> conventions.SurfaceTransfer:
    """Fit M5's tabulated surface transfer on a batch's **own training rows**.

    Kept for M5's scripts and its regression tests. Under the M8 sample
    definition the table maps ``rrs(theta_w)`` to ``Rrs(theta_air)`` on the
    above-water grid; task 4 replaces it with a Fresnel-based transfer.

    Raises
    ------
    ValueError
        Propagated from :func:`robust.rt.conventions.fit_surface_transfer` when a
        grid cell in the *selected* rows has no samples.
    """
    if batch.in_water_only:
        raise ValueError("fit_transfer: an 'in_water_only' batch has no Rrs")
    sub = select(batch, train)
    if not provenance:
        ids = np.unique(sub.realisation)
        n_cell = np.unique(
            np.stack([sub.theta_s, sub.theta_v, sub.dphi], axis=1), axis=0
        ).shape[0]
        resolution = batch.report.resolution if batch.report else DEFAULT_RESOLUTION
        provenance = (
            f"PB24 {resolution} (above-water geometry, rrs at theta_w), "
            f"{ids.size} training realisations "
            f"({ids.min()}-{ids.max()}), {n_cell} geometry cells, "
            f"{sub.n_sample} samples; one lstsq per cell"
        )
    return conventions.fit_surface_transfer(
        np.asarray(sub.rrs),
        np.asarray(sub.Rrs),
        sub.theta_s,
        sub.theta_v,
        sub.dphi,
        provenance=provenance,
    )
