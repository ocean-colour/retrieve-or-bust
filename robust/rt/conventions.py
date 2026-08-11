"""
Fixed conventions for the elastic-RT forward model.

One place for the choices that must be identical across every run, model, and
figure, so results stay comparable:

- the subsurface reflectance convention ``rrs = Rrs / (A + B * Rrs)`` with
  **A = 0.52, B = 1.7** (Lee et al. 2002), matching ``bing.rt``;
- the canonical wavelength grid (L23's 350-750 nm, 81 bands at 5 nm);
- pure-water backscattering ``bb_w(lambda)``, kept separate from ``bb_p``
  because the water/particle split is load-bearing for the physics (design 3);
- boundary validators, so a wrong grid or a negative IOP fails where it enters
  rather than as a puzzling number three milestones later.

**Validators are for boundaries, not for the hot path.** :func:`check_wave`,
:func:`check_iop`, and :func:`check_rrs` inspect concrete values and raise
``ValueError``. They cannot run inside ``jax.jit`` (a traced value has no
concrete truth), and they are not meant to: call them where data enters the
package -- the loader, a public constructor -- and leave ``forward`` clean.
``ValueError`` rather than bare ``assert`` because ``python -O`` strips
``assert``, and a silently skipped convention check is worse than no check.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

__all__ = [  # noqa: RUF022  - grouped by role, not alphabetical
    # Rrs <-> rrs
    "A_RRS",
    "B_RRS",
    "RRS_POLE",
    "Rrs_to_rrs",
    "rrs_to_Rrs",
    # Wavelength grid
    "WAVE_MIN",
    "WAVE_MAX",
    "WAVE_STEP",
    "N_WAVE",
    "WAVE",
    "canonical_wave",
    # Named grids (M5: a second dataset)
    "WaveGrid",
    "OLCI_WAVE",
    "L23_GRID",
    "OLCI_GRID",
    "GRIDS",
    "wave_grid",
    "grid_wave",
    # Pure water
    "BB_W_L23",
    "BB_W_RANGE",
    "BB_W_TAIL_EXPONENT",
    "bb_w",
    # Geometry-aware surface transfer (M5)
    "SurfaceTransfer",
    "SURFACE_TABLE",
    "interp_geometry",
    "fit_surface_transfer",
    "save_transfer",
    "load_transfer",
    "default_transfer",
    # Validators
    "check_wave",
    "check_bb_w_range",
    "check_iop",
    "check_rrs",
]


# --------------------------------------------------------------- Rrs <-> rrs -
#: Lee et al. (2002) coefficients for the above/below-water conversion. These
#: must equal ``bing.rt.A_Rrs`` / ``bing.rt.B_Rrs`` -- the two packages
#: disagreeing about what ``rrs`` means is exactly what fixing them prevents,
#: and a test asserts the equality rather than trusting this comment.
A_RRS = 0.52
B_RRS = 1.7

#: ``rrs`` value at which :func:`rrs_to_Rrs` diverges (``1 - B * rrs -> 0``).
#: Real-ocean ``rrs`` is ~1e-3 to 5e-2, so this is only reachable via a unit
#: error -- which is precisely why :func:`check_rrs` looks for it.
RRS_POLE = 1.0 / B_RRS


def _coefficients(geometry, transfer):
    """``(A, B)`` for a conversion: the nadir constants, or a fitted table.

    Passing neither is the M0-M4 path and returns :data:`A_RRS`, :data:`B_RRS`
    unchanged, so every existing number is reproducible from a clean checkout.
    """
    if geometry is None and transfer is None:
        return A_RRS, B_RRS
    if geometry is None:
        raise ValueError(
            "conventions: a surface transfer needs a geometry; pass geometry= "
            "or neither"
        )
    table = default_transfer() if transfer is None else transfer
    A, B = table.coefficients(geometry)
    return A[..., None], B[..., None]


def Rrs_to_rrs(
    Rrs: Float[Array, "..."], *, geometry=None, transfer=None
) -> Float[Array, "..."]:
    """Subsurface remote-sensing reflectance from the above-water value.

    ``rrs = Rrs / (A + B * Rrs)``.

    Parameters
    ----------
    Rrs : Array
        Above-water remote-sensing reflectance (sr^-1).
    geometry : robust.rt.types.Geometry, optional
        Viewing geometry. **Omit it and nothing changes**: the nadir constants
        :data:`A_RRS` / :data:`B_RRS` are used, which is what M0-M4 measured and
        what keeps their numbers reproducible. Supply it to use a fitted
        geometry-dependent transfer, which off-nadir is worth up to a factor of
        six -- see :class:`SurfaceTransfer`.
    transfer : SurfaceTransfer, optional
        Which table to use; defaults to :func:`default_transfer` when a geometry
        is given.

    Returns
    -------
    Array
        Subsurface remote-sensing reflectance (sr^-1). With a geometry, the
        trailing axis of ``Rrs`` is taken to be wavelength and the coefficients
        broadcast across it.
    """
    A, B = _coefficients(geometry, transfer)
    return Rrs / (A + B * Rrs)


def rrs_to_Rrs(
    rrs: Float[Array, "..."], *, geometry=None, transfer=None
) -> Float[Array, "..."]:
    """Above-water remote-sensing reflectance from the subsurface value.

    ``Rrs = A * rrs / (1 - B * rrs)``, the exact inverse of
    :func:`Rrs_to_rrs`. Diverges at ``rrs = 1 / B`` and returns *negative*
    values beyond it; see :func:`check_rrs`.

    Parameters
    ----------
    rrs : Array
        Subsurface remote-sensing reflectance (sr^-1).
    geometry, transfer
        As :func:`Rrs_to_rrs`; omitting both is the nadir path and is unchanged.

    Returns
    -------
    Array
        Above-water remote-sensing reflectance (sr^-1).
    """
    A, B = _coefficients(geometry, transfer)
    return A * rrs / (1.0 - B * rrs)


# ----------------------------------------------------------- wavelength grid -
WAVE_MIN = 350.0
WAVE_MAX = 750.0
WAVE_STEP = 5.0
N_WAVE = 81

#: The canonical grid, as NumPy. Deliberately *not* a device array built at
#: import: ``jnp`` would fix its dtype before a caller can enable float64.
#: These are exact multiples of 5, so float32 represents them without error.
WAVE = np.arange(WAVE_MIN, WAVE_MAX + WAVE_STEP / 2, WAVE_STEP)


def canonical_wave(dtype=None) -> Float[Array, " 81"]:
    """The canonical wavelength grid as a JAX array.

    Parameters
    ----------
    dtype : optional
        Passed to ``jnp.asarray``; default follows JAX's current default
        (float32, or float64 when ``jax_enable_x64`` is on).

    Returns
    -------
    Array
        Shape ``(81,)``, 350-750 nm in 5 nm steps.
    """
    return jnp.asarray(WAVE, dtype=dtype)


# ---------------------------------------------------------------- named grids -
#: PB24's OLCI band centres (nm), read from the files themselves rather than
#: transcribed: every ``SD_OLCI_no_R_*.nc`` carries this identical ``lambda``
#: coordinate. Note it is **not** a subsample of :data:`WAVE` -- half its bands
#: (412, 443, 673, 681, 709, 753) fall between 5 nm nodes -- and its last band
#: lies 3 nm beyond :data:`WAVE_MAX`, which is what :func:`check_bb_w_range` is
#: for.
OLCI_WAVE = np.array(
    [400.0, 412.0, 443.0, 490.0, 510.0, 560.0, 620.0, 665.0, 673.0, 681.0, 709.0, 753.0]
)


@dataclass(frozen=True)
class WaveGrid:
    """A named wavelength grid a dataset is defined on.

    M0-M4 had exactly one grid, so "the canonical grid" and "the wavelength grid"
    were the same sentence and :func:`check_wave` could hard-code it. M5 adds a
    second dataset on different bands, and the fix is deliberately *not* to
    loosen the check: a grid mismatch has caught real bugs, and it should keep
    catching them **per grid**. So the check gains a grid rather than losing its
    teeth.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"l23"``.
    wave : ndarray
        Band centres (nm), ascending.
    description : str
        One line, for error messages.
    """

    name: str
    wave: np.ndarray
    description: str = ""

    @property
    def n_wave(self) -> int:
        """Number of bands."""
        return int(self.wave.shape[0])

    @property
    def span(self) -> tuple[float, float]:
        """``(min, max)`` band centre, nm."""
        return float(self.wave[0]), float(self.wave[-1])


#: L23's grid -- the canonical one, and the default everywhere a grid is optional.
#: Named ``"canonical"`` rather than ``"l23"`` so the validator messages M0-M4
#: wrote (and their tests match on) are unchanged; ``"l23"`` is an alias.
L23_GRID = WaveGrid("canonical", WAVE, "L23 elastic release, 350-750 nm at 5 nm")

#: PB24's OLCI grid (Pitarch & Brando; see the M5 hand-off).
OLCI_GRID = WaveGrid("olci", OLCI_WAVE, "PB24 OLCI bands, 400-753 nm")

#: Every grid the package knows by name, plus the ``"l23"`` alias.
GRIDS = {g.name: g for g in (L23_GRID, OLCI_GRID)} | {"l23": L23_GRID}


def wave_grid(grid=None) -> WaveGrid:
    """Resolve a grid specification to a :class:`WaveGrid`.

    Parameters
    ----------
    grid : None, str, or WaveGrid, optional
        ``None`` (the default) means :data:`L23_GRID`, so every pre-M5 call site
        keeps its meaning. A string is looked up in :data:`GRIDS`; a
        :class:`WaveGrid` is returned unchanged.

    Returns
    -------
    WaveGrid

    Raises
    ------
    KeyError
        On an unknown name, listing the ones that exist -- a typo should not
        silently fall back to the canonical grid.
    """
    if grid is None:
        return L23_GRID
    if isinstance(grid, WaveGrid):
        return grid
    try:
        return GRIDS[grid]
    except KeyError:
        raise KeyError(
            f"unknown wavelength grid {grid!r}; known grids: {sorted(GRIDS)}"
        ) from None


def grid_wave(grid=None, dtype=None) -> Float[Array, " wave"]:
    """A named grid's band centres as a JAX array.

    Parameters
    ----------
    grid : None, str, or WaveGrid, optional
        As :func:`wave_grid`.
    dtype : optional
        Passed to ``jnp.asarray``.

    Returns
    -------
    Array
        Shape ``(n_wave,)``.
    """
    return jnp.asarray(wave_grid(grid).wave, dtype=dtype)


# --------------------------------------------------- pure-water backscattering
#: Pure-water backscattering on :data:`WAVE`, m^-1.
#:
#: Provenance: ``bb - bbnw`` from the L23 elastic file (``load_ds(1, 0)``). This
#: is not an approximation of L23's water model -- it *is* it, which is the
#: point: the forward model is trained against L23, so any other ``bb_w`` would
#: put a bias straight into ``bb_p = bb - bb_w``.
#:
#: Verified before embedding: the difference is constant to 1.6e-7 relative
#: (float32 storage noise) across all 3320 scenes, all three solar zeniths, and
#: both X=1 and X=4 -- so there is no scene to choose. ``bing``'s
#: ``bbNWModel.init_bbw`` and ``ocpy.water.scattering.bbw_from_l23`` compute the
#: same quantity but each picks an arbitrary scene index (0 and 170) without
#: checking that it does not matter; embedding the table removes the question,
#: and a test re-derives it from the netCDF.
#:
#: Falls as ``lambda^-4.2`` (fitted), consistent with molecular scattering's
#: ``-4.32`` (Morel 1974). The physically-parameterized alternative -- Zhang, Hu
#: & He (2009), with temperature and salinity -- is *not* usable today:
#: ``ocpy.water.scattering.betasw_ZHH2009`` raises "THIS IS NOT SUCCESFULLY
#: CONVERTED YET" (an unfinished MATLAB port). It becomes relevant at M5, when
#: new HydroLight runs may not share L23's water column.
BB_W_L23 = np.array(
    [
        5.91730000e-03, 5.55850007e-03, 5.22689987e-03, 4.92009986e-03,
        4.63580014e-03, 4.37199976e-03, 4.12699953e-03, 3.89909977e-03,
        3.68690025e-03, 3.48910014e-03, 3.30460002e-03, 3.13220010e-03,
        2.97100004e-03, 2.82019982e-03, 2.67890003e-03, 2.54639983e-03,
        2.42209993e-03, 2.30529997e-03, 2.19559995e-03, 2.09229998e-03,
        1.99509994e-03, 1.90350006e-03, 1.81720010e-03, 1.73569983e-03,
        1.65870006e-03, 1.58600020e-03, 1.51730003e-03, 1.45220000e-03,
        1.39070000e-03, 1.33230002e-03, 1.27710006e-03, 1.22460001e-03,
        1.17480010e-03, 1.12759997e-03, 1.08269998e-03, 1.04000000e-03,
        9.99439973e-04, 9.60820005e-04, 9.24060005e-04, 8.89040064e-04,
        8.55669961e-04, 8.23850045e-04, 7.93509942e-04, 7.64550059e-04,
        7.36900023e-04, 7.10489985e-04, 6.85259991e-04, 6.61139959e-04,
        6.38069992e-04, 6.16010046e-04, 5.94890036e-04, 5.74669975e-04,
        5.55300037e-04, 5.36739943e-04, 5.18949993e-04, 5.01889968e-04,
        4.85529978e-04, 4.69829974e-04, 4.54769994e-04, 4.40300006e-04,
        4.26400016e-04, 4.13039990e-04, 4.00209974e-04, 3.87869979e-04,
        3.75999982e-04, 3.64589971e-04, 3.53599986e-04, 3.43020016e-04,
        3.32840020e-04, 3.23030021e-04, 3.13579978e-04, 3.04479996e-04,
        2.95700040e-04, 2.87230010e-04, 2.79069995e-04, 2.71199999e-04,
        2.63599970e-04, 2.56259984e-04, 2.49179982e-04, 2.42340000e-04,
        2.35729996e-04,
    ]
)  # fmt: skip


#: The range :data:`BB_W_L23` actually supports, nm. Outside it, every answer is
#: a choice rather than a lookup -- see :func:`bb_w`'s ``mode``.
BB_W_RANGE = (WAVE_MIN, WAVE_MAX)

#: Power-law exponent of :data:`BB_W_L23` over its **red tail** (650-750 nm),
#: fitted in log-log: ``bb_w ~ lambda ** BB_W_TAIL_EXPONENT``. Measured, not
#: quoted -- it reproduces the tabulated tail to 2.2e-4 relative, and a test
#: re-derives it from the table. The whole-range fit gives -4.215 and the
#: literature's molecular value is -4.32 (Morel 1974); the tail fit is the right
#: one for extrapolating *past* 750 nm, which is the only thing it is used for.
BB_W_TAIL_EXPONENT = -4.140855


def bb_w(
    wave: Float[Array, "..."] | None = None, *, mode: str = "clamp"
) -> Float[Array, "..."]:
    """Pure-water backscattering coefficient.

    Linearly interpolates :data:`BB_W_L23`. Differentiable in ``wave`` and safe
    inside ``jit``; needs no data files.

    Parameters
    ----------
    wave : Array, optional
        Wavelengths (nm). Defaults to the canonical grid, where the table is
        returned exactly.
    mode : {"clamp", "extrapolate", "raise"}, optional
        What to do outside :data:`BB_W_RANGE`. ``"clamp"`` (default) holds the
        end points, which is ``jnp.interp``'s behaviour and what M0-M4 relied on;
        ``"extrapolate"`` continues the fitted red tail
        (:data:`BB_W_TAIL_EXPONENT`); ``"raise"`` refuses, and is a **boundary**
        option only -- it inspects concrete values, so it cannot run under
        ``jit``.

    Returns
    -------
    Array
        ``bb_w(wave)`` in m^-1.

    Raises
    ------
    ValueError
        For an unknown ``mode``, or under ``mode="raise"`` when any wavelength
        falls outside the table.

    Notes
    -----
    **Why this has a mode at all.** On L23's grid the question never arose: the
    table's support *is* the grid, so the clamp could not fire. PB24's OLCI grid
    ends at 753 nm, 3 nm past the table, where clamping overstates ``bb_w`` by
    1.6% -- small, but silent, and it grows to 23% at 800 nm where the
    hyperspectral files reach. Making the choice explicit is cheaper than
    discovering later which one a number was computed with.

    PB24 tabulates its own ``bbw`` per band, so its loader should prefer the
    file's values over this table entirely; the mode exists for callers that
    cannot.
    """
    if mode not in ("clamp", "extrapolate", "raise"):
        raise ValueError(
            f"bb_w: mode must be 'clamp', 'extrapolate' or 'raise'; got {mode!r}"
        )
    if wave is None:
        return jnp.asarray(BB_W_L23)
    if mode == "raise":
        check_bb_w_range(wave, name="bb_w wave")
    x = jnp.asarray(wave)
    table = jnp.interp(x, jnp.asarray(WAVE), jnp.asarray(BB_W_L23))
    if mode != "extrapolate":
        return table
    lo, hi = BB_W_RANGE
    tail = jnp.asarray(BB_W_L23)[-1] * (x / hi) ** BB_W_TAIL_EXPONENT
    head = jnp.asarray(BB_W_L23)[0] * (x / lo) ** BB_W_TAIL_EXPONENT
    return jnp.where(x > hi, tail, jnp.where(x < lo, head, table))


# ------------------------------------------------------------------ validators
def check_wave(wave, *, name: str = "wave", atol: float = 1e-3, grid=None) -> None:
    """Raise unless ``wave`` is the expected grid.

    Parameters
    ----------
    wave : array_like
        Candidate wavelength grid (nm).
    name : str, optional
        Name used in the error message.
    atol : float, optional
        Absolute tolerance in nm. The default 1e-3 nm is far tighter than any
        real grid difference but loose enough for float32 round-tripping.
    grid : None, str, or WaveGrid, optional
        Which grid to check against; ``None`` means :data:`L23_GRID`, so every
        pre-M5 call site is unchanged. The check is **per grid**, not relaxed:
        passing PB24's bands while expecting L23's still fails, which is the
        point.

    Raises
    ------
    ValueError
        If the shape or the values differ from the grid.
    KeyError
        If ``grid`` names a grid that does not exist.
    """
    g = wave_grid(grid)
    arr = np.asarray(wave, dtype=float)
    if arr.shape != (g.n_wave,):
        raise ValueError(
            f"{name}: expected the {g.name} grid of shape ({g.n_wave},), "
            f"got {arr.shape}"
        )
    if not np.allclose(arr, g.wave, atol=atol, rtol=0.0):
        worst = int(np.argmax(np.abs(arr - g.wave)))
        lo, hi = g.span
        raise ValueError(
            f"{name}: not the {g.name} {lo:.0f}-{hi:.0f} nm grid; "
            f"largest difference {arr[worst] - g.wave[worst]:+.4g} nm at index "
            f"{worst} (got {arr[worst]:.4g}, expected {g.wave[worst]:.4g})"
        )


def check_bb_w_range(wave, *, name: str = "wave") -> None:
    """Raise if any wavelength falls outside :data:`BB_W_L23`'s support.

    The boundary counterpart to :func:`bb_w`'s ``mode``: call it where a grid
    enters the package, so that a clamp -- which is silent by construction, being
    ``jnp.interp``'s default -- cannot be the thing nobody noticed.

    Parameters
    ----------
    wave : array_like
        Wavelengths (nm).
    name : str, optional
        Name used in the error message.

    Raises
    ------
    ValueError
        If any wavelength lies outside :data:`BB_W_RANGE`.
    """
    arr = np.asarray(wave, dtype=float)
    lo, hi = BB_W_RANGE
    outside = (arr < lo) | (arr > hi)
    if np.any(outside):
        bad = arr[outside]
        raise ValueError(
            f"{name}: {bad.size} wavelength(s) outside the bb_w table's "
            f"{lo:.0f}-{hi:.0f} nm support (e.g. {bad.min():.4g}, {bad.max():.4g} nm); "
            "bb_w would clamp there. Pass mode='extrapolate' to continue the fitted "
            "tail, or use the dataset's own bb_w."
        )


def check_iop(values, name: str = "iop") -> None:
    """Raise unless an inherent optical property is finite and non-negative.

    Parameters
    ----------
    values : array_like
        Absorption or backscattering values (m^-1).
    name : str, optional
        Name used in the error message.

    Raises
    ------
    ValueError
        If any value is NaN, infinite, or negative.
    """
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        n = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name}: {n} non-finite value(s)")
    if np.any(arr < 0.0):
        n = int((arr < 0.0).sum())
        raise ValueError(
            f"{name}: {n} negative value(s), minimum {arr.min():.6g}; "
            "IOPs are non-negative by definition"
        )


def check_rrs(values, name: str = "rrs", subsurface: bool = True) -> None:
    """Raise unless a reflectance is finite, non-negative, and convertible.

    Parameters
    ----------
    values : array_like
        Reflectance (sr^-1).
    name : str, optional
        Name used in the error message.
    subsurface : bool, optional
        When True (default) the values are subsurface ``rrs`` and are also
        checked against :data:`RRS_POLE`, past which :func:`rrs_to_Rrs`
        diverges and then goes negative. Above-water ``Rrs`` has no such pole.

    Raises
    ------
    ValueError
        If any value is non-finite, negative, or (subsurface only) at or beyond
        the conversion pole -- in practice the signature of a unit error.
    """
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        n = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name}: {n} non-finite value(s)")
    if np.any(arr < 0.0):
        n = int((arr < 0.0).sum())
        raise ValueError(f"{name}: {n} negative value(s), minimum {arr.min():.6g}")
    if subsurface and np.any(arr >= RRS_POLE):
        raise ValueError(
            f"{name}: {int((arr >= RRS_POLE).sum())} value(s) at or beyond the "
            f"rrs_to_Rrs pole {RRS_POLE:.4g} (maximum {arr.max():.6g}); "
            "ocean rrs is ~1e-3 to 5e-2, so check the units"
        )


# ----------------------------------------------- geometry-aware surface transfer
#: Where the fitted transfer table ships, relative to this package.
SURFACE_TABLE = "files/surface_pb24.npz"


@dataclass(frozen=True)
class SurfaceTransfer:
    """Lee-form coefficients ``A``, ``B`` tabulated against viewing geometry.

    :data:`A_RRS` and :data:`B_RRS` are *nadir* constants. Measured against PB24,
    ``Rrs = A rrs / (1 - B rrs)`` with those values is good at nadir and fails
    progressively off-nadir -- a median 6.6% over the sanctioned 0-70 degree
    window and 27% at ``theta_v = 60`` alone, because the true ``A`` falls from
    0.53 at nadir to 0.34 at 70 degrees as the Fresnel transmittance drops.

    So ``A`` and ``B`` become functions of ``(theta_s, theta_v, dphi)``,
    tabulated on the reference grid and interpolated trilinearly. Three findings
    shaped this:

    - **All three angles earn their place.** At ``theta_v = 60`` the per-geometry
      ``A`` still spans 0.28-0.46 across solar zenith and azimuth, so a
      ``A(theta_v)`` table alone leaves a median 3.4% and up to 70% error against
      the full one.
    - **``Q`` does not.** Lee's ``B = 1.7`` is really ``rbar * Q`` with ``Q``
      assumed ~3.5, and PB24 tabulates the real ``Q`` (0.9-6.0). Refitting with
      it in place -- ``1 - rbar Q rrs`` -- scores 1.71% against 1.74% for simply
      fitting ``B`` per geometry. It is not worth carrying, which is fortunate:
      :func:`robust.rt.forward` has no ``Q`` to offer.
    - **The residual does not go to zero.** Even fitting both coefficients at
      every geometry leaves a median 1.8%, so the Lee *form* is the floor here,
      not the coefficients. Reported rather than papered over.

    Attributes
    ----------
    theta_s, theta_v, dphi : numpy.ndarray
        Ascending node vectors, degrees.
    A, B : numpy.ndarray
        Coefficients of shape ``(n_theta_s, n_theta_v, n_dphi)``.
    provenance : str
        What was fitted, on what, and when.
    """

    theta_s: np.ndarray
    theta_v: np.ndarray
    dphi: np.ndarray
    A: np.ndarray
    B: np.ndarray
    provenance: str = ""

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_theta_s, n_theta_v, n_dphi)``."""
        return (self.theta_s.size, self.theta_v.size, self.dphi.size)

    def coefficients(self, geometry):
        """``(A, B)`` at a geometry, by trilinear interpolation.

        Parameters
        ----------
        geometry : robust.rt.types.Geometry
            Angles in degrees; any batch shape.

        Returns
        -------
        tuple of Array
            ``A`` and ``B``, broadcast to ``geometry.theta_s``'s shape.

        Notes
        -----
        Differentiable and ``jit``-safe. Values outside the node range clamp to
        the edge rather than extrapolating -- past 87.75 degrees the sun is below
        the horizon, so there is nothing to extrapolate *to*.

        **Piecewise linear means kinks at the nodes.** ``jax.grad`` returns a
        one-sided slope there while a central difference averages both sides, and
        they disagree by O(1) at an exact node. Gradient checks must be run
        *between* nodes -- the same care ``o25_coefficients`` needs (M4 gotcha 4).
        """
        nodes = (self.theta_s, self.theta_v, self.dphi)
        return (
            interp_geometry(self.A, nodes, geometry),
            interp_geometry(self.B, nodes, geometry),
        )


def _axis_weights(x, nodes):
    """Lower index and fractional weight for linear interpolation along one axis.

    A single-node axis degenerates to that node with zero weight, so a table that
    does not resolve an angle -- L23 is nadir-only, so its O25 refit has one
    ``theta_v`` node -- still evaluates rather than indexing out of bounds.
    """
    nodes = jnp.asarray(nodes)
    x = jnp.asarray(x)
    n = nodes.shape[0]
    if n == 1:
        return (
            jnp.zeros(jnp.shape(x), dtype=int),
            jnp.zeros(jnp.shape(x), dtype=jnp.asarray(x).dtype),
        )
    idx = jnp.clip(jnp.searchsorted(nodes, x, side="right") - 1, 0, n - 2)
    lo = nodes[idx]
    hi = nodes[idx + 1]
    return idx, jnp.clip((x - lo) / (hi - lo), 0.0, 1.0)


def interp_geometry(table, nodes, geometry):
    """Trilinear interpolation of a geometry-indexed table.

    Shared by :class:`SurfaceTransfer` and :class:`robust.rt.baselines.O25Table`,
    which index the same three angles on the same reference grid -- one
    implementation, tested once.

    Parameters
    ----------
    table : array_like
        Shape ``(n_theta_s, n_theta_v, n_dphi, ...)``; trailing axes are carried
        through, so a table of four O25 coefficients works as well as a scalar.
    nodes : tuple of array_like
        ``(theta_s, theta_v, dphi)`` node vectors, ascending, degrees.
    geometry : robust.rt.types.Geometry
        Angles in degrees, any batch shape.

    Returns
    -------
    Array
        Shape ``geometry.theta_s.shape + table.shape[3:]``.

    Notes
    -----
    Differentiable and ``jit``-safe. Values outside the nodes **clamp** rather
    than extrapolating. Piecewise linear, so there are kinks at the nodes: run
    gradient checks *between* them (M4 gotcha 4).
    """
    values = jnp.asarray(table)
    trailing = values.ndim - 3
    i_s, w_s = _axis_weights(geometry.theta_s, nodes[0])
    i_v, w_v = _axis_weights(geometry.theta_v, nodes[1])
    i_d, w_d = _axis_weights(geometry.dphi, nodes[2])
    if trailing:
        expand = (...,) + (None,) * trailing
        w_s, w_v, w_d = w_s[expand], w_v[expand], w_d[expand]

    total = 0.0
    for ds, ws in ((0, 1.0 - w_s), (1, w_s)):
        for dv, wv in ((0, 1.0 - w_v), (1, w_v)):
            for dd, wd in ((0, 1.0 - w_d), (1, w_d)):
                total = (
                    total
                    + ws
                    * wv
                    * wd
                    * values[
                        jnp.minimum(i_s + ds, values.shape[0] - 1),
                        jnp.minimum(i_v + dv, values.shape[1] - 1),
                        jnp.minimum(i_d + dd, values.shape[2] - 1),
                    ]
                )
    return total


def fit_surface_transfer(
    rrs, Rrs, theta_s, theta_v, dphi, *, provenance: str = ""
) -> SurfaceTransfer:
    """Fit ``A`` and ``B`` per geometry by least squares.

    ``Rrs = A rrs + B (Rrs rrs)`` is **linear in both coefficients**, so this is
    one ``lstsq`` per geometry -- no seed, no learning rate, no stopping rule, and
    the same fairness argument that applies to :func:`robust.rt.baselines.fit_o25`
    applies here.

    Parameters
    ----------
    rrs, Rrs : array_like
        Shape ``(n_sample, n_wave)``, paired subsurface and above-water
        reflectance.
    theta_s, theta_v, dphi : array_like
        Per-sample angles, shape ``(n_sample,)``, degrees.
    provenance : str, optional
        Recorded on the result.

    Returns
    -------
    SurfaceTransfer

    Raises
    ------
    ValueError
        If any grid cell has no samples -- a table with a hole would interpolate
        across it silently.
    """
    rrs = np.asarray(rrs, dtype=float)
    Rrs = np.asarray(Rrs, dtype=float)
    nodes = [np.unique(np.asarray(a, dtype=float)) for a in (theta_s, theta_v, dphi)]
    index = [
        np.searchsorted(node, np.asarray(a, dtype=float))
        for node, a in zip(nodes, (theta_s, theta_v, dphi), strict=True)
    ]

    shape = tuple(node.size for node in nodes)
    A = np.full(shape, np.nan)
    B = np.full(shape, np.nan)
    flat = (index[0] * shape[1] + index[1]) * shape[2] + index[2]
    for cell in np.unique(flat):
        mask = flat == cell
        r = rrs[mask].ravel()
        R = Rrs[mask].ravel()
        design = np.stack([r, R * r], axis=1)
        coef, *_ = np.linalg.lstsq(design, R, rcond=None)
        i = np.unravel_index(cell, shape)
        A[i], B[i] = coef

    if not np.isfinite(A).all():
        missing = int((~np.isfinite(A)).sum())
        raise ValueError(
            f"fit_surface_transfer: {missing} of {A.size} grid cells had no "
            "samples; a table with holes interpolates across them silently"
        )
    return SurfaceTransfer(
        theta_s=nodes[0],
        theta_v=nodes[1],
        dphi=nodes[2],
        A=A,
        B=B,
        provenance=provenance,
    )


def save_transfer(path, transfer: SurfaceTransfer) -> None:
    """Write a :class:`SurfaceTransfer` to ``.npz``, atomically.

    Temp file -> load back -> verify -> :func:`os.replace`, the pattern PR #11
    settled on: validate first, overwrite second.

    Parameters
    ----------
    path : str or pathlib.Path
    transfer : SurfaceTransfer
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(
            tmp,
            theta_s=transfer.theta_s,
            theta_v=transfer.theta_v,
            dphi=transfer.dphi,
            A=transfer.A,
            B=transfer.B,
            provenance=np.asarray(transfer.provenance),
        )
        back = load_transfer(tmp)
        if back.shape != transfer.shape or not np.allclose(back.A, transfer.A):
            raise ValueError(
                f"save_transfer: the written table does not load back equal; "
                f"{path} left untouched"
            )
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_transfer(path) -> SurfaceTransfer:
    """Read a :class:`SurfaceTransfer` from ``.npz``.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    SurfaceTransfer
    """
    data = np.load(path)
    return SurfaceTransfer(
        theta_s=data["theta_s"],
        theta_v=data["theta_v"],
        dphi=data["dphi"],
        A=data["A"],
        B=data["B"],
        provenance=str(data["provenance"]),
    )


def default_transfer() -> SurfaceTransfer:
    """The table shipped with the package, read once and cached.

    Returns
    -------
    SurfaceTransfer

    Raises
    ------
    FileNotFoundError
        If the table is absent -- regenerate with ``design/py/fit_surface.py``.
    """
    global _DEFAULT_TRANSFER
    if _DEFAULT_TRANSFER is None:
        path = Path(__file__).parent / SURFACE_TABLE
        if not path.is_file():
            raise FileNotFoundError(
                f"conventions: no fitted surface table at {path}; regenerate it "
                "with design/py/fit_surface.py"
            )
        _DEFAULT_TRANSFER = load_transfer(path)
    return _DEFAULT_TRANSFER


#: Cache for :func:`default_transfer`.
_DEFAULT_TRANSFER: SurfaceTransfer | None = None
