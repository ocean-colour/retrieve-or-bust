"""
Fixed conventions for the elastic-RT forward model.

One place for the choices that must be identical across every run, model, and
figure, so results stay comparable:

- the subsurface reflectance convention ``rrs = Rrs / (A + B * Rrs)`` with
  **A = 0.52, B = 1.7** (Lee et al. 2002), matching ``bing.rt``;
- the canonical wavelength grid (L23's 350-750 nm, 81 bands at 5 nm);
- pure-water backscattering ``bb_w(lambda)``, kept separate from ``bb_p``
  because the water/particle split is load-bearing for the physics (design 3);
- the Raman excitation grid (inelastic design §3): the 3400 cm^-1 water Raman
  shift, the wavenumber-form maps between emission and excitation wavelengths,
  and the one clamped-linear spectrum interpolation the package uses to put
  IOPs (and Ed) onto off-grid wavelengths, differentiable in the values;
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

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

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
    # Fresnel surface transfer (M8 task 4)
    "REFRACTIVE_INDEX",
    "fresnel_transmittance",
    "diffuse_transmittance",
    "TransferLike",
    "FresnelTransfer",
    "FRESNEL_A0",
    "FRESNEL_B",
    "FRESNEL_TD_A0",
    "FRESNEL_TD_B",
    "FRESNEL_N_TRAINING",
    "FRESNEL_TRAINING_DIGEST",
    "FRESNEL_PROVENANCE",
    "default_fresnel_transfer",
    "realisation_digest",
    "fresnel_normal_equations",
    "solve_fresnel",
    "fit_fresnel_transfer",
    # Geometry-aware surface transfer (M5)
    "SurfaceTransfer",
    "SURFACE_TABLE",
    "interp_geometry",
    "fit_surface_transfer",
    "save_transfer",
    "load_transfer",
    "default_transfer",
    # Raman excitation grid (inelastic M1)
    "RAMAN_SHIFT",
    "RAMAN_WAVE_MIN_OFFICIAL",
    "raman_excitation",
    "raman_emission",
    "interp_spectrum",
    # Validators
    "check_wave",
    "check_bb_w_range",
    "check_iop",
    "check_rrs",
]


# --------------------------------------------------------------- Rrs <-> rrs -
#: float: Numerator coefficient of the Lee et al. (2002) above/below-water
#: conversion, ``Rrs = A rrs / (1 - B rrs)``. Must equal ``bing.rt.A_Rrs`` --
#: the two packages disagreeing about what ``rrs`` means is exactly what
#: fixing them prevents, and a test asserts the equality rather than trusting
#: this comment.
A_RRS = 0.52
#: float: Denominator coefficient of the same Lee et al. (2002) conversion.
#: Must equal ``bing.rt.B_Rrs``, and the same test asserts it.
B_RRS = 1.7

#: ``rrs`` value at which :func:`rrs_to_Rrs` diverges (``1 - B * rrs -> 0``).
#: Real-ocean ``rrs`` is ~1e-3 to 5e-2, so this is only reachable via a unit
#: error -- which is precisely why :func:`check_rrs` looks for it.
RRS_POLE = 1.0 / B_RRS


def _coefficients(geometry, transfer):
    """``(A, B)`` for a conversion: the nadir constants, or a fitted transfer.

    Passing neither is the M0-M4 path and returns :data:`A_RRS`, :data:`B_RRS`
    unchanged, so every existing number is reproducible from a clean checkout.
    ``transfer`` is duck-typed (:class:`TransferLike`): anything with
    ``coefficients(geometry) -> (A, B)``, i.e. a :class:`SurfaceTransfer` or a
    :class:`FresnelTransfer`. With a geometry and no transfer the default is
    still M5's table (:func:`default_transfer`): M8 task 4 deliberately did
    not change it (the Fresnel transfer is opt-in).
    """
    if geometry is None and transfer is None:
        return A_RRS, B_RRS
    if geometry is None:
        raise ValueError(
            "conventions: a surface transfer needs a geometry; pass geometry= "
            "or neither"
        )
    table = default_transfer() if transfer is None else transfer
    if hasattr(table, "spectral_coefficients"):
        # Already shaped to broadcast against (..., n_wave): a FresnelTransfer
        # with the downwelling factor and a spectral diffuse fraction is
        # wavelength-dependent.
        return table.spectral_coefficients(geometry)
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
    transfer : TransferLike, optional
        Which transfer to use -- a :class:`SurfaceTransfer` table or a
        :class:`FresnelTransfer` (M8 task 4); defaults to
        :func:`default_transfer` (M5's table, superseded but still the default)
        when a geometry is given.

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
#: float: Blue edge of the canonical grid (nm) -- the L23 grid's own first
#: band, and the floor every clamp in the package extrapolates from.
WAVE_MIN = 350.0
#: float: Red edge of the canonical grid (nm).
WAVE_MAX = 750.0
#: float: Spacing of the canonical grid (nm).
WAVE_STEP = 5.0
#: int: Number of canonical grid points, ``(WAVE_MAX - WAVE_MIN) / WAVE_STEP
#: + 1``. Named because it is the shape every spectral array is checked
#: against by :func:`check_wave`.
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
    wave : numpy.ndarray
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
    mode : str, optional
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


# -------------------------------------------------- Raman excitation grid ----
#: Wavenumber shift of water Raman scattering, cm^-1 (Ge et al. 1993; the value
#: BING fixes as ``bing.rt.raman.WAVENUMBER_SHIFT_CENTER``, asserted equal by a
#: test). The physical band is ~3100-3700 cm^-1 wide; v1 uses the single-shift
#: center per the design's single-Gaussian emission choice (§4.4).
RAMAN_SHIFT = 3400.0

#: Official lower edge of the inelastic model's wavelength support, nm
#: (inelastic design §3). The Raman excitation for a 400 nm emission is
#: ``raman_excitation(400.) = 352.11 nm`` -- just inside the L23 grid
#: (:data:`WAVE_MIN` = 350). Below 400 nm the maps and interpolation still
#: *run* (no error, by design), but the excitation wavelengths leave the grid
#: and :func:`interp_spectrum` clamps to the 350 nm end value -- constant
#: extrapolation, a documented caveat rather than a gate.
RAMAN_WAVE_MIN_OFFICIAL = 400.0


def raman_excitation(
    wave: Float[Array, "..."], shift: float = RAMAN_SHIFT
) -> Float[Array, "..."]:
    """Excitation wavelength ``lambda'`` feeding Raman emission at ``wave``.

    The wavenumber form: ``1/lambda' = 1/lambda + shift`` (excitation photons
    are bluer than the emission by the water Raman shift). This is *the*
    excitation-grid map of the design (§3): the forward model interpolates the
    supplied IOP spectra onto ``raman_excitation(wave)``.

    Pure, ``jit``-able, differentiable. See :data:`RAMAN_WAVE_MIN_OFFICIAL`
    for the support caveat below 400 nm.

    Parameters
    ----------
    wave : Array
        Emission wavelengths (nm).
    shift : float, optional
        Wavenumber shift (cm^-1); default :data:`RAMAN_SHIFT`.

    Returns
    -------
    Array
        Excitation wavelengths (nm), same shape; e.g. 418.55 nm for 488 nm.
    """
    wave = jnp.asarray(wave)
    return 1.0 / (1.0 / wave + shift * 1e-7)


def raman_emission(
    wave_ex: Float[Array, "..."], shift: float = RAMAN_SHIFT
) -> Float[Array, "..."]:
    """Emission wavelength produced by Raman scattering of ``wave_ex``.

    The exact inverse of :func:`raman_excitation`:
    ``1/lambda = 1/lambda' - shift``. Matches
    ``bing.rt.raman.excitation_to_emission_wavelength`` (asserted by a test);
    e.g. 488 nm excitation emits at 585.08 nm.

    Parameters
    ----------
    wave_ex : Array
        Excitation wavelengths (nm).
    shift : float, optional
        Wavenumber shift (cm^-1); default :data:`RAMAN_SHIFT`.

    Returns
    -------
    Array
        Emission wavelengths (nm), same shape.
    """
    wave_ex = jnp.asarray(wave_ex)
    return 1.0 / (1.0 / wave_ex - shift * 1e-7)


def _as_float(x) -> Array:
    """``jnp.asarray``, with integer input coerced to the default float dtype."""
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.asarray(0.0).dtype)
    return x


def _interp_weights(
    x: Float[Array, "*any"], grid: Float[Array, " grid"]
) -> tuple[Array, Float[Array, "*any"]]:
    """Clamped-linear interpolation weights on a strictly increasing grid.

    Returns ``(idx, w)`` such that the interpolant is
    ``y[..., idx-1] * (1-w) + y[..., idx] * w``. Clipping the weight is what
    clamps beyond the grid ends -- no extrapolation, no boundary ``raise``
    that could not run under ``jit``. Shared by :func:`interp_spectrum` and
    :mod:`robust.rt.ed`'s zenith interpolation, so neither carries a private
    stride assumption.
    """
    idx = jnp.clip(jnp.searchsorted(grid, x, side="left"), 1, grid.shape[0] - 1)
    w = jnp.clip((x - grid[idx - 1]) / (grid[idx] - grid[idx - 1]), 0.0, 1.0)
    return idx, w


def interp_spectrum(
    wave_new: Float[Array, " wave_new"],
    grid: Float[Array, " grid"],
    spectra: Float[Array, "*batch grid"],
) -> Float[Array, "*batch wave_new"]:
    """Linear interpolation of (batched) spectra onto new wavelengths.

    The package's one interpolation rule (promoted here from
    :mod:`robust.rt.ed` at M1 task 2, so the Raman excitation grid and Ed use
    the same arithmetic): linear between grid points, **clamped** to the end
    values outside the grid (constant extrapolation -- the ``bb_w``
    precedent, and the documented sub-400 nm caveat of
    :data:`RAMAN_WAVE_MIN_OFFICIAL`).

    ``jnp.interp`` handles only 1-D inputs, so the weights are built once from
    ``wave_new``/``grid`` and applied by gathering -- batched, ``jit``-safe,
    and **differentiable in the spectrum values** (the property the Raman term
    needs: gradients flow through the excitation-grid IOPs back to the IOP
    inputs) as well as in ``wave_new``. All inputs are promoted to one common
    floating dtype -- integer wavelengths select float nodes, they never
    truncate the values (the PR #14 lesson, record §3.2.1).

    Parameters
    ----------
    wave_new : Array
        Wavelengths to evaluate at (nm), e.g. ``raman_excitation(wave)``.
    grid : Array
        Strictly increasing wavelengths (nm) the spectra are sampled on.
    spectra : Array
        Values on ``grid``, shape ``(..., n_grid)``.

    Returns
    -------
    Array
        Shape ``(..., n_wave_new)``.
    """
    wave_new = _as_float(wave_new)
    grid = _as_float(grid)
    spectra = _as_float(spectra)
    dtype = jnp.result_type(wave_new.dtype, grid.dtype, spectra.dtype)
    idx, w = _interp_weights(wave_new.astype(dtype), grid.astype(dtype))
    spectra = spectra.astype(dtype)
    return spectra[..., idx - 1] * (1.0 - w) + spectra[..., idx] * w


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


# ---------------------------------------------------- Fresnel surface transfer -
#: float: Refractive index of seawater, the value ZTT's Snell refraction uses
#: (``robust.rt.ztt.REFRACTIVE_INDEX`` is this constant) and PB24's HydroLight
#: runs assume for the flat-surface Fresnel law.
REFRACTIVE_INDEX = 1.34


def fresnel_transmittance(theta_air_deg, n: float = REFRACTIVE_INDEX):
    """Unpolarised flat-surface Fresnel transmittance, air side, ``1 - (Rs + Rp)/2``.

    For a direction at zenith ``θ`` in air and ``θt = asin(sin θ / n)`` in
    water, ``Rs = ((cos θ - n cos θt)/(cos θ + n cos θt))²`` and
    ``Rp = ((n cos θ - cos θt)/(n cos θ + cos θt))²``. Radiance crossing the
    surface in either direction sees the same ``T`` (reciprocity), so this is
    the factor between ``rrs(θ_w)`` and ``Rrs(θ)`` in the Lee form's ``A``
    (``rt_fit_pb24.md`` Q9c). ``T(0) = 1 - ((n-1)/(n+1))²`` = 0.97889 at
    n = 1.34; ``T/T(0)`` = 1.000 / 0.999 / 0.986 / 0.959 / 0.883 at
    10 / 30 / 50 / 60 / 70° (Q9 Round 2 table, pinned by a test).

    Parameters
    ----------
    theta_air_deg : Array
        Above-water zenith, degrees, in [0, 90].
    n : float, optional
        Relative refractive index water/air.

    Returns
    -------
    Array
        ``T``, dimensionless, the shape of ``theta_air_deg``.

    Notes
    -----
    Differentiable everywhere in [0, 90]: ``cos θt = sqrt(1 - sin²θ / n²)``
    stays away from 0 because ``n > 1``. ``jit``-safe; float32 unless x64 is
    enabled.
    """
    ti = jnp.deg2rad(jnp.asarray(theta_air_deg))
    cos_i = jnp.cos(ti)
    sin_t = jnp.sin(ti) / n
    cos_t = jnp.sqrt(1.0 - sin_t**2)
    rs = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    rp = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    return 1.0 - 0.5 * (rs**2 + rp**2)


def _fresnel_ratio_np(theta_air_deg, n: float = REFRACTIVE_INDEX) -> np.ndarray:
    """``T(θ)/T(0)`` in NumPy float64 -- the fit's copy of
    :func:`fresnel_transmittance`, so a fit does not depend on JAX's x64 flag."""
    ti = np.radians(np.asarray(theta_air_deg, dtype=float))

    def T(cos_i, sin_i):
        cos_t = np.sqrt(1.0 - (sin_i / n) ** 2)
        rs = (cos_i - n * cos_t) / (cos_i + n * cos_t)
        rp = (n * cos_i - cos_t) / (n * cos_i + cos_t)
        return 1.0 - 0.5 * (rs**2 + rp**2)

    return T(np.cos(ti), np.sin(ti)) / T(1.0, 0.0)


_DIFFUSE_T: dict[float, float] = {}


def diffuse_transmittance(n: float = REFRACTIVE_INDEX) -> float:
    """Transmittance of a **uniform-radiance sky** through a flat surface.

    ``T̄ = 2 ∫ T(θ) cos θ sin θ dθ`` over [0, 90°] (the cosine-weighted
    hemispherical mean of :func:`fresnel_transmittance`), by the trapezoid
    rule on 20,001 nodes (error < 1e-8). 0.93249 at n = 1.34. Cached per ``n``.

    Parameters
    ----------
    n : float, optional

    Returns
    -------
    float
    """
    key = float(n)
    if key not in _DIFFUSE_T:
        th = np.linspace(0.0, np.pi / 2, 20001)
        T = _fresnel_ratio_np(np.degrees(th), key) * _fresnel_T0_np(key)
        _DIFFUSE_T[key] = float(np.trapezoid(2.0 * T * np.cos(th) * np.sin(th), th))
    return _DIFFUSE_T[key]


def _fresnel_T0_np(n: float = REFRACTIVE_INDEX) -> float:
    """``T(0) = 1 - ((n - 1)/(n + 1))²``, float64."""
    return 1.0 - ((n - 1.0) / (n + 1.0)) ** 2


def _downwelling_ratio_np(theta_s_deg, H, n: float = REFRACTIVE_INDEX):
    """``t_d / T(0)`` in NumPy float64, ``t_d = (1 - H) T(θs) + H T̄``.

    ``H = None`` is the direct beam alone, ``T(θs)/T(0)``. Shapes broadcast.
    """
    direct = _fresnel_ratio_np(theta_s_deg, n)
    if H is None:
        return direct
    H = np.asarray(H, dtype=float)
    return (1.0 - H) * direct + H * (diffuse_transmittance(n) / _fresnel_T0_np(n))


@runtime_checkable
class TransferLike(Protocol):
    """What :func:`Rrs_to_rrs` / :func:`rrs_to_Rrs` accept as ``transfer=``.

    Duck-typed: anything with ``coefficients(geometry) -> (A, B)``, each
    broadcast to the geometry's batch shape, is a surface transfer. Both
    :class:`SurfaceTransfer` (M5's table) and :class:`FresnelTransfer`
    (M8 task 4) satisfy it; nothing checks ``isinstance`` on the hot path.
    A transfer may also define ``spectral_coefficients(geometry)``, returning
    ``(A, B)`` already shaped to broadcast against ``(..., n_wave)``; the
    converters prefer it when present (a wavelength-dependent transfer
    needs it).
    """

    def coefficients(self, geometry):
        """``(A, B)`` of the Lee form at ``geometry``."""
        ...


@dataclass(frozen=True)
class FresnelTransfer:
    """The Lee form with a Fresnel-scaled numerator: two numbers, any geometry.

    ``Rrs = A(θv) rrs / (1 - B rrs)``, ``A(θv) = A0 T(θv) / T(0)``, with ``T``
    the flat-surface :func:`fresnel_transmittance` at the **above-water**
    sensor zenith (``Geometry.theta_v``). ``rrs`` is the subsurface value at
    the refracted angle, ``asin(sin θv / n)``: the Snell pairing PB24 uses
    (``rt_fit_pb24.md`` Q9). ``θs`` and ``Δφ`` do not enter.

    This replaces M5's 2600-number :class:`SurfaceTransfer` table as the
    physical form (Q9c); see :data:`FRESNEL_A0` for the shipped numbers and
    :func:`default_fresnel_transfer`. It is **opt-in**: the package default
    when only a geometry is given is still M5's table, and with neither the
    nadir constants, so no L23 number moves.

    **The downwelling factor** (``downwelling=True``; M8 task 4, measured, not
    in the task's text). ``rrs`` is normalised by ``Ed(0-)`` and ``Rrs`` by
    ``Ed(0+)``, so the transfer also carries the surface's downwelling
    transmittance ``t_d = (1 - H) T(θs) + H T̄`` (direct beam at the sun's
    zenith, diffuse sky as a uniform-radiance hemisphere,
    :func:`diffuse_transmittance`):
    ``A = A0 · T(θv)/T(0) · t_d/T(0)``. ``H`` is ``geometry.diffuse_fraction``
    (the PB24 loader fills it; spectral, so ``A`` becomes
    ``(..., n_wave)``); ``None`` takes the direct beam alone (``H = 0``).
    Held out on PB24 it halves the window error of the θv-only form and
    removes its failure at large θs (``design/ztt_pb24/surface_fresnel.md``);
    its fitted ``A0`` is Lee's 0.52. Shipped as :data:`FRESNEL_TD_A0`,
    :data:`FRESNEL_TD_B`.

    Attributes
    ----------
    A0 : float
        The numerator at nadir (sun and view at zenith, ``H = 0``).
    B : float
        The denominator coefficient (Lee's ``B``), constant.
    n : float
        Refractive index used in ``T``.
    provenance : str
        What was fitted, on what, and when.
    downwelling : bool
        Include the downwelling factor ``t_d/T(0)`` (default ``False``: the
        θv-only form the task specified).
    """

    A0: float
    B: float
    n: float = REFRACTIVE_INDEX
    provenance: str = ""
    downwelling: bool = False

    def coefficients(self, geometry):
        """``(A, B)`` at a geometry.

        Parameters
        ----------
        geometry : robust.rt.types.Geometry
            Angles in degrees; any batch shape. Only ``theta_v`` (above-water)
            is used.

        Returns
        -------
        tuple of Array
            ``A = A0 T(θv)/T(0)`` (times ``t_d/T(0)`` with ``downwelling``) and
            ``B``, broadcast to the common shape of ``theta_s``, ``theta_v``
            and ``dphi`` -- plus a trailing wavelength axis when
            ``downwelling`` is on and ``geometry.diffuse_fraction`` is set.

        Notes
        -----
        Smooth in the angles (no table nodes, so no kinks) and in ``H``;
        differentiable and ``jit``-safe.
        """
        A = self._A(geometry)
        return A, jnp.broadcast_to(jnp.asarray(self.B, dtype=A.dtype), A.shape)

    def _spectral(self, geometry) -> bool:
        """Whether ``A`` carries a wavelength axis."""
        return self.downwelling and geometry.diffuse_fraction is not None

    def _A(self, geometry):
        """``A`` at ``geometry`` (see :meth:`coefficients`)."""
        theta_s = jnp.asarray(geometry.theta_s)
        theta_v = jnp.asarray(geometry.theta_v)
        shape = jnp.broadcast_shapes(
            jnp.shape(theta_s), jnp.shape(theta_v), jnp.shape(geometry.dphi)
        )
        T0 = fresnel_transmittance(jnp.zeros_like(theta_v), self.n)
        A = jnp.broadcast_to(
            self.A0 * fresnel_transmittance(theta_v, self.n) / T0, shape
        )
        if not self.downwelling:
            return A
        direct = jnp.broadcast_to(fresnel_transmittance(theta_s, self.n) / T0, shape)
        if geometry.diffuse_fraction is None:
            return A * direct
        H = jnp.asarray(geometry.diffuse_fraction)
        sky = diffuse_transmittance(self.n) / T0[..., None]
        t_d = (1.0 - H) * direct[..., None] + H * sky
        return A[..., None] * t_d

    def spectral_coefficients(self, geometry):
        """``(A, B)`` shaped to broadcast against ``(..., n_wave)``.

        What :func:`rrs_to_Rrs` / :func:`Rrs_to_rrs` use: a trailing
        length-1 axis for the per-geometry form, the wavelength axis itself
        when ``A`` is spectral.
        """
        A, B = self.coefficients(geometry)
        if self._spectral(geometry):
            return A, B
        return A[..., None], B[..., None]


#: float: **Shipped** nadir numerator of :class:`FresnelTransfer` (M8 task 4).
#: Fitted by ``design/py/fit_surface_fresnel.py`` (``--check`` recomputes it
#: bit for bit): PB24 hyperspectral cache ``pb24_v1`` (all 451 bands,
#: 350-800 nm), the **realisation split's training side** of
#: ``design/ztt_pb24/splits.json`` (all 4000 training realisations;
#: :data:`FRESNEL_TRAINING_DIGEST`), the window geometries (θs ≤ 70°,
#: θv_air ≤ 70°; 832 per realisation), ``rrs`` linearly interpolated to the
#: refracted angle as the loader does; relatively weighted linear least
#: squares ``1 = A0 (T/T0) rrs/Rrs + B rrs`` (normal equations in float64, no
#: optimiser); 2026-11-04. Held-out scores: ``design/ztt_pb24/surface_fresnel.md``.
FRESNEL_A0 = 0.505894615306294
#: float: **Shipped** denominator coefficient of :class:`FresnelTransfer`;
#: same fit as :data:`FRESNEL_A0`.
FRESNEL_B = 1.7281346247647804
#: float: **Shipped** ``A0`` of the ``downwelling=True`` form (the factor
#: ``t_d/T(0)`` with PB24's own ``H = 1 - Esdir_Es_ratio`` at each row's θs
#: and λ in the design matrix); otherwise the same rows, weighting, script and
#: date as :data:`FRESNEL_A0`.
FRESNEL_TD_A0 = 0.5199319425562341
#: float: **Shipped** ``B`` of the ``downwelling=True`` form.
FRESNEL_TD_B = 1.731512969766272
#: int: Training realisations behind :data:`FRESNEL_A0` / :data:`FRESNEL_B`.
FRESNEL_N_TRAINING = 4000
#: str: :func:`realisation_digest` of the realisation ids the shipped
#: constants were fitted on. A test checks it equals the digest of the
#: realisation split's **training** ids in ``splits.json`` -- so a refit that
#: touched one held-out realisation fails the suite.
FRESNEL_TRAINING_DIGEST = (
    "3dd4bcb017fcded0f23f7a899cf091b6fd64e74736f77261d1eec92ca541219c"
)
#: str: Provenance of the shipped constants, as one line.
FRESNEL_PROVENANCE = (
    "PB24 hyp cache pb24_v1, 451 bands; realisation-split training side of "
    "design/ztt_pb24/splits.json (4000 realisations); window geometries "
    "(theta_s, theta_v_air <= 70); 1,500,928,000 rows; relatively weighted "
    "linear least squares (float64 normal equations); "
    "design/py/fit_surface_fresnel.py; 2026-11-04"
)


def realisation_digest(ids) -> str:
    """SHA-256 of a set of realisation ids: sorted, unique, comma-joined ASCII.

    Parameters
    ----------
    ids : array_like of int

    Returns
    -------
    str
        Hex digest.
    """
    unique = np.unique(np.asarray(ids, dtype=int).ravel())
    return hashlib.sha256(",".join(str(int(i)) for i in unique).encode()).hexdigest()


def default_fresnel_transfer(*, downwelling: bool = False) -> FresnelTransfer:
    """The shipped :class:`FresnelTransfer`.

    Parameters
    ----------
    downwelling : bool, optional
        ``False`` (default): the θv-only form, :data:`FRESNEL_A0` /
        :data:`FRESNEL_B`. ``True``: with the downwelling factor,
        :data:`FRESNEL_TD_A0` / :data:`FRESNEL_TD_B` -- the better transfer
        on PB24 when ``Geometry.diffuse_fraction`` is set.

    Returns
    -------
    FresnelTransfer
    """
    if downwelling:
        return FresnelTransfer(
            A0=FRESNEL_TD_A0,
            B=FRESNEL_TD_B,
            provenance=FRESNEL_PROVENANCE,
            downwelling=True,
        )
    return FresnelTransfer(A0=FRESNEL_A0, B=FRESNEL_B, provenance=FRESNEL_PROVENANCE)


def fresnel_normal_equations(
    rrs,
    Rrs,
    theta_v,
    *,
    Q=None,
    factor=None,
    relative: bool = True,
    n: float = REFRACTIVE_INDEX,
):
    """Normal equations of the :class:`FresnelTransfer` fit, for streaming.

    ``Rrs = A0 r rrs + B (Rrs rrs)`` with ``r = T(θv)/T(0)`` is linear in
    ``(A0, B)``. ``relative=True`` (the shipped fit) divides each row by
    ``Rrs`` -- ``1 = A0 r rrs/Rrs + B rrs`` -- so the objective is the
    relative error the transfer is scored by; ``False`` is the unweighted
    regression of ``Rrs``. With ``Q`` the ``B`` column is multiplied by ``Q``
    (the ``Rrs = A rrs / (1 - B Q rrs)`` form, reported, not shipped). With
    ``factor`` the ``A0`` column is multiplied by it: the downwelling form
    passes ``t_d/T(0)`` (``_downwelling_ratio_np``).

    The sums are plain float64 ``numpy.sum`` (pairwise, single-threaded), so
    accumulating chunk by chunk in a fixed order is bit-reproducible.

    Parameters
    ----------
    rrs, Rrs : array_like
        Paired subsurface (at the refracted angle) and above-water
        reflectance, any common shape.
    theta_v : array_like
        Above-water view zenith (degrees), broadcastable against ``rrs``.
    Q : array_like, optional
        ``Q`` at the same samples, for the ``Q`` form.
    factor : array_like, optional
        Extra multiplier of ``r``, broadcastable against ``rrs``.
    relative : bool, optional
    n : float, optional

    Returns
    -------
    tuple
        ``(XtX, Xty, n_rows, n_dropped)``: a (2, 2) and a (2,) float64 array,
        the rows used and the rows dropped (non-finite or non-positive
        ``rrs``/``Rrs``).
    """
    rrs = np.asarray(rrs, dtype=float)
    Rrs = np.asarray(Rrs, dtype=float)
    r = _fresnel_ratio_np(theta_v, n)
    if factor is not None:
        r = r * np.asarray(factor, dtype=float)
    r = np.broadcast_to(r, rrs.shape)
    good = np.isfinite(rrs) & np.isfinite(Rrs) & (rrs > 0) & (Rrs > 0)
    second = rrs if Q is None else rrs * np.asarray(Q, dtype=float)
    if Q is not None:
        good &= np.isfinite(second)
    rrs_g, Rrs_g, r_g, second_g = rrs[good], Rrs[good], r[good], second[good]
    if relative:
        x0 = r_g * rrs_g / Rrs_g
        x1 = second_g
        y = np.ones_like(x0)
    else:
        x0 = r_g * rrs_g
        x1 = second_g * Rrs_g
        y = Rrs_g
    XtX = np.array(
        [
            [np.sum(x0 * x0), np.sum(x0 * x1)],
            [np.sum(x0 * x1), np.sum(x1 * x1)],
        ]
    )
    Xty = np.array([np.sum(x0 * y), np.sum(x1 * y)])
    return XtX, Xty, int(good.sum()), int((~good).sum())


def solve_fresnel(XtX, Xty) -> tuple[float, float]:
    """``(A0, B)`` from accumulated normal equations.

    Parameters
    ----------
    XtX, Xty : array_like
        As :func:`fresnel_normal_equations` returns, summed over chunks.

    Returns
    -------
    tuple of float
    """
    coef = np.linalg.solve(np.asarray(XtX, dtype=float), np.asarray(Xty, dtype=float))
    return float(coef[0]), float(coef[1])


def fit_fresnel_transfer(
    rrs,
    Rrs,
    theta_v,
    *,
    relative: bool = True,
    n: float = REFRACTIVE_INDEX,
    provenance: str = "",
) -> FresnelTransfer:
    """Fit a :class:`FresnelTransfer` in one pass (small batches, tests).

    ``design/py/fit_surface_fresnel.py`` streams the same normal equations
    over the cache; this is the one-chunk version.

    Parameters
    ----------
    rrs, Rrs : array_like
        ``(n_sample, n_wave)`` paired reflectances.
    theta_v : array_like
        ``(n_sample,)`` above-water view zenith, degrees.
    relative, n, provenance
        As :func:`fresnel_normal_equations`; ``provenance`` is recorded.

    Returns
    -------
    FresnelTransfer
    """
    theta_v = np.asarray(theta_v, dtype=float)
    rrs = np.asarray(rrs, dtype=float)
    if theta_v.ndim == rrs.ndim - 1:
        theta_v = theta_v[..., None]
    XtX, Xty, _, _ = fresnel_normal_equations(rrs, Rrs, theta_v, relative=relative, n=n)
    A0, B = solve_fresnel(XtX, Xty)
    return FresnelTransfer(A0=A0, B=B, n=n, provenance=provenance)


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
      :func:`~robust.rt.hybrid.forward` has no ``Q`` to offer.
    - **The residual does not go to zero.** Even fitting both coefficients at
      every geometry leaves a median 1.8%, so the Lee *form* is the floor here,
      not the coefficients. Reported rather than papered over.

    .. note:: **Superseded, 2026-11-04 (M8 task 4, ``rt_fit_pb24.md`` Q9c).**
       The numbers above, and the shipped table, come from M5's pairing of
       ``Rrs(θ)`` with ``rrs`` at the *same nominal* ``θ``; PB24 tabulates
       ``rrs`` at the in-water angle, so the right partner is ``rrs`` at
       ``asin(sin θ / 1.34)``. Most of "A falls from 0.53 to 0.34" was that
       mismatch plus Fresnel transmittance, not a property of the water. On
       correctly paired data the shipped table is *worse* than the nadir
       constants (16.0% vs 6.8% at θv = 60°, M8 task 3). The class still
       works for any table -- ``robust.rt.data.pb24.fit_transfer`` fits one on
       correctly paired rows -- and :class:`FresnelTransfer` is the physical
       two-number replacement.

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

    .. note:: **Superseded, 2026-11-04 (M8 task 4, Q9c).**
       ``surface_pb24.npz`` encodes M5's mismatched pairing (``rrs`` at the
       nominal rather than the refracted view angle) and is kept loadable for
       the record and because it is still what ``rrs_to_Rrs(...,
       geometry=...)`` uses when no ``transfer`` is passed (changing that
       default is a deliberate later act, not part of task 4). New work should
       pass :func:`default_fresnel_transfer` explicitly. Do not regenerate it
       with ``design/py/fit_surface.py``.

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
