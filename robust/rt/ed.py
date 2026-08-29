"""
Downwelling irradiance ``Ed(theta_s, lambda)`` at the surface, from packaged L23
spectra (inelastic design §4.2).

The inelastic terms are the only consumers: Raman needs the true ratio
``Ed(lambda') / Ed(lambda)`` (a flat-Ed correction is wrong by +60 % in the blue
to −50 % in the red — the assessment's headline), and fluorescence needs
``Ed(lambda')`` inside its excitation quadrature. The elastic model never calls
this module.

**What is packaged.** ``data/ed_l23.npz`` holds one ``Ed(0+)`` spectrum per L23
solar zenith (0/30/60 deg), extracted by ``design/py/gen_inelastic_fixture.py``
after asserting scene-independence (< 1e-3 relative scatter; measured ~5e-5) and
identity across the X=1/2/4 scenarios. Between the anchors, :func:`Ed`
interpolates linearly in ``theta_s``; outside 0–60 deg it clamps to the nearest
anchor (the :func:`robust.rt.conventions.bb_w` precedent: no silent
extrapolation, no error under ``jit``).

**The solar-model caveat (JXP, design DQ5).** The community's solar-irradiance
reference models are themselves imperfect — "a poor model of the Sun". v1
deliberately inherits whatever solar spectrum HydroLight/L23 used, because
consistency with the truth data trumps absolute solar accuracy for a forward
model scored against that data. When the effort moves to real PACE spectra, the
Ed source (TSIS-1-era references vs older standards) must be revisited; the
``Geometry.Ed`` override accepted here is the seam where that happens, with no
interface change.

**Override semantics.** ``override=(wave_Ed, Ed)`` — the :attr:`Geometry.Ed`
pair — replaces the packaged sky entirely, and ``theta_s`` is then ignored: an
override *is* one particular sky, zenith dependence included. Overrides are
interpolated onto the requested wavelengths with the same clamped linear rule.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions

__all__ = [  # noqa: RUF022  - pipeline order
    "ZENITH_ANCHORS",
    "load_table",
    "Ed",
    "ratio",
]

#: Solar zeniths of the packaged spectra, degrees. Uniformly spaced — the
#: interpolation below relies on the 30-degree stride.
ZENITH_ANCHORS = (0.0, 30.0, 60.0)

#: Lazy cache of the packaged table; see :func:`load_table`.
_TABLE: tuple[np.ndarray, np.ndarray] | None = None


def load_table() -> tuple[np.ndarray, np.ndarray]:
    """The packaged spectra as NumPy: ``(wave (81,), Ed (3, 81))``.

    Loaded lazily on first use and cached — importing :mod:`robust.rt` must not
    cost a file read — and returned as **NumPy, not JAX**, for the same reason
    ``conventions.WAVE`` is: a device array built here would fix its dtype
    before a caller can enable float64.

    Raises
    ------
    ValueError
        If the packaged file's zenith rows are not :data:`ZENITH_ANCHORS` —
        the file and this module version each other.
    """
    global _TABLE
    if _TABLE is None:
        from importlib import resources

        path = resources.files("robust.rt.data").joinpath("ed_l23.npz")
        with resources.as_file(path) as file:
            data = np.load(file)
            zeniths = tuple(float(z) for z in data["zeniths"])
            if zeniths != ZENITH_ANCHORS:
                raise ValueError(
                    f"ed_l23.npz holds zeniths {zeniths}, expected "
                    f"{ZENITH_ANCHORS}; regenerate with "
                    "design/py/gen_inelastic_fixture.py"
                )
            _TABLE = (data["wave"].astype(np.float64), data["Ed"].astype(np.float64))
    return _TABLE


def _interp_wave(
    wave: Float[Array, " wave"],
    grid: Float[Array, " grid"],
    spectra: Float[Array, "*batch grid"],
) -> Float[Array, "*batch wave"]:
    """Linear interpolation along the last axis, clamped at the grid ends.

    ``jnp.interp`` handles only 1-D inputs, so the weights are built once from
    ``wave``/``grid`` and applied by gathering — batched, ``jit``-safe, and
    differentiable in both ``wave`` and the spectrum values (the property M1's
    excitation-grid work needs; here it comes for free).
    """
    idx = jnp.clip(jnp.searchsorted(grid, wave, side="left"), 1, grid.shape[0] - 1)
    x0 = grid[idx - 1]
    x1 = grid[idx]
    # Clipping the weight is what clamps beyond the grid ends.
    w = jnp.clip((wave - x0) / (x1 - x0), 0.0, 1.0)
    return spectra[..., idx - 1] * (1.0 - w) + spectra[..., idx] * w


def Ed(
    theta_s: Float[Array, "*batch"] | float,
    wave: Float[Array, " wave"] | None = None,
    *,
    override: tuple[Float[Array, " wave_ed"], Float[Array, " wave_ed"]] | None = None,
) -> Float[Array, "*batch wave"]:
    """Surface downwelling irradiance ``Ed(0+)`` (W m^-2 nm^-1).

    Parameters
    ----------
    theta_s : Array or float
        Solar zenith (degrees), scalar or batched. Interpolated linearly
        between the packaged anchors 0/30/60 deg and **clamped** outside them.
        Ignored when ``override`` is given (module docstring).
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid. Off-grid values are
        interpolated; beyond 350–750 nm the spectrum is clamped at its ends.
    override : (Array, Array), optional
        A ``(wave_Ed, Ed)`` pair — pass :attr:`Geometry.Ed` here to use a real
        sky instead of the packaged L23 one.

    Returns
    -------
    Array
        ``Ed``, shape ``(*batch, n_wave)``; differentiable in ``theta_s`` and
        in the override values, and safe under ``jit``/``vmap``.
    """
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)

    if override is not None:
        wave_ed, ed = (jnp.asarray(part, dtype=wave.dtype) for part in override)
        return _interp_wave(wave, wave_ed, ed)

    grid_np, table_np = load_table()
    grid = jnp.asarray(grid_np, dtype=wave.dtype)
    table = jnp.asarray(table_np, dtype=wave.dtype)

    # Fractional position between anchors: 30-degree stride, clamped outside.
    t = jnp.clip(jnp.asarray(theta_s), ZENITH_ANCHORS[0], ZENITH_ANCHORS[-1]) / 30.0
    low = jnp.clip(jnp.floor(t), 0, len(ZENITH_ANCHORS) - 2).astype(int)
    frac = (t - low)[..., None]
    spectrum = table[low] * (1.0 - frac) + table[low + 1] * frac

    return _interp_wave(wave, grid, spectrum)


def ratio(
    theta_s: Float[Array, "*batch"] | float,
    wave_num: Float[Array, " wave"],
    wave_den: Float[Array, " wave"],
    *,
    override: tuple[Float[Array, " wave_ed"], Float[Array, " wave_ed"]] | None = None,
) -> Float[Array, "*batch wave"]:
    """The spectral ratio ``Ed(wave_num) / Ed(wave_den)`` at one geometry.

    The quantity the Raman term consumes: ``wave_num`` is the excitation grid
    ``lambda'`` and ``wave_den`` the emission wavelengths ``lambda``. Kept as a
    helper so both wavelength sets are guaranteed to be evaluated from the
    *same* sky — mixing the packaged Ed in the numerator with an override in
    the denominator can then never happen.

    Parameters
    ----------
    theta_s, override
        As :func:`Ed`.
    wave_num, wave_den : Array
        Wavelengths (nm) of the numerator and denominator, equal shapes.

    Returns
    -------
    Array
        The ratio, shape ``(*batch, n_wave)``.
    """
    return Ed(theta_s, wave_num, override=override) / Ed(
        theta_s, wave_den, override=override
    )
