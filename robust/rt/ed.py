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

**Override semantics.** ``override=(wave_Ed, Ed)`` — the
:attr:`Geometry.Ed <robust.rt.types.Geometry.Ed>` pair — replaces the packaged
sky entirely, and ``theta_s`` is then ignored: an override *is* one particular
sky, zenith dependence included. Overrides are interpolated onto the requested
wavelengths with the same clamped linear rule.
"""

from __future__ import annotations

import functools

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

#: Solar zeniths of the packaged spectra, degrees. Any strictly increasing
#: anchor set works — the interpolation below is searchsorted-based, not
#: stride-based (PR #14 review, record §2.7 finding 6).
ZENITH_ANCHORS = (0.0, 30.0, 60.0)


@functools.cache
def load_table() -> tuple[np.ndarray, np.ndarray]:
    """The packaged spectra as NumPy: ``(wave (81,), Ed (3, 81))``.

    Loaded lazily on first use and cached (``functools.cache``, the package's
    idiom — see ``emulator.load_default``) — importing :mod:`robust.rt` must
    not cost a file read — and returned as **NumPy, not JAX**, for the same
    reason ``conventions.WAVE`` is: a device array built here would fix its
    dtype before a caller can enable float64.

    Raises
    ------
    ValueError
        If the packaged file's zenith rows are not :data:`ZENITH_ANCHORS` —
        the file and this module version each other.
    """
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
        return (data["wave"].astype(np.float64), data["Ed"].astype(np.float64))


# The interpolation machinery was born here at M1 task 1 and promoted to
# conventions at task 2, where the Raman excitation grid shares it; these
# aliases keep this module readable and the arithmetic single-sourced.
_as_float = conventions._as_float
_interp_weights = conventions._interp_weights
_interp_wave = conventions.interp_spectrum


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
        A ``(wave_Ed, Ed)`` pair — pass
        :attr:`Geometry.Ed <robust.rt.types.Geometry.Ed>` here to use a real
        sky instead of the packaged L23 one.

    Returns
    -------
    Array
        ``Ed``, shape ``(*batch, n_wave)``; differentiable in ``theta_s`` and
        in the override values, and safe under ``jit``/``vmap``.
    """
    # Dtype rule (PR #14 review, record §2.7 finding 1): every input is
    # PROMOTED to one common floating dtype, never truncated. An integer
    # wavelength grid (a natural way to spell 400..700 nm) selects float
    # interpolation nodes rather than collapsing the irradiances to 0/1, and a
    # float64 theta_s keeps float64 arithmetic even when `wave` arrives
    # float32 — which it can mid-session, since a device conversion cached
    # before `jax_enable_x64` was toggled stays float32 (the jax_x64-fixture
    # situation; the gradient tests rely on this promotion).
    wave = _as_float(conventions.canonical_wave() if wave is None else wave)

    if override is not None:
        wave_ed, ed = (_as_float(part) for part in override)
        dtype = jnp.result_type(wave.dtype, wave_ed.dtype, ed.dtype)
        return _interp_wave(wave.astype(dtype), wave_ed.astype(dtype), ed.astype(dtype))

    theta = _as_float(theta_s)
    dtype = jnp.result_type(wave.dtype, theta.dtype)
    grid_np, table_np = load_table()
    grid = jnp.asarray(grid_np, dtype=dtype)
    table = jnp.asarray(table_np, dtype=dtype)

    # Linear in theta_s between the anchors, clamped outside — the same
    # searchsorted rule as the wavelength axis, so a future non-uniform anchor
    # set (M5) needs no code change here.
    anchors = jnp.asarray(ZENITH_ANCHORS, dtype=dtype)
    idx, w = _interp_weights(theta.astype(dtype), anchors)
    spectrum = table[idx - 1] * (1.0 - w[..., None]) + table[idx] * w[..., None]

    return _interp_wave(wave.astype(dtype), grid, spectrum)


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
