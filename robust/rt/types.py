"""
Input pytrees for the forward model.

The three arguments of :func:`robust.rt.forward`, as registered JAX pytrees so
``jit`` / ``vmap`` / ``grad`` traverse them, with light ``jaxtyping`` annotations
on the public signatures.

Because they are pytrees, ``jax.grad`` of a scalar function of an :class:`IOPs`
returns *an* :class:`IOPs` whose fields are the per-field derivatives. That is the
shape the future inversion wants, and it is the reason these are containers rather
than loose arrays.

**Registered with ``jax.tree_util.register_dataclass``, not
``flax.struct.dataclass``.** Both were available and both work; the deciding
argument is dependency direction. These types sit on the analytic path — the ZTT
backbone (M2) needs them and needs nothing from Flax — so having the core data
model import a neural-network library to describe a container would be backwards.
JAX's own mechanism is the more primitive and stable choice, and stdlib
``dataclasses`` means ``dataclasses.replace`` and ordinary ``repr`` come for free.
(Import cost was *not* the argument: measured, ``flax`` adds only ~0.08 s once
``jax`` is loaded. Flax arrives at M3, inside :mod:`robust.rt.emulator`, where it
earns its place.)

**Validation is explicit, never in ``__post_init__``.** Each type has a
``validate()`` that raises ``ValueError`` via :mod:`robust.rt.conventions`.
Validating in ``__post_init__`` would be worse than useless: under ``jit`` or
``vmap`` the fields are tracers with no concrete value, so the check would either
crash or silently pass. Call ``validate()`` where data enters — the loader, a
script — and leave the traced path clean.

**Angles are degrees**, not radians, matching L23 and the design's ``theta_s``
``in {0, 30, 60}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions

__all__ = [  # noqa: RUF022  - argument order of forward(), not alphabetical
    "IOPs",
    "PhaseParams",
    "Geometry",
]

#: A per-wavelength quantity, batched over any leading axes.
Spectrum = Float[Array, "*batch wave"]

#: A per-scene scalar, batched over any leading axes.
Scalar = Float[Array, "*batch"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IOPs:
    """Inherent optical properties on the canonical wavelength grid.

    The water/particle backscattering split is kept explicit rather than summed
    away: ``bb_w`` is a known constant of the medium and ``bb_p`` is what an
    inversion actually retrieves, and the synthesis work shows that separation is
    load-bearing for the physics (design §3). It is also free for us, since L23
    reports both.

    Attributes
    ----------
    a : Array
        Total absorption (m^-1), shape ``(..., n_wave)``.
    bb_w : Array
        Pure-water backscattering (m^-1), same shape as ``a``.
    bb_p : Array
        Particulate backscattering (m^-1), same shape as ``a``.
    """

    a: Spectrum
    bb_w: Spectrum
    bb_p: Spectrum

    @property
    def bb(self) -> Spectrum:
        """Total backscattering ``bb_w + bb_p`` (m^-1). Derived, not a leaf."""
        return self.bb_w + self.bb_p

    @property
    def u(self) -> Spectrum:
        """``bb / (a + bb)``, the single-scattering ratio Gordon-family models use.

        Derived, not a leaf. Note that ``Rrs`` is *not* univocal in ``u`` -- that
        is the central physical point motivating this project -- so ``u`` is a
        convenience, not a sufficient state.
        """
        bb = self.bb
        return bb / (self.a + bb)

    @property
    def n_wave(self) -> int:
        """Number of wavelength bands (the trailing axis)."""
        return self.a.shape[-1]

    @classmethod
    def from_total_bb(
        cls,
        a: Spectrum,
        bb: Spectrum,
        wave: Float[Array, " wave"] | None = None,
        *,
        bb_w_mode: str = "clamp",
    ) -> IOPs:
        """Build from total backscattering, splitting off pure water.

        The one place the water/particle split is performed, so every caller
        splits it the same way: ``bb_p = bb - bb_w(wave)`` with ``bb_w`` from
        :mod:`robust.rt.conventions`.

        ``bb_w`` is broadcast to ``a``'s full shape rather than stored as a bare
        ``(n_wave,)`` spectrum. It costs a little memory (~1 MB for a full L23
        batch) and buys a real convenience: every leaf then shares the batch
        shape, so plain ``jax.vmap(f, in_axes=0)`` works. Keeping it unbatched
        would force every caller to spell out
        ``in_axes=IOPs(a=0, bb_w=None, bb_p=0)``.

        Parameters
        ----------
        a : Array
            Total absorption (m^-1), shape ``(..., n_wave)``.
        bb : Array
            Total backscattering (m^-1), same shape as ``a``.
        wave : Array, optional
            Wavelengths (nm); defaults to the canonical grid.
        bb_w_mode : str, optional
            Passed to :func:`~robust.rt.conventions.bb_w`. The default
            ``"clamp"`` is what M0-M4 used and keeps every existing number
            identical; a grid reaching past 750 nm should say whether it wants
            ``"extrapolate"`` instead, because the clamp is otherwise silent.

        Returns
        -------
        IOPs

        Notes
        -----
        Does not validate. ``bb < bb_w`` yields a negative ``bb_p``, which is
        physically impossible and usually means ``bb`` was already the non-water
        part; call :meth:`validate` to catch it.

        A dataset that tabulates its own pure-water backscattering (PB24 does)
        should use those values rather than this constructor -- the table here is
        L23's water column, and nothing guarantees another campaign shares it.
        """
        a = jnp.asarray(a)
        bb = jnp.asarray(bb)
        bb_w = conventions.bb_w(
            conventions.canonical_wave() if wave is None else wave, mode=bb_w_mode
        )
        bb_w = jnp.broadcast_to(bb_w, a.shape)
        return cls(a=a, bb_w=bb_w, bb_p=bb - bb_w)

    def validate(self, wave: Float[Array, " wave"] | None = None, *, grid=None) -> None:
        """Raise ``ValueError`` unless the IOPs are physical and consistent.

        Boundary check only -- do not call inside ``jit``/``vmap``; see the module
        docstring.

        Parameters
        ----------
        wave : Array, optional
            If given, also require it to be a known grid and to match the
            trailing axis.
        grid : None, str, or WaveGrid, optional
            Which grid ``wave`` must be; ``None`` means L23's canonical grid, so
            every pre-M5 call site keeps its meaning. The trailing axis is
            checked against **that grid's** band count rather than against
            :data:`~robust.rt.conventions.N_WAVE`, which is what let this check
            be L23-only.

        Raises
        ------
        ValueError
            On a shape mismatch, or a non-finite or negative value.
        """
        if self.bb_w.shape != self.a.shape or self.bb_p.shape != self.a.shape:
            raise ValueError(
                f"IOPs: shapes must match; a {self.a.shape}, "
                f"bb_w {self.bb_w.shape}, bb_p {self.bb_p.shape}"
            )
        conventions.check_iop(self.a, "IOPs.a")
        conventions.check_iop(self.bb_w, "IOPs.bb_w")
        conventions.check_iop(self.bb_p, "IOPs.bb_p")
        if wave is not None:
            g = conventions.wave_grid(grid)
            conventions.check_wave(wave, grid=g)
            if self.n_wave != g.n_wave:
                raise ValueError(
                    f"IOPs: trailing axis {self.n_wave} does not match the "
                    f"{g.name} grid ({g.n_wave})"
                )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhaseParams:
    """Explicit phase-function descriptor.

    **This class is the extension point of the whole API, and M5 exercised it.**
    Week 1 carried a single scalar, the particulate backscattering ratio ``B_p``
    (design §4.2). M5 task 14 added the fuller ZTT backward-VSF parameters as
    *additional optional fields defaulting to ``None``* -- which is why the design
    insisted the phase function be a container rather than a bare array. Adding
    them changed neither :func:`robust.rt.forward`'s signature nor any existing
    call site, and every test written before them passed untouched.

    Two consequences of the ``None`` default worth knowing. A field left ``None``
    contributes no leaves, so gradients and ``tree_map`` ignore it. But the
    *treedef* does change when it becomes non-``None``, so ``jit`` will recompile
    once for each variant -- correct behaviour, and cheap, but visible in a
    profile.

    Attributes
    ----------
    beta_tilde_pi : Array or None
        **M5 (task 14).** The particulate backward phase function at exact
        backscatter, ``Pbb(180°) = βp(180°)/bb_p``, in sr^-1 — the ``β̃(π)`` the
        design names as the first of the fuller ZTT backward-VSF parameters
        (design §4.2). ``None`` means "use the fixed Sullivan & Twardowski (2009)
        shape", which is what M0-M4 did, so a ``None`` here reproduces every
        earlier number exactly.
    backward_slope : Array or None
        **M5 (task 14).** The second backward-VSF parameter: a dimensionless tilt
        of the shape across the backward hemisphere, 0 meaning "Sullivan's shape
        unchanged". See :func:`robust.rt.ztt.P_bb_from_phase` for the exact form
        and for what it does and does not claim.

        **Neither field is calibrated.** They are the *axis* the design asked for,
        plumbed through and gradient-checked; nothing in this repository has
        fitted them, because PB24 prescribes its phase functions and does not
        tabulate ``βp(ψ)``. Treat them as inputs to sweep, not as retrieved
        quantities.
    B_p : Array
        Particulate backscattering ratio ``bb_p / b_p``, dimensionless, typically
        ~0.005-0.03. Realized through a Fournier-Forand phase function. May be a
        per-scene scalar or a spectrum; any shape that broadcasts against the IOP
        spectra is accepted, and M1's loader records what L23 provides.
    """

    B_p: Float[Array, "..."]
    beta_tilde_pi: Float[Array, "..."] | None = None
    backward_slope: Float[Array, "..."] | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` unless ``B_p`` is a physical ratio.

        Checks only that it is finite and in ``(0, 1]`` -- the definitional bound
        for a backscattering *ratio*. The much tighter ~[0.004, 0.03] expected of
        real particles is the data loader's business (M1 task 3), not a
        type-level invariant: a synthetic sweep may legitimately go outside it.

        Raises
        ------
        ValueError
            If ``B_p`` is non-finite, non-positive, or greater than one.
        """
        arr = np.asarray(self.B_p, dtype=float)
        conventions.check_iop(arr, "PhaseParams.B_p")
        if np.any(arr <= 0.0) or np.any(arr > 1.0):
            raise ValueError(
                f"PhaseParams.B_p: must lie in (0, 1]; got range "
                f"[{arr.min():.6g}, {arr.max():.6g}]. B_p is a ratio "
                "bb_p / b_p, not a coefficient"
            )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Geometry:
    """Viewing and illumination geometry, in **degrees**.

    L23 fixes the view at nadir and varies only the solar zenith, but the API
    carries the full geometry from day one so M5's BRDF runs need no interface
    change (design §1).

    Attributes
    ----------
    theta_s : Array
        Solar zenith angle (degrees); 0 = sun overhead.
    theta_v : Array
        Sensor zenith angle (degrees); 0 = nadir view, as in L23.
    dphi : Array
        Sensor-sun relative azimuth (degrees).
    wind : Array or None
        Wind speed (m/s), optional surface-roughness input. ``None`` until a
        reference dataset varies it.
    """

    theta_s: Scalar
    theta_v: Scalar
    dphi: Scalar
    wind: Scalar | None = None

    @classmethod
    def nadir(cls, theta_s: Scalar, wind: Scalar | None = None) -> Geometry:
        """Nadir-viewing geometry at a given solar zenith -- the L23 case.

        Parameters
        ----------
        theta_s : Array
            Solar zenith angle (degrees).
        wind : Array, optional
            Wind speed (m/s).

        Returns
        -------
        Geometry
            With ``theta_v = 0`` and ``dphi = 0``.
        """
        theta_s = jnp.asarray(theta_s)
        zero = jnp.zeros_like(theta_s)
        return cls(theta_s=theta_s, theta_v=zero, dphi=zero, wind=wind)

    def validate(self) -> None:
        """Raise ``ValueError`` unless the angles are in range.

        Boundary check only -- do not call inside ``jit``/``vmap``.

        Raises
        ------
        ValueError
            If a zenith is outside [0, 90] deg, the azimuth outside [0, 360] deg,
            or the wind speed negative. A zenith beyond 90 deg puts the sun below
            the horizon or the sensor looking upward -- bad data or a convention
            mismatch.

        Notes
        -----
        This cannot catch the opposite unit error. An angle mistakenly given in
        *radians* is numerically small (30 deg -> 0.52), so it lands inside the
        valid range and passes; it would show up downstream as a poor fit rather
        than as an exception. A test pins that blind spot explicitly.
        """
        for name, value, hi in (
            ("theta_s", self.theta_s, 90.0),
            ("theta_v", self.theta_v, 90.0),
            ("dphi", self.dphi, 360.0),
        ):
            arr = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Geometry.{name}: non-finite value(s)")
            if np.any(arr < 0.0) or np.any(arr > hi):
                raise ValueError(
                    f"Geometry.{name}: must lie in [0, {hi:.0f}] degrees; got "
                    f"range [{arr.min():.6g}, {arr.max():.6g}] -- degrees, "
                    "not radians"
                )
        if self.wind is not None:
            conventions.check_iop(self.wind, "Geometry.wind")
