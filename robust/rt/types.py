"""
Input pytrees for the forward model.

The arguments of :func:`robust.rt.forward` — the elastic three plus the
:class:`Inelastic` configuration (inelastic coding plan, M0) — as registered JAX
pytrees so ``jit`` / ``vmap`` / ``grad`` traverse them, with light ``jaxtyping``
annotations on the public signatures.

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

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions

__all__ = [  # noqa: RUF022  - argument order of forward(), not alphabetical
    "IOPs",
    "PhaseParams",
    "Geometry",
    "Inelastic",
    "CDOMFl",
    "EMISSION_SHAPES",
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
    a_ph : Array or None
        Phytoplankton absorption (m^-1), same shape as ``a``; the fluorescence
        source term is ``b_F = phi_C * a_ph`` (inelastic design §3). Optional
        (default ``None``) and **ignored by the elastic path**: callers that
        already split ``a`` into components pay nothing, callers with only bulk
        ``a`` get elastic + Raman, and fluorescence *requires* the split — a
        physical requirement, not an API whim. Follows the ``Geometry.wind``
        precedent: an unset optional field contributes no leaves, but the
        treedef changes once it is set, so ``jit`` recompiles once per variant.
    a_cdom : Array or None
        CDOM absorption (m^-1), same shape as ``a``; the CDOM-fluorescence
        source term is proportional to ``a_cdom`` (CDOM design §2,
        ``design/rt_cdom_fluorescence_model.md``). Optional (default ``None``)
        and **ignored by the elastic, Raman, and Chl-fl paths**: only the
        CDOM-fluorescence term (``Inelastic.cdom_fl``) *requires* it — again a
        physical requirement, not an API whim. Same ``Geometry.wind``-style
        pytree behaviour as ``a_ph``: no leaves when unset, one treedef change
        (and ``jit`` recompile) when set. L23 populates it from a_g, which is
        stored separately from a_nap.
    """

    a: Spectrum
    bb_w: Spectrum
    bb_p: Spectrum
    a_ph: Spectrum | None = None
    a_cdom: Spectrum | None = None

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
        a_ph: Spectrum | None = None,
        a_cdom: Spectrum | None = None,
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
        a_ph : Array, optional
            Phytoplankton absorption (m^-1), passed through unchanged.
        a_cdom : Array, optional
            CDOM absorption (m^-1), passed through unchanged.

        Returns
        -------
        IOPs

        Notes
        -----
        Does not validate. ``bb < bb_w`` yields a negative ``bb_p``, which is
        physically impossible and usually means ``bb`` was already the non-water
        part; call :meth:`validate` to catch it.
        """
        a = jnp.asarray(a)
        bb = jnp.asarray(bb)
        bb_w = conventions.bb_w(conventions.canonical_wave() if wave is None else wave)
        bb_w = jnp.broadcast_to(bb_w, a.shape)
        if a_ph is not None:
            # Broadcast like bb_w, for the same reason: every leaf shares the
            # batch shape, so plain vmap(f, in_axes=0) works (PR #14 review).
            a_ph = jnp.broadcast_to(jnp.asarray(a_ph), a.shape)
        if a_cdom is not None:
            # Broadcast like bb_w, for the same reason: every leaf shares the
            # batch shape, so plain vmap(f, in_axes=0) works (PR #14 review).
            a_cdom = jnp.broadcast_to(jnp.asarray(a_cdom), a.shape)
        return cls(a=a, bb_w=bb_w, bb_p=bb - bb_w, a_ph=a_ph, a_cdom=a_cdom)

    def validate(self, wave: Float[Array, " wave"] | None = None) -> None:
        """Raise ``ValueError`` unless the IOPs are physical and consistent.

        Boundary check only -- do not call inside ``jit``/``vmap``; see the module
        docstring.

        Parameters
        ----------
        wave : Array, optional
            If given, also require it to be the canonical grid and to match the
            trailing axis.

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
        if self.a_ph is not None:
            if self.a_ph.shape != self.a.shape:
                raise ValueError(
                    f"IOPs: a_ph shape {self.a_ph.shape} does not match "
                    f"a {self.a.shape}"
                )
            conventions.check_iop(self.a_ph, "IOPs.a_ph")
            if np.any(np.asarray(self.a_ph) > np.asarray(self.a)):
                raise ValueError(
                    "IOPs: a_ph exceeds total absorption a somewhere -- a_ph is "
                    "a *component* of a, so this is a unit or bookkeeping error"
                )
        if self.a_cdom is not None:
            if self.a_cdom.shape != self.a.shape:
                raise ValueError(
                    f"IOPs: a_cdom shape {self.a_cdom.shape} does not match "
                    f"a {self.a.shape}"
                )
            conventions.check_iop(self.a_cdom, "IOPs.a_cdom")
            if np.any(np.asarray(self.a_cdom) > np.asarray(self.a)):
                raise ValueError(
                    "IOPs: a_cdom exceeds total absorption a somewhere -- a_cdom "
                    "is a *component* of a, so this is a unit or bookkeeping error"
                )
        if wave is not None:
            conventions.check_wave(wave)
            if self.n_wave != conventions.N_WAVE:
                raise ValueError(
                    f"IOPs: trailing axis {self.n_wave} does not match the "
                    f"canonical grid ({conventions.N_WAVE})"
                )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhaseParams:
    """Explicit phase-function descriptor.

    **This class is the extension point of the whole API.** Week 1 carries a
    single scalar, the particulate backscattering ratio ``B_p`` (design §4.2). At
    M5 the fuller ZTT backward-VSF parameters join it as *additional optional
    fields defaulting to ``None``* -- which is why the design insists the phase
    function be a container rather than a bare array. Adding a field changes
    neither :func:`robust.rt.forward`'s signature nor any existing call site.

    Two consequences of the ``None`` default worth knowing. A field left ``None``
    contributes no leaves, so gradients and ``tree_map`` ignore it. But the
    *treedef* does change when it becomes non-``None``, so ``jit`` will recompile
    once for each variant -- correct behaviour, and cheap, but visible in a
    profile.

    Attributes
    ----------
    B_p : Array
        Particulate backscattering ratio ``bb_p / b_p``, dimensionless, typically
        ~0.005-0.03. Realized through a Fournier-Forand phase function. May be a
        per-scene scalar or a spectrum; any shape that broadcasts against the IOP
        spectra is accepted, and M1's loader records what L23 provides.
    """

    B_p: Float[Array, "..."]

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
    Ed : (Array, Array) or None
        Optional downwelling-irradiance override, a ``(wave_Ed, Ed)`` pair of
        1-D arrays (nm, W m^-2 nm^-1) on their own grid (inelastic design §3).
        The inelastic terms need the *shape* of ``Ed(λ)`` (Raman uses the true
        ``Ed(λ')/Ed(λ)`` ratio); when ``None`` — the default — the packaged L23
        spectra interpolated in ``theta_s`` are used (M1's ``ed.py``). The
        elastic path ignores it entirely. This is the seam through which
        real-sky irradiances enter later without an interface change.
    """

    theta_s: Scalar
    theta_v: Scalar
    dphi: Scalar
    wind: Scalar | None = None
    Ed: tuple[Float[Array, " wave_ed"], Float[Array, " wave_ed"]] | None = None

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
        if self.Ed is not None:
            if len(self.Ed) != 2:
                raise ValueError(
                    f"Geometry.Ed: must be a (wave_Ed, Ed) pair; got "
                    f"{len(self.Ed)} elements"
                )
            wave_ed, ed = (np.asarray(part, dtype=float) for part in self.Ed)
            if wave_ed.ndim != 1 or ed.shape != wave_ed.shape:
                raise ValueError(
                    f"Geometry.Ed: wave_Ed and Ed must be 1-D and the same "
                    f"length; got shapes {wave_ed.shape} and {ed.shape}"
                )
            if np.any(np.diff(wave_ed) <= 0.0):
                raise ValueError(
                    "Geometry.Ed: wave_Ed must be strictly increasing (nm)"
                )
            conventions.check_iop(ed, "Geometry.Ed")


#: The fluorescence emission-profile options (inelastic design §4.4). ``'single'``
#: is the validated default; ``'double'`` (the PS I shoulder) is switchable but
#: untested against L23 — reported, never gated.
EMISSION_SHAPES = ("single", "double")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CDOMFl:
    """Configuration of the CDOM-fluorescence term (CDOM design §3, §4).

    The value of :attr:`Inelastic.cdom_fl` when CDOM fluorescence is on:
    ``Rrs_cdom = scale * K_cdom * (1 + delta_C)``, with ``K_cdom`` the analytic
    Hawes et al. (1992) kernel (``design/rt_cdom_fluorescence_model.md`` §2).
    A pytree rather than a bare scalar so that M6 can grow shape metadata
    (static fields) without an API break — the same extension-point argument
    made on :class:`PhaseParams`.

    Setting an instance requires ``IOPs.a_cdom`` — the physical source term —
    which ``forward`` enforces at call time, mirroring the
    ``fluorescence``/``a_ph`` guard.

    Attributes
    ----------
    scale : Array or float
        Amplitude ``s_C`` on the fixed Hawes reference kernel, dimensionless;
        scalar or batched per scene. Default 1.0 (the reference kernel as
        published). A differentiable pytree *leaf* — it is the
        ``phi_C``-analogue handle for the eventual inversion, so
        ``grad``/``vmap``/``jit`` must traverse it.
    """

    scale: Scalar | float = 1.0

    def validate(self) -> None:
        """Raise ``ValueError`` unless the configuration is usable.

        Boundary check only -- do not call inside ``jit``/``vmap``.

        Raises
        ------
        ValueError
            If ``scale`` is non-finite or not strictly positive. ``scale`` is
            an amplitude on the reference kernel; disable the process with
            ``Inelastic(cdom_fl=None)``, not ``scale=0``.
        """
        arr = np.asarray(self.scale, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("CDOMFl.scale: non-finite value(s)")
        if np.any(arr <= 0.0):
            raise ValueError(
                f"CDOMFl.scale: the amplitude on the Hawes reference kernel "
                f"must be positive; got range "
                f"[{arr.min():.6g}, {arr.max():.6g}]. Disable CDOM "
                "fluorescence with Inelastic(cdom_fl=None), not scale=0"
            )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Inelastic:
    """Configuration of the inelastic processes (inelastic design §3).

    The fifth argument of :func:`robust.rt.forward`. Passing ``None`` instead of
    an instance keeps the elastic path **bit-identical by construction** — the
    ``None`` branch takes the pre-existing code route, it does not multiply by
    one or add zero. Passing an instance raises ``NotImplementedError`` until M2
    lands the physics; the type itself is pinned at M0 so every later milestone
    builds against the same interface.

    **Leaves vs static fields.** ``phi_C`` (and, when set, ``cdom_fl.scale``)
    are pytree *leaves*: ``phi_C`` is a differentiable input — retrieving it is
    the point (design DQ4) — so ``grad``/``vmap`` must traverse it, and the
    nested :class:`CDOMFl` contributes its own leaf the same way. The process
    switches ``raman``/``fluorescence`` and the ``emission_shape`` selector are
    *static* metadata: they select code paths, so tracing them makes no sense and
    ``jit`` specializes on them instead (one recompile per configuration, like
    the treedef change documented on :class:`PhaseParams`).

    Attributes
    ----------
    phi_C : Array or float
        Chlorophyll fluorescence quantum yield, dimensionless; scalar or batched
        per scene. Default 0.02, the L23 assessment's value. The fluorescence
        source is ``b_F = phi_C * a_ph``, linear in ``phi_C`` by construction.
    raman : bool
        Include Raman scattering by water. Static. Default ``True``.
    fluorescence : bool
        Include chlorophyll-a fluorescence. Static. Default ``True``. Requires
        ``IOPs.a_ph`` — a physical requirement (the source term), enforced by
        ``forward`` when the physics lands (M2), not here.
    emission_shape : str
        One of :data:`EMISSION_SHAPES`. Static. Default ``'single'``.
    cdom_fl : CDOMFl or None
        CDOM-fluorescence configuration (CDOM design §3,
        ``design/rt_cdom_fluorescence_model.md``). **``None`` (the default)
        keeps the process off** — load-bearing: the shipped X4 truth omits CDOM
        fluorescence, so ``Inelastic(..., cdom_fl=None)`` must stay
        bit-identical to the shipped inelastic output. Set a :class:`CDOMFl`
        instance to turn the term on; that *requires* ``IOPs.a_cdom`` (the
        physical source term), which ``forward`` enforces at call time, and
        the nested ``scale`` becomes an additional differentiable leaf.
    """

    phi_C: Scalar | float = 0.02
    raman: bool = field(default=True, metadata=dict(static=True))
    fluorescence: bool = field(default=True, metadata=dict(static=True))
    emission_shape: str = field(default="single", metadata=dict(static=True))
    cdom_fl: CDOMFl | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` unless the configuration is usable.

        Boundary check only -- do not call inside ``jit``/``vmap``.

        Raises
        ------
        ValueError
            If ``phi_C`` is non-finite or outside ``(0, 1]`` (it is a quantum
            yield; real values are ~0.005-0.06, but only the definitional bound
            is a type-level invariant -- the looser philosophy of
            :meth:`PhaseParams.validate`); if ``emission_shape`` is not in
            :data:`EMISSION_SHAPES`; or if ``cdom_fl`` is set but is not a
            valid :class:`CDOMFl` (wrong type -- e.g. a bare scalar, the
            pre-M5 reserved-hook signature -- or one whose own ``validate()``
            rejects it).
        """
        arr = np.asarray(self.phi_C, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Inelastic.phi_C: non-finite value(s)")
        if np.any(arr <= 0.0) or np.any(arr > 1.0):
            raise ValueError(
                f"Inelastic.phi_C: a quantum yield must lie in (0, 1]; got "
                f"range [{arr.min():.6g}, {arr.max():.6g}]. Disable "
                "fluorescence with fluorescence=False, not phi_C=0"
            )
        if self.emission_shape not in EMISSION_SHAPES:
            raise ValueError(
                f"Inelastic.emission_shape: must be one of {EMISSION_SHAPES}; "
                f"got {self.emission_shape!r}"
            )
        if self.cdom_fl is not None:
            if not isinstance(self.cdom_fl, CDOMFl):
                raise ValueError(
                    f"Inelastic.cdom_fl: must be a CDOMFl instance (or None to "
                    f"keep CDOM fluorescence off); got "
                    f"{type(self.cdom_fl).__name__}. A bare scalar was the "
                    "pre-M5 reserved-hook signature -- wrap the amplitude as "
                    "CDOMFl(scale=...)"
                )
            self.cdom_fl.validate()
