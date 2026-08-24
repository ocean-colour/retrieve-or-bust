"""
The public forward model — ``Rrs = Rrs_ZTT + ΔRrs`` (M3).

One JAX function, differentiable end to end. The ``mode`` flag selects the analytic
backbone alone, the learned correction alone, or the hybrid, so all three options
compare on identical splits rather than on separately prepared data.

This module owns the only signature the rest of the world should depend on. The
inversion is out of scope (design §1), but it is the caller this interface is shaped
for: ``jax.grad`` / ``jax.jacobian`` of :func:`forward` give the input sensitivities
it will need, and a test pins them against finite differences.

**Additivity holds in ``rrs`` space, not in ``Rrs`` space.** The correction is
defined and trained below the surface, so
``rrs_forward(hybrid) == rrs_forward(ztt) + rrs_forward(emulator)`` exactly, while the
same identity in ``Rrs`` fails — the air-water interface (Lee 2002,
``A·rrs/(1 − B·rrs)``) is non-linear. Score in ``rrs`` (design §6); use
:func:`forward` when you want the above-water quantity an instrument would see.

**What ``mode="emulator"`` means here, and what it does not.** The design's three
options are analytic-only, learned-only, and hybrid. But M3's emulator is
parameterised as a *relative* correction, ``Δrrs = δ · rrs_ZTT`` (see
:mod:`robust.rt.emulator`, decision 1), so "the learned part on its own" is the
**correction term** ``Δrrs`` — not a standalone learned model that could replace the
physics. A genuine learned-only comparison needs a differently trained network, one
predicting ``rrs`` outright across four decades of dynamic range; that is a model to
add beside PR05 and O25 in M4's protocol, not a flag on this function. The mode is
still worth having: it isolates the learned contribution, which is exactly the number
M3 has to report.

**On the extrapolation warning.** M3 measured that the emulator's accuracy at an
unseen solar zenith is seed-dependent and can be worse than the backbone it corrects
(:mod:`robust.rt.emulator`, decision 4). JXP's call was that we do not use it at
larger angles without telling the user, so any mode that involves the emulator checks
the inputs against the trained :attr:`~robust.rt.emulator.Emulator.domain` and warns.
The check needs concrete values, so under ``jit`` it is skipped — deliberately, and
documented on :func:`forward`; ``jit`` is the hot path, and a check that cannot run
there should not silently pretend to.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from . import conventions
from . import inelastic as _inelastic
from . import ztt as _ztt

__all__ = [  # noqa: RUF022  - grouped by role
    "MODES",
    "OUT_OF_DOMAIN_POLICIES",
    "forward",
    "rrs_forward",
    "DomainWarning",
]

#: The three comparable configurations (design §4.5).
MODES = ("ztt", "emulator", "hybrid")

#: Modes whose result depends on the learned emulator.
_LEARNED_MODES = ("emulator", "hybrid")

#: What to do when the emulator is asked for inputs outside the accepted range.
#:
#: ``"warn"`` (the default, and the behaviour since M3) evaluates the emulator anyway
#: and raises :class:`DomainWarning`. ``"ztt"`` additionally **zeroes the learned
#: correction there**, so the model degrades to the analytic backbone exactly where
#: M3 measured the emulator to be unreliable.
#:
#: The second is the other half of JXP's instruction — *"we won't use the emulator at
#: larger angles (or will warn the user)"* (prompt 4, Q6; chosen in prompt 5, Q7). It
#: is an option rather than the default because switching it on changes numbers, and a
#: model whose output depends on a flag nobody set is its own kind of trap. The
#: threshold is :data:`robust.rt.emulator.SUPPORTED_THETA_S` for the solar zenith and
#: the trained range for everything else.
OUT_OF_DOMAIN_POLICIES = ("warn", "ztt")


class DomainWarning(UserWarning):
    """The emulator was asked for inputs outside its training range.

    Its own category, so a caller can promote it to an error
    (``warnings.simplefilter("error", DomainWarning)``) in a pipeline that must not
    silently extrapolate, or silence it once for a study that means to.
    """


def _resolve_emulator(emulator):
    """The caller's emulator, or the packaged default."""
    if emulator is not None:
        return emulator
    from . import emulator as _emulator

    return _emulator.load_default()


def _is_traced(*trees) -> bool:
    """Whether any leaf of any argument is a JAX tracer.

    Checks **every** leaf of the input pytrees, which is not fussiness: ``jax.grad``
    w.r.t. a single input traces only that one, so an earlier version that looked at
    ``iops.a`` and ``geometry.theta_s`` alone declared "not traced" while
    differentiating w.r.t. ``bb_p`` or ``B_p`` — and the domain check then died in
    ``np.asarray`` with a ``TracerArrayConversionError``. Differentiating w.r.t. the
    backscattering alone is precisely the inversion's use case, so that was the case
    that mattered most. A regression test pins each input separately.
    """
    return any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(trees)
    )


def _check_domain(emulator, iops, phase_params, geometry, wave) -> None:
    """Warn if the inputs fall outside the emulator's training range.

    Silent when the inputs are traced (no concrete values to compare) or when the
    emulator carries no domain.
    """
    if emulator.domain is None or _is_traced(iops, phase_params, geometry, wave):
        return
    report = emulator.out_of_domain(iops, phase_params, geometry, wave)
    if not report:
        return
    parts = [
        f"{b.feature} {b.fraction:.1%} of values outside [{b.lo:.4g}, {b.hi:.4g}], "
        f"worst {b.worst:.4g} — {b.excess:.0%} of the trained span beyond it"
        for b in report.values()
    ]
    warnings.warn(
        "the emulator is being evaluated outside its training range, where M3 "
        "measured its accuracy to be unreliable and occasionally worse than the "
        "analytic backbone: " + "; ".join(parts) + ". Consider mode='ztt'. "
        "(cos_theta_s decreases with solar zenith, so a low sun breaches the "
        "lower bound.)",
        DomainWarning,
        stacklevel=3,
    )


def rrs_forward(
    iops,
    phase_params,
    geometry,
    wave: Float[Array, " wave"] | None = None,
    mode: str = "hybrid",
    *,
    inelastic=None,
    emulator=None,
    check_domain: bool = True,
    on_out_of_domain: str = "warn",
) -> Float[Array, "*batch wave"]:
    """Subsurface remote-sensing reflectance ``rrs(wave)`` — the scored quantity.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Absorption and the water/particle backscattering split, on ``wave``.
        The optional ``a_ph`` field is ignored on the elastic path.
    phase_params : robust.rt.types.PhaseParams
        Explicit phase-function descriptor (week 1: ``B_p = bb_p / b_p``).
    geometry : robust.rt.types.Geometry
        Solar zenith, sensor zenith, relative azimuth; optional wind speed.
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid.
    mode : str, optional
        One of :data:`MODES`. Default ``'hybrid'``.

        - ``'ztt'`` — the analytic backbone alone, exactly
          :func:`robust.rt.ztt.rrs_ZTT`. Takes no emulator and cannot warn.
        - ``'emulator'`` — the learned correction ``Δrrs`` **alone**, which is a
          term and not a model; see the module docstring.
        - ``'hybrid'`` — ``rrs_ZTT + Δrrs``, the M3 deliverable.
    inelastic : robust.rt.types.Inelastic, optional
        Configuration of the inelastic processes (Raman scattering, chlorophyll
        fluorescence — inelastic design §3). ``None`` (the default) is the
        elastic model, **bit-identical by construction** to the pre-extension
        hybrid: the ``None`` branch returns the elastic result object
        untouched, rather than adding terms that happen to be zero. A test
        pins the fixture output hash. With ``raman=True`` (M2 task 1) the
        elastic result is multiplied by the analytic
        :func:`robust.rt.inelastic.raman_factor` in ``Rrs`` space (design §2;
        ``f_R = f_phys`` until M3 adds δ_R). ``fluorescence=True`` — the
        default — raises ``NotImplementedError`` until M2 task 2 lands the
        kernel; ask for the Raman-only model with
        ``Inelastic(fluorescence=False)``. Incompatible with
        ``mode='emulator'`` (a term, not a model — ``ValueError``).
    emulator : robust.rt.emulator.Emulator, optional
        Defaults to the packaged weights
        (:func:`robust.rt.emulator.load_default`). Ignored for ``mode='ztt'``, which
        is what lets the analytic path run without the ML stack or the weights file.
    check_domain : bool, optional
        Warn (:class:`DomainWarning`) when the inputs lie outside the emulator's
        training range. Default ``True``. Skipped automatically whenever any input
        is traced — under ``jit``, and under ``jax.grad`` even of a single input —
        since the check needs concrete values.
    on_out_of_domain : str, optional
        One of :data:`OUT_OF_DOMAIN_POLICIES`. ``"warn"`` (default) evaluates the
        emulator everywhere; ``"ztt"`` zeroes its correction on samples outside the
        accepted range, so those fall back to the analytic backbone. Unlike
        ``check_domain``, this one is **traceable** and therefore applies under
        ``jit`` and ``grad`` too — a policy that lapsed under compilation would be
        worse than none.

    Returns
    -------
    Array
        ``rrs``, sr⁻¹, shape ``(..., n_wave)``.

    Raises
    ------
    ValueError
        If ``mode`` is not in :data:`MODES`. Listed rather than silently defaulting:
        a typo'd mode returning the hybrid would corrupt a comparison table without
        any visible symptom.
    """
    if mode not in MODES:
        raise ValueError(f"forward: mode must be one of {MODES}; got {mode!r}")
    if on_out_of_domain not in OUT_OF_DOMAIN_POLICIES:
        raise ValueError(
            f"forward: on_out_of_domain must be one of {OUT_OF_DOMAIN_POLICIES}; "
            f"got {on_out_of_domain!r}"
        )
    if inelastic is not None:
        if inelastic.fluorescence:
            # M2 task 2 adds phi_C * K_fl; until then a configuration asking
            # for it must fail loudly, never return an array missing physics.
            raise NotImplementedError(
                "chlorophyll fluorescence lands at M2 task 2 of the inelastic "
                "coding plan; use Inelastic(fluorescence=False) for the "
                "Raman-only model, or inelastic=None for elastic"
            )
        if mode == "emulator":
            raise ValueError(
                "forward: mode='emulator' returns the learned correction term "
                "alone (a term, not a model — see the module docstring); the "
                "inelastic composition applies to a model output. Use "
                "mode='ztt' or mode='hybrid' with inelastic, or "
                "inelastic=None with mode='emulator'"
            )

    rrs_ztt = _ztt.rrs_ZTT(iops, phase_params, geometry, wave)
    if mode == "ztt":
        return _apply_inelastic(rrs_ztt, iops, geometry, wave, inelastic)

    emulator = _resolve_emulator(emulator)
    if check_domain:
        _check_domain(emulator, iops, phase_params, geometry, wave)
    delta_rrs = emulator.delta_rrs(iops, phase_params, geometry, wave, rrs_ztt=rrs_ztt)

    if on_out_of_domain == "ztt" and emulator.domain is not None:
        # Traceable, so the policy holds under jit exactly as it does outside it.
        outside = emulator.out_of_domain_mask(iops, phase_params, geometry, wave)
        delta_rrs = jnp.where(outside[..., None], 0.0, delta_rrs)

    if mode == "emulator":
        return delta_rrs
    return _apply_inelastic(rrs_ztt + delta_rrs, iops, geometry, wave, inelastic)


def _apply_inelastic(rrs, iops, geometry, wave, inelastic):
    """Compose the inelastic processes onto an elastic ``rrs`` (design §2).

    The composition law is written in ``Rrs`` space —
    ``Rrs_total = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl`` — so the elastic ``rrs``
    is converted up, composed, and converted back. ``forward``'s final
    ``rrs_to_Rrs`` then undoes the round trip exactly (algebraically; at
    ULP level in float — which is why the ``inelastic=None`` branch returns
    the *same object* untouched: the elastic path stays bit-identical by
    construction, never by cancellation. Notebook 1 §4 measured what that
    round trip does to bits.)

    Fluorescence joins here at M2 task 2 (``+ phi_C * K_fl`` before the
    down-conversion); the caller has already rejected configurations asking
    for it.
    """
    if inelastic is None:
        return rrs
    result = rrs
    if inelastic.raman:
        f_r = _inelastic.raman_factor(iops, geometry, wave)
        result = conventions.Rrs_to_rrs(conventions.rrs_to_Rrs(result) * f_r)
    return result


def forward(
    iops,
    phase_params,
    geometry,
    wave: Float[Array, " wave"] | None = None,
    mode: str = "hybrid",
    *,
    inelastic=None,
    emulator=None,
    check_domain: bool = True,
    on_out_of_domain: str = "warn",
) -> Float[Array, "*batch wave"]:
    """Elastic remote-sensing reflectance ``Rrs(wave)`` — the public forward model.

    Differentiable in JAX and batched over leading axes, so a full L23 batch
    (3320 scenes × 81 λ) is one call.

    Parameters
    ----------
    iops, phase_params, geometry, wave, mode, inelastic, emulator, check_domain, \
on_out_of_domain
        As :func:`rrs_forward`. In particular ``inelastic=None`` (the default)
        is the elastic model, bit-identical by construction to the
        pre-extension output; an instance raises ``NotImplementedError``
        until M2.

    Returns
    -------
    Array
        ``Rrs(wave)``, sr⁻¹, shape ``(..., n_wave)``.

    Raises
    ------
    ValueError
        If ``mode`` is not in :data:`MODES`.

    Notes
    -----
    ``mode='emulator'`` returns the ``Δrrs`` correction put through the interface,
    which is **not** additive with ``mode='ztt'`` in ``Rrs`` space — the conversion
    is non-linear. Use :func:`rrs_forward` when the parts must sum, and for all
    scoring (design §6).

    The gradient is the point of this function, not a by-product: ``jax.grad`` of a
    scalar of it w.r.t. an :class:`~robust.rt.types.IOPs` returns an ``IOPs`` of
    per-field derivatives, which is the shape the future inversion wants.
    """
    return conventions.rrs_to_Rrs(
        rrs_forward(
            iops,
            phase_params,
            geometry,
            wave,
            mode,
            inelastic=inelastic,
            emulator=emulator,
            check_domain=check_domain,
            on_out_of_domain=on_out_of_domain,
        )
    )
