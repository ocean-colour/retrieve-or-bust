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

from . import cdom_fl as _cdom_fl
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


def _resolve_corrections(corrections):
    """The heads `_apply_inelastic` should use, or ``None`` for analytic.

    ``None`` (the default) means the packaged trained heads —
    :func:`robust.rt.inelastic_corr.load_default`, which degrades to
    analytic-only behind a single ``MissingCorrectionWarning`` while the M3
    weights do not exist yet. ``False`` is analytic-only, explicit and
    silent (how the M2 characterization tests pin the analytic terms). An
    explicit :class:`~robust.rt.inelastic_corr.CorrectionHeads` is used as
    given. Only called when an inelastic process is actually on, so the
    elastic path never imports the ML stack (or warns) on its account.
    """
    if corrections is False:
        return None
    if corrections is None:
        from . import inelastic_corr

        return inelastic_corr.load_default()
    return corrections


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
    corrections=None,
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
        ``f_R = f_phys`` until M3 adds δ_R). With ``fluorescence=True`` —
        the default (M2 task 2) — the additive ``phi_C *``
        :func:`robust.rt.inelastic.fluorescence_kernel` term joins in the
        same space; it **requires** ``iops.a_ph``, the physical source term
        (``ValueError`` otherwise — use ``Inelastic(fluorescence=False)``
        for the Raman-only model). With ``cdom_fl=CDOMFl(...)`` (M5 task 5;
        ``None`` stays the default — the shipped X4 truth omits CDOM
        fluorescence) the additive ``scale *``
        :func:`robust.rt.cdom_fl.cdom_kernel` term joins too; it
        **requires** ``iops.a_cdom`` the same way. Incompatible with
        ``mode='emulator'`` (a term, not a model — ``ValueError``).
    corrections : optional
        The learned M3 corrections applied on top of the analytic terms
        (``f_R = 1 + (f_phys − 1)(1 + δ_R)``; ``× (1 + δ_F)`` on the
        kernel). ``None`` (default) — the packaged trained heads
        (:func:`robust.rt.inelastic_corr.load_default`), degrading to
        analytic-only behind one ``MissingCorrectionWarning`` while the M3
        weights are untrained/absent. ``False`` — analytic-only, explicit
        and silent (comparisons; the M2 characterization pins). Or an
        explicit :class:`~robust.rt.inelastic_corr.CorrectionHeads`.
        Ignored (never resolved, never warns) when ``inelastic`` is ``None``
        or all processes are off — the elastic path owes nothing to the ML
        stack.
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
        if mode == "emulator":
            raise ValueError(
                "forward: mode='emulator' returns the learned correction term "
                "alone (a term, not a model — see the module docstring); the "
                "inelastic composition applies to a model output. Use "
                "mode='ztt' or mode='hybrid' with inelastic, or "
                "inelastic=None with mode='emulator'"
            )
        if inelastic.fluorescence and iops.a_ph is None:
            # The kernel raises the same requirement; checking here fails
            # fast — before the emulator loads — where the M0/M1 guard sat.
            raise ValueError(
                "forward: Inelastic.fluorescence requires IOPs.a_ph — the "
                "fluorescence source term is b_F = phi_C * a_ph, and bulk "
                "absorption cannot stand in for the phytoplankton component. "
                "Provide a_ph (e.g. IOPs.from_total_bb(..., a_ph=...)) or "
                "use Inelastic(fluorescence=False)"
            )
        if inelastic.cdom_fl is not None and iops.a_cdom is None:
            # Same fail-fast as the a_ph guard above: the CDOM-fluorescence
            # source term is proportional to a_cdom (CDOM design §2), and bulk
            # absorption cannot stand in for the CDOM component.
            raise ValueError(
                "forward: Inelastic.cdom_fl requires IOPs.a_cdom — the "
                "CDOM-fluorescence source term is proportional to a_cdom, "
                "and bulk absorption cannot stand in for the CDOM component. "
                "Provide a_cdom (e.g. IOPs.from_total_bb(..., a_cdom=...)) "
                "or use Inelastic(cdom_fl=None)"
            )

    heads = None
    if inelastic is not None and (inelastic.raman or inelastic.fluorescence):
        heads = _resolve_corrections(corrections)

    rrs_ztt = _ztt.rrs_ZTT(iops, phase_params, geometry, wave)
    if mode == "ztt":
        return _apply_inelastic(rrs_ztt, iops, geometry, wave, inelastic, heads)

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
    return _apply_inelastic(rrs_ztt + delta_rrs, iops, geometry, wave, inelastic, heads)


def _apply_inelastic(rrs, iops, geometry, wave, inelastic, heads=None):
    """Compose the inelastic processes onto an elastic ``rrs`` (design §2).

    The composition law is written in ``Rrs`` space —
    ``Rrs_total = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl + Rrs_cdom`` — so the
    elastic ``rrs`` is converted up, composed (Raman multiplies, fluorescence
    adds ``phi_C * K_fl (1 + δ_F)``, CDOM fluorescence adds
    ``scale * K_cdom`` — CDOM design §2), and converted back. ``forward``'s
    final ``rrs_to_Rrs`` then undoes the round trip exactly (algebraically;
    at ULP level in float — which is why the ``inelastic=None`` and all-off
    branches return the *same object* untouched: the elastic path stays
    bit-identical by construction, never by cancellation. Notebook 1 §4
    measured what that round trip does to bits.)

    ``heads`` (resolved by the caller — :func:`_resolve_corrections`)
    carries the M3 learned corrections; a ``None`` container or a ``None``
    field means that term stays purely analytic (``f_R = f_phys``,
    ``δ_F = 0``) — by *omission* of the correction arithmetic, not by
    multiplying with a computed zero.
    """
    # cdom_fl counts as an active process here: without it, a caller setting
    # cdom_fl alone (raman=False, fluorescence=False) would silently get the
    # untouched elastic rrs back — a missing term, not a style nit (M5 task 5).
    if inelastic is None or not (
        inelastic.raman or inelastic.fluorescence or inelastic.cdom_fl is not None
    ):
        return rrs
    result = conventions.rrs_to_Rrs(rrs)
    if inelastic.raman:
        f_r = _inelastic.raman_factor(iops, geometry, wave)
        if heads is not None and heads.raman is not None:
            from . import inelastic_corr

            f_r = inelastic_corr.corrected_raman_factor(
                heads.raman.delta(iops, geometry, wave), f_r
            )
        result = result * f_r
    if inelastic.fluorescence:
        # K_fl raises the clear a_ph-is-required error; phi_C is a leaf
        # (possibly batched per scene), aligned onto the wavelength axis.
        k_fl = _inelastic.fluorescence_kernel(
            iops, geometry, wave, emission_shape=inelastic.emission_shape
        )
        if heads is not None and heads.fl is not None:
            from . import inelastic_corr

            k_fl = inelastic_corr.corrected_fluorescence(
                heads.fl.delta(iops, geometry, wave), k_fl
            )
        result = result + jnp.asarray(inelastic.phi_C)[..., None] * k_fl
    if inelastic.cdom_fl is not None:
        # K_cdom raises the clear a_cdom-is-required error; scale is the
        # CDOMFl leaf (possibly batched per scene), aligned onto the
        # wavelength axis. Task 6 will multiply this by (1 + delta_C) once
        # the head exists; until then this term IS the full CDOM
        # contribution — an absent/untrained head is (1 + 0) = 1 (CFQ3).
        k_cdom = _cdom_fl.cdom_kernel(iops, geometry, wave)
        result = result + jnp.asarray(inelastic.cdom_fl.scale)[..., None] * k_cdom
    return conventions.Rrs_to_rrs(result)


def forward(
    iops,
    phase_params,
    geometry,
    wave: Float[Array, " wave"] | None = None,
    mode: str = "hybrid",
    *,
    inelastic=None,
    corrections=None,
    emulator=None,
    check_domain: bool = True,
    on_out_of_domain: str = "warn",
) -> Float[Array, "*batch wave"]:
    """Elastic remote-sensing reflectance ``Rrs(wave)`` — the public forward model.

    Differentiable in JAX and batched over leading axes, so a full L23 batch
    (3320 scenes × 81 λ) is one call.

    Parameters
    ----------
    iops, phase_params, geometry, wave, mode, inelastic, corrections, emulator, \
check_domain, on_out_of_domain
        As :func:`rrs_forward`. In particular ``inelastic=None`` (the default)
        is the elastic model, bit-identical by construction to the
        pre-extension output; an instance composes the analytic Raman factor
        and/or the ``phi_C``-linear fluorescence term (M2) onto it.

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
            corrections=corrections,
            emulator=emulator,
            check_domain=check_domain,
            on_out_of_domain=on_out_of_domain,
        )
    )
