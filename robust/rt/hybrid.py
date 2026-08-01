"""
The public forward model — **stub (lands in M3)**.

``Rrs = Rrs_ZTT + ΔRrs``: one JAX function, differentiable end to end. The
``mode`` flag selects the analytic backbone alone, the emulator alone, or the
hybrid, so all three options compare on identical splits rather than on
separately prepared data.

This module owns the only signature the rest of the world should depend on. The
inversion is out of scope (design §1), but it is the caller this interface is
shaped for: ``jax.grad`` / ``jax.jacobian`` of :func:`forward` give the input
sensitivities it will need.
"""

from __future__ import annotations

#: The three comparable configurations (design §4.5).
MODES = ("ztt", "emulator", "hybrid")


def forward(iops, phase_params, geometry, wave, mode="hybrid"):
    """Elastic remote-sensing reflectance ``Rrs(wave)``.

    Differentiable in JAX and batched over leading axes, so a full L23 batch
    (3320 scenes × 81 λ) is one call.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Absorption and the water/particle backscattering split, on ``wave``.
    phase_params : robust.rt.types.PhaseParams
        Explicit phase-function descriptor (week 1: ``B_p = bb_p / b_p``).
    geometry : robust.rt.types.Geometry
        Solar zenith, sensor zenith, relative azimuth; optional wind speed.
    wave : jax.Array
        Wavelengths (nm).
    mode : str, optional
        One of :data:`MODES`. Default ``'hybrid'``.

    Returns
    -------
    jax.Array
        ``Rrs(wave)``, sr⁻¹, shape ``(..., n_wave)``.

    Raises
    ------
    NotImplementedError
        Always, in the M0 scaffold.
    """
    raise NotImplementedError(
        "forward lands in M3 (see design/rt_elastic_model_coding_plan.md); "
        f"mode={mode!r}"
    )
