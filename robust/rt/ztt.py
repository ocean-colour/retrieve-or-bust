"""
ZTT analytic backbone — **stub (lands in M2)**.

The Twardowski & Tonizzo (2018) ocean-colour model, transcribed into pure JAX so
it is differentiable: an ``Rrs`` relation in which the backward volume scattering
function enters *explicitly*, rather than being absorbed into fitted constants.
That explicitness is the reason it is the backbone — it gives the hybrid an
interpretable handle on phase-function shape, and it is scored standalone as the
analytical benchmark before any learned residual is added.

Source paper: ``context/RT/twardowski2018.pdf`` (Applied Sciences, "Ocean Color
Analytical Model Explicitly Dependent on the VSF").

Planned contents (design §4.3, coding plan M2)
----------------------------------------------
- :func:`Rrs_ZTT` below, as pure functions with no learned parameters.
- Week 1 passes phase-function shape as ``B_p``; the fuller ZTT backward-VSF
  parameterization slots in later behind the same signature.

M2 gates this against a reference case quoted in the paper *and* against
``jax.grad`` versus central finite differences. If a term in the paper proves
ambiguous, the coding plan's de-risking fallback is a Gordon/O25-in-JAX backbone
behind this same interface — a swap that must not reach :func:`robust.rt.forward`.
"""

from __future__ import annotations


def Rrs_ZTT(iops, phase_params, geometry, wave):
    """Remote-sensing reflectance from the ZTT analytic model.

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

    Returns
    -------
    jax.Array
        ``Rrs(wave)``, sr⁻¹, shape ``(..., n_wave)``; batched over leading axes.

    Raises
    ------
    NotImplementedError
        Always, in the M0 scaffold.
    """
    raise NotImplementedError(
        "Rrs_ZTT lands in M2 (see design/rt_elastic_model_coding_plan.md)"
    )
