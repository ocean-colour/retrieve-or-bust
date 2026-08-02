"""
Comparison models: the analytical baselines the hybrid must beat.

These are not part of the hybrid. They exist because every accuracy claim in this
project is **relative** -- "beats standard Gordon on the held-out splits" (coding
plan M3/M4) -- so the thing being beaten has to be in the repo, differentiable, and
computed on identical data rather than quoted from a paper.

Standard Gordon (1988) lands at M2; PR05 and O25 join at M4 when the full §6
protocol runs.

**Gordon takes the same arguments as** :func:`robust.rt.forward` **and ignores two
of them.** ``phase_params`` and ``geometry`` are accepted and discarded, which is
not an implementation shortcut but the model's defining limitation: standard Gordon
has no phase-function input and no solar-zenith dependence. L23 says ``Rrs`` falls
by a median 5.1% from 0 deg to 60 deg, so a zenith-blind model *must* mis-fit at
least one angle. That gap is what the ZTT backbone and the residual emulator are for,
and keeping the signatures interchangeable is what lets M4 score all of them in one
loop.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from . import conventions

__all__ = [  # noqa: RUF022  - grouped by role
    "G1_GORDON",
    "G2_GORDON",
    "rrs_gordon",
    "Rrs_gordon",
]

#: Canonical Gordon (1988) coefficients for ``rrs = G1 u + G2 u^2``. Fixed, not
#: fitted -- the "standard Gordon" of the coding plan's gates. The synthesis figures
#: (``context/RT/make_rt_elastic_figures.py``) use these same two numbers, so our
#: rRMS is comparable with the ladder in ``context/RT/fig_rrms_ladder.csv``.
G1_GORDON = 0.0949
G2_GORDON = 0.0794


def rrs_gordon(
    iops,
    phase_params=None,
    geometry=None,
    wave: Float[Array, " wave"] | None = None,
    *,
    g1: float = G1_GORDON,
    g2: float = G2_GORDON,
) -> Float[Array, "*batch wave"]:
    """Subsurface reflectance from the standard Gordon (1988) relation.

    ``rrs = g1 * u + g2 * u**2`` with ``u = bb / (a + bb)``.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Absorption and the water/particle backscattering split. Only the derived
        ``u`` is used -- Gordon cannot see the water/particle split, which is one
        of the two things it is blind to.
    phase_params : optional
        **Ignored.** Accepted so the signature matches
        :func:`robust.rt.forward`; standard Gordon has no phase-function input.
    geometry : optional
        **Ignored.** Same reason: no solar-zenith dependence.
    wave : Array, optional
        **Ignored.** The coefficients are wavelength-independent.
    g1, g2 : float, optional
        Coefficients, defaulting to :data:`G1_GORDON` / :data:`G2_GORDON`. Exposed
        so the per-lambda refits in the synthesis work can be reproduced, not
        because the standard values are negotiable.

    Returns
    -------
    Array
        ``rrs``, sr^-1, shape ``(..., n_wave)``.
    """
    u = iops.u
    return g1 * u + g2 * u**2


def Rrs_gordon(
    iops,
    phase_params=None,
    geometry=None,
    wave: Float[Array, " wave"] | None = None,
    *,
    g1: float = G1_GORDON,
    g2: float = G2_GORDON,
) -> Float[Array, "*batch wave"]:
    """Above-water reflectance from standard Gordon.

    :func:`rrs_gordon` put through the air-water interface.

    Parameters
    ----------
    iops, phase_params, geometry, wave, g1, g2
        As :func:`rrs_gordon`.

    Returns
    -------
    Array
        ``Rrs``, sr^-1, shape ``(..., n_wave)``.

    Notes
    -----
    Accuracy is scored in ``rrs`` space (design §6), so :func:`rrs_gordon` is the
    one the metrics call; this wrapper exists because ``Rrs`` is what a comparison
    against an observation or a plot wants.
    """
    return conventions.rrs_to_Rrs(
        rrs_gordon(iops, phase_params, geometry, wave, g1=g1, g2=g2)
    )
