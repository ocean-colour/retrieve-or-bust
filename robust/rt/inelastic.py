"""
Analytic inelastic terms — the physics backbone (inelastic coding plan, M2).

JAX ports of the **fixed** BING physics (branch ``inelastic-fixes``), pure and
differentiable, composed into :func:`robust.rt.forward` per the design's
composition law (§2)::

    Rrs_total(λ) = (Rrs_ZTT + ΔRrs) × f_R(λ)  +  Rrs_fl(λ)

**Raman is multiplicative.** :func:`raman_factor` returns
``f_phys = (R_E + R_Raman)/R_E`` in the Sathyendranath & Platt (1998) two-flow
framework — the *self-normalizing* form the assessment validated:
normalization errors cancel in the ratio, which is why BING's Raman was about
right while its (pre-fix) additive fluorescence was ×3 off. Until M3 adds the
learned δ_R, ``f_R = f_phys``.

**Known accuracy of this backbone** (post-fix BING vs L23, median increment
error 550–700 nm): **+1 % / −4 %** at θ_s = 30°/60° but **−39 % at 0°**, and
+30 % at 490 nm (30–60°). That structured residual is δ_R's job (M3), not a
bug here — M2's cross-check pins this code to BING, and its characterization
test pins the *measured* error bands, not zero.

Constants are HydroLight-consistent because the L23 truth is HydroLight:
``b_R(488) = 2.6e-4 m⁻¹`` is the HydroLight value (BING's default;
Bartlett 1998 measured 2.7e-4 — the M1 task-5 correction), the excitation
wavelength dependence is ``(488/λ′)^5.5`` (energy units), and the Raman phase
function contributes ``b_b/b = 1/2`` (depolarization 0.17 gives 0.489;
BING's default rounds to the Rayleigh-like 1/2, and matching BING is the M2
contract).

Fluorescence (:func:`fluorescence_kernel`) lands at M2 task 2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from . import conventions, ed

__all__ = [  # noqa: RUF022  - grouped by role
    # Constants (HydroLight-consistent; equal to bing.rt.raman's — tested)
    "B_RAMAN_488",
    "RAMAN_EXPONENT",
    "RAMAN_BB_RATIO",
    "MU_D",
    "MU_U",
    "MU_R",
    "S_E",
    # Terms
    "raman_bb",
    "raman_factor",
]

#: Raman scattering coefficient at 488 nm excitation, m^-1, **energy units** —
#: the HydroLight value (``bing.rt.raman.B_RAMAN_488_HYDROLIGHT``), i.e. what
#: generated the L23 truth. Bartlett et al. (1998) measured 2.7e-4; Desiderio
#: 2.4e-4. Matching the truth's generator wins (design §4.3).
B_RAMAN_488 = 2.6e-4

#: Excitation-wavelength exponent, energy units: b_R ∝ (488/λ')^5.5.
RAMAN_EXPONENT = 5.5

#: Backscattering fraction of the Raman phase function. Rayleigh-like:
#: the ρ = 0.17 depolarization gives 0.489; BING's default (and HydroLight's
#: convention) is the round 1/2, and the M2 contract is bing-equality.
RAMAN_BB_RATIO = 0.5

#: S&P98 two-flow mean cosines (design §4.3): downwelling (clear sky, high
#: sun), upwelling (diffuse), and Raman-scattered (isotropic) light.
MU_D = 0.9
MU_U = 0.5
MU_R = 0.5

#: Shape factor for elastic scattering in the two-flow terms (isotropic).
S_E = 1.0


def raman_bb(
    wave_ex: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Raman backscattering coefficient ``b_bR(λ')``, m^-1 (energy units).

    ``RAMAN_BB_RATIO × B_RAMAN_488 × (488/λ')**RAMAN_EXPONENT`` — Bartlett's
    wavelength dependence anchored at the HydroLight 488 nm value, times the
    Rayleigh-like backward fraction. Equals
    ``bing.rt.raman.raman_backscattering_coeff(λ')`` (pinned by the M2
    cross-check).

    Parameters
    ----------
    wave_ex : Array
        Excitation wavelengths (nm).

    Returns
    -------
    Array
        ``b_bR``, same shape.
    """
    wave_ex = jnp.asarray(wave_ex)
    return RAMAN_BB_RATIO * B_RAMAN_488 * (488.0 / wave_ex) ** RAMAN_EXPONENT


def raman_factor(
    iops,
    geometry,
    wave: Float[Array, " wave"] | None = None,
) -> Float[Array, "*batch wave"]:
    """The multiplicative Raman correction ``f_phys(λ) ≥ 1`` (design §4.3).

    The Sathyendranath & Platt (1998) two-flow assembly, term for term the
    fixed BING's ``calc_raman_correction_factor`` (their Eqs. 5, 11, 18, 23):

    - elastic reflectance ``R_E`` (Eq. 5);
    - first-order Raman ``R_R`` (Eq. 11) — excitation absorbed at λ′,
      re-emitted at λ;
    - both second-order terms (each ~10 % of first order): Raman-then-elastic
      ``R_RE`` (Eq. 18) and elastic-then-Raman ``R_ER`` (Eq. 23); the
      Raman-Raman terms (~1 %) are neglected, as in S&P98 §4.A;

    assembled as the ratio ``(R_E + R_R + R_RE + R_ER) / R_E``, with the
    **true Ed ratio** ``Ed(λ′)/Ed(λ)`` from :mod:`robust.rt.ed` — the
    assessment's flat-Ed error (+60 %/−50 %) is exactly what that input
    removes. The excitation-grid IOPs come from
    :func:`robust.rt.conventions.interp_spectrum` on the single-shift grid
    ``λ′ = raman_excitation(λ)``; below 400 nm the excitation leaves the L23
    grid and clamps (`RAMAN_WAVE_MIN_OFFICIAL` — documented caveat, no error).

    Pure JAX: batched over leading axes, ``jit``/``vmap``-safe, and
    differentiable in every input (the IOPs enter through the interpolation,
    which is differentiable in the spectrum values by construction — M1).

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Absorption and backscattering on ``wave``. ``a_ph`` is not used by
        Raman (water scatters regardless of what absorbs).
    geometry : robust.rt.types.Geometry
        ``theta_s`` selects the packaged sky; a ``geometry.Ed`` override
        replaces it entirely (the M1 seam, exercised end to end).
    wave : Array, optional
        Emission wavelengths (nm); defaults to the canonical grid.

    Returns
    -------
    Array
        ``f_phys``, shape ``(*batch, n_wave)``; 1.0076–2.5 on L23.
    """
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)

    a_em = iops.a
    bb_em = iops.bb
    wave_ex = conventions.raman_excitation(wave)
    a_ex = conventions.interp_spectrum(wave_ex, wave, a_em)
    bb_ex = conventions.interp_spectrum(wave_ex, wave, bb_em)
    bb_r = raman_bb(wave_ex)
    ed_ratio = ed.ratio(geometry.theta_s, wave_ex, wave, override=geometry.Ed)

    # Attenuation shorthands, S&P98: K = c_down/mu_d, kappa = c_up/mu_x.
    k_ex = (a_ex + bb_ex) / MU_D
    kappa_e_em = (a_em + bb_em) / MU_U
    kappa_r_em = (a_em + bb_em) / MU_R
    kappa_e_ex = (a_ex + bb_ex) / MU_U

    # Eq. (5): elastic two-flow reflectance.
    k_em = (a_em + bb_em) / MU_D
    r_e = (S_E * bb_em) / (MU_D * (k_em + kappa_e_em))

    # Eq. (11): first-order Raman.
    r_r = ed_ratio * (bb_r / MU_D) / (k_ex + kappa_r_em)

    # Eq. (18): Raman down, elastic up (kappa_RE = K_R = (a+bb)/mu_R at λ).
    r_re = (
        ed_ratio
        * (S_E * bb_em / MU_R)
        * (bb_r / MU_D)
        / ((k_ex + kappa_r_em) * (kappa_r_em + kappa_r_em))
    )

    # Eq. (23): elastic down, Raman up.
    r_er = (
        ed_ratio
        * (S_E * bb_ex / MU_D)
        * (bb_r / MU_U)
        / ((k_ex + kappa_e_ex) * (k_ex + kappa_r_em))
    )

    return (r_e + r_r + r_re + r_er) / r_e
