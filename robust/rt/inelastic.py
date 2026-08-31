"""
Analytic inelastic terms — the physics backbone (inelastic coding plan, M2).

JAX ports of the **fixed** BING physics (branch ``inelastic-fixes``), pure and
differentiable, composed into :func:`~robust.rt.hybrid.forward` per the design's
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

**Fluorescence is additive, and φ_C-linear by construction** (design §4.4,
DQ4): :func:`fluorescence_kernel` returns ``K_fl(λ)`` with the quantum yield
factored out, so the composed term is ``Rrs_fl = φ_C · K_fl`` and
``∂Rrs/∂φ_C = K_fl`` — the physiology handle the future inversion retrieves
rather than bakes in. The kernel is the fixed BING ``calc_Rrs_fluorescence``
per unit yield: Gordon (1979) fluorescence-as-inelastic-scattering in the
same S&P98 two-flow framework as Raman, with the **L_u = E_u/π** conversion
whose absence made pre-fix BING ×3 too bright (the assessment's second
headline). Unlike Raman, nothing divides out — every normalization here is
load-bearing, which is why fluorescence carries the π lesson and Raman never
felt it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from . import conventions, ed

__all__ = [  # noqa: RUF022  - grouped by role
    # Constants (HydroLight-consistent; equal to bing's — tested)
    "B_RAMAN_488",
    "RAMAN_EXPONENT",
    "RAMAN_BB_RATIO",
    "MU_D",
    "MU_U",
    "MU_R",
    "S_E",
    "MU_F",
    "PHI_C_REF",
    "LAMBDA_FL",
    "SIGMA_FL",
    "LAMBDA_FL_SECONDARY",
    "SIGMA_FL_SECONDARY",
    "FL_WEIGHT_PRIMARY",
    "FL_EX_MIN",
    "FL_EX_MAX",
    "FL_EX_STEP",
    # Terms
    "raman_bb",
    "raman_factor",
    "fl_excitation_grid",
    "emission_line",
    "fluorescence_kernel",
]

#: Raman scattering coefficient at 488 nm excitation, m^-1, **energy units** —
#: the HydroLight value (``bing.rt.raman.B_RAMAN_488_HYDROLIGHT``), i.e. what
#: generated the L23 truth. Bartlett et al. (1998) measured 2.7e-4; Desiderio
#: 2.4e-4. Matching the truth's generator wins (design §4.3).
B_RAMAN_488 = 2.6e-4

#: float: Excitation-wavelength exponent in energy units --
#: ``b_R ∝ (488/λ')^5.5``.
RAMAN_EXPONENT = 5.5

#: Backscattering fraction of the Raman phase function. Rayleigh-like:
#: the ρ = 0.17 depolarization gives 0.489; BING's default (and HydroLight's
#: convention) is the round 1/2, and the M2 contract is bing-equality.
RAMAN_BB_RATIO = 0.5

#: float: S&P98 two-flow mean cosine of the *downwelling* stream (clear sky,
#: high sun) -- one of the three fixed cosines of design §4.3.
MU_D = 0.9
#: float: S&P98 two-flow mean cosine of the *upwelling* (diffuse) stream.
MU_U = 0.5
#: float: S&P98 two-flow mean cosine of the *Raman-scattered* (isotropic)
#: stream.
MU_R = 0.5

#: Shape factor for elastic scattering in the two-flow terms (isotropic).
S_E = 1.0

#: Mean cosine of the upwelling *fluorescence* stream — isotropic emission,
#: same value (and same physics) as the Raman ``MU_R``; its own name because
#: BING keeps them as separate defaults (``mu_f``) and the M2 contract is
#: bing-equality term for term.
MU_F = 0.5

#: The reference quantum yield the kernel is evaluated at — HydroLight's
#: default and therefore the L23 truth's (``robust.rt.data.l23.PHI_C_L23``;
#: equality is tested). ``fluorescence_kernel`` returns
#: ``Rrs_fl(PHI_C_REF) / PHI_C_REF``, so ``φ_C · K_fl`` reproduces fixed BING
#: *exactly* at φ_C = 0.02 and is the design's φ_C-linear-by-construction
#: form elsewhere (§4.4): the only neglected nonlinearity is the
#: ``(1 − B·rrs)`` surface-transfer denominator, O(10⁻³) at fluorescence
#: amplitudes.
PHI_C_REF = 0.02

#: float: Center of the chlorophyll-a emission line (nm) -- the primary
#: (PS II) Gaussian, Gordon (1979); what L23/HydroLight used, hence
#: validatable.
LAMBDA_FL = 685.0
#: float: Gaussian width of that primary line (nm), i.e. FWHM 25 nm.
SIGMA_FL = 10.6

#: Secondary (PS I) shoulder of ``emission_shape='double'`` (FWHM 50 nm,
#: 0.75/0.25 weights) — physically better, **unvalidatable against L23**
#: (design §4.4): reported, never gated, off by default and off everywhere
#: in v1 training/validation.
LAMBDA_FL_SECONDARY = 730.0
#: float: Gaussian width of that secondary shoulder (nm), i.e. FWHM 50 nm.
SIGMA_FL_SECONDARY = 21.2
#: float: Weight carried by the primary line when
#: ``emission_shape='double'``; the shoulder carries ``1 - FL_WEIGHT_PRIMARY``.
FL_WEIGHT_PRIMARY = 0.75

#: Fluorescence excitation band (nm) — light absorbed by photosynthetic
#: pigments between these bounds can be re-emitted — and the quadrature step.
#: 5 nm puts the 65 nodes of :func:`fl_excitation_grid` exactly on canonical
#: grid points (so the excitation IOPs interpolate losslessly there) and is
#: the fixed contraction size the design budgets for (§4.6: 3320 × 81 × 65).
FL_EX_MIN = 370.0
#: float: Red edge of that fluorescence excitation band (nm).
FL_EX_MAX = 690.0
#: float: Quadrature step across it (nm) -- 5 nm, the canonical grid spacing,
#: which is what puts all 65 nodes exactly on canonical grid points.
FL_EX_STEP = 5.0


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


def fl_excitation_grid(dtype=None) -> Float[Array, " ex"]:
    """The fluorescence excitation quadrature nodes: 370–690 nm, 5 nm, 65 points.

    A fixed grid rather than a subset of the caller's ``wave``: the quadrature
    is then identical whatever emission grid is requested (a satellite band
    set, a slice), shapes stay static under ``jit``, and the cross-check can
    feed BING the very same nodes. On the canonical grid the nodes coincide
    with grid points, so the excitation IOPs interpolate losslessly.

    Parameters
    ----------
    dtype : optional
        Element dtype; defaults to JAX's current default float.

    Returns
    -------
    Array
        Excitation wavelengths (nm), shape ``(65,)``.
    """
    return jnp.arange(FL_EX_MIN, FL_EX_MAX + 0.5 * FL_EX_STEP, FL_EX_STEP, dtype=dtype)


def emission_line(
    wave: Float[Array, "..."],
    shape: str = "single",
) -> Float[Array, "..."]:
    """The chlorophyll emission line ``h_C(λ)`` (nm⁻¹), unit-normalized in λ.

    ``'single'`` — one Gaussian at 685 nm (σ = 10.6 nm), the validated
    default; ``'double'`` adds the 730 nm PS I shoulder (σ = 21.2 nm) at
    0.75/0.25 weights — implemented, **unvalidatable against L23** (module
    constants). Equals BING's ``chl_fl.emission_line_{single,double}_gaussian``.

    Parameters
    ----------
    wave : Array
        Emission wavelengths (nm).
    shape : str
        One of ``robust.rt.types.EMISSION_SHAPES``.

    Returns
    -------
    Array
        ``h_C``, same shape as ``wave``.

    Raises
    ------
    ValueError
        On an unknown ``shape`` — loudly, never a silently wrong line.
    """
    wave = jnp.asarray(wave)

    def gaussian(center, sigma):
        """One unit-area Gaussian in λ, evaluated on the enclosing ``wave``.

        Unit area (nm⁻¹) is what makes the line an emission *shape*: the
        weights in the double form then sum the two without changing the
        total emitted energy.
        """
        norm = 1.0 / (sigma * jnp.sqrt(2.0 * jnp.pi))
        return norm * jnp.exp(-0.5 * ((wave - center) / sigma) ** 2)

    if shape == "single":
        return gaussian(LAMBDA_FL, SIGMA_FL)
    if shape == "double":
        return FL_WEIGHT_PRIMARY * gaussian(LAMBDA_FL, SIGMA_FL) + (
            1.0 - FL_WEIGHT_PRIMARY
        ) * gaussian(LAMBDA_FL_SECONDARY, SIGMA_FL_SECONDARY)
    raise ValueError(
        f"emission_line: shape must be 'single' or 'double'; got {shape!r}"
    )


def fluorescence_kernel(
    iops,
    geometry,
    wave: Float[Array, " wave"] | None = None,
    emission_shape: str = "single",
) -> Float[Array, "*batch wave"]:
    """The φ_C-linear fluorescence kernel ``K_fl(λ)``: ``Rrs_fl = φ_C · K_fl``.

    The fixed BING ``calc_Rrs_fluorescence`` per unit quantum yield — Gordon
    (1979) fluorescence treated as inelastic scattering in the S&P98 two-flow
    frame (design §4.4), term for term:

    - source ``b_bF(λ′) = ½ · φ_C · a_ph(λ′)`` — isotropic emission, so half
      goes backward; **this is where ``a_ph`` earns its place in ``IOPs``**;
    - excitation integral (trapezoid) over :func:`fl_excitation_grid` with the
      ``λ′/λ`` quanta→energy factor and the true ``Ed(λ′)``, normalized by
      ``Ed(λ)`` — both from one sky (:mod:`robust.rt.ed`, override honored);
    - per-λ_em upwelling attenuation ``κ_F(λ) = (a + b_b)/μ_F`` (freezing it
      at 685 nm is the ~4× 730 nm bug BING's history warns about);
    - **L_u = E_u/π**: the emission is isotropic, so the upwelling radiance is
      ``E_u/π`` — the normalization whose absence made pre-fix BING ×3 too
      bright; the sentinel in the task-3 cross-check guards it;
    - the emission line ``h_C(λ)`` and the standard surface transfer
      :func:`robust.rt.conventions.rrs_to_Rrs` (A = 0.52, B = 1.7).

    Evaluated at :data:`PHI_C_REF` and divided by it, so ``φ_C · K_fl``
    equals BING exactly at the truth's φ_C = 0.02 and is φ_C-linear by
    construction elsewhere (the O(10⁻³) surface-transfer nonlinearity is the
    documented approximation, design §4.4). Known accuracy of this backbone
    (post-fix BING vs L23, 685 nm model/truth): **1.00 / 0.95 / 0.86** at
    θ_s = 0°/30°/60° — the trophic/zenith drift δ_F must close (M3).

    Pure JAX: batched over leading axes, ``jit``/``vmap``-safe (the
    ``(..., n_em, 65)`` contraction is fixed-size), differentiable in the
    IOPs (``a_ph`` included) and ``θ_s``.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Must carry ``a_ph`` — the source term is physically the light
        phytoplankton pigments absorbed; bulk ``a`` cannot stand in for it.
    geometry : robust.rt.types.Geometry
        ``theta_s`` selects the packaged sky; ``geometry.Ed`` overrides it
        (numerator and denominator from the same sky by construction).
    wave : Array, optional
        Emission wavelengths (nm); defaults to the canonical grid.
    emission_shape : str, optional
        ``'single'`` (default, validated) or ``'double'`` (see
        :func:`emission_line`). Static — pass
        ``Inelastic.emission_shape`` through, as ``forward`` does.

    Returns
    -------
    Array
        ``K_fl``, sr⁻¹ per unit φ_C, shape ``(*batch, n_wave)``; ≥ 0, peaked
        at 685 nm.

    Raises
    ------
    ValueError
        If ``iops.a_ph`` is ``None`` — a physical requirement, not an API
        whim: without the phytoplankton split there is no fluorescence
        source. Disable the process (``Inelastic(fluorescence=False)``)
        or load IOPs with the split.
    """
    if iops.a_ph is None:
        raise ValueError(
            "fluorescence_kernel: IOPs.a_ph is None, but the fluorescence "
            "source term is b_F = phi_C * a_ph — bulk absorption cannot "
            "stand in for the phytoplankton component. Provide a_ph (e.g. "
            "IOPs.from_total_bb(..., a_ph=...)) or turn the process off "
            "with Inelastic(fluorescence=False)"
        )
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)
    wave_ex = fl_excitation_grid(dtype=jnp.result_type(wave.dtype, float))

    a_ex = conventions.interp_spectrum(wave_ex, wave, iops.a)
    bb_ex = conventions.interp_spectrum(wave_ex, wave, iops.bb)
    aph_ex = conventions.interp_spectrum(wave_ex, wave, iops.a_ph)
    ed_ex = ed.Ed(geometry.theta_s, wave_ex, override=geometry.Ed)
    ed_em = ed.Ed(geometry.theta_s, wave, override=geometry.Ed)

    # Source and attenuation, S&P98 two-flow (the Raman Eq. 11 pattern with
    # the Raman b_bR replaced by the fluorescence source).
    bb_f = 0.5 * PHI_C_REF * aph_ex
    k_ex = (a_ex + bb_ex) / MU_D
    kappa_f_em = (iops.a + iops.bb) / MU_F

    # The (..., n_em, n_ex) contraction: K(λ') + κ_F(λ_em) is a sum, so it
    # cannot factor across the axes — this broadcast is the design's budget.
    # What *can* factor is everything λ'-only: the trapezoid weights, Ed(λ'),
    # the quanta→energy numerator and the source fold into one (..., n_ex)
    # array, so the big tensor appears in a single divide-and-reduce that XLA
    # fuses without materialising the integrand (the M4 speed fallback: the
    # unfused form cost 167 ms of a 33 ms elastic call; this one costs 4).
    # Algebraically identical to trapezoid(integrand)/(wave · Ed(λ_em));
    # float32 agreement with the unfused form is ~7e-7 (reordered sums).
    dx = jnp.diff(wave_ex)
    trapezoid_w = 0.5 * jnp.concatenate([dx[:1], dx[:-1] + dx[1:], dx[-1:]])
    numerator = trapezoid_w * wave_ex * ed_ex * bb_f / MU_D
    r_f = jnp.sum(
        numerator[..., None, :] / (k_ex[..., None, :] + kappa_f_em[..., :, None]),
        axis=-1,
    )
    # The barrier pins the reduced (..., n_em) result to materialise once:
    # without it XLA's consumer fusion re-runs the whole reduction (prelude
    # gathers included) for every use of r_f downstream — measured 17 ms vs
    # 3.8 ms on the full release, bit-identical output. Differentiable
    # (identity JVP) and jit/vmap-safe; the FD gradient gates run through it.
    r_f = jax.lax.optimization_barrier(r_f) / (wave * ed_em)

    # Isotropic emission: Lu = Eu/pi (the ×3 fix), then the standard surface
    # transfer and the emission line — bing's order, kept exactly.
    Rrs_f = conventions.rrs_to_Rrs(r_f / jnp.pi)
    return emission_line(wave, emission_shape) * Rrs_f / PHI_C_REF
