"""
ZTT analytic backbone — Twardowski & Tonizzo (2018), transcribed into JAX.

The Zaneveld-Twardowski-Tonizzo model: an analytical radiative-transfer expression
for remote-sensing reflectance in which the **backward volume scattering function
enters explicitly** rather than being absorbed into fitted coefficients. That
explicitness is the whole reason it is the backbone here — it gives the hybrid an
interpretable handle on phase-function shape that Gordon-family models structurally
lack, and it carries a real BRDF (solar zenith, view zenith, relative azimuth).

Source: *Appl. Sci.* **2018**, 8, 2684, doi:10.3390/app8122684, in
``context/RT/twardowski2018.pdf``. Every function below names the equation it
transcribes. The paper's own summary of the assembly is §2.9: "Equation (18) is the
final model ... the µd term is approximated by Equation (14), the β(ψ)/bb term by
Equation (10), the Ψ_KLu term by Equations (4) and (6), the µ∞ term by Equation (8),
and b̃b by Equation (11)."

**Elastic only.** Equation (18) is Equation (12) plus an additive ``rrs,Raman``
term. This package is elastic by scope (design §1), so :func:`rrs_ZTT` implements
Equation (12) and the Raman term is simply absent, not approximated.

Conventions, and one trap
-------------------------
- **Zenith angles in the paper are in-water and measured from straight down**, so
  **nadir viewing is θv = 180°**, giving ``-cos(θv) = +1`` in the denominator
  (paper §2, after Equation (2)). :class:`robust.rt.types.Geometry` uses the
  opposite and more usual convention, ``theta_v = 0`` for nadir, and both its
  ``theta_s`` and (since M8 task 4) its ``theta_v`` are *above-water* angles, the
  sun's as L23 reports it and the sensor's as PB24 tabulates ``Rrs``. All the
  conversions (Snell for both zeniths, then the nadir flip) happen in
  :func:`geometry_to_paper_angles`, in one place, because getting any of them
  backwards produces a plausible-looking BRDF that is wrong.
- Snell: ``θs' = asin(1.34 sin θs)`` relates in-water ``θs`` to above-water ``θs'``
  (paper §2.1). The atmospheric terms want ``θs'``.
- The water/particle backscattering split is *required*, not optional: it enters as
  ``η_bb = bbw / (bbp + bbw)`` (§2.2) and again in Equation (10). This is why
  :class:`robust.rt.types.IOPs` keeps ``bb_w`` and ``bb_p`` apart.

What the model needs supplied
-----------------------------
The paper is explicit (§2.9) that four parameters "must be provided from direct
measurements or through some assumptions": ``bbp``, ``apg``, ``Pbb(ψ)``, and
``b̃bp``. Two arrive inside :class:`robust.rt.types.IOPs`, ``b̃bp`` is
:class:`robust.rt.types.PhaseParams`'s ``B_p``, and ``Pbb(ψ)`` — the *shape* of the
particulate backward VSF — is the remaining input. Promoting ``PhaseParams`` from
``B_p`` alone to ``B_p`` plus a ``Pbb`` descriptor is precisely what the design's M5
"ZTT backward-VSF parameterization" means.

**One published coefficient set is missing, and it blocks µ∞.** Equation (8) fits
``µ∞(bb/a, η_bb)`` with sixteen coefficients ``m1..m16`` and states "Coefficients m
are provided in Appendix A, Table A2". They are **not there**: Table A2 as printed
lists coefficients for Equations (3), (4), (16) and (17) and runs from ``m*_d,8``
straight into Table A3. A full-text search finds ``m1`` and ``m16`` nowhere but
inside Equation (8) itself, and the MATLAB code the paper points to (ioccg.org) is
not at that address. :func:`mu_infinity` therefore implements Equation (8)'s
*structure* and **requires the coefficients from the caller** rather than inventing
them.

**The stand-in is the authors' own antecedent.** The same pair published
``µ∞(bb/a, η_bb)`` in Twardowski & Tonizzo (2017), *Optics Express* **25** (15),
18122 — reference [40], the study the 2018 text says Equation (8) "extended ... to
include near zero bb/a and increased resolution in η_bb". Its Table 1 is
transcribed here as :func:`mu_infinity_tt2017` and is what :func:`rrs_ZTT` uses
when no Equation (8) coefficients are supplied. It is published, checkable, and by
the same authors, but it is **not Equation (8)**: report results as *ZTT with the
TT2017 µ∞*. Passing ``mu_inf_coeffs=`` restores the 2018 model the moment those
sixteen numbers arrive. See Q4 in
``claude_prompts/RT/rt_elastic_coding_prompt_3.md``.
"""

from __future__ import annotations

import functools
import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions
from . import phase as _phase

__all__ = [  # noqa: RUF022  - grouped by role, following the paper's assembly
    # Coefficient tables (Appendix A)
    "FA_COEFFS",
    "E_COEFFS",
    "P3_COEFFS",
    "MD_STAR_COEFFS",
    "FL_AVE_WAVE",
    "FL_AVE",
    "FL_SIN_COEFF",
    "FL_OFFSET",
    "WATER_DEPOLARIZATION",
    "DEFAULT_VISIBILITY_KM",
    "REFRACTIVE_INDEX",
    # Geometry
    "above_water_zenith",
    "in_water_zenith",
    "scattering_angle",
    "geometry_to_paper_angles",
    # Model terms
    "F_psi",
    "psi_KLu",
    "beta_w_over_bb_w",
    "P_bb_sullivan",
    "P_bb_from_phase",
    "P_BB_PIVOT_DEG",
    "P_BB_ST_ANGLES",
    "P_BB_ST_MEAN",
    "P_BB_ST_COEFFS",
    "backward_phase_over_bb",
    "bb_tilde",
    "diffuse_fraction",
    "P3_cos",
    "Md_plus",
    "Md_star",
    "mu_d",
    "md_plus_correction",
    "CLAMP_WIDTH",
    "OUT_CLAMP_WIDTH",
    "smooth_clip",
    "log_inputs",
    "mu_infinity",
    "mu_infinity_tt2017",
    "MU_INF_TT2017_TABLE1",
    "f_L",
    # The model
    "rrs_ZTT",
    "Rrs_ZTT",
    # The L23 refit (M7)
    "MU_INF_L23_COEFFS",
    "MD_STAR_L23_COEFFS",
    "MD_PLUS_CORR_L23",
    "ZTT_L23_BOX",
    "MU_INF_L23_RANGE",
    "MD_STAR_L23_RANGE",
    "VARIANTS",
    "DEFAULT_VARIANT",
    "MU_INF_MODELS",
    "MU_D_MODELS",
    "mu_infinity_l23",
    "mu_infinity_l23_closure",
    "MU_INF_L23_CLOSURE_COEFFS",
    "MU_INF_L23_CLOSURE_RANGE",
    "mu_d_l23",
    "mu_inf_closure",
    # The PB24 fit (M8 task 8a)
    "H_PB24_FILE",
    "diffuse_fraction_pb24",
    "MU_INF_PB24_COEFFS",
    "MD_STAR_PB24_COEFFS",
    "MD_PLUS_CORR_PB24",
    "ZTT_PB24_BOX",
    "MU_INF_PB24_RANGE",
    "MD_STAR_PB24_RANGE",
    "ZTT_PB24_N_TRAINING",
    "ZTT_PB24_TRAINING_DIGEST",
    "mu_infinity_pb24",
    "mu_d_pb24",
    # The PB24 angular terms (M8 task 8b)
    "ANGULAR_MODELS",
    "PSI_KLU_PB24_COEFFS",
    "FL_AVE_PB24_WAVE",
    "FL_AVE_PB24",
    "FL_S_PB24",
    "D_FLOOR_PB24",
    "F_psi_pb24",
    "psi_KLu_pb24",
    "f_L_pb24",
    "denominator_floor_pb24",
    # The nadir-only PB24 fit (M8 task 9)
    "NADIR_TOL_DEG",
    "NADIR_BB_OVER_A_MAX",
    "PSI_KLU_PB24_NADIR_COEFFS",
    "PSI_KLU_PB24_NADIR_PSI_MIN",
    "FL_AVE_PB24_NADIR",
    "FL_S_PB24_NADIR",
    "psi_KLu_pb24_nadir",
    "f_L_pb24_nadir",
]

#: Refractive index of seawater used for Snell refraction (paper §2.1). The
#: same constant as :data:`robust.rt.conventions.REFRACTIVE_INDEX` (1.34), which
#: the Fresnel surface transfer uses.
REFRACTIVE_INDEX = conventions.REFRACTIVE_INDEX

#: Equation (4) — ``F(ψ) = fA1 ψ⁴ + fA2 ψ³ + fA3 ψ² + fA4 ψ + fA5``, ψ in
#: **degrees** (Table A2). Highest order first, for ``jnp.polyval``.
FA_COEFFS = (
    -3.79435531537314e-07,
    2.42117623125973e-04,
    -5.76056692150838e-02,
    6.04944577004764,
    -236.166389774491,
)

#: Equation (16) — diffuse fraction ``H(θs', V)``, coefficients ``e1..e18``
#: (Table A2), in the paper's order.
E_COEFFS = (
    -3.37021020153209e-12,
    2.25040435584125e-10,
    -2.25897880448836e-09,
    4.98402568695743e-10,
    -3.67440351688922e-08,
    4.02677827509591e-07,
    -2.52448256032736e-08,
    2.09631870150827e-06,
    -2.43068373614361e-05,
    5.98295717192273e-07,
    -5.36922068813161e-05,
    6.84105803724285e-04,
    -5.34168078899319e-06,
    4.95201118318049e-04,
    -6.09578731164684e-03,
    5.32097604773773e-04,
    -2.91276619216202e-02,
    0.589340234481004,
)

#: The ``P3[cos(θs')]`` polynomial quoted after Equation (16), highest order first.
P3_COEFFS = (0.7792, -1.7366, 1.1551, 0.7842)

#: Equation (17) — ``M*_d`` coefficients ``m*_d,1 .. m*_d,8`` (Table A2).
MD_STAR_COEFFS = (
    0.00611094400155735,
    -0.00104841847722295,
    0.0498255758922950,
    -0.0117672820980625,
    0.128019358635212,
    -0.0429896134897322,
    0.103528931695373,
    0.950921179229178,
)

#: Wavelengths of :data:`FL_AVE` (nm) — Table A3 spans 350-800 nm at 5 nm.
FL_AVE_WAVE = np.arange(350.0, 800.0 + 2.5, 5.0)

#: Table A3 — ``f_L,ave(λ)``, the spectral shape of the upwelling-radiance shape
#: factor, used by Equation (31). The paper notes its resemblance to an absorption
#: spectrum, an effect of multiple scattering (§4.1).
FL_AVE = np.array(
    [
        0.990, 0.990, 0.992, 0.992, 0.992, 0.995, 0.997, 0.997,
        0.998, 1.000, 1.000, 1.000, 1.002, 1.003, 1.006, 1.008,
        1.010, 1.013, 1.016, 1.020, 1.023, 1.024, 1.025, 1.025,
        1.026, 1.026, 1.026, 1.026, 1.026, 1.024, 1.022, 1.018,
        1.013, 1.009, 1.005, 1.002, 0.999, 0.996, 0.995, 0.992,
        0.989, 0.987, 0.985, 0.982, 0.981, 0.982, 0.983, 0.984,
        0.986, 0.987, 0.988, 0.988, 0.989, 0.989, 0.989, 0.990,
        0.990, 0.990, 0.990, 0.990, 0.990, 0.992, 0.993, 0.998,
        1.000, 1.001, 1.000, 0.995, 0.994, 0.993, 0.994, 0.994,
        0.996, 0.997, 0.999, 1.000, 1.000, 1.000, 0.999, 0.999,
        0.999, 0.999, 0.999, 0.999, 0.999, 1.000, 1.000, 1.001,
        1.002, 1.002, 1.002,
    ]
)  # fmt: skip

#: Equation (31) — the ψ-dependent scaling of ``f_L``.
FL_SIN_COEFF = 0.07762
#: float: Constant term of the same Equation (31) scaling,
#: ``FL_SIN_COEFF sin(ψ) + FL_OFFSET`` (see :func:`f_L`).
FL_OFFSET = 1.0405

#: Depolarization ratio of pure seawater — the default of Zhang et al. (2009), the
#: model the paper defers ``βw(ψ)`` to (§2.3, ref [52]).
WATER_DEPOLARIZATION = 0.039

#: Horizontal visibility (km). The paper notes 15 km is the Hydrolight default
#: (§2.7, after Equation (16)).
DEFAULT_VISIBILITY_KM = 15.0


# ------------------------------------------------------------------- geometry --
def above_water_zenith(theta_s_water_deg):
    """Above-water solar zenith ``θs'`` from in-water ``θs`` (paper §2.1).

    ``θs' = asin(1.34 sin θs)``.

    Parameters
    ----------
    theta_s_water_deg : Array
        In-water solar zenith angle (degrees).

    Returns
    -------
    Array
        Above-water solar zenith angle (degrees).
    """
    sin_air = REFRACTIVE_INDEX * jnp.sin(jnp.deg2rad(jnp.asarray(theta_s_water_deg)))
    return jnp.rad2deg(jnp.arcsin(jnp.clip(sin_air, -1.0, 1.0)))


def in_water_zenith(theta_s_air_deg):
    """In-water solar zenith ``θs`` from above-water ``θs'`` — inverse Snell.

    Plain Snell refraction, so it serves any direction: since M8 task 4
    :func:`geometry_to_paper_angles` also uses it for the view zenith. Exactly
    ``0`` at ``0`` (``asin(0) = 0``), which is what keeps nadir bit-identical.

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water zenith angle (degrees), as L23 reports it.

    Returns
    -------
    Array
        In-water zenith angle (degrees).
    """
    sin_water = jnp.sin(jnp.deg2rad(jnp.asarray(theta_s_air_deg))) / REFRACTIVE_INDEX
    return jnp.rad2deg(jnp.arcsin(jnp.clip(sin_water, -1.0, 1.0)))


def scattering_angle(theta_s_water_deg, theta_v_paper_deg, dphi_deg):
    """In-water scattering angle ψ (paper §2, the line after Equation (2)).

    ``cos ψ = cos θs cos θv - sin θs sin θv cos φ``, with all angles in the paper's
    in-water convention where **nadir viewing is θv = 180°**.

    Parameters
    ----------
    theta_s_water_deg, theta_v_paper_deg, dphi_deg : Array
        In-water solar zenith, paper-convention view zenith, and relative azimuth,
        in degrees.

    Returns
    -------
    Array
        Scattering angle ψ (degrees). For nadir viewing this reduces to
        ``ψ = 180° - θs``.
    """
    ts = jnp.deg2rad(jnp.asarray(theta_s_water_deg))
    tv = jnp.deg2rad(jnp.asarray(theta_v_paper_deg))
    phi = jnp.deg2rad(jnp.asarray(dphi_deg))
    cos_psi = jnp.cos(ts) * jnp.cos(tv) - jnp.sin(ts) * jnp.sin(tv) * jnp.cos(phi)
    return jnp.rad2deg(jnp.arccos(jnp.clip(cos_psi, -1.0, 1.0)))


def geometry_to_paper_angles(geometry, *, theta_v_in_water: bool = False):
    """Translate a :class:`robust.rt.types.Geometry` into the paper's angles.

    Three conversions, kept together because any one reversed yields a
    wrong-but-plausible BRDF:

    1. ``theta_s`` is the **above-water** solar zenith (L23's convention); the
       paper's ``θs`` is **in-water**, so Snell refracts it.
    2. ``theta_v`` is the **above-water** sensor zenith (M8 task 4,
       ``rt_fit_pb24.md`` Q9a); the paper's ``θv`` is in-water, so Snell
       refracts it too, ``θv_w = asin(sin θv / 1.34)``. Before M8 task 4 it was
       used unrefracted, i.e. read as in-water.
    3. ``theta_v = 0`` is nadir here; the paper's nadir is ``θv = 180°``, so
       ``θv_paper = 180° - θv_w``.

    At ``theta_v = 0`` step 2 returns exactly ``0``, so nadir (every L23
    number) is bit-identical to the pre-task-4 result.

    Parameters
    ----------
    geometry : robust.rt.types.Geometry
        Angles in degrees, this package's convention.
    theta_v_in_water : bool, optional
        ``True`` reads ``theta_v`` as an **in-water** angle and skips its
        refraction. Only for PB24's ``angles="in_water_only"`` batches, whose
        in-water nodes 50-87.5° lie beyond the critical angle and have no
        above-water direction (pass ``batch.in_water_only``). Default ``False``.

    Returns
    -------
    tuple of Array
        ``(theta_s_water, theta_s_air, theta_v_paper, psi)``, all degrees.
    """
    theta_s_air = jnp.asarray(geometry.theta_s)
    theta_s_water = in_water_zenith(theta_s_air)
    theta_v = jnp.asarray(geometry.theta_v)
    theta_v_water = theta_v if theta_v_in_water else in_water_zenith(theta_v)
    theta_v_paper = 180.0 - theta_v_water
    psi = scattering_angle(theta_s_water, theta_v_paper, jnp.asarray(geometry.dphi))
    return theta_s_water, theta_s_air, theta_v_paper, psi


# ------------------------------------------------------------ K_Lu / Psi_KLu ---
def F_psi(psi_deg: Float[Array, "..."]) -> Float[Array, "..."]:
    """Equation (4) — ``F(ψ)``, the fractional excess of ``K_Lu`` over ``K∞``.

    ``F(ψ) = fA1 ψ⁴ + fA2 ψ³ + fA3 ψ² + fA4 ψ + fA5`` with ψ in degrees
    (:data:`FA_COEFFS`). This is Equation (3) — originally a quartic in ``θs'`` —
    restated in terms of the in-water scattering angle, which is what lets one
    nadir-derived relation serve off-nadir viewing (paper §2.1).

    Parameters
    ----------
    psi_deg : Array
        In-water scattering angle (degrees). The paper's simulations cover
        ψ ≳ 134°, which it notes is >95% of the angles a polar orbiter sees.

    Returns
    -------
    Array
        ``F(ψ)``, dimensionless.

    Notes
    -----
    **Badly conditioned by construction.** At ψ = 180° the five terms are
    -398, +1412, -1866, +1089, -236 and they cancel to **0.0239** — a factor
    ~78,000. That is normal for a polynomial fitted over a narrow range in degrees,
    but it means float32 loses about five significant figures here: ``Ψ_KLu`` differs
    between float32 and float64 by ~5.6e-5 relative, and that single term accounts
    for essentially all of the model's float32/float64 disagreement (~5e-5 on
    ``rrs``). Negligible against a 3-5% model error, and irrelevant to the gradient
    gate, which runs in float64 — but do not rewrite this evaluation without
    re-checking, and do not tighten any float32 tolerance that depends on it. A test
    pins the current agreement. The paper's other polynomials cancel by factors of
    1-116 and are unaffected.
    """
    return jnp.polyval(jnp.asarray(FA_COEFFS), jnp.asarray(psi_deg))


def psi_KLu(psi_deg: Float[Array, "..."]) -> Float[Array, "..."]:
    """Equation (6) — ``Ψ_KLu(ψ) = K_Lu / K∞ = 1 + F(ψ)``.

    Assumed spectrally independent, with errors under 2% across water types
    (paper §2.1); it is the only term in the model with no λ dependence.

    Parameters
    ----------
    psi_deg : Array
        In-water scattering angle (degrees).

    Returns
    -------
    Array
        ``Ψ_KLu``, dimensionless.
    """
    return 1.0 + F_psi(psi_deg)


# -------------------------------------------------------- backward phase fn ----
def beta_w_over_bb_w(
    psi_deg: Float[Array, "..."], depolarization: float = WATER_DEPOLARIZATION
) -> Float[Array, "..."]:
    """Pure-water VSF normalized by its own backscattering, ``βw(ψ) / bbw``.

    The paper defers ``βw(ψ)`` to Zhang et al. (2009) as directly computable
    (§2.3), but that model's only implementation to hand
    (``ocpy.water.scattering.betasw_ZHH2009``) raises "THIS IS NOT SUCCESFULLY
    CONVERTED YET". Its *shape* is not in doubt: molecular scattering goes as

    ``βw(ψ) ∝ 1 + f cos²ψ``,  ``f = (1 - δ) / (1 + δ)``

    with δ the depolarization ratio. Normalizing over the backward hemisphere,
    ``bbw = 2π ∫_{π/2}^{π} βw sin ψ dψ = 2π βw(90°) (1 + f/3)``, gives the closed
    form used here. Only the shape is needed, since Equation (10) multiplies it by
    ``bbw``, so the unknown ``βw(90°)`` cancels.

    Checkable rather than assumed: at ψ = 180° this gives **0.2342 sr⁻¹** for
    δ = 0.039, against the 0.23 sr⁻¹ quoted for pure water in
    ``context/RT/rt_elastic_model.md`` §3.5 (citing Zhang 2009 and this paper).

    Parameters
    ----------
    psi_deg : Array
        Scattering angle (degrees).
    depolarization : float, optional
        δ; defaults to Zhang et al.'s 0.039.

    Returns
    -------
    Array
        ``βw(ψ) / bbw`` in sr⁻¹.
    """
    f = (1.0 - depolarization) / (1.0 + depolarization)
    cos_psi = jnp.cos(jnp.deg2rad(jnp.asarray(psi_deg)))
    return (1.0 + f * cos_psi**2) / (2.0 * jnp.pi * (1.0 + f / 3.0))


#: Sullivan & Twardowski (2009) Table 1 — the measured average particulate backward
#: phase function ``β̃bp(ψ) = βp(ψ)/bbp`` (sr⁻¹) from several million field VSFs,
#: with its standard deviation. *Appl. Opt.* **48** (35), 6811. Angles in degrees.
P_BB_ST_ANGLES = np.array([90.0, 100, 110, 120, 130, 140, 150, 160, 170])
#: numpy.ndarray: The nine tabulated ``β̃bp`` means (sr⁻¹) at
#: :data:`P_BB_ST_ANGLES`, from the same table.
P_BB_ST_MEAN = np.array([0.233, 0.186, 0.159, 0.145, 0.138, 0.137, 0.138, 0.141, 0.146])
P_BB_ST_STD = np.array([0.012, 0.007, 0.004, 0.004, 0.005, 0.006, 0.007, 0.007, 0.008])

#: Sullivan & Twardowski (2009) Table 2 — their fourth-order polynomial fit to the
#: values above, highest order first.
#:
#: **The published ``a3`` carries a typographic error.** Table 2 prints
#: ``8.007E−02``; that is impossible, since at ψ = 140° it contributes
#: ``19600 × 0.08 ≈ 1570`` against a tabulated value of 0.137. Refitting Table 1
#: here independently gives ``a3 = 7.79e-04`` while reproducing the other four
#: published coefficients closely (5.65e-9 vs 5.885e-9; −3.41e-6 vs −3.526e-6;
#: −7.98e-2 vs −8.150e-2; 3.215 vs 3.266), so the intended value is
#: **8.007E−04**. With that correction the published fit matches Table 1 to
#: 0.003 absolute, consistent with the paper's claim of "<0.5%".
P_BB_ST_COEFFS = (5.885e-09, -3.526e-06, 8.007e-04, -8.150e-02, 3.266)


def P_bb_sullivan(psi_deg: Float[Array, "..."]) -> Float[Array, "..."]:
    """``Pbb,ST(ψ)`` — the constant backward phase function shape of Sullivan &
    Twardowski (2009).

    This is the ``Pbb(ψ)`` input the 2018 ZTT paper calls for and reports its best
    performance with (§4.2): "If we assume βp(ψ)/bbp is a constant shape Pbb,ST(ψ)
    ... errors increase by only ~0.3%". "Constant" means constant **across water
    types**, not across angle — the shape still varies with ψ, and forgetting that
    inverts the modelled zenith trend (see ``robust/tests/test_ztt.py``).

    Evaluated from the paper's fourth-order polynomial (:data:`P_BB_ST_COEFFS`,
    with the ``a3`` typo corrected). Table 1 stops at 170°, so nadir viewing
    (ψ = 180°) is a short extrapolation, giving **0.153 sr⁻¹**; an independent
    refit of Table 1 gives 0.156, and fitting a constant against L23 at that
    geometry gives 0.148, so the extrapolation is well constrained.

    Parameters
    ----------
    psi_deg : Array
        Scattering angle (degrees). Meaningful over 90-180°; the measurements
        span 90-170°.

    Returns
    -------
    Array
        ``Pbb(ψ)`` in sr⁻¹, ~0.137 at its minimum near 140° rising to ~0.23 at 90°.
    """
    return jnp.polyval(jnp.asarray(P_BB_ST_COEFFS), jnp.asarray(psi_deg))


#: Reference angle the backward-shape tilt pivots about (degrees). 180° is exact
#: backscatter, where :attr:`~robust.rt.types.PhaseParams.beta_tilde_pi` is
#: defined, so the tilt leaves that value alone by construction.
P_BB_PIVOT_DEG = 180.0


def P_bb_from_phase(
    phase_params, psi_deg, *, use_ff: bool = False
) -> Float[Array, "..."]:
    """``Pbb(ψ) = βp(ψ)/bb_p`` from the phase-function container (M5 task 14).

    ZTT takes the particulate backward phase function as a **required model
    input** (paper §2.9), and M0-M4 always supplied the one fixed shape the paper
    reports its best results with, :func:`P_bb_sullivan`. Design §4.2 planned a
    2-3 parameter backward-VSF axis for when data sampled it; this is that axis,
    built on the two fields :class:`~robust.rt.types.PhaseParams` gained:

    ``Pbb(ψ) = beta_tilde_pi * S_ST(ψ)/S_ST(180°) * (ψ/180)**(-backward_slope)``

    - ``beta_tilde_pi`` **rescales** the shape so that ``Pbb(180°)`` equals it.
      Sullivan's own value is 0.153 sr^-1, so passing that reproduces the fixed
      shape exactly.
    - ``backward_slope`` **tilts** it across the backward hemisphere, pivoting at
      180° so the two parameters are independent: 0 leaves Sullivan's angular
      dependence untouched, positive values raise the shape away from
      backscatter.

    Either field left ``None`` takes its neutral value, so
    ``P_bb_from_phase(PhaseParams(B_p=...), psi)`` **is** ``P_bb_sullivan(psi)``,
    bit for bit. That is what keeps every M0-M4 number reproducible.

    **M7 task 2: a Fournier–Forand mixture instead of Sullivan.** With
    ``use_ff=True`` *and* ``phase_params.ff_B`` set, the shape is
    :func:`robust.rt.phase.P_bb_ff` -- for L23, its own two-component FF phase
    function -- in place of Sullivan's. Without ``ff_B`` it falls back to
    Sullivan **silently** (Q9): most callers have no component information. The
    switch is an explicit keyword, off by default, because the L23 loader now
    fills ``ff_B``. If ``rrs_ZTT`` honoured it unasked, every pinned hybrid
    number would move. M7 task 4's ``variant="l23"`` turns it on. The FF shape
    is not combined with the M5 tilt/rescale fields: asking for both raises.

    Parameters
    ----------
    phase_params : robust.rt.types.PhaseParams
    psi_deg : Array
        Scattering angle (degrees).
    use_ff : bool, optional
        Use the FF mixture when ``phase_params`` carries one (default False).

    Returns
    -------
    Array
        ``Pbb(ψ)`` in sr^-1.

    Raises
    ------
    ValueError
        If ``use_ff`` selects the FF mixture while ``beta_tilde_pi`` or
        ``backward_slope`` is also set.

    Notes
    -----
    **This parameterization is uncalibrated, and the form is a choice.** A power
    law in ``ψ/180`` is the simplest one-parameter tilt that is smooth,
    differentiable, and neutral at the pivot; it is not derived from measurements,
    and no dataset in this repository constrains it -- PB24 prescribes its phase
    functions and does not tabulate ``βp(ψ)`` (record §7.2). So this is an axis to
    sweep and an interface to build against, not a fitted model. Any result that
    varies these fields must say which values it used and that they are inputs.
    """
    if (
        use_ff
        and phase_params is not None
        and getattr(phase_params, "ff_B", None) is not None
    ):
        if (
            getattr(phase_params, "beta_tilde_pi", None) is not None
            or getattr(phase_params, "backward_slope", None) is not None
        ):
            raise ValueError(
                "P_bb_from_phase: the FF mixture (ff_B) and the M5 Sullivan "
                "tilt/rescale (beta_tilde_pi, backward_slope) are alternatives; "
                "set one or the other"
            )

        def align_components(value):
            """Give per-sample component arrays the trailing axes ``psi`` has.

            ``psi`` arrives as ``(n_sample, 1)``. A spectral ``(n, W, K)`` array
            already lines up, but a per-sample ``(n, K)`` would broadcast against
            ψ's ``(n, 1, 1)`` to ``(n, n, K)``: a silent cross-product of samples
            (M7 review, finding 1a). The same trap ``align`` closes below for
            the M5 fields.
            """
            value = jnp.asarray(value)
            psi_ndim = jnp.ndim(jnp.asarray(psi_deg))
            batch_ndim = value.ndim - 1
            if 0 < batch_ndim < psi_ndim:
                value = value.reshape(
                    value.shape[:-1] + (1,) * (psi_ndim - batch_ndim) + value.shape[-1:]
                )
            return value

        return _phase.P_bb_ff(
            psi_deg,
            align_components(phase_params.ff_B),
            align_components(phase_params.ff_weight),
        )

    shape = P_bb_sullivan(psi_deg)
    if phase_params is None:
        return shape

    def align(value):
        """Give a per-sample parameter the trailing axes ``psi`` already has.

        ``rrs_ZTT`` hands this function ``psi`` shaped ``(n_sample, 1)`` so it
        broadcasts against wavelength. A per-sample parameter arrives as
        ``(n_sample,)``, and ``(n, 1) * (n,)`` broadcasts to ``(n, n)`` -- not an
        error, just silently the wrong thing, which is the failure mode worth
        spending four lines on.
        """
        value = jnp.asarray(value)
        psi = jnp.asarray(psi_deg)
        if value.ndim and value.ndim < psi.ndim:
            return value.reshape(value.shape + (1,) * (psi.ndim - value.ndim))
        return value

    slope = getattr(phase_params, "backward_slope", None)
    if slope is not None:
        ratio = jnp.asarray(psi_deg) / P_BB_PIVOT_DEG
        shape = shape * ratio ** (-align(slope))

    pi_value = getattr(phase_params, "beta_tilde_pi", None)
    if pi_value is not None:
        pi_value = align(pi_value)
        # Renormalise so the caller's beta_tilde_pi is exactly Pbb(180 deg),
        # whatever the tilt did -- which is what makes the two independent.
        at_pivot = P_bb_sullivan(jnp.asarray(P_BB_PIVOT_DEG))
        shape = shape * (jnp.asarray(pi_value) / at_pivot)
    return shape


def backward_phase_over_bb(iops, P_bb, psi_deg) -> Float[Array, "..."]:
    """Equation (10) — ``β(ψ)/bb = [Pbb(ψ) bbp + βw(ψ)] / (bbp + bbw)``.

    The numerator of the model, and the reason ZTT is "explicitly dependent on the
    VSF": the particulate backward phase function appears as itself rather than
    inside a fitted constant.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Supplies ``bb_p`` and ``bb_w`` separately — Equation (10) cannot be
        written in terms of total ``bb`` alone.
    P_bb : Array
        ``Pbb(ψ) = βp(ψ)/bbp`` (sr⁻¹) at this geometry. A **required model input**
        (paper §2.9). The paper's best-performing choice is the constant shape
        ``Pbb,ST(ψ)`` of Sullivan & Twardowski (2009); that table is in *their*
        paper, not this one, so it is supplied by the caller.
    psi_deg : Array
        Scattering angle (degrees), for the water term.

    Returns
    -------
    Array
        ``β(ψ)/bb`` in sr⁻¹.
    """
    beta_w = beta_w_over_bb_w(psi_deg) * iops.bb_w
    return (jnp.asarray(P_bb) * iops.bb_p + beta_w) / (iops.bb_p + iops.bb_w)


def bb_tilde(iops, B_p: Float[Array, "..."]) -> Float[Array, "..."]:
    """Equation (11) — the total backscattering ratio ``b̃b = bb / b``.

    ``b̃b = (bbp + bbw) / (bbp / b̃bp + bw)``, where ``b̃bp`` is the *particulate*
    backscattering ratio. Pure water scatters symmetrically about 90°, so exactly
    half of its scattering is backward: ``bw = 2 bbw``. That identity is used here
    rather than carrying ``bw`` as a separate input.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Supplies ``bb_p`` and ``bb_w``.
    B_p : Array
        ``b̃bp`` — ``PhaseParams.B_p``. The paper notes reflectance moves by
        several percent across its natural ~0.003-0.03 range (§2.4), the same
        range M1 measured in L23.

    Returns
    -------
    Array
        ``b̃b``, dimensionless.
    """
    b_w = 2.0 * iops.bb_w
    return (iops.bb_p + iops.bb_w) / (iops.bb_p / jnp.asarray(B_p) + b_w)


# ----------------------------------------------------------------------- mu_d --
def diffuse_fraction(theta_s_air_deg, visibility_km: float = DEFAULT_VISIBILITY_KM):
    """Equation (16) — diffuse fraction ``H(θs', V) = Eds/Ed``.

    A fit to Gregg & Carder (1990): quintic in the above-water solar zenith with
    quadratic visibility dependence, coefficients ``e1..e18`` (:data:`E_COEFFS`).

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees).
    visibility_km : float, optional
        Horizontal visibility V (km); Hydrolight's default is 15 km.

    Returns
    -------
    Array
        ``H``, the diffuse fraction of downwelling irradiance.
    """
    e = E_COEFFS
    v = visibility_km
    quad = [
        e[0] * v**2 + e[1] * v + e[2],
        e[3] * v**2 + e[4] * v + e[5],
        e[6] * v**2 + e[7] * v + e[8],
        e[9] * v**2 + e[10] * v + e[11],
        e[12] * v**2 + e[13] * v + e[14],
        e[15] * v**2 + e[16] * v + e[17],
    ]
    return jnp.polyval(jnp.asarray(quad), jnp.asarray(theta_s_air_deg))


def P3_cos(theta_s_air_deg):
    """The ``P3[cos(θs')]`` correction quoted after Equation (16).

    ``0.7792 c³ - 1.7366 c² + 1.1551 c + 0.7842`` with ``c = cos θs'``. Corrects
    where Morel & Prieur's approximation departs from Gregg & Carder at large
    ``θs'``.

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees).

    Returns
    -------
    Array
        ``P3``, dimensionless.
    """
    return jnp.polyval(
        jnp.asarray(P3_COEFFS), jnp.cos(jnp.deg2rad(jnp.asarray(theta_s_air_deg)))
    )


def Md_plus(theta_s_air_deg, visibility_km: float = DEFAULT_VISIBILITY_KM, *, H=None):
    """Equation (15) — the atmospheric/geometric part of ``µd``.

    ``M+_d = [ (1-H)/µw + H/0.859 ]⁻¹ P3[cos θs']`` with ``µw = cos θs'``: the
    Morel & Prieur (1977) cardioidal-skylight form of Equation (13), generalized
    to a variable diffuse fraction.

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees).
    visibility_km : float, optional
        Horizontal visibility V (km). Unused when ``H`` is given.
    H : Array, optional
        **M8 task 4** (``rt_fit_pb24.md`` Q10a). The diffuse fraction
        ``1 - Edir/Ed`` supplied directly -- e.g. PB24's
        ``1 - Esdir_Es_ratio``, which the loader puts in
        ``Geometry.diffuse_fraction`` -- in place of TT's
        :func:`diffuse_fraction` ``(θs', V)``. Any shape broadcastable with
        ``theta_s_air_deg`` (a spectrum makes ``M+_d`` spectral). ``None``
        (default) is TT's Equation (16), bit for bit.

    Returns
    -------
    Array
        ``M+_d``, dimensionless.
    """
    theta = jnp.asarray(theta_s_air_deg)
    # mu_w is cos of the IN-WATER solar zenith. Equation (13) defines
    # "mu_w = cos(theta_s)" with theta_s unprimed, which section 2 fixes as
    # in-water, while H and P3 take the primed above-water angle. Using the
    # above-water cosine here gives mu_d = 0.573 at theta_s' = 62 deg against the
    # 0.79 the paper quotes; the in-water cosine reproduces 0.792 (and 0.936 at
    # 8 deg against their 0.94). A test pins both endpoints.
    mu_w = jnp.cos(jnp.deg2rad(in_water_zenith(theta)))
    H = diffuse_fraction(theta, visibility_km) if H is None else jnp.asarray(H)
    return P3_cos(theta) / ((1.0 - H) / mu_w + H / 0.859)


def Md_star(bb_over_a, eta_bb, coeffs=MD_STAR_COEFFS, *, box=None, out_range=None):
    """Equation (17) — the IOP-dependent part of ``µd``.

    A cubic in ``log10(bb/a)`` whose coefficients are linear in ``log10 η_bb``::

        M*_d = [m1 L + m2] X³ + [m3 L + m4] X² + [m5 L + m6] X + m7 L + m8

    with ``X = log10(bb/a)``, ``L = log10 η_bb``, ``m`` = :data:`MD_STAR_COEFFS`.

    Parameters
    ----------
    bb_over_a : Array
        ``bb / a``; the paper fitted over 1e-4 to 1e-1.
    eta_bb : Array
        ``η_bb = bbw / (bbp + bbw)``, the molecular fraction of backscattering;
        fitted over ~0.0098 to 0.98.
    coeffs : sequence of float, optional
        The eight ``m*_d`` coefficients.
    box : sequence of float, optional
        ``(X_lo, X_hi, L_lo, L_hi)``: clamp the log inputs into the fitted range
        (:func:`log_inputs`). ``None`` (the Table A2 default) extrapolates.
    out_range : sequence of float, optional
        ``(lo, hi)``: smoothly bound the output (:data:`OUT_CLAMP_WIDTH`). The
        input box is a rectangle but data fill only part of it, so a fitted
        polynomial can still misbehave in its empty corners.

    Returns
    -------
    Array
        ``M*_d``, dimensionless.
    """
    m = coeffs
    X, L = log_inputs(bb_over_a, eta_bb, box)
    value = (
        (m[0] * L + m[1]) * X**3
        + (m[2] * L + m[3]) * X**2
        + (m[4] * L + m[5]) * X
        + m[6] * L
        + m[7]
    )
    if out_range is not None:
        value = smooth_clip(value, out_range[0], out_range[1], OUT_CLAMP_WIDTH)
    return value


def mu_d(
    theta_s_air_deg,
    bb_over_a,
    eta_bb,
    visibility_km: float = DEFAULT_VISIBILITY_KM,
    *,
    md_star_coeffs=MD_STAR_COEFFS,
    md_plus_corr=None,
    box=None,
    md_star_range=None,
    H=None,
):
    """Equation (14) — ``µd ≈ M+_d(θs', V) × M*_d(bb/a, η_bb)``.

    The average cosine of the downwelling field just below the surface,
    factorized into an atmosphere/geometry part and an IOP part. The paper puts
    the error of the full expression below 1% (§2.7).

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees).
    bb_over_a, eta_bb : Array
        As :func:`Md_star`.
    visibility_km : float, optional
        Horizontal visibility V (km).
    md_star_coeffs : sequence of float, optional
        Equation (17)'s eight coefficients; Table A2's by default.
    md_plus_corr : sequence of float, optional
        ``(c1, c2)`` of :func:`md_plus_correction`, multiplying ``M+_d``; the
        M7 refit to L23 (Q10b). ``None`` leaves TT's ``M+_d`` as published.
    box : sequence of float, optional
        Input clamp for ``M*_d`` (:func:`Md_star`).
    md_star_range : sequence of float, optional
        Output bound for ``M*_d`` (:func:`Md_star`'s ``out_range``).
    H : Array, optional
        The diffuse fraction, in place of TT's ``H(θs', V)``; see
        :func:`Md_plus`. ``None`` (default) is TT's, bit for bit.

    Returns
    -------
    Array
        ``µd``, dimensionless.

    Notes
    -----
    With every keyword at its default this is exactly the published Equation
    (14), bit for bit; the keywords exist for the refit variant (M7 task 3)
    and the supplied diffuse fraction (M8 task 4).
    """
    plus = Md_plus(theta_s_air_deg, visibility_km, H=H)
    if md_plus_corr is not None:
        plus = plus * md_plus_correction(theta_s_air_deg, md_plus_corr)
    return plus * Md_star(
        bb_over_a, eta_bb, md_star_coeffs, box=box, out_range=md_star_range
    )


def md_plus_correction(theta_s_air_deg, corr):
    """The atmospheric correction ``c(θs') = 1 + c1 u + c2 u²``, ``u = 1 − cos θs'``.

    M7 (``rt_fit_ztt.md``, Q10b). Measured against L23, TT's ``M+_d`` puts ``µd``
    8.7% high at θs' = 60° and 2% low at 0°: TT's diffuse-fraction fit and
    ``P3`` term do not describe HydroLight's default clear sky. The correction
    is fitted *jointly* with ``M*_d``. ``c(0) = 1`` by construction, so the two
    factors stay separately identified. With L23's three sun angles it is
    exactly determined, and it is tied to L23's single atmosphere.

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees).
    corr : sequence of float
        ``(c1, c2)``.

    Returns
    -------
    Array
        ``c(θs')``, dimensionless.
    """
    u = 1.0 - jnp.cos(jnp.deg2rad(jnp.asarray(theta_s_air_deg)))
    return 1.0 + corr[0] * u + corr[1] * u**2


#: Width (in log10 units) of the soft corners of :func:`smooth_clip`. 0.01 decades
#: is 2.3% in the ratio itself, small against the fitted boxes' 1-4 decade sides.
CLAMP_WIDTH = 0.01


#: Width of the soft corners when a fitted polynomial's *output* is bounded
#: (``out_range``): 0.001 in µ. The guard moves a value by
#: ``width · ln(1 + e^{-d/width})`` at distance ``d`` inside its range: ~7e-6 at
#: five widths, but up to ~3e-4 within a width of an edge (review pass).
OUT_CLAMP_WIDTH = 0.001


def smooth_clip(x, lo, hi, width=CLAMP_WIDTH):
    """Clamp ``x`` into ``[lo, hi]`` with softplus corners: smooth, monotone, and
    the identity to within ``width · e^{-d/width}`` at distance ``d`` inside.

    ``jnp.clip`` would do the clamping, but its gradient is exactly zero
    outside and undefined at the edges. Here the gradient decays smoothly
    instead, which keeps an optimizer that strays out of the box informed.

    Parameters
    ----------
    x : Array
    lo, hi : float
    width : float, optional

    Returns
    -------
    Array
    """
    x = jnp.asarray(x)
    return (
        x
        - width * jnp.logaddexp(0.0, (x - hi) / width)
        + width * jnp.logaddexp(0.0, (lo - x) / width)
    )


def log_inputs(bb_over_a, eta_bb, box=None):
    """``X = log10(bb/a)`` and ``L = log10 η_bb``, optionally clamped to ``box``.

    Parameters
    ----------
    bb_over_a, eta_bb : Array
    box : sequence of float, optional
        ``(X_lo, X_hi, L_lo, L_hi)`` -- the range a fit was made over (Q3).

    Returns
    -------
    X, L : Array
    """
    X = jnp.log10(jnp.asarray(bb_over_a))
    L = jnp.log10(jnp.asarray(eta_bb))
    if box is not None:
        X = smooth_clip(X, box[0], box[1])
        L = smooth_clip(L, box[2], box[3])
    return X, L


# --------------------------------------------------------------------- mu_inf --
#: Twardowski & Tonizzo (2017) Table 1 — second-order fits of ``µ∞`` against
#: ``log10(bb/a)``, one per ``η_bb``: ``µ∞ = p0 + p1 L + p2 L²``.
#: *Optics Express* **25** (15), 18122, the antecedent (reference [40]) that the
#: 2018 paper's Equation (8) says it "extended". Columns: ``η_bb: (p0, p1, p2)``,
#: with the paper's quoted %δabs of 0.27, 1.1, 1.8, 2.4, 2.9, 3.6 respectively.
MU_INF_TT2017_TABLE1 = {
    0.9800: (0.714, -0.200, -0.0355),
    0.7800: (0.512, -0.325, -0.0554),
    0.4900: (0.352, -0.426, -0.0721),
    0.2500: (0.240, -0.501, -0.0855),
    0.0980: (0.201, -0.523, -0.0890),
    0.0098: (0.246, -0.458, -0.0711),
}

#: The fitted ranges the 2017 paper covers: ``bb/a`` in 1e-4..1e-1 and ``η_bb`` in
#: 0.0098..0.98. L23 reaches ``bb/a ≈ 0.31``, so the brightest scenes extrapolate.
MU_INF_TT2017_BB_OVER_A_RANGE = (1e-4, 1e-1)
MU_INF_TT2017_ETA_RANGE = (0.0098, 0.98)


#: Interpolation nodes derived once from :data:`MU_INF_TT2017_TABLE1`: the six
#: ``log10(η_bb)`` knots and the matching ``(p0, p1, p2)`` rows. Built at import
#: rather than per call.
_MU_INF_TT2017_LOG_ETA = np.log10(np.array(sorted(MU_INF_TT2017_TABLE1)))
_MU_INF_TT2017_P = np.array(
    [MU_INF_TT2017_TABLE1[e] for e in sorted(MU_INF_TT2017_TABLE1)]
)


def mu_infinity_tt2017(bb_over_a, eta_bb):
    """``µ∞(bb/a, η_bb)`` from Twardowski & Tonizzo (2017), Table 1.

    **A documented substitute for Equation (8), not Equation (8).** The 2018
    paper's own µ∞ coefficients are missing from its Table A2 (see
    :func:`mu_infinity`). This is the same authors' antecedent parameterization —
    the one the 2018 text says Equation (8) "extended ... to include near zero
    bb/a and increased resolution in η_bb" — so it is the closest published,
    checkable stand-in available, and results computed with it should be reported
    as *ZTT with the TT2017 µ∞*, never as the 2018 model.

    Table 1 gives ``µ∞ = p0 + p1 L + p2 L²``, ``L = log10(bb/a)``, at six discrete
    ``η_bb``. The three coefficients are interpolated linearly in ``log10 η_bb``
    to recover the two-dimensional surface Equation (8) would have provided;
    interpolation keeps it differentiable.

    Table 2's alternative quartics are *not* used: they reach ``µ∞ = 1.35`` at
    ``bb/a = 1e-4``, which is unphysical (µ∞ ≤ 1), and they carry no ``η_bb``
    dependence.

    Parameters
    ----------
    bb_over_a : Array
        ``bb / a``. Fitted over ``MU_INF_TT2017_BB_OVER_A_RANGE``; larger
        values extrapolate. Measured on the full 9960-sample L23 batch: median
        0.03, 99th percentile 0.31, **maximum 0.59**, with 23.6% of values above
        the fitted 0.1. (An earlier version of this line quoted the 99th
        percentile as what L23 "reaches", which understated the extrapolation
        by a factor of two.)
    eta_bb : Array
        ``η_bb = bbw / (bbp + bbw)``. Fitted over
        ``MU_INF_TT2017_ETA_RANGE``; outside it the coefficients clamp.

    Returns
    -------
    Array
        ``µ∞``, dimensionless.
    """
    log_eta_nodes = jnp.asarray(_MU_INF_TT2017_LOG_ETA)
    log_eta = jnp.log10(jnp.asarray(eta_bb))
    p0, p1, p2 = (
        jnp.interp(log_eta, log_eta_nodes, jnp.asarray(_MU_INF_TT2017_P[:, k]))
        for k in range(3)
    )

    L = jnp.log10(jnp.asarray(bb_over_a))
    return p0 + p1 * L + p2 * L**2


def mu_infinity(bb_over_a, eta_bb, coeffs=None, *, box=None, out_range=None):
    """Equation (8) — ``µ∞(bb/a, η_bb)``, the asymptotic average cosine.

    A cubic in ``log10(bb/a)`` whose four coefficients are each a cubic in
    ``log10 η_bb`` — sixteen fitted numbers ``m1..m16``::

        µ∞ = [m1 L³ + m2 L² + m3 L + m4] X³
           + [m5 L³ + m6 L² + m7 L + m8] X²
           + [m9 L³ + m10 L² + m11 L + m12] X
           + m13 L³ + m14 L² + m15 L + m16

    with ``X = log10(bb/a)`` and ``L = log10 η_bb``.

    **The coefficients are not available.** Equation (8) says "Coefficients m are
    provided in Appendix A, Table A2", but Table A2 as printed covers Equations
    (3), (4), (16) and (17) only, running from ``m*_d,8`` straight into Table A3.
    A full-text search finds ``m1`` and ``m16`` nowhere but inside Equation (8),
    and the MATLAB implementation the paper cites (ioccg.org) is not at that
    address. Rather than invent sixteen numbers — which would silently become
    "ZTT" in every downstream comparison — this function requires them.

    Parameters
    ----------
    bb_over_a, eta_bb : Array
        As :func:`Md_star`.
    coeffs : sequence of float, optional
        The sixteen coefficients ``m1..m16``. Supply them if you obtain them;
        there is no default.
    box : sequence of float, optional
        ``(X_lo, X_hi, L_lo, L_hi)`` input clamp (:func:`log_inputs`), for a
        fit made over a known range (M7, Q3).
    out_range : sequence of float, optional
        ``(lo, hi)`` smooth bound on the output, as :func:`Md_star`'s.

    Returns
    -------
    Array
        ``µ∞``, dimensionless, physically in (0, 1].

    Raises
    ------
    NotImplementedError
        If ``coeffs`` is None, with a pointer to the gap.
    ValueError
        If the wrong number of coefficients is supplied.
    """
    if coeffs is None:
        raise NotImplementedError(
            "ZTT Equation (8) coefficients m1..m16 for mu_infinity are missing "
            "from the published Table A2 (Twardowski & Tonizzo 2018), and the "
            "MATLAB code cited at ioccg.org is not there. Pass coeffs=(m1..m16) "
            "if you obtain them, or pass mu_inf=<value> to rrs_ZTT to supply "
            "mu_infinity directly -- but note a supplied constant is NOT the "
            "paper's parameterization and must not be reported as ZTT accuracy."
        )
    if len(coeffs) != 16:
        raise ValueError(f"mu_infinity: expected 16 coefficients, got {len(coeffs)}")

    m = coeffs
    X, L = log_inputs(bb_over_a, eta_bb, box)

    def cubic_in_L(c0, c1, c2, c3):
        """One cubic in ``L = log10(η_bb)`` -- the inner polynomial of
        Equation (8), evaluated four times to form the coefficients of the
        outer cubic in ``X = log10(bb/a)``."""
        return c0 * L**3 + c1 * L**2 + c2 * L + c3

    value = (
        cubic_in_L(m[0], m[1], m[2], m[3]) * X**3
        + cubic_in_L(m[4], m[5], m[6], m[7]) * X**2
        + cubic_in_L(m[8], m[9], m[10], m[11]) * X
        + cubic_in_L(m[12], m[13], m[14], m[15])
    )
    if out_range is not None:
        value = smooth_clip(value, out_range[0], out_range[1], OUT_CLAMP_WIDTH)
    return value


# ---------------------------------------------------- the L23 refit (M7) --
#: **"ZTT (L23 µ∞)"**: Equation (8)'s sixteen coefficients ``m1..m16`` refit to
#: L23 (M7 task 3, ``rt_fit_ztt.md``; record §7b.7). The target was µ∞ from the
#: asymptotic RTE eigenproblem (:mod:`robust.rt.asymptotic`) on every train-split
#: (scene, λ). The fit is linear least squares on relative residuals. Held out:
#: median |error| 0.21%, mean 0.31%. The eigen µ∞ itself reads ~1.3% high
#: against L23's profiles where those are asymptotic (Q13), a stated systematic.
#: These are **not** the authors' unpublished 2018 coefficients. Regenerate with
#: ``design/py/fit_ztt_l23.py``; ``design/ztt_l23/coefficients.json`` is the source.
MU_INF_L23_COEFFS = (
    0.07521986047292743,
    0.10972298647458721,
    0.0466331524399045,
    -0.01310755744139963,
    0.5271551241452678,
    0.7718313442009022,
    0.3325466606641836,
    -0.15564270735585706,
    1.2846484410844567,
    1.984711822640948,
    0.9657426806870779,
    -0.5574670296121988,
    1.1404716719460561,
    1.9912005403862445,
    1.200812992723473,
    0.3686930900105536,
)

#: Equation (17)'s eight ``M*_d`` coefficients, refit to L23's µd(0⁻) *jointly*
#: with :data:`MD_PLUS_CORR_L23` (M7 task 3, Q10b). Held out: 0.34 / 0.22 /
#: 0.46% at θs' = 0 / 30 / 60°, where Table A2's reach 2.05 / 0.50 / 8.75%.
MD_STAR_L23_COEFFS = (
    -0.003873912293755453,
    -0.010108051497584048,
    -0.02673631641171537,
    -0.08079295562105875,
    -0.052522890637756475,
    -0.21002435890920176,
    -0.026556917867871153,
    0.8419000299235042,
)

#: ``(c1, c2)`` of :func:`md_plus_correction` for L23's sky (HydroLight default
#: clear sky, 5 m/s wind): c(30°) = 0.983 and c(60°) = 0.907. Exactly determined
#: by L23's three sun angles, so its interpolation between them is untested
#: until PB24.
MD_PLUS_CORR_L23 = (-0.10656634715990398, -0.15864703687789908)

#: ``(X_lo, X_hi, L_lo, L_hi)``: the (log10 bb/a, log10 η_bb) range of the
#: training points. Both L23 refits clamp their inputs to it (Q3). The box is the
#: exact training range, unpadded, so points *on* its edges are shifted by up to
#: ``CLAMP_WIDTH · ln 2`` ≈ 0.007 decades (review pass). Measured net effect of
#: both guards: 1.5% of held-out µ∞ change by > 1e-4, median error unchanged.
ZTT_L23_BOX = (
    -3.9656776708214276,
    -0.22577186428964108,
    -1.7551743928189434,
    -0.015010866431198294,
)

#: Output guards (Q17). L23 fills only a triangle of :data:`ZTT_L23_BOX`, and in
#: the empty corners the bare Eq. (8) polynomial reaches −1.1 and 1.6. These are
#: the training ranges plus 5 × :data:`OUT_CLAMP_WIDTH`, with µ∞ capped below 1.
MU_INF_L23_RANGE = (0.4267627739922664, 0.999)
#: tuple: The same guard for ``M*_d`` (not a cosine, so not capped at 1).
MD_STAR_L23_RANGE = (0.881526035775638, 1.039261098123049)

#: **"ZTT (L23 closure µ∞)" -- a ceiling, not physics** (M7 task 5, Q1). Equation
#: (8)'s form fitted to the *closure* µ∞: for each L23 sample, the µ∞ that makes
#: the L23 variant reproduce the reference rrs exactly (:func:`mu_inf_closure`),
#: on train-split rows of all three zeniths. It absorbs the errors of every other
#: term (Ψ_KLu, f_L, Pbb, the residual µd), so it is not an average cosine of
#: anything. It is the most rrs that µ∞ alone could buy. Held-out rRMS: 1.20 /
#: 1.43 / 1.47% at 0 / 30 / 60° against 2.91 / 2.56 / 3.18% for "ZTT (L23 µ∞)".
#: Selected with ``mu_inf_model="l23_closure"``; never a default.
MU_INF_L23_CLOSURE_COEFFS = (
    -0.29563733771899775,
    -0.2984025525634349,
    -0.008195819323475718,
    -0.021091981813251153,
    -1.8403173054772548,
    -1.609279630269282,
    0.18259986740444667,
    -0.17366791155185418,
    -3.591180539083864,
    -2.479918768680893,
    0.9361370735221088,
    -0.5057020088394896,
    -2.139653284152173,
    -0.8943584651227547,
    0.9887988450116013,
    0.3983701582610333,
)
#: tuple: Output guard for the closure fit -- its training range plus
#: 5 × :data:`OUT_CLAMP_WIDTH`, capped below 1 (Q17).
MU_INF_L23_CLOSURE_RANGE = (0.4135738177525006, 0.994010053503653)


#: The backbone variants :func:`rrs_ZTT` knows. ``"l23"`` -- the M7 refit µ∞
#: and µd, plus L23's FF phase function where ``PhaseParams`` carries it -- was
#: the default from M7 to M8 task 9c. ``"tt2017"`` is the M0–M6 model, bit for
#: bit. ``"pb24"`` (M8
#: tasks 8a–8b) is "ZTT (PB24)": the PB24 diffuse-fraction table behind the
#: seam, µd and µ∞ refit to PB24, FF Pbb as ``"l23"``, Ψ_KLu and f_L refit
#: jointly through PB24's ``rrs`` (:func:`psi_KLu_pb24`, :func:`f_L_pb24`),
#: and the denominator floor (:func:`denominator_floor_pb24`).
#: ``"pb24_nadir"`` (M8 task 9, ``rt_fit_pb24.md`` Q45) is "ZTT (PB24_nadir)",
#: **the default**: the ``"pb24"`` variant's IOP-side terms (PB24's diffuse
#: fraction, µd and µ∞ of task 8a) with **TT's own Ψ_KLu and f_L** and the
#: denominator floor -- the M7 pattern on PB24. **For nadir viewing (θv = 0)
#: and bb/a below** :data:`NADIR_BB_OVER_A_MAX`; outside either it warns.
VARIANTS = ("l23", "tt2017", "pb24", "pb24_nadir")
#: str: The default of :func:`rrs_ZTT`'s ``variant``. ``"pb24_nadir"`` since M8
#: task 9c (JXP's decision, Q45): held out on PB24 at nadir it reads 4.36%
#: median against 4.73% for ``"l23"`` and 6.52% for Gordon, with no
#: non-physical value; on L23 its held-out rRMS is 3.64% with its own (PB24)
#: diffuse fraction and 3.39% with TT's, against 2.90% for ``"l23"``, which
#: was fitted there.
DEFAULT_VARIANT = "pb24_nadir"
#: µ∞ choices for ``mu_inf_model``. ``"l23_closure"`` is M7 task 5's ceiling.
MU_INF_MODELS = ("l23", "l23_closure", "tt2017", "pb24")
#: µd choices for ``mu_d_model`` (the ladder of M7 task 6 toggles it alone).
MU_D_MODELS = ("l23", "tt2017", "pb24")
#: Ψ_KLu / f_L choices for ``angular_model`` (M8 task 8b): ``"tt"`` is TT's
#: pair (Equations (4), (6) and (31)); ``"pb24"`` the joint PB24 refit.
#: ``"pb24_klu"`` is the nadir pair fitted to PB24's measured ``KLu`` (M8 task
#: 9b; :func:`psi_KLu_pb24_nadir`, :func:`f_L_pb24_nadir`) -- kept as a
#: record, used by no variant: it failed its gate (11.00% median).
ANGULAR_MODELS = ("tt", "pb24", "pb24_klu")


def mu_infinity_l23(bb_over_a, eta_bb):
    """µ∞ from the L23 refit of Equation (8), clamped and guarded.

    Parameters
    ----------
    bb_over_a, eta_bb : Array

    Returns
    -------
    Array
        µ∞, dimensionless.
    """
    return mu_infinity(
        bb_over_a,
        eta_bb,
        MU_INF_L23_COEFFS,
        box=ZTT_L23_BOX,
        out_range=MU_INF_L23_RANGE,
    )


def mu_infinity_l23_closure(bb_over_a, eta_bb):
    """The closure µ∞ fit (:data:`MU_INF_L23_CLOSURE_COEFFS`): **a ceiling, not
    physics**. Clamped to :data:`ZTT_L23_BOX` and guarded like
    :func:`mu_infinity_l23`.

    Parameters
    ----------
    bb_over_a, eta_bb : Array

    Returns
    -------
    Array
    """
    return mu_infinity(
        bb_over_a,
        eta_bb,
        MU_INF_L23_CLOSURE_COEFFS,
        box=ZTT_L23_BOX,
        out_range=MU_INF_L23_CLOSURE_RANGE,
    )


def mu_d_l23(
    theta_s_air_deg,
    bb_over_a,
    eta_bb,
    visibility_km=DEFAULT_VISIBILITY_KM,
    *,
    H=None,
):
    """µd from the L23 refit: TT's ``M+_d`` × c(θs') × refit ``M*_d``.

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees).
    bb_over_a, eta_bb : Array
    visibility_km : float, optional
        Enters TT's ``M+_d`` only; the correction was fitted at L23's sky.
    H : Array, optional
        The diffuse fraction in place of TT's ``H(θs', V)`` inside ``M+_d``
        (:func:`Md_plus`). M7's ``c(θs')`` still multiplies it: it was fitted
        jointly with TT's H on L23, so with a supplied H it is a correction
        for a different sky (whether to keep it is M8 task 7's question).
        ``None`` (default) is bit for bit the M7 result.

    Returns
    -------
    Array
        µd, dimensionless.
    """
    return mu_d(
        theta_s_air_deg,
        bb_over_a,
        eta_bb,
        visibility_km,
        md_star_coeffs=MD_STAR_L23_COEFFS,
        md_plus_corr=MD_PLUS_CORR_L23,
        box=ZTT_L23_BOX,
        md_star_range=MD_STAR_L23_RANGE,
        H=H,
    )


# ------------------------------------------- the PB24 diffuse fraction (M8) --
#: The shipped PB24 diffuse-fraction table (M8 task 8a, ``rt_fit_pb24.md`` Q21a):
#: ``H(θs', λ) = 1 − Esdir_Es_ratio`` on PB24's ten sun zeniths × 451 bands,
#: written by ``design/py/fit_ztt_pb24.py``. The field is an above-surface
#: quantity and is identical in every one of the 5,000 files (checked bit for
#: bit by the script and by a test), so the table *is* PB24's sky: RADTRAN-X
#: irradiances, 40 km visibility, US Navy maritime aerosol, 354.9 DU ozone
#: (Pitarch & Brando 2025). Another sky needs its own H through
#: ``Geometry.diffuse_fraction``.
H_PB24_FILE = "files/diffuse_fraction_pb24.npz"


@functools.lru_cache(maxsize=1)
def _h_pb24_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """``(sec θs' ascending, λ ascending, H (n_theta, n_wave), provenance)``
    from :data:`H_PB24_FILE`, read once."""
    path = Path(__file__).parent / H_PB24_FILE
    with np.load(path) as data:
        theta_s = np.asarray(data["theta_s"], dtype=np.float64)
        wave = np.asarray(data["wave"], dtype=np.float64)
        H = np.asarray(data["H"], dtype=np.float64)
        provenance = json.loads(str(data["provenance"]))
    sec = 1.0 / np.cos(np.deg2rad(theta_s))  # air mass, ascending with θs'
    order = np.argsort(sec)
    return sec[order], wave, H[order], provenance


def _bilinear(x_nodes, y_nodes, table, x, y):
    """Bilinear interpolation of ``table (n_x, n_y)`` at ``x (..., 1)`` or
    scalar and ``y (n,)``, giving ``(..., n)``; both node arrays ascending;
    inputs clamped to the node range (flat outside, as a sky table should be)."""
    xn, yn, F = jnp.asarray(x_nodes), jnp.asarray(y_nodes), jnp.asarray(table)
    x = jnp.clip(jnp.asarray(x), xn[0], xn[-1])
    y = jnp.clip(jnp.asarray(y), yn[0], yn[-1])
    i = jnp.clip(jnp.searchsorted(xn, x, side="right") - 1, 0, xn.size - 2)
    j = jnp.clip(jnp.searchsorted(yn, y, side="right") - 1, 0, yn.size - 2)
    tx = (x - xn[i]) / (xn[i + 1] - xn[i])
    ty = (y - yn[j]) / (yn[j + 1] - yn[j])
    f00, f01 = F[i, j], F[i, j + 1]
    f10, f11 = F[i + 1, j], F[i + 1, j + 1]
    return (1 - tx) * ((1 - ty) * f00 + ty * f01) + tx * ((1 - ty) * f10 + ty * f11)


def diffuse_fraction_pb24(theta_s_air_deg, wave: Float[Array, " wave"] | None = None):
    """PB24's diffuse fraction ``H(θs', λ)`` from the shipped table (M8 task 8a).

    Linear in the air mass ``sec θs'`` between PB24's ten sun zeniths
    (0–87.75°) and linear in λ between its 451 bands (350–800 nm); flat
    outside both ranges (so finite at θs' → 90°). The variable was chosen by
    measurement (M8 task 8a): leaving each interior node out and interpolating
    it from its neighbours, ``sec θs'`` reproduces H within 0.1–3.1% at
    10–60° where ``cos θs'`` gives 0.1–6.9% and θs' itself 1.5–8.3%; the real
    node spacing is half the leave-one-out spacing. At the nodes it is the
    files' ``1 − Esdir_Es_ratio`` exactly (the table stores the float32
    values). This is the default ``H`` of the ``"pb24"`` variant
    (:func:`mu_d_pb24`); a ``Geometry.diffuse_fraction`` overrides it.

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees), shape ``(...)`` -- one spectrum per
        entry; a trailing axis of length 1 (``rrs_ZTT`` passes ``(..., 1)``) is
        accepted as it is.
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid.

    Returns
    -------
    Array
        ``H``, shape ``(..., n_wave)``.
    """
    sec_nodes, wave_nodes, table, _ = _h_pb24_table()
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)
    theta = jnp.asarray(theta_s_air_deg)
    if theta.ndim and theta.shape[-1] != 1:
        theta = theta[..., None]
    x = 1.0 / jnp.cos(jnp.deg2rad(theta))
    return _bilinear(sec_nodes, wave_nodes, table, x, wave)


# ------------------------------------------------- the PB24 fit (M8 8a) --
#: **"ZTT (PB24)", the IOP-side terms** (M8 task 8a, ``rt_fit_pb24.md`` Q21–Q22;
#: record §7c.8). Fitted by ``design/py/fit_ztt_pb24.py`` on the realisation
#: split's 4,000 training realisations (``design/ztt_pb24/splits.json``), all
#: 451 bands, by streamed Householder QR; ``design/ztt_pb24/fit_coefficients.json``
#: is the source and ``--check`` keeps these honest. Equation (8)'s sixteen
#: coefficients refit to the eigen µ∞ on PB24's own IOPs and FF pair (task 6's
#: product), relative residuals, **no B_p term** (Q22): held out, median
#: |error| 1.10% (L23's fit: 0.90% in the bulk but +48% beyond its clamp), with
#: the B_p trend the form cannot absorb left in (+1.7% at B_p < 0.003, −3.1%
#: at > 0.03).
MU_INF_PB24_COEFFS = (
    0.001579666317308423,
    0.0021735695271569433,
    -0.03224377697203383,
    -0.013809383461016886,
    -0.0010122522850343992,
    -0.05143424378751467,
    -0.22799224742165938,
    -0.15578216091513747,
    -0.021181878575983446,
    -0.1528731881609277,
    -0.35546094744201245,
    -0.5634061625211141,
    0.006596981487185379,
    0.049053367638090636,
    0.10874196386435862,
    0.3399167524959902,
)

#: Equation (17)'s eight ``M*_d`` coefficients refit to PB24's ``mu_d`` *jointly*
#: with :data:`MD_PLUS_CORR_PB24`, with the PB24 diffuse fraction inside TT's
#: ``M+_d``, on the window suns (θs' ≤ 70°; 14.4 M rows). Held out: 0.4–1.0%
#: median per sun at 0–70°, 0.4–0.8% at bb/a < 0.59 and 1.9% / 3.0% at
#: 0.59–2 / > 2 (the L23 fit with the same H: 12% / 31% there).
MD_STAR_PB24_COEFFS = (
    -0.0038808591531856998,
    -0.012520018504516502,
    -0.01735221961009114,
    -0.09617021774965247,
    -0.013557833203465472,
    -0.23878541741698955,
    0.007519636892949682,
    0.8222051423461154,
)

#: ``(c1, c2)`` of :func:`md_plus_correction` for PB24's sky, fitted to the
#: seven free nodes 10–70° (c(30°) = 0.981, c(60°) = 0.933); the quadratic
#: leaves node residuals of up to 0.85% (at 30°), the form's limit.
MD_PLUS_CORR_PB24 = (-0.14396867927226809, 0.018416894768715587)

#: ``(X_lo, X_hi, L_lo, L_hi)``: the (log10 bb/a, log10 η_bb) range of PB24's
#: training rows -- bb/a 6e-5 to 17, η_bb 9e-6 to 1 -- which both PB24 refits
#: clamp their inputs to (as :data:`ZTT_L23_BOX` does for L23).
ZTT_PB24_BOX = (
    -4.205851915186109,
    1.2285661070062504,
    -5.056076609727998,
    -0.0017276344807632311,
)

#: Output guards, as M7's: the training range plus 5 × :data:`OUT_CLAMP_WIDTH`,
#: µ∞ capped below 1.
MU_INF_PB24_RANGE = (0.06568731709685627, 0.999)
#: tuple: The same guard for ``M*_d``.
MD_STAR_PB24_RANGE = (0.5188474099856758, 1.0579228041730246)

#: The leak guard (as the Fresnel transfer's): the number and the digest
#: (:func:`robust.rt.conventions.realisation_digest`) of the realisation ids
#: every PB24 fit above was made on. A test checks they are the realisation
#: split's training ids and nothing else.
ZTT_PB24_N_TRAINING = 4000
ZTT_PB24_TRAINING_DIGEST = (
    "3dd4bcb017fcded0f23f7a899cf091b6fd64e74736f77261d1eec92ca541219c"
)


def mu_infinity_pb24(bb_over_a, eta_bb):
    """µ∞ from the PB24 refit of Equation (8), clamped and guarded.

    Parameters
    ----------
    bb_over_a, eta_bb : Array

    Returns
    -------
    Array
        µ∞, dimensionless.
    """
    return mu_infinity(
        bb_over_a,
        eta_bb,
        MU_INF_PB24_COEFFS,
        box=ZTT_PB24_BOX,
        out_range=MU_INF_PB24_RANGE,
    )


def mu_d_pb24(
    theta_s_air_deg,
    bb_over_a,
    eta_bb,
    *,
    H=None,
    wave: Float[Array, " wave"] | None = None,
):
    """µd from the PB24 fit: TT's ``M+_d`` with PB24's H × c_PB24(θs') × refit
    ``M*_d``.

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees).
    bb_over_a, eta_bb : Array
    H : Array, optional
        The diffuse fraction. ``None`` (default) takes PB24's own,
        :func:`diffuse_fraction_pb24` at ``(theta_s_air_deg, wave)``; a
        ``Geometry.diffuse_fraction`` arrives here through :func:`rrs_ZTT`.
        TT's ``H(θs', V)`` is never used by this variant.
    wave : Array, optional
        Wavelengths (nm) for the table lookup when ``H`` is ``None``;
        defaults to the canonical grid.

    Returns
    -------
    Array
        µd, dimensionless.
    """
    if H is None:
        H = diffuse_fraction_pb24(theta_s_air_deg, wave)
    return mu_d(
        theta_s_air_deg,
        bb_over_a,
        eta_bb,
        md_star_coeffs=MD_STAR_PB24_COEFFS,
        md_plus_corr=MD_PLUS_CORR_PB24,
        box=ZTT_PB24_BOX,
        md_star_range=MD_STAR_PB24_RANGE,
        H=H,
    )


# ------------------------------------------- the PB24 angular terms (M8 8b) --
#: **"ZTT (PB24)", the angular terms** (M8 task 8b, ``rt_fit_pb24.md`` Q23–Q25,
#: Q30; record §7c.9). Fitted by ``design/py/fit_ztt_pb24_angular.py`` on the
#: window (θs, θv_air ≤ 70°) of the realisation split's 4,000 training
#: realisations, all 451 bands, **linearly in D-space**: with 8a's µd and µ∞
#: and the exact FF Pbb, Equation (12) inverts to a closure denominator
#: ``D* = [β(ψ)/bb] / (µd · rrs_PB24)`` per (sample, λ), and
#: ``D* − (a/bb)(1 + |cos θv|/µ∞) − 1/b̃b``
#: ``= (a/bb)(|cos θv|/µ∞) F(ψ) − (1/b̃b − 1) f_L(ψ, λ)``
#: is linear in the coefficients of ``F`` and ``f_L``; relative weighting in
#: D (relative weighting in rrs to first order); two alternating linear passes
#: for the bilinear ``f_L``; ``design/ztt_pb24/angular_coefficients.json`` is
#: the source and ``--check`` keeps these honest.
#:
#: **Off-nadir, Ψ_KLu is a calibration through rrs, not a measured K_Lu/K∞**:
#: PB24 tabulates ``KLu`` at nadir only, so the refit is physical at
#: ψ = 180° − θs_w (where it is checked against PB24's KLu/a) and a fitted
#: BRDF shape elsewhere.
#:
#: ``F(ψ)`` as a polynomial in ``u = 1 + cos ψ`` (a polynomial in cos ψ, Q30,
#: written so that ψ = 180° is ``u = 0`` and there is no cancellation: TT's
#: quartic in degrees cancels by 78,000× there). Highest order first, for
#: ``jnp.polyval``; Ψ_KLu = 1 + F. **Measured** (held out, window): the
#: refit Ψ_KLu is 0.74–0.89 over 91–180° (TT's: 1.02 at 180°, 1.32 at 135°),
#: not monotone; it fixes the BRDF *shape* (rrs(θv, Δφ)/rrs(nadir) to 0.4–3.8%
#: median per θv against 1.6–12.9% for TT's pair) and removes every
#: non-physical value, but the *level* is worse: window median 10.86%
#: against 8.96% with TT's pair, growing with θs, because one function of ψ
#: alone cannot follow both the nadir sun trend (K_Lu/a rises with θs; TT's
#: pair follows it at a constant 1.20×) and the off-nadir view shape (record
#: §7c.9). Adding F terms does not help (3–7 terms: 15.5% rRMS in D alike).
PSI_KLU_PB24_COEFFS = (
    -0.49165067238411775,
    1.7652985900107052,
    -1.261974730012199,
    -0.010032317710123306,
    -0.11519048273114063,
)

#: Wavelengths of :data:`FL_AVE_PB24` (nm): PB24's 1 nm grid, 350–800 nm.
FL_AVE_PB24_WAVE = np.arange(350.0, 800.0 + 0.5, 1.0)

#: ``f_L,ave(λ)`` refit per band (Q25), normalised to a mean of 1 over
#: 350–800 nm (the scale lives in :data:`FL_S_PB24`); rounded to 1e-6. Flatter
#: than Table A3 (0.993–1.010 against 0.981–1.026): with PB24's own µd and
#: FF Pbb in the closure, little of the spectral shape remains.
FL_AVE_PB24 = np.array(
    [
        1.009770, 1.009721, 1.009672, 1.009620, 1.009571, 1.009529, 1.009477, 1.009423,
        1.009369, 1.009317, 1.009269, 1.009220, 1.009184, 1.009134, 1.009084, 1.009033,
        1.008983, 1.008933, 1.008884, 1.008845, 1.008797, 1.008746, 1.008697, 1.008648,
        1.008598, 1.008548, 1.008499, 1.008460, 1.008410, 1.008361, 1.008311, 1.008260,
        1.008209, 1.008157, 1.008106, 1.008055, 1.008013, 1.007962, 1.007911, 1.007862,
        1.007812, 1.007761, 1.007711, 1.007660, 1.007612, 1.007582, 1.007535, 1.007487,
        1.007441, 1.007396, 1.007352, 1.007309, 1.007266, 1.007221, 1.007178, 1.007138,
        1.007094, 1.007046, 1.006999, 1.006953, 1.006908, 1.006861, 1.006814, 1.006769,
        1.006724, 1.006678, 1.006641, 1.006597, 1.006551, 1.006507, 1.006461, 1.006416,
        1.006370, 1.006324, 1.006278, 1.006234, 1.006190, 1.006146, 1.006112, 1.006070,
        1.006029, 1.005988, 1.005947, 1.005906, 1.005865, 1.005825, 1.005785, 1.005749,
        1.005712, 1.005676, 1.005639, 1.005612, 1.005576, 1.005539, 1.005501, 1.005467,
        1.005431, 1.005392, 1.005353, 1.005312, 1.005271, 1.005222, 1.005172, 1.005118,
        1.005065, 1.005010, 1.004963, 1.004908, 1.004853, 1.004798, 1.004743, 1.004690,
        1.004637, 1.004582, 1.004527, 1.004472, 1.004416, 1.004358, 1.004299, 1.004243,
        1.004187, 1.004130, 1.004080, 1.004023, 1.003965, 1.003908, 1.003850, 1.003793,
        1.003735, 1.003677, 1.003620, 1.003563, 1.003506, 1.003449, 1.003392, 1.003335,
        1.003278, 1.003223, 1.003168, 1.003114, 1.003067, 1.003014, 1.002960, 1.002909,
        1.002856, 1.002805, 1.002753, 1.002703, 1.002651, 1.002597, 1.002541, 1.002486,
        1.002429, 1.002373, 1.002316, 1.002263, 1.002209, 1.002158, 1.002106, 1.002054,
        1.002006, 1.001950, 1.001893, 1.001831, 1.001769, 1.001700, 1.001631, 1.001556,
        1.001480, 1.001406, 1.001331, 1.001255, 1.001177, 1.001100, 1.001024, 1.000948,
        1.000872, 1.000797, 1.000722, 1.000648, 1.000573, 1.000500, 1.000425, 1.000356,
        1.000282, 1.000208, 1.000134, 1.000060, 0.999986, 0.999912, 0.999837, 0.999762,
        0.999687, 0.999613, 0.999537, 0.999462, 0.999387, 0.999310, 0.999232, 0.999156,
        0.999079, 0.998998, 0.998916, 0.998831, 0.998747, 0.998661, 0.998575, 0.998488,
        0.998409, 0.998325, 0.998241, 0.998158, 0.998077, 0.997997, 0.997920, 0.997844,
        0.997771, 0.997703, 0.997637, 0.997575, 0.997514, 0.997458, 0.997404, 0.997355,
        0.997307, 0.997264, 0.997223, 0.997185, 0.997147, 0.997117, 0.997087, 0.997060,
        0.997035, 0.997016, 0.996996, 0.996978, 0.996959, 0.996942, 0.996921, 0.996906,
        0.996890, 0.996871, 0.996850, 0.996833, 0.996815, 0.996803, 0.996790, 0.996779,
        0.996769, 0.996774, 0.996781, 0.996746, 0.996714, 0.996707, 0.996702, 0.996688,
        0.996678, 0.996666, 0.996657, 0.996646, 0.996636, 0.996628, 0.996620, 0.996613,
        0.996606, 0.996599, 0.996591, 0.996583, 0.996574, 0.996569, 0.996567, 0.996557,
        0.996546, 0.996533, 0.996521, 0.996509, 0.996497, 0.996488, 0.996478, 0.996462,
        0.996445, 0.996436, 0.996428, 0.996415, 0.996402, 0.996388, 0.996373, 0.996359,
        0.996345, 0.996327, 0.996307, 0.996297, 0.996286, 0.996271, 0.996257, 0.996244,
        0.996232, 0.996228, 0.996229, 0.996231, 0.996240, 0.996273, 0.996315, 0.996368,
        0.996431, 0.996506, 0.996592, 0.996694, 0.996800, 0.996918, 0.997042, 0.997152,
        0.997264, 0.997371, 0.997476, 0.997573, 0.997664, 0.997749, 0.997826, 0.997889,
        0.997942, 0.997990, 0.998028, 0.998043, 0.998046, 0.998037, 0.998014, 0.997987,
        0.997943, 0.997871, 0.997783, 0.997689, 0.997579, 0.997451, 0.997306, 0.997146,
        0.996970, 0.996781, 0.996577, 0.996365, 0.996141, 0.995907, 0.995663, 0.995419,
        0.995171, 0.994928, 0.994690, 0.994456, 0.994232, 0.994014, 0.993814, 0.993655,
        0.993513, 0.993391, 0.993283, 0.993196, 0.993119, 0.993052, 0.992992, 0.992970,
        0.992960, 0.992950, 0.992946, 0.992961, 0.992979, 0.993028, 0.993079, 0.993148,
        0.993223, 0.993336, 0.993455, 0.993552, 0.993647, 0.993760, 0.993870, 0.994025,
        0.994181, 0.994328, 0.994471, 0.994757, 0.995043, 0.995142, 0.995247, 0.995360,
        0.995470, 0.995562, 0.995653, 0.995719, 0.995791, 0.995863, 0.995934, 0.995967,
        0.995999, 0.996030, 0.996068, 0.996090, 0.996111, 0.996113, 0.996116, 0.996124,
        0.996129, 0.996126, 0.996119, 0.996114, 0.996110, 0.996095, 0.996080, 0.996058,
        0.996043, 0.996022, 0.995999, 0.995970, 0.995941, 0.995916, 0.995887, 0.995860,
        0.995839, 0.995761, 0.995687, 0.995616, 0.995547, 0.995494, 0.995442, 0.995387,
        0.995330, 0.995264, 0.995190, 0.995101, 0.995016, 0.994922, 0.994832, 0.994745,
        0.994658, 0.994571, 0.994483, 0.994408, 0.994336, 0.994280, 0.994223, 0.994164,
        0.994106, 0.994043, 0.993981, 0.993929, 0.993882, 0.993870, 0.993855, 0.993831,
        0.993823, 0.993806, 0.993826,
    ]
)  # fmt: skip

#: ``(s0, s1, s2)`` of the ψ scaling ``s(ψ) = s0 + s1 sin ψ + s2 cos ψ`` of
#: :func:`f_L_pb24` (TT's Equation (31) has ``s2 = 0``): 1.023–1.034 over
#: 91–180° where TT's reads 1.04–1.12, so ``(1 − f_L)/b̃b`` is small and the
#: second sign-flip mechanism of M8 task 7 (f_L > 1 at low B_p) is nearly
#: gone -- 0.08% of window values unfloored, none floored.
FL_S_PB24 = (1.019010949931932, 0.004188609681042782, -0.01471603702979093)

#: ``(lo, width)`` of :func:`denominator_floor_pb24`: a smooth lower bound on
#: Equation (12)'s denominator D so that the variant is finite everywhere.
#: PB24's own D* exceeds 1.09 on every training window value; the floor at
#: 0.5 is invisible there (a softplus corner of width 0.01 moves D by
#: ``width · ln(1 + e^{(lo − D)/width})``: below float64 resolution for
#: D > 0.9). Measured footprint: 0.35% of held-out window values moved by
#: more than 1e-4, all where the unfloored D is near or below zero.
D_FLOOR_PB24 = (0.5, 0.01)


def F_psi_pb24(psi_deg: Float[Array, "..."], coeffs=None) -> Float[Array, "..."]:
    """``F(ψ)`` of the PB24 refit: a polynomial in ``u = 1 + cos ψ``.

    Parameters
    ----------
    psi_deg : Array
        In-water scattering angle (degrees).
    coeffs : sequence of float, optional
        Highest order first; default :data:`PSI_KLU_PB24_COEFFS`.

    Returns
    -------
    Array
        ``F(ψ)``, dimensionless; ``Ψ_KLu = 1 + F``.
    """
    c = PSI_KLU_PB24_COEFFS if coeffs is None else coeffs
    u = 1.0 + jnp.cos(jnp.deg2rad(jnp.asarray(psi_deg)))
    return jnp.polyval(jnp.asarray(c, dtype=u.dtype), u)


def psi_KLu_pb24(psi_deg: Float[Array, "..."], coeffs=None) -> Float[Array, "..."]:
    """``Ψ_KLu(ψ) = 1 + F(ψ)`` of the PB24 refit (:func:`F_psi_pb24`).

    Physical (K_Lu/K∞) at nadir, ψ = 180° − θs_w, where PB24 measures
    ``KLu``; off-nadir a calibration of Equation (12)'s BRDF shape through
    ``rrs`` (M8 task 8b).

    Parameters
    ----------
    psi_deg : Array
        In-water scattering angle (degrees).
    coeffs : sequence of float, optional
        As :func:`F_psi_pb24`.

    Returns
    -------
    Array
        ``Ψ_KLu``, dimensionless.
    """
    return 1.0 + F_psi_pb24(psi_deg, coeffs)


def f_L_pb24(
    psi_deg,
    wave: Float[Array, " wave"] | None = None,
    *,
    fl_ave=None,
    fl_wave=None,
    s=None,
):
    """``f_L(ψ, λ) = f_L,ave(λ) [s0 + s1 sin ψ + s2 cos ψ]`` of the PB24 refit.

    TT's factorisation (Equation (31)) with the spectral shape refit per band
    (:data:`FL_AVE_PB24`, Q25) and a cos ψ term added to the scaling
    (:data:`FL_S_PB24`).

    Parameters
    ----------
    psi_deg : Array
        In-water scattering angle (degrees).
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid. Linear interpolation
        on :data:`FL_AVE_PB24_WAVE`, flat outside 350–800 nm.
    fl_ave, fl_wave, s : optional
        Override the shipped spectral shape (on ``fl_wave``, default
        :data:`FL_AVE_PB24_WAVE`) and ``(s0, s1, s2)``; the fit script
        evaluates its candidates this way.

    Returns
    -------
    Array
        ``f_L``, dimensionless, broadcasting ψ against λ.
    """
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)
    fl_ave = FL_AVE_PB24 if fl_ave is None else fl_ave
    fl_wave = FL_AVE_PB24_WAVE if fl_wave is None else fl_wave
    s0, s1, s2 = FL_S_PB24 if s is None else s
    shape = jnp.interp(wave, jnp.asarray(fl_wave), jnp.asarray(fl_ave))
    psi = jnp.deg2rad(jnp.asarray(psi_deg))
    return shape * (s0 + s1 * jnp.sin(psi) + s2 * jnp.cos(psi))


def denominator_floor_pb24(D, lo: float | None = None, width: float | None = None):
    """A smooth lower bound on Equation (12)'s denominator: the softplus corner
    of :func:`smooth_clip`, one-sided.

    ``D + width · ln(1 + e^{(lo − D)/width})``: the identity to within
    ``width · e^{−d/width}`` at distance ``d`` above ``lo``, ``→ lo`` far
    below it, monotone and smooth, so ``rrs`` stays finite and differentiable
    where the fitted denominator would cross zero (the ψ < 110° sign flip of
    TT's form, record §7.12, and the low-B_p one of M8 task 7).

    Parameters
    ----------
    D : Array
    lo, width : float, optional
        Default :data:`D_FLOOR_PB24`.

    Returns
    -------
    Array
    """
    lo0, w0 = D_FLOOR_PB24
    lo = lo0 if lo is None else lo
    width = w0 if width is None else width
    D = jnp.asarray(D)
    return D + width * jnp.logaddexp(0.0, (lo - D) / width)


# --------------------------------- the nadir-only PB24 terms (M8 task 9) --
#: float: Above-water view zenith (degrees) beyond which
#: ``rrs_ZTT(variant="pb24_nadir")`` warns
#: (:class:`robust.rt.conventions.DomainWarning`) when ``geometry.theta_v`` is a
#: concrete array (``rt_fit_pb24.md`` Q40).
NADIR_TOL_DEG = 0.5

#: float: The ``bb/a`` above which ``rrs_ZTT(variant="pb24_nadir")`` warns
#: (``rt_fit_pb24.md`` Q45). Below it (95% of PB24's nadir values) the variant
#: reads 1.4–8.4% median per bb/a bin, held out (4.03% pooled); above it
#: Equation (12) with TT's f_L > 1 nears its second pole and reads 67% at
#: bb/a 0.59–2 and 228% beyond (Gordon: 16% and 20%). The floor keeps it
#: finite there, not right.
NADIR_BB_OVER_A_MAX = 0.5946

#: float: The smallest scattering angle (degrees) the nadir Ψ_KLu polynomial
#: is evaluated at; below it Ψ_KLu is held flat. Nadir viewing reaches
#: ψ = 131.8° at PB24's largest sun (87.75°), and the fit used 135.5–180°
#: (suns 0–70°), so anything below this is an off-nadir call the variant has
#: no calibration for.
PSI_KLU_PB24_NADIR_PSI_MIN = 130.0

#: **The measured-K_Lu nadir fit, Ψ_KLu** (``angular_model="pb24_klu"``; M8
#: task 9b, ``rt_fit_pb24.md`` Q35 (b), Q36 (a), Q37 (a); record §7c.12).
#: This was "ZTT (PB24_nadir)" as first built; after Q45 the variant of that
#: name uses TT's pair instead and this fit is kept as the record.
#: ``Ψ_KLu(ψ) = 1 + F(ψ)``, ``F`` a polynomial in
#: ``u = 1 + cos ψ`` (highest order first), IOP-independent as TT wrote it,
#: fitted by ``design/py/fit_ztt_pb24_nadir.py`` to PB24's **measured** nadir
#: ``KLu``: per window sun the geometric mean of ``(KLu/a) · µ∞`` (µ∞ the 8a
#: refit, :func:`mu_infinity_pb24`) over the realisation split's training
#: rows and all 451 bands, then the polynomial through those eight nodes.
#: ``design/ztt_pb24/nadir_coefficients.json`` is the source and ``--check``
#: keeps it honest.
#:
#: **Measured, and it fails its gate** (``design/ztt_pb24/nadir_metrics.md``;
#: held out, suns 0–70°, all bands): Ψ_KLu runs 0.80 at ψ = 180° to 1.17 at
#: 135.5° (TT's: 1.02 to 1.32) and reproduces PB24's KLu/a to 15.6% (median
#: |dev|; TT's pair 20.3%), but ``rrs`` reads **11.00% median** where TT's
#: Ψ_KLu and f_L on the same 8a terms read 4.36% and Gordon 6.52%; on L23 it
#: reads 11.8% rRMS against 2.90% for ``"l23"``. The measured K_Lu/K∞ is not
#: IOP-independent -- at θs = 0 it falls from 1.00 at bb/a < 0.001 (TT's
#: value) to 0.5 at bb/a > 0.6 -- so its average is 20% low exactly where
#: ``rrs`` is clearest. The pair is therefore **not used by any variant**;
#: ``"pb24_nadir"`` takes TT's pair (``rt_fit_pb24.md`` Q45).
PSI_KLU_PB24_NADIR_COEFFS = (
    1.6048298222426556,
    0.8039000600433032,
    -0.196688728781297,
)

#: ``f_L,ave(λ)`` of the nadir fit on :data:`FL_AVE_PB24_WAVE` (mean 1 over
#: 350–800 nm, rounded to 1e-6): per band, by linear closure on PB24's nadir
#: ``rrs`` with the Ψ_KLu above, 8a's µd and µ∞ and the exact FF Pbb.
FL_AVE_PB24_NADIR = np.array(
    [
        1.011986, 1.011917, 1.011849, 1.011778, 1.011711, 1.011647, 1.011576, 1.011502,
        1.011430, 1.011360, 1.011295, 1.011229, 1.011172, 1.011106, 1.011039, 1.010971,
        1.010906, 1.010840, 1.010775, 1.010716, 1.010653, 1.010587, 1.010523, 1.010459,
        1.010394, 1.010328, 1.010265, 1.010207, 1.010144, 1.010079, 1.010015, 1.009950,
        1.009884, 1.009817, 1.009750, 1.009685, 1.009625, 1.009560, 1.009495, 1.009431,
        1.009367, 1.009302, 1.009239, 1.009174, 1.009112, 1.009063, 1.009003, 1.008942,
        1.008884, 1.008827, 1.008772, 1.008717, 1.008663, 1.008607, 1.008553, 1.008500,
        1.008446, 1.008386, 1.008327, 1.008272, 1.008215, 1.008157, 1.008100, 1.008043,
        1.007988, 1.007931, 1.007879, 1.007824, 1.007768, 1.007713, 1.007657, 1.007600,
        1.007544, 1.007487, 1.007430, 1.007376, 1.007322, 1.007269, 1.007221, 1.007170,
        1.007120, 1.007069, 1.007019, 1.006969, 1.006919, 1.006869, 1.006820, 1.006776,
        1.006730, 1.006685, 1.006639, 1.006601, 1.006556, 1.006510, 1.006463, 1.006419,
        1.006374, 1.006326, 1.006277, 1.006226, 1.006175, 1.006115, 1.006055, 1.005989,
        1.005924, 1.005857, 1.005796, 1.005729, 1.005664, 1.005597, 1.005531, 1.005467,
        1.005403, 1.005337, 1.005271, 1.005204, 1.005137, 1.005067, 1.004997, 1.004930,
        1.004862, 1.004794, 1.004731, 1.004662, 1.004592, 1.004524, 1.004455, 1.004386,
        1.004318, 1.004249, 1.004180, 1.004112, 1.004044, 1.003976, 1.003908, 1.003841,
        1.003773, 1.003708, 1.003642, 1.003579, 1.003519, 1.003457, 1.003393, 1.003332,
        1.003269, 1.003209, 1.003148, 1.003088, 1.003027, 1.002963, 1.002896, 1.002832,
        1.002765, 1.002699, 1.002632, 1.002570, 1.002507, 1.002447, 1.002386, 1.002325,
        1.002267, 1.002202, 1.002135, 1.002063, 1.001991, 1.001910, 1.001829, 1.001741,
        1.001653, 1.001565, 1.001478, 1.001389, 1.001298, 1.001208, 1.001119, 1.001030,
        1.000942, 1.000854, 1.000767, 1.000680, 1.000593, 1.000507, 1.000420, 1.000337,
        1.000251, 1.000164, 1.000078, 0.999993, 0.999907, 0.999820, 0.999733, 0.999646,
        0.999558, 0.999473, 0.999385, 0.999299, 0.999211, 0.999122, 0.999032, 0.998943,
        0.998855, 0.998761, 0.998666, 0.998568, 0.998471, 0.998372, 0.998272, 0.998172,
        0.998077, 0.997980, 0.997884, 0.997788, 0.997694, 0.997602, 0.997513, 0.997426,
        0.997341, 0.997263, 0.997187, 0.997114, 0.997045, 0.996979, 0.996917, 0.996859,
        0.996803, 0.996753, 0.996705, 0.996661, 0.996617, 0.996580, 0.996546, 0.996514,
        0.996485, 0.996462, 0.996438, 0.996417, 0.996394, 0.996373, 0.996348, 0.996331,
        0.996312, 0.996289, 0.996265, 0.996246, 0.996225, 0.996213, 0.996197, 0.996186,
        0.996175, 0.996182, 0.996191, 0.996150, 0.996113, 0.996105, 0.996100, 0.996083,
        0.996072, 0.996057, 0.996046, 0.996032, 0.996019, 0.996009, 0.996000, 0.995990,
        0.995981, 0.995972, 0.995962, 0.995952, 0.995941, 0.995935, 0.995930, 0.995918,
        0.995904, 0.995890, 0.995875, 0.995861, 0.995847, 0.995836, 0.995824, 0.995805,
        0.995785, 0.995775, 0.995765, 0.995750, 0.995734, 0.995717, 0.995700, 0.995684,
        0.995667, 0.995646, 0.995623, 0.995611, 0.995599, 0.995581, 0.995565, 0.995549,
        0.995536, 0.995531, 0.995532, 0.995534, 0.995545, 0.995583, 0.995633, 0.995694,
        0.995767, 0.995856, 0.995956, 0.996074, 0.996198, 0.996338, 0.996484, 0.996614,
        0.996748, 0.996875, 0.997001, 0.997117, 0.997227, 0.997329, 0.997422, 0.997498,
        0.997562, 0.997621, 0.997666, 0.997686, 0.997689, 0.997680, 0.997652, 0.997619,
        0.997567, 0.997481, 0.997375, 0.997263, 0.997131, 0.996977, 0.996804, 0.996613,
        0.996404, 0.996179, 0.995938, 0.995688, 0.995425, 0.995151, 0.994867, 0.994583,
        0.994297, 0.994016, 0.993742, 0.993472, 0.993214, 0.992963, 0.992730, 0.992547,
        0.992383, 0.992241, 0.992116, 0.992014, 0.991925, 0.991846, 0.991775, 0.991748,
        0.991735, 0.991721, 0.991715, 0.991730, 0.991750, 0.991804, 0.991862, 0.991940,
        0.992025, 0.992154, 0.992291, 0.992401, 0.992510, 0.992639, 0.992765, 0.992943,
        0.993123, 0.993292, 0.993458, 0.993788, 0.994119, 0.994234, 0.994356, 0.994486,
        0.994613, 0.994720, 0.994826, 0.994902, 0.994985, 0.995068, 0.995151, 0.995188,
        0.995225, 0.995261, 0.995305, 0.995330, 0.995356, 0.995358, 0.995361, 0.995367,
        0.995373, 0.995369, 0.995361, 0.995355, 0.995351, 0.995332, 0.995315, 0.995289,
        0.995271, 0.995247, 0.995220, 0.995186, 0.995152, 0.995123, 0.995089, 0.995057,
        0.995032, 0.994941, 0.994854, 0.994772, 0.994691, 0.994630, 0.994569, 0.994504,
        0.994437, 0.994360, 0.994274, 0.994170, 0.994070, 0.993960, 0.993855, 0.993754,
        0.993652, 0.993550, 0.993448, 0.993361, 0.993277, 0.993212, 0.993146, 0.993077,
        0.993009, 0.992936, 0.992864, 0.992804, 0.992750, 0.992736, 0.992719, 0.992690,
        0.992680, 0.992661, 0.992683,
    ]
)  # fmt: skip

#: ``(s0, s1)`` of TT's Equation (31) scaling ``s(ψ) = s0 + s1 sin ψ``, refit
#: with :data:`FL_AVE_PB24_NADIR` (TT: 1.0405, 0.07762).
FL_S_PB24_NADIR = (1.0298524509404359, 0.024252437303698925)


def psi_KLu_pb24_nadir(
    psi_deg: Float[Array, "..."], coeffs=None
) -> Float[Array, "..."]:
    """``Ψ_KLu(ψ) = 1 + F(ψ)`` of the nadir-only PB24 fit (M8 task 9).

    A measured K_Lu/K∞: fitted to PB24's nadir ``KLu`` at ψ = 180° − θs_w.
    Held flat below :data:`PSI_KLU_PB24_NADIR_PSI_MIN`, where no nadir
    geometry reaches and the polynomial would extrapolate.

    Parameters
    ----------
    psi_deg : Array
        In-water scattering angle (degrees).
    coeffs : sequence of float, optional
        Polynomial in ``u = 1 + cos ψ``, highest order first; default
        :data:`PSI_KLU_PB24_NADIR_COEFFS`.

    Returns
    -------
    Array
        ``Ψ_KLu``, dimensionless.
    """
    c = PSI_KLU_PB24_NADIR_COEFFS if coeffs is None else coeffs
    u = 1.0 + jnp.cos(jnp.deg2rad(jnp.asarray(psi_deg)))
    u_max = 1.0 + np.cos(np.deg2rad(PSI_KLU_PB24_NADIR_PSI_MIN))
    u = jnp.minimum(u, u_max)
    return 1.0 + jnp.polyval(jnp.asarray(c, dtype=u.dtype), u)


def f_L_pb24_nadir(
    psi_deg,
    wave: Float[Array, " wave"] | None = None,
    *,
    fl_ave=None,
    fl_wave=None,
    s=None,
):
    """``f_L(ψ, λ) = f_L,ave(λ) [s0 + s1 sin ψ]`` of the nadir-only PB24 fit.

    TT's Equation (31) form with both factors refit (M8 task 9).

    Parameters
    ----------
    psi_deg : Array
        In-water scattering angle (degrees).
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid. Linear interpolation
        on :data:`FL_AVE_PB24_WAVE`, flat outside 350–800 nm.
    fl_ave, fl_wave, s : optional
        Override the shipped spectral shape (on ``fl_wave``) and ``(s0, s1)``;
        the fit script evaluates its candidates this way.

    Returns
    -------
    Array
        ``f_L``, dimensionless, broadcasting ψ against λ.
    """
    s0, s1 = FL_S_PB24_NADIR if s is None else s
    return f_L_pb24(
        psi_deg,
        wave,
        fl_ave=FL_AVE_PB24_NADIR if fl_ave is None else fl_ave,
        fl_wave=fl_wave,
        s=(s0, s1, 0.0),
    )


def _warn_if_outside_nadir_domain(theta_v, a, bb) -> None:
    """:class:`~robust.rt.conventions.DomainWarning` when a *concrete*
    ``theta_v`` exceeds :data:`NADIR_TOL_DEG` or a concrete ``bb/a`` exceeds
    :data:`NADIR_BB_OVER_A_MAX` (traced values cannot be inspected and pass
    silently)."""
    import jax

    if not isinstance(theta_v, jax.core.Tracer):
        worst = float(np.max(np.abs(np.asarray(theta_v))))
        if worst > NADIR_TOL_DEG:
            warnings.warn(
                f'rrs_ZTT(variant="pb24_nadir") is calibrated at nadir only; '
                f"geometry.theta_v reaches {worst:g}° (tolerance "
                f"{NADIR_TOL_DEG:g}°). Off-nadir it is uncalibrated.",
                conventions.DomainWarning,
                stacklevel=3,
            )
    if not isinstance(a, jax.core.Tracer) and not isinstance(bb, jax.core.Tracer):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.asarray(bb) / np.asarray(a)
        worst = float(np.nanmax(ratio)) if ratio.size else 0.0
        if worst > NADIR_BB_OVER_A_MAX:
            warnings.warn(
                f'rrs_ZTT(variant="pb24_nadir") is a clear-to-moderate-water '
                f"model; bb/a reaches {worst:.3g} (limit "
                f"{NADIR_BB_OVER_A_MAX:g}). Above it the error grows to tens of "
                "per cent and beyond.",
                conventions.DomainWarning,
                stacklevel=3,
            )


def _choose(name, value, allowed):
    """Validate a string option, raising with the allowed set."""
    if value not in allowed:
        raise ValueError(f"rrs_ZTT: {name} must be one of {allowed}; got {value!r}")
    return value


# ------------------------------------------------------------------------ f_L --
def f_L(psi_deg, wave: Float[Array, " wave"] | None = None):
    """Equation (31) — ``f_L(ψ, λ) = f_L,ave(λ) [0.07762 sin ψ + 1.0405]``.

    The upwelling-radiance shape factor. Zaneveld suggested a constant 1.05
    (§2.5); the paper instead fits a spectral shape (:data:`FL_AVE`, Table A3)
    scaled by scattering angle, and uses Equation (31) for every run after §4.1.

    Parameters
    ----------
    psi_deg : Array
        In-water scattering angle (degrees).
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid. Values outside
        350-800 nm clamp to the ends of Table A3.

    Returns
    -------
    Array
        ``f_L``, dimensionless, broadcasting ψ against λ.
    """
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)
    shape = jnp.interp(wave, jnp.asarray(FL_AVE_WAVE), jnp.asarray(FL_AVE))
    scale = FL_SIN_COEFF * jnp.sin(jnp.deg2rad(jnp.asarray(psi_deg))) + FL_OFFSET
    return shape * scale


# ------------------------------------------------------------------ the model --
def rrs_ZTT(
    iops,
    phase_params,
    geometry,
    wave: Float[Array, " wave"] | None = None,
    *,
    P_bb=None,
    mu_inf=None,
    mu_inf_coeffs=None,
    visibility_km: float = DEFAULT_VISIBILITY_KM,
    variant: str = DEFAULT_VARIANT,
    mu_inf_model: str | None = None,
    mu_d_model: str | None = None,
    angular_model: str | None = None,
    floor: bool | None = None,
    theta_v_in_water: bool = False,
) -> Float[Array, "*batch wave"]:
    """Equation (12) — subsurface ``rrs`` from the ZTT model.

    ``rrs = (1/µd)(β(ψ)/bb) / [ (a/bb)(1 - cos θv Ψ_KLu/µ∞) + f_L(1 - b̃b⁻¹) + b̃b⁻¹ ]``

    Equation (18) adds an ``rrs,Raman`` term; this package is elastic by scope, so
    that term is absent rather than approximated.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        ``a``, ``bb_w``, ``bb_p``; the water/particle split is required.
    phase_params : robust.rt.types.PhaseParams
        ``B_p`` is the paper's ``b̃bp`` (Equation (11)).
    geometry : robust.rt.types.Geometry
        Degrees, this package's convention (``theta_v`` above-water since M8
        task 4); converted by :func:`geometry_to_paper_angles`. If
        ``geometry.diffuse_fraction`` is set it replaces TT's ``H(θs', V)`` in
        µd (M8 task 4, Q10a); it must broadcast against the ``(..., n_wave)``
        spectra -- ``(n_sample, n_wave)`` from the PB24 loader, ``(n_wave,)``
        per sample under ``vmap``. ``None`` (default) is the variant's own H:
        the PB24 table for the two PB24 variants, TT's (bit for bit) otherwise.
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid.
    P_bb : Array, optional
        ``Pbb(ψ)`` in sr⁻¹ — a required model input (paper §2.9). Defaults to
        :func:`P_bb_sullivan`, the constant shape the paper reports its best
        performance with. Pass a constant only for sensitivity tests: holding it
        fixed in ψ inverts the modelled solar-zenith trend.
    mu_inf : Array, optional
        ``µ∞`` supplied directly — for sensitivity tests. Overrides everything.
    mu_inf_coeffs : sequence of float, optional
        The sixteen Equation (8) coefficients, if you obtain them; this is the
        path back to the model exactly as published in 2018 ("ZTT 2018").
    visibility_km : float, optional
        Horizontal visibility V (km).
    variant : {"pb24_nadir", "l23", "tt2017", "pb24"}, optional
        Which backbone (:data:`VARIANTS`). ``"pb24_nadir"``, **the default
        since M8 task 9c**, is "ZTT (PB24_nadir)": PB24's diffuse fraction
        (:func:`diffuse_fraction_pb24`, when ``geometry`` carries none), µd
        and µ∞ refit to PB24 (:func:`mu_d_pb24`, :func:`mu_infinity_pb24`),
        the FF phase function wherever ``phase_params`` carries ``ff_B``, TT's
        own Ψ_KLu and f_L (:func:`psi_KLu`, :func:`f_L`) and the denominator
        floor. **Nadir viewing and bb/a ≤** :data:`NADIR_BB_OVER_A_MAX`: a
        concrete ``geometry.theta_v`` beyond :data:`NADIR_TOL_DEG`, or a
        concrete bb/a beyond the limit, raises a
        :class:`~robust.rt.conventions.DomainWarning` (the value is still
        returned). ``"l23"``, the default from M7 until then, is
        "ZTT (L23 µ∞)": µ∞ and µd refit to L23 (:func:`mu_infinity_l23`,
        :func:`mu_d_l23`), and L23's two-component FF phase function wherever
        ``phase_params`` carries ``ff_B`` (Sullivan otherwise, silently).
        ``"tt2017"`` is "ZTT (TT2017 µ∞)", the M0–M6 model, bit for bit: the
        TT2017 µ∞, Table A2's µd, and Sullivan's Pbb. ``"pb24"`` (**M8 tasks
        8a–8b**) is "ZTT (PB24)": µ∞ and µd refit to PB24
        (:func:`mu_infinity_pb24`, :func:`mu_d_pb24`), with PB24's own diffuse
        fraction (:func:`diffuse_fraction_pb24`) when ``geometry`` carries
        none, the FF phase function as ``"l23"``, Ψ_KLu and f_L refit jointly
        through PB24's ``rrs`` (:func:`psi_KLu_pb24`, :func:`f_L_pb24`) and
        the denominator floor (:func:`denominator_floor_pb24`). **Off-nadir,
        that Ψ_KLu is a calibration through rrs, physical only at nadir**
        (PB24 measures ``KLu`` there alone).
    mu_inf_model : {"l23", "l23_closure", "tt2017", "pb24"}, optional
        Override the variant's µ∞ alone (:data:`MU_INF_MODELS`).
        ``"l23_closure"`` (M7 task 5) is the closure fit -- a ceiling, not
        physics (:func:`mu_infinity_l23_closure`). Precedence:
        ``mu_inf`` > ``mu_inf_coeffs`` > ``mu_inf_model`` > ``variant``.
    mu_d_model : {"l23", "tt2017", "pb24"}, optional
        Override the variant's µd alone (:data:`MU_D_MODELS`). The Pbb choice
        has its own override, ``P_bb``.
    angular_model : {"tt", "pb24", "pb24_klu"}, optional
        Override the variant's Ψ_KLu and f_L pair alone
        (:data:`ANGULAR_MODELS`; ``"pb24"`` for the ``"pb24"`` variant,
        ``"tt"`` otherwise). ``"pb24_klu"`` is the measured-K_Lu nadir fit of
        M8 task 9b, a record no variant uses.
    floor : bool, optional
        Apply :func:`denominator_floor_pb24` to Equation (12)'s denominator.
        Default: the variant's (``True`` for ``"pb24"`` and ``"pb24_nadir"``,
        ``False`` otherwise,
        which leaves ``"l23"`` and ``"tt2017"`` bit for bit).
    theta_v_in_water : bool, optional
        Read ``geometry.theta_v`` as in-water (no refraction); only for PB24's
        ``"in_water_only"`` batches. See :func:`geometry_to_paper_angles`.

    Returns
    -------
    Array
        ``rrs`` in sr⁻¹, shape ``(..., n_wave)``.

    Raises
    ------
    ValueError
        If both ``mu_inf`` and ``mu_inf_coeffs`` are given, or an option is not
        one of its allowed values.
    """
    if mu_inf is not None and mu_inf_coeffs is not None:
        raise ValueError(
            "rrs_ZTT: supply at most one of mu_inf (a value) or mu_inf_coeffs "
            "(Equation (8) coefficients m1..m16)."
        )
    _choose("variant", variant, VARIANTS)
    is_pb24 = variant in ("pb24", "pb24_nadir")
    iop_side = "pb24" if is_pb24 else variant
    mu_inf_model = _choose(
        "mu_inf_model",
        iop_side if mu_inf_model is None else mu_inf_model,
        MU_INF_MODELS,
    )
    mu_d_model = _choose(
        "mu_d_model", iop_side if mu_d_model is None else mu_d_model, MU_D_MODELS
    )
    variant_angular = "pb24" if variant == "pb24" else "tt"
    if variant == "pb24_nadir":
        _warn_if_outside_nadir_domain(geometry.theta_v, iops.a, iops.bb)
    angular_model = _choose(
        "angular_model",
        variant_angular if angular_model is None else angular_model,
        ANGULAR_MODELS,
    )
    if floor is None:
        floor = is_pb24

    _, theta_s_air, theta_v_paper, psi = geometry_to_paper_angles(
        geometry, theta_v_in_water=theta_v_in_water
    )
    H = geometry.diffuse_fraction

    # Per-sample angles gain a trailing axis so they broadcast against (..., wave).
    def spectral(x):
        """Give a per-sample angle a trailing length-1 axis so it broadcasts
        against the ``(..., wave)`` spectral arrays. Scalars pass through
        unchanged -- they already broadcast."""
        x = jnp.asarray(x)
        return x[..., None] if jnp.ndim(x) else x

    psi_b = spectral(psi)
    cos_theta_v = spectral(jnp.cos(jnp.deg2rad(theta_v_paper)))
    theta_s_air_b = spectral(theta_s_air)

    bb = iops.bb
    bb_over_a = bb / iops.a
    eta_bb = iops.bb_w / bb

    if mu_inf is not None:
        mu_inf_value = jnp.asarray(mu_inf)
    elif mu_inf_coeffs is not None:
        mu_inf_value = mu_infinity(bb_over_a, eta_bb, mu_inf_coeffs)
    elif mu_inf_model == "l23":
        mu_inf_value = mu_infinity_l23(bb_over_a, eta_bb)
    elif mu_inf_model == "l23_closure":
        mu_inf_value = mu_infinity_l23_closure(bb_over_a, eta_bb)
    elif mu_inf_model == "pb24":
        mu_inf_value = mu_infinity_pb24(bb_over_a, eta_bb)
    else:
        mu_inf_value = mu_infinity_tt2017(bb_over_a, eta_bb)

    # The backward VSF: an explicit override wins, else the phase container. That
    # falls back to Sullivan's fixed shape unless the M5 fields are set, or (the
    # L23 variant only) the FF mixture of M7 task 2 is present.
    P_bb_value = (
        P_bb_from_phase(phase_params, psi_b, use_ff=variant != "tt2017")
        if P_bb is None
        else P_bb
    )
    numerator = backward_phase_over_bb(iops, P_bb_value, psi_b)
    b_tilde = bb_tilde(iops, phase_params.B_p)

    if angular_model == "pb24":
        psi_term, fl_term = psi_KLu_pb24(psi_b), f_L_pb24(psi_b, wave)
    elif angular_model == "pb24_klu":
        psi_term, fl_term = psi_KLu_pb24_nadir(psi_b), f_L_pb24_nadir(psi_b, wave)
    else:
        psi_term, fl_term = psi_KLu(psi_b), f_L(psi_b, wave)
    denominator = (
        (iops.a / bb) * (1.0 - cos_theta_v * psi_term / mu_inf_value)
        + fl_term * (1.0 - 1.0 / b_tilde)
        + 1.0 / b_tilde
    )
    if floor:
        denominator = denominator_floor_pb24(denominator)

    if mu_d_model == "l23":
        mu_d_value = mu_d_l23(theta_s_air_b, bb_over_a, eta_bb, visibility_km, H=H)
    elif mu_d_model == "pb24":
        mu_d_value = mu_d_pb24(theta_s_air_b, bb_over_a, eta_bb, H=H, wave=wave)
    else:
        mu_d_value = mu_d(theta_s_air_b, bb_over_a, eta_bb, visibility_km, H=H)
    return numerator / (mu_d_value * denominator)


def mu_inf_closure(
    iops,
    phase_params,
    geometry,
    rrs,
    wave: Float[Array, " wave"] | None = None,
    *,
    visibility_km: float = DEFAULT_VISIBILITY_KM,
    theta_v_in_water: bool = False,
):
    """The µ∞ that makes the L23 variant reproduce ``rrs`` exactly (M7 task 5).

    Equation (12) is ``rrs = N / (µd D)`` with
    ``D = (a/bb)(1 − cos θv Ψ_KLu/µ∞) + R`` and ``R = f_L(1 − 1/b̃b) + 1/b̃b``.
    Given a reference ``rrs`` it is linear in ``1/µ∞``:

        µ∞ = −cos θv Ψ_KLu / [ (N / (µd rrs) − R)(bb/a) − 1 ]

    Every other term is the L23 variant's: the refit µd (:func:`mu_d_l23`) and
    L23's FF phase function where ``phase_params`` carries it. So the closure µ∞
    absorbs as little else as possible. It still absorbs **everything** else
    -- Ψ_KLu, f_L, the Pbb shape, the surface -- which is why it is *a ceiling,
    not physics*. The values can be ≤ 0 or > 1 where those errors are large.
    They are returned as computed, and the caller decides.

    Parameters
    ----------
    iops, phase_params, geometry, wave
        As :func:`rrs_ZTT`.
    rrs : Array
        Reference subsurface reflectance (sr^-1), ``(..., n_wave)``.
    visibility_km : float, optional
    theta_v_in_water : bool, optional
        As :func:`rrs_ZTT`. ``geometry.diffuse_fraction`` is used as there.

    Returns
    -------
    Array
        µ∞ per sample and wavelength, unclipped.
    """
    _, theta_s_air, theta_v_paper, psi = geometry_to_paper_angles(
        geometry, theta_v_in_water=theta_v_in_water
    )

    def spectral(x):
        """A per-sample angle gains a trailing axis to meet ``(..., wave)``."""
        x = jnp.asarray(x)
        return x[..., None] if jnp.ndim(x) else x

    psi_b = spectral(psi)
    cos_theta_v = spectral(jnp.cos(jnp.deg2rad(theta_v_paper)))
    theta_s_air_b = spectral(theta_s_air)
    bb = iops.bb
    bb_over_a = bb / iops.a
    eta_bb = iops.bb_w / bb

    numerator = backward_phase_over_bb(
        iops, P_bb_from_phase(phase_params, psi_b, use_ff=True), psi_b
    )
    b_tilde = bb_tilde(iops, phase_params.B_p)
    rest = f_L(psi_b, wave) * (1.0 - 1.0 / b_tilde) + 1.0 / b_tilde
    mu_d_value = mu_d_l23(
        theta_s_air_b, bb_over_a, eta_bb, visibility_km, H=geometry.diffuse_fraction
    )
    needed = numerator / (mu_d_value * jnp.asarray(rrs))
    return -cos_theta_v * psi_KLu(psi_b) / ((needed - rest) * bb_over_a - 1.0)


def Rrs_ZTT(
    iops,
    phase_params,
    geometry,
    wave: Float[Array, " wave"] | None = None,
    **kwargs,
) -> Float[Array, "*batch wave"]:
    """Above-water ``Rrs`` from the ZTT model — :func:`rrs_ZTT` plus the interface.

    Parameters
    ----------
    iops, phase_params, geometry, wave
        As :func:`rrs_ZTT`.
    **kwargs
        Forwarded to :func:`rrs_ZTT` (``P_bb``, ``mu_inf`` / ``mu_inf_coeffs``,
        ``visibility_km``, ``variant``, ``mu_inf_model``, ``mu_d_model``,
        ``theta_v_in_water``). ``geometry.diffuse_fraction`` reaches µd through
        :func:`rrs_ZTT`. The surface step is the nadir constants
        (:func:`robust.rt.conventions.rrs_to_Rrs` with no transfer), unchanged
        by M8 task 4: the Fresnel transfer is opt-in
        (:class:`robust.rt.conventions.FresnelTransfer`).

    Returns
    -------
    Array
        ``Rrs`` in sr⁻¹, shape ``(..., n_wave)``.
    """
    return conventions.rrs_to_Rrs(rrs_ZTT(iops, phase_params, geometry, wave, **kwargs))
