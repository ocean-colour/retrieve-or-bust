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
  opposite and more usual convention, ``theta_v = 0`` for nadir, and its
  ``theta_s`` is the *above-water* solar zenith as L23 reports it. Both conversions
  happen in :func:`geometry_to_paper_angles`, in one place, because getting either
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
``µ∞(bb/a, η_bb)`` in Twardowski & Tonizzo (2017), *Optics Express* **25**(15),
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

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions

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
    "mu_infinity",
    "mu_infinity_tt2017",
    "MU_INF_TT2017_TABLE1",
    "f_L",
    # The model
    "rrs_ZTT",
    "Rrs_ZTT",
]

#: Refractive index of seawater used for Snell refraction (paper §2.1).
REFRACTIVE_INDEX = 1.34

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

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith angle (degrees), as L23 reports it.

    Returns
    -------
    Array
        In-water solar zenith angle (degrees).
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


def geometry_to_paper_angles(geometry):
    """Translate a :class:`robust.rt.types.Geometry` into the paper's angles.

    Two conversions, kept together because either one reversed yields a
    wrong-but-plausible BRDF:

    1. ``theta_s`` is the **above-water** solar zenith (L23's convention); the
       paper's ``θs`` is **in-water**, so Snell refracts it.
    2. ``theta_v = 0`` is nadir here; the paper's nadir is ``θv = 180°``, so
       ``θv_paper = 180° - theta_v``.

    Parameters
    ----------
    geometry : robust.rt.types.Geometry
        Angles in degrees, this package's convention.

    Returns
    -------
    tuple of Array
        ``(theta_s_water, theta_s_air, theta_v_paper, psi)``, all degrees.
    """
    theta_s_air = jnp.asarray(geometry.theta_s)
    theta_s_water = in_water_zenith(theta_s_air)
    theta_v_paper = 180.0 - jnp.asarray(geometry.theta_v)
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
#: with its standard deviation. *Appl. Opt.* **48**(35), 6811. Angles in degrees.
P_BB_ST_ANGLES = np.array([90.0, 100, 110, 120, 130, 140, 150, 160, 170])
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


def Md_plus(theta_s_air_deg, visibility_km: float = DEFAULT_VISIBILITY_KM):
    """Equation (15) — the atmospheric/geometric part of ``µd``.

    ``M+_d = [ (1-H)/µw + H/0.859 ]⁻¹ P3[cos θs']`` with ``µw = cos θs'``: the
    Morel & Prieur (1977) cardioidal-skylight form of Equation (13), generalized
    to a variable diffuse fraction.

    Parameters
    ----------
    theta_s_air_deg : Array
        Above-water solar zenith (degrees).
    visibility_km : float, optional
        Horizontal visibility V (km).

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
    H = diffuse_fraction(theta, visibility_km)
    return P3_cos(theta) / ((1.0 - H) / mu_w + H / 0.859)


def Md_star(bb_over_a, eta_bb, coeffs=MD_STAR_COEFFS):
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

    Returns
    -------
    Array
        ``M*_d``, dimensionless.
    """
    m = coeffs
    X = jnp.log10(jnp.asarray(bb_over_a))
    L = jnp.log10(jnp.asarray(eta_bb))
    return (
        (m[0] * L + m[1]) * X**3
        + (m[2] * L + m[3]) * X**2
        + (m[4] * L + m[5]) * X
        + m[6] * L
        + m[7]
    )


def mu_d(
    theta_s_air_deg,
    bb_over_a,
    eta_bb,
    visibility_km: float = DEFAULT_VISIBILITY_KM,
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

    Returns
    -------
    Array
        ``µd``, dimensionless.
    """
    return Md_plus(theta_s_air_deg, visibility_km) * Md_star(bb_over_a, eta_bb)


# --------------------------------------------------------------------- mu_inf --
#: Twardowski & Tonizzo (2017) Table 1 — second-order fits of ``µ∞`` against
#: ``log10(bb/a)``, one per ``η_bb``: ``µ∞ = p0 + p1 L + p2 L²``.
#: *Optics Express* **25**(15), 18122, the antecedent (reference [40]) that the
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
        ``bb / a``. Fitted over :data:`MU_INF_TT2017_BB_OVER_A_RANGE`; larger
        values extrapolate (L23 reaches ~0.31).
    eta_bb : Array
        ``η_bb = bbw / (bbp + bbw)``. Fitted over
        :data:`MU_INF_TT2017_ETA_RANGE`; outside it the coefficients clamp.

    Returns
    -------
    Array
        ``µ∞``, dimensionless.
    """
    etas = np.array(sorted(MU_INF_TT2017_TABLE1))
    log_eta_nodes = jnp.asarray(np.log10(etas))
    table = np.array([MU_INF_TT2017_TABLE1[e] for e in etas])

    log_eta = jnp.log10(jnp.asarray(eta_bb))
    p0, p1, p2 = (
        jnp.interp(log_eta, log_eta_nodes, jnp.asarray(table[:, k])) for k in range(3)
    )

    L = jnp.log10(jnp.asarray(bb_over_a))
    return p0 + p1 * L + p2 * L**2


def mu_infinity(bb_over_a, eta_bb, coeffs=None):
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
    coeffs : sequence of 16 float, optional
        ``m1..m16``. Supply them if you obtain them; there is no default.

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
    X = jnp.log10(jnp.asarray(bb_over_a))
    L = jnp.log10(jnp.asarray(eta_bb))

    def cubic_in_L(c0, c1, c2, c3):
        return c0 * L**3 + c1 * L**2 + c2 * L + c3

    return (
        cubic_in_L(m[0], m[1], m[2], m[3]) * X**3
        + cubic_in_L(m[4], m[5], m[6], m[7]) * X**2
        + cubic_in_L(m[8], m[9], m[10], m[11]) * X
        + cubic_in_L(m[12], m[13], m[14], m[15])
    )


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
        Degrees, this package's convention; converted by
        :func:`geometry_to_paper_angles`.
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid.
    P_bb : Array, optional
        ``Pbb(ψ)`` in sr⁻¹ — a required model input (paper §2.9). Defaults to
        :func:`P_bb_sullivan`, the constant shape the paper reports its best
        performance with. Pass a constant only for sensitivity tests: holding it
        fixed in ψ inverts the modelled solar-zenith trend.
    mu_inf : Array, optional
        ``µ∞`` supplied directly — for sensitivity tests. Overrides everything.
    mu_inf_coeffs : sequence of 16 float, optional
        Equation (8) coefficients, if you obtain them; this is the path back to
        the model exactly as published in 2018.

        With neither given, ``µ∞`` falls back to :func:`mu_infinity_tt2017`, the
        same authors' 2017 parameterization, because the 2018 paper omits its own
        Equation (8) coefficients. **Report such results as "ZTT with the TT2017
        µ∞", not as the 2018 model.**
    visibility_km : float, optional
        Horizontal visibility V (km).

    Returns
    -------
    Array
        ``rrs`` in sr⁻¹, shape ``(..., n_wave)``.

    Raises
    ------
    ValueError
        If both ``mu_inf`` and ``mu_inf_coeffs`` are given.
    """
    if mu_inf is not None and mu_inf_coeffs is not None:
        raise ValueError(
            "rrs_ZTT: supply at most one of mu_inf (a value) or mu_inf_coeffs "
            "(Equation (8) coefficients m1..m16)."
        )

    _, theta_s_air, theta_v_paper, psi = geometry_to_paper_angles(geometry)

    # Per-sample angles gain a trailing axis so they broadcast against (..., wave).
    def spectral(x):
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
    else:
        mu_inf_value = mu_infinity_tt2017(bb_over_a, eta_bb)

    P_bb_value = P_bb_sullivan(psi_b) if P_bb is None else P_bb
    numerator = backward_phase_over_bb(iops, P_bb_value, psi_b)
    b_tilde = bb_tilde(iops, phase_params.B_p)

    denominator = (
        (iops.a / bb) * (1.0 - cos_theta_v * psi_KLu(psi_b) / mu_inf_value)
        + f_L(psi_b, wave) * (1.0 - 1.0 / b_tilde)
        + 1.0 / b_tilde
    )

    return numerator / (
        mu_d(theta_s_air_b, bb_over_a, eta_bb, visibility_km) * denominator
    )


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
        ``visibility_km``).

    Returns
    -------
    Array
        ``Rrs`` in sr⁻¹, shape ``(..., n_wave)``.
    """
    return conventions.rrs_to_Rrs(rrs_ZTT(iops, phase_params, geometry, wave, **kwargs))
