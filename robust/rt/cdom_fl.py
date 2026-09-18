"""
CDOM (yellow-matter) fluorescence — the analytic Hawes kernel (CDOM design M5).

The third inelastic term (``design/rt_cdom_fluorescence_model.md`` §2): an
additive emission term built on the Hawes et al. (1992) spectral fluorescence
quantum-efficiency parameterization, in the same Sathyendranath & Platt (1998)
two-flow / isotropic-emission / **L_u = E_u/π** / A·rrs/(1−B·rrs) machinery as
the validated chlorophyll-fluorescence kernel
(:func:`robust.rt.inelastic.fluorescence_kernel`). This module supplies the raw
kernel ``K_cdom(λ)`` only — the ``CDOMFl.scale`` amplitude and the (untrained)
``(1 + δ_C)`` head are applied where ``hybrid.forward()`` composes the term
(M5 task 5), never here.

**Functional form — peer-reviewed, verified directly.** Zhai, Hu, Lee et al.
(2017), *Optics Express* 25(8), A213–A235, Eqs. (5)–(8) (extracted verbatim
from the published PDF, 2026-08-29)::

    b_F(z, λ, λ_e) = a_F(z, λ_e) · f_F(z, λ, λ_e)                       [Eq 5]
    f_F(z, λ, λ_e) = η_F(z, λ, λ_e) · λ_e / λ                            [Eq 6]
    η_Y(λ, λ_e)    = g_Y(λ_e) · A0(λ_e)
                     · exp{ −[ (1/λ − A1/λ_e − B1) / (0.6·(A2/λ_e + B2)) ]² }
                                                                         [Eq 7]
    g_Y(λ_e)       = 1 for 310 ≤ λ_e ≤ 490 nm, else 0                    [Eq 8]

Read Eq. (7) literally: the Gaussian argument is in **reciprocal wavelength
(wavenumber)** — ``1/λ``, ``A1/λ_e``, ``A2/λ_e`` all carry explicit division.
For each excitation wavelength the emission distribution is approximately
Gaussian in wavenumber, with center, width, and amplitude depending on λ_e —
genuinely **non-separable** in (λ, λ_e), unlike the Chl-fl line shape (see
:func:`cdom_kernel` for the structural consequence).

**Constants — sourced from the Ocean Optics Web Book, accepted by JXP.**
A0–B2 below are the Hawes (1992) **Station FA7** fulvic-acid fit (Gulf of
Mexico, West Florida Shelf) — HydroLight's own default CDOM-fluorescence
choice (CFQ4/design §2), so the §7 truth runs and this kernel share constants
by construction. (Zhai et al. themselves used a 9:1 FA7:HA6 fulvic/humic
mixture — deliberately *not* followed here.) The numeric values were sourced
from Mobley's Ocean Optics Web Book
(``oceanopticsbook.info/view/scattering/level-2/cdom-fluorescence``, retrieved
2026-08-29 via an AI-mediated page fetch) and **accepted as-sourced by JXP
without independent primary-source verification** (Q&A CQ2 in the M5 prompt
doc, 2026-08-30: "I am good with the Ocean Optics Web Book"). That is a
provenance statement, not a peer-review claim — no cross-check against Hawes
(1992), Proc. SPIE 1750, or Light and Water §5.15 was performed, but it is no
longer an open action item. The *functional form* above and the 350 nm clamp
precedent are peer-reviewed-verified.

**The 350 nm excitation clamp.** ``g_Y`` nominally admits excitation down to
310 nm, but the L23/IOP grid starts at 350 nm, so the excitation quadrature is
hard-clamped to **350–490 nm** (design §2; the truncated-fraction diagnostic is
M5 task 4). Independent corroboration: Zhai et al. (2017) impose the same
λ_e ≥ 350 nm floor in their own model, citing "the strong ozone absorption and
decreased solar spectral irradiance in the UV" — the same physical reasoning
as our CFQ4 decision.

*Bibliographic note:* Zhai et al.'s reference list prints the Hawes (1992)
middle author as "C. K. Carder"; the well-known ocean-optics scientist of that
surname is Kendall L. Carder. The discrepancy is recorded here rather than
silently resolved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions, ed
from .inelastic import MU_D, MU_F

__all__ = [  # noqa: RUF022  - grouped by role
    # Hawes FA7 constants (Ocean Optics Web Book, JXP-accepted -- see module docstring)
    "HAWES_A0_WAVE",
    "HAWES_A0",
    "HAWES_A1",
    "HAWES_B1",
    "HAWES_A2",
    "HAWES_B2",
    "GY_EX_MIN",
    "GY_EX_MAX",
    # Excitation quadrature
    "CDOM_EX_MIN",
    "CDOM_EX_MAX",
    "CDOM_EX_STEP",
    "cdom_excitation_grid",
    # The Hawes function and the kernel
    "eta_hawes",
    "cdom_kernel",
    # The clamp's documented cost (task 4)
    "truncated_excitation_fraction",
]

# --------------------------------------------------------- Hawes FA7 constants
#: Excitation wavelengths (nm) at which Hawes tabulated the FA7 amplitude A0.
#:
#: PROVENANCE: this table and the four scalar constants below are the Station
#: FA7 values as reported on Mobley's Ocean Optics Web Book (retrieved
#: 2026-08-29), **accepted as-sourced by JXP without independent
#: primary-source verification** (M5 prompt doc, Q&A CQ2, 2026-08-30) — no
#: cross-check against Hawes (1992) / Light and Water primary text was
#: performed, and none is pending. The quoted fit quality is r² = 0.987.
HAWES_A0_WAVE = np.array(
    [310.0, 330.0, 350.0, 370.0, 390.0, 410.0, 430.0, 450.0, 470.0, 490.0]
)

#: FA7 amplitude ``A0(λ_e)`` at :data:`HAWES_A0_WAVE`, nm⁻¹ (table values are
#: ×10⁻⁵ nm⁻¹ on the source page). Same provenance flag as above. The kernel
#: interpolates **linearly** between these ten sparse nodes
#: (:func:`eta_hawes`) — a real, if secondary, source of kernel uncertainty
#: beyond the constants themselves.
HAWES_A0 = 1e-5 * np.array(
    [5.81, 6.34, 8.00, 9.89, 9.39, 10.48, 12.59, 13.48, 13.61, 9.24]
)

#: float: Slope term of the FA7 emission centre -- the peak emission
#: wavenumber is ``A1/λ_e + B1`` (nm⁻¹), with A1 dimensionless. Provenance as
#: above.
HAWES_A1 = 0.470
#: float: Offset term of that same emission centre, nm⁻¹.
HAWES_B1 = 8.077e-4

#: float: Slope term of the FA7 emission width -- the Gaussian denominator is
#: ``0.6 · (A2/λ_e + B2)`` (nm⁻¹), with A2 dimensionless. Provenance as above.
HAWES_A2 = 0.407
#: float: Offset term of that same emission width, nm⁻¹ (negative).
HAWES_B2 = -4.57e-4

#: Support of the excitation gate ``g_Y`` (Zhai et al. Eq. 8), nm.
GY_EX_MIN = 310.0
#: float: Upper edge of that ``g_Y`` support, nm.
GY_EX_MAX = 490.0

# ------------------------------------------------------- excitation quadrature
#: CDOM-fluorescence excitation band (nm) and quadrature step. The lower edge
#: is the **hard 350 nm clamp** (design §2): ``g_Y`` admits 310 nm, but the
#: L23/IOP/Ed grids start at 350 nm, so the kernel provably never reads below
#: it (the excitation grid *starts* there — no clamping arithmetic needed).
#: The upper edge is ``g_Y``'s own 490 nm cutoff. 5 nm matches the canonical
#: grid spacing (350–750 nm at 5 nm), so all 29 nodes land exactly on
#: canonical grid points and the excitation IOPs interpolate losslessly —
#: the same convention as the Chl-fl grid (``FL_EX_STEP``).
CDOM_EX_MIN = 350.0
#: float: Red edge of that CDOM-fluorescence excitation band (nm) --
#: ``g_Y``'s own cutoff.
CDOM_EX_MAX = 490.0
#: float: Quadrature step across it (nm), matching the canonical grid spacing.
CDOM_EX_STEP = 5.0


def cdom_excitation_grid(step: float = CDOM_EX_STEP, dtype=None) -> Float[Array, " ex"]:
    """The CDOM excitation quadrature nodes: 350–490 nm; 29 points at 5 nm.

    A fixed grid, for the same reasons as
    :func:`robust.rt.inelastic.fl_excitation_grid`: the quadrature is
    identical whatever emission grid is requested, shapes stay static under
    ``jit``, and the grid *starting* at :data:`CDOM_EX_MIN` is what makes the
    350 nm clamp provable rather than arithmetic.

    Parameters
    ----------
    step : float, optional
        Node spacing (nm); default :data:`CDOM_EX_STEP`. Non-default values
        exist for the quadrature-convergence gate (design §5.2), which
        compares the native grid against a 2× refinement.
    dtype : optional
        Element dtype; defaults to JAX's current default float.

    Returns
    -------
    Array
        Excitation wavelengths (nm), shape ``(29,)`` at the default step.
    """
    return jnp.arange(CDOM_EX_MIN, CDOM_EX_MAX + 0.5 * step, step, dtype=dtype)


def eta_hawes(
    wave_em: Float[Array, "..."],
    wave_ex: Float[Array, "..."],
) -> Float[Array, "..."]:
    """The Hawes FA7 spectral fluorescence quantum efficiency ``η_Y(λ, λ_e)``.

    Zhai et al. (2017) Eqs. (7)–(8) with the FA7 constants (module docstring
    carries the provenance/confidence flags)::

        η_Y = g_Y(λ_e) · A0(λ_e) · exp{ −[ (1/λ − A1/λ_e − B1)
                                           / (0.6·(A2/λ_e + B2)) ]² }

    Everything is in reciprocal wavelength: for each λ_e the emission is a
    Gaussian in *wavenumber* centered at ``A1/λ_e + B1`` with width
    ``0.6·(A2/λ_e + B2)``. Units: nm⁻¹ (a per-nm emission density — the
    quantum efficiency for emission into dλ is ``η_Y dλ``), so η_Y plays the
    role φ_C·h_C(λ) plays for Chl-fl, jointly. The center and width both move
    with λ_e — **not separable** into an excitation part times an emission
    part.

    ``A0(λ_e)`` is linearly interpolated between the ten tabulated nodes
    (:data:`HAWES_A0_WAVE`), clamped at the ends by ``jnp.interp`` — moot in
    practice, since ``g_Y`` zeroes everything outside [310, 490] nm anyway.

    Notes
    -----
    The Gaussian-in-wavenumber form does **not** strictly enforce a Stokes
    (red-only) shift: the *peak* emission is red of λ_e for every admissible
    λ_e (peak wavenumber ``A1/λ_e + B1 < 1/λ_e`` whenever
    ``λ_e < (1−A1)/B1 ≈ 656 nm``), but the blue tail is genuinely nonzero —
    η_Y at λ = λ_e is ~10 % of the peak value at λ_e = 400 nm. A test pins
    the red-shifted peak; nothing asserts the tail away.

    Parameters
    ----------
    wave_em : Array
        Emission wavelengths λ (nm). Broadcast against ``wave_ex``.
    wave_ex : Array
        Excitation wavelengths λ_e (nm).

    Returns
    -------
    Array
        ``η_Y`` (nm⁻¹), the broadcast shape of the inputs; ≥ 0 everywhere
        (A0 > 0 and exp > 0), zero outside ``g_Y``'s support.
    """
    wave_em = jnp.asarray(wave_em)
    wave_ex = jnp.asarray(wave_ex)
    a0 = jnp.interp(wave_ex, jnp.asarray(HAWES_A0_WAVE), jnp.asarray(HAWES_A0))
    g_y = jnp.where((wave_ex >= GY_EX_MIN) & (wave_ex <= GY_EX_MAX), 1.0, 0.0)
    center = HAWES_A1 / wave_ex + HAWES_B1
    width = 0.6 * (HAWES_A2 / wave_ex + HAWES_B2)
    z = (1.0 / wave_em - center) / width
    return g_y * a0 * jnp.exp(-(z**2))


def truncated_excitation_fraction(
    wave: Float[Array, " wave"] | None = None,
    quad_step: float = 0.25,
) -> Float[Array, " wave"]:
    """The fraction of Hawes-function emission the 350 nm clamp truncates.

    The committed diagnostic of design §2 / M5 task 4: the production kernel
    integrates excitation over 350–490 nm (:func:`cdom_excitation_grid`),
    but ``g_Y`` admits excitation down to 310 nm — so for each emission
    wavelength λ this quantifies what the clamp throws away, **from the Hawes
    function itself** (no IOPs, no Ed, no scene)::

        fraction(λ) = ∫_{310}^{350} η_Y(λ, λ_e) dλ_e
                      / ∫_{310}^{490} η_Y(λ, λ_e) dλ_e

    This function deliberately evaluates η_Y below 350 nm — unlike
    :func:`cdom_kernel`, which provably never reads IOPs or Ed there —
    precisely because its purpose is to characterize the clamp's cost. It is
    a property of the FA7 parameterization alone: the *actual* truncated
    contribution in a scene is further weighted by ``a_cdom(λ_e)·Ed(λ_e)``,
    and UV ``Ed`` is comparatively weak (Zhai et al.'s own rationale for the
    same clamp), so these numbers are a **conservative upper-bound flavor**
    of caveat, not a measured Rrs error.

    Quadrature: trapezoid on uniform ``quad_step`` sub-grids over each range
    (the default 0.25 nm is converged — measured max 3.5e-6 relative against
    a 2× refinement on the canonical grid; even 1 nm sits within 3e-5
    absolute). The denominator is
    strictly positive for every emission wavelength on the canonical grid
    (η_Y's Gaussian tails never underflow there — the far-red 750 nm row
    keeps exp(−z²) ≳ 1e-11 in float32), so the ``jnp.where`` guard below is
    inert in practice; it exists for pathological emission wavelengths far
    outside the Hawes band, where *both* integrals underflow to exactly 0.0
    and the honest answer is defined as 0 — no emission at all, so nothing
    was truncated — rather than a silent 0/0 NaN.

    Parameters
    ----------
    wave : Array, optional
        Emission wavelengths (nm); defaults to the canonical grid.
    quad_step : float, optional
        Sub-grid spacing (nm) of both trapezoid integrals; default 0.25.

    Returns
    -------
    Array
        The truncated fraction per emission wavelength, in [0, 1]; shape
        ``wave.shape``.
    """
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)
    n_lo = int(round((CDOM_EX_MIN - GY_EX_MIN) / quad_step)) + 1
    n_full = int(round((GY_EX_MAX - GY_EX_MIN) / quad_step)) + 1
    ex_lo = jnp.linspace(GY_EX_MIN, CDOM_EX_MIN, n_lo)
    ex_full = jnp.linspace(GY_EX_MIN, GY_EX_MAX, n_full)
    numerator = jnp.trapezoid(eta_hawes(wave[..., None], ex_lo), ex_lo, axis=-1)
    denominator = jnp.trapezoid(eta_hawes(wave[..., None], ex_full), ex_full, axis=-1)
    return jnp.where(denominator > 0.0, numerator / denominator, 0.0)


def cdom_kernel(
    iops,
    geometry,
    wave: Float[Array, " wave"] | None = None,
    wave_ex: Float[Array, " ex"] | None = None,
) -> Float[Array, "*batch wave"]:
    """The analytic CDOM-fluorescence kernel ``K_cdom(λ)`` at unit scale.

    The Chl-fl kernel's S&P98 machinery with the Hawes redistribution
    function η_Y in place of the separable emission line, term for term
    (design §2; compare :func:`robust.rt.inelastic.fluorescence_kernel`):

    - source ``b_bY(λ_e) = ½ · a_cdom(λ_e)`` — isotropic emission, half
      backward. **No reference-yield division**: Chl-fl factors out
      ``PHI_C_REF`` because its yield is a physical handle; here the
      amplitude handle is ``CDOMFl.scale`` (default 1.0), applied by the
      composition (task 5), so this function returns the raw kernel at
      scale = 1.
    - excitation integral (trapezoid) over :func:`cdom_excitation_grid`
      (350–490 nm — the hard clamp *is* the grid) with the true ``Ed(λ_e)``
      from :mod:`robust.rt.ed` (override honored), normalized by ``Ed(λ)``;
    - the ``λ_e/λ`` quanta→energy factor and ``η_Y(λ, λ_e)`` **inside** the
      excitation sum (Zhai et al. Eqs. 5–6: the emission function is
      ``a_cdom(λ_e) · η_Y(λ, λ_e) · λ_e/λ``);
    - attenuation ``K(λ_e) = (a+b_b)/μ_D`` downwelling at excitation,
      ``κ_Y(λ) = (a+b_b)/μ_F`` upwelling at emission (isotropic emission —
      the same ``MU_F`` as Chl-fl);
    - **L_u = E_u/π** and :func:`robust.rt.conventions.rrs_to_Rrs`, the same
      two calls in the same order as Chl-fl.

    **Structural departure from the Chl-fl kernel** (and its honest cost):
    Chl-fl's emission line is λ-only, so its numerator folds into a pure
    ``(..., n_ex)`` array and the line shape post-multiplies the reduced sum.
    η_Y depends on λ *and* λ_e jointly, so it multiplies the integrand
    **before** the reduction: the ``(..., n_em, n_ex)`` tensor here carries
    one extra elementwise multiply by the (batch-free) ``(n_em, n_ex)`` η_Y
    matrix that Chl-fl's fused form avoids. In exchange the contraction is
    29 excitation nodes against Chl-fl's 65, so this kernel should be *no
    more* expensive than Chl-fl's — the actual speed gate is task 7's job.

    Pure JAX: batched over leading axes, ``jit``/``vmap``-safe (fixed-size
    contraction), differentiable in the IOPs (``a_cdom`` included) and
    ``θ_s``.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Must carry ``a_cdom`` — the source term is physically the light CDOM
        absorbed; bulk ``a`` cannot stand in for it.
    geometry : robust.rt.types.Geometry
        ``theta_s`` selects the packaged sky; ``geometry.Ed`` overrides it
        (numerator and denominator from the same sky by construction).
    wave : Array, optional
        Emission wavelengths (nm); defaults to the canonical grid.
    wave_ex : Array, optional
        Excitation quadrature nodes (nm); defaults to
        :func:`cdom_excitation_grid`. Exists for the quadrature-convergence
        gate — passing nodes below 350 nm would defeat the clamp, so don't.

    Returns
    -------
    Array
        ``K_cdom``, sr⁻¹ at unit ``scale``, shape ``(*batch, n_wave)``; ≥ 0,
        broad and featureless across the blue-green (no 685 nm-style line).

    Raises
    ------
    ValueError
        If ``iops.a_cdom`` is ``None`` — a physical requirement: without the
        CDOM component there is no fluorescence source. Disable the process
        (``Inelastic(cdom_fl=None)``, the default) or load IOPs with the
        split (the L23 loaders populate it from ``ag``).
    """
    if iops.a_cdom is None:
        raise ValueError(
            "cdom_kernel: IOPs.a_cdom is None, but the CDOM-fluorescence "
            "source term is b_Y = 0.5 * a_cdom -- bulk absorption cannot "
            "stand in for the CDOM component. Provide a_cdom (e.g. "
            "IOPs.from_total_bb(..., a_cdom=...); the L23 loaders extract "
            "it from `ag`) or leave the process off with "
            "Inelastic(cdom_fl=None)"
        )
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)
    if wave_ex is None:
        wave_ex = cdom_excitation_grid(dtype=jnp.result_type(wave.dtype, float))
    else:
        wave_ex = jnp.asarray(wave_ex)

    a_ex = conventions.interp_spectrum(wave_ex, wave, iops.a)
    bb_ex = conventions.interp_spectrum(wave_ex, wave, iops.bb)
    acdom_ex = conventions.interp_spectrum(wave_ex, wave, iops.a_cdom)
    ed_ex = ed.Ed(geometry.theta_s, wave_ex, override=geometry.Ed)
    ed_em = ed.Ed(geometry.theta_s, wave, override=geometry.Ed)

    # Source and attenuation, S&P98 two-flow (the Chl-fl pattern; no
    # reference-yield factor -- see the docstring).
    bb_y = 0.5 * acdom_ex
    k_ex = (a_ex + bb_ex) / MU_D
    kappa_y_em = (iops.a + iops.bb) / MU_F

    # Everything λ_e-only still folds into one (..., n_ex) array; η_Y cannot
    # (it moves with both axes), so it enters the (..., n_em, n_ex) integrand
    # before the reduction. η_Y itself carries no batch axes -- the broadcast
    # multiply is against a fixed (n_em, n_ex) matrix.
    dx = jnp.diff(wave_ex)
    trapezoid_w = 0.5 * jnp.concatenate([dx[:1], dx[:-1] + dx[1:], dx[-1:]])
    numerator = trapezoid_w * wave_ex * ed_ex * bb_y / MU_D
    eta = eta_hawes(wave[:, None], wave_ex[None, :])
    r_y = jnp.sum(
        eta * numerator[..., None, :] / (k_ex[..., None, :] + kappa_y_em[..., :, None]),
        axis=-1,
    )
    # Same barrier rationale as the Chl-fl kernel: pin the reduced (..., n_em)
    # result so XLA's consumer fusion cannot re-run the reduction downstream.
    r_y = jax.lax.optimization_barrier(r_y) / (wave * ed_em)

    # Isotropic emission: Lu = Eu/pi, then the standard surface transfer --
    # the Chl-fl order, kept exactly.
    return conventions.rrs_to_Rrs(r_y / jnp.pi)
