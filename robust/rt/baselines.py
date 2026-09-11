"""
Comparison models: the analytical baselines the hybrid must beat.

These are not part of the hybrid. They exist because every accuracy claim in this
project is **relative** -- "beats standard Gordon on the held-out splits" (coding
plan M3/M4) -- so the thing being beaten has to be in the repo, differentiable, and
computed on identical data rather than quoted from a paper.

Standard Gordon (1988) landed at M2; **O25** (Pitarch et al. 2025) joins at M4.

**PR05 is deliberately absent.** Its coefficients are a 4-D
``(theta_s, theta_v, dphi, gamma_b)`` lookup table that the paper does not print and
that is not in this repo -- it exists behind a 2005 institutional URL or inside
POLYMER. And because L23 is nadir-only, a refit here could never populate the two
sensor-geometry axes, so it would be a different object wearing the same name. Recorded
as a gap rather than approximated (prompt 5, Q8).

**Gordon takes the same arguments as** :func:`~robust.rt.hybrid.forward` **and
ignores two of them.** ``phase_params`` and ``geometry`` are accepted and
discarded, which is not an implementation shortcut but the model's defining
limitation: standard Gordon has no phase-function input and no solar-zenith
dependence. L23 says ``Rrs`` falls by a median 5.1% from 0 deg to 60 deg, so a
zenith-blind model *must* mis-fit at least one angle. That gap is what the ZTT
backbone and the residual emulator are for, and keeping the signatures
interchangeable is what lets M4 score all of them in one loop.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions

__all__ = [  # noqa: RUF022  - grouped by role
    "G1_GORDON",
    "G2_GORDON",
    "rrs_gordon",
    "Rrs_gordon",
    "O25_L23_REFIT",
    "O25_RRS_CEILING",
    "o25_coefficients",
    "rrs_o25",
    "Rrs_o25",
    "fit_o25",
]

#: Canonical Gordon (1988) coefficients for ``rrs = G1 u + G2 u^2``. Fixed, not
#: fitted -- the "standard Gordon" of the coding plan's gates. The synthesis figures
#: (``context/RT/make_rt_elastic_figures.py``) use these same two numbers, so our
#: rRMS is comparable with the ladder in ``context/RT/fig_rrms_ladder.csv``.
G1_GORDON = 0.0949
#: float: The quadratic coefficient ``G2`` of that same Gordon (1988) form.
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
        :func:`~robust.rt.hybrid.forward`; standard Gordon has no phase-function input.
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


# --------------------------------------------------------------------------- O25

#: O25's four coefficients, refit on L23 — **not** the published values.
#:
#: Rows are ``(theta_s_deg, Gw0, Gw1, Gp0, Gp1)``. Fitted by :func:`fit_o25` on the
#: **``scene_train`` split only** (2656 samples per zenith x 81 lambda), by closed-form
#: relatively-weighted least squares, and embedded here so O25 is usable and testable
#: without ``$OS_COLOR`` — the same reasoning that puts ``BB_W_L23`` in
#: :mod:`robust.rt.conventions`. ``test_baselines.py`` refits and checks these numbers
#: reproduce, so the table cannot drift from the code that made it.
#:
#: **Why these are a refit and not Pitarch et al.'s own numbers.** The published ``G``
#: lookup tables are not printed in the paper (it shows them only as plots) and are not
#: in this repo; they live in the authors' code. Re-fitting is the alternative the paper
#: itself uses when it evaluates O25 on L23 (its Fig. 3). So every table and figure must
#: call this **"O25 form, refit on L23"** — it is a strong benchmark, but it is not a
#: statement about the published model, and it has seen our training data.
#:
#: **The refit has no sensor-zenith or azimuth axis**, because L23 is nadir-only. The
#: real O25 indexes all three geometry angles; ours indexes the solar zenith alone.
O25_L23_REFIT = (
    (0.0, 0.05866762, 0.02490574, 0.04018414, 0.12313511),
    (30.0, 0.05721442, 0.02932806, 0.04007408, 0.15273771),
    (60.0, 0.05249313, 0.03749838, 0.03936996, 0.20619877),
)

#: float: O25's stated validity ceiling -- the quadratic was fitted for ``Rrs``
#: up to this value.
#: L23 reaches 0.0248, well inside it, so nothing here extrapolates in brightness.
O25_RRS_CEILING = 0.06


def o25_coefficients(
    theta_s, coeffs=O25_L23_REFIT
) -> tuple[
    Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]
]:
    """Interpolate ``(Gw0, Gw1, Gp0, Gp1)`` to a solar zenith.

    O25's coefficients are a lookup table over geometry, so using it at an arbitrary
    angle means interpolating -- which is how the published model is used too.

    Parameters
    ----------
    theta_s : Array
        Solar zenith angle (degrees), any shape.
    coeffs : sequence of tuple, optional
        Rows of ``(theta_s_deg, Gw0, Gw1, Gp0, Gp1)``, ascending in angle. Defaults to
        :data:`O25_L23_REFIT`; pass the published table here to score the real O25.

    Returns
    -------
    tuple of Array
        The four coefficients, broadcast to ``theta_s``'s shape.

    Notes
    -----
    Linear in the angle and **flat outside the tabulated range** -- ``jnp.interp``
    clamps rather than extrapolating a ramp, which is the conservative choice: a linear
    extrapolation of ``Gp1`` past 60 deg grows without bound, while a held value at
    least stays inside the fitted family. Either way it is extrapolation, and the
    honest response is to report it rather than to trust it.

    Differentiable in ``theta_s`` **except at the tabulated angles themselves**,
    where a piecewise-linear lookup has a kink: ``jax.grad`` takes one one-sided
    slope there while a central difference averages both, and the two disagree by
    O(1) (measured: 69% at 30 deg). That is inherent to a lookup-table model rather
    than a defect here -- but it matters, because L23's three solar zeniths *are*
    the nodes, so a finite-difference check on L23 geometry lands on one every time.
    Check the gradient at an intermediate angle instead: at 45 deg it agrees to
    2.7e-10.
    """
    table = jnp.asarray(coeffs)
    angles = table[:, 0]
    theta_s = jnp.asarray(theta_s)
    return tuple(jnp.interp(theta_s, angles, table[:, i]) for i in range(1, 5))


def Rrs_o25(
    iops,
    phase_params=None,
    geometry=None,
    wave: Float[Array, " wave"] | None = None,
    *,
    coeffs=O25_L23_REFIT,
) -> Float[Array, "*batch wave"]:
    """Above-water reflectance from the O25 bivariate quadratic.

    ``Rrs = (Gw0 + Gw1 w_bw) w_bw + (Gp0 + Gp1 w_bp) w_bp``, with
    ``w_bw = bb_w/(a+bb)`` and ``w_bp = bb_p/(a+bb)`` (Pitarch et al. 2025,
    *Remote Sens. Environ.* **329**, 114920, Eqs. 3-4).

    **This is the one comparison model that is defined in ``Rrs``**, not ``rrs``, which
    is why the pair here is the reverse of :func:`rrs_gordon` / :func:`Rrs_gordon`:
    ``Rrs_o25`` is the primitive and :func:`rrs_o25` converts for scoring.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Absorption and the water/particle backscattering split. O25 uses the split
        directly -- it is a two-branch model, one branch per backscattering source --
        which is why it costs this project nothing: ``bb_w`` and ``bb_p`` are already
        separate fields (design §3).
    phase_params : optional
        **Ignored.** Not an oversight: O25 has no phase-function input at all. Its
        coefficients were calibrated on a set with *prescribed* Fournier-Forand phase
        functions, so phase-function shape is baked into the fitted numbers rather than
        being adjustable. That is the limitation the ZTT backbone exists to address.
    geometry : robust.rt.types.Geometry
        Solar zenith is **used** (unlike Gordon). ``theta_v`` and ``dphi`` are ignored
        by this refit because L23 is nadir-only; the published model indexes them.
    wave : Array, optional
        **Ignored.** The coefficients are wavelength-independent *by construction* --
        λ enters only through the IOPs. This is a design feature of O25, not a
        simplification made here.
    coeffs : sequence of tuple, optional
        As :func:`o25_coefficients`.

    Returns
    -------
    Array
        ``Rrs``, sr^-1, shape ``(..., n_wave)``.

    Raises
    ------
    ValueError
        If ``geometry`` is ``None``. Gordon can be called without one because it is
        genuinely zenith-blind; O25 cannot, and silently substituting a default angle
        would produce plausible numbers from the wrong coefficients.
    """
    if geometry is None:
        raise ValueError(
            "Rrs_o25 needs a geometry: its coefficients are indexed by solar zenith. "
            "(rrs_gordon accepts geometry=None because Gordon has no zenith term.)"
        )
    bb = iops.bb
    denom = iops.a + bb
    w_bw = iops.bb_w / denom
    w_bp = iops.bb_p / denom
    Gw0, Gw1, Gp0, Gp1 = o25_coefficients(geometry.theta_s, coeffs)
    # theta_s is per-sample; the albedos are per-sample-and-wavelength.
    Gw0, Gw1, Gp0, Gp1 = (g[..., None] for g in (Gw0, Gw1, Gp0, Gp1))
    return (Gw0 + Gw1 * w_bw) * w_bw + (Gp0 + Gp1 * w_bp) * w_bp


def rrs_o25(
    iops,
    phase_params=None,
    geometry=None,
    wave: Float[Array, " wave"] | None = None,
    *,
    coeffs=O25_L23_REFIT,
) -> Float[Array, "*batch wave"]:
    """Subsurface reflectance from O25 — :func:`Rrs_o25` through the interface.

    Parameters
    ----------
    iops, phase_params, geometry, wave, coeffs
        As :func:`Rrs_o25`.

    Returns
    -------
    Array
        ``rrs``, sr^-1, shape ``(..., n_wave)``.

    Notes
    -----
    Scoring happens in ``rrs`` (design §6), so this is the entry point the metrics
    call. Note the asymmetry it introduces: O25's coefficients were fitted against
    ``Rrs``, so the conversion sits *between* the fit and the score. Any table
    reporting O25's rRMS should say so.
    """
    return conventions.Rrs_to_rrs(
        Rrs_o25(iops, phase_params, geometry, wave, coeffs=coeffs)
    )


def fit_o25(
    iops,
    Rrs: Float[Array, "sample wave"],
    geometry,
    *,
    train,
    zeniths=(0.0, 30.0, 60.0),
    weighted: bool = True,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Fit O25's four coefficients per solar zenith, by least squares.

    **Closed form, not an optimisation.** The model is linear in its coefficients --
    ``Rrs = Gw0 w_bw + Gw1 w_bw^2 + Gp0 w_bp + Gp1 w_bp^2`` -- so this is one
    ``lstsq`` per zenith. There is no seed, no learning rate and no stopping rule, so
    the result is deterministic by construction rather than by care.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        The full batch.
    Rrs : Array
        Reference **above-water** reflectance, shape ``(n_sample, n_wave)``. O25 is
        defined in ``Rrs``, so it is fitted there.
    geometry : robust.rt.types.Geometry
        Per-sample solar zenith, used to group the fit.
    train : numpy.ndarray
        Boolean mask over the sample axis. **Required**: fitting a comparison model on
        the test split would flatter the model it is being compared against, and it is
        the one direction of bias nobody thinks to check.
    zeniths : sequence of float, optional
        The angles to fit separately.
    weighted : bool, optional
        Weight each residual by ``1/Rrs`` (default), so the objective matches the
        **relatively weighted** rRMS everything here is scored with. ``False``
        reproduces the paper's unweighted fit.

    Returns
    -------
    tuple of tuple
        Rows of ``(theta_s_deg, Gw0, Gw1, Gp0, Gp1)``, suitable for
        :data:`O25_L23_REFIT`.

    Raises
    ------
    ValueError
        If ``train`` selects no samples at one of ``zeniths``.

    Notes
    -----
    **The weighting is not a detail.** Measured on L23: the weighted fit reaches
    0.68-0.73% rRMS while the paper's unweighted fit reaches 2.5-2.7%, because an
    unweighted objective in ``Rrs`` optimises the bright blue and abandons the dark
    red -- exactly the failure mode the project's relative metric exists to expose.
    Fitting a *rival* model with the wrong objective would have made our own hybrid
    look four times better than a fair comparison allows, so the default is the fair
    one and the paper's choice is behind a flag.
    """
    a = np.asarray(iops.a)
    bb_w = np.asarray(iops.bb_w)
    bb_p = np.asarray(iops.bb_p)
    denom = a + bb_w + bb_p
    w_bw, w_bp = bb_w / denom, bb_p / denom
    Rrs = np.asarray(Rrs)
    theta_s = np.asarray(geometry.theta_s)
    train = np.asarray(train)

    rows = []
    for zenith in zeniths:
        mask = train & (theta_s == zenith)
        if not mask.any():
            raise ValueError(
                f"fit_o25: the train mask selects no samples at theta_s = {zenith}"
            )
        design = np.stack(
            [w_bw[mask], w_bw[mask] ** 2, w_bp[mask], w_bp[mask] ** 2], axis=-1
        ).reshape(-1, 4)
        target = Rrs[mask].reshape(-1)
        if weighted:
            # min sum(((X b - y)/y)^2) == min sum((X/y . b - 1)^2)
            beta, *_ = np.linalg.lstsq(
                design / target[:, None], np.ones_like(target), rcond=None
            )
        else:
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        rows.append((float(zenith), *(float(v) for v in beta)))
    return tuple(rows)
