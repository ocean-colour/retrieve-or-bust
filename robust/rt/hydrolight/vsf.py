"""
The volume-scattering-function design set — spec Appendix B.

Twenty-seven designs, each of which HydroLight can be pointed at: **25 are
tabulated** by this module as a discrete ``beta_tilde(psi)``, and **2 are named
built-ins** we deliberately do not reproduce (see *Built-ins* below).

**Fournier-Forand is a two-parameter family.** It is usually *used* as a
one-parameter family indexed by the backscatter fraction ``B_p``, but it is
derived from the real refractive index ``n`` and the Junge slope ``mu``, so
distinct ``(n, mu)`` pairs give the **same** ``B_p`` with different shapes. That
is what makes a matched-``B_p`` shape experiment possible at all, and it needs no
scattering code beyond the closed forms here.

**What that experiment can and cannot vary** (measured, not assumed — see
:func:`shape_table`, which the test suite regenerates rather than quoting):
at matched ``B_p`` the three FF branches differ by ~50 % at ``psi = 1 deg`` and
~35 % at 10 deg, but by only **1-3 % beyond 120 deg**. Two-component mixtures do
better and not by much: ~5 % at 180 deg. So at matched bulk ``B_p``, nature
leaves very little freedom in the backscatter hemisphere, and the honest lever on
the *backward* axis is the out-of-family anchors and the designs that
deliberately break ``B_p`` matching — not the matched-``B_p`` set. The spec says
so in its Section 11; this docstring says so where the code lives.

**Built-ins, and why they are not tabulated here.** Design ``petzold`` is
Petzold's average-particle VSF and design ``sullivan_twardowski`` is a measured
VSF of that family. Both are *data* we do not hold. HydroLight ships the Petzold
table itself, so the intermediate representation references it **by name** and
the operator's own copy is used; inventing numbers for it would be worse than
useless, because the whole point of an out-of-family anchor is that it is not
something we constructed. ``sullivan_twardowski`` is marked optional and is
dropped from a batch if the operator has no tabulated copy.

References
----------
Fournier, G. R. and J. L. Forand (1994), *Analytic phase function for ocean
water*, Proc. SPIE 2258, 194-201; and Mobley, *Light and Water* / the Ocean
Optics Web Book for the ``B_p(n, mu)`` closed form used here.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "PSI_GRID",
    "TARGET_BP",
    "MIXTURE_RECIPES",
    "VSFDesign",
    "ff_backscatter_fraction",
    "ff_phase",
    "numeric_backscatter_fraction",
    "numeric_norm",
    "FF_BRANCHES",
    "solve_ff_branches",
    "deliverable",
    "mixture_fraction",
    "mixture_brackets",
    "designs",
    "shape_table",
]


def _psi_grid() -> np.ndarray:
    """The tabulation angles, degrees: dense where the phase function moves.

    The forward peak reaches ``psi_min = 1e-3 deg`` deliberately. At 0.1 deg the
    delivered table integrates to as little as 0.32 of unity for the sharpest
    designs, and its quadrature ``B_p`` is then wrong by a factor of three --
    measured, not feared. See :func:`deliverable`.
    """
    fine = np.logspace(-3.0, 1.0, 400)
    coarse = np.arange(10.0, 180.0 + 1e-9, 0.5)
    return np.unique(np.concatenate([fine, coarse]))


#: Scattering angles the tabulated ``beta_tilde`` is delivered on, degrees.
#: Logarithmic below 10 deg because the forward peak spans two decades there,
#: 1 deg steps above it. 180 deg is included exactly -- ``beta_tilde(pi)`` is a
#: model parameter (``PhaseParams.beta_tilde_pi``) and must not be interpolated.
PSI_GRID = _psi_grid()

#: The five bulk backscatter fractions every tabulated design is built at.
#: Spans ~7.5x, which is the design's nominal band; L23 spans 1.7x.
TARGET_BP = (0.004, 0.007, 0.012, 0.020, 0.030)

#: The Fournier-Forand branches, ``{target B_p: ((n, mu), ...)}`` -- **measured,
#: not chosen**. Each entry is a deliverable ``(n, mu)`` solution (see
#: :func:`deliverable`) reproducing its target within 4e-4, and the entries at
#: one target differ in ``n`` by at least 0.02 so that "a branch" means a
#: genuinely different particle population rather than a neighbouring point on
#: one ridge.
#:
#: **The counts are uneven, and that is the finding.** At ``B_p = 0.004`` exactly
#: one deliverable solution exists; at 0.030 there are six. Low backscatter
#: fractions pin the Fournier-Forand shape almost completely, so the
#: matched-``B_p`` shape experiment has leverage only at the higher ``B_p`` end.
#: :func:`solve_ff_branches` re-derives this table from scratch and the test
#: suite asserts the two agree.
FF_BRANCHES = {
    0.004: ((1.020, 3.72),),
    0.007: ((1.020, 3.86), (1.040, 3.64)),
    0.012: ((1.020, 4.00), (1.040, 3.80), (1.060, 3.64)),
    0.02: ((1.020, 4.14), (1.045, 3.92), (1.065, 3.78), (1.085, 3.68), (1.105, 3.60)),
    0.03: (
        (1.025, 4.20),
        (1.055, 3.98),
        (1.090, 3.80),
        (1.110, 3.72),
        (1.140, 3.62),
        (1.160, 3.56),
    ),
}

#: The two mixture recipes, ``(low-B_p component, high-B_p component)`` in
#: ``(n, mu)``. Both components of both recipes are themselves deliverable, which
#: is the binding constraint: the deliverable set spans ``B_p`` 0.0037 to 0.0483,
#: so ``mixA`` (its two extremes) brackets all five targets and ``mixB`` -- a
#: deliberately narrower, differently-shaped pair -- brackets four.
MIXTURE_RECIPES = {
    "mixA": ((1.020, 3.70), (1.260, 3.54)),
    "mixB": ((1.025, 3.68), (1.165, 3.64)),
}

#: Minimum fraction of unit normalisation a delivered table must reach, and the
#: maximum ``|B_p(quadrature) - B_p(closed form)|`` it may carry.
NORM_FLOOR = 0.995
BP_TOL = 4e-4

#: Junge slope held fixed when a single FF parameter is inferred from a measured
#: ``B_p`` (used by :mod:`robust.rt.hydrolight.l23_recon`). FF has two
#: parameters and a ``B_p`` constrains only one combination of them, so one must
#: be pinned and **recorded**; 3.5 sits mid-range for oceanic particles.
FF_MU_FIXED = 3.5


def ff_backscatter_fraction(n, mu):
    """Fournier-Forand bulk backscatter fraction ``B_p``, in closed form.

    Parameters
    ----------
    n : float or array
        Real refractive index of the particles relative to water.
    mu : float or array
        Junge (hyperbolic) slope of the size distribution.

    Returns
    -------
    float or ndarray
        ``B_p``, the fraction of scattering into the backward hemisphere.
    """
    n = np.asarray(n, dtype=float)
    mu = np.asarray(mu, dtype=float)
    nu = (3.0 - mu) / 2.0
    d90 = 2.0 / (3.0 * (n - 1.0) ** 2)
    num = 1.0 - d90 ** (nu + 1.0) - 0.5 * (1.0 - d90**nu)
    den = (1.0 - d90) * d90**nu
    return 1.0 - num / den


def ff_phase(psi_deg, n, mu):
    """Fournier-Forand phase function ``beta_tilde(psi)``, sr^-1.

    Parameters
    ----------
    psi_deg : array_like
        Scattering angles, degrees. ``psi = 0`` is excluded (the function
        diverges there); :data:`PSI_GRID` starts at 0.1 deg for that reason.
    n, mu : float
        As :func:`ff_backscatter_fraction`.

    Returns
    -------
    ndarray
        ``beta_tilde`` at each angle, normalised so that ``int beta_tilde dOmega
        = 1`` analytically. The delivered table is renormalised on
        :data:`PSI_GRID` (see :func:`designs`), because a discrete table is what
        HydroLight integrates and the forward peak is not fully resolved by any
        finite grid.
    """
    psi = np.radians(np.asarray(psi_deg, dtype=float))
    nu = (3.0 - mu) / 2.0
    u = 2.0 * np.sin(psi / 2.0)
    d = u**2 / (3.0 * (n - 1.0) ** 2)
    d180 = 4.0 / (3.0 * (n - 1.0) ** 2)
    t1 = 1.0 / (4.0 * np.pi * (1.0 - d) ** 2 * d**nu)
    t2 = (
        nu * (1.0 - d)
        - (1.0 - d**nu)
        + (d * (1.0 - d**nu) - nu * (1.0 - d)) / np.sin(psi / 2.0) ** 2
    )
    t3 = (
        (1.0 - d180**nu)
        / (16.0 * np.pi * (d180 - 1.0) * d180**nu)
        * (3.0 * np.cos(psi) ** 2 - 1.0)
    )
    return t1 * t2 + t3


def _cos_trapz(psi_deg, beta, back_only=False):
    """``int beta dOmega`` by trapezoid in ``cos psi`` -- the natural variable.

    ``dOmega = 2 pi sin(psi) dpsi = -2 pi d(cos psi)``, so integrating against
    ``cos psi`` removes the ``sin psi`` factor that a naive trapezoid in ``psi``
    handles badly near the forward peak.
    """
    mu = np.cos(np.radians(np.asarray(psi_deg, dtype=float)))
    order = np.argsort(mu)
    mu = mu[order]
    b = np.asarray(beta, dtype=float)[order]
    if back_only:
        keep = mu <= 0.0
        mu, b = mu[keep], b[keep]
    return 2.0 * np.pi * float(np.trapezoid(b, mu))


def numeric_norm(psi_deg, beta_tilde):
    """``int beta_tilde dOmega`` on a discrete grid -- 1 for a perfect table."""
    return _cos_trapz(psi_deg, beta_tilde)


def numeric_backscatter_fraction(psi_deg, beta_tilde):
    """``B_p`` by quadrature over the backward hemisphere of a tabulated VSF.

    The independent check on :func:`ff_backscatter_fraction`: the closed form and
    the table must agree, or one of them is wrong -- or, as it turned out, the
    grid is too coarse to carry the design at all (:func:`deliverable`).
    """
    total = _cos_trapz(psi_deg, beta_tilde)
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    return _cos_trapz(psi_deg, beta_tilde, back_only=True) / total


def deliverable(n, mu, psi_deg=None, norm_floor=NORM_FLOOR, bp_tol=BP_TOL):
    """Whether ``(n, mu)`` can be *shipped* as a discrete table, and why.

    A Fournier-Forand design is only useful to us if the table we hand the
    operator carries the phase function we meant. For a Junge slope near 3 it
    does not: the forward peak then behaves like ``psi**-1.93``, whose integral
    converges so slowly that at ``psi_min = 1e-5 deg`` the table still holds only
    64 % of the scattering and its quadrature ``B_p`` is out by a factor of
    three. **Those designs are not deliverable on any finite grid**, and the
    deliverable set turns out to require ``mu >= 3.52``.

    This is the criterion that shrank the design set from the 15 Fournier-Forand
    branches originally specified to 17 unevenly distributed ones, and it is
    applied here rather than assumed.

    Parameters
    ----------
    n, mu : float
        Fournier-Forand parameters.
    psi_deg : array_like, optional
        The grid to test on; defaults to :data:`PSI_GRID`.
    norm_floor : float, optional
        Minimum ``int beta_tilde dOmega`` the raw table must reach.
    bp_tol : float, optional
        Maximum ``|B_p(quadrature) - B_p(closed form)|``.

    Returns
    -------
    tuple
        ``(ok, closed_form_B_p, norm, numeric_B_p)``.
    """
    psi = PSI_GRID if psi_deg is None else np.asarray(psi_deg, dtype=float)
    cf = float(ff_backscatter_fraction(n, mu))
    if not np.isfinite(cf) or cf <= 0.0:
        return False, cf, float("nan"), float("nan")
    beta = ff_phase(psi, n, mu)
    norm = numeric_norm(psi, beta)
    bp = numeric_backscatter_fraction(psi, beta)
    ok = bool(
        np.isfinite(norm)
        and np.isfinite(bp)
        and norm >= norm_floor
        and abs(bp - cf) < bp_tol
    )
    return ok, cf, norm, bp


#: The ``(n, mu)`` search grid :func:`solve_ff_branches` scans. Fixed, so a
#: branch is "the closest point of this grid", never the output of an iterative
#: solve whose path could drift between releases.
_N_SCAN = np.round(np.arange(1.020, 1.2601, 0.005), 4)
_MU_SCAN = np.round(np.arange(3.50, 4.501, 0.02), 3)

#: Minimum separation in ``n`` for two solutions to count as distinct branches.
_MIN_N_GAP = 0.02


def solve_ff_branches(target_bp, bp_tol=BP_TOL, min_n_gap=_MIN_N_GAP):
    """Re-derive the deliverable ``(n, mu)`` branches for one target ``B_p``.

    The generator of :data:`FF_BRANCHES`. Kept as live code rather than a
    comment so the test suite can assert that the shipped constants are still
    what the criterion produces.

    Parameters
    ----------
    target_bp : float
        Desired bulk backscatter fraction.
    bp_tol : float, optional
        Tolerance on ``B_p``.
    min_n_gap : float, optional
        Minimum separation in ``n`` between branches.

    Returns
    -------
    tuple of tuple
        ``(n, mu)`` pairs, ordered by ``n``. May be empty, and may hold fewer
        than three entries -- at low ``B_p`` it holds exactly one.
    """
    hits = []
    for n in _N_SCAN:
        best = None
        for mu in _MU_SCAN:
            ok, cf, _, _ = deliverable(n, mu, bp_tol=bp_tol)
            if not ok or abs(cf - target_bp) >= bp_tol:
                continue
            if best is None or abs(cf - target_bp) < abs(best[2] - target_bp):
                best = (float(n), float(mu), cf)
        if best is not None:
            hits.append(best)
    picked = []
    for n, mu, _ in sorted(hits):
        if all(abs(n - p[0]) >= min_n_gap for p in picked):
            picked.append((round(n, 3), round(mu, 2)))
    return tuple(picked)


def mixture_fraction(target_bp, low, high):
    """Scattering-weighted fraction of the high-``B_p`` component.

    Parameters
    ----------
    target_bp : float
        Desired bulk backscatter fraction of the mixture.
    low, high : tuple
        ``(n, mu)`` of the two components; ``low`` must have the smaller ``B_p``.

    Returns
    -------
    float
        ``f`` in ``(0, 1)`` with ``(1 - f) B_low + f B_high = target_bp``.

    Raises
    ------
    ValueError
        If the components do not bracket the target.
    """
    b_lo = float(ff_backscatter_fraction(*low))
    b_hi = float(ff_backscatter_fraction(*high))
    if not b_lo < target_bp < b_hi:
        raise ValueError(
            f"components B_p=({b_lo:.5f}, {b_hi:.5f}) do not bracket {target_bp}"
        )
    return (target_bp - b_lo) / (b_hi - b_lo)


def mixture_brackets(recipe, target_bp):
    """Whether ``recipe``'s two components bracket ``target_bp``."""
    low, high = MIXTURE_RECIPES[recipe]
    return (
        float(ff_backscatter_fraction(*low))
        < target_bp
        < float(ff_backscatter_fraction(*high))
    )


@dataclasses.dataclass(frozen=True)
class VSFDesign:
    """One volume-scattering-function design.

    Attributes
    ----------
    name : str
        Stable identifier, used in water-body ids and in the manifest.
    kind : str
        ``"ff"``, ``"mixture"`` or ``"builtin"``.
    target_bp : float or None
        The bulk backscatter fraction the design was built at; None for the
        built-ins, whose ``B_p`` is whatever their table says.
    params : dict
        The construction parameters, recorded verbatim for the manifest.
    beta_tilde : ndarray or None
        ``beta_tilde`` on :data:`PSI_GRID`, sr^-1, renormalised to integrate to
        1 on that grid. None for the built-ins.
    builtin : str or None
        The operator-side table this design refers to instead of carrying one.
    optional : bool
        True when the batch may proceed without this design (the operator may
        not hold the table).
    """

    name: str
    kind: str
    target_bp: float | None
    params: dict
    beta_tilde: np.ndarray | None
    builtin: str | None = None
    optional: bool = False

    @property
    def tabulated(self) -> bool:
        """Whether this design carries its own table."""
        return self.beta_tilde is not None

    def realised_bp(self) -> float | None:
        """``B_p`` of the delivered table, by quadrature; None for built-ins."""
        if self.beta_tilde is None:
            return None
        return numeric_backscatter_fraction(PSI_GRID, self.beta_tilde)


def _normalise(beta):
    """Scale a table so it integrates to 1 on :data:`PSI_GRID`."""
    return np.asarray(beta, dtype=float) / numeric_norm(PSI_GRID, beta)


def designs():
    """The volume-scattering-function design set of spec Appendix B.

    Returns
    -------
    tuple of VSFDesign
        **28 designs**: 17 Fournier-Forand branches (:data:`FF_BRANCHES`), 9
        two-component mixtures (every ``(recipe, target)`` pair the recipe
        brackets), and 2 named built-ins. Deterministic -- the same objects in
        the same order on every call.

    Notes
    -----
    The composition differs from the specification's "15 FF + 10 mixtures + 2",
    which was written before :func:`deliverable` was applied. The total is 28
    rather than 27 and the split has moved; what did **not** change is that the
    tabulated designs span the same five target ``B_p`` values.
    """
    out = []
    for target, pairs in FF_BRANCHES.items():
        for k, (n, mu) in enumerate(pairs):
            out.append(
                VSFDesign(
                    name=f"ff_bp{target:g}_br{k + 1}",
                    kind="ff",
                    target_bp=target,
                    params={
                        "n": n,
                        "mu": mu,
                        "B_p_closed_form": float(ff_backscatter_fraction(n, mu)),
                    },
                    beta_tilde=_normalise(ff_phase(PSI_GRID, n, mu)),
                )
            )
    for recipe, (low, high) in MIXTURE_RECIPES.items():
        for target in TARGET_BP:
            if not mixture_brackets(recipe, target):
                continue
            f = mixture_fraction(target, low, high)
            beta = (1.0 - f) * ff_phase(PSI_GRID, *low) + f * ff_phase(PSI_GRID, *high)
            out.append(
                VSFDesign(
                    name=f"{recipe}_bp{target:g}",
                    kind="mixture",
                    target_bp=target,
                    params={
                        "low_n": low[0],
                        "low_mu": low[1],
                        "low_B_p": float(ff_backscatter_fraction(*low)),
                        "high_n": high[0],
                        "high_mu": high[1],
                        "high_B_p": float(ff_backscatter_fraction(*high)),
                        "fraction_high": f,
                    },
                    beta_tilde=_normalise(beta),
                )
            )
    out.append(
        VSFDesign(
            name="petzold",
            kind="builtin",
            target_bp=None,
            params={"note": "Petzold average-particle; B_p approx 0.0183"},
            beta_tilde=None,
            builtin="petzold_average_particle",
        )
    )
    out.append(
        VSFDesign(
            name="sullivan_twardowski",
            kind="builtin",
            target_bp=None,
            params={"note": "measured VSF; supplied by the operator if held"},
            beta_tilde=None,
            builtin="sullivan_twardowski_measured",
            optional=True,
        )
    )
    return tuple(out)


def shape_table(target_bp=0.030, angles=(1, 10, 45, 90, 120, 135, 160, 180)):
    """The matched-``B_p`` shape comparison of spec Appendix B.

    Regenerated rather than quoted: if the Fournier-Forand algebra were wrong,
    or the deliverable branch set moved, the specification's central claim about
    the backscatter hemisphere would fail here rather than stand unchallenged in
    a document. It has already earned that -- the first version of this table was
    computed from ``(n, mu)`` pairs that :func:`deliverable` later rejected, and
    it understated the backward contrast by roughly a factor of five.

    Parameters
    ----------
    target_bp : float, optional
        The matched backscatter fraction, default 0.030 -- the target carrying
        the most branches and therefore the most contrast to report.
    angles : sequence of float, optional
        Scattering angles, degrees.

    Returns
    -------
    dict
        ``{"target_bp", "angles", "branches", "ratio", "spread"}``. Ratios are
        taken against the first (lowest ``n``) branch; ``spread`` is
        ``max/min - 1`` across branches at each angle.
    """
    angles = np.asarray(angles, dtype=float)
    pairs = FF_BRANCHES[target_bp]
    vals = {f"br{k + 1}": ff_phase(angles, n, mu) for k, (n, mu) in enumerate(pairs)}
    ref = vals["br1"]
    stack = np.stack(list(vals.values()))
    return {
        "target_bp": target_bp,
        "angles": angles,
        "branches": vals,
        "ratio": {k: v / ref for k, v in vals.items()},
        "spread": stack.max(axis=0) / stack.min(axis=0) - 1.0,
    }
