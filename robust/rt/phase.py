"""
Fournier–Forand particle phase functions, in JAX (M7 task 2, ``rt_fit_ztt.md``).

L23 (Loisel et al. 2023 §2) prescribes its particle phase functions as
Fournier & Forand (1994) with a fixed backscatter ratio per component:
``b̃b = 0.010`` for phytoplankton and ``0.018`` for non-algal particles. The FF
function has two shape parameters, the real refractive index ``n`` and the
Junge slope ``µ``, so a backscatter ratio alone does not fix it. L23 does not
say which ``(n, µ)`` it used. This module adopts **HydroLight's own "FF from
bb/b" convention**, the Mobley et al. (2002) relation

    n = 1.01 + 0.1542 (µ − 3),

and solves for ``µ`` from ``B``. That is an *assumption* (Q8). M7 task 1 gave
it mild support: among five conventions it gave the asymptotic µ∞ closest to
L23's profiles (``design/py/explore_mu_inf_eigen.py``).

Everything on the forward path is pure JAX and differentiable, including the
``B → µ`` inversion (table interpolation refined by Newton steps).

Formulas: Mobley, Sundman & Boss (2002), *Appl. Opt.* 41, 1035, Eqs. (5)–(7).
With ``ν = (3 − µ)/2`` and ``δ(ψ) = 4 sin²(ψ/2) / (3 (n − 1)²)``:

    p̃(ψ) = [ν(1−δ) − (1−δ^ν) + (δ(1−δ^ν) − ν(1−δ)) / sin²(ψ/2)]
           / (4π (1−δ)² δ^ν)
         + (1 − δ180^ν) (3 cos²ψ − 1) / (16π (δ180 − 1) δ180^ν)

    B = 1 − [1 − δ90^{ν+1} − ½(1 − δ90^ν)] / [(1 − δ90) δ90^ν]

``p̃`` is normalized so that ``2π ∫ p̃ sin ψ dψ = 1`` over ``[0, π]``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

__all__ = [
    "MOBLEY_N0",
    "MOBLEY_DN_DMU",
    "mobley_n",
    "ff_phase",
    "ff_backscatter_ratio",
    "ff_mu_for_B",
    "ff_phase_for_B",
    "P_bb_ff",
]

#: Mobley et al. (2002): ``n = MOBLEY_N0 + MOBLEY_DN_DMU (µ − 3)``.
MOBLEY_N0 = 1.01
#: float: Slope of the same relation.
MOBLEY_DN_DMU = 0.1542

#: The ``µ`` grid of the ``B → µ`` table. B is monotone increasing in µ under
#: the Mobley relation: B(3.02) ≈ 1.6e-5, B(4.8) ≈ 0.40. That covers every
#: particle backscatter ratio in L23 (0.010–0.018) and PB24 (0.0010–0.036)
#: with room to spare.
_MU_TABLE = np.linspace(3.02, 4.8, 4097)

#: Newton refinements applied after table interpolation. Two take the round
#: trip ``B → µ → B`` to float64 precision (tested).
_NEWTON_STEPS = 2


def mobley_n(mu):
    """Refractive index tied to the Junge slope, Mobley et al. (2002).

    Parameters
    ----------
    mu : Array
        Junge slope ``µ`` (dimensionless, > 3).

    Returns
    -------
    Array
        Real refractive index ``n`` relative to water.
    """
    return MOBLEY_N0 + MOBLEY_DN_DMU * (jnp.asarray(mu) - 3.0)


def ff_phase(psi_deg, n, mu) -> Float[Array, "..."]:
    """The Fournier–Forand phase function ``p̃(ψ)`` (sr^-1).

    Parameters
    ----------
    psi_deg : Array
        Scattering angle (degrees), in ``(0, 180]``. At exactly 0 the formula is
        singular; the function is integrable there but has no finite value.
    n, mu : Array
        Real refractive index and Junge slope. Broadcast against ``psi_deg``.

    Returns
    -------
    Array
        ``p̃(ψ)``, normalized to one over the sphere.
    """
    psi = jnp.deg2rad(jnp.asarray(psi_deg))
    n = jnp.asarray(n)
    nu = (3.0 - jnp.asarray(mu)) / 2.0
    scale = 4.0 / (3.0 * (n - 1.0) ** 2)
    s2 = jnp.sin(psi / 2.0) ** 2
    d = scale * s2
    d180 = scale
    d_nu = d**nu
    main = (
        nu * (1.0 - d) - (1.0 - d_nu) + (d * (1.0 - d_nu) - nu * (1.0 - d)) / s2
    ) / (4.0 * jnp.pi * (1.0 - d) ** 2 * d_nu)
    d180_nu = d180**nu
    tail = (
        (1.0 - d180_nu)
        * (3.0 * jnp.cos(psi) ** 2 - 1.0)
        / (16.0 * jnp.pi * (d180 - 1.0) * d180_nu)
    )
    return main + tail


def ff_backscatter_ratio(n, mu):
    """Backscatter fraction ``B = bb/b`` of an FF phase function, closed form.

    Parameters
    ----------
    n, mu : Array
        Real refractive index and Junge slope.

    Returns
    -------
    Array
        ``B``, dimensionless.
    """
    nu = (3.0 - jnp.asarray(mu)) / 2.0
    d90 = 4.0 / (3.0 * (jnp.asarray(n) - 1.0) ** 2) * 0.5
    return 1.0 - (1.0 - d90 ** (nu + 1.0) - 0.5 * (1.0 - d90**nu)) / (
        (1.0 - d90) * d90**nu
    )


def _B_of_mu(mu):
    """``B(µ)`` along the Mobley relation."""
    return ff_backscatter_ratio(mobley_n(mu), mu)


def _B_of_mu_numpy(mu):
    """``B(µ)`` in NumPy float64 -- for building the table at import, where
    JAX would silently compute in float32."""
    n = MOBLEY_N0 + MOBLEY_DN_DMU * (mu - 3.0)
    nu = (3.0 - mu) / 2.0
    d90 = 4.0 / (3.0 * (n - 1.0) ** 2) * 0.5
    return 1.0 - (1.0 - d90 ** (nu + 1.0) - 0.5 * (1.0 - d90**nu)) / (
        (1.0 - d90) * d90**nu
    )


_LOG_B_TABLE = np.log(_B_of_mu_numpy(_MU_TABLE))
if not np.all(np.diff(_LOG_B_TABLE) > 0):  # pragma: no cover - import-time guard
    raise RuntimeError("phase: B(µ) is not monotone on the table grid")


def ff_mu_for_B(B):
    """The Junge slope ``µ`` whose Mobley-convention FF has backscatter ratio B.

    Linear interpolation in ``log B`` on a 4097-point monotone table, refined by
    Newton steps on ``log B(µ) − log B``. Differentiable with the exact
    implicit-function derivative ``dµ/dB = 1 / B'(µ)``: the iterations run on
    constants, and a final Newton step carries the gradient.

    Parameters
    ----------
    B : Array
        Particle backscatter ratio, within the table's range (~1.6e-5 to 0.40).
        Values outside it clamp to the table ends before refinement.

    Returns
    -------
    Array
        ``µ``, same shape as ``B``.
    """
    log_B = jnp.log(jnp.asarray(B))
    log_B_const = jax.lax.stop_gradient(log_B)

    def log_B_of(m):
        return jnp.log(_B_of_mu(m))

    dtype = log_B_const.dtype  # tables follow the caller's precision, silently
    mu = jnp.interp(
        log_B_const,
        jnp.asarray(_LOG_B_TABLE, dtype=dtype),
        jnp.asarray(_MU_TABLE, dtype=dtype),
    )
    for _ in range(_NEWTON_STEPS):
        # The closed form is elementwise, so one forward-mode JVP with a unit
        # tangent gives d log B / dµ for every element at once -- exactly.
        f, slope = jax.jvp(log_B_of, (mu,), (jnp.ones_like(mu),))
        mu = mu - (f - log_B_const) / slope
    # The value is converged; now attach the *implicit* derivative. Gradients
    # through unrolled Newton steps are only approximately dµ/dB (measured
    # 1.2e-5 relative), so the iterations above see constants, and one last
    # step carries the exact dµ/d log B = 1 / (d log B / dµ) at the root.
    mu = jax.lax.stop_gradient(mu)
    f, slope = jax.jvp(log_B_of, (mu,), (jnp.ones_like(mu),))
    return mu - (f - log_B) / jax.lax.stop_gradient(slope)


def ff_phase_for_B(psi_deg, B):
    """The Mobley-convention FF phase function with backscatter ratio ``B``.

    Parameters
    ----------
    psi_deg : Array
        Scattering angle (degrees).
    B : Array
        Backscatter ratio; broadcast against ``psi_deg``.

    Returns
    -------
    Array
        ``p̃(ψ; B)`` in sr^-1.
    """
    mu = ff_mu_for_B(B)
    return ff_phase(psi_deg, mobley_n(mu), mu)


def P_bb_ff(psi_deg, ff_B, ff_weight) -> Float[Array, "..."]:
    """Particulate backward phase function of a mixture of FF components.

    ``Pbb(ψ) = βp(ψ) / bbp = Σ_k w_k p̃(ψ; B_k) / Σ_k w_k B_k``,

    where ``w_k = b_k / Σ b`` is component ``k``'s share of particle
    scattering. This is ZTT's ``Pbb`` (Eq. 10) for L23's particles exactly: two
    FF components with B = 0.010 (phytoplankton) and 0.018 (non-algal).

    Parameters
    ----------
    psi_deg : Array
        Scattering angle (degrees), shape broadcasting against ``ff_B[..., 0]``.
    ff_B : Array
        Per-component backscatter ratios, **component axis last**.
    ff_weight : Array
        Per-component scattering fractions, component axis last; need not be
        normalized (only ratios enter). May be spectral.

    Returns
    -------
    Array
        ``Pbb(ψ)`` in sr^-1, the broadcast shape without the component axis.
    """
    psi = jnp.asarray(psi_deg)[..., None]
    B = jnp.asarray(ff_B)
    w = jnp.asarray(ff_weight)
    return jnp.sum(w * ff_phase_for_B(psi, B), axis=-1) / jnp.sum(w * B, axis=-1)
