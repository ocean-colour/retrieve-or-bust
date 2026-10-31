"""
The asymptotic average cosine µ∞ as an eigenvalue of the RTE (M7 task 3).

**Fitting-only, NumPy, never on the differentiable path.** This module computes
the *target* that ZTT's Equation (8) is fitted to; the model itself evaluates
the fitted polynomial (:func:`robust.rt.ztt.mu_infinity`).

Why an eigenproblem (``rt_fit_ztt.md``, Q13, answer (a)). µ∞ is an IOP: deep
in a homogeneous ocean the radiance distribution relaxes to a shape that
depends only on absorption, scattering and the phase function, and decays as
``e^{−K∞ z}``. M7 task 1 found L23's 50 m profiles reach that regime on only
2.4% of (scene, λ), so the profiles cannot supply µ∞ over L23's domain. The
asymptotic mode can be computed directly instead.

Azimuthally averaging the RTE and writing ``L(z, µ) = f(µ) e^{−K z}``:

    (c I − b P̃ W) f = K diag(µ) f

on a double-Gauss grid ``µ_i`` with weights ``w_i`` (``W = 2π diag(w)``), where
``P̃_ij`` is the azimuth-averaged phase function. ``K∞`` is the smallest positive
eigenvalue; by Gershun's law ``µ∞ = a / K∞``.

Numerics, each choice tested in ``robust/tests/test_asymptotic.py``:

- **δ-truncation of the forward peak** inside :data:`TRUNCATION_DEG`.
  Fournier–Forand goes as ψ^(µ−5) near 0, which no fixed angular quadrature
  resolves. The truncated fraction ``F`` is moved into a δ-function, i.e.
  removed from ``b`` (``b* = (1 − F) b``). Exact forward scattering does not
  alter radiance, so K∞ is unchanged up to the truncation's angular width,
  which is why the angle must stay small (:data:`TRUNCATION_DEG`).
- **Energy renormalization** of each kernel row: the discrete kernel's missing
  or extra mass goes onto the diagonal (forward scattering on the grid).
- **Component kernels are precomputed once.** L23's three scatterers (water,
  phytoplankton FF B = 0.010, non-algal FF B = 0.018) have fixed shapes, so a
  point's kernel is a linear combination of three matrices. The ~269k L23
  eigenproblems then cost only the solves.

Measured: against the 2.4% of L23 points where the profiles *are*
asymptotic, µ∞ comes out about **1.3% high** (task 1's prototype, at 5°
truncation, gave 1.4%), with ±0.3% scatter. The offset doesn't change with the
stream count or water depolarization, and it is carried as a stated systematic
(Q13). ``design/ztt_l23/metrics.md`` has the current figure.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import jax
import numpy as np
from scipy.integrate import quad

from . import phase as _phase
from . import ztt as _ztt

__all__ = [
    "N_STREAMS",
    "TRUNCATION_DEG",
    "N_PHI",
    "Kernel",
    "double_gauss",
    "water_phase",
    "ff_phase_numpy",
    "build_kernel",
    "l23_kernels",
    "mu_infinity_eigen",
]

#: Streams in µ (double-Gauss, half on each hemisphere). 80 → 160 moves µ∞ by
#: ≤ 1.7e-4 on L23 points. The known weak spot is low single-scattering albedo:
#: against the exact isotropic solution 80 streams are -7e-4 at ω₀ = 0.1 and
#: exact to rounding for ω₀ ≥ 0.5 (``test_asymptotic.py``).
N_STREAMS = 80

#: Forward-peak δ-truncation angle (degrees). Converged: on 300 random L23
#: points at 80 streams, 0°, 1° and 2° agree with a 160-stream 1° reference to
#: ≤ 1.7e-4, while 5° biases µ∞ high by 8e-4 (median) and 10° by ~3.5e-3. The
#: truncated band is not exactly forward scattering, so it must stay narrow.
TRUNCATION_DEG = 2.0

#: Azimuth samples for the kernel average (midpoint rule, periodic, so
#: spectrally accurate for a smooth integrand).
N_PHI = 2048

#: Points per batched eigensolve.
_CHUNK = 4096


@contextlib.contextmanager
def _float64():
    """Run JAX in float64 for the enclosed block, restoring the prior setting.

    The FF functions live in :mod:`robust.rt.phase` (JAX, float32 by default).
    Building a fitting target in float32 would make it storage-precision
    dependent, so they are evaluated in float64 here.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def double_gauss(n_streams: int = N_STREAMS):
    """Double-Gauss nodes and weights in µ on ``[−1, 1]``.

    Gauss–Legendre on each hemisphere separately, so no node sits at µ = 0
    (the eigenproblem divides by µ) and the two hemispheres integrate exactly.

    Returns
    -------
    mu, weight : numpy.ndarray
        Each ``(n_streams,)``, ascending µ.
    """
    if n_streams % 2:
        raise ValueError("double_gauss: n_streams must be even")
    x, w = np.polynomial.legendre.leggauss(n_streams // 2)
    up, w_up = 0.5 * (x + 1.0), 0.5 * w
    return np.concatenate([-up[::-1], up]), np.concatenate([w_up[::-1], w_up])


def water_phase(depolarization: float = _ztt.WATER_DEPOLARIZATION):
    """Pure-water phase function ``p̃_w(ψ) = (1 + f cos²ψ) / (4π (1 + f/3))``.

    ``f = (1 − δ)/(1 + δ)``. L23 computes water scattering after Zhang et al.
    (2009) (Loisel et al. 2023 §2), whose depolarization ratio is
    ``ztt.WATER_DEPOLARIZATION``, the same δ ZTT's ``βw`` uses.

    Returns
    -------
    callable
        ``psi_rad -> p̃`` (sr^-1), NumPy.
    """
    f = (1.0 - depolarization) / (1.0 + depolarization)
    return lambda psi: (1.0 + f * np.cos(psi) ** 2) / (4.0 * np.pi * (1.0 + f / 3.0))


def ff_phase_numpy(B: float):
    """The Mobley-convention FF phase function for backscatter ratio ``B``, as
    a float64 NumPy callable of ψ in radians (wraps :mod:`robust.rt.phase`)."""
    with _float64():
        mu = float(_phase.ff_mu_for_B(B))
        n = float(_phase.mobley_n(mu))

    def p(psi):
        with _float64():
            return np.asarray(_phase.ff_phase(np.rad2deg(psi), n, mu), dtype=float)

    return p


@dataclass(frozen=True)
class Kernel:
    """One scatterer's discretized phase function.

    Attributes
    ----------
    matrix : numpy.ndarray
        ``P̃_ij`` (sr^-1), azimuth-averaged, energy-renormalized,
        ``(n_streams, n_streams)``.
    truncated : float
        Fraction ``F`` of scattering moved into the forward δ.
    """

    matrix: np.ndarray
    truncated: float


def build_kernel(
    p,
    mu: np.ndarray,
    weight: np.ndarray,
    *,
    truncation_deg: float = TRUNCATION_DEG,
    n_phi: int = N_PHI,
) -> Kernel:
    """Azimuth-averaged, δ-truncated, renormalized kernel of phase function ``p``.

    Parameters
    ----------
    p : callable
        ``psi_rad -> p̃`` (sr^-1), normalized over the sphere.
    mu, weight : numpy.ndarray
        The stream grid (:func:`double_gauss`).
    truncation_deg : float, optional
        δ-truncation angle; 0 disables it.
    n_phi : int, optional
        Azimuth samples.

    Returns
    -------
    Kernel
    """
    psic = np.deg2rad(truncation_deg)
    if psic > 0:
        inside = (
            2
            * np.pi
            * quad(
                lambda t: float(p(np.asarray(t))) * np.sin(t),
                0.0,
                psic,
                limit=400,
                points=[1e-6, 1e-4, 1e-2],
            )[0]
        )
        flat_value = float(p(np.asarray(psic)))
        flat = flat_value * 2 * np.pi * (1.0 - np.cos(psic))
        truncated = inside - flat

        def kept(psi):
            return np.where(psi < psic, flat_value, p(np.maximum(psi, psic))) / (
                1.0 - truncated
            )
    else:
        truncated, kept = 0.0, p

    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    sin_mu = np.sqrt(1.0 - mu**2)
    matrix = np.empty((mu.size, mu.size))
    for i in range(mu.size):  # row by row: (n, n_phi) at a time, not (n, n, n_phi)
        cos_t = mu[i] * mu[:, None] + sin_mu[i] * sin_mu[:, None] * np.cos(phi)
        matrix[i] = kept(np.arccos(np.clip(cos_t, -1.0, 1.0))).mean(axis=1)
    row_sum = 2 * np.pi * (matrix * weight[None, :]).sum(axis=1)
    matrix += np.diag((1.0 - row_sum) / (2 * np.pi * weight))
    return Kernel(matrix=matrix, truncated=float(truncated))


def l23_kernels(
    n_streams: int = N_STREAMS,
    *,
    truncation_deg: float = TRUNCATION_DEG,
    n_phi: int = N_PHI,
):
    """The stream grid and the three L23 scatterers' kernels.

    Returns
    -------
    mu, weight : numpy.ndarray
    kernels : tuple of Kernel
        Water, phytoplankton (FF, B = 0.010), non-algal (FF, B = 0.018) -- the
        order of :func:`mu_infinity_eigen`'s ``b`` components.
    """
    from .data import l23

    mu, weight = double_gauss(n_streams)
    shapes = (water_phase(), *(ff_phase_numpy(B) for B in l23.L23_FF_B))
    kernels = tuple(
        build_kernel(p, mu, weight, truncation_deg=truncation_deg, n_phi=n_phi)
        for p in shapes
    )
    return mu, weight, kernels


def mu_infinity_eigen(a, b, mu, weight, kernels) -> np.ndarray:
    """µ∞ = a / K∞ from the asymptotic eigenproblem, for many points at once.

    Parameters
    ----------
    a : array_like
        Absorption (m^-1), any shape ``S``.
    b : array_like
        Scattering per scatterer (m^-1), shape ``S + (n_kernel,)``, in the
        order of ``kernels``.
    mu, weight : numpy.ndarray
        The stream grid the kernels were built on.
    kernels : sequence of Kernel

    Returns
    -------
    numpy.ndarray
        µ∞, shape ``S``, float64.

    Raises
    ------
    ValueError
        On a shape mismatch, or if any point has no positive real eigenvalue
        (which a physical, absorbing medium always has).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.shape != a.shape + (len(kernels),):
        raise ValueError(
            f"mu_infinity_eigen: b {b.shape} should be a {a.shape} + ({len(kernels)},)"
        )
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1, len(kernels))
    keep = np.array([1.0 - k.truncated for k in kernels])
    b_star = flat_b * keep  # δ-truncated scattering
    c_star = flat_a + b_star.sum(axis=1)
    # Each kernel pre-multiplied by the quadrature weight, ready to scale.
    scaled = np.stack([2 * np.pi * k.matrix * weight[None, :] for k in kernels])
    inv_mu = 1.0 / mu

    out = np.empty(flat_a.size)
    eye = np.eye(mu.size)
    for start in range(0, flat_a.size, _CHUNK):
        stop = min(start + _CHUNK, flat_a.size)
        S = np.einsum("pk,kij->pij", b_star[start:stop], scaled)
        A = (c_star[start:stop, None, None] * eye - S) * inv_mu[None, :, None]
        ev = np.linalg.eigvals(A)
        real_positive = np.where(
            (ev.real > 0) & (np.abs(ev.imag) <= 1e-9 * np.abs(ev.real)),
            ev.real,
            np.inf,
        )
        K = real_positive.min(axis=1)
        if not np.all(np.isfinite(K)):
            raise ValueError("mu_infinity_eigen: no positive real eigenvalue")
        out[start:stop] = flat_a[start:stop] / K
    return out.reshape(a.shape)
