"""
Tests for :mod:`robust.rt.asymptotic` -- µ∞ from the asymptotic RTE eigenproblem
(M7 task 3, ``rt_fit_ztt.md`` Q13).

The solver is checked against physics with closed-form answers before it is
trusted on L23:

- **isotropic scattering**: K∞ solves the classical transcendental equation
  ``(ω₀ / 2k) ln[(1 + k)/(1 − k)] = 1``, ``k = K∞/c``;
- **no scattering**: the slowest-decaying mode is the most vertical stream, so
  µ∞ → the largest node;
- **δ-truncation invariance**: moving the truncation angle barely moves µ∞;
- and, on the release, the measured ~+1.3% systematic against the L23 profiles
  where those are asymptotic is pinned, so a change in it cannot go unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from robust.rt import asymptotic as A
from robust.tests.conftest import needs_l23_profiles


def isotropic(psi):
    return np.full_like(np.asarray(psi, dtype=float), 1.0 / (4.0 * np.pi))


@pytest.fixture(scope="module")
def grid():
    return A.double_gauss(40)


@pytest.fixture(scope="module")
def l23_kernels_40():
    """The L23 kernels at 40 streams -- the task-1 sweep showed < 1e-4 change
    from 80, and it keeps the module fast."""
    return A.l23_kernels(40)


def test_double_gauss(grid):
    mu, w = grid
    assert mu.size == 40 and np.all(np.diff(mu) > 0)
    assert np.sum(w) == pytest.approx(2.0, abs=1e-13)
    np.testing.assert_allclose(mu, -mu[::-1], atol=1e-15)
    assert not np.any(mu == 0)
    with pytest.raises(ValueError, match="even"):
        A.double_gauss(41)


@pytest.mark.parametrize("which", ["water", "ff_ph", "ff_d"])
def test_kernels_conserve_energy(l23_kernels_40, which):
    mu, w, kernels = l23_kernels_40
    k = kernels[["water", "ff_ph", "ff_d"].index(which)]
    np.testing.assert_allclose(2 * np.pi * k.matrix @ w, 1.0, atol=1e-12)


def test_truncated_fractions(l23_kernels_40):
    """Water is not truncated; the FF forward peaks hold most of the scattering
    within 2° (measured 0.39 / 0.28; the more forward-peaked phytoplankton FF
    the more)."""
    _, _, (water, ph, d) = l23_kernels_40
    # Water is smooth: "truncating" it moves only its curvature, ~5e-6.
    assert abs(water.truncated) < 1e-5
    assert 0.3 < ph.truncated < 0.5
    assert 0.2 < d.truncated < ph.truncated


#: Discrete-ordinate error at the production stream count, by albedo: exact to
#: rounding for ω₀ ≥ 0.5, and -7e-4 / -3e-4 at ω₀ = 0.1 / 0.2. There the
#: asymptotic mode crowds towards µ = 1, between the last node and the pole.
#: Measured, and pinned so a regression shows; 16% of L23 points have
#: ω₀ < 0.09, which puts a ≲ 0.1% low bias on their µ∞ target.
ISOTROPIC_TOLERANCE = {0.1: 1e-3, 0.2: 4e-4, 0.5: 1e-9, 0.8: 1e-9, 0.95: 1e-9}


def exact_isotropic_mu_inf(omega0):
    """µ∞ for isotropic scattering, c = 1: solves (ω₀/2k) ln[(1+k)/(1−k)] = 1
    in t = ln(1 − k), which stays well conditioned as k → 1."""

    def f(t):
        return omega0 / (2 * (1 - np.exp(t))) * (np.log(2 - np.exp(t)) - t) - 1

    k = 1 - np.exp(brentq(f, -700.0, -1e-12))
    return (1 - omega0) / k


@pytest.mark.parametrize("omega0", sorted(ISOTROPIC_TOLERANCE))
def test_isotropic_matches_the_transcendental_equation(omega0):
    mu, w = A.double_gauss(A.N_STREAMS)
    kernel = A.build_kernel(isotropic, mu, w, truncation_deg=0.0)
    got = A.mu_infinity_eigen(
        np.asarray(1 - omega0), np.asarray([omega0]), mu, w, (kernel,)
    )
    expect = exact_isotropic_mu_inf(omega0)
    assert abs(float(got) / expect - 1) < ISOTROPIC_TOLERANCE[omega0]


def test_no_scattering_is_the_most_vertical_stream(grid):
    mu, w = grid
    kernel = A.build_kernel(isotropic, mu, w, truncation_deg=0.0)
    got = A.mu_infinity_eigen(np.asarray(0.1), np.asarray([0.0]), mu, w, (kernel,))
    assert float(got) == pytest.approx(mu.max(), rel=1e-12)


def test_truncation_converged_at_the_production_angle():
    """At 80 streams, 1° and the production 2° agree to < 3e-4 on an L23-like
    point (the convergence study found ≤ 1.7e-4 over 300 points). 8° does not
    -- the truncated band is not exactly forward -- which is why 2° is used."""
    mu, w = A.double_gauss(A.N_STREAMS)
    shapes = (A.water_phase(), A.ff_phase_numpy(0.010))
    a, b = np.asarray(0.05), np.asarray([0.003, 0.2])
    values = {}
    for angle in (1.0, A.TRUNCATION_DEG, 8.0):
        kernels = tuple(A.build_kernel(p, mu, w, truncation_deg=angle) for p in shapes)
        values[angle] = float(A.mu_infinity_eigen(a, b, mu, w, kernels))
    assert values[A.TRUNCATION_DEG] == pytest.approx(values[1.0], rel=3e-4)
    assert abs(values[8.0] / values[1.0] - 1) > 1e-3


def test_vectorized_and_deterministic(l23_kernels_40):
    mu, w, kernels = l23_kernels_40
    rng = np.random.default_rng(0)
    a = rng.uniform(0.01, 1.0, (3, 5))
    b = rng.uniform(0.001, 0.5, (3, 5, 3))
    out = A.mu_infinity_eigen(a, b, mu, w, kernels)
    assert out.shape == (3, 5)
    assert np.all((out > 0) & (out <= 1))
    one = A.mu_infinity_eigen(a[1, 2], b[1, 2], mu, w, kernels)
    assert float(one) == out[1, 2]
    np.testing.assert_array_equal(A.mu_infinity_eigen(a, b, mu, w, kernels), out)


def test_more_scattering_lowers_mu_inf(l23_kernels_40):
    mu, w, kernels = l23_kernels_40
    b = np.asarray([[0.003, 0.0, s] for s in (0.01, 0.1, 1.0)])
    out = A.mu_infinity_eigen(np.full(3, 0.05), b, mu, w, kernels)
    assert np.all(np.diff(out) < 0)


def test_shape_mismatch_raises(l23_kernels_40):
    mu, w, kernels = l23_kernels_40
    with pytest.raises(ValueError, match="should be"):
        A.mu_infinity_eigen(np.ones(4), np.ones((4, 2)), mu, w, kernels)


@needs_l23_profiles
def test_offset_against_asymptotic_profiles_is_pinned(l23_kernels_40):
    """The ~+1.3% systematic of Q13, on 300 profile-asymptotic points."""
    from ocpy.hydrolight import loisel23

    from robust.rt.data import l23 as L

    product = L.build_profile_product()
    ds = loisel23.load_ds(1, 0)
    a, b, bnw, bph, bd = (
        np.asarray(ds[v].data, dtype=float) for v in ("a", "b", "bnw", "bph", "bd")
    )
    ok = np.argwhere(product.flag == L.FLAG_ASYMPTOTIC)
    pick = ok[np.random.default_rng(0).choice(len(ok), 300, replace=False)]
    s, k = pick[:, 0], pick[:, 1]
    mu, w, kernels = l23_kernels_40
    got = A.mu_infinity_eigen(
        a[s, k], np.stack([(b - bnw)[s, k], bph[s, k], bd[s, k]], -1), mu, w, kernels
    )
    ratio = got / product.mu_inf[s, k]
    assert 1.008 < np.median(ratio) < 1.017
    assert np.percentile(ratio, 84) - np.percentile(ratio, 16) < 0.01
