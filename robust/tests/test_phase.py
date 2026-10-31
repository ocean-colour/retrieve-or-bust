"""
Tests for the Fournier–Forand phase functions and L23's FF mixture (M7 task 2,
``rt_fit_ztt.md``).

The task's gates:

- ``p̃FF`` integrates to one over the sphere (float64 quadrature);
- ``B → µ → B`` round-trips 0.010 and 0.018 (and the range PB24 needs);
- FF ``Pbb(180°)`` for both L23 components, reported against Sullivan's 0.153;
- ``ff_B=None`` -- or ``use_ff=False`` -- reproduces today's ``Pbb`` bit for bit;
- float64 ``jax.grad`` against central finite differences on the new path.

Plus the plumbing: ``PhaseParams`` validation, and the L23 loader filling the
two new fields from ``bph`` / ``bd``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import phase as P
from robust.rt import ztt as Z
from robust.rt.data import l23 as L
from robust.rt.types import PhaseParams
from robust.tests.conftest import L23_SMALL_FIXTURE, needs_l23

#: The two L23 components, in ``ff_B`` order.
B_PH, B_D = L.L23_FF_B


@functools.cache
def sphere_quadrature(n=1500, k=6):
    """Nodes ψ (rad) and weights for ``2π ∫ f(ψ) sin ψ dψ`` on ``[0, π]``.

    Substitutes ``ψ = π s^k``: FF goes as ψ^(µ−5) near 0, so ``p̃ sin ψ`` is
    integrable but singular (as ψ^-0.8 at B = 0.001). With k = 6 the
    transformed integrand vanishes at s = 0 for every µ > 3.17, which covers
    B ≥ 1e-3. Cached: ``leggauss`` is an n×n eigensolve.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (x + 1)
    ws = 0.5 * w
    psi = np.pi * s**k
    return psi, 2 * np.pi * np.sin(psi) * np.pi * k * s ** (k - 1) * ws


# --------------------------------------------------------------- FF itself --
@pytest.mark.parametrize("B", [0.001, B_PH, B_D, 0.035])
def test_ff_normalizes_and_backscatters_B(jax_x64, B):
    """Gate: ∫ p̃ dΩ = 1, and the backward half of it is the closed-form B."""
    psi, weight = sphere_quadrature()
    p = np.asarray(jax.jit(P.ff_phase_for_B)(np.rad2deg(psi), B))
    assert abs(np.sum(p * weight) - 1.0) < 1e-6
    # The backward hemisphere is smooth: plain Gauss-Legendre on [90°, 180°].
    x, w = np.polynomial.legendre.leggauss(200)
    back_psi = np.pi / 2 * (1.5 + 0.5 * x)
    back_w = np.pi / 4 * w * 2 * np.pi * np.sin(back_psi)
    p_back = np.asarray(jax.jit(P.ff_phase_for_B)(np.rad2deg(back_psi), B))
    assert np.sum(p_back * back_w) == pytest.approx(B, rel=1e-8)


@pytest.mark.parametrize("B", [1e-4, 0.001, B_PH, B_D, 0.035, 0.1])
def test_B_mu_round_trip(jax_x64, B):
    """Gate: ``B → µ → B`` is exact to float64 working precision."""
    mu = P.ff_mu_for_B(B)
    back = P.ff_backscatter_ratio(P.mobley_n(mu), mu)
    assert float(back) / B - 1 == pytest.approx(0.0, abs=1e-11)


def test_B_mu_round_trip_float32():
    """In the float32 default the round trip still holds to ~1e-5."""
    B = jnp.asarray([B_PH, B_D], dtype=jnp.float32)
    mu = P.ff_mu_for_B(B)
    back = P.ff_backscatter_ratio(P.mobley_n(mu), mu)
    np.testing.assert_allclose(np.asarray(back), [B_PH, B_D], rtol=2e-5)


def test_mobley_parameters_for_L23(jax_x64):
    """The (n, µ) pairs the convention assigns L23's components, pinned."""
    mu = np.asarray(P.ff_mu_for_B(jnp.asarray([B_PH, B_D])))
    np.testing.assert_allclose(mu, [3.470254, 3.580006], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(P.mobley_n(mu)), [1.082513, 1.099437], atol=1e-6
    )


def test_Pbb_180_against_sullivan(jax_x64):
    """Gate (reported): FF ``Pbb(180°)`` for both components vs Sullivan's 0.153.

    Measured 0.1506 (B = 0.010) and 0.1559 (B = 0.018) against Sullivan &
    Twardowski's 0.1529: within 2%, bracketing it. Pinned so a change shows.
    """
    one = jnp.ones(1)
    ph = float(P.P_bb_ff(180.0, jnp.asarray([B_PH]), one)[()])
    d = float(P.P_bb_ff(180.0, jnp.asarray([B_D]), one)[()])
    assert ph == pytest.approx(0.150619, abs=1e-5)
    assert d == pytest.approx(0.155883, abs=1e-5)
    sullivan = float(Z.P_bb_sullivan(180.0))
    assert ph < sullivan < d


def test_mixture_is_the_weighted_formula(jax_x64):
    """``P_bb_ff`` is Σ w p̃ / Σ w B, and one component is p̃ / B."""
    psi = jnp.asarray([120.0, 150.0, 180.0])
    single = P.P_bb_ff(psi, jnp.asarray([B_D]), jnp.ones(1))
    np.testing.assert_allclose(single, P.ff_phase_for_B(psi, B_D) / B_D, rtol=1e-12)

    w = jnp.asarray([0.3, 0.7])
    mixed = P.P_bb_ff(psi, jnp.asarray([B_PH, B_D]), w)
    expect = (0.3 * P.ff_phase_for_B(psi, B_PH) + 0.7 * P.ff_phase_for_B(psi, B_D)) / (
        0.3 * B_PH + 0.7 * B_D
    )
    np.testing.assert_allclose(mixed, expect, rtol=1e-12)
    # Weights need not be normalized.
    np.testing.assert_allclose(
        P.P_bb_ff(psi, jnp.asarray([B_PH, B_D]), 10 * w), mixed, rtol=1e-12
    )


def test_mixture_broadcasts_per_sample_and_wavelength():
    """L23 layout: ψ (n, 1), ff_B / ff_weight (n, W, 2) → Pbb (n, W)."""
    n, W = 4, 5
    psi = jnp.linspace(140.0, 180.0, n)[:, None]
    ff_B = jnp.broadcast_to(jnp.asarray([B_PH, B_D]), (n, W, 2))
    w = jnp.stack([jnp.linspace(0.1, 0.9, W)] * n)
    ff_w = jnp.stack([w, 1 - w], axis=-1)
    out = P.P_bb_ff(psi, ff_B, ff_w)
    assert out.shape == (n, W)
    # One element by hand.
    i, j = 2, 3
    by_hand = P.P_bb_ff(psi[i, 0], ff_B[i, j], ff_w[i, j])
    np.testing.assert_allclose(out[i, j], by_hand, rtol=1e-6)


# ------------------------------------------------------ ZTT's Pbb switch --
def l23_like_phase(n=3, W=4):
    """A PhaseParams with the FF mixture set, L23-shaped."""
    w = jnp.linspace(0.2, 0.8, W)
    ff_w = jnp.broadcast_to(jnp.stack([w, 1 - w], axis=-1), (n, W, 2))
    return PhaseParams(
        B_p=jnp.full((n, W), 0.014),
        ff_B=jnp.broadcast_to(jnp.asarray([B_PH, B_D]), (n, W, 2)),
        ff_weight=ff_w,
    )


def test_without_ff_B_pbb_is_bit_for_bit_sullivan():
    """Gate: ``ff_B=None`` reproduces today's Pbb exactly, with or without
    ``use_ff`` -- the silent fallback of Q9."""
    psi = jnp.linspace(135.0, 180.0, 7)[:, None]
    plain = PhaseParams(B_p=jnp.full((7, 1), 0.014))
    reference = Z.P_bb_sullivan(psi)
    for use_ff in (False, True):
        out = Z.P_bb_from_phase(plain, psi, use_ff=use_ff)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(reference))


def test_ff_fields_are_inert_unless_asked_for():
    """With ``ff_B`` set but ``use_ff=False`` (today's ``rrs_ZTT``), still
    Sullivan bit for bit: the loader filling the fields moves no number."""
    phase = l23_like_phase()
    psi = jnp.asarray([150.0, 160.0, 180.0])[:, None]
    np.testing.assert_array_equal(
        np.asarray(Z.P_bb_from_phase(phase, psi)),
        np.asarray(Z.P_bb_sullivan(psi)),
    )


def test_use_ff_selects_the_mixture():
    phase = l23_like_phase()
    psi = jnp.asarray([150.0, 160.0, 180.0])[:, None]
    out = Z.P_bb_from_phase(phase, psi, use_ff=True)
    np.testing.assert_allclose(
        out, P.P_bb_ff(psi, phase.ff_B, phase.ff_weight), rtol=1e-7
    )
    assert out.shape == (3, 4)


def test_ff_and_m5_tilt_are_alternatives():
    phase = PhaseParams(
        B_p=jnp.asarray(0.014),
        beta_tilde_pi=jnp.asarray(0.15),
        ff_B=jnp.asarray([B_PH, B_D]),
        ff_weight=jnp.asarray([0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="alternatives"):
        Z.P_bb_from_phase(phase, 180.0, use_ff=True)
    # Without use_ff the FF fields are ignored, so no conflict arises.
    Z.P_bb_from_phase(phase, 180.0)


def test_rrs_ztt_tt2017_ignores_ff_fields_l23_uses_them():
    """``variant="tt2017"`` is blind to the FF fields (bit for bit); the default
    ``"l23"`` variant uses them (M7 task 4)."""
    from robust.rt.types import Geometry, IOPs

    n, W = 3, 4
    wave = jnp.linspace(400.0, 700.0, W)
    iops = IOPs(
        a=jnp.full((n, W), 0.1),
        bb_w=jnp.full((n, W), 0.002),
        bb_p=jnp.full((n, W), 0.003),
    )
    geometry = Geometry.nadir(jnp.asarray([0.0, 30.0, 60.0]))
    with_ff = l23_like_phase(n, W)
    without = PhaseParams(B_p=with_ff.B_p)
    np.testing.assert_array_equal(
        np.asarray(Z.rrs_ZTT(iops, with_ff, geometry, wave, variant="tt2017")),
        np.asarray(Z.rrs_ZTT(iops, without, geometry, wave, variant="tt2017")),
    )
    assert not np.array_equal(
        np.asarray(Z.rrs_ZTT(iops, with_ff, geometry, wave)),
        np.asarray(Z.rrs_ZTT(iops, without, geometry, wave)),
    )


def test_ff_path_jits():
    phase = l23_like_phase()
    psi = jnp.asarray([150.0, 160.0, 180.0])[:, None]
    f = jax.jit(lambda p, s: Z.P_bb_from_phase(p, s, use_ff=True))
    # float32: XLA fuses differently under jit, so agreement is ~1e-6, not bits.
    np.testing.assert_allclose(
        f(phase, psi), Z.P_bb_from_phase(phase, psi, use_ff=True), rtol=1e-5
    )


# -------------------------------------------------------------- gradients --
def central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_grad_mu_for_B_matches_fd(jax_x64):
    """Gate: dµ/dB through the table + Newton inversion."""
    for B in (B_PH, B_D):
        g = float(jax.grad(P.ff_mu_for_B)(B))
        fd = float(central_fd(P.ff_mu_for_B, B, 1e-7))
        assert g == pytest.approx(fd, rel=1e-6)


def test_grad_pbb_ff_matches_fd(jax_x64):
    """Gate: ∂Pbb/∂(w_ph, B_ph, B_d, ψ) on the FF path, float64."""
    psi0, w0 = 155.0, 0.35

    def pbb(psi, w, b_ph, b_d):
        return P.P_bb_ff(psi, jnp.stack([b_ph, b_d]), jnp.stack([w, 1.0 - w]))

    args = (psi0, w0, B_PH, B_D)
    steps = (1e-4, 1e-6, 1e-8, 1e-8)
    for k, h in enumerate(steps):
        g = float(jax.grad(pbb, argnums=k)(*args))

        def along(x, k=k):
            changed = list(args)
            changed[k] = x
            return pbb(*changed)

        fd = float(central_fd(along, args[k], h))
        assert g == pytest.approx(fd, rel=1e-5, abs=1e-12), k


# ------------------------------------------------------------- validation --
def test_phase_params_validate_ff():
    ok = l23_like_phase()
    ok.validate()
    with pytest.raises(ValueError, match="together"):
        PhaseParams(B_p=jnp.asarray(0.01), ff_B=jnp.asarray([0.01])).validate()
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        PhaseParams(
            B_p=jnp.asarray(0.01), ff_B=jnp.asarray([0.0]), ff_weight=jnp.ones(1)
        ).validate()
    with pytest.raises(ValueError, match=">= 0"):
        PhaseParams(
            B_p=jnp.asarray(0.01), ff_B=jnp.asarray([0.01]), ff_weight=-jnp.ones(1)
        ).validate()
    with pytest.raises(ValueError, match="broadcast|component"):
        PhaseParams(
            B_p=jnp.asarray(0.01), ff_B=jnp.asarray([0.01, 0.02]), ff_weight=jnp.ones(3)
        ).validate()


# ----------------------------------------------------------------- loader --
def reader_with_components(n_scene=3, W=81):
    """A load_batch reader serving the raw fields plus bph / bd."""
    wave = np.arange(350.0, 751.0, 5.0)
    rng = np.random.default_rng(0)

    def read(x, zenith):
        bph = rng.uniform(0.0, 0.2, (n_scene, W))
        bph[0, :3] = 0.0  # L23 has scenes with no phytoplankton scattering
        bd = rng.uniform(0.01, 0.2, (n_scene, W))
        bnw = bph + bd
        bbnw = B_PH * bph + B_D * bd
        a = np.full((n_scene, W), 0.1)
        return {
            "wave": wave,
            "Rrs": np.full((n_scene, W), 0.004),
            "a": a,
            "bb": bbnw + 0.002,
            "bbnw": bbnw,
            "bnw": bnw,
            "bph": bph,
            "bd": bd,
        }

    return read


def test_loader_fills_ff_fields_from_components():
    batch = L.load_batch(reader=reader_with_components())
    pp = batch.phase_params
    assert pp.ff_B.shape == pp.ff_weight.shape == (9, 81, 2)
    # Stored at the JAX default dtype (float32 here).
    np.testing.assert_allclose(np.asarray(pp.ff_B[..., 0]), B_PH, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(pp.ff_B[..., 1]), B_D, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(pp.ff_weight.sum(-1)), 1.0, rtol=1e-6)
    # Σ w B is the bulk B_p the loader already computes -- one consistent story.
    np.testing.assert_allclose(
        np.asarray((pp.ff_weight * pp.ff_B).sum(-1)), np.asarray(pp.B_p), rtol=1e-6
    )
    pp.validate()


def test_select_carries_ff_fields():
    batch = L.load_batch(reader=reader_with_components())
    mask = np.zeros(batch.n_sample, dtype=bool)
    mask[[1, 5]] = True
    sub = L.select(batch, mask)
    np.testing.assert_array_equal(
        np.asarray(sub.phase_params.ff_weight),
        np.asarray(batch.phase_params.ff_weight)[mask],
    )


def test_fixture_batch_has_no_ff_fields():
    """The hash-pinned fixture predates bph/bd: its batches carry no FF info."""
    if not L23_SMALL_FIXTURE.is_file():
        pytest.skip(f"fixture missing: {L23_SMALL_FIXTURE}")
    batch = L.load_batch(reader=L.npz_reader(L23_SMALL_FIXTURE))
    assert batch.phase_params.ff_B is None
    assert batch.phase_params.ff_weight is None


@needs_l23
def test_full_release_ff_weights():
    """On the release, the FF mixture reproduces L23's bulk B_p to float32
    storage precision (bbnw is stored, not recomputed)."""
    batch = L.load_batch(zeniths=(0,))
    pp = batch.phase_params
    implied = np.asarray((pp.ff_weight * pp.ff_B).sum(-1))
    np.testing.assert_allclose(implied, np.asarray(pp.B_p), rtol=1e-3)
    assert float(pp.ff_weight[..., 0].min()) == 0.0  # pure-NAP scenes exist
