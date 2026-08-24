"""
Tests for :mod:`robust.rt.inelastic_corr` — M3 task 1, the head machinery.

No trained weights exist yet (task 2), so what this module gates is the
*machinery*, on the properties that must hold regardless of training:

- **Zero-init is the analytic model.** A fresh head's δ is exactly 0
  (zero-initialised output layer), so the corrected forward equals the
  analytic forward — training starts from the physics.
- **The bound binds.** |δ| < ``delta_max`` whatever the parameters.
- **δ_F cannot see φ_C.** Its feature list has no yield column, and the
  composed forward stays exactly linear in φ_C with a (randomised) head
  active — the design §4.4 promise, now enforced against the M3 code.
- **The fallback warns once, then behaves.** Absent weight files:
  ``corrections=None`` degrades to analytic-only behind a single
  ``MissingCorrectionWarning``; ``corrections=False`` is the same numbers,
  silent.
- **Round trip + refusal.** ``save_head``/``load_head`` reproduce the head;
  a file whose feature list mismatches the code is refused, not warned.

The held-out ≤ 5 % gates arrive with task 3, against the *committed*
weights.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import hybrid as H
from robust.rt import inelastic as I
from robust.rt import inelastic_corr as IC
from robust.rt.types import Inelastic

#: One fixed configuration per head kind, small and fast.
RAMAN_CONFIG = IC.HeadConfig("raman", delta_max=1.0)
FL_CONFIG = IC.HeadConfig("fl", delta_max=0.5)


def randomized(head, scale=3.0, seed=7):
    """The head with its parameters replaced by noise — a nonzero δ.

    ``init_head`` is deliberately a zero function, so bound/effect tests
    need parameters that actually fire; scale 3 pushes the tanh well into
    its nonlinear range.
    """
    leaves, treedef = jax.tree_util.tree_flatten(head.params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    noisy = [
        scale * jax.random.normal(k, leaf.shape, leaf.dtype)
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    return dataclasses.replace(
        head, params=jax.tree_util.tree_unflatten(treedef, noisy)
    )


def batch_args(batch):
    return (batch.iops, batch.phase_params, batch.geometry, batch.wave)


# ---------------------------------------------------------------- features ----


def test_feature_lists_are_the_designs():
    """§4.5 features, in order — and no φ_C anywhere near δ_F."""
    assert IC.RAMAN_FEATURES == (
        "log10_a_em",
        "log10_bb_em",
        "log10_a_ex",
        "log10_bb_ex",
        "cos_theta_s",
        "wave",
    )
    assert IC.FL_FEATURES == (
        "log10_a_ph440",
        "log10_a_em",
        "log10_bb_em",
        "log10_a_490",
        "cos_theta_s",
        "wave",
    )
    assert not any("phi" in name.lower() for name in IC.FL_FEATURES)


def test_features_shapes_and_finiteness(l23_small_inelastic_batch):
    """(..., n_wave, 6) per head, finite everywhere on the fixture."""
    batch = l23_small_inelastic_batch
    for fn in (IC.features_raman, IC.features_fl):
        x = np.asarray(fn(batch.iops, batch.geometry, batch.wave))
        assert x.shape == (*np.asarray(batch.iops.a).shape, 6)
        assert np.all(np.isfinite(x))


def test_features_fl_requires_aph(l23_small_inelastic_batch):
    """The same physical requirement as the kernel, same loud error."""
    batch = l23_small_inelastic_batch
    iops = dataclasses.replace(batch.iops, a_ph=None)
    with pytest.raises(ValueError, match="a_ph"):
        IC.features_fl(iops, batch.geometry, batch.wave)


# ------------------------------------------------------------------- heads ----


@pytest.mark.parametrize("config", [RAMAN_CONFIG, FL_CONFIG], ids=["raman", "fl"])
def test_fresh_head_is_exactly_zero(config, l23_small_inelastic_batch):
    """Zero-init output layer → δ ≡ 0: training starts from the physics."""
    batch = l23_small_inelastic_batch
    head = IC.init_head(config.kind, config)
    delta = np.asarray(head.delta(batch.iops, batch.geometry, batch.wave))
    assert np.all(delta == 0.0)


@pytest.mark.parametrize("config", [RAMAN_CONFIG, FL_CONFIG], ids=["raman", "fl"])
def test_delta_is_bounded(config, l23_small_inelastic_batch):
    """|δ| < delta_max for arbitrary (noise) parameters — the tanh bound."""
    batch = l23_small_inelastic_batch
    head = randomized(IC.init_head(config.kind, config))
    delta = np.asarray(head.delta(batch.iops, batch.geometry, batch.wave))
    assert np.all(np.isfinite(delta))
    assert np.abs(delta).max() < config.delta_max
    assert np.abs(delta).max() > 0.0  # the noise actually fired


def test_corrected_factor_form(l23_small_inelastic_batch):
    """f_R = 1 + (f_phys − 1)(1 + δ): identity at δ=0, →1 with the increment."""
    batch = l23_small_inelastic_batch
    f_phys = I.raman_factor(batch.iops, batch.geometry, batch.wave)
    assert np.array_equal(
        np.asarray(IC.corrected_raman_factor(jnp.zeros(()), f_phys)),
        np.asarray(f_phys),
    )
    # Where the increment is zero, no delta can move the factor off 1.
    ones = jnp.ones_like(f_phys)
    corrected = IC.corrected_raman_factor(jnp.full((), 0.9), ones)
    assert np.array_equal(np.asarray(corrected), np.asarray(ones))


# ------------------------------------------------------------------ wiring ----


def test_zero_heads_equal_analytic_forward(l23_small_inelastic_batch):
    """forward with fresh heads == forward with corrections=False.

    rtol 1e-6, not exact: (f−1)+1 == f is exact only below 2 (Sterbenz),
    and the fixture's Raman factor stays under 1.4 — but the guarantee
    worth pinning is agreement, not bit-identity.
    """
    batch = l23_small_inelastic_batch
    heads = IC.CorrectionHeads(
        raman=IC.init_head("raman", RAMAN_CONFIG), fl=IC.init_head("fl", FL_CONFIG)
    )
    analytic = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=False,
            check_domain=False,
        )
    )
    corrected = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=heads,
            check_domain=False,
        )
    )
    np.testing.assert_allclose(corrected, analytic, rtol=1e-6)


def test_random_heads_move_the_right_bands(l23_small_inelastic_batch):
    """Each head reaches its own term: Raman across the spectrum, fl at 685."""
    batch = l23_small_inelastic_batch
    wave = np.asarray(batch.wave)
    i550 = int(np.abs(wave - 550.0).argmin())
    i685 = int(np.abs(wave - 685.0).argmin())
    analytic = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=False,
            check_domain=False,
        )
    )

    raman_only_head = IC.CorrectionHeads(
        raman=randomized(IC.init_head("raman", RAMAN_CONFIG))
    )
    moved = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=raman_only_head,
            check_domain=False,
        )
    )
    assert not np.allclose(moved[:, i550], analytic[:, i550], rtol=1e-4)

    fl_only_head = IC.CorrectionHeads(fl=randomized(IC.init_head("fl", FL_CONFIG)))
    moved = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=fl_only_head,
            check_domain=False,
        )
    )
    assert not np.allclose(moved[:, i685], analytic[:, i685], rtol=1e-4)
    # ... and only the additive line: far from 685 the fl head is inert.
    np.testing.assert_allclose(moved[:, i550], analytic[:, i550], rtol=1e-6)


def test_phi_C_linearity_survives_a_live_fl_head(l23_small_inelastic_batch):
    """The §4.4 promise against the M3 code: δ_F never breaks the yield.

    With a randomised (nonzero) fl head active, the fluorescence term is
    still exactly proportional to φ_C — doubling the yield doubles the
    delta over the raman-only model.
    """
    batch = l23_small_inelastic_batch
    heads = IC.CorrectionHeads(fl=randomized(IC.init_head("fl", FL_CONFIG)))
    base = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(fluorescence=False),
            corrections=False,
            check_domain=False,
        )
    )

    def with_phi(phi):
        return np.asarray(
            H.forward(
                *batch_args(batch),
                inelastic=Inelastic(phi_C=jnp.asarray(phi)),
                corrections=heads,
                check_domain=False,
            )
        )

    i685 = int(np.abs(np.asarray(batch.wave) - 685.0).argmin())
    delta_1x = with_phi(0.02)[:, i685] - base[:, i685]
    delta_2x = with_phi(0.04)[:, i685] - base[:, i685]
    np.testing.assert_allclose(delta_2x, 2.0 * delta_1x, rtol=1e-4)


def test_gradients_flow_through_heads(l23_small_inelastic_batch):
    """grad w.r.t. head parameters and w.r.t. inputs, both finite.

    The parameter gradient is what task 2 trains on; the input gradient is
    the inversion's path through the corrected model.
    """
    batch = l23_small_inelastic_batch
    row = jax.tree_util.tree_map(lambda x: x[75:76], batch.iops)
    pp = jax.tree_util.tree_map(lambda x: x[75:76], batch.phase_params)
    geom = jax.tree_util.tree_map(lambda x: x[75:76], batch.geometry)
    heads = IC.CorrectionHeads(
        raman=randomized(IC.init_head("raman", RAMAN_CONFIG)),
        fl=randomized(IC.init_head("fl", FL_CONFIG)),
    )

    def scalar_of_heads(hd):
        return H.rrs_forward(
            row,
            pp,
            geom,
            batch.wave,
            inelastic=Inelastic(),
            corrections=hd,
            check_domain=False,
        ).sum()

    g_heads = jax.grad(scalar_of_heads)(heads)
    for leaf in jax.tree_util.tree_leaves(g_heads):
        assert np.all(np.isfinite(np.asarray(leaf)))

    g_iops = jax.grad(
        lambda io: H.rrs_forward(
            io,
            pp,
            geom,
            batch.wave,
            inelastic=Inelastic(),
            corrections=heads,
            check_domain=False,
        ).sum()
    )(row)
    for leaf in (g_iops.a, g_iops.bb_p, g_iops.a_ph):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_corrected_forward_jits(l23_small_inelastic_batch):
    """The corrected path compiles; eager and compiled agree."""
    batch = l23_small_inelastic_batch
    heads = IC.CorrectionHeads(
        raman=randomized(IC.init_head("raman", RAMAN_CONFIG)),
        fl=randomized(IC.init_head("fl", FL_CONFIG)),
    )

    def f(iops, pp, geom, hd):
        return H.rrs_forward(
            iops,
            pp,
            geom,
            batch.wave,
            "ztt",
            inelastic=Inelastic(),
            corrections=hd,
        )

    eager = np.asarray(f(*batch_args(batch)[:3], heads))
    jitted = np.asarray(jax.jit(f)(*batch_args(batch)[:3], heads))
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)


# ------------------------------------------------------- weights + fallback ----


@pytest.mark.parametrize("config", [RAMAN_CONFIG, FL_CONFIG], ids=["raman", "fl"])
def test_save_load_round_trip(config, tmp_path, l23_small_inelastic_batch):
    """A saved head reloads to the same δ on real inputs."""
    batch = l23_small_inelastic_batch
    head = randomized(IC.init_head(config.kind, config))
    path = tmp_path / "head.npz"
    IC.save_head(head, path)
    loaded = IC.load_head(path)
    assert loaded.config == head.config
    np.testing.assert_array_equal(
        np.asarray(loaded.delta(batch.iops, batch.geometry, batch.wave)),
        np.asarray(head.delta(batch.iops, batch.geometry, batch.wave)),
    )


def test_load_refuses_feature_mismatch(tmp_path):
    """Weights against the wrong feature vector are refused, not warned."""
    head = IC.init_head("raman", RAMAN_CONFIG)
    path = tmp_path / "head.npz"
    IC.save_head(head, path)

    import numpy as _np

    data = dict(_np.load(path, allow_pickle=False))
    data["features"] = _np.asarray(("something", "else"))
    _np.savez(path, **data)
    with pytest.raises(ValueError, match="features"):
        IC.load_head(path)


def test_default_resolution_warns_once_and_is_analytic(l23_small_inelastic_batch):
    """Absent weights: corrections=None == corrections=False, plus one warning.

    ``load_default`` is memoised, so the warning is once per process — the
    cache is cleared here to make the test order-independent, and cleared
    again afterwards so this test does not eat another test's warning.
    Skipped once trained weights are committed (task 2): the fallback path
    then no longer exists to test.
    """
    if IC.DEFAULT_RAMAN_WEIGHTS.exists() and IC.DEFAULT_FL_WEIGHTS.exists():
        pytest.skip("trained weights are committed; the fallback path is gone")
    batch = l23_small_inelastic_batch

    IC.load_default.cache_clear()
    try:
        with pytest.warns(IC.MissingCorrectionWarning, match="ANALYTIC-ONLY"):
            default = np.asarray(
                H.forward(*batch_args(batch), inelastic=Inelastic(), check_domain=False)
            )
        explicit = np.asarray(
            H.forward(
                *batch_args(batch),
                inelastic=Inelastic(),
                corrections=False,
                check_domain=False,
            )
        )
        np.testing.assert_array_equal(default, explicit)
    finally:
        IC.load_default.cache_clear()


def test_elastic_path_never_resolves_corrections(l23_small_inelastic_batch):
    """inelastic=None (and all-off) never touch the loader — no warning.

    The elastic path owes nothing to the ML stack; a warning about missing
    correction weights on a purely elastic call would be noise about
    physics the caller never asked for.
    """
    batch = l23_small_inelastic_batch
    IC.load_default.cache_clear()
    try:
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", IC.MissingCorrectionWarning)
            H.forward(*batch_args(batch), check_domain=False)
            H.forward(
                *batch_args(batch),
                inelastic=Inelastic(raman=False, fluorescence=False),
                check_domain=False,
            )
    finally:
        IC.load_default.cache_clear()
