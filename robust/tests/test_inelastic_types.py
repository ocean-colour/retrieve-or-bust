"""
Tests for the M0 inelastic API extension (inelastic coding plan, M0 task 3).

Three concerns, in order of importance:

- **The elastic hash-regression.** ``forward(..., inelastic=None)`` on the
  committed 50-scene CI fixture must be *bit-identical* — SHA-256 of the output
  bytes — to the pre-extension hybrid. The hashes below were computed on the
  unmodified elastic code (branch point of ``inelastic-rt``, 2026-08-20, tank
  server) immediately *before* ``types.py``/``hybrid.py`` were touched, and
  verified deterministic across processes. This is the design §1 guarantee, and
  it guards every later milestone: M2's composition must route around the
  elastic path, never through arithmetic on it.

- **The new pytree**: :class:`robust.rt.types.Inelastic` — defaults,
  flatten/unflatten, which fields are leaves vs static, and that ``jit`` /
  ``vmap`` / ``grad`` traverse it. ``phi_C`` must be a *leaf* (differentiable —
  retrieving it is the point, design DQ4); the process switches must be
  *static* (they select code paths).

- **The extended fields**: ``IOPs.a_ph`` and ``Geometry.Ed``, both optional and
  defaulting to ``None`` following the ``Geometry.wind`` precedent — no leaves
  when unset, so every existing call site and ``vmap`` axis spec is untouched.

Everything here runs on the committed fixture or on synthetic inputs — no
``$OS_COLOR`` needed, so the whole module runs in CI.
"""

from __future__ import annotations

import dataclasses
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import hybrid as H
from robust.rt import types as T

N = C.N_WAVE

#: SHA-256 of ``np.asarray(out).tobytes()`` for ``forward``/``rrs_forward`` on
#: the 50-scene fixture (150 samples x 81 bands, float32), computed on the
#: PRE-EXTENSION elastic code with ``check_domain=False`` and defaults otherwise.
#: If either test fails after a code change, the elastic path is no longer the
#: pre-change code route — that is a gate failure, not a tolerance question.
#: (If a *platform* change — CPU, jax/XLA version — ever trips this without any
#: code change, that is a finding to take to JXP, not a hash to update quietly.)
PRE_CHANGE_SHA256_RRS_ABOVE = (
    "aaa0616119f179551e64969cd8407ed44e8eb0f8f5d9b27ba6ac7c97d826bbc7"
)
PRE_CHANGE_SHA256_RRS_BELOW = (
    "d111464020aacb47bbc9dd9aa027dd11b2e15e019a735687b6c6c0fa504c2c38"
)


def tiny_args():
    """Minimal synthetic inputs for tests that never score accuracy."""
    wave = jnp.asarray([440.0, 550.0])
    iops = T.IOPs(
        a=jnp.asarray([0.15, 0.12]),
        bb_w=C.bb_w(wave),
        bb_p=jnp.asarray([0.003, 0.003]),
    )
    return (
        iops,
        T.PhaseParams(B_p=jnp.asarray(0.0126)),
        T.Geometry.nadir(jnp.asarray(30.0)),
        wave,
    )


def sha256_of(array) -> str:
    """The regression hash: SHA-256 over the raw bytes, dtype and all."""
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


# ------------------------------------------------- elastic hash-regression ----


def test_elastic_hash_regression_Rrs(l23_small_batch):
    """``forward(..., inelastic=None)`` is bit-identical to the pre-change hybrid.

    ``check_domain=False`` because the full fixture includes the 60-deg samples
    the emulator warns about, and the pre-change pin was computed the same way.
    """
    batch = l23_small_batch
    out = H.forward(
        batch.iops,
        batch.phase_params,
        batch.geometry,
        batch.wave,
        inelastic=None,
        check_domain=False,
    )
    assert np.asarray(out).dtype == np.float32
    assert sha256_of(out) == PRE_CHANGE_SHA256_RRS_ABOVE


def test_elastic_hash_regression_rrs(l23_small_batch):
    """Same pin below the surface — the scored quantity (design §6)."""
    batch = l23_small_batch
    out = H.rrs_forward(
        batch.iops,
        batch.phase_params,
        batch.geometry,
        batch.wave,
        inelastic=None,
        check_domain=False,
    )
    assert sha256_of(out) == PRE_CHANGE_SHA256_RRS_BELOW


def test_inelastic_none_is_the_default(l23_small_batch):
    """Omitting ``inelastic`` and passing ``None`` are the same call, bitwise."""
    batch = l23_small_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    a = H.forward(*args, check_domain=False)
    b = H.forward(*args, inelastic=None, check_domain=False)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_inelastic_instance_raises_until_m2():
    """An actual Inelastic raises NotImplementedError naming the milestone.

    The elastic M0 convention: a caller gets a loud error, never a
    plausible-looking array with no inelastic physics in it.
    """
    args = tiny_args()
    with pytest.raises(NotImplementedError, match="M2"):
        H.forward(*args, inelastic=T.Inelastic())
    with pytest.raises(NotImplementedError, match="M2"):
        H.rrs_forward(*args, inelastic=T.Inelastic())


def test_elastic_route_still_jits_with_inelastic_none():
    """``inelastic=None`` is static, so the elastic route compiles as before."""
    iops, phase_params, geometry, wave = tiny_args()

    @jax.jit
    def f(iops, phase_params, geometry, wave):
        return H.rrs_forward(iops, phase_params, geometry, wave, "ztt", inelastic=None)

    out = f(iops, phase_params, geometry, wave)
    assert np.all(np.isfinite(np.asarray(out)))


# ----------------------------------------------------------- Inelastic type ----


def test_inelastic_defaults():
    """The design §3 signature, field for field."""
    inel = T.Inelastic()
    assert float(np.asarray(inel.phi_C)) == pytest.approx(0.02)
    assert inel.raman is True
    assert inel.fluorescence is True
    assert inel.emission_shape == "single"
    assert inel.cdom_fl is None


def test_inelastic_flatten_unflatten_roundtrip():
    """Flatten/unflatten reproduces the instance, static fields included."""
    inel = T.Inelastic(phi_C=jnp.asarray(0.01), raman=False, emission_shape="double")
    leaves, treedef = jax.tree_util.tree_flatten(inel)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, T.Inelastic)
    assert float(rebuilt.phi_C) == pytest.approx(0.01)
    assert rebuilt.raman is False
    assert rebuilt.fluorescence is True
    assert rebuilt.emission_shape == "double"


def test_inelastic_phi_c_is_the_only_default_leaf():
    """``phi_C`` is a leaf; the switches are static; unset ``cdom_fl`` adds none.

    This is the contract ``grad``/``vmap`` rely on: differentiating an
    ``Inelastic`` touches the quantum yield and nothing else.
    """
    leaves = jax.tree_util.tree_leaves(T.Inelastic())
    assert len(leaves) == 1
    assert float(np.asarray(leaves[0])) == pytest.approx(0.02)


def test_inelastic_static_fields_change_the_treedef():
    """The switches live in the treedef, so ``jit`` specializes per configuration."""
    _, td_on = jax.tree_util.tree_flatten(T.Inelastic())
    _, td_off = jax.tree_util.tree_flatten(T.Inelastic(raman=False))
    assert td_on != td_off


def test_inelastic_replace_and_frozen():
    """stdlib dataclass behaviour: frozen, and ``replace`` works."""
    inel = T.Inelastic()
    with pytest.raises(dataclasses.FrozenInstanceError):
        inel.phi_C = 0.05  # type: ignore[misc]
    bumped = dataclasses.replace(inel, phi_C=jnp.asarray(0.05))
    assert float(bumped.phi_C) == pytest.approx(0.05)
    assert float(np.asarray(inel.phi_C)) == pytest.approx(0.02)


def test_inelastic_jit_traversal():
    """``jit`` accepts an Inelastic argument and traces its leaf."""

    @jax.jit
    def yield_squared(inel: T.Inelastic):
        return inel.phi_C**2

    out = yield_squared(T.Inelastic(phi_C=jnp.asarray(0.03)))
    assert float(out) == pytest.approx(9e-4)


def test_inelastic_vmap_traversal():
    """``vmap`` maps over a batched ``phi_C`` — per-scene yields, design §3."""
    batched = T.Inelastic(phi_C=jnp.asarray([0.01, 0.02, 0.04]))
    out = jax.vmap(lambda inel: 2.0 * inel.phi_C)(batched)
    np.testing.assert_allclose(np.asarray(out), [0.02, 0.04, 0.08], rtol=1e-6)


def test_inelastic_grad_returns_the_container():
    """``grad`` w.r.t. an Inelastic returns an Inelastic — d/d(phi_C) labelled.

    ``phi_C`` enters the fluorescence term linearly (``Rrs_fl = phi_C * K_fl``),
    so its derivative is the kernel itself; here a stand-in scalar checks the
    plumbing that M2's real gradient gate will exercise.
    """
    grad = jax.grad(lambda inel: 3.0 * inel.phi_C)(T.Inelastic(phi_C=jnp.asarray(0.02)))
    assert isinstance(grad, T.Inelastic)
    assert float(grad.phi_C) == pytest.approx(3.0)


def test_inelastic_validate_accepts_defaults():
    T.Inelastic().validate()


@pytest.mark.parametrize(
    "bad",
    [
        T.Inelastic(phi_C=jnp.asarray(0.0)),
        T.Inelastic(phi_C=jnp.asarray(-0.01)),
        T.Inelastic(phi_C=jnp.asarray(1.5)),
        T.Inelastic(phi_C=jnp.asarray(jnp.nan)),
        T.Inelastic(emission_shape="triple"),
        T.Inelastic(cdom_fl=jnp.asarray(0.1)),
    ],
    ids=["zero", "negative", "above-one", "nan", "bad-shape", "cdom-set"],
)
def test_inelastic_validate_rejects(bad):
    """Each documented failure mode raises; phi_C=0 points at the boolean."""
    with pytest.raises(ValueError):
        bad.validate()


# --------------------------------------------------------------- IOPs.a_ph ----


def test_iops_a_ph_defaults_to_none_with_unchanged_leaves():
    """No ``a_ph`` → exactly the elastic three leaves, as every M1-M4 test built."""
    iops = T.IOPs(
        a=jnp.full(N, 0.15),
        bb_w=jnp.broadcast_to(C.bb_w(), (N,)),
        bb_p=jnp.full(N, 0.003),
    )
    assert iops.a_ph is None
    assert len(jax.tree_util.tree_leaves(iops)) == 3


def test_iops_a_ph_set_becomes_a_leaf_and_survives_jit_vmap():
    """With ``a_ph`` set there are four leaves and JAX traverses them all."""
    shape = (5, N)
    iops = T.IOPs(
        a=jnp.full(shape, 0.15),
        bb_w=jnp.broadcast_to(C.bb_w(), shape),
        bb_p=jnp.full(shape, 0.003),
        a_ph=jnp.full(shape, 0.05),
    )
    assert len(jax.tree_util.tree_leaves(iops)) == 4
    total = jax.jit(lambda i: i.a_ph.sum())(iops)
    # rel=1e-5, not 1e-6: a 405-term float32 sum carries ~1e-6 of accumulation
    # noise on its own (the float32-tolerance gotcha the elastic record §2 pins).
    assert float(total) == pytest.approx(0.05 * 5 * N, rel=1e-5)
    per_scene = jax.vmap(lambda i: i.a_ph.mean())(iops)
    assert per_scene.shape == (5,)


def test_iops_from_total_bb_passes_a_ph_through():
    a = jnp.full(N, 0.15)
    bb = jnp.broadcast_to(C.bb_w(), (N,)) + 0.003
    a_ph = jnp.full(N, 0.05)
    iops = T.IOPs.from_total_bb(a, bb, a_ph=a_ph)
    np.testing.assert_array_equal(np.asarray(iops.a_ph), np.asarray(a_ph))
    assert T.IOPs.from_total_bb(a, bb).a_ph is None


def test_iops_validate_rejects_bad_a_ph():
    """Shape mismatch, negative values, and a_ph > a are each caught."""
    good = T.IOPs(
        a=jnp.full(N, 0.15),
        bb_w=jnp.broadcast_to(C.bb_w(), (N,)),
        bb_p=jnp.full(N, 0.003),
        a_ph=jnp.full(N, 0.05),
    )
    good.validate()
    with pytest.raises(ValueError, match="a_ph"):
        dataclasses.replace(good, a_ph=jnp.full(N - 1, 0.05)).validate()
    with pytest.raises(ValueError, match="a_ph"):
        dataclasses.replace(good, a_ph=jnp.full(N, -0.05)).validate()
    with pytest.raises(ValueError, match="component"):
        dataclasses.replace(good, a_ph=jnp.full(N, 0.2)).validate()


def test_elastic_forward_ignores_a_ph():
    """The elastic path gives bit-identical output with and without ``a_ph``.

    'Ignores it' is the design's word (§3); this is that word as arithmetic.
    """
    iops, phase_params, geometry, wave = tiny_args()
    with_a_ph = dataclasses.replace(iops, a_ph=jnp.asarray([0.05, 0.03]))
    a = H.rrs_forward(iops, phase_params, geometry, wave, "ztt")
    b = H.rrs_forward(with_a_ph, phase_params, geometry, wave, "ztt")
    assert np.array_equal(np.asarray(a), np.asarray(b))


# ------------------------------------------------------------- Geometry.Ed ----


def test_geometry_ed_defaults_to_none_with_unchanged_leaves():
    geom = T.Geometry.nadir(jnp.asarray(30.0))
    assert geom.Ed is None
    assert len(jax.tree_util.tree_leaves(geom)) == 3  # wind=None, Ed=None


def test_geometry_ed_set_becomes_leaves():
    """The override's two arrays are leaves, so they trace and differentiate."""
    wave_ed = jnp.linspace(350.0, 750.0, 11)
    ed = jnp.full(11, 1.2)
    geom = dataclasses.replace(T.Geometry.nadir(jnp.asarray(30.0)), Ed=(wave_ed, ed))
    assert len(jax.tree_util.tree_leaves(geom)) == 5
    geom.validate()


def test_geometry_ed_validate_rejects_bad_overrides():
    base = T.Geometry.nadir(jnp.asarray(30.0))
    wave_ed = jnp.linspace(350.0, 750.0, 11)
    ed = jnp.full(11, 1.2)
    with pytest.raises(ValueError, match="pair"):
        dataclasses.replace(base, Ed=(wave_ed, ed, ed)).validate()
    with pytest.raises(ValueError, match="same"):
        dataclasses.replace(base, Ed=(wave_ed, ed[:-1])).validate()
    with pytest.raises(ValueError, match="increasing"):
        dataclasses.replace(base, Ed=(wave_ed[::-1], ed)).validate()
    with pytest.raises(ValueError, match="Ed"):
        dataclasses.replace(base, Ed=(wave_ed, -ed)).validate()


def test_elastic_forward_ignores_ed():
    """The elastic path gives bit-identical output with and without the override."""
    iops, phase_params, geometry, wave = tiny_args()
    with_ed = dataclasses.replace(
        geometry, Ed=(jnp.linspace(350.0, 750.0, 5), jnp.full(5, 1.0))
    )
    a = H.rrs_forward(iops, phase_params, geometry, wave, "ztt")
    b = H.rrs_forward(iops, phase_params, with_ed, wave, "ztt")
    assert np.array_equal(np.asarray(a), np.asarray(b))


# ------------------------------------------------------------------ exports ----


def test_inelastic_is_exported():
    """``from robust import rt`` carries the new type beside the elastic three."""
    from robust import rt

    assert rt.Inelastic is T.Inelastic
    assert "Inelastic" in rt.__all__
