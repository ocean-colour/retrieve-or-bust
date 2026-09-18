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

  **The gate has two tiers** (M0 task 7; prompt 1 Q&A Q2). Bit-identity is
  anchored to the machine that pinned it: GitHub's heterogeneous runner fleet
  reproduced the tank bits on some runners and not others *within one CI run*,
  so the strict bitwise tests skip under CI (the M2 bing-xcheck precedent) and
  are mandatory-green **on their anchor machine** — since M5 there are two pin
  sets on two different machines, so each strict tier now also skips where it
  is not anchored rather than failing there (:func:`strict_bits_on_anchor`,
  docs prompt 1 Q&A Q12). CI instead runs a *closeness*
  regression against the committed pre-change outputs
  (``files/elastic_reference_outputs.npz``, whose bytes hash to the pins
  below) at ULP-scale tolerance — tight enough to catch any real change to
  the elastic route, loose enough to admit cross-platform float32 noise.
  Do not "fix" a strict-tier failure by re-pinning to a new hash without
  understanding what changed: the pin is the guard, not the target.

- **The new pytree**: :class:`robust.rt.types.Inelastic` — defaults,
  flatten/unflatten, which fields are leaves vs static, and that ``jit`` /
  ``vmap`` / ``grad`` traverse it. ``phi_C`` must be a *leaf* (differentiable —
  retrieving it is the point, design DQ4); the process switches must be
  *static* (they select code paths).

- **The extended fields**: ``IOPs.a_ph`` and ``Geometry.Ed``, both optional and
  defaulting to ``None`` following the ``Geometry.wind`` precedent — no leaves
  when unset, so every existing call site and ``vmap`` axis spec is untouched.

M5 (CDOM-fluorescence design, task 1) extends the same contracts: ``IOPs``
gains optional ``a_cdom`` (mirroring ``a_ph`` exactly), ``CDOMFl`` is a new
registered pytree (``scale`` its single differentiable leaf), and
``Inelastic.cdom_fl`` is retyped from the reserved always-reject scalar hook to
``CDOMFl | None`` — ``None`` stays the default, and the set-but-unused states
stay bit-identical (the design §3 extended bit-identity requirement).

Everything here runs on the committed fixture or on synthetic inputs — no
``$OS_COLOR`` needed, so the whole module runs in CI.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import platform

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import hybrid as H
from robust.rt import types as T

N = C.N_WAVE

#: The committed pre-change outputs (see the module docstring); regenerate with
#: ``design/py/gen_inelastic_fixture.py`` only on a deliberate elastic change.
ELASTIC_REFERENCE = (
    pathlib.Path(__file__).parent / "files" / "elastic_reference_outputs.npz"
)

#: The machine each strict pin set was computed on (see the pin blocks below).
#: They are deliberately *different* machines, which is why the strict tiers
#: have to be selected per anchor rather than simply "not CI".
ELASTIC_PIN_ANCHOR = "tank"
INELASTIC_PIN_ANCHOR = "mac"

#: Hosts whose anchor identity we can name, so the two anchor machines keep
#: their bitwise guard without anyone having to remember an environment
#: variable. Keys are ``platform.node()``, lowercased. Add the tank server's
#: node name here the first time the suite runs there.
ANCHOR_HOSTS = {"mac.lan": INELASTIC_PIN_ANCHOR}

#: Environment override, read once: ``ROBUST_HASH_ANCHOR=tank``, ``=mac``, a
#: comma-separated list (a machine that anchors both), or ``=none`` to disown
#: every pin set on a machine that :data:`ANCHOR_HOSTS` would otherwise claim.


def this_machines_anchors() -> frozenset[str]:
    """Which strict pin sets this machine is the anchor for.

    ``ROBUST_HASH_ANCHOR`` wins when set; otherwise :data:`ANCHOR_HOSTS` maps
    the hostname. An unknown machine anchors **nothing**, which is the safe
    default: a strict tier that cannot be trusted here skips with a reason
    instead of failing, and the closeness tiers — which pass everywhere — stay
    the regression guard.
    """
    declared = os.environ.get("ROBUST_HASH_ANCHOR", "").strip().lower()
    if declared:
        return frozenset(a for a in (p.strip() for p in declared.split(",")) if a)
    host = platform.node().strip().lower()
    return frozenset({ANCHOR_HOSTS[host]} if host in ANCHOR_HOSTS else ())


def strict_bits_on_anchor(anchor: str):
    """Skip a strict bitwise tier unless this machine is its anchor.

    Bit-identity is machine-anchored twice over: the elastic pins were
    computed on the tank server and the inelastic ones on JXP's Mac, so no
    single machine reproduces both and an unconditional strict tier turns a
    routine ``pytest`` run permanently red — which is a weaker regression
    signal than a green suite, not a stronger one (prompt-1 Q&A Q12,
    option 1). CI skips both tiers for the older reason (Q&A Q2): GitHub's
    heterogeneous runners reproduced the tank bits on some machines and not
    others *within one run*.
    """
    if os.environ.get("CI", "") != "":
        return pytest.mark.skipif(
            True,
            reason="bitwise hash pins are machine-anchored; CI runners "
            "reproduce them only sometimes — the closeness tier runs instead",
        )
    here = sorted(this_machines_anchors())
    return pytest.mark.skipif(
        anchor not in here,
        reason=f"bitwise hash pins are anchored to the {anchor!r} machine; "
        f"this one is {'+'.join(here) or 'unanchored'} — set "
        f"ROBUST_HASH_ANCHOR={anchor} to claim it. The closeness tier carries "
        "the regression here",
    )


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

#: The committed pre-CDOM-wiring default-inelastic outputs (M5 task 5) —
#: ``forward``/``rrs_forward`` with ``inelastic=Inelastic()`` (``raman=True,
#: fluorescence=True, cdom_fl=None``) on the 150-sample inelastic fixture;
#: regenerate with ``gen_inelastic_fixture.write_inelastic_default_reference``
#: only on a deliberate change to the shipped inelastic model.
INELASTIC_DEFAULT_REFERENCE = (
    pathlib.Path(__file__).parent / "files" / "inelastic_default_reference_outputs.npz"
)

#: SHA-256 of ``np.asarray(out).tobytes()`` for the default-inelastic
#: configuration above, computed on the PRE-CDOM-WIRING code (M5 task 5, step
#: 5a: the reference was written and hashed **before** ``hybrid.py`` grew the
#: ``Rrs_cdom`` composition) with the committed trained heads
#: (``corrections=None``) and ``check_domain=False``. These pins prove the
#: CDOM branch is unreachable — a no-op by construction — when ``cdom_fl``
#: stays ``None`` (CDOM design §3). Machine anchoring: pinned on JXP's Mac
#: (darwin, 2026-08-29) — a *different* machine from the tank server that
#: anchored the elastic pins above, so no one machine can reproduce both pin
#: sets. Each strict tier therefore runs only on its own anchor and skips
#: elsewhere (:func:`strict_bits_on_anchor`); the closeness tiers carry the
#: guard everywhere (the finding recorded in the M5 prompt doc's task-1 log,
#: decided at docs prompt 1 Q&A Q12).
PRE_CDOM_SHA256_RRS_ABOVE = (
    "0dd365158e3037261ee061777fe51da8fa132d4f0972792ad068b9c73641291a"
)
PRE_CDOM_SHA256_RRS_BELOW = (
    "72d4a308e2222c802e18e1878d00f26853db831d9db82a8e529cfead883cc0b8"
)


from robust.tests.conftest import needs_weights, tiny_args  # noqa: E402 - shared


def sha256_of(array) -> str:
    """The regression hash: SHA-256 over the raw bytes, dtype and all."""
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


# ------------------------------------------------- elastic hash-regression ----


def elastic_outputs(batch):
    """The two regression quantities, computed exactly as the pins were.

    ``check_domain=False`` because the full fixture includes the 60-deg samples
    the emulator warns about, and the pre-change pins were computed the same
    way.
    """
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    return (
        H.forward(*args, inelastic=None, check_domain=False),
        H.rrs_forward(*args, inelastic=None, check_domain=False),
    )


@strict_bits_on_anchor(ELASTIC_PIN_ANCHOR)
def test_elastic_hash_regression_strict(l23_small_batch):
    """``forward(..., inelastic=None)`` is bit-identical to the pre-change hybrid.

    Strict tier: the SHA-256 pins *and* element-wise equality with the
    committed reference (same content — the arrays' bytes hash to the pins;
    the array comparison is here so a failure names positions, not just
    digests). Skips under CI and off the tank server that pinned it
    (``ROBUST_HASH_ANCHOR=tank``); mandatory-green there.
    """
    Rrs, rrs = elastic_outputs(l23_small_batch)
    assert np.asarray(Rrs).dtype == np.float32
    assert sha256_of(Rrs) == PRE_CHANGE_SHA256_RRS_ABOVE
    assert sha256_of(rrs) == PRE_CHANGE_SHA256_RRS_BELOW

    reference = np.load(ELASTIC_REFERENCE)
    np.testing.assert_array_equal(np.asarray(Rrs), reference["Rrs"])
    np.testing.assert_array_equal(np.asarray(rrs), reference["rrs"])


def test_elastic_regression_close_everywhere(l23_small_batch):
    """Closeness tier: the elastic route matches the committed pre-change
    outputs to ULP scale, on every platform.

    The committed arrays are the pinned bytes (their SHA-256 *is* the strict
    pin), so this is the same guard at the tolerance cross-platform float32
    allows: rtol 5e-7 is ~4 ULP — CI's observed runner-to-runner spread is
    1-2 ULP, while any genuine restructuring of the elastic route (e.g. an
    extra pass through the ``rrs <-> Rrs`` conversion, notebook 1 §4) shows
    up at this tolerance as a broad, not marginal, failure.
    """
    Rrs, rrs = elastic_outputs(l23_small_batch)
    reference = np.load(ELASTIC_REFERENCE)
    assert sha256_of(reference["Rrs"]) == PRE_CHANGE_SHA256_RRS_ABOVE
    assert sha256_of(reference["rrs"]) == PRE_CHANGE_SHA256_RRS_BELOW
    np.testing.assert_allclose(np.asarray(Rrs), reference["Rrs"], rtol=5e-7, atol=0.0)
    np.testing.assert_allclose(np.asarray(rrs), reference["rrs"], rtol=5e-7, atol=0.0)


# ---------------------------------- inelastic-default hash-regression (M5) ----


def inelastic_default_outputs(batch):
    """The two M5 regression quantities, computed exactly as the pins were.

    The default ``Inelastic()`` — ``cdom_fl=None`` implicitly — with the
    packaged trained heads (``corrections=None``, the shipped default) and
    ``check_domain=False``; the :func:`elastic_outputs` pattern on the
    inelastic fixture, whose samples carry the ``a_ph``/``a_cdom`` the
    default configuration exercises.
    """
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    return (
        H.forward(*args, inelastic=T.Inelastic(), check_domain=False),
        H.rrs_forward(*args, inelastic=T.Inelastic(), check_domain=False),
    )


@needs_weights
@strict_bits_on_anchor(INELASTIC_PIN_ANCHOR)
def test_inelastic_default_hash_regression_strict(l23_small_inelastic_batch):
    """``forward(..., inelastic=Inelastic())`` is bit-identical to the
    pre-CDOM-wiring shipped inelastic model.

    The M5 task-5 extension of the elastic strict tier (CDOM design §3): the
    pins and reference were computed **before** ``hybrid.py`` grew the
    ``Rrs_cdom`` composition, so this passing after the wiring proves the new
    branch is unreachable when ``cdom_fl`` stays ``None`` — the default model
    remains provably CDOM-fl-free, keeping the X4-truth 0.34 % gate's claims
    valid. Skips under CI, off the Mac that pinned it
    (``ROBUST_HASH_ANCHOR=mac``), and without the committed heads (the default
    ``corrections=None`` resolves them; absent weights would silently change
    the bytes under comparison).
    """
    Rrs, rrs = inelastic_default_outputs(l23_small_inelastic_batch)
    assert np.asarray(Rrs).dtype == np.float32
    assert sha256_of(Rrs) == PRE_CDOM_SHA256_RRS_ABOVE
    assert sha256_of(rrs) == PRE_CDOM_SHA256_RRS_BELOW

    reference = np.load(INELASTIC_DEFAULT_REFERENCE)
    np.testing.assert_array_equal(np.asarray(Rrs), reference["Rrs"])
    np.testing.assert_array_equal(np.asarray(rrs), reference["rrs"])


@needs_weights
def test_inelastic_default_regression_close_everywhere(l23_small_inelastic_batch):
    """Closeness tier: the default-inelastic route matches the committed
    pre-CDOM-wiring outputs to ULP scale, on every platform.

    The committed arrays are the pinned bytes (their SHA-256 *is* the strict
    pin), so this is the same guard at the tolerance cross-platform float32
    allows — rtol 5e-7 (~4 ULP), exactly the elastic closeness tier's. Any
    genuine change to the default inelastic route — e.g. CDOM arithmetic
    leaking into the ``cdom_fl=None`` path — shows up broadly at this
    tolerance, on the tank server and CI as well as the pinning Mac.
    """
    Rrs, rrs = inelastic_default_outputs(l23_small_inelastic_batch)
    reference = np.load(INELASTIC_DEFAULT_REFERENCE)
    assert sha256_of(reference["Rrs"]) == PRE_CDOM_SHA256_RRS_ABOVE
    assert sha256_of(reference["rrs"]) == PRE_CDOM_SHA256_RRS_BELOW
    np.testing.assert_allclose(np.asarray(Rrs), reference["Rrs"], rtol=5e-7, atol=0.0)
    np.testing.assert_allclose(np.asarray(rrs), reference["rrs"], rtol=5e-7, atol=0.0)


def test_inelastic_none_is_the_default(l23_small_batch):
    """Omitting ``inelastic`` and passing ``None`` are the same call, bitwise."""
    batch = l23_small_batch
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    a = H.forward(*args, check_domain=False)
    b = H.forward(*args, inelastic=None, check_domain=False)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_inelastic_fluorescence_without_aph_raises():
    """Fluorescence without its source term raises, loudly.

    The elastic M0 convention: a caller gets a loud error, never a
    plausible-looking array with missing physics. Updated deliberately at
    each M2 task (as the M1 hand-off required): Raman composes since task 1,
    the fluorescence kernel since task 2 — so the *unimplemented*-physics
    guard has retired, and what this call now guarantees is the *physical*
    requirement: the default ``Inelastic()`` asks for fluorescence, whose
    source term is ``b_F = phi_C * a_ph``, and ``tiny_args`` carries no
    ``a_ph`` — ``ValueError`` before any model work, never an array with a
    silently missing term.
    """
    args = tiny_args()
    with pytest.raises(ValueError, match="a_ph"):
        H.forward(*args, inelastic=T.Inelastic())
    with pytest.raises(ValueError, match="a_ph"):
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
        # A bare scalar was the pre-M5 reserved-hook signature; since the M5
        # retype it is rejected for a *different* reason — cdom_fl must be a
        # CDOMFl instance, and validate() type-checks it so the old calling
        # convention fails loudly instead of half-configuring the process.
        T.Inelastic(cdom_fl=jnp.asarray(0.1)),
        # A CDOMFl instance is accepted, but validate() delegates to
        # CDOMFl.validate(), so an unusable nested configuration still raises.
        T.Inelastic(cdom_fl=T.CDOMFl(scale=jnp.asarray(0.0))),
    ],
    ids=[
        "zero",
        "negative",
        "above-one",
        "nan",
        "bad-shape",
        "cdom-not-a-cdomfl",
        "cdom-bad-scale",
    ],
)
def test_inelastic_validate_rejects(bad):
    """Each documented failure mode raises; phi_C=0 points at the boolean."""
    with pytest.raises(ValueError):
        bad.validate()


# ------------------------------------------------------------- CDOMFl type ----


def test_cdom_fl_defaults():
    """The CDOM design §3 signature: ``CDOMFl(scale=1.0)``."""
    cdom = T.CDOMFl()
    assert float(np.asarray(cdom.scale)) == pytest.approx(1.0)


def test_cdom_fl_flatten_unflatten_roundtrip():
    """Flatten/unflatten reproduces the instance."""
    cdom = T.CDOMFl(scale=jnp.asarray(1.5))
    leaves, treedef = jax.tree_util.tree_flatten(cdom)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, T.CDOMFl)
    assert float(rebuilt.scale) == pytest.approx(1.5)


def test_cdom_fl_scale_is_the_only_leaf():
    """``scale`` is the single leaf — the contract ``grad``/``vmap`` rely on."""
    leaves = jax.tree_util.tree_leaves(T.CDOMFl())
    assert len(leaves) == 1
    assert float(np.asarray(leaves[0])) == pytest.approx(1.0)


def test_cdom_fl_replace_and_frozen():
    """stdlib dataclass behaviour: frozen, and ``replace`` works."""
    cdom = T.CDOMFl()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cdom.scale = 2.0  # type: ignore[misc]
    bumped = dataclasses.replace(cdom, scale=jnp.asarray(2.0))
    assert float(bumped.scale) == pytest.approx(2.0)
    assert float(np.asarray(cdom.scale)) == pytest.approx(1.0)


def test_cdom_fl_jit_traversal():
    """``jit`` accepts a CDOMFl argument and traces its leaf."""

    @jax.jit
    def scale_squared(cdom: T.CDOMFl):
        return cdom.scale**2

    out = scale_squared(T.CDOMFl(scale=jnp.asarray(1.5)))
    assert float(out) == pytest.approx(2.25)


def test_cdom_fl_vmap_traversal():
    """``vmap`` maps over a batched ``scale`` — per-scene amplitudes."""
    batched = T.CDOMFl(scale=jnp.asarray([0.5, 1.0, 2.0]))
    out = jax.vmap(lambda cdom: 2.0 * cdom.scale)(batched)
    np.testing.assert_allclose(np.asarray(out), [1.0, 2.0, 4.0], rtol=1e-6)


def test_cdom_fl_grad_returns_the_container():
    """``grad`` w.r.t. a CDOMFl returns a CDOMFl — d/d(scale) labelled.

    ``scale`` enters the CDOM-fluorescence term linearly
    (``Rrs_cdom = scale * K_cdom * (1 + delta_C)``), so its derivative is the
    kernel itself; here a stand-in scalar checks the plumbing that M5 task 7's
    real gradient gate will exercise.
    """
    grad = jax.grad(lambda cdom: 3.0 * cdom.scale)(T.CDOMFl(scale=jnp.asarray(1.0)))
    assert isinstance(grad, T.CDOMFl)
    assert float(grad.scale) == pytest.approx(3.0)


def test_cdom_fl_validate_accepts_default():
    T.CDOMFl().validate()


@pytest.mark.parametrize(
    "bad",
    [
        T.CDOMFl(scale=jnp.asarray(0.0)),
        T.CDOMFl(scale=jnp.asarray(-1.0)),
        T.CDOMFl(scale=jnp.asarray(jnp.nan)),
    ],
    ids=["zero", "negative", "nan"],
)
def test_cdom_fl_validate_rejects(bad):
    """Non-finite or non-positive ``scale`` raises; zero points at cdom_fl=None."""
    with pytest.raises(ValueError, match="scale"):
        bad.validate()


def test_inelastic_accepts_cdom_fl_instance():
    """A CDOMFl instance is now accepted — the reserved hook retired at M5."""
    T.Inelastic(cdom_fl=T.CDOMFl()).validate()


def test_inelastic_cdom_fl_set_nests_its_leaf():
    """With ``cdom_fl`` set, ``phi_C`` and the nested ``scale`` are both leaves;
    unset, the default single-leaf contract of
    :func:`test_inelastic_phi_c_is_the_only_default_leaf` is unchanged.
    """
    leaves = jax.tree_util.tree_leaves(
        T.Inelastic(cdom_fl=T.CDOMFl(scale=jnp.asarray(2.0)))
    )
    assert len(leaves) == 2
    values = sorted(float(np.asarray(leaf)) for leaf in leaves)
    assert values[0] == pytest.approx(0.02)  # phi_C
    assert values[1] == pytest.approx(2.0)  # cdom_fl.scale
    assert len(jax.tree_util.tree_leaves(T.Inelastic(cdom_fl=None))) == 1


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


def test_iops_from_total_bb_broadcasts_a_ph_like_bb_w():
    """A shared (n_wave,) a_ph is broadcast to the batch shape, as bb_w is.

    Regression for PR #14 review finding 3: a_ph was passed through
    unbroadcast, so the constructor's own uniform-batch-shape guarantee (the
    reason plain ``vmap(f, in_axes=0)`` works) broke the moment a shared
    spectrum was supplied.
    """
    shape = (5, N)
    iops = T.IOPs.from_total_bb(
        jnp.full(shape, 0.15),
        jnp.broadcast_to(C.bb_w(), shape) + 0.003,
        a_ph=jnp.full(N, 0.05),
    )
    assert iops.a_ph.shape == shape
    iops.validate()
    per_scene = jax.vmap(lambda i: i.a_ph.mean())(iops)
    assert per_scene.shape == (5,)


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


# ------------------------------------------------------------- IOPs.a_cdom ----


def test_iops_a_cdom_defaults_to_none_with_unchanged_leaves():
    """No ``a_cdom`` → exactly the elastic three leaves, as before M5."""
    iops = T.IOPs(
        a=jnp.full(N, 0.15),
        bb_w=jnp.broadcast_to(C.bb_w(), (N,)),
        bb_p=jnp.full(N, 0.003),
    )
    assert iops.a_cdom is None
    assert len(jax.tree_util.tree_leaves(iops)) == 3


def test_iops_a_cdom_set_becomes_a_leaf_and_survives_jit_vmap():
    """With ``a_cdom`` set there are four leaves and JAX traverses them all."""
    shape = (5, N)
    iops = T.IOPs(
        a=jnp.full(shape, 0.15),
        bb_w=jnp.broadcast_to(C.bb_w(), shape),
        bb_p=jnp.full(shape, 0.003),
        a_cdom=jnp.full(shape, 0.05),
    )
    assert len(jax.tree_util.tree_leaves(iops)) == 4
    total = jax.jit(lambda i: i.a_cdom.sum())(iops)
    # rel=1e-5, not 1e-6: a 405-term float32 sum carries ~1e-6 of accumulation
    # noise on its own (the float32-tolerance gotcha the elastic record §2 pins).
    assert float(total) == pytest.approx(0.05 * 5 * N, rel=1e-5)
    per_scene = jax.vmap(lambda i: i.a_cdom.mean())(iops)
    assert per_scene.shape == (5,)


def test_iops_from_total_bb_passes_a_cdom_through():
    a = jnp.full(N, 0.15)
    bb = jnp.broadcast_to(C.bb_w(), (N,)) + 0.003
    a_cdom = jnp.full(N, 0.05)
    iops = T.IOPs.from_total_bb(a, bb, a_cdom=a_cdom)
    np.testing.assert_array_equal(np.asarray(iops.a_cdom), np.asarray(a_cdom))
    assert T.IOPs.from_total_bb(a, bb).a_cdom is None


def test_iops_from_total_bb_broadcasts_a_cdom_like_bb_w():
    """A shared (n_wave,) a_cdom is broadcast to the batch shape, as bb_w is.

    The a_ph analogue (PR #14 review finding 3) applied to the new field: the
    constructor's uniform-batch-shape guarantee — the reason plain
    ``vmap(f, in_axes=0)`` works — must hold for a shared spectrum too.
    """
    shape = (5, N)
    iops = T.IOPs.from_total_bb(
        jnp.full(shape, 0.15),
        jnp.broadcast_to(C.bb_w(), shape) + 0.003,
        a_cdom=jnp.full(N, 0.05),
    )
    assert iops.a_cdom.shape == shape
    iops.validate()
    per_scene = jax.vmap(lambda i: i.a_cdom.mean())(iops)
    assert per_scene.shape == (5,)


def test_iops_validate_rejects_bad_a_cdom():
    """Shape mismatch, negative values, and a_cdom > a are each caught."""
    good = T.IOPs(
        a=jnp.full(N, 0.15),
        bb_w=jnp.broadcast_to(C.bb_w(), (N,)),
        bb_p=jnp.full(N, 0.003),
        a_cdom=jnp.full(N, 0.05),
    )
    good.validate()
    with pytest.raises(ValueError, match="a_cdom"):
        dataclasses.replace(good, a_cdom=jnp.full(N - 1, 0.05)).validate()
    with pytest.raises(ValueError, match="a_cdom"):
        dataclasses.replace(good, a_cdom=jnp.full(N, -0.05)).validate()
    with pytest.raises(ValueError, match="component"):
        dataclasses.replace(good, a_cdom=jnp.full(N, 0.2)).validate()


def test_elastic_forward_ignores_a_cdom():
    """The elastic path gives bit-identical output with and without ``a_cdom``.

    'Ignores it' is the CDOM design's word (§3); this is that word as
    arithmetic — the :func:`test_elastic_forward_ignores_a_ph` twin.
    """
    iops, phase_params, geometry, wave = tiny_args()
    with_a_cdom = dataclasses.replace(iops, a_cdom=jnp.asarray([0.05, 0.03]))
    a = H.rrs_forward(iops, phase_params, geometry, wave, "ztt")
    b = H.rrs_forward(with_a_cdom, phase_params, geometry, wave, "ztt")
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_inelastic_forward_ignores_unused_a_cdom(l23_small_inelastic_batch):
    """The inelastic path (``cdom_fl=None``) is bitwise indifferent to a set
    ``a_cdom`` — the design §3 extended bit-identity requirement, on the real
    Raman + Chl-fl route over the committed 50-scene fixture.
    """
    batch = l23_small_inelastic_batch
    with_a_cdom = dataclasses.replace(batch.iops, a_cdom=0.3 * batch.iops.a)
    kwargs = dict(inelastic=T.Inelastic(), corrections=False, check_domain=False)
    a = H.forward(batch.iops, batch.phase_params, batch.geometry, batch.wave, **kwargs)
    b = H.forward(with_a_cdom, batch.phase_params, batch.geometry, batch.wave, **kwargs)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_inelastic_cdom_fl_without_a_cdom_raises():
    """CDOM fluorescence without its source term raises, loudly.

    The :func:`test_inelastic_fluorescence_without_aph_raises` twin: setting
    ``cdom_fl`` asks for a term whose source is proportional to ``a_cdom``
    (CDOM design §2), and ``tiny_args`` carries no ``a_cdom`` — ``ValueError``
    before any model work, never an array with a silently missing term.
    ``fluorescence=False`` so the ``a_ph`` guard (also unmet here) cannot mask
    the one under test.
    """
    args = tiny_args()
    inelastic = T.Inelastic(fluorescence=False, cdom_fl=T.CDOMFl())
    with pytest.raises(ValueError, match="a_cdom"):
        H.forward(*args, inelastic=inelastic)
    with pytest.raises(ValueError, match="a_cdom"):
        H.rrs_forward(*args, inelastic=inelastic)


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


def test_cdom_fl_is_exported():
    """``robust.rt`` re-exports ``CDOMFl`` (M5 task 5 — deferred from task 1,
    which left ``__init__.py`` untouched; the composition wiring makes
    ``CDOMFl`` a genuine ``forward()`` argument type, so it exports beside
    ``Inelastic``, and the ``cdom_fl`` submodule joins the roster)."""
    from robust import rt

    assert rt.CDOMFl is T.CDOMFl
    assert "CDOMFl" in rt.__all__
    assert "cdom_fl" in rt.__all__
