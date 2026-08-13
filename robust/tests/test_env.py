"""
The M0 gate: the environment and the package scaffold.

Nothing scientific — this module answers "can the prototype be built here at
all?". It checks the three JAX properties the later milestones actually rely on
(a CPU backend, float64 on demand, working autodiff), that the ``robust.rt``
scaffold imports and exposes the designed API, and that the unimplemented pieces
say so loudly instead of returning something wrong.
"""

import numpy as np
import pytest

from robust.tests.conftest import l23_available

# ------------------------------------------------------------------ JAX -----


def test_jax_imports_and_runs_on_cpu():
    """``import jax; jax.numpy.ones(3)`` works, on the CPU backend."""
    import jax
    import jax.numpy as jnp

    x = jnp.ones(3)

    np.testing.assert_array_equal(np.asarray(x), np.ones(3))
    assert jax.default_backend() == "cpu"
    # CPU-only for now (coding plan CQ5): no accelerator should sneak in.
    assert [d.platform for d in jax.devices()] == ["cpu"]
    assert all(d.platform == "cpu" for d in x.devices())


def test_jax_float64_available(jax_x64):
    """float64 is reachable — the M2 finite-difference gate needs it.

    In the default float32 regime the differencing noise would swamp a
    ``jax.grad`` comparison, so this is a real prerequisite, not a nicety.
    """
    assert jax_x64.numpy.ones(3).dtype == np.float64


def test_jax_x64_fixture_restores_default():
    """Outside the fixture, JAX is back to its float32 default.

    Guards the fixture itself: a leaked global x64 flag would quietly change the
    dtype regime of every test that runs afterwards.
    """
    import jax.numpy as jnp

    assert jnp.ones(3).dtype == np.float32


def test_jax_grad_works():
    """Autodiff runs: d/dx sum(x**2) = 2x."""
    import jax
    import jax.numpy as jnp

    grad = jax.grad(lambda v: jnp.sum(v**2))(3.0)

    assert float(grad) == pytest.approx(6.0)


def test_jax_jit_works():
    """``jit`` compiles and returns the right answer.

    ``forward`` is specified as ``jit``/``vmap``-friendly (design §3), so a
    broken XLA compile should fail here rather than at M3.
    """
    import jax
    import jax.numpy as jnp

    f = jax.jit(lambda v: jnp.sum(v**2))

    assert float(f(jnp.arange(4.0))) == pytest.approx(14.0)


def test_ml_stack_imports():
    """flax and optax import — the M3 emulator's dependencies."""
    import flax
    import optax

    assert flax.__version__
    assert optax.__version__


# ------------------------------------------------------------ robust.rt -----


def test_robust_rt_imports():
    """``from robust import rt`` succeeds."""
    from robust import rt

    assert rt.__doc__


def test_rt_exposes_scaffold_modules():
    """Every module of the planned layout is importable and re-exported."""
    from robust import rt

    expected = [
        "conventions",
        "types",
        "data",
        "ztt",
        "emulator",
        "hybrid",
        "validation",
    ]

    for name in expected:
        assert hasattr(rt, name), f"robust.rt.{name} missing"
    assert hasattr(rt.data, "l23")
    # The public entry point of the design is exported from day one.
    assert callable(rt.forward)


def test_no_stubs_remain():
    """Every designed callable is implemented; nothing raises "not yet written".

    This test asserted ``NotImplementedError`` from each stub as long as any
    remained: ``ztt.Rrs_ZTT`` came off the list at M2, and ``forward`` — the
    last one — at M3, so what is left to pin is that the scaffold era is over.
    ``mu_infinity`` still raises, but because the published paper omits its
    coefficients — a data gap, not an unwritten function — and ``test_ztt.py``
    checks it. ``robust.rt.hybrid`` now has its own tests (``test_hybrid.py``).
    """
    import jax.numpy as jnp

    from robust import rt

    wave = jnp.asarray([440.0, 550.0])
    iops = rt.IOPs(
        a=jnp.asarray([0.15, 0.12]),
        bb_w=rt.conventions.bb_w(wave),
        bb_p=jnp.asarray([0.003, 0.003]),
    )
    Rrs = rt.forward(
        iops,
        rt.PhaseParams(B_p=jnp.asarray(0.0126)),
        rt.Geometry.nadir(jnp.asarray(30.0)),
        wave,
    )

    assert np.all(np.isfinite(np.asarray(Rrs)))


def test_hybrid_modes_declared():
    """The three comparable configurations are named (design §4.5)."""
    from robust import rt

    assert rt.hybrid.MODES == ("ztt", "emulator", "hybrid")


# ------------------------------------------------------- reference data -----


def test_l23_availability_check_runs():
    """The conftest data probe answers without raising, either way.

    It gates every M1+ data test, so it must never be the thing that breaks;
    a machine with no ``$OS_COLOR`` mount should get ``False``, not an error.
    """
    assert isinstance(l23_available(), bool)


def test_ocpy_l23_loader_importable():
    """``ocpy``'s L23 loader is installed — M1 builds on it, not its own copy."""
    from ocpy.hydrolight import loisel23

    assert hasattr(loisel23, "l23_path")


# ------------------------------------ the frozen forward API (M5 task 15) ----
# `robust.rt.forward` is the shared engine for training-data generation and for
# the separately designed inversion. Both need it to stop moving. These tests are
# the freeze: they fail on any change to the call surface, so a change becomes a
# deliberate act with a diff attached rather than something noticed downstream.
#
# What the freeze permits and forbids is stated in the implementation record §8.
# In short: adding a keyword-only argument with a behaviour-preserving default is
# permitted, and so is adding a `None`-defaulted field to one of the pytree
# containers (M1 designed for it; M5 task 14 did it). Renaming, reordering,
# removing, or changing what a default *does* are not.

import inspect  # noqa: E402
from pathlib import Path  # noqa: E402

#: The frozen call surface of ``robust.rt.forward``: name -> (kind, default).
#: ``inspect.Parameter.empty`` means "no default", i.e. the caller must supply it.
FROZEN_FORWARD = (
    ("iops", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("phase_params", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("geometry", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("wave", "POSITIONAL_OR_KEYWORD", None),
    ("mode", "POSITIONAL_OR_KEYWORD", "hybrid"),
    ("emulator", "KEYWORD_ONLY", None),
    ("check_domain", "KEYWORD_ONLY", True),
    ("on_out_of_domain", "KEYWORD_ONLY", "warn"),
)

#: The frozen field order of the three argument containers. Order matters as much
#: as membership: these are registered pytrees, so a reordering silently changes
#: what ``tree_flatten`` produces and would misalign anything that zips leaves.
FROZEN_CONTAINERS = {
    "IOPs": ("a", "bb_w", "bb_p"),
    "PhaseParams": ("B_p", "beta_tilde_pi", "backward_slope"),
    "Geometry": ("theta_s", "theta_v", "dphi", "wind"),
}


def test_forward_signature_is_frozen():
    """**The freeze.** Names, order, kind, and defaults of the public entry point."""
    import robust.rt as rt

    observed = tuple(
        (name, p.kind.name, p.default)
        for name, p in inspect.signature(rt.forward).parameters.items()
    )

    assert observed == FROZEN_FORWARD, (
        "robust.rt.forward's call surface changed. That is allowed only "
        "deliberately — see the implementation record §8 for what the freeze "
        "permits (a new keyword-only argument with a behaviour-preserving "
        "default) and forbids (renaming, reordering, removing, or changing what "
        "an existing default does).\n"
        f"expected: {FROZEN_FORWARD}\nobserved: {observed}"
    )


def test_forwards_argument_containers_are_frozen():
    """The pytree field order is part of the API, not an implementation detail."""
    import dataclasses

    from robust.rt import types

    for name, expected in FROZEN_CONTAINERS.items():
        observed = tuple(f.name for f in dataclasses.fields(getattr(types, name)))

        assert observed == expected, (
            f"{name}'s fields changed from {expected} to {observed}. Appending a "
            "field that defaults to None is permitted; reordering or renaming is "
            "not, because these are registered pytrees and the leaf order is "
            "observable."
        )


def test_the_optional_container_fields_all_default_to_none():
    """What makes appending a field safe: it must change nothing by default."""
    import dataclasses

    from robust.rt import types

    for name in FROZEN_CONTAINERS:
        cls = getattr(types, name)
        for field in dataclasses.fields(cls):
            if field.default is dataclasses.MISSING:
                continue
            assert field.default is None, (
                f"{name}.{field.name} defaults to {field.default!r}; an optional "
                "field on a frozen container must default to None, or adding it "
                "changed behaviour for every existing caller"
            )


def test_the_enumerations_the_signature_refers_to_are_frozen():
    """``mode`` and ``on_out_of_domain`` are strings validated against these."""
    from robust.rt import hybrid

    assert hybrid.MODES == ("ztt", "emulator", "hybrid")
    assert hybrid.OUT_OF_DOMAIN_POLICIES == ("warn", "ztt")


def test_forward_is_callable_positionally_as_frozen():
    """The freeze is behavioural too: the pinned order must actually work."""
    import jax.numpy as jnp

    from robust.rt import conventions, types

    wave = conventions.canonical_wave()
    n = wave.shape[0]
    iops = types.IOPs(
        a=jnp.full((2, n), 0.1),
        bb_w=jnp.broadcast_to(conventions.bb_w(wave), (2, n)),
        bb_p=jnp.full((2, n), 3e-3),
    )
    phase = types.PhaseParams(B_p=jnp.full((2, n), 0.012))
    geometry = types.Geometry.nadir(jnp.full(2, 30.0))

    import robust.rt as rt

    out = rt.forward(iops, phase, geometry, wave, "ztt")

    assert out.shape == (2, n)


# --- what the signature freeze does NOT cover, closed here (M5 task 15) ------
# An audit of the freeze found three ways a downstream caller gets hurt with
# every signature test green. These close them.


def test_the_default_path_produces_pinned_numbers(l23_small_batch):
    """**Gap 1.** The signature can be frozen while the *numbers* drift.

    ``forward``'s default path resolves ``emulator=None`` through
    ``load_default()``, so the answer depends on a 6.5 kB file the project has
    explicitly reserved the right to replace (Q13's promotion rule, §7.15). A
    caller generating training data, or an inversion calibrated against it, would
    silently stop matching a later regeneration.

    So the shipped weights are pinned by digest **and** the output is pinned by
    value. The digest says *which* model; the value says the whole path from
    inputs to ``Rrs`` still computes what it computed. Changing the default model
    is allowed — it just has to be a deliberate edit here, with a diff.
    """
    import hashlib

    import jax.numpy as jnp

    import robust.rt as rt
    from robust.rt import emulator

    digest = hashlib.sha256(Path(emulator.DEFAULT_WEIGHTS).read_bytes()).hexdigest()
    assert digest.startswith("a49d2454ae8332df"), (
        "the shipped emulator weights changed. If that is deliberate — a retrain, "
        "or the PB24 model winning L23's held-out split — update this digest and "
        "the golden values below in the same commit."
    )

    batch = l23_small_batch
    out = rt.forward(batch.iops, batch.phase_params, batch.geometry, batch.wave)

    assert out.shape == (batch.n_sample, batch.n_wave)
    assert out.dtype == jnp.float32  # dtype is part of the contract too
    assert float(jnp.mean(out)) == pytest.approx(2.943628933281e-03, rel=1e-6)
    assert float(out[0, 40]) == pytest.approx(1.207069493830e-03, rel=1e-6)


def test_forward_returns_Rrs_at_off_nadir_geometry_too(l23_small_batch):
    """**Gap 2.** The Rrs/rrs convention, pinned where it could actually slip.

    ``forward`` returns above-water ``Rrs``; ``rrs_forward`` returns the
    subsurface value. Every hybrid test runs on L23, which is nadir-only — so
    wiring M5's geometry-aware surface transfer (§7.10) into ``forward`` *for
    off-nadir geometries only* would pass the entire suite while changing every
    number a multi-angular caller sees. This asserts the relationship at an
    off-nadir geometry, which is exactly where that change would hide.
    """
    import jax.numpy as jnp
    import numpy as np

    import robust.rt as rt
    from robust.rt import conventions, hybrid, types

    batch = l23_small_batch
    n = batch.n_sample
    geometry = types.Geometry(
        theta_s=batch.geometry.theta_s,
        theta_v=jnp.full(n, 35.0),
        dphi=jnp.full(n, 80.0),
    )
    args = (batch.iops, batch.phase_params, geometry, batch.wave)

    above = rt.forward(*args, "ztt")
    below = hybrid.rrs_forward(*args, "ztt")

    np.testing.assert_array_equal(
        np.asarray(above), np.asarray(conventions.rrs_to_Rrs(below))
    )


def test_rrs_forward_signature_is_frozen_too():
    """**Gap 3.** ``rrs_forward`` is the engine training and scoring actually use.

    Scoring happens in ``rrs`` (design §6) and the emulator trains there, so a
    downstream consumer is at least as likely to build against ``rrs_forward`` as
    against ``forward``. Freezing only the latter would leave the more load-bearing
    surface free to move.
    """
    from robust.rt import hybrid

    observed = tuple(
        (name, p.kind.name, p.default)
        for name, p in inspect.signature(hybrid.rrs_forward).parameters.items()
    )

    assert observed == FROZEN_FORWARD, (
        f"rrs_forward's call surface diverged from forward's.\n"
        f"expected: {FROZEN_FORWARD}\nobserved: {observed}"
    )


def test_the_emulator_pytree_contract_holds_across_a_transform():
    """**Gap 4.** The static/leaf split the inversion is told to rely on.

    ``Emulator.config`` and ``envelope`` are static; ``params``/``mean``/``std``/
    ``domain`` are leaves. Every test in the suite *closes over* an emulator
    rather than passing it through ``jit``/``grad``, and closures are not
    flattened — so the ``static`` metadata could be dropped and everything would
    stay green until an inversion tried to differentiate w.r.t. the weights.
    """
    import jax
    import jax.numpy as jnp

    from robust.rt import emulator as E

    model = E.load_default()
    leaves = jax.tree_util.tree_leaves(model)

    # config and envelope must NOT be leaves; the four arrays must be.
    assert len(leaves) == len(jax.tree_util.tree_leaves(model.params)) + 3
    assert all(hasattr(leaf, "dtype") for leaf in leaves)

    # and it survives being an *argument* rather than a closure
    def total(em, x):
        return jnp.sum(em._standardise(x))

    x = jnp.zeros((2, len(E.FEATURES)))
    jitted = jax.jit(total)(model, x)
    assert jnp.isfinite(jitted)

    grads = jax.grad(total)(model, x)
    assert grads.config is model.config  # static passes through untouched
    assert grads.envelope == model.envelope
    assert np.asarray(grads.mean).shape == np.asarray(model.mean).shape
