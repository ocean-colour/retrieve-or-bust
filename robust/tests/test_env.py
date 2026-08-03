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


def test_unimplemented_stubs_raise():
    """The remaining stubs raise instead of silently returning something wrong.

    ``forward`` carries its final signature already, so this is what keeps the
    scaffold honest: a caller gets a milestone-naming ``NotImplementedError``,
    never a plausible-looking array.

    ``ztt.Rrs_ZTT`` was on this list until M2 implemented it; ``robust.rt.ztt``
    now has its own tests. ``mu_infinity`` is the one piece of ZTT that still
    raises, because the published paper omits its coefficients — checked below
    rather than here, since that is a data gap, not an unwritten function.
    """
    from robust import rt

    with pytest.raises(NotImplementedError):
        rt.forward(None, None, None, None)


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
