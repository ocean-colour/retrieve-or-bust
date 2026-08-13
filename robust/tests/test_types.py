"""
Tests for :mod:`robust.rt.types` (M1 task 2).

The point of these types is that they are *pytrees*, so most of what is worth
testing is JAX behaviour rather than attribute access: that ``jit``, ``vmap``,
``grad``, and ``tree_map`` all traverse them, and that ``grad`` comes back as the
same container with per-field derivatives -- the shape the future inversion wants.

Two design claims are also tested rather than asserted in prose:

- **``PhaseParams`` is extensible without touching ``forward``.** A local
  M5-shaped variant with an extra optional field is defined and pushed through
  ``jit``/``grad``, so the promise in the design is exercised, not just written
  down.
- **Validation cannot be silently skipped under tracing.** ``validate()`` is
  documented as a boundary check; a test confirms it genuinely fails inside
  ``jit`` rather than passing vacuously, which is what would happen if someone
  later "helpfully" moved it into ``__post_init__``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import types as T

N = C.N_WAVE


def make_iops(batch: int | None = None) -> T.IOPs:
    """A physical IOPs instance, optionally batched over scenes."""
    shape = (N,) if batch is None else (batch, N)
    return T.IOPs(
        a=jnp.full(shape, 0.15),
        bb_w=jnp.broadcast_to(C.bb_w(), shape),
        bb_p=jnp.full(shape, 0.003),
    )


# ------------------------------------------------------------------- IOPs ----


def test_iops_fields_and_derived_quantities():
    """``bb`` and ``u`` are derived from the three stored fields."""
    iops = make_iops()

    np.testing.assert_allclose(
        np.asarray(iops.bb), np.asarray(iops.bb_w) + np.asarray(iops.bb_p), rtol=1e-6
    )
    expected_u = np.asarray(iops.bb) / (np.asarray(iops.a) + np.asarray(iops.bb))
    np.testing.assert_allclose(np.asarray(iops.u), expected_u, rtol=1e-6)
    assert iops.n_wave == N


def test_iops_is_a_pytree_with_three_leaves():
    """Only the stored fields are leaves; ``bb``/``u`` are properties."""
    leaves = jax.tree_util.tree_leaves(make_iops())

    assert len(leaves) == 3


def test_iops_is_frozen():
    """Immutable, so a pytree cannot be mutated behind JAX's back."""
    iops = make_iops()

    with pytest.raises(dataclasses.FrozenInstanceError):
        iops.a = jnp.zeros(N)


def test_iops_replace_is_functional():
    """``dataclasses.replace`` works and leaves the original alone."""
    iops = make_iops()

    updated = dataclasses.replace(iops, a=jnp.full((N,), 0.5))

    assert float(updated.a[0]) == pytest.approx(0.5)
    assert float(iops.a[0]) == pytest.approx(0.15)


def test_from_total_bb_splits_off_water():
    """``bb_p = bb - bb_w``, with ``bb_w`` taken from the conventions module."""
    a = jnp.full((N,), 0.15)
    bb_p_true = jnp.full((N,), 0.003)
    bb_total = C.bb_w() + bb_p_true

    iops = T.IOPs.from_total_bb(a, bb_total)

    np.testing.assert_allclose(np.asarray(iops.bb_w), C.BB_W_L23, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(iops.bb_p), np.asarray(bb_p_true), atol=1e-9)
    np.testing.assert_allclose(np.asarray(iops.bb), np.asarray(bb_total), rtol=1e-6)


def test_from_total_bb_broadcasts_bb_w_over_the_batch():
    """Every leaf shares the batch shape, so plain ``vmap(in_axes=0)`` works.

    This is the reason ``bb_w`` is broadcast rather than stored as ``(81,)``:
    otherwise callers would have to spell out a per-field ``in_axes``.
    """
    a = jnp.full((7, N), 0.15)
    bb = jnp.broadcast_to(C.bb_w(), (7, N)) + 0.003

    iops = T.IOPs.from_total_bb(a, bb)

    assert iops.a.shape == iops.bb_w.shape == iops.bb_p.shape == (7, N)
    # The payoff: no custom in_axes needed.
    out = jax.vmap(lambda x: jnp.sum(x.u))(iops)
    assert out.shape == (7,)


def test_from_total_bb_can_produce_negative_bb_p_and_validate_catches_it():
    """The documented failure mode: passing *non-water* bb as if it were total.

    ``from_total_bb`` does not validate (it must stay traceable), so this is
    exactly the mistake ``validate()`` exists to catch.
    """
    a = jnp.full((N,), 0.15)
    bb_nonwater_only = jnp.full((N,), 1e-4)  # smaller than bb_w in the blue

    iops = T.IOPs.from_total_bb(a, bb_nonwater_only)

    assert float(jnp.min(iops.bb_p)) < 0.0
    with pytest.raises(ValueError, match="negative"):
        iops.validate()


def test_iops_validate_accepts_physical_values():
    make_iops().validate()
    make_iops(batch=3).validate(wave=C.canonical_wave())


def test_iops_validate_rejects_shape_mismatch():
    iops = T.IOPs(a=jnp.ones(N), bb_w=jnp.ones(N - 1), bb_p=jnp.ones(N))

    with pytest.raises(ValueError, match="shapes must match"):
        iops.validate()


def test_iops_validate_rejects_nonfinite():
    iops = dataclasses.replace(make_iops(), a=jnp.full((N,), jnp.nan))

    with pytest.raises(ValueError, match="non-finite"):
        iops.validate()


def test_iops_validate_rejects_wrong_wave_length():
    """A short spectrum is caught when the grid is supplied for cross-checking."""
    iops = T.IOPs(a=jnp.ones(10), bb_w=jnp.ones(10), bb_p=jnp.ones(10))

    with pytest.raises(ValueError, match="canonical grid"):
        iops.validate(wave=C.canonical_wave())


# -------------------------------------------------------------- PhaseParams --


def test_phase_params_is_a_pytree():
    params = T.PhaseParams(B_p=jnp.asarray(0.0126))

    assert jax.tree_util.tree_leaves(params) == [params.B_p]


def test_phase_params_validate_accepts_realistic_values():
    T.PhaseParams(B_p=jnp.asarray(0.0126)).validate()
    T.PhaseParams(B_p=jnp.linspace(0.005, 0.03, N)).validate()


@pytest.mark.parametrize("bad", [0.0, -0.01, 1.5])
def test_phase_params_validate_rejects_non_ratios(bad):
    """``B_p`` is a ratio in (0, 1], not a coefficient."""
    with pytest.raises(ValueError):
        T.PhaseParams(B_p=jnp.asarray(bad)).validate()


def test_phase_params_validate_allows_values_outside_the_l23_range():
    """A synthetic sweep may leave ~[0.004, 0.03]; that is the loader's business.

    Keeping the tight range out of the type means M2/M3 can sweep ``B_p`` to probe
    the model without fighting a type-level invariant.
    """
    T.PhaseParams(B_p=jnp.asarray(0.3)).validate()


def test_phase_params_extends_without_changing_the_forward_signature():
    """The M5 shape: an extra optional field, defaulting to None.

    Exercises the design's extensibility claim -- ``forward(iops, phase_params,
    geometry, wave)`` is unchanged, the container simply carries more.
    """

    @jax.tree_util.register_dataclass
    @dataclass(frozen=True)
    class ZTTPhaseParams:
        B_p: jax.Array
        beta_tilde_pi: jax.Array | None = None

    minimal = ZTTPhaseParams(B_p=jnp.asarray(0.0126))
    extended = ZTTPhaseParams(B_p=jnp.asarray(0.0126), beta_tilde_pi=jnp.asarray(0.5))

    # An unset optional field contributes no leaves...
    assert len(jax.tree_util.tree_leaves(minimal)) == 1
    assert len(jax.tree_util.tree_leaves(extended)) == 2
    # ...but it does change the treedef, so jit recompiles per variant.
    assert jax.tree_util.tree_structure(minimal) != jax.tree_util.tree_structure(
        extended
    )

    # Both traverse jit and grad.
    assert float(jax.jit(lambda p: p.B_p * 2.0)(extended)) == pytest.approx(0.0252)
    g = jax.grad(lambda p: p.B_p * 2.0)(extended)
    assert float(g.B_p) == pytest.approx(2.0)
    assert g.beta_tilde_pi is None or float(g.beta_tilde_pi) == pytest.approx(0.0)


# ----------------------------------------------------------------- Geometry --


def test_geometry_nadir_constructor():
    """The L23 case: nadir view, so only the solar zenith varies."""
    geom = T.Geometry.nadir(jnp.asarray(30.0))

    assert float(geom.theta_s) == pytest.approx(30.0)
    assert float(geom.theta_v) == 0.0
    assert float(geom.dphi) == 0.0
    assert geom.wind is None


def test_geometry_optional_wind_changes_the_leaf_count():
    """``wind=None`` contributes no leaves; supplying it adds one."""
    without = T.Geometry.nadir(jnp.asarray(30.0))
    with_wind = T.Geometry.nadir(jnp.asarray(30.0), wind=jnp.asarray(5.0))

    assert len(jax.tree_util.tree_leaves(without)) == 3
    assert len(jax.tree_util.tree_leaves(with_wind)) == 4


def test_geometry_validate_accepts_the_l23_zeniths():
    for theta_s in (0.0, 30.0, 60.0):
        T.Geometry.nadir(jnp.asarray(theta_s)).validate()
    T.Geometry.nadir(jnp.asarray(30.0), wind=jnp.asarray(5.0)).validate()


def test_validate_cannot_detect_radians_a_known_limitation():
    """Pins a gap rather than a guarantee, so nobody relies on the wrong thing.

    30 degrees expressed in radians is 0.52, which sits happily inside [0, 90],
    so the range check **cannot** see that mix-up -- it would surface at M3 as a
    poor fit, not as an error. What the check does catch is the reverse mistake:
    feeding degrees to something expecting radians, or any angle past the
    horizon. Documented here because a validator's blind spots matter as much as
    its coverage.
    """
    T.Geometry.nadir(jnp.asarray(np.deg2rad(30.0))).validate()  # passes: 0.52 deg

    with pytest.raises(ValueError, match="degrees"):
        T.Geometry.nadir(jnp.asarray(95.0)).validate()  # past the horizon


@pytest.mark.parametrize(
    ("field", "value"),
    [("theta_s", 91.0), ("theta_s", -1.0), ("theta_v", 120.0), ("dphi", 400.0)],
)
def test_geometry_validate_rejects_out_of_range_angles(field, value):
    kwargs = {
        "theta_s": jnp.asarray(30.0),
        "theta_v": jnp.asarray(0.0),
        "dphi": jnp.asarray(0.0),
    }
    kwargs[field] = jnp.asarray(value)

    with pytest.raises(ValueError, match=field):
        T.Geometry(**kwargs).validate()


def test_geometry_validate_rejects_negative_wind():
    with pytest.raises(ValueError, match="negative"):
        T.Geometry.nadir(jnp.asarray(30.0), wind=jnp.asarray(-1.0)).validate()


# ------------------------------------------------- JAX transforms end-to-end --


def test_grad_returns_the_same_container_with_per_field_derivatives():
    """The property the future inversion is built on.

    ``jax.grad`` of a scalar of an ``IOPs`` is an ``IOPs``, so sensitivities stay
    labelled instead of arriving as an anonymous flat vector.
    """
    iops = make_iops()

    grads = jax.grad(lambda x: jnp.sum(x.u))(iops)

    assert isinstance(grads, T.IOPs)
    assert grads.a.shape == iops.a.shape
    # More absorption lowers u; more particulate backscatter raises it.
    assert float(grads.a[0]) < 0.0
    assert float(grads.bb_p[0]) > 0.0


def test_jit_and_tree_map_traverse_the_containers():
    iops = make_iops()

    total = jax.jit(lambda x: jnp.sum(x.bb))(iops)
    doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, iops)

    assert float(total) > 0.0
    np.testing.assert_allclose(
        np.asarray(doubled.bb_p), 2.0 * np.asarray(iops.bb_p), rtol=1e-6
    )


def test_geometry_and_phase_params_pass_through_jit_together():
    """All three containers in one traced call, as ``forward`` will take them."""
    iops = make_iops()
    params = T.PhaseParams(B_p=jnp.asarray(0.0126))
    geom = T.Geometry.nadir(jnp.asarray(30.0))

    @jax.jit
    def summarize(iops, params, geom):
        return jnp.sum(iops.u) * params.B_p * jnp.cos(jnp.deg2rad(geom.theta_s))

    assert float(summarize(iops, params, geom)) > 0.0


def test_types_are_exported_from_the_package_root():
    """``robust.rt`` re-exports the public types, per the coding-plan layout."""
    from robust import rt

    assert rt.IOPs is T.IOPs
    assert rt.PhaseParams is T.PhaseParams
    assert rt.Geometry is T.Geometry
    for name in ("IOPs", "PhaseParams", "Geometry"):
        assert name in rt.__all__


def test_validate_is_not_usable_under_jit():
    """Confirms the documented boundary-only contract.

    If a future change moved validation into ``__post_init__``, it would run on
    tracers -- either raising here or, worse, passing vacuously. This test pins
    the current, deliberate behaviour: validation is for concrete data.
    """
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.jit(lambda x: (x.validate(), 0.0)[1])(make_iops())


# ------------------------------------------------- a second wavelength grid --
# M5 task 3. `IOPs.validate` used to compare the trailing axis against
# `conventions.N_WAVE`, which is what made it L23-only.


def test_validate_accepts_a_pb24_grid_batch():
    """A 12-band OLCI batch validates against its own grid."""
    wave = C.grid_wave("olci")
    n = C.OLCI_GRID.n_wave
    iops = T.IOPs(
        a=jnp.full((4, n), 0.05),
        bb_w=jnp.full((4, n), 1e-3),
        bb_p=jnp.full((4, n), 2e-3),
    )

    iops.validate(wave, grid="olci")  # must not raise


def test_validate_still_refuses_the_wrong_grid():
    """The check moved, it did not soften: OLCI data against L23 still fails."""
    wave = C.grid_wave("olci")
    n = C.OLCI_GRID.n_wave
    iops = T.IOPs(
        a=jnp.full((2, n), 0.05),
        bb_w=jnp.full((2, n), 1e-3),
        bb_p=jnp.full((2, n), 2e-3),
    )

    with pytest.raises(ValueError, match="expected the canonical grid"):
        iops.validate(wave)


def test_validate_catches_a_trailing_axis_that_lies_about_its_grid():
    """Right grid, wrong number of bands in the arrays."""
    wave = C.grid_wave("olci")
    iops = T.IOPs(
        a=jnp.full((2, 11), 0.05),
        bb_w=jnp.full((2, 11), 1e-3),
        bb_p=jnp.full((2, 11), 2e-3),
    )

    with pytest.raises(ValueError, match="does not match the olci grid"):
        iops.validate(wave, grid="olci")


def test_from_total_bb_default_is_the_old_clamp_exactly():
    """The M0-M4 path is bit-identical; the mode only changes what you ask for."""
    wave = C.grid_wave("olci")
    a = jnp.full((3, C.OLCI_GRID.n_wave), 0.05)
    bb = jnp.full((3, C.OLCI_GRID.n_wave), 5e-3)

    clamped = T.IOPs.from_total_bb(a, bb, wave)
    explicit = T.IOPs.from_total_bb(a, bb, wave, bb_w_mode="clamp")
    extrapolated = T.IOPs.from_total_bb(a, bb, wave, bb_w_mode="extrapolate")

    np.testing.assert_array_equal(np.asarray(clamped.bb_w), np.asarray(explicit.bb_w))
    # they differ only in the one band past the table, and only there
    differs = np.asarray(clamped.bb_w[0] != extrapolated.bb_w[0])
    assert differs.sum() == 1
    assert differs[-1]  # 753 nm


def test_validate_checks_the_m5_phase_fields():
    """**Regression (M5 review, L5).** They flow into Pbb, so they need checking.

    A negative ``beta_tilde_pi`` produces a negative backward phase function and
    a negative ``rrs``, with no symptom anywhere — the silent path ``validate()``
    exists to close for ``B_p``.
    """
    base = dict(B_p=jnp.asarray(0.012))

    T.PhaseParams(**base, beta_tilde_pi=jnp.asarray(0.15)).validate()
    T.PhaseParams(**base, backward_slope=jnp.asarray(-0.5)).validate()  # sign is fine

    with pytest.raises(ValueError, match="beta_tilde_pi"):
        T.PhaseParams(**base, beta_tilde_pi=jnp.asarray(-0.1)).validate()
    with pytest.raises(ValueError, match="backward_slope"):
        T.PhaseParams(**base, backward_slope=jnp.asarray(jnp.nan)).validate()
