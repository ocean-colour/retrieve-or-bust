"""
Differentiable elastic radiative-transfer forward model
=======================================================

A fast, accurate, **differentiable** map ``Rrs(λ; a, bb, phase function,
geometry)``, built in JAX as the hybrid

    Rrs = Rrs_ZTT(θ)  +  ΔRrs_emulator(θ)

where ``Rrs_ZTT`` is the Twardowski & Tonizzo (2018) analytic backbone — with an
*explicit* phase-function dependence — and ``ΔRrs`` is a small learned residual
(multiple scattering and phase-function effects the backbone misses).

Elastic physics only so far — but the *interface* now carries the inelastic
extension (design ``design/rt_inelastic_model.md``, M0): ``forward(...,
inelastic=None)`` with the :class:`~robust.rt.types.Inelastic` configuration
pytree. ``inelastic=None`` is bit-identical to the elastic hybrid by
construction; passing an instance raises until M2 lands Raman + fluorescence.

Design    : ``design/rt_elastic_model.md``
Plan      : ``design/rt_elastic_model_coding_plan.md``
Built log : ``design/rt_elastic_implementation.md``

Submodules
----------
conventions
    ``Rrs`` <-> ``rrs`` (A = 0.52, B = 1.7), the wavelength grid, ``bb_w(λ)``.
types
    ``IOPs`` / ``PhaseParams`` / ``Geometry`` pytrees.
data.l23
    Loisel+2023 (L23) elastic reference batches, via ``ocpy``.
ztt
    ``Rrs_ZTT`` — the analytic backbone.
emulator
    The Flax residual MLP and its Optax training loop.
hybrid
    ``forward()`` — the public entry point.
validation
    The accuracy / speed / gradient protocol; rrms from M2.
baselines
    Comparison models the hybrid must beat -- standard Gordon (M2), PR05/O25 (M4).

**Status.** ``conventions``, ``types``, and ``data.l23`` are implemented (M1);
``baselines`` (standard Gordon) and ``validation.rrms`` landed with M2. Still
documented stubs whose callables raise :class:`NotImplementedError`: ``ztt`` (M2),
``emulator`` and ``hybrid`` (M3), and the rest of ``validation`` (M4). The
signatures are already those of the design, so nothing downstream has to change as
the bodies land.
"""

from . import baselines, conventions, data, emulator, hybrid, types, validation, ztt
from .hybrid import forward
from .types import Geometry, Inelastic, IOPs, PhaseParams

# Grouped by role, and ordered as the pipeline runs (conventions -> data ->
# backbone -> emulator -> hybrid -> validation), not alphabetically: the order is
# the point. Hence the noqa.
__all__ = [  # noqa: RUF022
    # Submodules
    "conventions",
    "types",
    "data",
    "ztt",
    "emulator",
    "hybrid",
    "validation",
    "baselines",
    # Public API
    "forward",
    # Public types — the arguments of forward(), in signature order
    "IOPs",
    "PhaseParams",
    "Geometry",
    "Inelastic",
]
