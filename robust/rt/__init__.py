"""
Differentiable elastic radiative-transfer forward model
=======================================================

A fast, accurate, **differentiable** map ``Rrs(λ; a, bb, phase function,
geometry)``, built in JAX as the hybrid

    Rrs = Rrs_ZTT(θ)  +  ΔRrs_emulator(θ)

where ``Rrs_ZTT`` is the Twardowski & Tonizzo (2018) analytic backbone — with an
*explicit* phase-function dependence — and ``ΔRrs`` is a small learned residual
(multiple scattering and phase-function effects the backbone misses).

Elastic only: no Raman, no fluorescence.

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
    The accuracy / speed / gradient protocol.

**Status.** ``conventions`` and ``types`` are implemented (M1). The rest are
documented stubs whose callables raise :class:`NotImplementedError` naming the
milestone that fills them in — ``data.l23`` at M1, ``ztt`` at M2, ``emulator`` and
``hybrid`` at M3, ``validation`` at M4. The signatures are already those of the
design, so nothing downstream has to change as the bodies land.
"""

from . import conventions, data, emulator, hybrid, types, validation, ztt
from .hybrid import forward
from .types import Geometry, IOPs, PhaseParams

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
    # Public API
    "forward",
    # Public types — the arguments of forward(), in signature order
    "IOPs",
    "PhaseParams",
    "Geometry",
]
