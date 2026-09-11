"""
Differentiable elastic radiative-transfer forward model
=======================================================

A fast, accurate, **differentiable** map ``Rrs(λ; a, bb, phase function,
geometry)``, built in JAX as the hybrid

    Rrs = Rrs_ZTT(θ)  +  ΔRrs_emulator(θ)

where ``Rrs_ZTT`` is the Twardowski & Tonizzo (2018) analytic backbone — with an
*explicit* phase-function dependence — and ``ΔRrs`` is a small learned residual
(multiple scattering and phase-function effects the backbone misses).

The inelastic extension (design ``design/rt_inelastic_model.md``) rides on
top: ``forward(..., inelastic=None)`` with the
:class:`~robust.rt.types.Inelastic` configuration pytree. ``inelastic=None``
(the default) is bit-identical to the elastic hybrid by construction; an
instance composes Raman, chlorophyll fluorescence, and (when
``cdom_fl`` is set) CDOM fluorescence.

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
ed
    ``Ed(theta_s, lambda)`` from packaged L23 spectra + the ``Geometry.Ed``
    override — consumed by the inelastic terms only (M1).
inelastic
    The analytic inelastic terms: ``raman_factor`` and
    ``fluorescence_kernel`` (M2).
cdom_fl
    The analytic CDOM-fluorescence term: ``eta_hawes``, ``cdom_kernel``,
    the 350 nm-clamp diagnostic (M5).
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

**Status.** The elastic Week-1 prototype is **complete** — every submodule
above is implemented and `forward()` is the working hybrid (elastic record
§6). The inelastic Raman + chlorophyll-fluorescence prototype (M0–M4) is
likewise **complete and gate-passed** — held-out 0.34 % rRMS vs the X4
truth, bit-identical elastic off-state (``reports/report_rt_inelastic_model.md``,
v1.0). M5 adds **CDOM fluorescence** as a third inelastic term
(``design/rt_cdom_fluorescence_model.md``): **analytic-only** (the Hawes FA7
kernel; the δ_C head is defined but untrained), **default-off**
(``Inelastic(cdom_fl=None)`` — the default — stays bit-identical to the
shipped inelastic model, because the X4 truth omits CDOM-fl), and
**unvalidated until M6**, pending the design-§7 HydroLight truth runs.
"""

from . import (
    baselines,
    cdom_fl,
    conventions,
    data,
    ed,
    emulator,
    hybrid,
    inelastic,
    types,
    validation,
    ztt,
)
from .hybrid import forward
from .types import CDOMFl, Geometry, Inelastic, IOPs, PhaseParams

# Grouped by role, and ordered as the pipeline runs (conventions -> data ->
# backbone -> emulator -> hybrid -> validation), not alphabetically: the order is
# the point. Hence the noqa.
__all__ = [  # noqa: RUF022
    # Submodules
    "conventions",
    "types",
    "data",
    "ed",
    "inelastic",
    "cdom_fl",
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
    "CDOMFl",
]
