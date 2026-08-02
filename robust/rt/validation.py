"""
Validation protocol — **mostly a stub (lands in M4)**; ``rrms`` is live from M2.

Scores the forward model on the three axes that define acceptance: *accurate*,
*fast*, *differentiable*. Deliberately a **protocol, not a target** — no blind
absolute numbers, consistent with the project's unbiased-uncertainty stance.
Absolute rRMS and latency are reported; only the *relative* comparison is gated.

Planned contents (design §6, coding plan M4)
--------------------------------------------
- **Accurate.** rRMS in ``rrs`` space, relatively weighted, broken out per λ, per
  solar zenith, and per ``B_p`` bin — always alongside Gordon, PR05, and O25 on
  the same splits, plus the two held-out splits (random 20% of scenes; the unseen
  60° zenith).
- **Fast.** Throughput (scenes·λ/s, batched) and single-call latency, so it is
  visible if the emulator erases the speed advantage over calling an RT solver.
- **Differentiable.** ``jax.grad`` versus central finite differences w.r.t. ``a``,
  ``bb_p``, ``B_p``, and geometry, across a random batch. A hard gate: it is the
  property the future inversion depends on.

The M4 acceptance gate — hybrid beats standard Gordon on **both** held-out splits
*and* passes the gradient check — is what makes the week-1 prototype "done".
Run-and-figure scripts live in ``design/py/``, outside the package.

**Why :func:`rrms` is here already.** M2 needs it to score its analytic backbone
against the Gordon baseline, and the *whole point* of the metric is that one
definition is shared: the number in the M2 log, the M4 table, and the synthesis
figures must be the same quantity or none of the comparisons mean anything. So it
lands with M2 rather than being written twice.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["rrms"]


def rrms(
    truth: Float[Array, "..."],
    pred: Float[Array, "..."],
    axis: int | tuple[int, ...] | None = None,
) -> Float[Array, "..."]:
    """Relative RMS error, in percent.

    ``100 * sqrt(mean(((pred - truth) / truth)**2))``

    The **relative** form is deliberate and is the project's standing convention
    (design §6). ``Rrs`` spans more than a decade across the spectrum -- L23 runs
    from ~2.5e-2 in the blue to ~6e-6 in the red -- so an *absolute* RMS would be
    almost entirely a statement about the blue, and a model could look excellent
    while being useless past 600 nm. It is also the definition used by BING and by
    ``context/RT/make_rt_elastic_figures.py``, so numbers here are directly
    comparable with the rRMS ladder in ``context/RT/fig_rrms_ladder.csv``.

    Parameters
    ----------
    truth : Array
        Reference values. Must be non-zero: this metric divides by them.
    pred : Array
        Model values, broadcastable against ``truth``.
    axis : int or tuple of int, optional
        Axis or axes to reduce over. ``None`` (default) reduces everything to a
        scalar; ``axis=0`` over a ``(sample, wave)`` array gives the per-wavelength
        ladder, which is how the design asks for it to be reported.

    Returns
    -------
    Array
        Relative RMS in percent.

    Notes
    -----
    Score in ``rrs`` space, not ``Rrs`` (design §6). The two are not
    interchangeable: the interface conversion is non-linear, so a 6-14% departure
    from a linear rescaling sits between them over the ocean range (M1's notebook
    §1). Pure JAX and differentiable, so it can double as a training loss at M3.
    """
    residual = (pred - truth) / truth
    return 100.0 * jnp.sqrt(jnp.mean(residual**2, axis=axis))
