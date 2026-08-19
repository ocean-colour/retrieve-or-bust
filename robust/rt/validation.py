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

import dataclasses
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import types

__all__ = [  # noqa: RUF022  - grouped by role
    "rrms",
    "rrms_per_wavelength",
    "group_rrms",
    "bp_bin_labels",
    "throughput",
    "gradient_report",
    "FD_STEPS",
    "GRADIENT_TOL",
    "score_models",
    "markdown_table",
]

#: Per-variable central-difference steps for :func:`gradient_report`.
#:
#: Measured at M2 and unchanged since: no single step clears the tolerance for all
#: four, because ``theta_s`` is O(30) degrees while the IOP-like variables are
#: O(1e-3) m^-1. Too large a step also leaves the physical domain -- ``bb_p`` goes
#: negative and the model returns NaN.
FD_STEPS = {"a": 1e-6, "bb_p": 1e-9, "B_p": 1e-8, "theta_s": 1e-3}

#: The tolerance the gradient gate has held to since M2, in relative terms.
GRADIENT_TOL = 1e-6


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


# ------------------------------------------------------------------ breakdowns


def rrms_per_wavelength(
    truth: Float[Array, "sample wave"], pred: Float[Array, "sample wave"]
) -> Float[Array, " wave"]:
    """The per-λ error ladder the design asks results to be reported as.

    ``rrms(..., axis=0)`` under a name, because "rRMS per wavelength" is a named
    object in this project: it is what shows whether a model buys a good total by
    fixing the bright blue and abandoning the dark red. Gordon's ladder does exactly
    that (2.5% at 400 nm rising to 9.0% at 700 nm, ``context/RT/fig_rrms_ladder.csv``),
    which is why a single scalar was never enough.

    Parameters
    ----------
    truth, pred : Array
        Shape ``(n_sample, n_wave)``.

    Returns
    -------
    Array
        rRMS in percent, shape ``(n_wave,)``.
    """
    return rrms(truth, pred, axis=0)


def group_rrms(
    truth: Float[Array, "sample wave"],
    pred: Float[Array, "sample wave"],
    labels: np.ndarray,
) -> dict[float, float]:
    """rRMS within each group of samples — the per-zenith and per-``B_p``-bin cut.

    Parameters
    ----------
    truth, pred : Array
        Shape ``(n_sample, n_wave)``.
    labels : numpy.ndarray
        Per-sample group label, shape ``(n_sample,)``. Host-side (it is metadata,
        not something to differentiate), e.g. ``batch.zenith`` or the output of
        :func:`bp_bin_labels`.

    Returns
    -------
    dict
        ``{label: rRMS percent}``, in ascending label order. Empty groups are
        omitted rather than reported as NaN.
    """
    labels = np.asarray(labels)
    out = {}
    for value in np.unique(labels):
        mask = labels == value
        if not mask.any():
            continue
        out[float(value)] = float(rrms(truth[mask], pred[mask]))
    return out


def bp_bin_labels(
    B_p: Float[Array, "sample wave"], n_bins: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Bin samples by their mean ``B_p``, into equal-count bins.

    **Quantile bins, not equal-width**, because L23's ``B_p`` spans only a factor
    1.75 (0.0103–0.0180) against the design's ~7× nominal band: equal-width bins over
    so narrow a range would put nearly every sample in the middle two and report the
    outer two on a handful of scenes. Equal counts at least make each row's error bar
    comparable.

    This breakdown is required by design §6, but read it with the range in mind: it
    cannot speak to phase-function *generalisation*, only to whether accuracy varies
    across the narrow slice L23 happens to cover.

    Parameters
    ----------
    B_p : Array
        Particulate backscattering ratio, shape ``(n_sample, n_wave)`` or
        ``(n_sample,)``.
    n_bins : int, optional
        Number of quantile bins.

    Returns
    -------
    labels : numpy.ndarray
        Bin index per sample, shape ``(n_sample,)``.
    edges : numpy.ndarray
        The ``n_bins + 1`` bin edges, so a table can name the range it is reporting.

    Notes
    -----
    "Equal count" holds only when the values are mostly distinct. Heavy ties collapse
    quantile edges and can leave a bin empty — with 40% of samples sharing the
    minimum, the counts come out ``[0, 40, 30, 30]``, and identical values put
    everything in the last bin. L23 is well behaved here (exactly 2490 per bin on the
    full batch), but check the counts before reading a per-bin table from other data.
    """
    values = np.asarray(B_p)
    if values.ndim > 1:
        values = values.mean(axis=tuple(range(1, values.ndim)))
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    # right=False with the final edge nudged so the maximum lands in the last bin.
    labels = np.clip(np.digitize(values, edges[1:-1], right=False), 0, n_bins - 1)
    return labels, edges


# ------------------------------------------------------------------ fast, and
# ------------------------------------------------------------- differentiable


def throughput(model, *args, repeats: int = 5) -> tuple[float, float]:
    """Seconds per jitted call, and samples·λ per second.

    Parameters
    ----------
    model : callable
        ``model(*args) -> Array`` of shape ``(n_sample, n_wave)``.
    *args
        Passed straight through; the first is assumed to carry the batch.
    repeats : int, optional
        Timed calls after one warm-up (which pays the compilation).

    Returns
    -------
    seconds : float
        Mean wall time per call.
    rate : float
        ``n_sample * n_wave / seconds``.

    Notes
    -----
    Wall-clock on a shared machine wanders ~20% between runs, so **report the ratio
    between models rather than the milliseconds** — that is the number that
    reproduces. The compile is deliberately outside the timed region: it is paid once
    per process, and including it would measure XLA rather than the model.
    """
    if repeats < 1:
        raise ValueError(f"throughput: repeats must be >= 1; got {repeats}")
    compiled = jax.jit(model)
    out = compiled(*args)
    out.block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        out = compiled(*args)
        out.block_until_ready()
    seconds = (time.perf_counter() - start) / repeats
    return seconds, float(out.size) / seconds


def gradient_report(
    model,
    iops,
    phase_params,
    geometry,
    wave,
    *,
    steps: dict[str, float] | None = None,
) -> dict[str, float]:
    """``jax.grad`` versus central finite differences, per input variable.

    The differentiability half of the design's three axes, as a function rather than
    a test, so the acceptance gate and the report both call the same code.

    Parameters
    ----------
    model : callable
        ``model(iops, phase_params, geometry, wave) -> Array``.
    iops, phase_params, geometry, wave
        A **small** batch — a few samples is plenty, and each variable costs two
        extra forward passes. The geometry is carried through intact apart from
        ``theta_s``: an earlier version rebuilt it with ``Geometry.nadir``, silently
        discarding the caller's ``theta_v``/``dphi``, which would have certified the
        gradient at the wrong geometry the moment M5 goes off-nadir.
    steps : dict, optional
        Per-variable finite-difference step. Defaults to :data:`FD_STEPS`.

    Returns
    -------
    dict
        ``{variable: relative difference}`` for ``a``, ``bb_p``, ``B_p``,
        ``theta_s``. ``inf`` where the finite difference left the physical domain
        and returned a non-finite value.

    Notes
    -----
    **Run this under float64** (``jax.config.update("jax_enable_x64", True)``) and
    pass arrays already cast to it: in float32 the differencing noise swamps the
    comparison. The steps differ per variable by an order of magnitude or more —
    ``theta_s`` is O(30) and wants ~1e-3 while ``bb_p`` is O(1e-3) and wants ~1e-9 —
    because no single step clears the tolerance for all four (M2 measured this).
    A step larger than the variable can drive ``bb_p`` negative, where the model
    returns NaN; that is a bad *step*, not a bad gradient, so it is reported as
    ``inf`` rather than silently compared. A variable the model genuinely ignores
    reports ``0.0`` -- both derivatives are exactly zero, which is agreement.

    **Lookup-table models need care at their nodes.** ``o25_coefficients`` is
    piecewise linear in ``theta_s``, so the tabulated angles are kinks: autodiff
    returns one one-sided slope while the central difference averages both, and they
    disagree by O(1). L23's angles *are* the nodes, so evaluate that variable at an
    intermediate angle (45 deg, say) before concluding anything about O25's
    gradient.
    """
    steps = dict(FD_STEPS if steps is None else steps)
    if set(steps) != set(FD_STEPS):
        # A missing key used to raise KeyError deep inside the closure; an extra one
        # was worse -- it reported 0.0, i.e. "perfect agreement", for a variable that
        # is never perturbed at all.
        raise ValueError(
            f"gradient_report: steps must name exactly {sorted(FD_STEPS)}; "
            f"got {sorted(steps)}"
        )
    a0, bb_w0, bb_p0 = iops.a, iops.bb_w, iops.bb_p
    B_p0, theta0 = phase_params.B_p, geometry.theta_s

    def scalar(shift, *, name):
        """Mean model output with one variable shifted by ``shift``."""
        offsets = dict.fromkeys(steps, 0.0)
        offsets[name] = shift
        return jnp.mean(
            model(
                types.IOPs(
                    a=a0 + offsets["a"], bb_w=bb_w0, bb_p=bb_p0 + offsets["bb_p"]
                ),
                types.PhaseParams(B_p=B_p0 + offsets["B_p"]),
                dataclasses.replace(geometry, theta_s=theta0 + offsets["theta_s"]),
                wave,
            )
        )

    report = {}
    for name, step in steps.items():
        zero = jnp.zeros((), dtype=jnp.asarray(a0).dtype)
        analytic = float(jax.grad(partial(scalar, name=name))(zero))
        h = jnp.asarray(step, dtype=zero.dtype)
        numeric = float((scalar(h, name=name) - scalar(-h, name=name)) / (2.0 * h))
        if not np.isfinite(numeric):
            # The step left the physical domain (bb_p driven negative -> NaN). A bad
            # step, not a bad gradient, so it is flagged rather than compared.
            report[name] = float("inf")
        elif numeric == 0.0:
            # A model that genuinely does not depend on this variable: O25 ignores
            # B_p, so both sides are exactly zero and that is perfect agreement. A
            # ratio would have reported it as infinitely wrong.
            report[name] = 0.0 if analytic == 0.0 else float("inf")
        else:
            report[name] = abs(analytic / numeric - 1.0)
    return report


# ------------------------------------------------------------------- reporting


def score_models(
    models: dict, truth: Float[Array, "sample wave"], masks: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """rRMS of every model on every split, on identical data.

    Parameters
    ----------
    models : dict
        ``{name: prediction array}``, each shape ``(n_sample, n_wave)`` in ``rrs``
        space. Predictions rather than callables, so every model is evaluated once
        on the full batch and then sliced — which also guarantees that "identical
        data" is literally true rather than a claim about two code paths.
    truth : Array
        Reference ``rrs``, shape ``(n_sample, n_wave)``.
    masks : dict
        ``{split name: boolean mask}`` over the sample axis.

    Returns
    -------
    dict
        ``{model name: {split name: rRMS percent}}``.
    """
    return {
        name: {
            split: float(rrms(truth[mask], pred[mask])) for split, mask in masks.items()
        }
        for name, pred in models.items()
    }


def markdown_table(rows: list[list], headers: list[str]) -> str:
    """A GitHub-flavoured markdown table, right-aligned for numbers.

    Parameters
    ----------
    rows : list of list
        Cell values; numbers are formatted to two decimals, everything else
        stringified.
    headers : list of str

    Returns
    -------
    str
    """

    def cell(value):
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)
