"""
Validation protocol — the design-§6 machinery for both the elastic and the
inelastic acceptance gates.

Scores the forward model on the three axes that define acceptance: *accurate*,
*fast*, *differentiable*. Deliberately a **protocol, not a target** — no blind
absolute numbers, consistent with the project's unbiased-uncertainty stance.
Absolute rRMS and latency are reported; only the *relative* comparison is gated.

Elastic protocol (elastic design §6, elastic coding plan M4)
------------------------------------------------------------
- **Accurate.** rRMS in ``rrs`` space, relatively weighted, broken out per λ, per
  solar zenith, and per ``B_p`` bin — always alongside Gordon, PR05, and O25 on
  the same splits, plus the two held-out splits (random 20% of scenes; the unseen
  60° zenith).
- **Fast.** Throughput (scenes·λ/s, batched) and single-call latency, so it is
  visible if the emulator erases the speed advantage over calling an RT solver.
- **Differentiable.** ``jax.grad`` versus central finite differences w.r.t. ``a``,
  ``bb_p``, ``B_p``, and geometry, across a random batch. A hard gate: it is the
  property the future inversion depends on.

Inelastic protocol (inelastic design §6, inelastic coding plan M4)
------------------------------------------------------------------
The same three axes over the all-processes-on model, with the truth channel
``Rrs_X4`` and the elastic splits reused verbatim:

- **Accurate.** Held-out (by scene) total rRMS vs ``Rrs_X4`` in ``rrs`` space —
  :func:`group_rrms` per zenith, :func:`rrms_per_wavelength` per λ — plus the
  per-process delta metrics the M3 gates pinned: :func:`median_increment_error`
  (Raman, vs ``Rrs_X2/Rrs_X1``) and :func:`peak_ratio_error` (fluorescence
  685 nm, vs ``Rrs_X4 − Rrs_X2``). One definition each, shared with
  ``test_inelastic_corr.py`` — the number in the M3 log and the M4 table are the
  same quantity. Diagnostics (reported, not gated): a_ph(440)-decile performance
  via :func:`quantile_bin_labels`, and :func:`phi_c_linearity` — the scaled-truth
  construction, since no varied-φ_C truth exists.
- **Fast.** :func:`speed_ratio` — the inelastic forward against the elastic
  hybrid, same batch, same machine; the gate is the *ratio* (≤ 2×), because
  wall-clock wanders and the ratio is what reproduces.
- **Differentiable.** :func:`inelastic_gradient_report` — the M2/M3 FD protocol
  (float64, per-variable steps, θ_s off the Ed anchors) for all inputs of the
  composed corrected forward, now including ``a_ph`` and ``φ_C``.

The gate assertions live in ``robust/tests/`` (``test_validation.py``,
``test_inelastic_validation.py``); run-and-figure scripts in ``design/py/``,
outside the package. The elastic hash-regression — gate line (4),
``inelastic=None`` bit-identical — is ``test_inelastic_types.py``'s standing
two-tier pin and is *reported through*, not re-derived here.

**Why :func:`rrms` predates the rest.** M2 needed it to score its analytic
backbone against the Gordon baseline, and the *whole point* of the metric is that
one definition is shared: the number in the M2 log, the M4 table, and the
synthesis figures must be the same quantity or none of the comparisons mean
anything. So it landed with M2 rather than being written twice — the rule every
function here follows.
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
    "quantile_bin_labels",
    "median_increment_error",
    "peak_ratio_error",
    "phi_c_linearity",
    "throughput",
    "speed_ratio",
    "gradient_report",
    "inelastic_gradient_report",
    "FD_STEPS",
    "INELASTIC_FD_STEPS",
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

#: Per-variable steps for :func:`inelastic_gradient_report` — the M2/M3 gate's
#: values (``test_inelastic.py``/``test_inelastic_corr.py``), with the elastic
#: :data:`FD_STEPS` for the variables the two protocols share. ``a_ph`` is
#: IOP-like (O(1e-2) m^-1 at 440 nm); ``phi_C`` is O(0.02).
INELASTIC_FD_STEPS = {
    "a": 1e-6,
    "bb_p": 1e-9,
    "B_p": 1e-8,
    "a_ph": 1e-8,
    "phi_C": 1e-6,
    "theta_s": 1e-3,
}

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
    return quantile_bin_labels(B_p, n_bins)


def quantile_bin_labels(
    values: Float[Array, "sample ..."], n_bins: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count quantile bins over a per-sample quantity.

    The generic form of :func:`bp_bin_labels` (which delegates here), factored
    out for the inelastic protocol's a_ph(440)-decile diagnostic — the known
    failure axis of the analytic fluorescence amplitude (design §6): pass
    ``a_ph[:, i440]`` with ``n_bins=10`` and feed the labels to
    :func:`group_rrms` or :func:`peak_ratio_error`.

    Parameters
    ----------
    values : Array
        Per-sample values, shape ``(n_sample,)`` — or ``(n_sample, ...)``, which
        is reduced to a per-sample mean first (the :func:`bp_bin_labels` use).
    n_bins : int, optional
        Number of quantile bins.

    Returns
    -------
    labels : numpy.ndarray
        Bin index per sample, shape ``(n_sample,)``.
    edges : numpy.ndarray
        The ``n_bins + 1`` bin edges.

    Notes
    -----
    The heavy-ties caveat on :func:`bp_bin_labels` applies unchanged: check the
    per-bin counts before reading a table built from data with many repeats.
    """
    values = np.asarray(values)
    if values.ndim > 1:
        values = values.mean(axis=tuple(range(1, values.ndim)))
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    # right=False with the final edge nudged so the maximum lands in the last bin.
    labels = np.clip(np.digitize(values, edges[1:-1], right=False), 0, n_bins - 1)
    return labels, edges


# ------------------------------------------------- per-process delta metrics


def median_increment_error(
    model_factor: Float[Array, "sample wave"],
    truth_factor: Float[Array, "sample wave"],
    labels: np.ndarray,
    band: np.ndarray,
) -> dict[float, float]:
    """The Raman delta metric: median relative increment error, per group.

    ``median((f_model − 1) / (f_truth − 1) − 1)`` over a wavelength band within
    each label group — gate line (2) of the inelastic design §6, scored on the
    *increment* because the factor itself is 1 + small and a ratio of factors
    would hide a large error in the small part (the assessment's lesson: the
    analytic backbone is off by −39 % at 0° in this metric while its factor
    looks fine).

    This is the ``test_inelastic_corr.py`` definition under its permanent name;
    the M3 log, the held-out gate, and the M4 table all report this quantity.

    Parameters
    ----------
    model_factor, truth_factor : Array
        Multiplicative factors (model ``f_R``; truth ``Rrs_X2 / Rrs_X1``),
        shape ``(n_sample, n_wave)``.
    labels : numpy.ndarray
        Per-sample group label, shape ``(n_sample,)`` — usually the zenith.
    band : numpy.ndarray
        Boolean mask over the wavelength axis (e.g. 550–700 nm, or a single
        442-nm-style line).

    Returns
    -------
    dict
        ``{label: median increment error}``, signed, in ascending label order.
        Fractional, not percent — the ≤ 5 % gate is ``abs(value) <= 0.05``.
    """
    inc = (np.asarray(model_factor) - 1.0) / (np.asarray(truth_factor) - 1.0) - 1.0
    labels = np.asarray(labels)
    band = np.asarray(band)
    out = {}
    for value in np.unique(labels):
        rows = labels == value
        out[float(value)] = float(np.median(inc[rows][:, band]))
    return out


def peak_ratio_error(
    model: Float[Array, "sample wave"],
    truth: Float[Array, "sample wave"],
    labels: np.ndarray,
    index: int,
) -> dict[float, float]:
    """The fluorescence delta metric: median model/truth − 1 at one band, per group.

    Gate line (3) of the inelastic design §6, evaluated at the 685 nm peak:
    ``median(model[:, index] / truth[:, index]) − 1`` within each label group.
    The median of the *ratio* rather than the ratio of medians, so every scene
    counts once and a handful of eutrophic outliers cannot carry the statistic.
    Same definition as ``test_inelastic_corr.py``'s held-out gate.

    Parameters
    ----------
    model, truth : Array
        Additive terms (model ``φ_C·K_fl·(1+δ_F)``; truth ``Rrs_X4 − Rrs_X2``),
        shape ``(n_sample, n_wave)``. ``truth`` must be non-zero at ``index`` —
        true at 685 nm in every L23 sample.
    labels : numpy.ndarray
        Per-sample group label, shape ``(n_sample,)``.
    index : int
        Wavelength index of the peak.

    Returns
    -------
    dict
        ``{label: median error}``, signed, fractional, ascending label order.
    """
    model = np.asarray(model)
    truth = np.asarray(truth)
    labels = np.asarray(labels)
    out = {}
    for value in np.unique(labels):
        rows = labels == value
        ratio = model[rows, index] / truth[rows, index]
        out[float(value)] = float(np.median(ratio)) - 1.0
    return out


def phi_c_linearity(
    fluorescence_delta,
    truth_delta: Float[Array, "sample wave"],
    labels: np.ndarray,
    index: int,
    *,
    phi_ref: float,
    scales: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
) -> dict[float, dict[float, float]]:
    """The φ_C-linearity diagnostic against the scaled-truth construction.

    No varied-φ_C truth exists — HydroLight ran X4 at exactly one yield — so
    the design (§6, §8) asks for the next best thing: scale the *truth* term
    linearly, ``truth(s) = s · (Rrs_X4 − Rrs_X2)``, evaluate the model at
    ``φ_C = s · phi_ref``, and report :func:`peak_ratio_error` at each scale.
    A model that is exactly linear in φ_C (the §4.4 construction: δ_F has no
    yield column) reports the *same* error at every scale; any drift across
    scales is nonlinearity leaking in. Reported, never gated — the construction
    has real truth only at ``s = 1``.

    Parameters
    ----------
    fluorescence_delta : callable
        ``phi -> Array (n_sample, n_wave)`` — the model's fluorescence-only
        term in ``Rrs`` space at yield ``phi`` (e.g. the full forward minus the
        Raman-only forward).
    truth_delta : Array
        ``Rrs_X4 − Rrs_X2``, the truth at ``phi_ref``.
    labels : numpy.ndarray
        Per-sample group label (usually the zenith).
    index : int
        Wavelength index of the 685 nm peak.
    phi_ref : float
        The truth's yield (:data:`robust.rt.data.l23.PHI_C_L23`).
    scales : tuple of float, optional
        Multiples of ``phi_ref`` to probe. Includes 1.0 so the table carries
        its own anchor to the gated metric.

    Returns
    -------
    dict
        ``{scale: {label: median 685 nm error}}``, as :func:`peak_ratio_error`.
    """
    truth = np.asarray(truth_delta)
    return {
        float(s): peak_ratio_error(
            np.asarray(fluorescence_delta(s * phi_ref)), s * truth, labels, index
        )
        for s in scales
    }


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


def speed_ratio(
    candidate, reference, *args, repeats: int = 5
) -> tuple[float, float, float]:
    """Candidate runtime over reference runtime, jitted, on identical arguments.

    The inelastic speed gate (design §6 line 6) in function form: the full-batch
    inelastic forward against the elastic hybrid on the same batch, same
    machine, ≤ 2×. The *ratio* is the gated number — :func:`throughput`'s
    milliseconds wander ~20 % between runs on a shared machine, while the ratio
    of two back-to-back timings reproduces.

    Parameters
    ----------
    candidate, reference : callable
        ``model(*args) -> Array``. Close extra configuration (``inelastic=``,
        ``corrections=``) over in a lambda so both take the same positionals —
        that is what makes "identical arguments" literal.
    *args
        Passed to both models.
    repeats : int, optional
        Timed calls per model after each one's compile-paying warm-up.

    Returns
    -------
    ratio : float
        ``candidate_seconds / reference_seconds``.
    candidate_seconds, reference_seconds : float
        The per-call means, for the report table.
    """
    candidate_seconds, _ = throughput(candidate, *args, repeats=repeats)
    reference_seconds, _ = throughput(reference, *args, repeats=repeats)
    return candidate_seconds / reference_seconds, candidate_seconds, reference_seconds


def _grad_vs_fd(scalar, name: str, step: float, dtype) -> float:
    """One variable's ``jax.grad`` vs central-difference disagreement.

    The comparison and its two edge cases, shared by :func:`gradient_report`
    and :func:`inelastic_gradient_report` so the M4 protocol cannot drift from
    the elastic gate's classification rules.
    """
    zero = jnp.zeros((), dtype=dtype)
    analytic = float(jax.grad(partial(scalar, name=name))(zero))
    h = jnp.asarray(step, dtype=dtype)
    numeric = float((scalar(h, name=name) - scalar(-h, name=name)) / (2.0 * h))
    if not np.isfinite(numeric):
        # The step left the physical domain (bb_p driven negative -> NaN). A bad
        # step, not a bad gradient, so it is flagged rather than compared.
        return float("inf")
    if numeric == 0.0:
        # A model that genuinely does not depend on this variable: O25 ignores
        # B_p, so both sides are exactly zero and that is perfect agreement. A
        # ratio would have reported it as infinitely wrong.
        return 0.0 if analytic == 0.0 else float("inf")
    return abs(analytic / numeric - 1.0)


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

    dtype = jnp.asarray(a0).dtype
    return {
        name: _grad_vs_fd(scalar, name, step, dtype) for name, step in steps.items()
    }


def inelastic_gradient_report(
    model,
    iops,
    phase_params,
    geometry,
    wave,
    *,
    phi_C,
    steps: dict[str, float] | None = None,
) -> dict[str, float]:
    """:func:`gradient_report` for the composed inelastic forward — gate line (5).

    The M2/M3 FD protocol as a protocol function: ``jax.grad`` versus central
    finite differences for **all** inputs of the corrected inelastic model —
    the elastic four plus ``a_ph`` (the fluorescence source term) and ``φ_C``
    (the physiology handle the future inversion retrieves). Same classification
    rules as the elastic report (shared helper), same :data:`GRADIENT_TOL`.

    Parameters
    ----------
    model : callable
        ``model(iops, phase_params, geometry, wave, phi_C) -> Array`` — close
        ``inelastic=Inelastic(phi_C=...)``, ``corrections=...`` over in a
        wrapper. Taking ``phi_C`` positionally keeps this module free of any
        import from :mod:`robust.rt.hybrid`.
    iops
        Must carry ``a_ph`` (``ValueError`` otherwise): a report that silently
        skipped the fluorescence source would certify the wrong model.
    phase_params, geometry, wave
        As :func:`gradient_report` — a small batch, geometry carried intact
        apart from ``theta_s``.
    phi_C : float or Array
        The yield to evaluate at (scalar or per-sample leaf).
    steps : dict, optional
        Per-variable steps. Defaults to :data:`INELASTIC_FD_STEPS`; must name
        exactly its variables (same refusal rule as the elastic report — a
        missing key would report "perfect agreement" for a variable never
        perturbed).

    Returns
    -------
    dict
        ``{variable: relative difference}`` for ``a``, ``bb_p``, ``B_p``,
        ``a_ph``, ``phi_C``, ``theta_s``; ``inf`` where the step left the
        physical domain.

    Notes
    -----
    Run under float64 with pre-cast arrays, as :func:`gradient_report`. **Keep
    ``theta_s`` off the packaged-Ed anchors** (0/30/60°): the Ed table is
    piecewise-linear in θ_s, so at an anchor the derivative is one-sided —
    autodiff takes one side, a central difference averages both, and they
    disagree at the 7th digit (M2 finding, record §4.4). The standing gates
    evaluate at 35°; L23 batches arrive *at* the anchors, so shift before
    calling.
    """
    steps = dict(INELASTIC_FD_STEPS if steps is None else steps)
    if set(steps) != set(INELASTIC_FD_STEPS):
        raise ValueError(
            f"inelastic_gradient_report: steps must name exactly "
            f"{sorted(INELASTIC_FD_STEPS)}; got {sorted(steps)}"
        )
    if iops.a_ph is None:
        raise ValueError(
            "inelastic_gradient_report: iops.a_ph is required — the report "
            "certifies the composed inelastic model, whose fluorescence source "
            "is phi_C * a_ph"
        )
    a0, bb_w0, bb_p0, a_ph0 = iops.a, iops.bb_w, iops.bb_p, iops.a_ph
    B_p0, theta0 = phase_params.B_p, geometry.theta_s
    phi0 = jnp.asarray(phi_C)

    def scalar(shift, *, name):
        """Mean model output with one variable shifted by ``shift``."""
        offsets = dict.fromkeys(steps, 0.0)
        offsets[name] = shift
        return jnp.mean(
            model(
                types.IOPs(
                    a=a0 + offsets["a"],
                    bb_w=bb_w0,
                    bb_p=bb_p0 + offsets["bb_p"],
                    a_ph=a_ph0 + offsets["a_ph"],
                ),
                types.PhaseParams(B_p=B_p0 + offsets["B_p"]),
                dataclasses.replace(geometry, theta_s=theta0 + offsets["theta_s"]),
                wave,
                phi0 + offsets["phi_C"],
            )
        )

    dtype = jnp.asarray(a0).dtype
    return {
        name: _grad_vs_fd(scalar, name, step, dtype) for name, step in steps.items()
    }


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
