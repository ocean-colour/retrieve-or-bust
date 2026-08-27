#!/usr/bin/env python
"""Run the design §6 validation protocols and regenerate the metrics tables + figures.

**Elastic** (the default): every model on identical data, scored in ``rrs`` space —
standard Gordon, the ZTT backbone, the O25 form refit on L23, and the hybrid (linear
and MLP emulators, with and without the out-of-domain fallback). Broken out per
wavelength, per solar zenith and per ``B_p`` bin, on both held-out splits, plus
throughput and the gradient gate.

**Inelastic** (``--inelastic``, M4 task 2): the inelastic design-§6 gate table and
diagnostics — held-out total rRMS vs ``Rrs_X4`` (gated over the 400–700 nm band,
prompt 5 Q&A Q1; the full grid reported), the per-process delta metrics, the speed
ratio vs the elastic hybrid, the six-variable gradient report, and the
reported-not-gated diagnostics (a_ph(440) deciles, φ_C-linearity,
``emission_shape='double'``). Every metric goes through
:mod:`robust.rt.validation`'s protocol functions — the same definitions the gate
tests assert — and every number quoted in the implementation record §6 regenerates
here. Artifacts land *alongside* the elastic ones (``*_inelastic.*``,
``inelastic_*.png``), never replacing them.

Usage
-----
    python design/py/run_validation.py                  # design/validation/ (elastic)
    python design/py/run_validation.py --inelastic      # -> the inelastic artifacts
    python design/py/run_validation.py --seeds 23 1 7   # the emulator seed spread
    python design/py/run_validation.py --quick          # fewer steps, for a smoke test

Requires ``$OS_COLOR`` (the L23 netCDFs) and the ``ocean14`` environment. The elastic
run takes a few minutes (each emulator fit is ~60 s and the zenith study fits one per
seed); the inelastic run trains nothing (committed weights) and takes ~2 minutes,
most of it the speed trials.

Every output is written to a temporary file and moved into place only once complete, so
an interrupted run cannot leave a half-written table or figure behind that a later
reader would mistake for results (the rule this repo adopted after PR #11). Note the
limit of that guarantee: each *file* lands atomically, but the five are replaced one
after another, so a run killed midway can leave a new table beside an older figure.
The aggregation check in ``test_validation.py`` is what notices if that happens.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from robust.rt import baselines as B  # noqa: E402
from robust.rt import conventions as C  # noqa: E402
from robust.rt import emulator as E  # noqa: E402
from robust.rt import hybrid as H  # noqa: E402
from robust.rt import inelastic as INE  # noqa: E402
from robust.rt import inelastic_corr as IC  # noqa: E402
from robust.rt import validation as V  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import l23 as L  # noqa: E402
from robust.rt.types import Geometry, Inelastic, IOPs, PhaseParams  # noqa: E402

#: Where the committed artefacts live.
OUT_DIR = Path(__file__).resolve().parents[1] / "validation"

#: Solar zenith to check O25's gradient at — deliberately *between* its table nodes.
#: L23's own angles are the nodes, where a piecewise-linear lookup has a kink and a
#: central difference is meaningless (see ``baselines.o25_coefficients``).
GRADIENT_CHECK_ZENITH = 45.0


#: House style and the CVD-validated categorical set, as in the notebooks. The
#: worst adjacent pair is dE 21.9 (protan); #56B4E9 falls below 3:1 contrast on
#: white, so every series is directly labelled rather than identified by a swatch.
INK, INK_MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"
C_A, C_B, C_C, MUTED = "#0072B2", "#D55E00", "#56B4E9", "#9a9a9a"


def apply_house_style():
    """Select the Agg backend and set the house rcParams, returning pyplot.

    One definition for every figure writer in this script (the M4 review found
    the block pasted twice): a style tweak lands everywhere or nowhere, so the
    committed elastic and inelastic figures cannot silently diverge.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 10.5,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def save_figure_atomically(fig, path: Path) -> None:
    """Render a figure to a temporary file, then move it into place.

    Same rule as the tables: ``savefig`` straight to the destination would leave a
    truncated PNG if the process died mid-write, and a corrupt figure in a committed
    directory is worse than a missing one.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem, suffix=path.suffix
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        fig.savefig(tmp)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def make_figures(out_dir: Path, wave, per_lambda: dict, zenith_rows: list) -> None:
    """Write the two committed figures: the per-λ ladder and the unseen-zenith story.

    Parameters
    ----------
    out_dir : pathlib.Path
    wave : numpy.ndarray
        Wavelength grid, nm.
    per_lambda : dict
        ``{model name: rRMS per wavelength}`` on the held-out scenes.
    zenith_rows : list
        ``(name, median, lo, hi)`` from :func:`zenith_study`; ``lo``/``hi`` are
        ``None`` for the models that have no seed.
    """
    plt = apply_house_style()

    styles = {
        "standard Gordon": (C_C, 1.8, "-"),
        "ZTT backbone": (C_B, 1.8, "-"),
        "O25 form, refit on L23": (C_A, 2.0, "-"),
        "hybrid, linear": (MUTED, 1.6, "--"),
        "hybrid, MLP": (INK, 2.2, "-"),
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for name, (colour, width, dash) in styles.items():
        if name in per_lambda:
            ax.plot(wave, per_lambda[name], color=colour, lw=width, ls=dash, label=name)
    ax.set_yscale("log")
    ax.set_xlim(wave.min(), wave.max())
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("rRMS [%]   (log)")
    ax.legend(loc="lower left", ncols=2, fontsize=9.5)
    fig.suptitle(
        "Held-out scenes, per wavelength: O25 is the benchmark to beat, not Gordon",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure_atomically(fig, out_dir / "rrms_per_wavelength.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    names = [row[0] for row in zenith_rows]
    positions = np.arange(len(names))
    for y, (name, median, lo, hi) in zip(positions, zenith_rows, strict=True):
        seeded = lo is not None
        colour = INK if "MLP" in name else (C_A if "O25" in name else INK_MUTED)
        spread = seeded and hi > lo
        if spread:
            ax.plot(
                [lo, hi], [y, y], color=colour, lw=6, alpha=0.30, solid_capstyle="butt"
            )
        ax.plot([median], [y], "o", color=colour, ms=8)
        # Every row gets a number. A seeded model whose range collapses -- the linear
        # emulator is deterministic -- is worth saying out loud rather than leaving as
        # a bare dot the reader has to measure off the axis.
        label = (
            f"{lo:.2f}–{hi:.2f} over seeds"
            if spread
            else (f"{median:.2f}  (same for every seed)" if seeded else f"{median:.2f}")
        )
        ax.annotate(
            label,
            xy=((hi if spread else median) + 0.35, y),
            va="center",
            fontsize=9,
            color=colour,
        )
    ax.set_yticks(positions, names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("rRMS at the unseen 60° [%]   (dot = median over seeds)")
    ax.set_xlim(0, max(row[3] or row[1] for row in zenith_rows) * 1.45)
    fig.suptitle(
        "Trained on 0°/30°, scored at the unseen 60°: the refit O25 wins, and\n"
        "only the MLP's answer depends on its seed",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure_atomically(fig, out_dir / "unseen_zenith.png")
    plt.close(fig)


def write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem, suffix=path.suffix
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text)
        # mkstemp creates 0600; a committed artefact must be readable by everyone
        # who can read the repo, so restore the permissions a plain open() would
        # have given it under the process umask.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv_atomically(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write a CSV through :mod:`csv`, so the quoting is not our problem.

    The first version of this joined fields with commas by hand, and the model names
    contain commas — "O25 form, refit on L23", "hybrid, MLP". The result parsed, and
    parsed *wrongly*: `metrics.csv` had four fields where its header promised three,
    and the ladder's header expanded seven model names into ten columns, so any
    consumer would have silently mis-labelled every column. Nothing crashed, which is
    what made it worth fixing properly rather than renaming the models.

    Parameters
    ----------
    path : pathlib.Path
    header : list of str
    rows : list of list of str
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    write_atomically(path, buffer.getvalue())


def build_models(batch, splits, *, steps: int, seed: int):
    """Every model's prediction on the full batch, in ``rrs`` space.

    Predictions rather than callables, so "identical data" is literal: each model is
    evaluated once on the whole batch and the splits are slices of that.
    """
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    rrs_ztt = Z.rrs_ZTT(*args)

    config = E.EmulatorConfig(steps=steps, seed=seed)
    linear = E.EmulatorConfig(hidden=(), steps=steps, seed=seed)
    mlp_em, _ = E.fit_l23(batch, splits, config=config, rrs_ztt=rrs_ztt)
    lin_em, _ = E.fit_l23(batch, splits, config=linear, rrs_ztt=rrs_ztt)

    models = {
        "standard Gordon": B.rrs_gordon(*args),
        "ZTT backbone": rrs_ztt,
        "O25 form, refit on L23": B.rrs_o25(*args),
        "hybrid, linear": rrs_ztt * (1.0 + lin_em.relative_delta(*args)),
        "hybrid, MLP": rrs_ztt * (1.0 + mlp_em.relative_delta(*args)),
    }
    return models, mlp_em


def zenith_study(batch, splits, truth, rrs_ztt, seeds, *, steps: int):
    """Train on 0/30 deg only and score the unseen 60 deg — the hard split.

    Every trained model gets the **seed spread**, not a single fit: M3 measured a
    2.6x range here, so one number would be a claim about a seed rather than about
    the method.
    """
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    train, test = splits.zenith_train, splits.zenith_test
    rows = []

    for name, pred in (
        ("standard Gordon", B.rrs_gordon(*args)),
        ("ZTT backbone", rrs_ztt),
    ):
        rows.append((name, float(V.rrms(truth[test], pred[test])), None, None))

    # O25 refit on 0/30 only: its lookup then has no 60 deg row and clamps to 30 deg.
    o25_rows = B.fit_o25(
        batch.iops, batch.Rrs, batch.geometry, train=train, zeniths=(0.0, 30.0)
    )
    o25 = B.rrs_o25(*args, coeffs=o25_rows)
    rows.append(
        ("O25 form, refit on 0/30", float(V.rrms(truth[test], o25[test])), None, None)
    )

    for label, hidden in (("hybrid, linear", ()), ("hybrid, MLP", (16, 16))):
        scores = []
        for seed in seeds:
            config = E.EmulatorConfig(hidden=hidden, steps=steps, seed=seed)
            emulator, _ = E.fit(
                *args, truth, train=train, config=config, rrs_ztt=rrs_ztt
            )
            hybrid = rrs_ztt * (1.0 + emulator.relative_delta(*args))
            scores.append(float(V.rrms(truth[test], hybrid[test])))
        rows.append((label, float(np.median(scores)), min(scores), max(scores)))
    return rows


def make_inelastic_figures(
    out_dir: Path, wave, ladders: dict, delta_rows: list
) -> None:
    """Write the two committed inelastic figures.

    Parameters
    ----------
    out_dir : pathlib.Path
    wave : numpy.ndarray
        Wavelength grid, nm (the full 350–750 grid — the ladder shows the
        sub-400 cliff *because* it is excluded from the gate).
    ladders : dict
        ``{model name: rRMS per wavelength}`` vs ``Rrs_X4``, held-out scenes.
    delta_rows : list
        ``(label, analytic_pct, corrected_pct)`` per (metric, zenith) — the
        before/after story of the correction heads.
    """
    plt = apply_house_style()
    lo, hi = V.INELASTIC_GATE_BAND

    # --- the vs-X4 ladder: elastic-only -> analytic -> corrected -------------
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    styles = {
        "elastic-only": (C_B, 1.8, "-"),
        "analytic inelastic": (C_A, 1.8, "-"),
        "corrected inelastic": (INK, 2.2, "-"),
    }
    for name, (colour, width, dash) in styles.items():
        ax.plot(wave, ladders[name], color=colour, lw=width, ls=dash, label=name)
    ax.set_yscale("log")
    ax.set_xlim(wave.min(), wave.max())
    # The excluded regions, shaded rather than cropped: the sub-400 cliff is
    # the honest reason the gate band exists (Q&A Q1), so it stays visible.
    ax.axvspan(wave.min(), lo, color=GRID, alpha=0.45, zorder=0)
    ax.axvspan(hi, wave.max(), color=GRID, alpha=0.45, zorder=0)
    bar = V.INELASTIC_GATE_TOTAL_RRMS
    ax.plot([lo, hi], [bar, bar], color=INK_MUTED, lw=1.2, ls=":")
    ax.annotate(
        f"{bar:g} % gate, {lo:.0f}–{hi:.0f} nm",
        xy=(hi - 8, bar * 1.1),
        ha="right",
        fontsize=9,
        color=INK_MUTED,
    )
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("rRMS vs Rrs_X4 [%]   (log)")
    # Lower left sits on the shaded sub-400 band, clear of all three curves.
    ax.legend(loc="lower left", fontsize=9.5)
    fig.suptitle(
        "Held-out scenes vs the all-processes truth: the corrected model sits\n"
        "under the 0.5 % gate across its 400–700 nm domain (shaded = ungated)",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure_atomically(fig, out_dir / "inelastic_rrms_per_wavelength.png")
    plt.close(fig)

    # --- the per-process before/after ----------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    positions = np.arange(len(delta_rows))
    delta_bar = 100.0 * V.INELASTIC_GATE_DELTA
    ax.axvspan(-delta_bar, delta_bar, color=GRID, alpha=0.45, zorder=0)
    for y, (_label, analytic, corrected) in zip(positions, delta_rows, strict=True):
        ax.plot([analytic, corrected], [y, y], color=INK_MUTED, lw=1.0, zorder=2)
        ax.plot([analytic], [y], "o", color=MUTED, ms=7, zorder=3)
        ax.plot([corrected], [y], "o", color=INK, ms=7, zorder=4)
        ax.annotate(
            f"{analytic:+.1f} → {corrected:+.2f}",
            xy=(max(analytic, 6.5) + 1.5, y),
            va="center",
            fontsize=9,
            color=INK_MUTED,
        )
    ax.set_yticks(positions, [row[0] for row in delta_rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(
        "median error [%]   (grey dot = analytic backbone, black = corrected; "
        f"band = ±{delta_bar:g} % gate)"
    )
    fig.suptitle(
        "The per-process gates: what the trained heads closed (held-out medians)",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure_atomically(fig, out_dir / "inelastic_deltas.png")
    plt.close(fig)


def inelastic_main(args_cli) -> int:
    """The inelastic §6 protocol: gate table, diagnostics, artifacts (M4 task 2).

    No training happens here — the committed weights *are* the model — so the
    run is deterministic apart from the speed trials, and every number in the
    implementation record §6 regenerates from this function.
    """
    try:
        batch = L.load_inelastic_batch()
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(
            "the L23 inelastic reference data is not available -- set $OS_COLOR "
            f"to the directory holding the Hydrolight netCDFs. ({exc})"
        ) from exc
    if not (IC.DEFAULT_RAMAN_WEIGHTS.exists() and IC.DEFAULT_FL_WEIGHTS.exists()):
        raise SystemExit(
            "committed correction weights missing -- run "
            "design/py/train_inelastic_corr.py first"
        )
    splits = L.make_splits(batch)
    heads = IC.load_default()
    # Eager, before anything jits: a memoised loader first touched inside a
    # jit trace caches tracers and every later call explodes (the task-1 lesson).
    em = E.load_default()

    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    wave = np.asarray(batch.wave)
    zen = batch.zenith
    held = splits.scene_test
    lo, hi = V.INELASTIC_GATE_BAND
    band = (wave >= lo) & (wave <= hi)
    truth = np.asarray(C.Rrs_to_rrs(batch.Rrs_x4))

    # ---------------------------------------------- total rRMS vs X4 (line 1)
    rrs_models = {
        "elastic-only": H.rrs_forward(*args, "hybrid", emulator=em, check_domain=False),
        "analytic inelastic": H.rrs_forward(
            *args,
            "hybrid",
            inelastic=Inelastic(),
            corrections=False,
            emulator=em,
            check_domain=False,
        ),
        "corrected inelastic": H.rrs_forward(
            *args,
            "hybrid",
            inelastic=Inelastic(),
            corrections=heads,
            emulator=em,
            check_domain=False,
        ),
    }
    rrs_models = {name: np.asarray(pred) for name, pred in rrs_models.items()}

    def zenith_rrms(pred, columns):
        return V.group_rrms(
            jnp.asarray(truth[held][:, columns]),
            jnp.asarray(pred[held][:, columns]),
            zen[held],
        )

    gate_rrms = {name: zenith_rrms(pred, band) for name, pred in rrs_models.items()}
    full_rrms = {
        name: zenith_rrms(pred, slice(None)) for name, pred in rrs_models.items()
    }
    ladders = {
        name: np.asarray(V.rrms_per_wavelength(truth[held], pred[held]))
        for name, pred in rrs_models.items()
    }

    # ------------------------------------------- per-process deltas (lines 2-3)
    f_phys = np.asarray(INE.raman_factor(batch.iops, batch.geometry, batch.wave))
    delta_r = np.asarray(heads.raman.delta(batch.iops, batch.geometry, batch.wave))
    f_corr = np.asarray(IC.corrected_raman_factor(delta_r, f_phys))
    truth_r = np.asarray(batch.truth_raman_factor)
    k_fl = np.asarray(INE.fluorescence_kernel(batch.iops, batch.geometry, batch.wave))
    delta_f = np.asarray(heads.fl.delta(batch.iops, batch.geometry, batch.wave))
    fl_analytic = L.PHI_C_L23 * k_fl
    # The shared composition helper — the expression forward() runs (review).
    fl_corrected = L.PHI_C_L23 * np.asarray(IC.corrected_fluorescence(delta_f, k_fl))
    truth_fl = np.asarray(batch.truth_fluorescence)
    i685 = int(np.abs(wave - 685.0).argmin())

    raman_bands = {
        "Raman 550-700 nm": (wave >= 550.0) & (wave <= 700.0),
        "Raman 490 nm": np.abs(wave - 490.0) < 1e-6,
    }
    deltas = {}  # {metric: {zenith: (analytic, corrected)}}, fractional
    for label, columns in raman_bands.items():
        analytic = V.median_increment_error(
            f_phys[held], truth_r[held], zen[held], columns
        )
        corrected = V.median_increment_error(
            f_corr[held], truth_r[held], zen[held], columns
        )
        deltas[label] = {z: (analytic[z], corrected[z]) for z in analytic}
    fl_a = V.peak_ratio_error(fl_analytic[held], truth_fl[held], zen[held], i685)
    fl_c = V.peak_ratio_error(fl_corrected[held], truth_fl[held], zen[held], i685)
    deltas["fluorescence 685 nm"] = {z: (fl_a[z], fl_c[z]) for z in fl_a}

    # ------------------------------------------------- bit-identity (line 4)
    # The baseline is free: forward() is literally rrs_to_Rrs(rrs_forward()),
    # so the elastic-only rrs already computed converts to the identical bytes
    # (M4 review finding — this was a redundant full-batch forward).
    omitted = np.asarray(C.rrs_to_Rrs(jnp.asarray(rrs_models["elastic-only"])))
    bit_identical = np.array_equal(
        omitted,
        np.asarray(H.forward(*args, inelastic=None, emulator=em, check_domain=False)),
    ) and np.array_equal(
        omitted,
        np.asarray(
            H.forward(
                *args,
                inelastic=Inelastic(raman=False, fluorescence=False),
                emulator=em,
                check_domain=False,
            )
        ),
    )

    # -------------------------------------------------------- speed (line 6)
    # jit once here: throughput reuses an already-wrapped callable, so the
    # trials repay the timing loop and not one XLA compile per trial; the
    # measurement order alternates so an ordering bias cannot repeat into the
    # median (both M4 review findings).
    @jax.jit
    def corrected_fn(i, p, g, w):
        return H.forward(
            i,
            p,
            g,
            w,
            "hybrid",
            inelastic=Inelastic(),
            corrections=heads,
            emulator=em,
            check_domain=False,
        )

    @jax.jit
    def elastic_fn(i, p, g, w):
        return H.forward(i, p, g, w, "hybrid", emulator=em, check_domain=False)

    trials = [
        V.speed_ratio(corrected_fn, elastic_fn, *args, repeats=10, reverse=bool(t % 2))
        for t in range(args_cli.speed_trials)
    ]
    speed_median = float(np.median([t[0] for t in trials]))
    inelastic_ms = 1e3 * float(np.median([t[1] for t in trials]))
    elastic_ms = 1e3 * float(np.median([t[2] for t in trials]))

    # ---------------------------------------------------- gradients (line 5)
    # try/finally so an exception cannot leave float64 on for the rest of the
    # process (M4 review finding) — every later float32 comparison would then
    # run on different bytes.
    jax.config.update("jax_enable_x64", True)
    try:
        rows = np.where(zen == 30)[0][:3]
        f64 = lambda x: jnp.asarray(  # noqa: E731
            np.asarray(x)[rows], dtype=jnp.float64
        )
        grad_report = V.inelastic_gradient_report(
            lambda i, p, g, w, phi: H.forward(
                i,
                p,
                g,
                w,
                "hybrid",
                inelastic=Inelastic(phi_C=phi),
                corrections=heads,
                emulator=em,
                check_domain=False,
            ),
            IOPs(
                a=f64(batch.iops.a),
                bb_w=f64(batch.iops.bb_w),
                bb_p=f64(batch.iops.bb_p),
                a_ph=f64(batch.iops.a_ph),
            ),
            PhaseParams(B_p=f64(batch.phase_params.B_p)),
            # 35 deg: off the packaged-Ed anchors (0/30/60), where the
            # derivative is one-sided (record §4.4) — the standing gates' choice.
            Geometry.nadir(jnp.full((len(rows),), 35.0, dtype=jnp.float64)),
            jnp.asarray(wave, dtype=jnp.float64),
            phi_C=jnp.asarray(L.PHI_C_L23, jnp.float64),
        )
    finally:
        jax.config.update("jax_enable_x64", False)

    # ----------------------------------------------- diagnostics (not gated)
    i440 = int(np.abs(wave - 440.0).argmin())
    decile_labels, decile_edges = V.quantile_bin_labels(
        np.asarray(batch.iops.a_ph)[held][:, i440], n_bins=10
    )
    deciles = V.peak_ratio_error(
        fl_corrected[held], truth_fl[held], decile_labels, i685
    )

    raman_only = np.asarray(
        H.forward(
            *args,
            "hybrid",
            inelastic=Inelastic(fluorescence=False),
            corrections=heads,
            emulator=em,
            check_domain=False,
        )
    )

    def fluorescence_delta(phi):
        full = np.asarray(
            H.forward(
                *args,
                "hybrid",
                inelastic=Inelastic(phi_C=jnp.asarray(phi)),
                corrections=heads,
                emulator=em,
                check_domain=False,
            )
        )
        return (full - raman_only)[held]

    linearity = V.phi_c_linearity(
        fluorescence_delta, truth_fl[held], zen[held], i685, phi_ref=L.PHI_C_L23
    )

    double = np.asarray(
        H.forward(
            *args,
            "hybrid",
            inelastic=Inelastic(emission_shape="double"),
            corrections=heads,
            emulator=em,
            check_domain=False,
        )
    )
    # Same reuse rule as the bit-identity baseline: the corrected rrs is in hand.
    single = np.asarray(C.rrs_to_Rrs(jnp.asarray(rrs_models["corrected inelastic"])))
    i730 = int(np.abs(wave - 730.0).argmin())
    double_685 = 100.0 * float(np.median(double[:, i685] / single[:, i685] - 1.0))
    double_730 = 100.0 * float(np.median(double[:, i730] / single[:, i730] - 1.0))
    double_vs_truth = V.peak_ratio_error(
        (double - raman_only)[held], truth_fl[held], zen[held], i685
    )

    # ------------------------------------------------------------ the gate
    worst_total = max(gate_rrms["corrected inelastic"].values())
    worst_raman = max(
        abs(corrected)
        for label in raman_bands
        for _, corrected in deltas[label].values()
    )
    worst_fl = max(
        abs(corrected) for _, corrected in deltas["fluorescence 685 nm"].values()
    )
    worst_grad = max(grad_report.values())
    gate_rows = [
        [
            "1 total rRMS vs X4",
            f"{worst_total:.3f} % (worst zenith)",
            f"<= {V.INELASTIC_GATE_TOTAL_RRMS:g} %",
            "PASS" if worst_total <= V.INELASTIC_GATE_TOTAL_RRMS else "FAIL",
        ],
        [
            "2 Raman delta incl. 0 deg",
            f"{100 * worst_raman:.2f} % (worst)",
            f"<= {100 * V.INELASTIC_GATE_DELTA:g} %",
            "PASS" if worst_raman <= V.INELASTIC_GATE_DELTA else "FAIL",
        ],
        [
            "3 fluorescence delta",
            f"{100 * worst_fl:.2f} % (worst)",
            f"<= {100 * V.INELASTIC_GATE_DELTA:g} %",
            "PASS" if worst_fl <= V.INELASTIC_GATE_DELTA else "FAIL",
        ],
        [
            "4 inelastic=None bit-identical",
            str(bit_identical),
            "True",
            "PASS" if bit_identical else "FAIL",
        ],
        [
            "5 gradients incl. phi_C",
            f"{worst_grad:.1e} (worst of 6)",
            "<= 1e-06",
            "PASS" if worst_grad <= V.GRADIENT_TOL else "FAIL",
        ],
        [
            "6 speed vs elastic hybrid",
            f"{speed_median:.2f}x (median)",
            f"<= {V.INELASTIC_GATE_SPEED:g}x",
            "PASS" if speed_median <= V.INELASTIC_GATE_SPEED else "FAIL",
        ],
    ]
    all_pass = all(row[-1] == "PASS" for row in gate_rows)

    # ------------------------------------------------------------- artifacts
    zeniths = (0.0, 30.0, 60.0)
    lines = [
        "# M4 validation — the inelastic design §6 protocol",
        "",
        f"*Generated by `design/py/run_validation.py --inelastic`: committed "
        f"weights only (no training), {args_cli.speed_trials} speed trials x 10 "
        "calls. Regenerate after any change to the model, the metric, the "
        "weights or the splits.*",
        "",
        f"L23 inelastic (X=1/2/4), {batch.n_sample} samples x {batch.n_wave} "
        f"lambda; {int(held.sum())} held-out samples (elastic scene split). "
        "rRMS in `rrs` space vs `Rrs_X4`, all processes on, phi_C = 0.02.",
        "",
        f"## The acceptance gate — {'**PASSED**' if all_pass else '**FAILED**'}",
        "",
        f"The total-rRMS line is scored over **{lo:.0f}-{hi:.0f} nm** (prompt 5 "
        "Q&A Q1: the model's stated domain — below 400 nm the Raman excitation "
        "clamps and the heads never trained). The full 350-750 nm grid is "
        "reported below, never gated.",
        "",
        V.markdown_table(gate_rows, ["line", "measured", "bar", "verdict"]),
        "",
        "## Total held-out rRMS vs Rrs_X4, per zenith",
        "",
        V.markdown_table(
            [
                [name, *(scores[z] for z in zeniths)]
                for name, scores in gate_rrms.items()
            ],
            [f"model ({lo:.0f}-{hi:.0f} nm)", "0 deg", "30 deg", "60 deg"],
        ),
        "",
        V.markdown_table(
            [
                [name, *(scores[z] for z in zeniths)]
                for name, scores in full_rrms.items()
            ],
            ["model (full 350-750 nm, reported)", "0 deg", "30 deg", "60 deg"],
        ),
        "",
        "## Per-process deltas (median error, held-out; analytic -> corrected)",
        "",
        V.markdown_table(
            [
                [label]
                + [
                    f"{100 * a:+.2f} -> {100 * c:+.2f}"
                    for a, c in (deltas[label][z] for z in zeniths)
                ]
                for label in deltas
            ],
            ["metric [%]", "0 deg", "30 deg", "60 deg"],
        ),
        "",
        "## Throughput (full batch, jitted, CPU; medians over trials)",
        "",
        V.markdown_table(
            [
                ["elastic hybrid", elastic_ms, 1.0],
                ["corrected inelastic", inelastic_ms, speed_median],
            ],
            ["model", "ms / call", "x elastic"],
        ),
        "",
        f"Trial ratios: {', '.join(f'{t[0]:.2f}' for t in trials)} — wall-clock "
        "on a shared machine wanders ~5 %, which is why the gate asserts the "
        "median.",
        "",
        "## Gradient gate (jax.grad vs central differences, float64, theta_s = 35 deg)",
        "",
        V.markdown_table(
            [[name, f"{grad_report[name]:.1e}"] for name in V.INELASTIC_FD_STEPS],
            ["variable", "relative difference"],
        ),
        "",
        "## Diagnostics (reported, not gated)",
        "",
        "**a_ph(440) deciles** (held-out fluorescence 685 nm error; edges "
        f"{decile_edges[0]:.4f}-{decile_edges[-1]:.4f} m^-1): "
        + ", ".join(f"{100 * deciles[k]:+.2f}" for k in sorted(deciles))
        + f" % — max |err| {100 * max(abs(v) for v in deciles.values()):.2f} %, "
        "flat across trophic state (the analytic term drifts -11 to +11 %).",
        "",
        "**phi_C-linearity** (scaled-truth construction; 685 nm error per "
        "zenith at each multiple of phi_C = 0.02):",
        "",
        V.markdown_table(
            [
                [f"{scale:g}x", *(f"{100 * linearity[scale][z]:+.3f}" for z in zeniths)]
                for scale in sorted(linearity)
            ],
            ["scale", "0 deg [%]", "30 deg [%]", "60 deg [%]"],
        ),
        "",
        "Identical rows = linear by construction (design §4.4); truth exists "
        "only at 1x.",
        "",
        f"**emission_shape='double'**: vs 'single', median Rrs {double_685:+.1f} % "
        f"at 685 nm and {double_730:+.1f} % at 730 nm; scored against the "
        "single-shape truth the 685 nm term sits at "
        + ", ".join(f"{100 * double_vs_truth[z]:+.1f} %" for z in zeniths)
        + " (0/30/60 deg) — consistent with moving 25 % of the emission into a "
        "shoulder L23 cannot see. Unvalidatable, off everywhere in v1.",
        "",
        "## Caveats (measured, record §5-6)",
        "",
        "- **Geometry**: the heads interpolate in cos(theta_s) over three "
        "zeniths; a delta_R trained without 60 deg errs by -74 % there "
        "(notebook 4 §3). Unseen geometries need coverage or a domain guard.",
        "- **Domain**: lambda >= 400 nm (excitation clamp below), nadir view, "
        "L23-like water; phi_C-linearity has truth only at 0.02.",
        "- **theta_s derivative** is one-sided at the packaged-Ed anchors "
        "(0/30/60 deg) — differentiate off the anchors (record §4.4).",
        "",
    ]

    write_atomically(args_cli.out / "metrics_inelastic.md", "\n".join(lines) + "\n")

    csv_rows = []
    for name in rrs_models:
        for z in zeniths:
            csv_rows.append(
                [
                    "total_rrms",
                    f"{name} {lo:.0f}-{hi:.0f}nm",
                    f"{z:.0f}",
                    f"{gate_rrms[name][z]:.4f}",
                ]
            )
            csv_rows.append(
                [
                    "total_rrms",
                    f"{name} full-grid",
                    f"{z:.0f}",
                    f"{full_rrms[name][z]:.4f}",
                ]
            )
    for label in deltas:
        for z in zeniths:
            analytic, corrected = deltas[label][z]
            csv_rows.append(
                ["delta_analytic_pct", label, f"{z:.0f}", f"{100 * analytic:.4f}"]
            )
            csv_rows.append(
                ["delta_corrected_pct", label, f"{z:.0f}", f"{100 * corrected:.4f}"]
            )
    csv_rows.append(["speed", "median_ratio", "", f"{speed_median:.4f}"])
    csv_rows.append(["speed", "inelastic_ms", "", f"{inelastic_ms:.2f}"])
    csv_rows.append(["speed", "elastic_ms", "", f"{elastic_ms:.2f}"])
    for name, value in grad_report.items():
        csv_rows.append(["gradient_rel_diff", name, "", f"{value:.3e}"])
    csv_rows.append(["bit_identical", "inelastic_none", "", str(bit_identical)])
    write_csv_atomically(
        args_cli.out / "metrics_inelastic.csv",
        ["section", "metric", "zenith", "value"],
        csv_rows,
    )

    write_csv_atomically(
        args_cli.out / "rrms_per_wavelength_inelastic.csv",
        ["wavelength_nm", *rrs_models],
        [
            [f"{wave[i]:.1f}", *(f"{ladders[n][i]:.4f}" for n in rrs_models)]
            for i in range(len(wave))
        ],
    )

    delta_fig_rows = [
        (
            f"{label}, {z:.0f} deg",
            100.0 * deltas[label][z][0],
            100.0 * deltas[label][z][1],
        )
        for label in deltas
        for z in zeniths
    ]
    make_inelastic_figures(args_cli.out, wave, ladders, delta_fig_rows)

    print("\n".join(lines))
    print(
        f"wrote {args_cli.out}/metrics_inelastic.md, metrics_inelastic.csv, "
        "rrms_per_wavelength_inelastic.csv + 2 figures"
    )
    return 0 if all_pass else 1


def main() -> int:
    """Score every model on every split and write the artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 1, 7, 101, 2024])
    parser.add_argument("--quick", action="store_true", help="short fits, for a check")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--inelastic",
        action="store_true",
        help="regenerate the inelastic §6 artifacts instead of the elastic ones",
    )
    parser.add_argument(
        "--speed-trials",
        type=int,
        default=5,
        help="speed_ratio trials for the inelastic gate line 6 (median reported)",
    )
    args_cli = parser.parse_args()
    if args_cli.inelastic:
        if args_cli.quick:
            # The inelastic run trains nothing, so there is nothing to shorten;
            # a --quick that silently did the full run would be a lie.
            raise SystemExit("--quick applies to the elastic emulator fits only")
        return inelastic_main(args_cli)
    steps = 300 if args_cli.quick else E.EmulatorConfig().steps

    if args_cli.quick and args_cli.out == OUT_DIR:
        # --quick fits for 300 steps instead of 3000. Letting that overwrite the
        # committed artefacts would leave numbers indistinguishable from real ones
        # except by being wrong, so it has to be deliberate.
        raise SystemExit(
            "refusing --quick with the default --out: it would overwrite the "
            f"committed artefacts in {OUT_DIR} with short-fit numbers. Pass an "
            "explicit --out for a smoke test."
        )
    try:
        batch = L.load_batch()
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(
            "the L23 reference data is not available -- set $OS_COLOR to the "
            f"directory holding the Hydrolight netCDFs. ({exc})"
        ) from exc
    splits = L.make_splits(batch)
    truth = C.Rrs_to_rrs(batch.Rrs)
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    rrs_ztt = Z.rrs_ZTT(*args)
    zen = batch.zenith

    models, mlp_em = build_models(batch, splits, steps=steps, seed=args_cli.seeds[0])
    # The fallback policy only differs outside the supported envelope, so on L23 --
    # every angle of which is inside it -- it is the same model. Reported to show that.
    models["hybrid, MLP + fallback"] = H.rrs_forward(
        *args, "hybrid", emulator=mlp_em, on_out_of_domain="ztt", check_domain=False
    )

    masks = {
        "train": splits.scene_train,
        "held-out scenes": splits.scene_test,
        "held-out @60": splits.scene_test & (zen == 60),
    }
    table = V.score_models(models, truth, masks)

    lines = ["# M4 validation — the design §6 protocol", ""]
    lines += [
        f"*Generated by `design/py/run_validation.py`: {steps} Adam steps per "
        f"emulator fit, seeds {tuple(args_cli.seeds)}"
        + (", **--quick**" if args_cli.quick else "")
        + ". Regenerate after any change to the models, the metric or the splits.*",
        "",
        f"L23 elastic, {batch.n_sample} samples x {batch.n_wave} lambda; "
        f"{int(splits.scene_train.sum())} train / {int(splits.scene_test.sum())} "
        "held-out scenes. rRMS %, in `rrs` space (design §6).",
        "",
        "## Scene split",
        "",
        V.markdown_table(
            [[name, *(scores[s] for s in masks)] for name, scores in table.items()],
            ["model", *masks],
        ),
        "",
        "## Per solar zenith (held-out scenes)",
        "",
    ]
    held = splits.scene_test
    lines.append(
        V.markdown_table(
            [
                [name, *V.group_rrms(truth[held], pred[held], zen[held]).values()]
                for name, pred in models.items()
            ],
            ["model", "0 deg", "30 deg", "60 deg"],
        )
    )

    bp_labels, bp_edges = V.bp_bin_labels(batch.phase_params.B_p)
    lines += [
        "",
        "## Per `B_p` bin (held-out scenes, equal-count bins)",
        "",
        "Bin edges: "
        + ", ".join(f"{e:.5f}" for e in bp_edges)
        + f" — a factor {bp_edges[-1] / bp_edges[0]:.2f} in total, against the design's"
        " ~7x nominal band, so this cut cannot speak to phase-function"
        " generalisation.",
        "",
        V.markdown_table(
            [
                [name, *V.group_rrms(truth[held], pred[held], bp_labels[held]).values()]
                for name, pred in models.items()
            ],
            ["model", "bin 0", "bin 1", "bin 2", "bin 3"],
        ),
        "",
        "## Unseen 60 deg (trained on 0/30 only)",
        "",
        f"Emulators over seeds {tuple(args_cli.seeds)}; analytic models have no seed.",
        "",
    ]
    zenith_rows = zenith_study(
        batch, splits, truth, rrs_ztt, args_cli.seeds, steps=steps
    )
    lines.append(
        V.markdown_table(
            [
                [name, median, "" if lo is None else f"{lo:.2f}–{hi:.2f}"]
                for name, median, lo, hi in zenith_rows
            ],
            ["model", "rRMS (median)", "range over seeds"],
        )
    )

    # Speed, reported as a ratio: wall-clock wanders ~20% between runs. The ratio is
    # taken against the ZTT row *from this same loop*, not a separate timing of it --
    # timing the reference twice made the table say ZTT was 0.72x itself.
    timings = {}
    for name, fn in (
        ("standard Gordon", lambda i, p, g, w: B.rrs_gordon(i, p, g, w)),
        ("ZTT backbone", lambda i, p, g, w: Z.rrs_ZTT(i, p, g, w)),
        ("O25 form", lambda i, p, g, w: B.rrs_o25(i, p, g, w)),
        (
            "hybrid, MLP",
            lambda i, p, g, w: H.rrs_forward(
                i, p, g, w, "hybrid", emulator=mlp_em, check_domain=False
            ),
        ),
    ):
        timings[name] = V.throughput(fn, *args)
    reference = timings["ZTT backbone"][0]
    speed_rows = [
        [name, seconds * 1e3, rate / 1e6, seconds / reference]
        for name, (seconds, rate) in timings.items()
    ]
    lines += [
        "",
        "## Throughput (jitted, CPU)",
        "",
        V.markdown_table(speed_rows, ["model", "ms / call", "M sample·λ/s", "x ZTT"]),
        "",
        "## Gradient gate",
        "",
        f"`jax.grad` vs central differences, float64, per-variable steps "
        f"{V.FD_STEPS}, tolerance {V.GRADIENT_TOL:g}. Evaluated at "
        f"{GRADIENT_CHECK_ZENITH:.0f} deg — *between* O25's table nodes, where a "
        "piecewise-linear lookup is differentiable (L23's own angles are the nodes).",
        "",
    ]

    # try/finally for the same reason as the inelastic section: an exception
    # here must not leave float64 on for the rest of the process.
    jax.config.update("jax_enable_x64", True)
    try:
        rows = np.where(zen == 30)[0][:3]
        f64 = lambda x: jnp.asarray(  # noqa: E731
            np.asarray(x)[rows], dtype=jnp.float64
        )
        iops64 = IOPs(
            a=f64(batch.iops.a), bb_w=f64(batch.iops.bb_w), bb_p=f64(batch.iops.bb_p)
        )
        phase64 = PhaseParams(B_p=f64(batch.phase_params.B_p))
        geom64 = Geometry.nadir(
            jnp.full((len(rows),), GRADIENT_CHECK_ZENITH, dtype=jnp.float64)
        )
        wave64 = jnp.asarray(np.asarray(batch.wave), dtype=jnp.float64)
        grad_rows = []
        for name, fn in (
            ("ZTT backbone", lambda i, p, g, w: Z.rrs_ZTT(i, p, g, w)),
            ("O25 form", lambda i, p, g, w: B.rrs_o25(i, p, g, w)),
            (
                "hybrid, MLP",
                lambda i, p, g, w: H.rrs_forward(
                    i, p, g, w, "hybrid", emulator=mlp_em, check_domain=False
                ),
            ),
        ):
            report = V.gradient_report(fn, iops64, phase64, geom64, wave64)
            grad_rows.append([name, *(f"{report[k]:.1e}" for k in V.FD_STEPS)])
    finally:
        jax.config.update("jax_enable_x64", False)
    lines.append(V.markdown_table(grad_rows, ["model", *V.FD_STEPS]))
    lines += [
        "",
        "`B_p` reads 0.0e+00 for O25 because it genuinely ignores the phase function:"
        " both derivatives are exactly zero, which is agreement.",
        "",
    ]

    write_atomically(args_cli.out / "metrics.md", "\n".join(lines) + "\n")

    write_csv_atomically(
        args_cli.out / "metrics.csv",
        ["model", "split", "rrms_percent"],
        [
            [name, split, f"{value:.4f}"]
            for name, scores in table.items()
            for split, value in scores.items()
        ],
    )

    per_lambda = {
        name: np.asarray(V.rrms_per_wavelength(truth[held], pred[held]))
        for name, pred in models.items()
    }
    wave = np.asarray(batch.wave)
    write_csv_atomically(
        args_cli.out / "rrms_per_wavelength.csv",
        ["wavelength_nm", *models],
        [
            [f"{wave[i]:.1f}", *(f"{per_lambda[n][i]:.4f}" for n in models)]
            for i in range(len(wave))
        ],
    )

    make_figures(args_cli.out, wave, per_lambda, zenith_rows)

    print("\n".join(lines))
    print(f"wrote {args_cli.out}/metrics.md, metrics.csv, rrms_per_wavelength.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
