#!/usr/bin/env python
"""Run the design §6 validation protocol and regenerate the metrics table + figures.

Every model on identical data, scored in ``rrs`` space: standard Gordon, the ZTT
backbone, the O25 form refit on L23, and the hybrid (linear and MLP emulators, with
and without the out-of-domain fallback). Broken out per wavelength, per solar zenith
and per ``B_p`` bin, on both held-out splits, plus throughput and the gradient gate.

Usage
-----
    python design/py/run_validation.py                  # -> design/validation/
    python design/py/run_validation.py --seeds 23 1 7   # the emulator seed spread
    python design/py/run_validation.py --quick          # fewer steps, for a smoke test

Requires ``$OS_COLOR`` (the L23 netCDFs) and the ``ocean14`` environment. Takes a few
minutes: each emulator fit is ~60 s and the zenith study fits one per seed.

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
from robust.rt import validation as V  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import l23 as L  # noqa: E402
from robust.rt.types import Geometry, IOPs, PhaseParams  # noqa: E402

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


def main() -> int:
    """Score every model on every split and write the artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 1, 7, 101, 2024])
    parser.add_argument("--quick", action="store_true", help="short fits, for a check")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args_cli = parser.parse_args()
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
    # Headers derived from the labels rather than hard-coded, and the groups
    # requested by name: zipping `.values()` against a fixed header list is only
    # safe while every group is guaranteed non-empty, which L23's three fixed
    # zeniths happen to be and PB24's eight are not (M5 task 6).
    zenith_groups = {
        name: V.group_rrms(truth[held], pred[held], zen[held], expected=L.ZENITHS)
        for name, pred in models.items()
    }
    zenith_labels = sorted(next(iter(zenith_groups.values())))
    lines.append(
        V.markdown_table(
            [
                [name, *(groups[label] for label in zenith_labels)]
                for name, groups in zenith_groups.items()
            ],
            ["model", *(f"{label:.0f} deg" for label in zenith_labels)],
        )
    )

    bp_labels, bp_edges = V.bp_bin_labels(batch.phase_params.B_p)
    bp_groups = {
        name: V.group_rrms(
            truth[held],
            pred[held],
            bp_labels[held],
            expected=range(len(bp_edges) - 1),
        )
        for name, pred in models.items()
    }
    bp_bins = sorted(next(iter(bp_groups.values())))
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
                [name, *(groups[label] for label in bp_bins)]
                for name, groups in bp_groups.items()
            ],
            ["model", *(f"bin {label:.0f}" for label in bp_bins)],
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

    jax.config.update("jax_enable_x64", True)
    rows = np.where(zen == 30)[0][:3]
    f64 = lambda x: jnp.asarray(np.asarray(x)[rows], dtype=jnp.float64)  # noqa: E731
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
