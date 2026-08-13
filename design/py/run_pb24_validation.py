#!/usr/bin/env python
"""
Score the analytic models on PB24 — M5 task 9, *before* anything is trained.

The point of running this first is that it fixes the target. M4's numbers were
set on L23, where O25 refit reaches 0.69%; PB24 is a different dataset with a
12x span in ``B_p`` and a full BRDF, and until it is measured nobody knows what
"good" is here.

It also answers a question the prototype never could: **does the analytic
backbone degrade off-nadir on its own?** ZTT is genuinely BRDF-aware -- it forms
the scattering angle from ``theta_v`` and ``dphi`` -- but L23 fixed the view at
nadir, so that machinery has never been exercised against data.

Three models, on identical data:

- **standard Gordon** — no geometry at all, the floor.
- **ZTT backbone** — the analytic model, geometry-aware, no fitted parameters
  beyond the published ones.
- **O25 form, refit on PB24** — the benchmark, given its best shot: coefficients
  on the full ``(theta_s, theta_v, dphi)`` grid (M5 task 8) and scored through
  the fitted surface transfer (task 7), because with the nadir transfer the
  interface error swamps the comparison.

Every model is refit **on each split's own training mask** and scored on that
split's held-out side.

    python design/py/run_pb24_validation.py                 # write the artefacts
    python design/py/run_pb24_validation.py --quick --out /tmp/x
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robust.rt import baselines as B  # noqa: E402
from robust.rt import conventions as C  # noqa: E402
from robust.rt import validation as V  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import pb24 as P  # noqa: E402

#: Where the artefacts land.
OUT_DIR = Path(__file__).resolve().parents[1] / "validation_pb24"

#: Realisations loaded for the committed run. Measured stability of the held-out
#: numbers across 50 / 100 / 200 / 400 realisations: Gordon 21.4-20.5%, O25
#: 6.31-6.77% -- so a few percent relative, and 200 is comfortably enough.
#: **ZTT is the exception**: its rRMS ranges over 7061-18972% across the same
#: sizes because a handful of non-physical predictions determine it. Its stable
#: statistics are the median (10.9-14.2%) and the non-physical share
#: (14.2-14.8%), which is why both are reported.
N_REALISATION = 200

#: Geometry stride. 1 = every geometry. Kept explicit because a stride sharing a
#: factor with the 13 azimuths silently deletes the BRDF axis (record §7.7).
GEOMETRY_STRIDE = 1

#: The CVD-safe palette M4 settled on.
COLOURS = {
    "standard Gordon": "#9a9a9a",
    "ZTT backbone": "#0072B2",
    "O25 form, refit on PB24": "#D55E00",
}


def write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary file in the same directory."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem, suffix=path.suffix
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv_atomically(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write a CSV through :mod:`csv` — the model names contain commas."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    write_atomically(path, buffer.getvalue())


def save_figure_atomically(fig, path: Path) -> None:
    """Render to bytes, then replace — never leave a half-written PNG committed."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".png")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        fig.savefig(tmp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build_models(batch, train, transfer):
    """Every model's prediction on the whole batch, in ``rrs`` space.

    Predictions rather than callables, so "identical data" is literal rather than
    a claim about two code paths. O25 is refit on ``train`` alone.
    """
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    o25 = B.fit_o25_table(batch.iops, batch.Rrs, batch.geometry, train=train)
    return {
        "standard Gordon": B.rrs_gordon(*args),
        "ZTT backbone": Z.rrs_ZTT(*args),
        "O25 form, refit on PB24": B.rrs_o25(*args, coeffs=o25, transfer=transfer),
    }, o25


def cut_table(models, truth, labels, mask, expected, fmt):
    """One breakdown table: rows are models, columns are label groups.

    Returns ``(headers, rows, groups)`` where ``groups`` is the per-model dict, so
    a caller can assert the headers and the values came from the same labels
    rather than from two independent orderings (M5 task 6).
    """
    groups = {
        name: V.group_rrms(truth[mask], pred[mask], labels[mask], expected=expected)
        for name, pred in models.items()
    }
    keys = sorted(next(iter(groups.values())))
    headers = ["model", *(fmt(k) for k in keys)]
    rows = [[name, *(groups[name][k] for k in keys)] for name in groups]
    return headers, rows, groups


def make_figures(out_dir, wave, per_lambda, view_rows, bp_rows):
    """Three figures: the spectral ladder, the view-angle ladder, the B_p ladder."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, values in per_lambda.items():
        ax.plot(wave, values, color=COLOURS[name], linewidth=2, label=name)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("rRMS (%)")
    ax.set_title("PB24, held-out realisations: error per wavelength")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    save_figure_atomically(fig, out_dir / "rrms_per_wavelength.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, (angles, values) in view_rows.items():
        ax.plot(
            angles,
            values,
            color=COLOURS[name],
            linewidth=2,
            marker="o",
            markersize=5,
            label=name,
        )
    ax.set_xlabel("sensor zenith (degrees)")
    ax.set_ylabel("rRMS (%)")
    ax.set_title("PB24, held-out realisations: error per view angle")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    save_figure_atomically(fig, out_dir / "rrms_per_view_angle.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, (bins, values) in bp_rows.items():
        ax.plot(
            bins,
            values,
            color=COLOURS[name],
            linewidth=2,
            marker="s",
            markersize=5,
            label=name,
        )
    ax.set_xlabel("$B_p$ bin (equal count, ascending)")
    ax.set_ylabel("rRMS (%)")
    ax.set_title("PB24, held-out realisations: error per $B_p$ bin")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    save_figure_atomically(fig, out_dir / "rrms_per_bp_bin.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realisations", type=int, default=N_REALISATION)
    parser.add_argument("--stride", type=int, default=GEOMETRY_STRIDE)
    parser.add_argument("--quick", action="store_true", help="40 realisations")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.quick and args.out == OUT_DIR:
        raise SystemExit(
            "refusing --quick with the default --out: it would overwrite the "
            f"committed artefacts in {OUT_DIR} with small-sample numbers."
        )
    if not os.environ.get("OS_COLOR"):
        raise SystemExit("$OS_COLOR is not set, so PB24 cannot be read.")

    n = 40 if args.quick else args.realisations
    print(f"loading {n} realisations, all geometries ...")
    batch = P.load_batch(realisations=n, angles="all", geometry_stride=args.stride)
    print(f"  {batch.report.summary()}")

    splits = P.make_splits(batch, kinds=P.SPLIT_KINDS)
    for kind in P.SPLIT_KINDS:
        print(f"  {splits.reports[kind].summary()}")

    transfer = C.default_transfer()
    truth = batch.rrs
    theta_v = batch.theta_v
    theta_s = batch.theta_s
    bp_labels, bp_edges = V.bp_bin_labels(batch.phase_params.B_p)

    # --- the headline: every split, each with its own refit ------------------
    # The realisation and B_p splits are restricted to the **sanctioned window**
    # (Q14's 0-70 deg). Left unrestricted on an angles="all" batch they would mix
    # in-window and shell samples on both sides, so a single number would answer
    # two questions at once -- "does it generalise to new water?" and "does it
    # extrapolate past 70 deg?" -- and neither cleanly. The geometry split is the
    # one that carries the shell, by construction.
    in_window = (theta_s <= P.ANGLE_WINDOW_MAX) & (theta_v <= P.ANGLE_WINDOW_MAX)
    restrict = {
        "realisation": in_window,
        "bp_band": in_window,
        "geometry": np.ones_like(in_window),
    }

    table: dict[str, dict[str, float]] = {}
    for kind in P.SPLIT_KINDS:
        keep = restrict[kind]
        train, test = splits.train(kind) & keep, splits.test(kind) & keep
        models, _ = build_models(batch, train, transfer)
        scores = V.score_models(models, truth, {"train": train, "test": test})
        for name, by_split in scores.items():
            table.setdefault(name, {})[f"{kind} (train)"] = by_split["train"]
            table.setdefault(name, {})[f"{kind} (held out)"] = by_split["test"]

    # --- breakdowns, on the realisation split's held-out side ----------------
    # These keep the full angle range on purpose: the per-view-angle cut is the
    # point, and cutting it off at 70 deg would hide the shell it exists to show.
    train, held = splits.train("realisation"), splits.test("realisation")
    models, _ = build_models(batch, train & in_window, transfer)

    # A model that returns a non-physical reflectance is not "inaccurate", it is
    # out of its domain, and an rRMS cannot say so -- the same few samples that
    # dominate the RMS also hide how many there are. So count them.
    nonphysical = {
        name: float((np.asarray(pred) <= 0.0).mean()) * 100.0
        for name, pred in models.items()
    }
    bb_over_a = np.asarray(batch.iops.bb / batch.iops.a)
    mu_inf = np.asarray(
        Z.mu_infinity_tt2017(
            batch.iops.bb / batch.iops.a, batch.iops.bb_w / batch.iops.bb
        )
    )
    domain = {
        "bb_over_a_max": float(bb_over_a.max()),
        "mu_inf_min": float(mu_inf.min()),
        "mu_inf_nonpositive_pct": float((mu_inf <= 0.0).mean()) * 100.0,
        "beyond_l23_bb_over_a_pct": float((bb_over_a > 0.5946).mean()) * 100.0,
        "ztt_median_pct": float(
            np.median(
                np.abs(np.asarray(models["ZTT backbone"])[held] / truth[held] - 1.0)
            )
        )
        * 100.0,
    }

    per_lambda = {
        name: np.asarray(V.rrms_per_wavelength(truth[held], pred[held]))
        for name, pred in models.items()
    }
    view_headers, view_rows, view_groups = cut_table(
        models, truth, theta_v, held, np.unique(theta_v), lambda k: f"{k:.1f} deg"
    )
    sun_headers, sun_rows, _ = cut_table(
        models, truth, theta_s, held, np.unique(theta_s), lambda k: f"{k:.1f} deg"
    )
    bp_headers, bp_rows, bp_groups = cut_table(
        models,
        truth,
        bp_labels,
        held,
        range(len(bp_edges) - 1),
        lambda k: f"bin {k:.0f}",
    )

    # --- the document --------------------------------------------------------
    columns = [f"{k} ({s})" for k in P.SPLIT_KINDS for s in ("train", "held out")]
    lines = [
        "# M5 task 9 — the analytic models on PB24",
        "",
        f"*Generated by `design/py/run_pb24_validation.py` on {n} realisations "
        f"({batch.n_sample} samples), stride {args.stride}. rRMS %, in `rrs` space, "
        "lower is better.*",
        "",
        "**O25 is a refit, not the published model**, and it is refit on **each "
        "split's own training mask**. Its coefficients span the full "
        "`(theta_s, theta_v, dphi)` grid (task 8) and it is scored through the "
        "fitted surface transfer (task 7); with the nadir transfer the interface "
        "error swamps the comparison and O25 looks ~3x worse than it is.",
        "",
        "## Every model on every split",
        "",
        V.markdown_table(
            [
                [name, *(scores[c] for c in columns), nonphysical[name]]
                for name, scores in table.items()
            ],
            ["model", *columns, "non-physical %"],
        ),
        "",
        "The last column is the share of predicted `rrs` values that are **zero or "
        "negative**. A model producing those is not inaccurate, it is outside its "
        "domain, and an rRMS cannot say so — the same handful of samples that "
        "dominate the RMS also conceal how many there are.",
        "",
        "## ZTT is outside its validity domain on PB24",
        "",
        f"ZTT's `mu_infinity` comes from Twardowski & Tonizzo (2017), a fit in "
        f"`bb/a` and `eta_bb`. L23 spans `bb/a` up to **0.59**; PB24 reaches "
        f"**{domain['bb_over_a_max']:.2f}**, and "
        f"{domain['beyond_l23_bb_over_a_pct']:.1f}% of its values lie beyond "
        "anything L23 probed. Out there the fitted polynomial goes **negative** "
        f"(minimum {domain['mu_inf_min']:.4f}, non-positive for "
        f"{domain['mu_inf_nonpositive_pct']:.3f}% of values), which flips the sign "
        "of the `(a/bb)(1 - cos(theta_v) psi / mu_inf)` term and with it the whole "
        "denominator.",
        "",
        "So the ZTT row above is **not a statement about the backbone's accuracy**; "
        "it is a statement that the backbone has an unstated validity domain which "
        "L23 never left and PB24 leaves routinely. Its rRMS is not even a stable "
        "quantity — measured over 50/100/200/400 realisations it ranges over "
        "7061-18972%, because a handful of sign-flipped predictions determine it. "
        f"The statistics that *are* stable: a median relative error of "
        f"{domain['ztt_median_pct']:.1f}% (against 5.9% on L23) and a non-physical "
        f"share of {nonphysical['ZTT backbone']:.1f}%.",
        "",
        "**This matters directly for task 11.** The hybrid is "
        "`rrs_ZTT * (1 + delta)` with `|delta| <= 0.5`, so a bounded *relative* "
        "correction cannot repair a backbone that has the wrong sign — on roughly "
        "a fifth of PB24 the hybrid as constructed has nothing to correct toward. "
        "See **Q17**.",
        "",
        "## Per sensor zenith (held-out realisations)",
        "",
        V.markdown_table(view_rows, view_headers),
        "",
        "## Per solar zenith (held-out realisations)",
        "",
        V.markdown_table(sun_rows, sun_headers),
        "",
        "## Per `B_p` bin (held-out realisations, equal-count bins)",
        "",
        "Bin edges: "
        + ", ".join(f"{e:.5f}" for e in bp_edges)
        + f" — a factor {bp_edges[-1] / bp_edges[0]:.2f} in total. Unlike L23's 1.75x, "
        "this **does** speak to phase-function dependence.",
        "",
        V.markdown_table(bp_rows, bp_headers),
        "",
    ]

    make_figures(
        args.out,
        np.asarray(batch.wave),
        per_lambda,
        {
            n_: (np.array(sorted(g)), np.array([g[k] for k in sorted(g)]))
            for n_, g in view_groups.items()
        },
        {
            n_: (np.array(sorted(g)), np.array([g[k] for k in sorted(g)]))
            for n_, g in bp_groups.items()
        },
    )

    write_atomically(args.out / "metrics.md", "\n".join(lines) + "\n")
    write_csv_atomically(
        args.out / "metrics.csv",
        ["model", *columns, "non_physical_pct"],
        [
            [name, *(f"{scores[c]:.4f}" for c in columns), f"{nonphysical[name]:.4f}"]
            for name, scores in table.items()
        ],
    )
    write_csv_atomically(
        args.out / "rrms_per_wavelength.csv",
        ["wavelength_nm", *per_lambda],
        [
            [f"{w:.1f}", *(f"{per_lambda[name][i]:.4f}" for name in per_lambda)]
            for i, w in enumerate(np.asarray(batch.wave))
        ],
    )
    write_csv_atomically(
        args.out / "rrms_per_view_angle.csv",
        view_headers,
        [[row[0], *(f"{v:.4f}" for v in row[1:])] for row in view_rows],
    )
    print(
        f"\nwrote {args.out}/metrics.md, metrics.csv, rrms_per_wavelength.csv, "
        "rrms_per_view_angle.csv and 3 figures"
    )
    print("\n" + "\n".join(lines[7:13]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
