#!/usr/bin/env python
"""
Re-score the ZTT backbone on L23 — the M7 acceptance gate (task 6, ``rt_fit_ztt.md``).

An **additive ladder**, each step switching on one L23 term, so each gain is
attributed to the term that earned it:

1. ZTT (TT2017 µ∞)                 ``variant="tt2017"`` -- the M0–M6 backbone
2. + L23 µd                         ``mu_d_model="l23"``
3. + L23 µ∞                         ``mu_inf_model="l23"`` (Sullivan Pbb still)
4. + FF Pbb  = **ZTT (L23 µ∞)**     ``variant="l23"`` -- the new default
5. closure µ∞ (a ceiling)           ``mu_inf_model="l23_closure"``

plus one **diagnostic** step (Q16 answer (a)): the L23 variant with
Ψ_KLu/µ∞ replaced by L23's *measured* KLu(0⁻)/a (an oracle; nadir only). It
sizes what correcting Ψ_KLu could be worth before anyone fits a correction.
Standard Gordon and the O25 form refit on L23 come from the M4 ladder.

Because ladder gains depend on the order the terms go in, the attribution is
also reported both ways: each term **added alone** to TT2017, and **removed
alone** from the L23 variant.

Acceptance (task 6): ZTT (L23 µ∞) beats ZTT (TT2017 µ∞) on held-out scenes at
**each** of 0°, 30°, 60°. Reported plainly either way; nothing is tuned.

Writes ``design/ztt_l23/ladder.{md,csv}``, ``ladder_per_wavelength.csv``,
``fig_ladder_per_wavelength.png``, ``fig_ladder_per_zenith.png``. (Not
``metrics.*``, the name the prompt used: ``fit_ztt_l23.py`` already writes those.)

    python design/py/run_ztt_l23.py
    python design/py/run_ztt_l23.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robust.rt import baselines as B  # noqa: E402
from robust.rt import conventions as C  # noqa: E402
from robust.rt import validation as V  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import l23 as L  # noqa: E402

OUT = ROOT / "design" / "ztt_l23"

#: The ladder, in order: (label, rrs_ZTT keywords).
LADDER = (
    ("ZTT (TT2017 µ∞)", {"variant": "tt2017"}),
    ("+ L23 µd", {"variant": "tt2017", "mu_d_model": "l23"}),
    ("+ L23 µ∞", {"variant": "tt2017", "mu_d_model": "l23", "mu_inf_model": "l23"}),
    ("+ FF Pbb = ZTT (L23 µ∞)", {"variant": "l23"}),
    ("closure µ∞ (ceiling)", {"variant": "l23", "mu_inf_model": "l23_closure"}),
)
BASE, NEW = LADDER[0][0], LADDER[3][0]
ORACLE = "L23 variant + oracle KLu/a (diagnostic)"

#: Each term alone: added to TT2017, and removed from L23.
ALONE = {
    "µd": (
        {"variant": "tt2017", "mu_d_model": "l23"},
        {"variant": "l23", "mu_d_model": "tt2017"},
    ),
    "µ∞": (
        {"variant": "tt2017", "mu_inf_model": "l23"},
        {"variant": "l23", "mu_inf_model": "tt2017"},
    ),
    "FF Pbb": (None, None),  # filled in run(): the P_bb override
}

ZENITHS = L.ZENITHS


def oracle_mu_inf(batch, product):
    """µ∞ that makes ZTT's Ψ_KLu/µ∞ equal L23's measured KLu(0⁻)/a.

    At nadir the term is (a/bb)(1 + Ψ_KLu/µ∞); setting µ∞ = Ψ_KLu(ψ) a / KLu0
    replaces the whole ratio with the measurement. Batch order is zenith-major,
    matching the product's leading zenith axis.
    """
    _, _, _, psi = Z.geometry_to_paper_angles(batch.geometry)
    psi_klu = np.asarray(Z.psi_KLu(psi))[:, None]
    klu0 = product.KLu0.astype(float).reshape(-1, product.KLu0.shape[-1])
    return psi_klu * np.asarray(batch.iops.a) / klu0


def run():
    """Every model's rrs on the full batch, and the masks to score them on."""
    batch = L.load_batch()  # netCDF: FF fields present
    splits = L.make_splits(batch)
    truth = np.asarray(C.Rrs_to_rrs(batch.Rrs))
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    product = L.load_profile_product(L.default_profile_product_path())

    models = {"standard Gordon": np.asarray(B.rrs_gordon(*args))}
    models["O25 form, refit on L23"] = np.asarray(B.rrs_o25(*args))
    for label, kw in LADDER:
        models[label] = np.asarray(Z.rrs_ZTT(*args, **kw))
    models[ORACLE] = np.asarray(Z.rrs_ZTT(*args, mu_inf=oracle_mu_inf(batch, product)))

    # Each term alone. FF Pbb: TT2017 + the FF shape, and L23 with Sullivan's.
    _, _, _, psi = Z.geometry_to_paper_angles(batch.geometry)
    psi_b = np.asarray(psi)[:, None]
    ff = Z.P_bb_from_phase(batch.phase_params, psi_b, use_ff=True)
    sullivan = Z.P_bb_sullivan(psi_b)
    alone = dict(ALONE)
    alone["FF Pbb"] = (
        {"variant": "tt2017", "P_bb": ff},
        {"variant": "l23", "P_bb": sullivan},
    )
    alone_models = {
        term: tuple(np.asarray(Z.rrs_ZTT(*args, **kw)) for kw in pair)
        for term, pair in alone.items()
    }

    zen = batch.zenith
    masks = {"train": splits.scene_train, "held-out": splits.scene_test}
    for z in ZENITHS:
        masks[f"held-out {z}°"] = splits.scene_test & (zen == z)
    bp_labels, bp_edges = V.bp_bin_labels(np.asarray(batch.phase_params.B_p))
    oracle_depths = oracle_depth_sweep(batch, truth, args, splits.scene_test)
    return (
        batch,
        truth,
        models,
        alone_models,
        masks,
        (bp_labels, bp_edges),
        oracle_depths,
    )


#: Depths (m) at which the oracle's KLu is also taken. ZTT's K_Lu is the
#: attenuation of upwelling radiance along the path, which the paper does not
#: pin to the surface, so the oracle is checked at several depths before its
#: verdict is read as being about Ψ_KLu rather than about where KLu is sampled.
ORACLE_DEPTHS = (0.0, 1.0, 5.0, 10.0, 20.0)


def oracle_depth_sweep(batch, truth, args, held):
    """Held-out rRMS of the oracle step with KLu from each depth, and the
    quantiles of ZTT's implied K_Lu (Ψ_KLu a / µ∞_L23) over the measured one."""
    _, _, _, psi = Z.geometry_to_paper_angles(batch.geometry)
    psi_klu = np.asarray(Z.psi_KLu(psi))[:, None]
    a = np.asarray(batch.iops.a)
    x = np.asarray(batch.iops.bb / batch.iops.a)
    eta = np.asarray(batch.iops.bb_w / batch.iops.bb)
    implied = psi_klu * a / np.asarray(Z.mu_infinity_l23(x, eta))
    reads = [
        L._read_profile(L.ELASTIC_X, z, ("KLu_z",), ORACLE_DEPTHS) for z in ZENITHS
    ]
    rows = []
    for k, depth in enumerate(ORACLE_DEPTHS):
        klu = np.concatenate([r["KLu_z"][k] for r in reads]).astype(float)
        rrs = np.asarray(Z.rrs_ZTT(*args, mu_inf=psi_klu * a / klu))
        ratio = (implied / klu)[held]
        rows.append(
            (
                depth,
                float(V.rrms(truth[held], rrs[held])),
                [float(v) for v in np.nanpercentile(ratio, (16, 50, 84))],
            )
        )
    return rows


def tables(batch, truth, models, alone_models, masks, bp, oracle_depths):
    """Markdown report, CSV rows, per-λ rows, and the gate verdict."""
    scores = V.score_models(models, truth, masks)
    held = masks["held-out"]
    per_lambda = {
        name: np.asarray(V.rrms_per_wavelength(truth[held], pred[held]))
        for name, pred in models.items()
    }
    bp_labels, bp_edges = bp
    per_bp = {
        name: V.group_rrms(
            truth[held], pred[held], bp_labels[held], expected=range(len(bp_edges) - 1)
        )
        for name, pred in models.items()
    }
    gate = {
        z: scores[NEW][f"held-out {z}°"] < scores[BASE][f"held-out {z}°"]
        for z in ZENITHS
    }

    split_names = list(masks)
    lines = [
        "# ZTT (L23 µ∞) vs ZTT (TT2017 µ∞) — the M7 ladder (task 6)",
        "",
        "*Generated by `design/py/run_ztt_l23.py`. Do not edit by hand.*",
        "",
        "rRMS (%) in rrs space on L23 elastic (3320 scenes × 3 zeniths × 81 λ); "
        f"held-out = the {int(np.unique(batch.scene[held]).size)}-scene split "
        f"(seed {L.SPLIT_SEED}). Each ladder step switches on one L23 term.",
        "",
        "## Acceptance gate",
        "",
        "ZTT (L23 µ∞) beats ZTT (TT2017 µ∞) on held-out scenes at each zenith: "
        + ", ".join(
            f"{z}° {'✅' if ok else '❌'} ({scores[NEW][f'held-out {z}°']:.2f} vs "
            f"{scores[BASE][f'held-out {z}°']:.2f})"
            for z, ok in gate.items()
        )
        + f" — **{'PASS' if all(gate.values()) else 'FAIL'}**.",
        "",
        "## The ladder",
        "",
        "| model | " + " | ".join(split_names) + " |",
        "|---" * (len(split_names) + 1) + "|",
    ]
    order = [label for label, _ in LADDER] + [
        ORACLE,
        "O25 form, refit on L23",
        "standard Gordon",
    ]
    for name in order:
        cells = " | ".join(f"{scores[name][s]:.2f}" for s in split_names)
        bold = "**" if name == NEW else ""
        lines.append(f"| {bold}{name}{bold} | {cells} |")

    lines += [
        "",
        "## Attribution (held-out rRMS change, percentage points)",
        "",
        "Ladder order: the change each step makes in the order above. Alone: the "
        "change from adding the term by itself to TT2017, and the change from "
        "removing it by itself from the L23 variant (positive = removing it hurts).",
        "",
        "| term | ladder step: all / 0° / 30° / 60° | added alone to TT2017 | "
        "removed alone from L23 |",
        "|---|---|---|---|",
    ]
    ladder_names = [label for label, _ in LADDER]
    for term, step in (("µd", 1), ("µ∞", 2), ("FF Pbb", 3)):
        prev, cur = ladder_names[step - 1], ladder_names[step]
        cols = [
            scores[cur][s] - scores[prev][s]
            for s in ["held-out", *[f"held-out {z}°" for z in ZENITHS]]
        ]
        add, rem = alone_models[term]
        add_d = float(V.rrms(truth[held], add[held])) - scores[BASE]["held-out"]
        rem_d = float(V.rrms(truth[held], rem[held])) - scores[NEW]["held-out"]
        lines.append(
            f"| {term} | "
            + " / ".join(f"{c:+.2f}" for c in cols)
            + f" | {add_d:+.2f} | {rem_d:+.2f} |"
        )
    oracle_gain = scores[ORACLE]["held-out"] - scores[NEW]["held-out"]
    closure_gain = scores[LADDER[4][0]]["held-out"] - scores[NEW]["held-out"]
    lines += [
        f"| closure µ∞ (ceiling) | {closure_gain:+.2f} (from the L23 variant) | | |",
        f"| oracle KLu/a (Q16) | {oracle_gain:+.2f} (from the L23 variant) | | |",
        "",
        "## The oracle KLu/a step, by sampling depth (Q16)",
        "",
        "Ψ_KLu/µ∞ replaced by L23's measured KLu(z)/a. ZTT's own implied K_Lu is "
        "Ψ_KLu a / µ∞(L23).",
        "",
        "| KLu taken at | held-out rRMS | ZTT implied / measured K_Lu, 16 / 50 / 84% |",
        "|---|---|---|",
        *[
            f"| {d:.0f} m | {r:.2f}% | " + " / ".join(f"{v:.3f}" for v in q) + " |"
            for d, r, q in oracle_depths
        ],
        f"| *(ZTT (L23 µ∞), for reference)* | {scores[NEW]['held-out']:.2f}% | |",
        "",
        "## Per B_p bin (held out; equal-count bins of the sample-mean B_p)",
        "",
        "| model | "
        + " | ".join(
            f"[{bp_edges[i]:.4f}, {bp_edges[i + 1]:.4f})"
            for i in range(len(bp_edges) - 1)
        )
        + " |",
        "|---" * len(bp_edges) + "|",
    ]
    for name in order:
        lines.append(
            f"| {name} | "
            + " | ".join(f"{per_bp[name][i]:.2f}" for i in range(len(bp_edges) - 1))
            + " |"
        )
    wave = np.asarray(batch.wave)
    pick = [
        int(np.argmin(np.abs(wave - w)))
        for w in (350, 400, 450, 500, 550, 600, 650, 700, 750)
    ]
    lines += [
        "",
        "## Per wavelength (held out; full table in `ladder_per_wavelength.csv`)",
        "",
        "| model | " + " | ".join(f"{wave[i]:.0f}" for i in pick) + " |",
        "|---" * (len(pick) + 1) + "|",
    ]
    for name in order:
        lines.append(
            f"| {name} | "
            + " | ".join(f"{per_lambda[name][i]:.2f}" for i in pick)
            + " |"
        )
    lines.append("")

    csv_rows = [
        [name, *(f"{scores[name][s]:.6f}" for s in split_names)] for name in order
    ]
    lambda_rows = [
        [f"{w:.0f}", *(f"{per_lambda[n][i]:.6f}" for n in order)]
        for i, w in enumerate(wave)
    ]
    return (
        "\n".join(lines),
        ["model", *split_names],
        csv_rows,
        ["wave", *order],
        lambda_rows,
        gate,
        scores,
        per_lambda,
        order,
    )


def to_csv(header, rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


#: Okabe–Ito, CVD-safe, assigned in fixed order to the ladder (never cycled).
PALETTE = ("#E69F00", "#56B4E9", "#009E73", "#0072B2", "#CC79A7", "#D55E00")


def figures(batch, scores, per_lambda, order, out: Path):
    from run_validation import INK_MUTED, apply_house_style, save_figure_atomically

    plt = apply_house_style()
    wave = np.asarray(batch.wave)
    shown = [label for label, _ in LADDER] + [ORACLE]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for k, name in enumerate(shown):
        ax.plot(
            wave,
            per_lambda[name],
            color=PALETTE[k],
            lw=2.4 if name == NEW else 1.6,
            ls="--" if name in (LADDER[4][0], ORACLE) else "-",
            label=name,
        )
    ax.plot(
        wave,
        per_lambda["O25 form, refit on L23"],
        color=INK_MUTED,
        lw=1.2,
        ls=":",
        label="O25 form, refit on L23",
    )
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("held-out rRMS (%)")
    ax.set_ylim(bottom=0)
    ax.set_title(
        "ZTT ladder on L23, held-out scenes (all zeniths)", loc="left", fontsize=10.5
    )
    ax.legend(fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout()
    save_figure_atomically(fig, out / "fig_ladder_per_wavelength.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    width = 0.8 / len(shown)
    x = np.arange(len(ZENITHS))
    for k, name in enumerate(shown):
        vals = [scores[name][f"held-out {z}°"] for z in ZENITHS]
        bars = ax.bar(
            x + (k - (len(shown) - 1) / 2) * width,
            vals,
            width * 0.92,
            color=PALETTE[k],
            label=name,
        )
        if name in (shown[0], NEW):
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"θs = {z}°" for z in ZENITHS])
    ax.set_ylabel("held-out rRMS (%)")
    ax.set_title("Per solar zenith", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=2, loc="upper left")
    ax.set_ylim(
        0, 1.25 * max(scores[n][f"held-out {z}°"] for n in shown for z in ZENITHS)
    )
    fig.tight_layout()
    save_figure_atomically(fig, out / "fig_ladder_per_zenith.png")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="write nothing")
    args = parser.parse_args(argv)

    batch, truth, models, alone_models, masks, bp, oracle_depths = run()
    (text, header, rows, lheader, lrows, gate, scores, per_lambda, order) = tables(
        batch, truth, models, alone_models, masks, bp, oracle_depths
    )
    print(text)
    if args.dry_run:
        return 0 if all(gate.values()) else 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ladder.md").write_text(text)
    (OUT / "ladder.csv").write_text(to_csv(header, rows))
    (OUT / "ladder_per_wavelength.csv").write_text(to_csv(lheader, lrows))
    figures(batch, scores, per_lambda, order, OUT)
    print(f"wrote ladder.md, ladder.csv, ladder_per_wavelength.csv, figures to {OUT}")
    return 0 if all(gate.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
