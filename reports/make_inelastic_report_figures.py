""" Generate the figures for reports/report_rt_inelastic_model.md.

Data sources (the RQ4 rule, inherited from make_report_figures.py: draw from
the committed validation artefacts, never recompute the science):

- ``design/validation/rrms_per_wavelength_inelastic.csv`` -- the per-wavelength
  total-rRMS ladder vs Rrs_X4 on held-out scenes, written by
  ``design/py/run_validation.py --inelastic``.
- ``design/validation/metrics_inelastic.csv`` -- the scalar gate/delta table,
  same provenance.
- The zenith-holdout numbers are transcribed from
  ``design/rt_inelastic_implementation.md`` (S5.3; notebook 4 S3) because the
  crippled retrain is deliberately never shipped or written to a CSV;
  ``design/py/train_inelastic_corr.py`` trains and reports it (the
  "zenith-holdout 0/30->60" variant) alongside the shipped heads.

Figures written beside this script:

- ``fig_inelastic_architecture.png``  -- the composed forward-model dataflow.
- ``fig_inelastic_rrms_ladder.png``   -- per-wavelength total rRMS vs X4,
  three models, held-out scenes, ungated regions shaded.
- ``fig_inelastic_deltas.png``        -- the nine per-process delta rows,
  analytic backbone -> corrected, against the +-5 % gate band.
- ``fig_inelastic_unseen_zenith.png`` -- the unseen-60 deg holdout cliff.

Palette: the three model hues reuse the house palette of
``make_report_figures.py`` (Okabe-Ito subset) in the same adjacent stacking
order it validated for CVD safety (protan/deutan dE >= 11.4, normal-vision
floor 18.4 on white); every series carries a direct label, so identity is
never color-alone.

Run with the ``ocean14`` conda environment from anywhere:

    python reports/make_inelastic_report_figures.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent / "design" / "validation"

# One color per model state, identical across figures (see module docstring).
COLOR = {
    "elastic-only": "#E69F00",
    "analytic inelastic": "#009E73",
    "corrected inelastic": "#0072B2",
    "holdout retrain": "#D55E00",
}
INK, INK_2 = "#333333", "#666666"
BAND_GRAY = "#eeeeee"  # ungated spectral regions / the +-5 % gate band

GATE_BAND = (400.0, 700.0)  # validation.INELASTIC_GATE_BAND (Q&A/Report ctx)
GATE_TOTAL = 0.5            # %  (design S6 line 1)
GATE_DELTA = 5.0            # %  (design S6 lines 2-3)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "font.size": 10.5,
})


def read_ladder():
    """Read the per-wavelength total-rRMS ladder (held-out scenes vs X4).

    Returns
    -------
    wave : np.ndarray
        Wavelength grid [nm].
    ladder : dict
        ``{model name: rRMS % per wavelength}``.
    """
    with open(VALIDATION / "rrms_per_wavelength_inelastic.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    wave = np.array([float(r["wavelength_nm"]) for r in rows])
    models = [k for k in rows[0] if k != "wavelength_nm"]
    ladder = {m: np.array([float(r[m]) for r in rows]) for m in models}
    return wave, ladder


def read_deltas():
    """Read the per-process delta table from metrics_inelastic.csv.

    Returns
    -------
    dict
        ``{(metric, zenith): (analytic %, corrected %)}`` with metric one of
        "Raman 550-700 nm", "Raman 490 nm", "fluorescence 685 nm".
    """
    with open(VALIDATION / "metrics_inelastic.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r["section"] not in ("delta_analytic_pct", "delta_corrected_pct"):
            continue
        key = (r["metric"], int(r["zenith"]))
        pair = out.setdefault(key, [None, None])
        pair[0 if r["section"] == "delta_analytic_pct" else 1] = float(r["value"])
    return {k: tuple(v) for k, v in out.items()}


def fig_architecture():
    """The composed elastic + inelastic forward model as a dataflow schematic."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axis_off()
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 68)

    def box(x, y, w, h, text, edge=INK_2, face="white", fs=8.8, lw=1.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face,
                                   edgecolor=edge, linewidth=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)

    def arrow(x0, y0, x1, y1, color=INK_2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3))

    el = COLOR["elastic-only"]
    phys = COLOR["analytic inelastic"]
    corr = COLOR["corrected inelastic"]

    # Inputs (the two new fields flagged inline).
    box(1, 40, 23, 24,
        "IOPs  $a$, $bb_w$, $bb_p$\n+ $a_{ph}$  (new)\n"
        "PhaseParams  $B_p$\nGeometry  $\\theta_s$ (+ opt. $E_d$)\n"
        "Inelastic  $\\varphi_C$  (new)\nwavelength  $\\lambda$")

    # Elastic hybrid (the PR #13 deliverable, untouched).
    box(30, 52, 34, 10,
        "elastic hybrid (prior effort)\n"
        r"$rrs = rrs_{\rm ZTT}\,(1+\delta_{\rm em})\ \to\ R_{rs}^{\rm el}$",
        edge=el, lw=1.8)
    ax.text(47, 63.5, "elastic path: bit-identical when inelastic=None",
            color=el, fontsize=8.5, ha="center", style="italic")

    # Raman branch.
    box(30, 33, 34, 14,
        "Raman (S&P98 two-flow, fixed BING)\n"
        "excitation $1/\\lambda' = 1/\\lambda + 3400\\,{\\rm cm^{-1}}$\n"
        "true sky ratio $E_d(\\lambda')/E_d(\\lambda)$\n"
        r"$\to$ analytic factor  $f_{\rm phys}$", edge=phys, lw=1.8)
    box(69, 35, 26, 10,
        "bounded head  $\\delta_R$  (129 par)\n"
        r"$f_R = 1 + (f_{\rm phys}\!-\!1)(1+\delta_R)$", edge=corr, lw=1.8)

    # Fluorescence branch.
    box(30, 14, 34, 14,
        "Chl-a fluorescence (Gordon-type)\n"
        "$b_{bF} = \\frac{1}{2}\\varphi_C\\,a_{ph}(\\lambda')$;  "
        "$L_u = E_u/\\pi$\n"
        "single Gaussian 685 nm\n"
        r"$\to$ $\varphi_C$-linear kernel  $K_{\rm fl}$", edge=phys, lw=1.8)
    box(69, 16, 26, 10,
        "bounded head  $\\delta_F$  (129 par)\n"
        r"$R_{rs}^{\rm fl} = \varphi_C\,K_{\rm fl}\,(1+\delta_F)$",
        edge=corr, lw=1.8)

    # Composition.
    box(76, 47, 26, 10,
        "composition\n"
        r"$R_{rs} = R_{rs}^{\rm el}\times f_R + R_{rs}^{\rm fl}$",
        edge=INK, lw=1.6)

    arrow(24, 57, 29, 57)
    arrow(24, 47, 29, 42)
    arrow(24, 44, 29, 23)
    arrow(64, 57, 78, 57)   # elastic Rrs into composition
    arrow(64, 40, 68, 40)
    arrow(64, 21, 68, 21)
    arrow(89, 45, 89, 46.8)   # Raman head up into composition
    arrow(95, 21, 100, 46.5)  # fluorescence into composition
    arrow(102, 52, 106, 52)
    ax.text(106.5, 54.5, "$R_{rs}$", fontsize=12, ha="right")

    ax.text(3, 30,
            "physics carries the shape;\nthe heads absorb the measured\n"
            "two-flow errors (zenith, trophic)",
            fontsize=8.3, color=phys, va="top", style="italic")
    ax.text(3, 12,
            "Everything is JAX: differentiable in every input incl. "
            "$\\partial R_{rs}/\\partial\\varphi_C$\n"
            "(gradients match finite differences to $\\leq 6\\times10^{-9}$). "
            "inelastic=None takes the\nelastic code route untouched "
            "(bit-identical, SHA-256-pinned).",
            fontsize=8.3, color=INK_2, va="top")
    ax.set_title(
        "The complete forward model:  "
        r"$R_{rs} = R_{rs}^{\rm el}\,\times f_R \; + \; "
        r"\varphi_C\,K_{\rm fl}\,(1+\delta_F)$",
        fontsize=12, color=INK, pad=8)

    fig.tight_layout()
    fig.savefig(HERE / "fig_inelastic_architecture.png", dpi=200)
    plt.close(fig)


def fig_rrms_ladder():
    """Per-wavelength total rRMS vs X4, three models, ungated regions shaded."""
    wave, ladder = read_ladder()
    order = ["elastic-only", "analytic inelastic", "corrected inelastic"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    # Ungated spectral regions (reported, never gated).
    ax.axvspan(wave[0], GATE_BAND[0], color=BAND_GRAY, zorder=0)
    ax.axvspan(GATE_BAND[1], wave[-1], color=BAND_GRAY, zorder=0)
    ax.text(375, 30, "reported,\nnot gated", ha="center", fontsize=8.5,
            color=INK_2)
    ax.text(725, 30, "reported,\nnot gated", ha="center", fontsize=8.5,
            color=INK_2)

    for name in order:
        emph = name == "corrected inelastic"
        ax.plot(wave, ladder[name], color=COLOR[name],
                lw=2.6 if emph else 1.8, zorder=3 if emph else 2, label=name)

    # The gate bar, drawn only over the band it applies to.
    ax.plot(GATE_BAND, [GATE_TOTAL, GATE_TOTAL], ls="--", lw=1.2, color=INK_2,
            zorder=2)
    ax.text(548, GATE_TOTAL * 1.13, "gate: per-zenith total $\\leq$ 0.5 %",
            fontsize=8.5, color=INK_2)

    # Direct labels at the right edge.
    label = {
        "elastic-only": "elastic-only\n(inelastic=None)",
        "analytic inelastic": "analytic backbone\n(no heads)",
        "corrected inelastic": "corrected\n(the default model)",
    }
    dy = {"elastic-only": 4, "analytic inelastic": 0, "corrected inelastic": -2}
    for name in order:
        ax.annotate(label[name], (wave[-1], ladder[name][-1]),
                    xytext=(6, dy[name]), textcoords="offset points",
                    va="center", fontsize=9, color=INK)

    ax.set_yscale("log")
    ax.set_xlim(350, 750)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30],
                  labels=["0.1", "0.3", "1", "3", "10", "30"])
    ax.grid(True, axis="y", which="major")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("total rRMS vs $R_{rs}^{X4}$ in $rrs$ space [%]")
    ax.set_title("Error against the all-processes-on ocean (X4) - "
                 "held-out scenes, all three zeniths", fontsize=11.5)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    fig.savefig(HERE / "fig_inelastic_rrms_ladder.png", dpi=200)
    plt.close(fig)


def fig_deltas():
    """The nine per-process delta rows: analytic -> corrected vs the gate band."""
    deltas = read_deltas()
    metrics = ["Raman 550-700 nm", "Raman 490 nm", "fluorescence 685 nm"]
    titles = {
        "Raman 550-700 nm": "Raman increment, median error 550-700 nm",
        "Raman 490 nm": "Raman increment, median error at 490 nm",
        "fluorescence 685 nm": "fluorescence peak, median error at 685 nm",
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.axvspan(-GATE_DELTA, GATE_DELTA, color=BAND_GRAY, zorder=0)
    ax.axvline(0, color=INK_2, lw=0.8, zorder=1)

    ypos, ylabels = [], []
    y = 0
    for metric in metrics:
        for zen in (0, 30, 60):
            analytic, corrected = deltas[(metric, zen)]
            ax.annotate("", xy=(corrected, y), xytext=(analytic, y),
                        arrowprops=dict(arrowstyle="->", lw=1.1,
                                        color=INK_2, alpha=0.8))
            ax.plot([analytic], [y], "o", ms=8,
                    color=COLOR["analytic inelastic"], zorder=3)
            ax.plot([corrected], [y], "o", ms=8,
                    color=COLOR["corrected inelastic"], zorder=4)
            ypos.append(y)
            ylabels.append(f"{zen}°")
            y += 1
        y += 0.9  # group gap

    # Group titles between the blocks.
    for i, metric in enumerate(metrics):
        ax.text(-45, i * 3.9 + 1.0, titles[metric], fontsize=9.5, color=INK,
                ha="left", va="center", fontweight="bold")

    # Direct labels for the two states (identity never color-alone).
    a0, c0 = deltas[("Raman 550-700 nm", 0)]
    ax.annotate("analytic backbone", (a0, 0), xytext=(0, 11),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=COLOR["analytic inelastic"])
    ax.annotate("corrected", (c0, 0), xytext=(8, 11),
                textcoords="offset points", ha="left", fontsize=8.5,
                color=COLOR["corrected inelastic"])
    ax.text(0, y - 0.4, "±5 % gate band", ha="center", fontsize=8.5,
            color=INK_2)

    ax.set_yticks(ypos, labels=ylabels, fontsize=9)
    ax.set_ylabel("solar zenith")
    ax.set_xlim(-45, 37)
    ax.set_ylim(-0.9, y + 0.2)
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.set_xlabel("median per-process delta error, held-out scenes [%]")
    ax.set_title("Per-process errors: the heads move every row into the "
                 "gate band", fontsize=11.5)

    fig.tight_layout()
    fig.savefig(HERE / "fig_inelastic_deltas.png", dpi=200)
    plt.close(fig)


def fig_unseen_zenith():
    """The zenith-holdout diagnostic: heads retrained without 60 deg, scored
    at 60 deg.

    Numbers transcribed from the implementation record S5.3 (the shipped and
    analytic values also live in ``metrics_inelastic.csv``; the crippled
    retrain is reported-only by design and is never committed as weights).
    """
    rows = [  # (process, analytic, shipped, crippled)
        ("Raman increment\n550-700 nm @ 60°", -4.21, -0.21, -74.0),
        ("fluorescence peak\n685 nm @ 60°", -13.71, +0.10, -9.2),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.axvspan(-GATE_DELTA, GATE_DELTA, color=BAND_GRAY, zorder=0)
    ax.axvline(0, color=INK_2, lw=0.8, zorder=1)

    spec = [
        ("analytic backbone (no head)", "analytic inelastic", 1),
        ("shipped head (trained on 0°/30°/60°)", "corrected inelastic", 2),
        ("head retrained without 60°", "holdout retrain", 3),
    ]
    for y, (_name, analytic, shipped, crippled) in enumerate(rows):
        for value, (_, key, _) in zip((analytic, shipped, crippled), spec, strict=True):
            ax.plot([value], [y], "o", ms=9, color=COLOR[key], zorder=3)
        ax.annotate(f"{crippled:+.0f} %", (crippled, y), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=COLOR["holdout retrain"])

    handles = [plt.Line2D([], [], marker="o", ls="", ms=8,
                          color=COLOR[key], label=name)
               for name, key, _ in spec]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False)

    ax.set_yticks(range(len(rows)), labels=[r[0] for r in rows], fontsize=9)
    ax.set_xlim(-82, 12)
    ax.set_ylim(-0.6, 1.6)
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.set_xlabel("median delta error at 60° [%]   "
                  "(shaded: the ±5 % gate band)")
    ax.set_title("The heads are interpolators in geometry: an unseen zenith "
                 "breaks δ$_R$", fontsize=11.5)

    fig.tight_layout()
    fig.savefig(HERE / "fig_inelastic_unseen_zenith.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_rrms_ladder()
    fig_deltas()
    fig_unseen_zenith()
    print(f"wrote 4 figures to {HERE}")
