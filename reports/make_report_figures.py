""" Generate the figures for reports/report_rt_elastic_model.md.

Data sources (RQ3: draw from the committed validation artefacts):

- ``design/validation/rrms_per_wavelength.csv`` -- the per-wavelength rRMS
  ladder for every model on the held-out scenes, written by
  ``design/py/run_validation.py``.
- ``design/validation/metrics.csv`` -- the scalar model x split table, same
  provenance.
- The unseen-60 deg numbers are transcribed from
  ``design/validation/metrics.md`` ("Unseen 60 deg" table, seeds 23, 1, 7,
  101, 2024) because ``run_validation.py`` does not write them to a CSV;
  regenerate them with that script after any change to the models or splits.

Figures written beside this script:

- ``fig_architecture.png``  -- the hybrid forward-model dataflow.
- ``fig_rrms_ladder.png``   -- per-wavelength rRMS, five models, held-out scenes.
- ``fig_unseen_zenith.png`` -- the unseen-60 deg comparison with the seed spread.

Run with the ``ocean14`` conda environment from anywhere:

    python reports/make_report_figures.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent / "design" / "validation"

# House palette (Okabe-Ito subset), validated for CVD safety in the adjacent
# stacking order of the ladder (worst model on top): protan/deutan dE >= 11.4,
# normal-vision floor 18.4. #E69F00 and #56B4E9 sit below 3:1 contrast on
# white, so every series carries a direct label -- identity is never
# color-alone. One color per model, identical across figures.
COLOR = {
    "standard Gordon": "#E69F00",
    "ZTT backbone": "#009E73",
    "hybrid, linear": "#56B4E9",
    "O25 form, refit on L23": "#D55E00",
    "hybrid, MLP": "#0072B2",
}
INK, INK_2 = "#333333", "#666666"

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
    """Read the per-wavelength rRMS ladder (held-out scenes) for every model.

    Returns
    -------
    wave : np.ndarray
        Wavelength grid [nm].
    ladder : dict
        ``{model name: rRMS % per wavelength}``.
    """
    with open(VALIDATION / "rrms_per_wavelength.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    wave = np.array([float(r["wavelength_nm"]) for r in rows])
    models = [k for k in rows[0] if k != "wavelength_nm"]
    ladder = {m: np.array([float(r[m]) for r in rows]) for m in models}
    return wave, ladder


def fig_architecture():
    """The hybrid forward model as a dataflow schematic."""
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 60)

    def box(x, y, w, h, text, edge=INK_2, face="white", fs=9, lw=1.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face,
                                   edgecolor=edge, linewidth=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)

    def arrow(x0, y0, x1, y1, color=INK_2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3))

    phys, learn = COLOR["ZTT backbone"], COLOR["hybrid, MLP"]

    # Inputs.
    box(1, 38, 21, 16,
        "IOPs  $a$, $bb_w$, $bb_p$\nPhaseParams  $B_p$\nGeometry "
        r"$\theta_s,\theta_v,\Delta\phi$" + "\nwavelength  $\\lambda$")

    # Physics path.
    box(28, 42, 31, 12,
        "ZTT analytic backbone\nexplicit backward VSF\n"
        r"(TT2017 $\mu_\infty$ stand-in)", edge=phys, lw=1.8)
    box(64, 44, 12, 8, r"$rrs_{\rm ZTT}$", edge=phys, lw=1.4)
    ax.text(43.5, 55.5, "unfitted physics", color=phys, fontsize=9,
            ha="center", style="italic")

    # Learned path.
    box(28, 19, 31, 12,
        "7 dimensionless features\n$\\log_{10}u$, $\\eta_{bb}$, $B_p$, "
        "$\\lambda$,\n$\\cos\\theta_s$, $\\cos\\theta_v$, $\\cos\\Delta\\phi$",
        edge=learn, lw=1.4)
    box(28, 3, 31, 12,
        "MLP 7$\\to$16$\\to$16$\\to$1 (tanh)\n417 trained parameters\n"
        "$\\delta = 0.5\\,\\tanh(\\cdot)$", edge=learn, lw=1.8)
    ax.text(43.5, 32.5, "learned residual correction", color=learn,
            fontsize=9, ha="center", style="italic")

    # Combine and interface.
    box(64, 18, 14, 12, "$rrs =$\n$rrs_{\\rm ZTT}(1+\\delta)$",
        edge=INK, lw=1.4)
    box(83, 28, 20, 14,
        "air-water interface\n\n$R_{rs} = \\dfrac{0.52\\,rrs}{1 - 1.7\\,rrs}$",
        fs=9)

    arrow(22, 48, 27, 48)
    arrow(22, 40, 27, 27)
    arrow(59, 48, 63, 48)
    arrow(43.5, 19, 43.5, 15.5)
    arrow(59, 9, 66, 17.5)
    arrow(70, 44, 70, 30.5)
    arrow(78, 25, 84, 31)
    arrow(103, 35, 105.5, 35)
    ax.text(105.5, 37.5, "$R_{rs}$", fontsize=11, ha="right")

    ax.text(64, 8,
            "Everything is JAX: end-to-end differentiable\n"
            "(gradients match finite differences to $\\leq 5\\times10^{-9}$).\n"
            "Outside the trained domain: DomainWarning,\n"
            "or fallback to the backbone.",
            fontsize=8.5, color=INK_2, va="top")
    ax.set_title("The hybrid forward model:  "
                 r"$rrs = rrs_{\rm ZTT}\,(1 + \delta_{\rm emulator})$",
                 fontsize=12, color=INK, pad=10)

    fig.tight_layout()
    fig.savefig(HERE / "fig_architecture.png", dpi=200)
    plt.close(fig)


def fig_rrms_ladder():
    """Per-wavelength rRMS on held-out scenes, five models, log axis."""
    wave, ladder = read_ladder()
    # "hybrid, MLP + fallback" duplicates "hybrid, MLP" on L23 (the fallback
    # never fires inside the sanctioned envelope) -- drop it for clarity.
    order = ["standard Gordon", "ZTT backbone", "hybrid, linear",
             "O25 form, refit on L23", "hybrid, MLP"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for name in order:
        emph = name == "hybrid, MLP"
        ax.plot(wave, ladder[name], color=COLOR[name],
                lw=2.6 if emph else 1.8, zorder=3 if emph else 2, label=name)

    # Direct labels at the right edge (relief for the two low-contrast hues,
    # and identity never carried by color alone).
    label = {"standard Gordon": "standard Gordon (1988)",
             "ZTT backbone": "ZTT backbone",
             "hybrid, linear": "hybrid, linear (8 par)",
             "O25 form, refit on L23": "O25 form, refit (12 par)",
             "hybrid, MLP": "hybrid, MLP (417 par)"}
    # The O25 and MLP curves end close together; nudge their labels apart.
    dy = {"O25 form, refit on L23": 5, "hybrid, MLP": -6}
    for name in order:
        ax.annotate(label[name], (wave[-1], ladder[name][-1]),
                    xytext=(6, dy.get(name, 0)), textcoords="offset points",
                    va="center", fontsize=9, color=INK)

    ax.set_yscale("log")
    ax.set_xlim(350, 750)
    ax.set_yticks([0.3, 1, 3, 10], labels=["0.3", "1", "3", "10"])
    ax.grid(True, axis="y", which="major")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("rRMS in $rrs$ space [%]")
    ax.set_title("Forward-model error per wavelength - held-out scenes, "
                 "all three solar zeniths", fontsize=11.5)
    ax.legend(loc="center left", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(HERE / "fig_rrms_ladder.png", dpi=200)
    plt.close(fig)


def fig_unseen_zenith():
    """The unseen-60 deg comparison: analytic models are deterministic dots;
    the MLP hybrid is a seed spread.

    Numbers transcribed from ``design/validation/metrics.md`` (regenerated by
    ``design/py/run_validation.py``; emulators trained on 0/30 deg only,
    seeds 23, 1, 7, 101, 2024).
    """
    mlp_seeds = [4.74, 8.37, 7.75, 5.40, 12.24]
    rows = [  # (label, value or None, model key)
        ("standard Gordon (1988)", 9.01, "standard Gordon"),
        ("ZTT backbone", 8.09, "ZTT backbone"),
        ("hybrid, linear (8 par)", 6.16, "hybrid, linear"),
        ("O25 form, refit on 0°/30°", 4.63, "O25 form, refit on L23"),
        ("hybrid, MLP (417 par)", None, "hybrid, MLP"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for y, (name, value, key) in enumerate(rows):
        c = COLOR[key]
        if value is not None:
            ax.plot([value], [y], "o", ms=9, color=c, zorder=3)
        else:
            lo, hi, med = min(mlp_seeds), max(mlp_seeds), np.median(mlp_seeds)
            ax.plot([lo, hi], [y, y], "-", lw=2.5, color=c, alpha=0.45,
                    solid_capstyle="round", zorder=2)
            ax.plot(mlp_seeds, [y] * len(mlp_seeds), "o", ms=5.5, color="white",
                    markeredgecolor=c, markeredgewidth=1.4, zorder=3)
            ax.plot([med], [y], "o", ms=9, color=c, zorder=4)
            ax.annotate("median of 5 seeds", (med, y), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=INK_2)

    ax.set_yticks(range(len(rows)), labels=[r[0] for r in rows])
    ax.set_xlabel("rRMS at the unseen 60° solar zenith [%]  "
                  "(trained/fitted on 0°/30° only)")
    ax.set_xlim(0, 13)
    ax.grid(True, axis="x")
    ax.invert_yaxis()
    ax.set_title("Unseen 60° zenith: the MLP's answer depends on the seed",
                 fontsize=11.5)

    fig.tight_layout()
    fig.savefig(HERE / "fig_unseen_zenith.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_rrms_ladder()
    fig_unseen_zenith()
    print(f"wrote 3 figures to {HERE}")
