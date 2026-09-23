"""
M2 + M4 -- select batch 0's water bodies and write the delivery for the operator.

Batch 0 is the pilot of spec Section 4: **172 runs, and nothing else is
commissioned until it passes**. Its purpose is to answer three questions that
cannot be answered after the compute is spent -- do our decks reproduce L23, what
is HydroLight's own numerical error here, and does the delivery format work.

**Selection is reproducible, not hand-picked.** Every water body is chosen by a
stated rule from a stated quantile of a stated quantity, so the same 21 bodies
come back on any machine:

* **P1, the L23 reproduction test (10 bodies, 90 runs).** Ten L23 scenes at
  evenly spaced quantiles of ``a_ph(440)``, so the test spans the trophic range
  rather than clustering where the ensemble is dense. These are the bodies whose
  delivered ``Rrs`` is compared against L23's published ``Rrs`` -- if they match,
  every convention is validated at once (sky model, wind, surface, band
  structure, Raman and fluorescence constants, and the output convention).
* **P2, the R0 convergence sweep (5 bodies, 70 runs).** Five L23 scenes at
  evenly spaced quantiles of ``bb(440)/a(440)``, which is the axis the solver's
  discretisation should bite hardest along, run at two zeniths under seven solver
  configurations. This is the only measurement anywhere of the truth's own error
  bar, against which the project's 0.30 % and 0.34 % gates are quoted.
* **P3, the grid corners (6 bodies, 12 runs).** The extremes of batch A's
  *realizable* node set -- minimum and maximum ``bb/a`` at the extreme
  ``eta_bb``, plus the two remaining box corners that survive the physical mask.
  These exercise the user-supplied-IOP path at its limits and the 80 degree
  grazing shell.

Run
---
``python design/py/make_hydrolight_batch0.py --out <dir>``
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "robust" / "tests" / "files"

#: The seven R0 solver configurations: a baseline plus each knob moved in both
#: directions. The operator chooses the actual settings; what matters is that
#: each knob moves both ways from whatever the campaign's baseline will be.
R0_CONFIGS = (
    "baseline",
    "quads_coarser",
    "quads_finer",
    "depth_grid_coarser",
    "depth_grid_finer",
    "inelastic_iterations_fewer",
    "inelastic_iterations_more",
)


def _quantile_picks(values, n):
    """Indices at ``n`` evenly spaced quantiles of ``values`` -- deterministic."""
    order = np.argsort(values, kind="stable")
    q = np.linspace(0, len(order) - 1, n).round().astype(int)
    return order[q]


def select(reader):
    """Batch 0's 21 water bodies, by the rules in the module docstring."""
    from robust.rt.hydrolight import deck, grid, l23_recon

    raw = reader()
    wave = raw["wave"]
    i440 = int(np.argmin(np.abs(wave - 440.0)))
    aph440 = raw["aph"][:, i440]
    ratio440 = raw["bb"][:, i440] / raw["a"][:, i440]

    p1_idx = _quantile_picks(aph440, 10)
    p2_idx = _quantile_picks(ratio440, 5)

    p1 = l23_recon.reconstruct(
        reader, indices=p1_idx, scenarios=("S1", "S2", "S4"), block="pilot-P1"
    )
    for wb, idx in zip(p1, p1_idx, strict=True):
        wb.theta_s = (0, 30, 60)
        wb.provenance["selection"] = {
            "rule": "evenly spaced quantiles of a_ph(440)",
            "a_ph_440": float(aph440[idx]),
        }

    p2 = l23_recon.reconstruct(
        reader, indices=p2_idx, scenarios=("S4",), block="pilot-P2"
    )
    for wb, idx in zip(p2, p2_idx, strict=True):
        wb.theta_s = (0, 60)
        wb.configs = R0_CONFIGS
        wb.provenance["selection"] = {
            "rule": "evenly spaced quantiles of bb(440)/a(440)",
            "bb_over_a_440": float(ratio440[idx]),
        }

    bodies_a = grid.grid_bodies()
    bb_a = np.array([b.provenance["target_bb_over_a"] for b in bodies_a])
    eta = np.array([b.provenance["target_eta_bb"] for b in bodies_a])
    # Corners of the realizable region: extremes of each coordinate, and the two
    # surviving diagonal corners.
    want = [
        (bb_a.min(), eta.min()),
        (bb_a.min(), eta.max()),
        (bb_a.max(), eta.min()),
        (bb_a.max(), eta.max()),
        (np.median(np.unique(bb_a)), eta.min()),
        (np.median(np.unique(bb_a)), eta.max()),
    ]
    p3 = []
    seen = set()
    for tx, ty in want:
        d = (np.log(bb_a / tx)) ** 2 + (np.log(eta / ty)) ** 2
        for k in np.argsort(d):
            key = (bb_a[k], eta[k])
            if key in seen:
                continue
            seen.add(key)
            wb = bodies_a[int(k)]
            wb.wbid = f"pilot-P3-{len(p3):02d}"
            wb.block = "pilot-P3"
            wb.theta_s = (0, 80)
            wb.provenance["selection"] = {
                "rule": "corner of the realizable (bb/a, eta_bb) region",
                "target_bb_over_a": float(bb_a[k]),
                "target_eta_bb": float(eta[k]),
            }
            p3.append(wb)
            break
    assert len(p3) == 6, len(p3)
    _ = deck  # imported for its side-effect-free constants in the docstring
    return p1, p2, p3


README = """\
{release} {version} -- batch 0 (the pilot)

{n_bodies} water bodies, {n_runs} runs. Nothing else in the campaign is
commissioned until this batch passes; the full specification is
design/hydrolight_runs.md.

WHAT IS IN HERE
  manifest.json      every water body, every convention, and the ten-item
                     metadata contract with its pending entries named
  ir/<wbid>.json     one run description per water body (no HydroLight syntax)
  ir/<wbid>.npz      a, b, bb, b_p on the 85-band grid, float64
  vsf/<name>.txt     tabulated beta_tilde(psi), {n_psi} angles, normalised
  provisional/       a readable description of each run -- NOT a deck

WHY THERE ARE NO DECKS YET
  We do not know your HydroLight version's deck syntax. That is question 7
  below. Everything physical is in ir/ and vsf/ and does not depend on the
  syntax; the adapter that renders a real deck is a short piece of code we will
  write the moment you send one example deck from your working setup.

CONVENTIONS (spec Section 2)
  wavelengths     330-750 nm, 5 nm bands, {n_wave} bands. The 350-750 subset is
                  bit-identical to the L23 grid, which is what makes the
                  reproduction test below a like-for-like comparison.
  solar zenith    as listed per water body; the campaign grid is 0-80 deg in
                  10 deg steps, with 0-70 the sanctioned window and 80 a
                  deliberate extrapolation shell.
  view geometry   the standard quad layout -- no custom quad file. Please
                  output the FULL upwelling radiance distribution.
  scenarios       named by switch, not by L23's X label:
                    S1  Raman off, Chl-fl off, CDOM-fl off   (= L23 X=1)
                    S2  Raman ON,  Chl-fl off, CDOM-fl off   (= L23 X=2)
                    S4  Raman ON,  Chl-fl ON,  CDOM-fl off   (= L23 X=4)
                    S5  Raman ON,  Chl-fl ON,  CDOM-fl ON    (new)
  water column    homogeneous, optically deep, no bottom.
  sky             clear, {sky_model}, wind {wind} m/s.
  precision       float64 on all radiometric fields, please. Single precision
                  underflows rrs to exactly zero at grazing geometries and our
                  metric divides by it.
  outputs         Rrs (above water) AND rrs (just below), Lu(0-) at all quads,
                  Ed(0+), Ed(0-), Eu(0-), K_d, K_u, K_Lu, mu_d, mu_u, mu_tot,
                  and the IOPs as you received them (so we can verify the deck
                  we wrote is the deck that ran).
                  For the P3 bodies only, also K_d(z), K_u(z), K_Lu(z),
                  mu_bar(z) to at least 25 optical depths, and the asymptotic
                  radiance distribution and K_inf if your build reports them.

WHAT THE THREE GROUPS ARE FOR
  P1 ({n_p1} bodies, {r_p1} runs) -- these are L23 water bodies rebuilt from
     L23's own published IOPs. We compare your Rrs against L23's published Rrs
     for the same scenes. If they agree, every convention above is validated at
     once. If they do not, we learn which one is wrong while it still costs 90
     runs to find out.
  P2 ({n_p2} bodies, {r_p2} runs) -- the same run repeated under seven solver
     configurations: a baseline, plus quad resolution, depth-grid resolution and
     inelastic-source iteration count each moved coarser and finer. Please
     choose the actual settings. This measures HydroLight's own numerical error,
     which nobody has ever recorded for this project, and against which our
     accuracy claims are quoted.
  P3 ({n_p3} bodies, {r_p3} runs) -- the corners of our IOP grid, to exercise
     the user-supplied-IOP path at its limits and the 80 degree grazing shell.

TEN QUESTIONS (spec Section 10), in order of how much each answer moves the
specification:
{questions}
"""

QUESTIONS = """\
   1. Does ONE run return the full radiance distribution over all view quads,
      so that only solar zenith and the inelastic switches multiply the run
      count? The entire campaign sizing depends on this.
   2. Does your build report the asymptotic radiance distribution and K_inf for
      homogeneous, optically deep water -- and under which switch?
   3. Exact version and build string, and any local modifications.
   4. Your Raman and fluorescence data files, verbatim: the Raman coefficient
      and redistribution function, the chlorophyll fluorescence efficiency and
      emission shape, and specifically WHICH published Hawes variant the CDOM
      fluorescence option uses, with its constants. This is the one part of the
      metadata contract we cannot satisfy from our own decks.
   5. Will you accept user-supplied IOP files -- tabulated a(lambda), b(lambda)
      and a discretised beta_tilde(psi) per water body -- and in what format?
   6. What does a run cost in wall-clock, for an 85-band elastic run and for an
      inelastic one, and how many run in parallel? This turns ~157,000 runs
      into a date.
   7. One example deck and its outputs from your working setup, as the template
      we generate the rest from.
   8. Can the band set start at 330 nm, and does your Raman implementation
      handle excitation at the band-set floor by clipping or by extrapolation?
   9. Which sky model, and is a fixed 5 m/s wind acceptable?
  10. Disk and transfer -- what can you produce, and how do we take delivery?
"""


def main(argv=None):
    """Select batch 0 and write the delivery tree."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "build" / "hydrolight" / "batch0"))
    ap.add_argument("--release", action="store_true", help="use the full L23 release")
    ap.add_argument(
        "--created", default="2026-09-23", help="manifest date; fixed for determinism"
    )
    ap.add_argument("--generator-version", default="spec-prompt-2")
    args = ap.parse_args(argv)

    from robust.rt.hydrolight import deck, l23_recon, manifest, vsf

    reader = (
        l23_recon.release_reader()
        if args.release
        else l23_recon.fixture_reader(
            FIXTURES / "l23_small.npz", FIXTURES / "l23_inelastic_fixture.npz"
        )
    )
    p1, p2, p3 = select(reader)
    bodies = p1 + p2 + p3

    out = pathlib.Path(args.out)
    (out / "ir").mkdir(parents=True, exist_ok=True)
    (out / "vsf").mkdir(parents=True, exist_ok=True)
    (out / "provisional").mkdir(parents=True, exist_ok=True)

    files = []
    for wb in bodies:
        files += deck.write_water_body(out / "ir", wb)
        for theta in wb.theta_s:
            for scenario in wb.scenarios:
                path = out / "provisional" / f"{wb.wbid}_{scenario}_sza{theta:02d}.txt"
                path.write_text(deck.render_provisional_deck(wb, theta, scenario))
                files.append(path)

    used = {wb.vsf_design for wb in bodies}
    designs = [d for d in vsf.designs() if d.name in used]
    for d in designs:
        if not d.tabulated:
            continue
        path = out / "vsf" / f"{d.name}.txt"
        rows = "\n".join(
            f"{psi:12.6f} {beta:20.12e}"
            for psi, beta in zip(vsf.PSI_GRID, d.beta_tilde, strict=True)
        )
        path.write_text(
            f"# beta_tilde(psi), sr^-1, normalised on this grid\n"
            f"# design {d.name}  kind {d.kind}  target B_p {d.target_bp}\n"
            f"# delivered B_p (quadrature) {d.realised_bp():.6f}\n"
            f"# psi_deg  beta_tilde\n{rows}\n"
        )
        files.append(path)

    man = manifest.build(
        "pilot",
        bodies,
        files,
        generator_version=args.generator_version,
        designs=designs,
        created=args.created,
    )
    man["pilot_groups"] = {
        "P1": {"bodies": len(p1), "runs": sum(b.n_runs() for b in p1)},
        "P2": {
            "bodies": len(p2),
            "runs": sum(b.n_runs() for b in p2),
            "configs": list(R0_CONFIGS),
        },
        "P3": {"bodies": len(p3), "runs": sum(b.n_runs() for b in p3)},
    }
    manifest.write(out / "manifest.json", man)

    (out / "README.txt").write_text(
        README.format(
            release=manifest.RELEASE,
            version=manifest.VERSION,
            n_bodies=len(bodies),
            n_runs=sum(b.n_runs() for b in bodies),
            n_psi=vsf.PSI_GRID.size,
            n_wave=deck.WAVE_HL.size,
            sky_model=deck.SKY["model"],
            wind=deck.SKY["wind_speed_m_s"],
            n_p1=len(p1),
            r_p1=sum(b.n_runs() for b in p1),
            n_p2=len(p2),
            r_p2=sum(b.n_runs() for b in p2),
            n_p3=len(p3),
            r_p3=sum(b.n_runs() for b in p3),
            questions=QUESTIONS,
        )
    )

    print(f"batch 0 written to {out}")
    for label, group in (("P1", p1), ("P2", p2), ("P3", p3)):
        print(
            f"  {label}: {len(group):2d} bodies, "
            f"{sum(b.n_runs() for b in group):3d} runs"
        )
    print(f"  total: {len(bodies)} bodies, {sum(b.n_runs() for b in bodies)} runs")
    print(f"  VSF tables: {sum(d.tabulated for d in designs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
