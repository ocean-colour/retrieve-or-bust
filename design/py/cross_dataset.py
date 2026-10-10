#!/usr/bin/env python
"""
Train on PB24, judge on L23 — M5 task 12, the cross-dataset check.

Q13 asked for this: a model trained on one HydroLight campaign and scored on
another, without refitting, is the strongest generalisation evidence this project
can produce. It also decides the promotion rule, written down before any number
existed — ``load_default()`` moves to a PB24 model **only** if that model wins
L23's own held-out split.

Task 11 gives the check a sharper question than it was designed for. The PB24
gate failed structurally: the analytic backbone is non-physical on ~20% of PB24,
and no bounded relative correction can repair it. So an emulator trained there
had two things it could have learned — the residual physics, or how to compensate
for a broken backbone. **L23 separates them**, because on L23 the backbone is
healthy (5.93%). If the PB24-trained correction helps on L23, it learned physics;
if it hurts, it learned the pathology.

Two caveats are printed with every number rather than left to the reader:

- **The grids do not match.** L23 spans 350-750 nm in 81 bands; PB24's OLCI grid
  spans 400-753 in 12. A PB24-trained emulator's ``wave_nm`` domain therefore
  excludes 10 of L23's bands outright, so the overlap is computed and the
  non-overlapping bands are reported separately, never folded in.
- **The model is out of its domain.** It flags essentially every L23 sample, and
  the score is what it does when asked anyway. That is worth measuring and must
  never be quoted as though the model were being used properly.

    python design/py/cross_dataset.py
    python design/py/cross_dataset.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust.rt import conventions as C  # noqa: E402
from robust.rt import emulator as E  # noqa: E402
from robust.rt import validation as V  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import l23 as L  # noqa: E402
from robust.rt.data import pb24 as P  # noqa: E402

#: PB24 realisations to train on.
N_REALISATION = 150

#: Geometry stride, per axis (record §7.14).
STRIDE = (1, 2, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realisations", type=int, default=N_REALISATION)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OS_COLOR"):
        raise SystemExit("$OS_COLOR is not set, so neither dataset can be read.")

    n = 30 if args.quick else args.realisations
    steps = 200 if args.quick else E.EmulatorConfig().steps

    print(f"training on {n} PB24 realisations, stride {STRIDE}, {steps} steps ...")
    pb = P.load_batch(realisations=n, angles="window", geometry_stride=STRIDE)
    splits_pb = P.make_splits(pb, kinds=("realisation",))
    emulator, _, coverage = E.fit_pb24(
        pb,
        splits_pb,
        "realisation",
        config=E.EmulatorConfig(steps=steps, seed=args.seed),
    )
    j = E.FEATURES.index("wave_nm")
    domain = np.asarray(emulator.domain)
    print(
        f"  trained on {coverage['n_train']} samples; "
        f"wave_nm domain {domain[0, j]:.0f}-{domain[1, j]:.0f} nm"
    )

    batch = L.load_batch()
    splits = L.make_splits(batch)
    held = splits.scene_test
    wave = np.asarray(batch.wave)
    truth = np.asarray(C.Rrs_to_rrs(batch.Rrs))
    ztt = np.asarray(Z.rrs_ZTT(batch.iops, batch.phase_params, batch.geometry, wave))

    flagged = np.asarray(
        emulator.out_of_domain_mask(
            batch.iops, batch.phase_params, batch.geometry, wave
        )
    )
    overlap = (wave >= domain[0, j]) & (wave <= domain[1, j])
    if not overlap.any():
        raise SystemExit("the two wavelength grids do not overlap at all")
    print(
        f"\nL23 samples the PB24 model flags out of domain: {100 * flagged.mean():.1f}%"
        f"\nband overlap: {overlap.sum()} of {wave.size} L23 bands "
        f"({wave[overlap].min():.0f}-{wave[overlap].max():.0f} nm); "
        f"{(~overlap).sum()} bands below it are extrapolation"
    )

    delta = np.asarray(
        emulator.relative_delta(batch.iops, batch.phase_params, batch.geometry, wave)
    )
    shipped = E.load_default()
    delta_l23 = np.asarray(
        shipped.relative_delta(batch.iops, batch.phase_params, batch.geometry, wave)
    )

    models = {
        "ZTT backbone": ztt,
        "hybrid, PB24-trained": ztt * (1.0 + delta),
        "hybrid, L23-trained (shipped)": ztt * (1.0 + delta_l23),
    }

    def rrms(pred, bands):
        return float(V.rrms(truth[held][:, bands], pred[held][:, bands]))

    print(
        f"\nL23 held-out scenes, rRMS in rrs:\n"
        f"{'model':32s} {'all 81 bands':>13s} {'overlap':>10s} {'350-395 nm':>12s}"
    )
    for name, pred in models.items():
        print(
            f"{name:32s} {rrms(pred, slice(None)):12.2f}% "
            f"{rrms(pred, overlap):9.2f}% {rrms(pred, ~overlap):11.2f}%"
        )

    print(
        f"\nthe PB24 model's correction on L23: median |delta| "
        f"{100 * np.median(np.abs(delta)):.1f}%, max {100 * np.abs(delta).max():.1f}% "
        f"(bound {100 * emulator.config.delta_max:.0f}%)"
    )

    incumbent = rrms(models["hybrid, L23-trained (shipped)"], slice(None))
    challenger = rrms(models["hybrid, PB24-trained"], slice(None))
    verdict = (
        "PROMOTE the PB24 model"
        if challenger < incumbent
        else "keep the L23 model as load_default()"
    )
    print(
        f"\npromotion rule (Q13): PB24 {challenger:.2f}% vs L23 {incumbent:.2f}% "
        f"on L23's held-out split -> {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
