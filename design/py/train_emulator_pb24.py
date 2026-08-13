#!/usr/bin/env python
"""
Train the residual emulator on PB24 and ship the weights — M5 task 11.

Adds a model, it does not replace one (Q13): `robust/rt/files/emulator_pb24.npz`
lands beside the L23 weights and ``emulator.load_default()`` keeps returning the
L23 model, so every M4 number stays reproducible from a clean checkout.

Four decisions from the Q&A are wired in here rather than left to a caller:

- **Q16 — train on a subsample.** All realisations, subsampled *geometries*.
  The factor is not assumed: the script trains at **two** strides and reports
  both, so the choice is evidenced.
- **Q14 — the sanctioned window** is 0–70 degrees in both zeniths, carried on the
  emulator itself (M5 task 10) rather than in a module constant.
- **Q17 — restrict, and report.** Training excludes samples whose analytic
  backbone is non-physical, because ``rrs_ZTT * (1 + delta)`` with
  ``|delta| <= 0.5`` cannot repair a negative backbone. The excluded share is
  printed with every number.
- **Q15 — the gate** is O25 refit on PB24, on the realisation split *and* the
  held-out-``B_p`` split.

    python design/py/train_emulator_pb24.py            # fit, report, write
    python design/py/train_emulator_pb24.py --quick    # short fit, no write
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust.rt import baselines as B  # noqa: E402
from robust.rt import conventions as C  # noqa: E402
from robust.rt import emulator as E  # noqa: E402
from robust.rt import validation as V  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import pb24 as P  # noqa: E402

#: Where the weights ship.
OUT = (
    Path(__file__).resolve().parents[2]
    / "robust"
    / "rt"
    / "files"
    / "emulator_pb24.npz"
)

#: All realisations carry the phase-function variation M5 is about, so the
#: subsample is in *geometry* (Q16). Two strides, compared rather than assumed.
#:
#: **Per axis, not flat.** A flat stride over the flattened geometry list does not
#: preserve the grid's product structure, so the O25 table this is gated against
#: cannot be fitted on the same batch -- it finds most of its cells empty. Striding
#: each angle axis keeps the full product of what survives.
STRIDES = ((1, 2, 2), (1, 3, 4))

#: Correction bounds to train at. ``0.5`` is the shipped value, chosen at M3 to
#: stop the emulator swamping the physics. On PB24 it is the **binding
#: constraint**: the *oracle* -- the best any emulator could do, using the truth
#: to choose delta -- scores 69.5% at 0.5 and 0.00% at 1.0 on samples with a
#: usable backbone. So a FAIL at 0.5 says nothing about the network, and both are
#: trained here to make that visible rather than arguable (Q17 option 4).
DELTA_MAXES = (0.5, 1.0)

#: Realisations. The full 5000 is ~2 GB resident before training touches it
#: (record §7.7); 800 with a coprime geometry stride is ~90k samples per fit.
N_REALISATION = 800

#: The architecture the shipped weights must have, checked on the emulator being
#: serialised rather than on a config constant (PR #12's finding).
SHIPPED_CONFIG = E.EmulatorConfig()


def check_architecture(config: E.EmulatorConfig, what: str) -> None:
    """Refuse to ship anything but the validated architecture."""
    if (
        config.hidden != SHIPPED_CONFIG.hidden
        or config.delta_max != SHIPPED_CONFIG.delta_max
    ):
        raise SystemExit(
            f"{what}: refusing to ship hidden={config.hidden}, "
            f"delta_max={config.delta_max}; the validated architecture is "
            f"hidden={SHIPPED_CONFIG.hidden}, delta_max={SHIPPED_CONFIG.delta_max}"
        )


def score(pred, truth, mask) -> float:
    """rRMS on a mask, in percent."""
    return float(V.rrms(truth[mask], pred[mask]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realisations", type=int, default=N_REALISATION)
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 1, 7, 101, 2024])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--ship-anyway", action="store_true", help="write weights despite a failed gate"
    )
    args = parser.parse_args()

    if args.quick and args.out == OUT and not args.dry_run:
        parser.error("--quick would ship short-fit weights; pass --dry-run or --out")
    if not os.environ.get("OS_COLOR"):
        raise SystemExit("$OS_COLOR is not set, so PB24 cannot be read.")

    steps = 300 if args.quick else SHIPPED_CONFIG.steps
    n = 80 if args.quick else args.realisations
    strides = STRIDES[:1] if args.quick else STRIDES

    runs = [(strides[0], d) for d in DELTA_MAXES]
    if len(strides) > 1:
        runs.append((strides[1], DELTA_MAXES[0]))  # Q16: a second stride, compared

    results = {}
    loaded = {}
    for stride, delta_max in runs:
        print(
            f"\n=== stride {stride} (theta_s, theta_v, dphi), delta_max {delta_max} ==="
        )
        if stride not in loaded:
            batch = P.load_batch(
                realisations=n, angles="window", geometry_stride=stride
            )
            splits = P.make_splits(batch, kinds=("realisation", "bp_band"))
            rrs_ztt = Z.rrs_ZTT(
                batch.iops, batch.phase_params, batch.geometry, batch.wave
            )
            loaded[stride] = (batch, splits, rrs_ztt)
            print(f"  {batch.report.summary()}")
        batch, splits, rrs_ztt = loaded[stride]
        transfer = C.default_transfer()
        truth = batch.rrs
        usable_all = E.backbone_is_usable(rrs_ztt)

        # The ceiling this functional form cannot pass, whatever the network
        # learns: delta chosen with knowledge of the truth, then clipped.
        with np.errstate(divide="ignore", invalid="ignore"):
            best_delta = np.clip(
                np.asarray(truth) / np.asarray(rrs_ztt) - 1.0, -delta_max, delta_max
            )
        oracle = np.asarray(rrs_ztt) * (1.0 + best_delta)

        per_seed = {}
        for seed in args.seeds:
            row = {}
            for kind in ("realisation", "bp_band"):
                # **Refit per split.** Training once on the realisation split and
                # scoring the bp_band test set is a leak, not a shortcut: 75% of
                # the bp_band held-out realisations are in the realisation split's
                # training set, so that number would have been training error
                # compared against an honestly held-out rival. `fit_pb24` takes
                # `kind` precisely so this is a one-word choice.
                config = E.EmulatorConfig(
                    steps=steps, seed=seed, delta_max=delta_max
                )
                emulator, _, coverage = E.fit_pb24(
                    batch, splits, kind, config=config, rrs_ztt=rrs_ztt
                )
                pred = rrs_ztt * (
                    1.0
                    + emulator.relative_delta(
                        batch.iops, batch.phase_params, batch.geometry, batch.wave
                    )
                )
                # **Score everything on the full test set.** An earlier version
                # scored both models only where the backbone is physical, which
                # is indefensible twice over: it drops the samples only the
                # *hybrid's* form cannot represent, and -- measured -- those
                # samples cluster at large theta_v and dphi near 180, so the
                # restriction quietly narrows the geometry range as well (it
                # empties 17 of 224 O25 grid cells outright). The hybrid trains
                # on what its form can represent, which is necessary and
                # disclosed; it is *judged* on everything, which is the only
                # comparison that means anything.
                train = splits.train(kind)
                test = splits.test(kind)
                dropped = test & ~usable_all

                # O25 fitted directly in `rrs`, the space everything is scored
                # in. Fitting it in `Rrs` and converting charges it with the
                # surface transfer's own residual (a median ~1.8%, record §7.10)
                # that the emulator never pays, because the emulator trains on
                # PB24's tabulated `rrs`. Both are reported.
                o25_rrs = B.fit_o25_table(
                    batch.iops, truth, batch.geometry, train=train
                )
                o25_Rrs = B.fit_o25_table(
                    batch.iops, batch.Rrs, batch.geometry, train=train
                )
                args_m = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
                pred_o25_rrs = B.Rrs_o25(*args_m, coeffs=o25_rrs)
                pred_o25_Rrs = B.rrs_o25(
                    *args_m, coeffs=o25_Rrs, transfer=transfer
                )

                row[kind] = {
                    "hybrid": score(pred, truth, test),
                    "o25": score(pred_o25_rrs, truth, test),
                    "o25_via_Rrs": score(pred_o25_Rrs, truth, test),
                    "ztt": score(rrs_ztt, truth, test),
                    "oracle": score(oracle, truth, test),
                    "hybrid_usable": score(pred, truth, test & usable_all),
                    "o25_usable": score(pred_o25_rrs, truth, test & usable_all),
                    "o25_on_dropped": (
                        score(pred_o25_rrs, truth, dropped)
                        if dropped.any()
                        else float("nan")
                    ),
                    "n_test": int(test.sum()),
                    "n_dropped": int(dropped.sum()),
                }
                if seed == args.seeds[0] and kind == "realisation":
                    results[(stride, delta_max)] = {
                        "emulator": emulator,
                        "coverage": coverage,
                    }
            per_seed[seed] = row
            print(
                f"  seed {seed:5d}: "
                + "   |   ".join(
                    f"{k} hybrid {row[k]['hybrid']:7.2f}% vs O25 {row[k]['o25']:6.2f}%"
                    for k in ("realisation", "bp_band")
                )
            )

        results[(stride, delta_max)]["per_seed"] = per_seed
        cov = results[(stride, delta_max)]["coverage"]
        print(
            f"  coverage: trained on {cov['n_train']} of {cov['n_total']} samples; "
            f"excluded {cov['n_excluded_backbone']} for a non-physical backbone "
            f"({100 * cov['n_excluded_backbone'] / cov['n_total']:.1f}%)"
        )

    print("\n=== summary: median over seeds, held-out ===")
    header = (
        f"{'stride':>12s} {'dmax':>5s} {'split':>12s} {'hybrid':>9s} "
        f"{'oracle':>9s} {'O25':>9s} {'gate':>6s}"
    )
    print(header)
    for (stride, delta_max), payload in results.items():
        for kind in ("realisation", "bp_band"):
            rows = [payload["per_seed"][s][kind] for s in args.seeds]
            hyb = np.median([r["hybrid"] for r in rows])
            o25 = np.median([r["o25"] for r in rows])
            orc = np.median([r["oracle"] for r in rows])
            spread = (
                min(r["hybrid"] for r in rows),
                max(r["hybrid"] for r in rows),
            )
            print(
                f"{str(stride):>12s} {delta_max:5.1f} {kind:>12s} {hyb:8.2f}% "
                f"{orc:8.2f}% {o25:8.2f}% "
                f"{'PASS' if hyb < o25 else 'FAIL':>6s}   "
                f"seeds {spread[0]:.2f}-{spread[1]:.2f}%"
            )
    first = results[next(iter(results))]["per_seed"][args.seeds[0]]["realisation"]
    print(
        f"\nfairness checks (realisation, seed {args.seeds[0]}):"
        f"\n  O25 fitted in rrs {first['o25']:.2f}%  vs via Rrs+transfer "
        f"{first['o25_via_Rrs']:.2f}%  (the transfer's residual is a handicap the "
        "hybrid never pays)"
        f"\n  restricted to a physical backbone: hybrid "
        f"{first['hybrid_usable']:.2f}%  O25 {first['o25_usable']:.2f}%"
        f"\n  on the {first['n_dropped']} samples that restriction would drop: "
        f"O25 {first['o25_on_dropped']:.2f}%  -- the rival handles them, so "
        "excluding them would flatter us"
    )

    # Refuse to ship a model that fails its gate. A committed `.npz` is a claim
    # that the thing is usable, and `load()` gives no hint of provenance -- so the
    # refusal belongs here, where the numbers are, rather than in a reviewer's
    # memory.
    passed = all(
        np.median([payload["per_seed"][s][kind]["hybrid"] for s in args.seeds])
        < np.median([payload["per_seed"][s][kind]["o25"] for s in args.seeds])
        for payload in results.values()
        for kind in ("realisation", "bp_band")
    )
    if not passed and not args.ship_anyway:
        print(
            "\nGATE FAILED — refusing to ship. The bound on the relative "
            "correction, not the network, is what fails: compare the hybrid "
            "column with the oracle column, which is the best any emulator could "
            "do at that bound. Pass --ship-anyway to override."
        )
        return 1

    chosen = (strides[0], DELTA_MAXES[0])
    emulator = results[chosen]["emulator"]
    check_architecture(emulator.config, "train_emulator_pb24")
    print(f"\nshipping the {chosen} (stride, delta_max) model, seed {args.seeds[0]}")
    print(f"  envelope: {emulator.envelope.describe()}")

    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    E.save(emulator, args.out)
    back = E.load(args.out)
    check_architecture(back.config, "train_emulator_pb24 (written file)")
    if back.envelope != emulator.envelope:
        raise SystemExit("the written envelope does not read back equal")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
