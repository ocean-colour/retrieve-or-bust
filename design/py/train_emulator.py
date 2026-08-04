#!/usr/bin/env python
"""Train the residual emulator on L23 and write the weights the package ships.

The real M3 fit: ~60 s for 3000 full-batch Adam steps on the 7968-sample
``scene_train`` split. It lives here rather than in the test suite for the reason
PAB's MCMC does — CI must not depend on a minute of optimisation, or on the 17 MB
netCDFs. The suite trains toy-size on the committed 50-scene fixture instead, and
exercises *these* weights for the real numbers.

Usage
-----
    python design/py/train_emulator.py            # -> robust/rt/files/emulator_l23.npz
    python design/py/train_emulator.py --dry-run  # train and report, write nothing

Requires ``$OS_COLOR`` (the L23 netCDFs) and the ``ocean14`` environment.

Regenerate after any change to ``emulator.FEATURES``, the architecture, the loss, or
the splits: the shipped weights are only meaningful against the code that made them,
and ``emulator.load`` refuses a file whose feature list no longer matches.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust.rt import baselines as B  # noqa: E402
from robust.rt import conventions as C  # noqa: E402
from robust.rt import emulator as E  # noqa: E402
from robust.rt import validation as V  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import l23 as L  # noqa: E402


def report(truth, pred, mask, label: str) -> float:
    """Print and return rRMS on one mask."""
    value = float(V.rrms(truth[mask], pred[mask]))
    print(f"    {label:34s} {value:6.2f}%")
    return value


def main() -> int:
    """Train, report against every reference, and write the weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="do not write weights")
    parser.add_argument(
        "--out", type=Path, default=E.DEFAULT_WEIGHTS, help="destination .npz"
    )
    args = parser.parse_args()

    batch = L.load_batch()
    splits = L.make_splits(batch)
    truth = C.Rrs_to_rrs(batch.Rrs)
    rrs_ztt = Z.rrs_ZTT(batch.iops, batch.phase_params, batch.geometry, batch.wave)
    rrs_gordon = B.rrs_gordon(batch.iops)
    print(
        f"L23 elastic: {batch.n_sample} samples x {batch.n_wave} lambda; "
        f"{int(splits.scene_train.sum())} train / "
        f"{int(splits.scene_test.sum())} held-out scenes"
    )

    # The linear model first: it is the baseline the MLP has to beat, and reporting
    # it second would invite reading the MLP's gain as larger than it is.
    for label, config in (("linear baseline", E.LINEAR_CONFIG), ("MLP", None)):
        t0 = time.perf_counter()
        emulator, history = E.fit_l23(batch, splits, config=config, rrs_ztt=rrs_ztt)
        elapsed = time.perf_counter() - t0
        delta = emulator.relative_delta(
            batch.iops, batch.phase_params, batch.geometry, batch.wave
        )
        hybrid = rrs_ztt * (1.0 + delta)
        n_par = sum(np.asarray(p).size for p in _leaves(emulator.params))
        print(
            f"\n  {label}: hidden={emulator.config.hidden}, {n_par} params, "
            f"{elapsed:.0f} s, |delta| rms {history.delta_rms[-1]:.2f}%"
        )
        for name, mask in (
            ("train", splits.scene_train),
            ("held-out scenes", splits.scene_test),
            ("held-out scenes @60 deg", splits.scene_test & (batch.zenith == 60)),
        ):
            report(truth, hybrid, mask, f"hybrid, {name}")
        for name, mask in (("train", splits.scene_train),):
            report(truth, rrs_ztt, mask, f"ZTT alone, {name}")
            report(truth, rrs_gordon, mask, f"Gordon, {name}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    E.save(emulator, args.out)
    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out} ({size_kb:.1f} KB)")

    # Round-trip, because a weights file that loads to something different is the
    # one failure mode that would not show up until a much later comparison.
    reloaded = E.load(args.out)
    same = np.allclose(
        np.asarray(
            reloaded.relative_delta(
                batch.iops, batch.phase_params, batch.geometry, batch.wave
            )
        ),
        np.asarray(delta),
    )
    print(f"round-trip reproduces the correction: {same}")
    return 0 if same else 1


def _leaves(tree):
    """Parameter leaves, without importing jax at module scope here."""
    import jax

    return jax.tree_util.tree_leaves(tree)


if __name__ == "__main__":
    raise SystemExit(main())
