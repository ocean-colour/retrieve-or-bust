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
import os
import sys
import tempfile
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

#: Names for the two fits, so the one that gets shipped is selected by name rather
#: than by whatever the training loop last left in scope (PR #11 review).
BASELINE = "linear baseline"
SHIPPED = "MLP"

#: The configuration that actually gets shipped. Named so the architecture guard in
#: :func:`main` can check it *before* training rather than after.
SHIPPED_CONFIG = E.EmulatorConfig()


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

    # Check the architecture up front rather than after two minutes of fitting -- and
    # before the --dry-run exit, so a maintainer who breaks this finds out from a dry
    # run instead of from a shipped file. SHIPPED_CONFIG is what will be trained, so
    # the guard needs nothing that training produces.
    expected = E.EmulatorConfig().hidden
    if SHIPPED_CONFIG.hidden != expected:
        raise SystemExit(
            f"refusing to run: the config selected for shipping has "
            f"hidden={SHIPPED_CONFIG.hidden}, but the package's default architecture "
            f"is {expected}. The weights file is loaded by emulator.load_default() "
            "and used by forward(mode='hybrid'), so a different architecture here "
            "would silently become the shipped model"
        )

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
    # it second would invite reading the MLP's gain as larger than it is. Results are
    # collected into a dict keyed by name -- SHIPPED, below, then names what is
    # written rather than inheriting whatever the loop left in scope.
    trained = {}
    for label, config in ((BASELINE, E.LINEAR_CONFIG), (SHIPPED, SHIPPED_CONFIG)):
        t0 = time.perf_counter()
        emulator, history = E.fit_l23(batch, splits, config=config, rrs_ztt=rrs_ztt)
        elapsed = time.perf_counter() - t0
        delta = emulator.relative_delta(
            batch.iops, batch.phase_params, batch.geometry, batch.wave
        )
        hybrid = rrs_ztt * (1.0 + delta)
        n_par = sum(np.asarray(p).size for p in _leaves(emulator.params))
        trained[label] = (emulator, delta)
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

    emulator, delta = trained[SHIPPED]
    return write_weights(emulator, delta, batch, args.out)


def write_weights(emulator, delta, batch, out: Path) -> int:
    """Validate the weights **before** they can replace a known-good file.

    The round-trip check below is the only thing standing between a broken
    serialisation and a silently wrong shipped model, so it has to run on a file
    that is not yet the destination. Writing first and checking afterwards -- the
    original shape of this function, caught in review of PR #11 -- meant a failed
    check reported an error with the good weights already overwritten. Now the
    candidate is written beside the destination, verified, and only then moved into
    place with :func:`os.replace`, which is atomic: readers see either the old file
    or the new one, never a half-written one.

    Parameters
    ----------
    emulator : robust.rt.emulator.Emulator
        The emulator to ship.
    delta : Array
        Its relative correction on ``batch``, computed before saving; the reloaded
        copy has to reproduce this.
    batch : robust.rt.data.l23.L23Batch
        The batch to compare on.
    out : pathlib.Path
        Destination.

    Returns
    -------
    int
        Process exit status: 0 if the round-trip reproduced the correction.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=out.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        E.save(emulator, tmp)
        reloaded = E.load(tmp)
        same = np.allclose(
            np.asarray(
                reloaded.relative_delta(
                    batch.iops, batch.phase_params, batch.geometry, batch.wave
                )
            ),
            np.asarray(delta),
        )
        if not same:
            print(
                f"round-trip FAILED: the reloaded emulator does not reproduce the "
                f"correction. {out} left untouched; candidate discarded"
            )
            return 1
        # mkstemp creates 0600; a committed artefact must be readable by everyone
        # who can read the repo, so restore the permissions a plain open() would
        # have given it under the process umask.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, out)
    finally:
        # Covers all three exits: after a successful replace the candidate is gone
        # already, and on a failed check or an exception it must not be left behind.
        tmp.unlink(missing_ok=True)
    print(f"\nwrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    print("round-trip reproduces the correction: True")
    return 0


def _leaves(tree):
    """Parameter leaves, without importing jax at module scope here."""
    import jax

    return jax.tree_util.tree_leaves(tree)


if __name__ == "__main__":
    raise SystemExit(main())
