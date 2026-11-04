#!/usr/bin/env python
"""
Build the PB24 held-out splits once, over all 5000 realisations (M8 task 3).

Reads only the per-realisation **table** of the hyperspectral cache
(``robust.rt.data.pb24_cache``; no 4-D reads): the realisation ids, the bulk
``B_p = (bbph + bbNAP) / (bph + bNAP)`` averaged over the 451 bands, and the
labels and IOP summaries the confound is reported on (``C``, ``N``, ``Y``,
mean ``a``, mean ``bb_p``). Then :func:`robust.rt.data.pb24.build_splits`:

- ``realisation``: a ``SPLIT_SEED = 23`` permutation's first
  ``TEST_FRACTION = 0.2`` of the ids, held out whole;
- ``bp_band``: the realisations whose mean ``B_p`` lies in its (0.4, 0.6)
  quantile band over all 5000, held out whole;
- ``geometry``: a **rule**, stored as text -- held out ``theta_v_air >= 80`` or
  ``theta_s >= 80`` (above-water angles);

and writes ``design/ztt_pb24/splits.json`` (sorted keys, no timestamp, so a
rebuild is byte-identical) and ``design/ztt_pb24/splits.md`` (counts and the
chlorophyll confound). :func:`robust.rt.data.pb24.make_splits` loads the JSON.

    python design/py/make_pb24_splits.py            # build and write both files
    python design/py/make_pb24_splits.py --check    # rebuild in memory, compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robust.rt.data import pb24 as P  # noqa: E402
from robust.rt.data import pb24_cache as PC  # noqa: E402

OUT_DIR = ROOT / "design" / "ztt_pb24"
MD_FILE = OUT_DIR / "splits.md"


def per_realisation(cache_dir=None) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """Ids, mean ``B_p`` and confound columns from the cache table."""
    cache = PC.open_cache(cache_dir)
    t = {
        name: np.asarray(cache.table(name), dtype=float)
        for name in (
            "C",
            "N",
            "Y",
            "aw",
            "aph",
            "ag",
            "aNAP",
            "bph",
            "bNAP",
            "bbph",
            "bbNAP",
        )
    }
    bb_p = t["bbph"] + t["bbNAP"]
    B_p = bb_p / (t["bph"] + t["bNAP"])
    labels = {
        "C": t["C"],
        "N": t["N"],
        "Y": t["Y"],
        "a_mean": (t["aw"] + t["aph"] + t["ag"] + t["aNAP"]).mean(axis=-1),
        "bb_p_mean": bb_p.mean(axis=-1),
    }
    side = cache.sidecar
    source = (
        f"PB24 hyperspectral cache pb24_v{side['cache_version']} table "
        f"(reader version {side['reader_version']}); B_p_mean = mean over "
        f"{cache.grids['lambda'].size} bands of (bbph + bbNAP) / (bph + bNAP)"
    )
    return cache.realisations, B_p.mean(axis=-1), labels, source


def markdown(content: dict) -> str:
    """The short human summary written beside the JSON."""
    ids = np.asarray(content["realisations"])
    conf = content["confound"]
    ref = conf["random_reference"]
    lines = [
        "# PB24 splits (M8 task 3)",
        "",
        "Written by `design/py/make_pb24_splits.py` with",
        "`design/ztt_pb24/splits.json`; do not edit by hand. Loaded by",
        "`robust.rt.data.pb24.make_splits`.",
        "",
        f"Source: {content['source']}.",
        "",
        "| split | definition | held out | train |",
        "|---|---|---|---|",
    ]
    n_r = len(content["realisation"]["test"])
    n_b = len(content["bp_band"]["test"])
    lines.append(
        f"| realisation | seed {content['seed']}, test fraction "
        f"{content['test_fraction']:g} | {n_r} realisations | {ids.size - n_r} |"
    )
    lo, hi = content["bp_band"]["lo"], content["bp_band"]["hi"]
    q = content["bp_band_quantiles"]
    lines.append(
        f"| bp_band | mean B_p in its ({q[0]:g}, {q[1]:g}) quantile band, "
        f"[{lo:.5f}, {hi:.5f}] | {n_b} realisations | {ids.size - n_b} |"
    )
    lines.append(
        f"| geometry | {content['geometry']['rule']} | 468 of 1300 air "
        "geometries per realisation | 832 (the window) |"
    )
    overlap = len(set(content["realisation"]["test"]) & set(content["bp_band"]["test"]))
    bp = np.asarray(content["B_p_mean"])
    lines += [
        "",
        f"Realisations: {ids.size} ({ids.min()}-{ids.max()}). Mean B_p spans "
        f"{bp.min():.5f}-{bp.max():.5f} ({bp.max() / bp.min():.1f}x). "
        f"{overlap} realisations are held out by both the realisation and the "
        "B_p-band split.",
        "",
        "## The confound",
        "",
        f"corr(log B_p_mean, log C) = **{conf['corr_log_B_p_log_C']:.3f}** over all "
        f"{ids.size} realisations.",
        "",
        "Test/train ratio of medians, against what a random 20% hold-out does "
        f"(median [min, max] over seeds {conf['random_reference_seeds'][0]}-"
        f"{conf['random_reference_seeds'][-1]}):",
        "",
        "| label | realisation | bp_band | random reference |",
        "|---|---|---|---|",
    ]
    for name in conf["bp_band"]:
        med, mn, mx = ref[name]
        lines.append(
            f"| {name} | {conf['realisation'][name]:.3f} | "
            f"{conf['bp_band'][name]:.3f} | {med:.3f} [{mn:.3f}, {mx:.3f}] |"
        )
    lines += [
        "",
        "The B_p band is interior, so its train side is both tails and its",
        "`B_p_mean` ratio sits near 1 by construction; the separation lives in the",
        "band edges. Read every other entry against the random reference.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", default=None, help="hyperspectral cache")
    parser.add_argument("--out", default=str(P.DEFAULT_SPLITS_FILE))
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and compare with the files on disk",
    )
    args = parser.parse_args(argv)

    ids, bp_mean, labels, source = per_realisation(args.cache_dir)
    content = P.build_splits(ids, bp_mean, labels, source=source)
    text = json.dumps(content, indent=1, sort_keys=True) + "\n"
    md = markdown(content)
    out = Path(args.out)
    md_path = out.with_suffix(".md")

    print(md)
    if args.check:
        same_json = out.is_file() and out.read_text() == text
        same_md = md_path.is_file() and md_path.read_text() == md
        print(
            f"check: {out} {'identical' if same_json else 'DIFFERS'}; "
            f"{md_path} {'identical' if same_md else 'DIFFERS'}"
        )
        return 0 if same_json and same_md else 1

    P.write_splits(out, content)
    md_path.write_text(md)
    print(f"wrote {out} ({out.stat().st_size} bytes) and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
