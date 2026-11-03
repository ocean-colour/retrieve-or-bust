#!/usr/bin/env python
"""
Build, verify and rebuild-check the PB24 consolidation cache (M8 task 2).

Opens each hyperspectral ``SD_hyp_no_R_NNNN.nc`` **once** through
``ocpy.hydrolight.pb24`` and writes one float32 ``.npy`` per field plus
``sidecar.json`` with :func:`robust.rt.data.pb24_cache.build` -- by default
into ``$OS_COLOR/SD/derived/pb24_v1/`` (~36 GB, not committed). Prints the
file sizes and the wall time.

``--check`` runs the task-2 gate through :func:`~robust.rt.data.pb24_cache.verify`:
every table field of every cached realisation, realisations 1, 2500 and 5000
(those present) in full, and a random 2% of every other realisation's 4-D
arrays (fixed seed), all compared bit for bit with fresh per-file reads.

``--rebuild-check`` builds a second copy into a scratch directory beside the
cache (same volume), compares every file byte for byte (SHA-256 and
``filecmp``), sidecar included, reports, and deletes the copy.

Deterministic: the output is byte-identical for any ``--workers``.

    python design/py/build_pb24_cache.py --workers 8                 # full build
    python design/py/build_pb24_cache.py --skip-build --check        # gate only
    python design/py/build_pb24_cache.py --skip-build --rebuild-check
    # the committed test fixture (3 realisations, every 10th band in 4-D),
    # built in a scratch directory and packed into one deterministic zip:
    python design/py/build_pb24_cache.py \\
        --cache-dir /tmp/pb24_cache_small \\
        --realisations 1 993 2500 --band-step 10 \\
        --pack robust/tests/files/pb24_cache_small.zip
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robust.rt.data import pb24_cache as PC  # noqa: E402


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in 64 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 26):
            h.update(chunk)
    return h.hexdigest()


def rebuild_check(cache_dir: Path, realisations, workers: int, bands) -> bool:
    """Build a second copy beside ``cache_dir``, compare every file, delete it."""
    scratch = cache_dir.parent / f".{cache_dir.name}_rebuild_check"
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        summary = PC.build(
            scratch,
            realisations=realisations,
            workers=workers,
            progress=True,
            bands=bands,
        )
        print(f"second build: {summary.wall_s:.1f} s into {scratch}")
        names = sorted(p.name for p in cache_dir.iterdir() if p.is_file())
        other = sorted(p.name for p in scratch.iterdir() if p.is_file())
        ok = names == other
        if not ok:
            print(f"FILE LISTS DIFFER: {names} vs {other}")
        for name in names:
            if name not in other:
                continue
            a, b = cache_dir / name, scratch / name
            ha, hb = sha256(a), sha256(b)
            same = ha == hb and filecmp.cmp(a, b, shallow=False)
            ok &= same
            print(f"  {'identical' if same else 'DIFFERENT'}  {ha[:16]}  {name}")
        print(
            f"rebuild-check: {'byte-identical' if ok else 'FAILED'} "
            f"({len(names)} files)"
        )
        return ok
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"deleted {scratch}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="destination (default $OS_COLOR/SD/derived/pb24_v1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="reader processes (output identical for any value)",
    )
    parser.add_argument(
        "--realisations",
        type=int,
        nargs="+",
        default=None,
        help="realisation ids, for a miniature (default: all)",
    )
    parser.add_argument(
        "--band-step",
        type=int,
        default=None,
        help="keep every Nth band in the 4-D fields (miniature)",
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="use the existing cache"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the cache bit for bit against the netCDFs",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.02,
        help="share of each 4-D array sampled by --check",
    )
    parser.add_argument("--seed", type=int, default=PC.VERIFY_SEED)
    parser.add_argument(
        "--rebuild-check",
        action="store_true",
        help="build a second copy and compare byte for byte",
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=None,
        metavar="ZIP",
        help="after building, pack the cache directory into this one zip file",
    )
    args = parser.parse_args(argv)

    cache_dir = PC.default_cache_dir() if args.cache_dir is None else args.cache_dir
    bands = None
    if args.band_step is not None:
        from ocpy.hydrolight import pb24

        bands = range(0, pb24.grids("hyp")["lambda"].size, args.band_step)

    ok = True
    if not args.skip_build:
        summary = PC.build(
            cache_dir,
            realisations=args.realisations,
            workers=args.workers,
            progress=True,
            bands=bands,
        )
        print(f"built {summary.cache_dir}")
        print(summary.table())

    if args.check:
        cache = PC.open_cache(cache_dir)
        report = PC.verify(
            cache_dir,
            fraction=args.fraction,
            seed=args.seed,
            full=[r for r in PC.GATE_FULL_REALISATIONS if r in set(cache.realisations)],
            workers=args.workers,
            raise_on_mismatch=False,
        )
        print(report.summary())
        ok &= report.ok

    if args.rebuild_check:
        cache = PC.open_cache(cache_dir)
        if args.skip_build:  # rebuild what is there, as it was built
            realisations = cache.realisations
            bands = cache.four_d_bands
        else:
            realisations = args.realisations
        ok &= rebuild_check(cache_dir, realisations, args.workers, bands)

    if args.pack is not None:
        zip_path = PC.pack(cache_dir, args.pack)
        print(f"packed {cache_dir} -> {zip_path} ({zip_path.stat().st_size:,} bytes)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
