#!/usr/bin/env python
"""
Fit the geometry-aware surface transfer from PB24 and ship the table (M5 task 7).

``conventions.A_RRS`` / ``B_RRS`` are Lee (2002)'s *nadir* constants. Measured
against PB24 they are good at nadir and fail progressively off-nadir, because the
true ``A`` tracks the Fresnel transmittance and falls from ~0.53 at nadir to
~0.34 at 70 degrees. This script fits ``A`` and ``B`` at every node of PB24's
angle grid and writes ``robust/rt/files/surface_pb24.npz``.

The fit is **linear in both coefficients** -- ``Rrs = A rrs + B (Rrs rrs)`` -- so
it is one ``lstsq`` per geometry: no seed, no learning rate, no stopping rule, and
therefore bit-reproducible.

Fitted on the **training** side of the realisation split, and reported on the
held-out side, because a table with 2600 free numbers fitted and scored on the
same water bodies would tell us nothing.

    python design/py/fit_surface.py                # fit, report, write
    python design/py/fit_surface.py --quick        # fewer realisations
    python design/py/fit_surface.py --dry-run      # report without writing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust.rt import conventions as C  # noqa: E402
from robust.rt.data import pb24 as P  # noqa: E402

#: Realisations used for the shipped table. 400 x 1300 geometries x 12 bands is
#: ~6 M paired points, ~4800 per grid cell -- ample for two coefficients, and it
#: keeps the fit to a couple of minutes. More would not move the numbers.
N_REALISATION = 400

#: Where the table ships.
OUT = (
    Path(__file__).resolve().parents[2] / "robust" / "rt" / "files" / "surface_pb24.npz"
)


def relative_error(rrs, Rrs, A, B):
    """``|predicted / true - 1|`` for the Lee form with the given coefficients."""
    pred = A * rrs / (1.0 - B * rrs)
    return np.abs(pred / Rrs - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="60 realisations")
    parser.add_argument("--dry-run", action="store_true", help="report, do not write")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.quick and args.out == OUT and not args.dry_run:
        parser.error(
            "--quick writes a table fitted on too few realisations; pass --out or "
            "--dry-run"
        )
    if not os.environ.get("OS_COLOR"):
        print("$OS_COLOR is not set, so PB24 cannot be read.", file=sys.stderr)
        return 2

    n = 60 if args.quick else N_REALISATION
    print(f"loading {n} realisations, all geometries ...")
    batch = P.load_batch(realisations=n, angles="all")
    print(f"  {batch.report.summary()}")

    splits = P.make_splits(batch, kinds=("realisation",))
    train, test = splits.train("realisation"), splits.test("realisation")
    print(f"  {splits.reports['realisation'].summary()}")

    rrs = np.asarray(batch.rrs)
    Rrs = np.asarray(batch.Rrs)
    transfer = C.fit_surface_transfer(
        rrs[train],
        Rrs[train],
        batch.theta_s[train],
        batch.theta_v[train],
        batch.dphi[train],
        provenance=(
            f"PB24 OLCI, {int(np.unique(batch.realisation[train]).size)} training "
            f"realisations, all {batch.report.n_geometry_available} geometries; "
            "one lstsq per grid cell"
        ),
    )
    print(
        f"  table {transfer.shape}, A {transfer.A.min():.4f}-{transfer.A.max():.4f}, "
        f"B {transfer.B.min():.3f}-{transfer.B.max():.3f}"
    )

    # Report on HELD-OUT realisations only.
    from robust.rt.types import Geometry

    geometry = Geometry(
        theta_s=batch.geometry.theta_s[test],
        theta_v=batch.geometry.theta_v[test],
        dphi=batch.geometry.dphi[test],
    )
    A, B = transfer.coefficients(geometry)
    A = np.asarray(A)[:, None]
    B = np.asarray(B)[:, None]

    nadir = relative_error(rrs[test], Rrs[test], C.A_RRS, C.B_RRS)
    fitted = relative_error(rrs[test], Rrs[test], A, B)
    theta_v = batch.theta_v[test]
    window = (batch.theta_s[test] <= 70.0) & (theta_v <= 70.0)

    print("\nheld-out realisations, relative error on Rrs:")
    print(f"{'theta_v':>8s} {'nadir const':>12s} {'fitted':>10s} {'gain':>7s}")
    for v in np.unique(theta_v):
        sel = theta_v == v
        a = float(np.median(nadir[sel]))
        b = float(np.median(fitted[sel]))
        print(f"{v:8.2f} {a * 100:11.2f}% {b * 100:9.2f}% {a / b:6.1f}x")
    a = float(np.median(nadir[window]))
    b = float(np.median(fitted[window]))
    print(f"{'window':>8s} {a * 100:11.2f}% {b * 100:9.2f}% {a / b:6.1f}x")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    C.save_transfer(args.out, transfer)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
