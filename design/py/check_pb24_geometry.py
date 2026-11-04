#!/usr/bin/env python
"""
Measure the M8 task-3 sample definition on the whole PB24 release.

Two measurements, written to ``design/ztt_pb24/geometry_check.md``:

1. **The Fresnel check** (Q9 of ``rt_fit_pb24.md``) on every realisation of the
   hyperspectral cache: ``Rrs(theta_air) / rrs(theta_w)``, ``rrs`` interpolated
   linearly in theta to ``theta_w = asin(sin(theta_air) / 1.34)``, normalised at
   nadir, against the flat-surface Fresnel ratio ``T(theta_air) / T(0)``; every
   ``theta_s``, ``dphi`` and band. Run through :func:`robust.rt.data.pb24.load_batch`
   itself (which checks every realisation it reads) with a one-geometry stride,
   so the batch stays tiny while the check sees all 1300 geometries.
2. **Linear against cubic interpolation** of ``rrs`` and ``Q`` to ``theta_w``
   (:func:`robust.rt.data.pb24.interpolation_error`) on a random sample of
   realisations (seed 23), ``theta_s <= 70``.

Deterministic. ``--n-interp`` sets the sample size (default 250); ``--quick``
runs the Fresnel check on every 50th realisation only.

    python design/py/check_pb24_geometry.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robust.rt.data import pb24 as P  # noqa: E402
from robust.rt.data import pb24_cache as PC  # noqa: E402

OUT = ROOT / "design" / "ztt_pb24" / "geometry_check.md"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--n-interp", type=int, default=250)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    ids = PC.open_cache(args.cache_dir).realisations
    if args.quick:
        ids = ids[::50]

    t0 = time.time()
    batch = P.load_batch(
        realisations=tuple(int(i) for i in ids),
        angles="all",
        geometry_stride=(10, 10, 13),
        extras=(),
        cache_dir=args.cache_dir,
    )
    fresnel = batch.report.fresnel
    t_fresnel = time.time() - t0

    rng = np.random.default_rng(P.SPLIT_SEED)
    sample = np.sort(rng.choice(ids, size=min(args.n_interp, ids.size), replace=False))
    t0 = time.time()
    interp = P.interpolation_error(
        tuple(int(i) for i in sample), cache_dir=args.cache_dir
    )
    t_interp = time.time() - t0

    by_s = ", ".join(
        f"{ts:g}: {m * 100:.2f}%" for ts, m in fresnel.max_abs_by_theta_s.items()
    )
    lines = [
        "# PB24 above-water geometry: the Fresnel check and the interpolation error",
        "",
        "Written by `design/py/check_pb24_geometry.py` (M8 task 3); do not edit.",
        "",
        "## Fresnel check",
        "",
        f"{fresnel.n_realisation} realisations (hyperspectral cache, 451 bands), "
        "every theta_s, dphi and band; deviation = (Rrs/rrs normalised at nadir) / "
        "(T/T0) - 1. Quantiles q0 and q1 exact, the rest to 5e-5. "
        f"{fresnel.n_excluded} values excluded (non-finite ratio). "
        f"Within {P.FRESNEL_TOLERANCE:.0%} for theta_v_air <= "
        f"{P.ANGLE_WINDOW_MAX:g}: **{fresnel.within()}** (worst "
        f"{fresnel.worst() * 100:.2f}%; {fresnel.n_beyond} values beyond, in "
        f"realisations {list(fresnel.realisations_beyond)}).",
        "",
        fresnel.table(),
        "",
        f"max |deviation| over theta_v_air <= 70, per theta_s: {by_s}.",
        "",
        "## Linear against cubic interpolation to theta_w",
        "",
        f"{interp.n_realisation} realisations drawn with seed {P.SPLIT_SEED}, "
        "theta_s <= 70, every dphi and band; error = |linear / cubic - 1|, the "
        "cubic a not-a-knot spline through the in-water nodes 0-80. Quantiles q0 "
        "and q1 exact, the rest to 5e-6.",
        "",
        "`rrs`:",
        "",
        interp.table("rrs"),
        "",
        "`Q`:",
        "",
        interp.table("Q"),
        "",
    ]
    text = "\n".join(lines)
    Path(args.out).write_text(text)
    print(text)
    print(f"Fresnel check {t_fresnel:.1f} s; interpolation {t_interp:.1f} s")
    print(batch.report.summary())
    if not fresnel.within():
        print(
            f"Fresnel check beyond {P.FRESNEL_TOLERANCE:.0%}: {fresnel.n_beyond} "
            f"values in realisations {list(fresnel.realisations_beyond)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
