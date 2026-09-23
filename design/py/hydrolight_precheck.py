"""
M3 -- the in-house pre-check, before Henry runs anything.

Pushes the **reconstructed** L23 water bodies of
:mod:`robust.rt.hydrolight.l23_recon` through the package's existing forward
model and compares against L23's own published ``Rrs``. This does **not**
validate HydroLight: it validates *our reconstruction pipeline*, and it costs no
compute. If the reconstruction is broken we find it here rather than in the
pilot's 90 runs.

Three questions, in increasing strength:

1. **Do the IOPs round-trip?** Reconstructed ``a``, ``bb`` and ``b_p`` against
   the values in the L23 file, on L23's own 81-band grid. This is the one that
   caught the original defect: rebuilding ``a`` from ``a_w + a_ph + a_g +
   a_nap`` let the difference between our pure-water table and L23's reach the
   total, by up to **150 %** in the red. Total ``a`` is now carried verbatim.
2. **Does the forward model agree with itself?** ``forward`` on the
   reconstructed IOPs against ``forward`` on the loader's IOPs. Any difference
   here is the reconstruction, not the model.
3. **Does it still reproduce L23?** rRMS against L23's X=1 ``Rrs``, which should
   land on the package's published number for whichever mode is scored -- not
   better, not worse. A *change* here would mean the reconstruction had altered
   the water.

Run
---
``python design/py/hydrolight_precheck.py``  (uses the committed fixtures; add
``--release`` for all 3320 scenes, which needs ``$OS_COLOR``).
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


def main(argv=None):
    """Run the three checks and print a report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--release",
        action="store_true",
        help="use the full 3320-scene release instead of the committed fixtures",
    )
    ap.add_argument("--zenith", type=int, default=0, help="solar zenith, degrees")
    args = ap.parse_args(argv)

    import jax.numpy as jnp

    from robust.rt import validation
    from robust.rt.data import l23
    from robust.rt.hybrid import forward
    from robust.rt.hydrolight import deck, l23_recon
    from robust.rt.types import Geometry, IOPs, PhaseParams

    if args.release:
        reader = l23_recon.release_reader()
        raw = l23._read_file(l23.ELASTIC_X, args.zenith)
        label = "full release (3320 scenes)"
    else:
        reader = l23_recon.fixture_reader(
            FIXTURES / "l23_small.npz", FIXTURES / "l23_inelastic_fixture.npz"
        )
        raw = l23.npz_reader(FIXTURES / "l23_small.npz")(l23.ELASTIC_X, args.zenith)
        label = "committed 50-scene fixture"

    bodies = l23_recon.reconstruct(reader)
    sub = deck.WAVE_HL >= raw["wave"][0]
    wave = jnp.asarray(raw["wave"])

    a_rec = np.array([b.a[sub] for b in bodies])
    bb_rec = np.array([b.bb[sub] for b in bodies])
    bp_rec = np.array([b.b_p[sub] for b in bodies])

    print(f"M3 pre-check -- {label}, solar zenith {args.zenith} deg")
    print(f"  {len(bodies)} water bodies x {sub.sum()} bands\n")

    print("1. IOP round-trip (reconstructed vs the L23 file)")
    worst = 0.0
    for name, got, want in (
        ("a", a_rec, raw["a"]),
        ("bb", bb_rec, raw["bb"]),
        ("b_p", bp_rec, raw["bnw"]),
    ):
        err = np.abs(got - want) / np.abs(want)
        worst = max(worst, float(err.max()))
        print(
            f"   {name:4s} median {np.median(err):.3e}"
            f"   p99 {np.percentile(err, 99):.3e}   max {err.max():.3e}"
        )
    print(f"   -> worst relative IOP error: {worst:.3e}\n")

    # bb_w from the file itself, as l23.load_batch does, so the comparison is
    # like-for-like rather than against this module's own water table.
    bb_w_file = raw["bb"] - raw["bbnw"]
    geometry = Geometry.nadir(jnp.asarray(float(args.zenith)))

    def run(a, bb, bb_p, B_p):
        iops = IOPs(
            a=jnp.asarray(a), bb_w=jnp.asarray(bb - bb_p), bb_p=jnp.asarray(bb_p)
        )
        return forward(
            iops,
            PhaseParams(B_p=jnp.asarray(B_p)),
            geometry,
            wave,
            check_domain=False,
        )

    rrs_file = run(raw["a"], raw["bb"], raw["bbnw"], raw["bbnw"] / raw["bnw"])
    rrs_rec = run(a_rec, bb_rec, bb_rec - bb_w_file, (bb_rec - bb_w_file) / bp_rec)

    print("2. forward(reconstructed) vs forward(loader IOPs)")
    d = np.abs(np.asarray(rrs_rec) - np.asarray(rrs_file)) / np.abs(rrs_file)
    print(f"   median {np.median(d):.3e}   max {d.max():.3e}\n")

    print("3. rRMS against L23's published X=1 Rrs")
    truth = jnp.asarray(raw["Rrs"])
    for name, pred in (("loader IOPs", rrs_file), ("reconstructed", rrs_rec)):
        print(f"   {name:16s} {float(validation.rrms(truth, pred)):8.4f} %")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
