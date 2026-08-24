"""
Generate the packaged data artifacts of the inelastic effort (coding plan M1).

Part 1 (M1 task 1, this file's ``extract_ed``): the three L23 ``Ed(0+)(lambda)``
spectra -> ``robust/rt/data/ed_l23.npz``, consumed by :mod:`robust.rt.ed`.
Part 2 (M1 task 3): the sibling CI fixture
``robust/tests/files/l23_inelastic_fixture.npz`` -- not written yet; this
docstring is its reservation.

``Ed(0+)`` is a *sky* property: HydroLight computed one downwelling irradiance
per solar zenith, so the 3320 per-scene copies in each file must agree. That is
asserted (< 1e-3 relative scatter, the coding-plan gate; measured ~5e-5, float32
storage noise) **before** collapsing to one spectrum per zenith, and the spectra
are also cross-checked to be identical across the X=1/2/4 scenarios at each
zenith -- if either assumption ever fails in a new release, this script stops
rather than packaging an average of different skies.

Follows the BING generator this pattern is borrowed from
(``bing/bing/tests/files/gen_l23_inelastic_fixture.py``) and the elastic
``write_fixture`` discipline: the output is written to a temporary file,
verified, and only then moved into place atomically (PR #11's lesson --
validate first, overwrite second).

Run once (``ocean14``) on a machine with the L23 store::

    python design/py/gen_inelastic_fixture.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from robust.rt import conventions  # noqa: E402

#: Destination of the packaged Ed spectra (robust.rt.ed reads this).
ED_PATH = REPO / "robust" / "rt" / "data" / "ed_l23.npz"

#: Solar zeniths in the L23 release, degrees.
ZENITHS = (0, 30, 60)

#: The scene-independence gate (coding plan M1): maximum relative scatter of
#: Ed(0+) across the 3320 scenes of one file. Measured ~5e-5 (float32 noise).
MAX_REL_SCATTER = 1e-3


def extract_ed() -> dict[str, np.ndarray]:
    """Extract one ``Ed(0+)`` spectrum per zenith from the X=2 files.

    Returns
    -------
    dict
        ``wave`` (81,), ``zeniths`` (3,), ``Ed`` (3, 81) -- float32, matching
        the netCDF storage dtype; the mean is taken in float64.

    Raises
    ------
    AssertionError
        If Ed varies across scenes beyond :data:`MAX_REL_SCATTER`, differs
        between the X scenarios, or the wavelength grid is not the canonical
        one -- each would mean the packaged spectra misrepresent the sky.
    """
    from ocpy.hydrolight import loisel23

    wave = None
    spectra = []
    for zenith in ZENITHS:
        ds = loisel23.load_ds(2, zenith)
        ed = ds["Ed_0+"].values
        scatter = float((ed.std(axis=0) / ed.mean(axis=0)).max())
        assert scatter < MAX_REL_SCATTER, (
            f"Ed(0+) varies across scenes at Y={zenith}: relative scatter "
            f"{scatter:.2e} >= {MAX_REL_SCATTER:.0e} -- not a sky property?"
        )
        if wave is None:
            wave = np.asarray(ds["Lambda"].values, dtype=np.float32)
            conventions.check_wave(wave, name=f"L23 Lambda (X=2, Y={zenith})")
        collapsed = ed.mean(axis=0, dtype=np.float64)

        # The sky must not depend on the IOP scenario: X=1 and X=4 carry the
        # same illumination. rtol 1e-6 -- float32 data, expected bit-identical.
        for x_other in (1, 4):
            other = loisel23.load_ds(x_other, zenith)
            assert np.allclose(other["Ed_0+"].values, ed, rtol=1e-6, atol=0.0), (
                f"Ed(0+) differs between X=2 and X={x_other} at Y={zenith}"
            )
            other.close()

        print(
            f"  Y={zenith:>2}: scatter {scatter:.1e}, "
            f"Ed(440) = {collapsed[np.abs(wave - 440).argmin()]:.4f} W/m^2/nm"
        )
        spectra.append(collapsed.astype(np.float32))
        ds.close()

    return {
        "wave": wave,
        "zeniths": np.asarray(ZENITHS, dtype=np.int32),
        "Ed": np.stack(spectra),
    }


def write_ed(path: Path = ED_PATH) -> None:
    """Write the packaged Ed file atomically, verifying before replacing."""
    arrays = extract_ed()

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)

        # Verify the candidate before it replaces anything (PR #11).
        check = np.load(tmp)
        assert set(check.files) == {"wave", "zeniths", "Ed"}
        assert check["Ed"].shape == (len(ZENITHS), conventions.N_WAVE)
        assert check["Ed"].dtype == np.float32
        assert np.all(np.isfinite(check["Ed"])) and np.all(check["Ed"] > 0.0)
        conventions.check_wave(check["wave"])
        # Lower sun -> less irradiance, at every wavelength. A violated ordering
        # means the zenith rows were scrambled.
        assert np.all(np.diff(check["Ed"], axis=0) < 0.0), "Ed rows out of order"
        check.close()

        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} kB)")
    finally:
        tmp.unlink(missing_ok=True)


#: Destination of the elastic reference outputs (the hash-regression's
#: everywhere-runnable form; robust/tests/test_inelastic_types.py reads it).
ELASTIC_REF_PATH = REPO / "robust" / "tests" / "files" / "elastic_reference_outputs.npz"


def write_elastic_reference(path: Path = ELASTIC_REF_PATH) -> None:
    """Write the elastic `forward`/`rrs_forward` outputs on the CI fixture.

    The M0 hash-regression pins SHA-256 of these arrays, but bit-identity is
    anchored to the machine that computed them (prompt 1, Q&A Q2) — GitHub's
    heterogeneous runner fleet reproduces the bits on some runners and not
    others. Committing the arrays themselves lets every platform run a tight
    *closeness* regression while the strict bitwise gate stays scoped to
    non-CI machines. Regenerate only if the elastic model itself is
    deliberately changed — which the inelastic effort must never do.
    """
    from robust.rt import hybrid
    from robust.rt.data import l23

    batch = l23.load_batch(
        reader=l23.npz_reader(REPO / "robust" / "tests" / "files" / "l23_small.npz")
    )
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    arrays = {
        "Rrs": np.asarray(hybrid.forward(*args, check_domain=False)),
        "rrs": np.asarray(hybrid.rrs_forward(*args, check_domain=False)),
    }

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        check = np.load(tmp)
        assert set(check.files) == {"Rrs", "rrs"}
        for key in ("Rrs", "rrs"):
            assert check[key].dtype == np.float32
            assert check[key].shape == arrays[key].shape
            assert np.array_equal(check[key], arrays[key])
        check.close()
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} kB)")
    finally:
        tmp.unlink(missing_ok=True)


#: Destination of the sibling CI fixture (CQ4; M1 task 3).
INELASTIC_FIXTURE_PATH = (
    REPO / "robust" / "tests" / "files" / "l23_inelastic_fixture.npz"
)

#: Scenes in the sibling fixture -- the elastic fixture's own 50 (its
#: ``write_fixture`` takes the *first* ``n_scene`` scenes; CQ4 requires the
#: same indices, and the loader's cross-fixture equality check enforces it).
N_SCENE_FIXTURE = 50


def write_inelastic_fixture(path: Path = INELASTIC_FIXTURE_PATH) -> None:
    """Part 2: the sibling CI fixture (coding plan CQ4).

    The elastic fixture's 50 scene indices x {a, aph, bb, Rrs1, Rrs2, Rrs4}
    at all three zeniths, float32, and nothing else (Ed ships in the package
    data; ``bbnw``/``bnw`` stay in the elastic fixture, whose bytes this
    script never touches). The a/bb/Rrs1 copies exist so
    ``l23.inelastic_npz_reader`` can *prove* the two fixtures describe the
    same water rather than assuming it.

    Verified before replacing anything: loaded back through the real
    ``load_inelastic_batch`` via the real fixture reader, validated, and
    sample-counted (the write_fixture / PR #11 discipline).
    """
    from robust.rt.data import l23

    arrays: dict[str, np.ndarray] = {
        "zeniths": np.asarray(ZENITHS),
        "n_scene": np.asarray(N_SCENE_FIXTURE),
    }
    for zenith in ZENITHS:
        raw = l23._read_inelastic_file(zenith)  # noqa: SLF001 - the loader's own reader
        arrays["wave"] = raw["wave"].astype(np.float32)
        for field in ("a", "aph", "bb", "Rrs1", "Rrs2", "Rrs4"):
            arrays[f"{field}_{zenith}"] = raw[field][:N_SCENE_FIXTURE].astype(
                np.float32
            )

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        batch = l23.load_inelastic_batch(
            reader=l23.inelastic_npz_reader(
                tmp, REPO / "robust" / "tests" / "files" / "l23_small.npz"
            )
        )
        batch.validate()
        expected = len(ZENITHS) * N_SCENE_FIXTURE
        if batch.n_sample != expected:
            raise ValueError(
                f"write_inelastic_fixture: snapshot loads {batch.n_sample} "
                f"samples, expected {expected}; {path} left untouched"
            )
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} kB)")
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    print("Part 1: Ed(0+) spectra (X=2 files)")
    write_ed()
    print("Elastic reference outputs (hash-regression companion)")
    write_elastic_reference()
    print("Part 2: sibling CI fixture (X1/X2/X4 channels + aph)")
    write_inelastic_fixture()


if __name__ == "__main__":
    main()
