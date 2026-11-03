"""
Tests for :mod:`robust.rt.data.pb24_cache` (M8 task 2).

Four things are being defended.

**The reader works without the data.** The committed miniature
(:data:`PB24_CACHE_FIXTURE`, one zip: three realisations, every 10th band in
4-D) was written by the same :func:`~robust.rt.data.pb24_cache.build` as the
full cache, is unpacked once per session (the ``pb24_cache_dir`` fixture) and
is opened by the same :func:`~robust.rt.data.pb24_cache.open_cache`, so the
sidecar, the memmaps and every slicing path run in CI.

**Slicing returns what plain indexing would.** Every selector (ids, angle
indices or values, band indices or wavelengths) is checked against direct
numpy indexing of the whole memmap, and slices are checked to be views.

**A stale or broken cache is refused**: wrong ``CACHE_VERSION``, wrong
``READER_VERSION``, no sidecar (which is what a crashed build leaves).

**The cache is the files.** With the data: the Task-1 golden values read from
the cache equal the ocpy reader bit for bit; ``verify`` passes on
realisations 1, 2500 and 5000 in full; and rebuilding the fixture from the
netCDFs reproduces it byte for byte, for one worker and for several.
"""

from __future__ import annotations

import filecmp
import json

import numpy as np
import pytest

from robust.rt.data import pb24_cache as PC
from robust.tests.conftest import (
    PB24_CACHE_FIXTURE,
    PB24_CACHE_FIXTURE_BANDS,
    PB24_FIXTURE_REALISATIONS,
    needs_pb24_cache,
    needs_pb24_hyp,
)

#: Task-1 golden values (hyp): (realisation, λ, θs, θ, φ) ->
#: rrs, Rrs, aph(λ), Esdir_Es_ratio(λ, θs), float32 as stored.
GOLDEN = {
    (1, 443, 30, 10, 90): (0.0071871569, 0.0037851033, 0.0881650001, 0.7778172493),
    (2500, 560, 60, 40, 135): (0.0865792707, 0.0447451025, 0.0140349995, 0.7885347009),
    (5000, 700, 0, 0, 0): (6.9509493e-05, 3.6750615e-05, 6.30e-05, 0.9163848162),
}


@pytest.fixture(scope="module")
def mini(pb24_cache_dir):
    """The committed miniature cache, unpacked and opened through the reader."""
    return PC.open_cache(pb24_cache_dir)


# -- the fixture, no data needed -------------------------------------------------


def test_fields_constants():
    assert len(PC.IOP_FIELDS) == 10
    assert len(PC.AOP_FIELDS) == 12  # eleven AOPs + Esdir_Es_ratio
    assert set(PC.FIELDS) == set(PC.SOURCE_DIMS) == set(PC.CACHE_DIMS)
    assert len(set(PC.FIELDS)) == len(PC.FIELDS) == 28


def test_sidecar_round_trip(mini, pb24_cache_dir):
    text = (pb24_cache_dir / PC.SIDECAR).read_text()
    sidecar = json.loads(text)
    assert sidecar == mini.sidecar == PC.read_sidecar(pb24_cache_dir)
    # Canonical form: sorted keys, so a rebuild writes the same bytes.
    assert json.dumps(sidecar, sort_keys=True, indent=1) + "\n" == text
    assert sidecar["cache_version"] == PC.CACHE_VERSION
    assert sidecar["resolution"] == "hyp"
    assert "time" not in text.lower() and "date" not in text.lower()
    assert sidecar["realisations"] == list(PB24_FIXTURE_REALISATIONS)
    assert [s["name"] for s in sidecar["sources"]] == [
        f"SD_hyp_no_R_{r:04d}.nc" for r in PB24_FIXTURE_REALISATIONS
    ]
    assert sidecar["four_d_bands"] == list(PB24_CACHE_FIXTURE_BANDS)


def test_grids(mini):
    g = mini.grids
    assert g["lambda"].dtype == np.float32
    np.testing.assert_array_equal(g["lambda"], np.arange(350, 801, dtype=np.float32))
    assert g["theta_s"].size == g["theta"].size == 10 and g["phi"].size == 13
    assert g["theta_s"][-1] == np.float32(87.75) and g["theta"][-1] == 87.5
    np.testing.assert_array_equal(mini.four_d_wavelengths, np.arange(350, 801, 10))


def test_shapes_and_memmaps(mini):
    n = len(PB24_FIXTURE_REALISATIONS)
    nb = len(PB24_CACHE_FIXTURE_BANDS)
    expect = {
        **{f: (n,) for f in PC.SCALAR_FIELDS},
        **{f: (n, 451) for f in PC.IOP_FIELDS},
        **{f: (n, 10, 451) for f in PC.AOP_FIELDS},
        **{f: (n, 10, 10, 13, nb) for f in PC.FOUR_D_FIELDS},
    }
    assert set(mini.fields) == set(expect)
    for name, shape in expect.items():
        arr = mini.array(name)
        assert isinstance(arr, np.memmap)
        assert arr.shape == shape and arr.dtype == np.float32
        assert not arr.flags.writeable
        assert mini.sidecar["fields"][name]["shape"] == list(shape)
    assert mini.array("rrs") is mini.array("rrs")  # opened once, lazily


def test_values_are_physical(mini):
    """Spot checks that the transposes put the right axes in the right place."""
    # Water IOPs do not depend on the realisation.
    aw = mini.table("aw")
    np.testing.assert_array_equal(aw[0], aw[1])
    # Esdir_Es_ratio is above-surface: identical across realisations, and the
    # direct share falls with sun zenith (axis 1 is theta_s).
    e = mini.table("Esdir_Es_ratio")
    np.testing.assert_array_equal(e[0], e[2])
    assert np.all(np.diff(e[0, :, 93]) < 0)
    # rrs * Q = R (an identity per in-water direction) at stored bands.
    rrs, q = mini.array("rrs"), mini.array("Q")
    R = mini.table("R")[:, :, mini.four_d_bands]
    ratio = rrs[:, :, 0, 0, :] * q[:, :, 0, 0, :] / R
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-3)


def test_realisation_index(mini):
    assert mini.realisation_index(993) == 1
    np.testing.assert_array_equal(mini.realisation_index([2500, 1]), [2, 0])
    with pytest.raises(KeyError):
        mini.realisation_index([1, 2])
    with pytest.raises(KeyError):
        mini.realisation_index(99999)


def test_table_slicing(mini):
    full = mini.array("Kd")
    np.testing.assert_array_equal(mini.table("Kd", 993), full[1])
    np.testing.assert_array_equal(mini.table("Kd", [2500, 1]), full[[2, 0]])
    view = mini.table("Kd", [1, 993])  # contiguous rows: a view
    assert np.shares_memory(view, full)
    with pytest.raises(KeyError):
        mini.table("rrs")


def test_four_d_by_index(mini):
    full = np.asarray(mini.array("Rrs"))
    stored = list(PB24_CACHE_FIXTURE_BANDS)
    got = mini.four_d("Rrs", realisations=[2500, 1], theta_s=[3, 0], theta=2, phi=6)
    want = full[[2, 0]][:, [3, 0]][:, :, 2, 6, :]
    np.testing.assert_array_equal(got, want)
    # Bands are full-grid indices, mapped to stored positions.
    got = mini.four_d("Rrs", 993, 1, 1, 1, bands=[0, 90, 450])
    want = full[1, 1, 1, 1, [0, stored.index(90), stored.index(450)]]
    np.testing.assert_array_equal(got, want)
    # Basic selections and evenly spaced runs are views.
    v = mini.four_d("Rrs", realisations=[1, 993], theta_s=slice(0, 4), phi=[0, 2, 4])
    assert v.shape == (2, 4, 10, 3, len(stored)) and np.shares_memory(
        v, mini.array("Rrs")
    )
    with pytest.raises(KeyError):
        mini.four_d("Rrs", bands=5)  # not stored in the miniature
    with pytest.raises(KeyError):
        mini.four_d("aph")


def test_four_d_by_value(mini):
    full = np.asarray(mini.array("rrs"))
    stored = list(PB24_CACHE_FIXTURE_BANDS)
    got = mini.four_d_at("rrs", 1, theta_s=30, theta=10, phi=90, wavelengths=440)
    assert got == full[0, 3, 1, 6, stored.index(90)]
    got = mini.four_d_at("rrs", [1, 2500], theta_s=87.75, wavelengths=[350, 800])
    np.testing.assert_array_equal(got, full[[0, 2]][:, 9][..., [0, len(stored) - 1]])
    with pytest.raises(KeyError):
        mini.four_d_at("rrs", theta=45)  # not a node: no rounding
    with pytest.raises(KeyError):
        mini.four_d_at("rrs", wavelengths=443)  # a node, but not stored here


def _copy_fixture(tmp_path):
    dst = tmp_path / "cache"
    PC.unpack(PB24_CACHE_FIXTURE, dst)
    return dst


def test_pack_is_deterministic_and_round_trips(tmp_path, pb24_cache_dir):
    """Packing twice gives identical bytes; unpacking restores every file."""
    z1 = PC.pack(pb24_cache_dir, tmp_path / "a.zip")
    z2 = PC.pack(pb24_cache_dir, tmp_path / "b.zip")
    assert filecmp.cmp(z1, z2, shallow=False)
    assert filecmp.cmp(z1, PB24_CACHE_FIXTURE, shallow=False)
    back = PC.unpack(z1, tmp_path / "back")
    names = sorted(p.name for p in pb24_cache_dir.iterdir())
    assert sorted(p.name for p in back.iterdir()) == names
    for name in names:
        assert filecmp.cmp(pb24_cache_dir / name, back / name, shallow=False), name


@pytest.mark.parametrize("key", ["cache_version", "reader_version"])
def test_version_mismatch_raises(tmp_path, mini, key):
    dst = _copy_fixture(tmp_path)
    sidecar = json.loads((dst / PC.SIDECAR).read_text())
    sidecar[key] += 1
    (dst / PC.SIDECAR).write_text(json.dumps(sidecar, sort_keys=True))
    with pytest.raises(ValueError, match="version"):
        PC.open_cache(dst)


def test_no_sidecar_is_no_cache(tmp_path, mini):
    dst = _copy_fixture(tmp_path)
    (dst / PC.SIDECAR).unlink()
    with pytest.raises(FileNotFoundError):
        PC.open_cache(dst)


def test_shape_mismatch_raises(tmp_path, mini):
    dst = _copy_fixture(tmp_path)
    np.save(dst / "aph.npy", np.zeros((2, 451), np.float32))
    cache = PC.open_cache(dst)
    with pytest.raises(ValueError, match="sidecar"):
        cache.array("aph")


def test_default_cache_dir_follows_os_color():
    from ocpy.hydrolight import pb24

    d = PC.default_cache_dir()
    assert d.name == f"pb24_v{PC.CACHE_VERSION}"
    assert d.parent.parent == __import__("pathlib").Path(pb24.pb24_path).parent


# -- with the netCDFs -----------------------------------------------------------


@needs_pb24_hyp
@pytest.mark.parametrize("workers", [1, 2])
def test_build_reproduces_fixture(tmp_path, pb24_cache_dir, workers):
    summary = PC.build(
        tmp_path / "rebuilt",
        realisations=PB24_FIXTURE_REALISATIONS,
        bands=PB24_CACHE_FIXTURE_BANDS,
        workers=workers,
    )
    names = sorted(p.name for p in pb24_cache_dir.iterdir())
    assert sorted(p.name for p in summary.cache_dir.iterdir()) == names
    for name in names:
        assert filecmp.cmp(
            pb24_cache_dir / name, summary.cache_dir / name, shallow=False
        ), name
    assert set(summary.sizes) == set(names)


@needs_pb24_hyp
def test_verify_fixture(pb24_cache_dir):
    report = PC.verify(pb24_cache_dir, full=PB24_FIXTURE_REALISATIONS)
    assert report.ok and report.n_four_d_values == report.n_four_d_total


@needs_pb24_hyp
def test_crashed_build_leaves_no_cache(tmp_path, monkeypatch):
    def boom(task):
        raise RuntimeError("simulated crash")

    dst = tmp_path / "crashed"
    monkeypatch.setattr(PC, "_load_one", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        PC.build(dst, realisations=(1, 2))
    assert list(dst.iterdir()) == []  # no sidecar, no partial arrays


@needs_pb24_hyp
def test_verify_catches_a_flipped_bit(tmp_path, mini):
    dst = _copy_fixture(tmp_path)
    arr = np.load(dst / "Kd.npy")
    arr.view(np.uint32)[1, 4, 200] ^= 1  # one ulp in one value
    np.save(dst / "Kd.npy", arr)
    report = PC.verify(dst, fields=("Kd",), raise_on_mismatch=False)
    assert report.mismatches == [(993, "Kd", 1)]
    with pytest.raises(AssertionError):
        PC.verify(dst, fields=("Kd",))


# -- with the full cache --------------------------------------------------------


@pytest.fixture(scope="module")
def full_cache():
    return PC.open_cache()


@needs_pb24_cache
def test_full_cache_layout(full_cache):
    assert full_cache.realisations.tolist() == list(range(1, 5001))
    assert full_cache.four_d_bands.tolist() == list(range(451))
    assert full_cache.array("rrs").shape == (5000, 10, 10, 13, 451)
    assert full_cache.array("Kd").shape == (5000, 10, 451)


@needs_pb24_cache
@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_golden_values(full_cache, key):
    from ocpy.hydrolight import pb24

    r, lam, ts, th, ph = key
    ds = pb24.load_ds(r)
    sel = {"lambda": lam, "theta_s": ts, "theta": th, "phi": ph}
    for name, golden in zip(("rrs", "Rrs"), GOLDEN[key][:2], strict=True):
        cached = full_cache.four_d_at(name, r, ts, th, ph, lam)
        raw = ds[name].sel(sel).values
        assert cached.dtype == np.float32
        assert cached.tobytes() == raw.tobytes()
        assert cached == pytest.approx(golden, rel=1e-7)
    i = full_cache.band_index(lam)
    aph = full_cache.table("aph", r)[i]
    assert aph.tobytes() == ds["aph"].sel({"lambda": lam}).values.tobytes()
    assert aph == pytest.approx(GOLDEN[key][2], rel=1e-7)
    j = full_cache.angle_index("theta_s", ts)
    e = full_cache.table("Esdir_Es_ratio", r)[j, i]
    raw = ds["Esdir_Es_ratio"].sel({"lambda": lam, "theta_s": ts}).values
    assert e.tobytes() == raw.tobytes()
    assert e == pytest.approx(GOLDEN[key][3], rel=1e-7)


@needs_pb24_cache
def test_verify_gate_realisations(full_cache):
    report = PC.verify(
        realisations=PC.GATE_FULL_REALISATIONS, full=PC.GATE_FULL_REALISATIONS
    )
    assert report.ok
    assert report.n_four_d_values == report.n_four_d_total == 3 * 3 * 1300 * 451
