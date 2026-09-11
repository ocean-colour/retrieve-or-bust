"""
Tests for the inelastic data pipeline (inelastic coding plan, M1 task 3).

The load-bearing property is **split reuse**: M3's per-process gates are only
meaningful if the inelastic truth channels are held out on exactly the elastic
splits, so mask-for-mask equality is *proved* on the fixture (everywhere) and
on the full release (``needs_l23`` + ``needs_l23_inelastic``), not assumed
from the shared seed.

Everything else follows the elastic test philosophy: the committed sibling
fixture feeds the *real* loader through the *real* reader on every machine
(CI included), golden rows are cross-checked against the raw netCDFs where
the data exists, and absolute pins catch a changed reference release that
row-by-row comparisons would follow blindly.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from robust.rt.data import l23 as L
from robust.tests.conftest import (
    L23_INELASTIC_FIXTURE,
    L23_SMALL_FIXTURE,
    needs_l23,
    needs_l23_inelastic,
)

#: SHA-256 of the elastic fixture's bytes. CQ4 says the sibling must not touch
#: them; this pin turns that sentence into a failing test. Update only on a
#: *deliberate*, logged regeneration of the elastic fixture.
ELASTIC_FIXTURE_SHA256 = (
    "697dd44977ebd7aa23a7931a934e3cee8a3fbbe5e1cdff7ec58915d367841610"
)


# ------------------------------------------------------------- the fixture ----


def test_sibling_fixture_is_committed_and_small():
    """Present, and within the (corrected) ≲300 kB budget."""
    assert L23_INELASTIC_FIXTURE.is_file()
    assert L23_INELASTIC_FIXTURE.stat().st_size < 300 * 1024


def test_elastic_fixture_bytes_are_untouched():
    """CQ4: the sibling adds channels; the elastic fixture's bytes stay put."""
    digest = hashlib.sha256(L23_SMALL_FIXTURE.read_bytes()).hexdigest()
    assert digest == ELASTIC_FIXTURE_SHA256


def test_fixture_loads_without_os_color(l23_small_inelastic_batch):
    """The real loader runs from the two committed files alone."""
    batch = l23_small_inelastic_batch
    assert batch.n_sample == 150
    assert batch.n_wave == 81
    for channel in (batch.Rrs_x1, batch.Rrs_x2, batch.Rrs_x4):
        assert channel.shape == (150, 81)
    assert batch.iops.a_ph is not None
    assert batch.iops.a_ph.shape == (150, 81)
    assert batch.iops.a_cdom is not None
    assert batch.iops.a_cdom.shape == (150, 81)
    np.testing.assert_array_equal(np.unique(batch.zenith), [0.0, 30.0, 60.0])


def test_a_cdom_decomposition_bookkeeping(l23_small_inelastic_batch):
    """The a_dg double-counting foot-gun, pinned (CDOM design §8, M5 task 2).

    L23 stores ``ag`` (CDOM) separately from ``ad`` (detritus), so with
    ``a = a_water + a_ph + a_cdom + a_detrital`` the loaded components must
    satisfy ``a_cdom ≥ 0`` and ``a_ph + a_cdom ≤ a`` *everywhere* — a real
    assertion on real numbers, not an assumption about the decomposition.
    (Measured margin on the fixture: max(a_ph + a_cdom − a) ≈ −5.3e-3.)
    """
    batch = l23_small_inelastic_batch
    a = np.asarray(batch.iops.a)
    a_ph = np.asarray(batch.iops.a_ph)
    a_cdom = np.asarray(batch.iops.a_cdom)
    assert a_cdom.min() >= 0.0
    assert np.all(a_ph + a_cdom <= a)


def test_reader_rejects_mismatched_fixtures(tmp_path):
    """A sibling that disagrees with the elastic fixture raises, not pairs.

    The cross-fixture equality check is the reader's whole safety story:
    corrupt one shared array and the load must fail loudly.
    """
    data = dict(np.load(L23_INELASTIC_FIXTURE))
    data["a_30"] = data["a_30"] * 1.001
    bad = tmp_path / "bad_sibling.npz"
    np.savez_compressed(bad, **data)

    reader = L.inelastic_npz_reader(bad, L23_SMALL_FIXTURE)
    reader(0)  # untouched zenith still reads
    with pytest.raises(ValueError, match="do not describe the same scenes"):
        reader(30)


# ------------------------------------------------------------ truth channels ----


def test_truth_channels_are_the_documented_identities(l23_small_inelastic_batch):
    """``truth_raman_factor`` and ``truth_fluorescence`` are exactly their
    definitions — bitwise, so nothing can drift between property and prose."""
    batch = l23_small_inelastic_batch
    np.testing.assert_array_equal(
        np.asarray(batch.truth_raman_factor),
        np.asarray(batch.Rrs_x2) / np.asarray(batch.Rrs_x1),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.truth_fluorescence),
        np.asarray(batch.Rrs_x4) - np.asarray(batch.Rrs_x2),
    )


def test_golden_absolute_values():
    """Absolute pins on fixture rows — the guard row-by-row checks cannot give.

    If the L23 release itself changes, the netCDF cross-check would follow it;
    these numbers would not.
    """
    data = np.load(L23_INELASTIC_FIXTURE)
    i440 = int(np.abs(data["wave"] - 440.0).argmin())
    i685 = int(np.abs(data["wave"] - 685.0).argmin())
    assert data["Rrs2_0"][0, i440] == pytest.approx(9.2462e-03, rel=1e-4)
    assert data["Rrs4_0"][0, i685] == pytest.approx(1.2273e-04, rel=1e-4)
    assert data["aph_0"][0, i440] == pytest.approx(3.7110e-03, rel=1e-4)
    assert data["ag_0"][0, i440] == pytest.approx(5.7960e-03, rel=1e-4)
    assert data["ag_30"][7, i440] == pytest.approx(3.1390e-03, rel=1e-4)


def test_raman_factor_is_physical(l23_small_inelastic_batch):
    """Raman only adds photons: X2/X1 ≥ 1 everywhere (measured min 1.0076),
    and the gain grows toward the red — the shape M2's f_phys must reproduce."""
    factor = np.asarray(l23_small_inelastic_batch.truth_raman_factor)
    assert factor.min() >= 1.0
    wave = np.asarray(l23_small_inelastic_batch.wave)
    blue = np.median(factor[:, np.abs(wave - 450.0).argmin()])
    red = np.median(factor[:, np.abs(wave - 650.0).argmin()])
    assert red > blue > 1.0


def test_fluorescence_peaks_at_685(l23_small_inelastic_batch):
    """The fluorescence delta is strictly positive at 685 nm in every sample,
    and 685 nm is its median spectral argmax within the emission band."""
    batch = l23_small_inelastic_batch
    wave = np.asarray(batch.wave)
    delta = np.asarray(batch.truth_fluorescence)
    i685 = int(np.abs(wave - 685.0).argmin())
    assert np.all(delta[:, i685] > 0.0)
    band = (wave >= 640.0) & (wave <= 720.0)
    peak_wave = wave[band][np.argmax(np.median(delta[:, band], axis=0))]
    assert peak_wave == pytest.approx(685.0, abs=5.0)


# --------------------------------------------------------------- split reuse ----


def test_splits_equal_the_elastic_splits(l23_small_inelastic_batch, l23_small_batch):
    """The load-bearing gate: identical masks, mask for mask, scene for scene.

    ``make_splits`` reads only ``scene``/``zenith``, and the inelastic loader
    reproduces the elastic layout exactly — proved here rather than assumed
    from the shared seed.
    """
    elastic = L.make_splits(l23_small_batch)
    inelastic = L.make_splits(l23_small_inelastic_batch)
    np.testing.assert_array_equal(inelastic.test_scenes, elastic.test_scenes)
    for field in ("scene_train", "scene_test", "zenith_train", "zenith_test"):
        np.testing.assert_array_equal(
            getattr(inelastic, field), getattr(elastic, field)
        )


def test_sample_ordering_matches_the_elastic_batch(
    l23_small_inelastic_batch, l23_small_batch
):
    """Same scenes, same zeniths, same order — and bit-identical shared data.

    ``Rrs_x1`` *is* the elastic reference; ``a`` is the same water. Bitwise
    equality here is what makes every elastic-side quantity directly
    comparable to the inelastic channels, row for row.
    """
    inelastic, elastic = l23_small_inelastic_batch, l23_small_batch
    np.testing.assert_array_equal(inelastic.scene, elastic.scene)
    np.testing.assert_array_equal(inelastic.zenith, elastic.zenith)
    np.testing.assert_array_equal(np.asarray(inelastic.Rrs_x1), np.asarray(elastic.Rrs))
    np.testing.assert_array_equal(
        np.asarray(inelastic.iops.a), np.asarray(elastic.iops.a)
    )
    np.testing.assert_array_equal(
        np.asarray(inelastic.phase_params.B_p), np.asarray(elastic.phase_params.B_p)
    )


# ------------------------------------------------------------------ subsetting ----


def test_select_inelastic_subsets_consistently(l23_small_inelastic_batch):
    """Channels, IOPs (a_ph included), and labels move together."""
    batch = l23_small_inelastic_batch
    splits = L.make_splits(batch)

    test = L.select_inelastic(batch, splits.scene_test)

    n = int(splits.scene_test.sum())
    assert test.n_sample == n
    assert test.iops.a_ph.shape == (n, 81)
    assert test.Rrs_x2.shape == (n, 81)
    keep = np.flatnonzero(splits.scene_test)
    np.testing.assert_array_equal(
        np.asarray(test.Rrs_x4), np.asarray(batch.Rrs_x4)[keep]
    )
    test.validate()


def test_select_inelastic_rejects_a_bad_mask(l23_small_inelastic_batch):
    with pytest.raises(ValueError, match="boolean mask"):
        L.select_inelastic(l23_small_inelastic_batch, np.ones(5, dtype=bool))


def test_validate_requires_a_ph(l23_small_inelastic_batch):
    """An inelastic batch without the fluorescence source term is a bug."""
    import dataclasses

    stripped = dataclasses.replace(
        l23_small_inelastic_batch,
        iops=dataclasses.replace(l23_small_inelastic_batch.iops, a_ph=None),
    )
    with pytest.raises(ValueError, match="a_ph"):
        stripped.validate()


def test_validate_requires_a_cdom(l23_small_inelastic_batch):
    """An inelastic batch without the CDOM-fl source term is likewise a bug."""
    import dataclasses

    stripped = dataclasses.replace(
        l23_small_inelastic_batch,
        iops=dataclasses.replace(l23_small_inelastic_batch.iops, a_cdom=None),
    )
    with pytest.raises(ValueError, match="a_cdom"):
        stripped.validate()


# --------------------------------------------------------- against the netCDF ----


@needs_l23_inelastic
def test_fixture_rows_match_the_raw_netcdf():
    """Golden: every fixture array is the raw file's first 50 rows, bit-faithful
    to the float32 the netCDF stores."""
    from ocpy.hydrolight import loisel23

    data = np.load(L23_INELASTIC_FIXTURE)
    for zenith in (0, 30, 60):
        ds1 = loisel23.load_ds(1, zenith)
        ds2 = loisel23.load_ds(2, zenith)
        ds4 = loisel23.load_ds(4, zenith)
        np.testing.assert_array_equal(
            data[f"aph_{zenith}"], ds1.aph.data[:50].astype(np.float32)
        )
        np.testing.assert_array_equal(
            data[f"ag_{zenith}"], ds1.ag.data[:50].astype(np.float32)
        )
        np.testing.assert_array_equal(
            data[f"Rrs2_{zenith}"], ds2.Rrs.data[:50].astype(np.float32)
        )
        np.testing.assert_array_equal(
            data[f"Rrs4_{zenith}"], ds4.Rrs.data[:50].astype(np.float32)
        )
        for ds in (ds1, ds2, ds4):
            ds.close()


@needs_l23
@needs_l23_inelastic
def test_loader_a_cdom_golden_against_the_raw_netcdf():
    """Golden a_cdom rows: the loader reproduces the raw ``ag`` bit for bit.

    Mirrors the a_ph golden pattern — the value is read *directly from the
    raw file* (float64, no fixture in the loop) and the live loader must
    reproduce it at specific (scene, zenith) samples.
    """
    from ocpy.hydrolight import loisel23

    n_scene = 10
    batch = L.load_inelastic_batch(zeniths=(0, 60), scenes=slice(0, n_scene))
    a_cdom = np.asarray(batch.iops.a_cdom)

    for block, zenith in enumerate((0, 60)):
        ds = loisel23.load_ds(1, zenith)
        raw = np.asarray(ds.ag.data, dtype=float)
        ds.close()
        for scene in (0, 7):
            row = block * n_scene + scene
            assert batch.zenith[row] == zenith and batch.scene[row] == scene
            np.testing.assert_array_equal(a_cdom[row], raw[scene])


@needs_l23
@needs_l23_inelastic
def test_full_release_a_cdom_decomposition_bookkeeping():
    """The §8 foot-gun pins at full scale: over all 9960 samples of the live
    release, ``a_cdom ≥ 0`` and ``a_ph + a_cdom ≤ a`` everywhere."""
    batch = L.load_inelastic_batch()
    a = np.asarray(batch.iops.a)
    a_ph = np.asarray(batch.iops.a_ph)
    a_cdom = np.asarray(batch.iops.a_cdom)
    assert batch.n_sample == 3 * L.N_SCENES
    assert a_cdom.min() >= 0.0
    assert np.all(a_ph + a_cdom <= a)


@needs_l23
@needs_l23_inelastic
def test_full_batch_loads_and_reuses_the_elastic_splits():
    """The full release: 9960 samples, and split equality holds at scale."""
    inelastic = L.load_inelastic_batch()
    elastic = L.load_batch()

    assert inelastic.n_sample == 3 * L.N_SCENES
    np.testing.assert_array_equal(inelastic.scene, elastic.scene)

    s_in = L.make_splits(inelastic)
    s_el = L.make_splits(elastic)
    np.testing.assert_array_equal(s_in.test_scenes, s_el.test_scenes)
    np.testing.assert_array_equal(s_in.scene_test, s_el.scene_test)
