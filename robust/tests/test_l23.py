"""
Tests for :mod:`robust.rt.data.l23` (M1 task 3).

Deliberately in three layers, by what each can be trusted to prove.

**Logic, on a synthetic batch** -- no files at all. The splits *are* the M4
acceptance gate: if scene leakage crept in, every held-out number afterwards would
be optimistic, so that must be covered on machines (CI included) without the
dataset.

**The real loader, on the committed fixture** (``files/l23_small.npz``, 50 scenes
x 3 zeniths). The fixture stores the loader's *input*, so ``load_batch`` genuinely
runs against real L23 numbers here; a snapshot of its output could only be checked
for staleness. This is what gives CI real-data coverage of the loader.

**The full release, behind ``needs_l23``** -- the claims only 3320 scenes x 3
zeniths can support: the ``B_p`` range over all 81 bands, the golden row
cross-checked against the raw netCDF, and the IOPs-identical-across-zeniths
observation.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt.data import l23 as L
from robust.rt.types import Geometry, IOPs, PhaseParams
from robust.tests.conftest import needs_l23

N = C.N_WAVE


def synthetic_batch(n_scene: int = 10, zeniths=L.ZENITHS) -> L.L23Batch:
    """A physically plausible batch with no file access, for logic tests."""
    n = n_scene * len(zeniths)
    wave = C.canonical_wave()
    bb_w = np.broadcast_to(C.BB_W_L23, (n, N))
    return L.L23Batch(
        iops=IOPs(
            a=jnp.asarray(np.full((n, N), 0.15)),
            bb_w=jnp.asarray(bb_w),
            bb_p=jnp.asarray(np.full((n, N), 0.003)),
        ),
        phase_params=PhaseParams(B_p=jnp.asarray(np.full((n, N), 0.0126))),
        geometry=Geometry.nadir(
            jnp.asarray(np.repeat(np.asarray(zeniths, dtype=float), n_scene))
        ),
        Rrs=jnp.asarray(np.full((n, N), 5e-3)),
        wave=wave,
        scene=np.tile(np.arange(n_scene), len(zeniths)),
    )


# ----------------------------------------------------------- container logic -


def test_batch_reports_its_own_shape():
    batch = synthetic_batch(n_scene=10)

    assert batch.n_sample == 30
    assert batch.n_wave == N
    np.testing.assert_array_equal(np.unique(batch.zenith), [0.0, 30.0, 60.0])


def test_batch_validate_accepts_a_plausible_batch():
    synthetic_batch().validate()


def test_batch_validate_rejects_rrs_shape_mismatch():
    batch = synthetic_batch(n_scene=4)
    broken = dataclasses.replace(batch, Rrs=batch.Rrs[:, :-1])

    with pytest.raises(ValueError, match="does not match"):
        broken.validate()


def test_batch_validate_rejects_bad_scene_labels():
    batch = synthetic_batch(n_scene=4)
    broken = dataclasses.replace(batch, scene=batch.scene[:-1])

    with pytest.raises(ValueError, match="does not label"):
        broken.validate()


def test_batch_validate_rejects_nonpositive_rrs():
    batch = synthetic_batch(n_scene=4)
    broken = dataclasses.replace(batch, Rrs=batch.Rrs.at[0, 0].set(-1e-6))

    with pytest.raises(ValueError, match="negative"):
        broken.validate()


def test_batch_is_not_a_pytree():
    """It holds host-side integer labels, which must never be traced.

    Registering it would invite ``jax.grad`` over a scene index.
    """
    import jax

    leaves = jax.tree_util.tree_leaves(synthetic_batch(n_scene=2))

    # A plain dataclass is an opaque leaf, not a container of arrays.
    assert len(leaves) == 1


# ------------------------------------------------------------------- splits --


def test_scene_split_holds_out_the_requested_fraction():
    batch = synthetic_batch(n_scene=100)

    splits = L.make_splits(batch)

    assert splits.test_scenes.size == 20  # 20% of 100 scenes
    # Every zenith of every held-out scene is held out with it.
    assert splits.scene_test.sum() == 20 * len(L.ZENITHS)
    assert splits.scene_train.sum() == 80 * len(L.ZENITHS)


def test_scene_split_does_not_leak_scenes_between_train_and_test():
    """The property the whole M4 gate rests on.

    A per-*sample* split would put the same water body in both sets at different
    sun angles, and every held-out number afterwards would be optimistic.
    """
    batch = synthetic_batch(n_scene=50)

    splits = L.make_splits(batch)

    train_scenes = set(batch.scene[splits.scene_train].tolist())
    test_scenes = set(batch.scene[splits.scene_test].tolist())

    assert train_scenes.isdisjoint(test_scenes)
    assert train_scenes | test_scenes == set(batch.scene.tolist())


def test_scene_split_masks_are_complementary():
    batch = synthetic_batch(n_scene=25)

    splits = L.make_splits(batch)

    np.testing.assert_array_equal(splits.scene_train, ~splits.scene_test)


def test_scene_split_is_deterministic_for_a_given_seed():
    batch = synthetic_batch(n_scene=40)

    first = L.make_splits(batch)
    again = L.make_splits(batch)
    other = L.make_splits(batch, seed=L.SPLIT_SEED + 1)

    np.testing.assert_array_equal(first.test_scenes, again.test_scenes)
    assert not np.array_equal(first.test_scenes, other.test_scenes)


def test_zenith_split_holds_out_sixty_degrees():
    batch = synthetic_batch(n_scene=10)

    splits = L.make_splits(batch)

    assert set(batch.zenith[splits.zenith_train].tolist()) == {0.0, 30.0}
    assert set(batch.zenith[splits.zenith_test].tolist()) == {60.0}


def test_make_splits_rejects_a_missing_held_out_zenith():
    """A vacuous gate is worse than a failing one."""
    batch = synthetic_batch(n_scene=5, zeniths=(0, 30))

    with pytest.raises(ValueError, match="no samples at 60"):
        L.make_splits(batch)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_make_splits_rejects_bad_fractions(bad):
    with pytest.raises(ValueError, match="test_fraction"):
        L.make_splits(synthetic_batch(n_scene=5), test_fraction=bad)


# ------------------------------------------------------------------ select ---


def test_select_subsets_every_field_consistently():
    batch = synthetic_batch(n_scene=10)
    splits = L.make_splits(batch)

    test = L.select(batch, splits.scene_test)

    assert test.n_sample == int(splits.scene_test.sum())
    assert test.iops.a.shape == (test.n_sample, N)
    assert test.phase_params.B_p.shape == (test.n_sample, N)
    assert test.Rrs.shape == (test.n_sample, N)
    assert test.scene.shape == (test.n_sample,)
    np.testing.assert_array_equal(np.asarray(test.wave), np.asarray(batch.wave))
    test.validate()


def test_select_keeps_the_right_rows():
    batch = synthetic_batch(n_scene=6)
    mask = np.zeros(batch.n_sample, dtype=bool)
    mask[[0, 5, 11]] = True

    picked = L.select(batch, mask)

    np.testing.assert_array_equal(picked.scene, batch.scene[mask])
    np.testing.assert_array_equal(picked.zenith, batch.zenith[mask])


@pytest.mark.parametrize(
    "bad",
    [np.ones(5, dtype=bool), np.ones(30, dtype=int)],
    ids=["wrong-length", "not-boolean"],
)
def test_select_rejects_a_bad_mask(bad):
    with pytest.raises(ValueError, match="boolean mask"):
        L.select(synthetic_batch(n_scene=10), bad)


# ------------------------------------------------------------ B_p reporting --


def test_B_p_outside_the_expected_range_warns_but_does_not_clip():
    """Reported, never squashed: clipping would hide a change in the reference."""
    B_p = np.array([0.0126, 0.5])

    with pytest.warns(UserWarning, match="leaves the expected range"):
        L._warn_if_B_p_unexpected(B_p)

    assert B_p[1] == 0.5  # untouched


def test_B_p_inside_the_expected_range_is_silent():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L._warn_if_B_p_unexpected(np.array([0.005, 0.0126, 0.029]))


# ------------------------------------- the cached fixture: real data, no mount -
#
# These run everywhere, CI included. The fixture stores the loader's *input*, so
# `load_batch` genuinely executes here against real L23 numbers -- a snapshot of
# its output could only ever be checked for staleness. 50 scenes x 3 zeniths.


def test_fixture_drives_the_real_loader(l23_small_batch):
    """Shapes and grid, straight out of ``load_batch``."""
    assert l23_small_batch.n_sample == 3 * 50
    assert l23_small_batch.n_wave == C.N_WAVE
    C.check_wave(np.asarray(l23_small_batch.wave))
    np.testing.assert_array_equal(np.unique(l23_small_batch.zenith), [0.0, 30.0, 60.0])
    np.testing.assert_array_equal(l23_small_batch.scene[:50], np.arange(50))


def test_fixture_batch_is_physical(l23_small_batch):
    l23_small_batch.validate()

    assert float(jnp.min(l23_small_batch.iops.a)) >= 0.0
    assert float(jnp.min(l23_small_batch.iops.bb_p)) >= 0.0
    assert float(jnp.min(l23_small_batch.Rrs)) > 0.0


def test_fixture_B_p_is_inside_the_design_band(l23_small_batch):
    """The range assert, exercised in CI rather than only on the dev machine."""
    B_p = np.asarray(l23_small_batch.phase_params.B_p)
    lo, hi = L.B_P_EXPECTED

    assert B_p.min() >= lo
    assert B_p.max() <= hi
    assert np.all(B_p.min(axis=0) >= lo)  # per band, not just globally


def test_fixture_bb_w_agrees_with_the_conventions_table(l23_small_batch):
    """Cross-module agreement, checked without the dataset (see rtol note below)."""
    bb_w = np.asarray(l23_small_batch.iops.bb_w)

    np.testing.assert_allclose(bb_w, np.broadcast_to(C.BB_W_L23, bb_w.shape), rtol=1e-5)


def test_fixture_rrs_falls_with_solar_zenith(l23_small_batch):
    """The geometry signal, reproduced on the 50-scene subset."""
    Rrs = np.asarray(l23_small_batch.Rrs).reshape(len(L.ZENITHS), 50, C.N_WAVE)

    ratio_30 = np.median(Rrs[1] / Rrs[0])
    ratio_60 = np.median(Rrs[2] / Rrs[0])

    assert ratio_30 == pytest.approx(0.990, abs=0.005)
    assert ratio_60 == pytest.approx(0.946, abs=0.005)
    assert ratio_60 < ratio_30 < 1.0


def test_fixture_golden_value(l23_small_batch):
    """The same absolute pin as the full-data test, available in CI."""
    i440 = int(np.argmin(np.abs(np.asarray(l23_small_batch.wave) - 440.0)))

    assert float(l23_small_batch.Rrs[0, i440]) == pytest.approx(
        8.53329990e-03, rel=1e-6
    )


def test_fixture_splits_and_select_on_real_labels(l23_small_batch):
    """Splits and subsetting over genuine scene labels, not synthetic ones."""
    splits = L.make_splits(l23_small_batch)

    assert splits.test_scenes.size == 10  # 20% of 50
    assert splits.scene_test.sum() == 10 * len(L.ZENITHS)

    train_scenes = set(l23_small_batch.scene[splits.scene_train].tolist())
    test_scenes = set(l23_small_batch.scene[splits.scene_test].tolist())
    assert train_scenes.isdisjoint(test_scenes)

    held = L.select(l23_small_batch, splits.zenith_test)
    assert set(held.zenith.tolist()) == {60.0}
    held.validate()


def test_fixture_iops_are_identical_across_zeniths(l23_small_batch):
    """The zenith-invariance claim, checkable without the dataset.

    The full-release version of this test is the authority; this one makes the
    same claim fail loudly in CI, where the netCDFs are absent. Added with the
    PR #9 review fix, on the principle that a claim worth asserting is worth
    asserting where the suite actually runs.
    """
    fields = {
        "a": l23_small_batch.iops.a,
        "bb_w": l23_small_batch.iops.bb_w,
        "bb_p": l23_small_batch.iops.bb_p,
        "B_p": l23_small_batch.phase_params.B_p,
    }
    n_scene = l23_small_batch.n_sample // len(L.ZENITHS)

    for name, values in fields.items():
        stacked = np.asarray(values).reshape(len(L.ZENITHS), n_scene, C.N_WAVE)
        for zi, zenith in enumerate(L.ZENITHS[1:], start=1):
            np.testing.assert_array_equal(
                stacked[0],
                stacked[zi],
                err_msg=f"{name} differs between {L.ZENITHS[0]} deg and {zenith} deg",
            )


def test_npz_reader_refuses_the_wrong_scenario():
    """Serving the wrong angle or scenario silently would be far worse."""
    from robust.tests.conftest import L23_SMALL_FIXTURE

    read = L.npz_reader(L23_SMALL_FIXTURE)

    with pytest.raises(ValueError, match="X="):
        read(4, 0)
    with pytest.raises(ValueError, match="zeniths"):
        read(L.ELASTIC_X, 45)


def test_cached_fixture_stays_small():
    """A fixture is only worth committing while it is small.

    Guards against the file quietly growing into a data dump in git history.
    """
    from robust.tests.conftest import L23_SMALL_FIXTURE

    assert L23_SMALL_FIXTURE.stat().st_size < 512 * 1024


# ------------------------------------------------------- the reference data --


@needs_l23
def test_batch_shapes_and_grid(l23_batch):
    """The gate's shape requirement: 3320 scenes x 81 bands, three zeniths."""
    assert l23_batch.n_sample == len(L.ZENITHS) * L.N_SCENES
    assert l23_batch.n_wave == C.N_WAVE
    assert l23_batch.iops.a.shape == (len(L.ZENITHS) * L.N_SCENES, C.N_WAVE)
    C.check_wave(np.asarray(l23_batch.wave))
    np.testing.assert_array_equal(np.unique(l23_batch.zenith), [0.0, 30.0, 60.0])


@needs_l23
def test_iops_and_rrs_are_physical(l23_batch):
    """``a``, ``bb`` >= 0 and ``Rrs`` > 0 -- the rest of the gate."""
    l23_batch.validate()  # raises on any non-finite/negative IOP or Rrs

    assert float(jnp.min(l23_batch.iops.a)) >= 0.0
    assert float(jnp.min(l23_batch.iops.bb)) >= 0.0
    assert float(jnp.min(l23_batch.iops.bb_p)) >= 0.0
    assert float(jnp.min(l23_batch.Rrs)) > 0.0


@needs_l23
def test_B_p_is_within_the_design_range_at_every_wavelength(l23_batch):
    """The measurement prompt 2 asked for: all 81 bands, all three zeniths.

    No failure in the UV or the far red -- 100% of the values sit inside the
    design's nominal band, so the range assert needs no wavelength-dependent
    escape hatch.
    """
    B_p = np.asarray(l23_batch.phase_params.B_p)
    lo, hi = L.B_P_EXPECTED

    assert B_p.shape == (len(L.ZENITHS) * L.N_SCENES, C.N_WAVE)
    assert np.all(np.isfinite(B_p))
    assert B_p.min() >= lo, f"B_p dips to {B_p.min():.5g} below {lo}"
    assert B_p.max() <= hi, f"B_p reaches {B_p.max():.5g} above {hi}"

    # Per-band, not just globally: a band-specific excursion would hide in the
    # global min/max if it were small.
    per_band_min, per_band_max = B_p.min(axis=0), B_p.max(axis=0)
    assert np.all(per_band_min >= lo)
    assert np.all(per_band_max <= hi)


@needs_l23
def test_B_p_observed_range_matches_what_was_recorded(l23_batch):
    """Pins the narrow slice of phase-function space L23 actually covers.

    Recorded because it bounds what "explicit phase-function dependence" can mean
    before M5: a factor ~1.75 in ``B_p``, where the design's nominal band spans
    ~7. If a future release widened it, this failure is the good news.
    """
    B_p = np.asarray(l23_batch.phase_params.B_p)

    assert B_p.min() == pytest.approx(0.01026, abs=1e-5)
    assert B_p.max() == pytest.approx(0.01800, abs=1e-5)


@needs_l23
def test_B_p_varies_with_wavelength(l23_batch):
    """So it is carried as a spectrum, not reduced to a per-scene scalar."""
    B_p = np.asarray(l23_batch.phase_params.B_p)

    assert not np.allclose(B_p, B_p[:, :1], rtol=1e-4)


@needs_l23
def test_loader_bb_w_agrees_with_the_conventions_table(l23_batch):
    """Closes the loop between the loader and :mod:`robust.rt.conventions`.

    The loader takes ``bb_w`` from the file (``bb - bbnw``); ``conventions`` ships
    the same quantity as an embedded table. If they ever diverged, ``bb_p`` would
    mean something different in the two places.

    The tolerance is 1e-5, not 1e-6, and the reason is the data rather than the
    code: L23 stores float32, and the two paths differ in *where* the subtraction
    happens. The table was extracted with float32 arithmetic (exactly what the
    file's own dtype gives); the loader upcasts to float64 first, which is
    slightly more accurate. Where ``bb_w`` is smallest -- the red tail, ~3e-4 --
    that shows up as up to 3.4e-6 relative disagreement on ~0.1% of elements.
    Both values are well inside the reference data's own precision, so the
    tolerance follows float32 rather than pretending to a precision the netCDF
    does not carry.
    """
    bb_w = np.asarray(l23_batch.iops.bb_w)

    np.testing.assert_allclose(bb_w, np.broadcast_to(C.BB_W_L23, bb_w.shape), rtol=1e-5)


@needs_l23
def test_golden_row_matches_the_raw_netcdf():
    """Golden value cross-checked against the file, not against a copied number.

    Re-reads the netCDF and compares the assembled batch row by row, so a
    transposed axis, an off-by-one in the scene labels, or a mis-ordered zenith
    concatenation shows up here.
    """
    from ocpy.hydrolight import loisel23

    batch = L.load_batch(scenes=slice(0, 3))
    wave = np.asarray(batch.wave)
    i440 = int(np.argmin(np.abs(wave - 440.0)))

    for zi, zenith in enumerate(L.ZENITHS):
        ds = loisel23.load_ds(L.ELASTIC_X, zenith)
        for scene in range(3):
            row = zi * 3 + scene
            assert batch.scene[row] == scene
            assert batch.zenith[row] == float(zenith)
            assert float(batch.Rrs[row, i440]) == pytest.approx(
                float(ds.Rrs.data[scene, i440]), rel=1e-6
            )
            assert float(batch.iops.a[row, i440]) == pytest.approx(
                float(ds.a.data[scene, i440]), rel=1e-6
            )
            assert float(batch.iops.bb[row, i440]) == pytest.approx(
                float(ds.bb.data[scene, i440]), rel=1e-6
            )
            assert float(batch.iops.bb_p[row, i440]) == pytest.approx(
                float(ds.bbnw.data[scene, i440]), rel=1e-6
            )


@needs_l23
def test_golden_values_are_pinned_absolutely():
    """A second, independent pin: the numbers themselves.

    The cross-check above would still pass if the underlying release changed.
    These constants were read from the files on 2026-08-01; a failure here means
    the reference data moved, which is worth knowing loudly.
    """
    batch = L.load_batch(zeniths=(0,), scenes=slice(0, 1))
    i440 = int(np.argmin(np.abs(np.asarray(batch.wave) - 440.0)))

    assert float(batch.Rrs[0, i440]) == pytest.approx(8.53329990e-03, rel=1e-6)
    assert float(batch.iops.a[0, i440]) == pytest.approx(1.66950002e-02, rel=1e-6)
    assert float(batch.iops.bb[0, i440]) == pytest.approx(2.91789998e-03, rel=1e-6)
    assert float(batch.phase_params.B_p[0, i440]) == pytest.approx(
        7.22300028e-04 / 5.46935983e-02, rel=1e-6
    )


@needs_l23
def test_iops_are_identical_across_zeniths(l23_batch):
    """An observed property of the release, not something the loader assumes.

    The same 3320 water bodies are illuminated three ways, so only ``Rrs``
    differs. The loader reads each file's own IOPs anyway -- this test is what
    would tell us if that ever stopped being true.

    Every field is compared against every other zenith rather than a hand-picked
    pair or two. The first version of this test checked ``a`` at all three angles
    but ``bb_p`` only at 0 deg and 60 deg, so a 30 deg mismatch could not have
    failed it -- caught in review (PR #9). Looping over the fields removes the
    class of omission, not just the instance.
    """
    fields = {
        "a": l23_batch.iops.a,
        "bb_w": l23_batch.iops.bb_w,
        "bb_p": l23_batch.iops.bb_p,
        "B_p": l23_batch.phase_params.B_p,
    }

    for name, values in fields.items():
        stacked = np.asarray(values).reshape(len(L.ZENITHS), L.N_SCENES, C.N_WAVE)
        for zi, zenith in enumerate(L.ZENITHS[1:], start=1):
            np.testing.assert_array_equal(
                stacked[0],
                stacked[zi],
                err_msg=f"{name} differs between {L.ZENITHS[0]} deg and {zenith} deg",
            )


@needs_l23
def test_rrs_falls_with_solar_zenith(l23_batch):
    """The only geometry signal in hand -- and the one Gordon cannot express.

    Standard Gordon has no solar-zenith dependence at all, so this ~5% effect at
    60 deg is what the M3/M4 comparison is built to exploit.
    """
    Rrs = np.asarray(l23_batch.Rrs).reshape(len(L.ZENITHS), L.N_SCENES, C.N_WAVE)

    ratio_30 = np.median(Rrs[1] / Rrs[0])
    ratio_60 = np.median(Rrs[2] / Rrs[0])

    assert ratio_30 == pytest.approx(0.990, abs=0.005)
    assert ratio_60 == pytest.approx(0.949, abs=0.005)
    assert ratio_60 < ratio_30 < 1.0


@needs_l23
def test_real_data_raises_no_B_p_warning():
    """The reference data is inside the expected band, so loading is silent."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        L.load_batch(scenes=slice(0, 50))


@needs_l23
def test_scenes_subset_keeps_labels_and_shapes():
    batch = L.load_batch(scenes=slice(10, 20))

    assert batch.n_sample == 3 * 10
    np.testing.assert_array_equal(batch.scene[:10], np.arange(10, 20))


@needs_l23
def test_splits_on_the_real_batch(l23_batch):
    """The two M4 gates, sized on the real dataset."""
    splits = L.make_splits(l23_batch)

    assert splits.test_scenes.size == round(L.TEST_FRACTION * L.N_SCENES)
    assert splits.scene_test.sum() == splits.test_scenes.size * len(L.ZENITHS)
    assert splits.zenith_test.sum() == L.N_SCENES
    assert set(l23_batch.zenith[splits.zenith_train].tolist()) == {0.0, 30.0}


@needs_l23
def test_rejects_a_nonelastic_scenario_only_by_documentation(l23_batch):
    """X is a parameter, and the default is the elastic set.

    Guards against a silent switch: the batch the prototype trains on must be
    X=1, since the model represents no inelastic process.
    """
    assert L.ELASTIC_X == 1


@needs_l23
def test_write_fixture_reproduces_the_committed_snapshot(tmp_path):
    """Regenerating the fixture gives back the committed bytes, and verifies itself.

    Two things at once. **Reproducibility**: the committed
    ``robust/tests/files/l23_small.npz`` is what the whole suite runs on without
    ``$OS_COLOR``, so it has to be exactly what the current code would write --
    otherwise the fixture and the loader have silently drifted apart.

    **The write is validated before it lands.** ``write_fixture`` snapshots to a
    temporary file, loads it back through :func:`npz_reader` and
    :func:`load_batch`, and only then replaces the destination. Writing first and
    checking afterwards is the defect class reported against
    ``design/py/train_emulator.py`` in PR #11; the destination here is a committed
    file that CI depends on, so it got the same treatment. This test pins that the
    happy path is byte-exact; the guard itself was demonstrated by forcing the
    verification to fail and confirming the destination survived untouched.
    """
    from robust.tests.conftest import L23_SMALL_FIXTURE

    out = tmp_path / "l23_small.npz"
    L.write_fixture(out)

    committed = np.load(L23_SMALL_FIXTURE)
    regenerated = np.load(out)
    assert set(regenerated.files) == set(committed.files)
    for key in committed.files:
        np.testing.assert_array_equal(
            regenerated[key],
            committed[key],
            err_msg=f"{key} differs from the committed fixture",
        )

    # And it loads to the batch the fixture-based tests expect.
    batch = L.load_batch(reader=L.npz_reader(out))
    batch.validate()
    assert batch.n_sample == len(L.ZENITHS) * 50
