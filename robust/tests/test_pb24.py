"""
Tests for :mod:`robust.rt.data.pb24` (M5 task 4).

Five things are being defended.

**The loader runs on real numbers without the dataset.** The committed fixture
stores the loader's *input* -- three whole realisations, all 1300 geometries each
-- so ``load_batch`` genuinely executes in CI. Same seam as L23's, for the same
reason: a snapshot of the loader's *output* could only ever prove itself
unchanged.

**Golden values against the raw netCDF.** Named realisations at named geometries,
so the flattening order, the IOP sums and the ``mu`` gather are checked against
the source rather than against themselves. Needs ``$OS_COLOR``.

**Nothing is dropped silently.** The angle window, the stride and the zero-``rrs``
filter each report a count, and the tests assert those counts against
independently computed truth. The zero-``rrs`` assertions run on the **shell**
load, because inside the Q14 window that filter removes exactly nothing and
asserting its count there would pass vacuously -- the failure mode M4's review
found twice.

**The stride cannot alias unnoticed.** Flattening ``(theta_s, theta, phi)`` in C
order puts the 13 azimuths innermost, so ``geometry_stride=13`` keeps one azimuth
and silently deletes the BRDF axis this milestone exists to study. Found by
running the loader, not by reading it.

**The grid is identical across files**, and the loader stops rather than
concatenating two different angle grids into one mislabelled batch.
"""

from __future__ import annotations

import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt.data import pb24 as P
from robust.tests.conftest import (
    PB24_FIXTURE_REALISATIONS,
    PB24_SMALL_FIXTURE,
    needs_pb24,
)

#: Geometry counts on PB24's grid: 10 x 10 x 13.
N_GEOMETRY = 1300

#: Inside the Q14 window, theta_s and theta_v each keep 8 of 10 values.
N_WINDOW = 8 * 8 * 13

#: The realisation carrying the only zero-``rrs`` values in the OLCI set.
ZERO_REALISATION = 993


@pytest.fixture(scope="module")
def shell_batch(pb24_reader):
    """The extrapolation shell, unfiltered -- where the zero ``rrs`` values live."""
    return P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS,
        angles="shell",
        drop_zero_rrs=False,
        reader=pb24_reader,
    )


# ------------------------------------------------------------------- shapes --


def test_window_load_has_the_expected_shape(pb24_small_batch):
    """3 realisations x 832 in-window geometries, 12 OLCI bands."""
    batch = pb24_small_batch

    assert batch.n_wave == 12
    assert batch.n_sample == len(PB24_FIXTURE_REALISATIONS) * N_WINDOW
    assert batch.iops.a.shape == (batch.n_sample, 12)
    assert batch.rrs.shape == batch.Rrs.shape == batch.iops.a.shape
    C.check_wave(np.asarray(batch.wave), grid="olci")


def test_window_and_shell_partition_the_grid(
    pb24_reader, shell_batch, pb24_small_batch
):
    """Window + shell = all, with nothing counted twice."""
    assert pb24_small_batch.report.n_geometry == N_WINDOW
    assert shell_batch.report.n_geometry == N_GEOMETRY - N_WINDOW

    everything = P.load_batch(
        realisations=(1,), angles="all", drop_zero_rrs=False, reader=pb24_reader
    )

    assert everything.report.n_geometry == N_GEOMETRY
    assert everything.report.n_dropped_angle == 0


def test_the_window_is_q14s_envelope(pb24_small_batch):
    """Both zeniths reach 70 degrees and stop there; all azimuths are present."""
    batch = pb24_small_batch

    assert batch.theta_s.max() == P.ANGLE_WINDOW_MAX == 70.0
    assert batch.theta_v.max() == P.ANGLE_WINDOW_MAX
    assert sorted(set(batch.theta_s.tolist())) == [0, 10, 20, 30, 40, 50, 60, 70]
    assert len(set(batch.dphi.tolist())) == 13


def test_the_shell_is_the_extrapolation_set(shell_batch):
    """Every shell sample has at least one angle past the window."""
    beyond = (shell_batch.theta_s > P.ANGLE_WINDOW_MAX) | (
        shell_batch.theta_v > P.ANGLE_WINDOW_MAX
    )

    assert bool(beyond.all())
    assert shell_batch.theta_s.max() == pytest.approx(87.75)
    assert shell_batch.theta_v.max() == pytest.approx(87.5)


# --------------------------------------------------- nothing dropped silently -


def test_the_angle_window_reports_the_count_it_removed(pb24_small_batch):
    """The report is checkable arithmetic, not a label."""
    report = pb24_small_batch.report

    assert report.n_dropped_angle == N_GEOMETRY - N_WINDOW == 468
    assert report.n_geometry + report.n_dropped_angle == report.n_geometry_available


def test_the_stride_reports_what_it_dropped(pb24_reader):
    """A subsample is visible at the call site, not implied."""
    batch = P.load_batch(realisations=(1,), geometry_stride=3, reader=pb24_reader)
    report = batch.report

    kept = len(range(0, N_WINDOW, 3))
    assert report.n_geometry == kept
    assert report.n_dropped_stride == N_WINDOW - kept
    assert batch.n_sample == kept
    assert "stride=3" in report.summary()


def test_zero_rrs_is_filtered_and_counted_on_the_shell(pb24_reader, shell_batch):
    """The gate that would have been vacuous inside the window.

    Inside Q14's window the filter removes exactly zero samples, so asserting its
    count there proves nothing. On the shell it has real work: realisation 993
    carries two exactly-zero ``rrs`` values, both at 753 nm.
    """
    unfiltered = shell_batch
    filtered = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS,
        angles="shell",
        drop_zero_rrs=True,
        reader=pb24_reader,
    )

    assert unfiltered.report.n_zero_bands == 2
    assert unfiltered.report.n_dropped_zero_rrs == 0  # not asked to drop
    assert filtered.report.n_dropped_zero_rrs == 2
    assert filtered.n_sample == unfiltered.n_sample - 2
    assert not bool((np.asarray(filtered.rrs) <= 0.0).any())


def test_the_zero_rrs_filter_is_inert_inside_the_window(pb24_small_batch):
    """Stated so the previous test's choice of the shell is not folklore."""
    assert pb24_small_batch.report.n_zero_bands == 0
    assert pb24_small_batch.report.n_dropped_zero_rrs == 0


def test_dropping_whole_spectra_has_a_visible_cost(pb24_reader):
    """Eleven good bands go with each bad one, and the report shows it."""
    filtered = P.load_batch(
        realisations=(ZERO_REALISATION,), angles="shell", reader=pb24_reader
    )
    report = filtered.report

    # 2 zero values, 2 spectra removed, so 2 * (12 - 1) = 22 good bands lost.
    assert report.n_zero_bands == 2
    assert report.n_dropped_zero_rrs == 2
    lost_good = report.n_dropped_zero_rrs * 12 - report.n_zero_bands
    assert lost_good == 22


# ------------------------------------------------------------ stride aliasing -


def test_a_stride_that_aliases_the_azimuth_axis_warns(pb24_reader):
    """13 azimuths innermost: stride 13 keeps one azimuth and no BRDF at all."""
    with pytest.warns(UserWarning, match="loses whole angle values"):
        batch = P.load_batch(realisations=(1,), geometry_stride=13, reader=pb24_reader)

    assert batch.report.aliased_axes == ("dphi",)
    assert len(set(batch.dphi.tolist())) == 1
    assert "ALIASED" in batch.report.summary()


def test_a_coprime_stride_keeps_every_angle(pb24_reader):
    """The healthy case, so the warning above is not merely always-on."""
    batch = P.load_batch(realisations=(1,), geometry_stride=7, reader=pb24_reader)

    assert batch.report.aliased_axes == ()
    assert batch.report.coverage == {
        "theta_s": (8, 8),
        "theta_v": (8, 8),
        "dphi": (13, 13),
    }


# ------------------------------------------------------------------ contents --


def test_iops_are_component_sums_not_the_files_totals(pb24_small_batch):
    """``a`` and ``bb_p`` are built from components; ``bb_w`` is the file's own."""
    batch = pb24_small_batch

    assert float(batch.iops.a.min()) > 0.0
    assert float(batch.iops.bb_p.min()) > 0.0
    # bb_w must be PB24's own water, not conventions.bb_w (which is L23's).
    ours = np.asarray(C.bb_w(batch.wave))
    theirs = np.asarray(batch.iops.bb_w[0])
    assert not np.allclose(ours, theirs, rtol=1e-3)


def test_B_p_spans_more_than_l23_and_varies_with_wavelength(pb24_small_batch):
    """The property that makes a held-out-``B_p`` split possible."""
    B_p = np.asarray(pb24_small_batch.phase_params.B_p)

    assert B_p.min() > 0.0
    within = B_p.max(axis=-1) / B_p.min(axis=-1)
    assert within.max() > 1.001  # not flat in wavelength, unlike each component


def test_extras_are_gathered_at_each_samples_solar_zenith(pb24_small_batch):
    """``mu_d`` is tabulated per zenith; every sample must get its own."""
    batch = pb24_small_batch

    assert set(batch.aops) == {"Q", "mu_d", "mu_u", "mu_tot"}
    for name, values in batch.aops.items():
        assert values.shape == batch.rrs.shape, name

    mu_d = np.asarray(batch.aops["mu_d"])
    first = batch.realisation == PB24_FIXTURE_REALISATIONS[0]

    # Within one realisation mu_d depends on theta_s and nothing else, so the
    # gather must produce exactly one distinct value per zenith -- 8 in the
    # window. A broadcast bug would give 1; a mis-indexed gather would give more.
    distinct = np.unique(np.round(mu_d[first][:, 0], 12))
    assert distinct.size == 8

    # And it falls as the sun moves off zenith.
    at_0 = mu_d[first & (batch.theta_s == 0.0)][:, 0]
    at_70 = mu_d[first & (batch.theta_s == 70.0)][:, 0]
    assert at_0.mean() > at_70.mean()


def test_extras_can_be_requested_and_are_validated(pb24_reader):
    """Opt-in K's work; an unknown name fails rather than being ignored."""
    batch = P.load_batch(
        realisations=(1,), extras=("mu_d", "Kd"), geometry_stride=7, reader=pb24_reader
    )

    assert set(batch.aops) == {"Q", "mu_d", "Kd"}

    with pytest.raises(ValueError, match="unknown extras"):
        P.load_batch(realisations=(1,), extras=("mu_x",), reader=pb24_reader)


def test_water_classes_are_opt_in(pb24_reader):
    """The default attaches none, so a fixture load is environment-independent."""
    default = P.load_batch(realisations=(1,), geometry_stride=7, reader=pb24_reader)
    explicit = P.load_batch(
        realisations=(1,),
        geometry_stride=7,
        water_classes=np.full(P.N_REALISATION, 7),
        reader=pb24_reader,
    )

    assert default.water_class is None
    assert explicit.water_class is not None
    assert set(explicit.water_class.tolist()) == {7}

    with pytest.raises(ValueError, match="must be an array, None or 'auto'"):
        P.load_batch(realisations=(1,), water_classes="all", reader=pb24_reader)


# -------------------------------------------------------------- the API edges -


def test_select_subsets_every_leaf(pb24_small_batch):
    """Including the extras, which a naive implementation forgets."""
    batch = pb24_small_batch
    mask = batch.theta_v == 0.0

    subset = P.select(batch, mask)

    assert subset.n_sample == int(mask.sum())
    assert subset.iops.a.shape[0] == subset.n_sample
    assert subset.realisation.shape == (subset.n_sample,)
    for name, values in subset.aops.items():
        assert values.shape[0] == subset.n_sample, name
    assert bool((np.asarray(subset.theta_v) == 0.0).all())


def test_select_rejects_a_wrong_mask(pb24_small_batch):
    with pytest.raises(ValueError, match="boolean mask"):
        P.select(pb24_small_batch, np.ones(3, dtype=bool))


def test_load_batch_rejects_nonsense_arguments(pb24_reader):
    """Every knob fails loudly rather than guessing."""
    with pytest.raises(ValueError, match="angles must be one of"):
        P.load_batch(realisations=(1,), angles="everything", reader=pb24_reader)
    with pytest.raises(ValueError, match="geometry_stride must be"):
        P.load_batch(realisations=(1,), geometry_stride=0, reader=pb24_reader)
    with pytest.raises(ValueError, match="selects nothing"):
        P.load_batch(realisations=(), reader=pb24_reader)


def test_npz_reader_refuses_what_it_does_not_hold():
    """Serving the wrong realisation silently would be far worse than failing."""
    read = P.npz_reader(PB24_SMALL_FIXTURE)

    with pytest.raises(ValueError, match="was asked for 7"):
        read(7)
    with pytest.raises(ValueError, match="resolution"):
        read(1, "hyp")


def test_file_path_rejects_an_out_of_range_index():
    with pytest.raises(ValueError, match=r"outside \[1, 5000\]"):
        P.file_path(0)
    with pytest.raises(ValueError, match="unknown resolution"):
        P.file_path(1, "hyperspectral")


# ------------------------------------------------------- against the raw data -


@needs_pb24
def test_golden_values_against_the_raw_netcdf():
    """Named realisations at named geometries, checked against the source.

    This is what pins the flattening order. A transposed grid would still load,
    still validate, and still produce plausible numbers -- it would simply label
    every sample with the wrong geometry.
    """
    import xarray as xr

    for index in (1, 2500):
        batch = P.load_batch(realisations=(index,), angles="all", drop_zero_rrs=False)
        with xr.open_dataset(P.file_path(index)) as ds:
            ts = np.asarray(ds["theta_s"].values, dtype=float)
            tv = np.asarray(ds["theta"].values, dtype=float)
            ph = np.asarray(ds["phi"].values, dtype=float)
            for i_ts, i_tv, i_ph, i_w in ((0, 0, 0, 2), (3, 5, 7, 0), (9, 9, 12, 11)):
                truth = float(
                    ds["rrs"].isel(
                        theta_s=i_ts, theta=i_tv, phi=i_ph, **{"lambda": i_w}
                    )
                )
                flat = (i_ts * tv.size + i_tv) * ph.size + i_ph
                got = float(batch.rrs[flat, i_w])

                assert got == pytest.approx(truth, rel=1e-6)
                assert batch.theta_s[flat] == pytest.approx(ts[i_ts])
                assert batch.theta_v[flat] == pytest.approx(tv[i_tv])
                assert batch.dphi[flat] == pytest.approx(ph[i_ph])


@needs_pb24
def test_golden_iops_and_mu_against_the_raw_netcdf():
    """The component sums and the per-zenith gather, against the source."""
    import xarray as xr

    index = 2500
    batch = P.load_batch(realisations=(index,), angles="all", drop_zero_rrs=False)
    with xr.open_dataset(P.file_path(index)) as ds:
        i_w = 4
        a_truth = float(
            ds["aw"][i_w] + ds["aph"][i_w] + ds["ag"][i_w] + ds["aNAP"][i_w]
        )
        bbp_truth = float(ds["bbph"][i_w] + ds["bbNAP"][i_w])
        bp_truth = float(ds["bph"][i_w] + ds["bNAP"][i_w])

        assert float(batch.iops.a[0, i_w]) == pytest.approx(a_truth, rel=1e-6)
        assert float(batch.iops.bb_p[0, i_w]) == pytest.approx(bbp_truth, rel=1e-6)
        assert float(batch.iops.bb_w[0, i_w]) == pytest.approx(
            float(ds["bbw"][i_w]), rel=1e-6
        )
        assert float(batch.phase_params.B_p[0, i_w]) == pytest.approx(
            bbp_truth / bp_truth, rel=1e-6
        )

        # mu_d is (lambda, theta_s): sample 0 is theta_s index 0, and the last
        # geometry of the grid is theta_s index 9.
        assert float(batch.aops["mu_d"][0, i_w]) == pytest.approx(
            float(ds["mu_d"][i_w, 0]), rel=1e-6
        )
        assert float(batch.aops["mu_d"][-1, i_w]) == pytest.approx(
            float(ds["mu_d"][i_w, 9]), rel=1e-6
        )


@needs_pb24
def test_the_angle_grid_is_identical_across_files():
    """Asserted, not assumed -- the loader relies on it to label samples."""
    import xarray as xr

    reference = None
    for index in (1, 993, 2500, 5000):
        with xr.open_dataset(P.file_path(index)) as ds:
            grids = tuple(
                np.asarray(ds[name].values, dtype=float)
                for name in ("lambda", "theta_s", "theta", "phi")
            )
        if reference is None:
            reference = grids
        else:
            for name, got, want in zip(
                ("lambda", "theta_s", "theta", "phi"), grids, reference, strict=True
            ):
                assert np.array_equal(got, want), (
                    f"{name} differs at realisation {index}"
                )


@needs_pb24
def test_loader_stops_on_a_mismatched_grid(pb24_reader):
    """A batch spanning two angle grids would be silently mislabelled."""
    good = P.npz_reader(PB24_SMALL_FIXTURE)

    def bad(index, resolution="olci"):
        raw = dict(good(index, resolution))
        if index != PB24_FIXTURE_REALISATIONS[0]:
            raw["phi"] = raw["phi"] + 1.0
        return raw

    with pytest.raises(ValueError, match="different 'phi' grid"):
        P.load_batch(realisations=PB24_FIXTURE_REALISATIONS, reader=bad)


@needs_pb24
def test_the_committed_fixture_reproduces_the_snapshot(tmp_path):
    """Regenerating the fixture gives the same bytes back.

    The same gate L23's fixture carries: if this drifts, every CI number computed
    from the fixture drifted with it and nothing else would have said so.
    """
    regenerated = tmp_path / "pb24_small.npz"
    P.write_fixture(regenerated, realisations=PB24_FIXTURE_REALISATIONS)

    fresh = np.load(regenerated)
    committed = np.load(PB24_SMALL_FIXTURE)

    assert sorted(fresh.files) == sorted(committed.files)
    for name in committed.files:
        np.testing.assert_array_equal(
            fresh[name], committed[name], err_msg=f"fixture field {name} drifted"
        )


@needs_pb24
def test_read_classes_matches_the_sidecar():
    """12 unbalanced classes, one per realisation."""
    classes = P.read_classes()

    assert classes.shape == (P.N_REALISATION,)
    assert set(np.unique(classes).tolist()) == set(range(1, P.N_CLASS + 1))
    counts = np.bincount(classes)[1:]
    assert counts.min() < counts.max() / 5  # very unbalanced, as recorded


def test_data_dir_explains_itself_without_os_color(monkeypatch):
    """A missing mount must say so, not raise a KeyError three frames down."""
    monkeypatch.delenv("OS_COLOR", raising=False)

    with pytest.raises(RuntimeError, match=r"\$OS_COLOR is not set"):
        P.data_dir()


# ------------------------------------------------------------------- splits --
# M5 task 5. The gate is mostly about failure modes that do not look like
# failures: an empty side scores nothing and reports success, and a `B_p` split
# quietly moves the water type along with the phase function.


def test_every_split_is_disjoint_and_exhaustive(pb24_reader):
    """Train and test partition the sample axis, for every kind."""
    batch = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS, angles="all", reader=pb24_reader
    )
    splits = P.make_splits(batch, kinds=P.SPLIT_KINDS)

    for kind in P.SPLIT_KINDS:
        train, test = splits.train(kind), splits.test(kind)

        assert train.shape == test.shape == (batch.n_sample,)
        assert not bool((train & test).any()), kind
        assert bool((train | test).all()), kind
        assert int(train.sum()) + int(test.sum()) == batch.n_sample, kind


def test_every_split_has_a_non_empty_test_set(pb24_reader):
    """The assertion the gate exists for: an empty side is a silent pass."""
    batch = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS, angles="all", reader=pb24_reader
    )
    splits = P.make_splits(batch, kinds=P.SPLIT_KINDS)

    for kind in P.SPLIT_KINDS:
        assert splits.reports[kind].n_test > 0, kind
        assert splits.reports[kind].n_train > 0, kind


def test_splits_are_deterministic_and_seed_dependent(pb24_small_batch):
    """Same seed, same masks; and the seed genuinely reaches the draw.

    The second half needs care with three realisations: two seeds can easily
    agree by chance, so asserting ``seed=7 != seed=8`` would be flaky. Asserting
    that the seed *field* differs would be worse -- it cannot fail. So: sweep
    seeds and require the draw to land on more than one realisation.
    """
    a = P.make_splits(pb24_small_batch, kinds=("realisation",), seed=7)
    b = P.make_splits(pb24_small_batch, kinds=("realisation",), seed=7)

    np.testing.assert_array_equal(a.test("realisation"), b.test("realisation"))
    np.testing.assert_array_equal(a.train("realisation"), b.train("realisation"))

    held = set()
    for seed in range(10):
        splits = P.make_splits(pb24_small_batch, kinds=("realisation",), seed=seed)
        held.add(
            tuple(
                sorted(
                    set(
                        pb24_small_batch.realisation[
                            splits.test("realisation")
                        ].tolist()
                    )
                )
            )
        )

    assert len(held) > 1, f"the seed never changed the draw: {held}"


def test_realisation_split_moves_whole_water_bodies(pb24_small_batch):
    """Splitting per sample would leak: one realisation appears 832 times."""
    splits = P.make_splits(pb24_small_batch, kinds=("realisation",))
    train, test = splits.train("realisation"), splits.test("realisation")

    on_train = set(pb24_small_batch.realisation[train].tolist())
    on_test = set(pb24_small_batch.realisation[test].tolist())

    assert on_train.isdisjoint(on_test)
    assert on_train | on_test == set(PB24_FIXTURE_REALISATIONS)


def test_bp_band_holds_out_an_interior_band(pb24_small_batch):
    """Train is both tails, so the split tests interpolation, not extrapolation."""
    report = P.make_splits(pb24_small_batch, kinds=("bp_band",)).reports["bp_band"]
    detail = report.detail

    assert detail["B_p_train_lo"] < detail["B_p_band_lo"]
    assert detail["B_p_train_hi"] > detail["B_p_band_hi"]
    # and the separation is clean: no training realisation inside the held band
    assert detail["n_train_inside_band"] == 0.0


def test_bp_band_reports_a_confound_that_is_not_decorative(pb24_small_batch):
    """The gate: the confound must be visible in the artefact, and be real."""
    report = P.make_splits(pb24_small_batch, kinds=("bp_band",)).reports["bp_band"]

    assert set(report.confound) >= {"C", "N", "Y", "a_mean", "bb_p_mean", "B_p_mean"}
    assert all(np.isfinite(v) and v > 0 for v in report.confound.values())

    # It must actually measure something -- a report of all ones would mean the
    # computation never ran against real data.
    assert any(abs(v - 1.0) > 0.05 for v in report.confound.values())
    assert "side-effect" in report.summary()


def test_confound_reference_is_a_band_not_a_point(pb24_reader):
    """A random hold-out already moves heavy-tailed labels; that is the yardstick."""
    batch = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS, geometry_stride=7, reader=pb24_reader
    )
    reference = P.confound_reference(batch, seeds=range(5))

    assert set(reference) >= {"C", "B_p_mean"}
    for name, (median, lo, hi) in reference.items():
        assert lo <= median <= hi, name


def test_geometry_split_refuses_a_window_only_batch(pb24_small_batch):
    """The interaction that would have made this split score nothing at all."""
    with pytest.raises(ValueError, match="leaves its test side empty"):
        P.make_splits(pb24_small_batch, kinds=("geometry",))

    with pytest.raises(ValueError, match="angles='all'"):
        P.make_splits(pb24_small_batch, kinds=("geometry",))


def test_geometry_split_tests_only_beyond_the_window(pb24_reader):
    """Train stops at 70 degrees; test starts at 80."""
    batch = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS, angles="all", reader=pb24_reader
    )
    splits = P.make_splits(batch, kinds=("geometry",))
    train, test = splits.train("geometry"), splits.test("geometry")
    worst = np.maximum(batch.theta_s, batch.theta_v)

    assert worst[train].max() == P.ANGLE_WINDOW_MAX == 70.0
    assert worst[test].min() == 80.0
    assert splits.reports["geometry"].detail["worst_angle_max_test"] == pytest.approx(
        87.75
    )


def test_geometry_split_divides_angles_not_water_bodies(pb24_reader):
    """Every realisation appears on both sides -- that is the point of it."""
    batch = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS, angles="all", reader=pb24_reader
    )
    report = P.make_splits(batch, kinds=("geometry",)).reports["geometry"]

    assert report.n_train_realisation == report.n_test_realisation
    assert report.n_train_realisation == len(PB24_FIXTURE_REALISATIONS)
    # and so it moves no IOP statistic at all
    assert all(v == pytest.approx(1.0) for v in report.confound.values())


def test_make_splits_rejects_nonsense_arguments(pb24_small_batch):
    with pytest.raises(ValueError, match="unknown split kind"):
        P.make_splits(pb24_small_batch, kinds=("realization",))
    with pytest.raises(ValueError, match="selects nothing"):
        P.make_splits(pb24_small_batch, kinds=())
    with pytest.raises(ValueError, match="test_fraction"):
        P.make_splits(pb24_small_batch, test_fraction=0.0)
    with pytest.raises(ValueError, match="bp_quantiles"):
        P.make_splits(pb24_small_batch, bp_quantiles=(0.6, 0.4))


def test_default_kinds_are_the_ones_a_window_load_supports(pb24_small_batch):
    """So the default never raises on the default load."""
    splits = P.make_splits(pb24_small_batch)

    assert tuple(splits.masks) == P.DEFAULT_SPLIT_KINDS
    assert "geometry" not in splits.masks


@needs_pb24
def test_the_bp_confound_is_real_at_scale():
    """The correlation this split has to disclose, measured on real numbers.

    Task 0's audit reported -0.65 for ``bbph/bph`` against chlorophyll; this is
    the *bulk* ``B_p``, which is the quantity the split actually uses, and it is
    weaker. Pinned here so the documented number and the split agree.
    """
    batch = P.load_batch(realisations=200, geometry_stride=11)
    realisations = np.unique(batch.realisation)
    B_p = np.asarray(batch.phase_params.B_p).mean(axis=-1)
    per_bp = np.array([B_p[batch.realisation == i].mean() for i in realisations])
    per_c = np.array(
        [batch.labels["C"][batch.realisation == i][0] for i in realisations]
    )

    corr = float(np.corrcoef(np.log(per_bp), np.log(per_c))[0, 1])

    assert -0.75 < corr < -0.3, f"corr(log B_p, log C) = {corr}"

    # And the split it justifies really does separate B_p at this scale.
    detail = P.make_splits(batch, kinds=("bp_band",)).reports["bp_band"].detail
    assert detail["n_train_inside_band"] == 0.0
    assert detail["B_p_span_all"] > 5.0
