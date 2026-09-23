"""
Tests for :mod:`robust.rt.hydrolight` -- the deck generator.

Pinning **measured properties of the design**, not its documentation, in the
style of ``test_l23.py`` and ``test_pb24.py``. Three of these have already
earned their place by failing:

* :func:`test_shipped_branches_are_deliverable` and
  :func:`test_low_mu_is_not_deliverable` encode the discovery that half of
  Fournier-Forand's parameter space cannot be shipped as a discrete table at
  all -- the forward peak for ``mu`` near 3 behaves like ``psi**-1.93`` and at
  ``psi_min = 1e-5 deg`` the table still holds only 64 % of the scattering. The
  design set was built from those parameters before this was measured.
* :func:`test_shape_table_backward_contrast` regenerates the specification's
  central claim about the backscatter hemisphere. Computed from the
  *undeliverable* branches it read 1-3 %; from the deliverable ones it is an
  order of magnitude larger. A number quoted into a document cannot fail; this
  can.
* :func:`test_l23_round_trip` pins that total absorption is carried **verbatim**.
  Rebuilding it from components let the difference between our pure-water table
  and L23's reach ``a`` itself, by up to 150 % in the red.

No ``$OS_COLOR`` is needed: everything runs against the committed L23 fixtures.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt.hydrolight import deck, ensemble, grid, l23_recon, manifest, vsf
from robust.tests.conftest import L23_INELASTIC_FIXTURE, L23_SMALL_FIXTURE


@pytest.fixture(scope="module")
def reader():
    """An L23 reader over the committed fixtures."""
    for fixture in (L23_SMALL_FIXTURE, L23_INELASTIC_FIXTURE):
        if not fixture.is_file():
            pytest.skip(f"cached L23 fixture missing: {fixture}")
    return l23_recon.fixture_reader(L23_SMALL_FIXTURE, L23_INELASTIC_FIXTURE)


# --------------------------------------------------------------------------
# vsf
# --------------------------------------------------------------------------


def test_designs_inventory():
    """28 designs: 17 Fournier-Forand, 9 mixtures, 2 named built-ins."""
    d = vsf.designs()
    kinds = {k: sum(x.kind == k for x in d) for k in ("ff", "mixture", "builtin")}
    assert len(d) == 28
    assert kinds == {"ff": 17, "mixture": 9, "builtin": 2}
    assert sum(x.tabulated for x in d) == 26
    assert len({x.name for x in d}) == len(d), "design names must be unique"


def test_designs_are_deterministic():
    """Two calls give identical tables -- the manifest hashes depend on it."""
    a, b = vsf.designs(), vsf.designs()
    for x, y in zip(a, b, strict=True):
        assert x.name == y.name
        if x.tabulated:
            assert np.array_equal(x.beta_tilde, y.beta_tilde)


def test_delivered_backscatter_fractions():
    """Every tabulated design carries the ``B_p`` it claims.

    Two stages, each held to 4e-4: the closed form must hit the target, and the
    delivered table must hit the closed form. Their sum bounds the total drift.
    """
    for d in vsf.designs():
        if not d.tabulated:
            continue
        closed = (
            d.params["B_p_closed_form"]
            if d.kind == "ff"
            else d.target_bp  # mixtures are linear in B_p by construction
        )
        assert abs(closed - d.target_bp) < 4e-4, d.name
        assert abs(d.realised_bp() - closed) < 4e-4, d.name
        assert abs(d.realised_bp() - d.target_bp) < 8e-4, d.name


def test_tables_are_normalised():
    """Each delivered table integrates to 1 over 4 pi on its own grid."""
    for d in vsf.designs():
        if d.tabulated:
            assert vsf.numeric_norm(vsf.PSI_GRID, d.beta_tilde) == pytest.approx(
                1.0, abs=1e-9
            ), d.name


def test_psi_grid_includes_exact_backscatter():
    """180 deg is a grid point: ``beta_tilde(pi)`` is a model parameter."""
    assert vsf.PSI_GRID[-1] == 180.0
    assert vsf.PSI_GRID[0] == pytest.approx(1e-3)


def test_shipped_branches_are_deliverable():
    """Every ``(n, mu)`` in :data:`vsf.FF_BRANCHES` passes the criterion."""
    for target, pairs in vsf.FF_BRANCHES.items():
        for n, mu in pairs:
            ok, closed, norm, numeric = vsf.deliverable(n, mu)
            assert ok, f"B_p={target} (n={n}, mu={mu}): norm={norm}, B={numeric}"
            assert abs(closed - target) < 4e-4


def test_low_mu_is_not_deliverable():
    """The forward peak for ``mu`` near 3 cannot be tabulated -- measured."""
    ok, closed, norm, numeric = vsf.deliverable(1.190, 3.24)
    assert not ok
    assert norm < 0.95, "expected a table that loses a large part of the peak"
    assert abs(numeric - closed) > 4e-4


def test_branch_counts_are_uneven():
    """Low ``B_p`` pins the shape: one branch at 0.004, six at 0.030."""
    counts = {t: len(p) for t, p in vsf.FF_BRANCHES.items()}
    assert counts == {0.004: 1, 0.007: 2, 0.012: 3, 0.020: 5, 0.030: 6}


def test_solver_reproduces_the_shipped_branches():
    """:func:`vsf.solve_ff_branches` still yields the shipped constants."""
    for target in (0.004, 0.012):
        assert vsf.solve_ff_branches(target) == vsf.FF_BRANCHES[target]


def test_shape_table_backward_contrast():
    """The matched-``B_p`` shape claim, regenerated rather than quoted.

    At matched ``B_p`` the deliverable branches differ strongly in the forward
    peak and appreciably -- not negligibly -- in the backscatter hemisphere.
    """
    st = vsf.shape_table(0.030)
    spread = dict(zip(st["angles"], st["spread"], strict=True))
    assert spread[1.0] > 0.4, "forward peak should vary by tens of percent"
    assert spread[45.0] > 0.3
    assert 0.08 < spread[180.0] < 0.4, (
        "backward contrast is real but modest; a value near 0.02 would mean the "
        "undeliverable branches had crept back in"
    )
    assert spread[180.0] > spread[120.0]


def test_mixtures_bracket_their_targets():
    """A mixture exists only where its two components bracket the target."""
    for recipe in vsf.MIXTURE_RECIPES:
        built = {d.target_bp for d in vsf.designs() if d.name.startswith(recipe)}
        expected = {t for t in vsf.TARGET_BP if vsf.mixture_brackets(recipe, t)}
        assert built == expected
    with pytest.raises(ValueError, match="do not bracket"):
        vsf.mixture_fraction(0.5, *vsf.MIXTURE_RECIPES["mixA"])


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------


def test_realizability_mask_is_63_of_96():
    """The catalogue asked for 96 nodes; 63 are physical water, at every B_p."""
    for B_p in (0.004, 0.012, 0.030):
        assert grid.n_realizable_nodes(B_p) == 63


def test_realizability_ceiling_is_a_property_of_pure_water():
    """``eta_bb * (bb/a) <= bb_w/a_w`` -- the constraint, stated as a test."""
    a_w, bb_w, b_w = grid.water_tables()
    i = int(np.argmin(np.abs(deck.WAVE_HL - grid.LAMBDA_REF)))
    ceiling = bb_w[i] / a_w[i]
    assert ceiling == pytest.approx(0.498, rel=0.02)
    assert grid.realize_node(3.0, 0.98, 0.012, a_w[i], bb_w[i]) is None
    assert grid.realize_node(1e-4, 0.5, 0.012, a_w[i], bb_w[i]) is None  # a_nw > 20
    assert grid.realize_node(0.01, 0.5, 0.012, a_w[i], bb_w[i]) is not None


def test_water_tables_agree_with_the_package():
    """``bb_w`` on the shared 350-750 range is the packaged table, not a refit."""
    _, bb_w, b_w = grid.water_tables()
    sub = deck.WAVE_HL >= C.WAVE[0]
    assert np.allclose(bb_w[sub], C.BB_W_L23, rtol=1e-12)
    assert np.allclose(b_w, 2.0 * bb_w)


def test_grid_box_occupancy_and_named_gaps():
    """The spectral sweep recovers ~90 % of the box; the gaps are enumerated."""
    bodies = grid.grid_bodies() + grid.fill_bodies()
    occ = grid.box_occupancy(bodies)
    assert occ["fraction"] > 0.85
    assert len(occ["empty_cells"]) == occ["n_cells"] - occ["occupied"]
    for cell in occ["empty_cells"]:
        assert set(cell) == {"i", "j", "bb_over_a", "eta_bb"}


def test_batch_a_is_deterministic():
    """Same bodies, same ids, same arrays on every call."""
    a = grid.grid_bodies() + grid.fill_bodies()
    b = grid.grid_bodies() + grid.fill_bodies()
    assert [x.wbid for x in a] == [x.wbid for x in b]
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.a, y.a) and np.array_equal(x.bb, y.bb)


# --------------------------------------------------------------------------
# l23_recon
# --------------------------------------------------------------------------


def test_l23_round_trip(reader):
    """Reconstructed IOPs equal the file's, on L23's own grid.

    ``a`` must be **exact**: it is carried verbatim, not rebuilt from
    components. ``bb`` carries only the fixture's float32 precision.
    """
    raw = reader()
    bodies = l23_recon.reconstruct(reader)
    sub = deck.WAVE_HL >= raw["wave"][0]
    a = np.array([b.a[sub] for b in bodies])
    bb = np.array([b.bb[sub] for b in bodies])
    b_p = np.array([b.b_p[sub] for b in bodies])
    assert np.allclose(a, raw["a"], rtol=0, atol=0)
    assert np.allclose(b_p, raw["bnw"], rtol=0, atol=0)
    assert np.max(np.abs(bb - raw["bb"]) / raw["bb"]) < 1e-6


def test_l23_caveats_are_data_not_comments(reader):
    """Every reconstructed body carries the three caveats as recorded fields."""
    body = l23_recon.reconstruct(reader, indices=[0])[0]
    caveats = body.provenance["caveats"]
    assert set(caveats) == set(l23_recon.CAVEATS)
    assert all(caveats.values())
    assert body.provenance["ff_mu_fixed"] == vsf.FF_MU_FIXED
    assert "component_residual_min_m^-1" in body.provenance


def test_infer_ff_n_inverts_the_closed_form():
    """The inferred FF parameter reproduces the ``B_p`` it was solved from."""
    target = np.array([0.005, 0.012, 0.02, 0.03])
    n = l23_recon.infer_ff_n(target)
    assert np.allclose(
        vsf.ff_backscatter_fraction(n, vsf.FF_MU_FIXED), target, atol=1e-7
    )


def test_uv_extension_is_finite_and_positive(reader):
    """The four bands below L23's floor extend, rather than clamp to nothing."""
    body = l23_recon.reconstruct(reader, indices=[0])[0]
    below = deck.WAVE_HL < 350.0
    assert below.sum() == 4
    for field in (body.a, body.b, body.bb, body.b_p):
        assert np.all(np.isfinite(field[below]))
        assert np.all(field[below] > 0)


# --------------------------------------------------------------------------
# ensemble
# --------------------------------------------------------------------------


def test_design_block_is_reproducible(reader):
    """Same seed, same draws -- the ensemble is recoverable from the manifest."""
    a = ensemble.design_block(reader, n=25)
    b = ensemble.design_block(reader, n=25)
    assert [x.wbid for x in a] == [x.wbid for x in b]
    assert np.array_equal(a[7].a, b[7].a)


def test_design_block_crosses_the_vsf_axis(reader):
    """Batch B carries the phase-function designs -- the point of HD4."""
    bodies = ensemble.design_block(reader, n=60)
    assert len({b.vsf_design for b in bodies}) > 10
    assert all(b.scenarios == ("S1", "S2", "S4") for b in bodies)


def test_r5_oversamples_the_cdom_tail(reader):
    """R5's subset is drawn toward high ``a_cdom(440)``, deliberately."""
    bodies = ensemble.design_block(reader, n=200)
    i440 = int(np.argmin(np.abs(deck.WAVE_HL - 440.0)))
    picked = ensemble.r5_subset(bodies, n=40)
    assert np.median([b.a_cdom[i440] for b in picked]) > np.median(
        [b.a_cdom[i440] for b in bodies]
    )
    assert all(b.scenarios == ("S5",) for b in picked)


def test_r4_is_a_phi_series(reader):
    """R4 varies ``phi_C`` on shared optics, not the water."""
    bodies = ensemble.design_block(reader, n=40)
    picked = ensemble.r4_subset(bodies, n=5)
    assert len(picked) == 20
    assert sorted({b.phi_C for b in picked}) == [0.005, 0.01, 0.04, 0.06]
    same = [
        b
        for b in picked
        if b.provenance["derived_from"] == picked[0].provenance["derived_from"]
    ]
    assert all(np.array_equal(b.a, same[0].a) for b in same)


def test_r8_has_a_homogeneous_control(reader):
    """Sixty profiles, one third of them controls."""
    bodies = ensemble.design_block(reader, n=40)
    picked = ensemble.r8_profiles(bodies, n_base=20)
    assert len(picked) == 60
    assert sum(b.depth_profile is None for b in picked) == 20


# --------------------------------------------------------------------------
# deck and manifest
# --------------------------------------------------------------------------


def test_wave_grid_subset_is_bit_identical_to_l23():
    """The 350-750 subset *is* ``conventions.WAVE`` -- not merely close."""
    assert np.array_equal(deck.WAVE_HL[deck.WAVE_HL >= 350.0], C.WAVE)
    assert deck.WAVE_HL.size == 85


def test_scenarios_carry_the_x_mapping():
    """Switches are named; the L23 X label is recorded, not relied on."""
    assert deck.SCENARIOS["S4"] == {
        "raman": True,
        "chl_fl": True,
        "cdom_fl": False,
        "l23_X": 4,
    }
    assert deck.SCENARIOS["S5"]["l23_X"] is None


def test_render_deck_refuses_without_a_template(reader):
    """A nearly-right deck is worse than none: it would run."""
    body = l23_recon.reconstruct(reader, indices=[0])[0]
    with pytest.raises(NotImplementedError, match="not known yet"):
        deck.render_deck(body, 0, "S1")
    text = deck.render_provisional_deck(body, 30, "S4")
    assert "NOT a HydroLight input deck" in text
    assert "phi_C = 0.02" in text


def test_write_arrays_is_byte_identical(tmp_path, reader):
    """Regeneration must be reproducible, or the manifest hashes are worthless."""
    body = l23_recon.reconstruct(reader, indices=[0])[0]
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    deck.write_arrays(first, body.arrays())
    deck.write_arrays(second, body.arrays())
    assert first.read_bytes() == second.read_bytes()
    assert manifest.sha256(first) == manifest.sha256(second)


def test_written_arrays_round_trip(tmp_path, reader):
    """The deterministic writer still produces a readable ``.npz``."""
    body = l23_recon.reconstruct(reader, indices=[0])[0]
    path = tmp_path / "wb.npz"
    deck.write_arrays(path, body.arrays())
    loaded = np.load(path)
    assert np.array_equal(loaded["a"], body.a)
    assert loaded["a"].dtype == np.float64


def test_manifest_covers_the_ten_contract_items(tmp_path, reader):
    """All ten items present; the operator's four marked pending with a reason."""
    bodies = l23_recon.reconstruct(reader, indices=[0, 1])
    files = []
    for wb in bodies:
        files += deck.write_water_body(tmp_path, wb)
    man = manifest.build(
        "pilot", bodies, files, generator_version="test", created="2026-09-23"
    )
    contract = man["metadata_contract"]
    assert len(contract) == 10
    pending = {k for k, v in contract.items() if v["status"] == "pending"}
    assert pending == {
        "1_hydrolight_version_and_switches",
        "3_raman_coefficient_and_redistribution",
        "4_chl_fluorescence_efficiency_and_shape",
        "5_cdom_fluorescence_hawes_variant_and_constants",
        "6_solar_spectrum_and_sky_model",
        "8_output_convention_Rrs_vs_rrs_above_vs_below",
    }
    assert all("reason" in contract[k] for k in pending)
    assert man["totals"]["runs"] == sum(b.n_runs() for b in bodies)
    assert all(len(f["sha256"]) == 64 for f in man["files"])


def test_manifest_is_deterministic_json(tmp_path, reader):
    """Same inputs, same bytes."""
    bodies = l23_recon.reconstruct(reader, indices=[0])
    files = deck.write_water_body(tmp_path, bodies[0])
    kwargs = dict(generator_version="test", created="2026-09-23")
    a = manifest.build("pilot", bodies, files, **kwargs)
    b = manifest.build("pilot", bodies, files, **kwargs)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_manifest_flags_the_run_count_assumption(tmp_path, reader):
    """The sizing rests on an unconfirmed model; the file must say so."""
    bodies = l23_recon.reconstruct(reader, indices=[0])
    man = manifest.build(
        "pilot", bodies, [], generator_version="test", created="2026-09-23"
    )
    assert "CONFIRM WITH THE OPERATOR" in man["run_count_model"]


# --------------------------------------------------------------------------
# batch 0
# --------------------------------------------------------------------------


def test_batch0_is_172_runs(reader):
    """The pilot is exactly the 172 runs the specification commits to."""
    import sys

    sys.path.insert(0, str(L23_SMALL_FIXTURE.parents[3] / "design" / "py"))
    from make_hydrolight_batch0 import select

    p1, p2, p3 = select(reader)
    assert (len(p1), len(p2), len(p3)) == (10, 5, 6)
    assert sum(b.n_runs() for b in p1) == 90
    assert sum(b.n_runs() for b in p2) == 70
    assert sum(b.n_runs() for b in p3) == 12
    assert sum(b.n_runs() for b in p1 + p2 + p3) == 172


def test_batch0_selection_is_reproducible(reader):
    """Chosen by rule from a stated quantile -- not hand-picked."""
    import sys

    sys.path.insert(0, str(L23_SMALL_FIXTURE.parents[3] / "design" / "py"))
    from make_hydrolight_batch0 import select

    first = select(reader)
    second = select(reader)
    for a, b in zip(first, second, strict=True):
        assert [x.wbid for x in a] == [x.wbid for x in b]
    for wb in first[0]:
        assert "selection" in wb.provenance
