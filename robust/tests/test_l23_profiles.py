"""
Tests for the L23 depth-profile product (M7 task 1, ``rt_fit_ztt.md``).

Three layers, as in ``test_l23.py``:

**Logic, on synthetic profiles** -- the flag rules, including ``unusable``,
which the real release never produces (its shallowest usable depth is 18 m).

**The real builder, on the committed fixture** (``files/l23_profile_fixture.npz``,
16 scenes x 8 depths x 3 zeniths of *raw* fields), so CI runs the product code
on real numbers: shapes, flags, bit-for-bit reproducibility, save/load.

**The full release, behind ``needs_l23_profiles``** -- the task's gates:
Gershun's law at depth, golden rows read straight from the netCDF by a separate
path, and zenith-independence of µ∞ where the product calls it asymptotic.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from robust.rt.data import l23 as L
from robust.tests.conftest import L23_PROFILE_FIXTURE, needs_l23_profiles


@pytest.fixture(scope="module")
def fixture_reader():
    """A reader over the committed profile fixture."""
    if not L23_PROFILE_FIXTURE.is_file():
        pytest.skip(f"profile fixture missing: {L23_PROFILE_FIXTURE}")
    return L.profile_npz_reader(L23_PROFILE_FIXTURE)


@pytest.fixture(scope="module")
def fixture_product(fixture_reader):
    """The product built from the fixture through the real builder."""
    return L.build_profile_product(reader=fixture_reader)


# --------------------------------------------------------------- synthetic --
def synthetic_reader(mu_by_zenith, *, z=(0.0, 10.0, 20.0, 50.0), Eo=None):
    """A reader serving profiles whose µ at every depth is ``mu_by_zenith[Y]``.

    One scene, one wavelength. ``Eo`` (per depth) defaults to 1 everywhere; pass
    zeros to make a depth unusable.
    """
    z = np.asarray(z)
    Eo = np.ones(z.size) if Eo is None else np.asarray(Eo, dtype=float)

    def read(x, zenith, fields, depths=None):
        mu = mu_by_zenith[zenith]
        Eu = 0.01 * Eo
        Ed = mu * Eo + Eu  # so (Ed - Eu) / Eo = mu
        cube = {
            "Ed_z": Ed,
            "Eu_z": Eu,
            "Eo_z": Eo,
            "md_z": np.full(z.size, 0.9),
            "KLu_z": np.full(z.size, 0.05),
        }
        out = {"wave": np.array([500.0]), "z": z, "scene": np.array([7])}
        for f in fields:
            out[f] = cube[f].astype(np.float32)[:, None, None]
        return out

    return read


def test_flags_asymptotic_and_not():
    """Three suns agreeing to < 1% are asymptotic; 5% apart are not."""
    agree = L.build_profile_product(
        reader=synthetic_reader({0: 0.800, 30: 0.802, 60: 0.798})
    )
    assert agree.flag[0, 0] == L.FLAG_ASYMPTOTIC
    np.testing.assert_allclose(agree.mu_inf[0, 0], 0.8, rtol=1e-6)

    apart = L.build_profile_product(
        reader=synthetic_reader({0: 0.82, 30: 0.80, 60: 0.78})
    )
    assert apart.flag[0, 0] == L.FLAG_NOT_ASYMPTOTIC
    assert np.isnan(apart.mu_inf[0, 0])


def test_flag_unusable_when_only_shallow_depths_have_light():
    """Light that underflows below 5 m leaves no deep layer: ``unusable``."""
    product = L.build_profile_product(
        reader=synthetic_reader(
            {0: 0.8, 30: 0.8, 60: 0.8}, z=(0.0, 5.0, 20.0, 50.0), Eo=[1, 1, 0, 0]
        )
    )
    assert product.z_deep[0, 0] == 5.0  # usable, but shallower than the minimum
    assert product.flag[0, 0] == L.FLAG_UNUSABLE
    assert np.isnan(product.mu_inf[0, 0])


def test_denormal_irradiance_is_not_data():
    """Below PROFILE_E_MIN the deepest usable depth moves up, not to the noise."""
    product = L.build_profile_product(
        reader=synthetic_reader({0: 0.8, 30: 0.8, 60: 0.8}, Eo=[1, 1, 1, 1e-40])
    )
    assert product.z_deep[0, 0] == 20.0


def test_scene_labels_come_from_the_reader():
    """The product carries the reader's release indices, not 0..n-1."""
    product = L.build_profile_product(
        reader=synthetic_reader({0: 0.8, 30: 0.8, 60: 0.8})
    )
    assert product.scene.tolist() == [7]


def test_needs_two_zeniths():
    with pytest.raises(ValueError, match="two zeniths"):
        L.build_profile_product(zeniths=(0,))


# ----------------------------------------------------------------- fixture --
def test_fixture_product_shapes_and_flags(fixture_product):
    p = fixture_product
    n_scene, n_wave = len(L.PROFILE_FIXTURE_SCENES), 81
    assert p.scene.tolist() == list(L.PROFILE_FIXTURE_SCENES)
    assert p.mu_deep.shape == (3, n_scene, n_wave)
    assert p.mu_d0.shape == p.KLu0.shape == (3, n_scene, n_wave)
    assert p.flag.dtype == np.int8
    present = set(np.unique(p.flag).tolist())
    assert {L.FLAG_ASYMPTOTIC, L.FLAG_NOT_ASYMPTOTIC} <= present
    # mu_inf is defined exactly where the flag says asymptotic.
    np.testing.assert_array_equal(np.isfinite(p.mu_inf), p.flag == L.FLAG_ASYMPTOTIC)
    # Physical ranges.
    finite = np.isfinite(p.mu_deep)
    assert np.all((p.mu_deep[finite] > 0) & (p.mu_deep[finite] <= 1))
    assert np.all((p.mu_d0 > 0.5) & (p.mu_d0 <= 1))
    assert np.all(p.KLu0 > 0)


def test_fixture_product_is_bit_reproducible(fixture_reader, fixture_product, tmp_path):
    """Gate: rebuilding gives identical arrays, and so an identical file."""
    again = L.build_profile_product(reader=fixture_reader)
    for name in L.L23ProfileProduct.ARRAYS:
        np.testing.assert_array_equal(
            getattr(again, name), getattr(fixture_product, name)
        )
    assert again.provenance == fixture_product.provenance

    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    L.save_profile_product(fixture_product, a)
    L.save_profile_product(again, b)
    # Compare the decompressed members: zip containers carry timestamps.
    with np.load(a) as da, np.load(b) as db:
        assert sorted(da.files) == sorted(db.files)
        for key in da.files:
            np.testing.assert_array_equal(da[key], db[key])


def test_save_load_round_trip_and_stale_version(fixture_product, tmp_path):
    path = tmp_path / "product.npz"
    L.save_profile_product(fixture_product, path)
    back = L.load_profile_product(path)
    for name in L.L23ProfileProduct.ARRAYS:
        np.testing.assert_array_equal(
            getattr(back, name), getattr(fixture_product, name)
        )
    assert back.provenance == fixture_product.provenance

    # save_profile_product validates by loading, so write the stale file raw.
    stale = dict(fixture_product.provenance, version=-1)
    arrays = {n: getattr(fixture_product, n) for n in L.L23ProfileProduct.ARRAYS}
    np.savez(tmp_path / "stale.npz", **arrays, provenance=np.asarray(json.dumps(stale)))
    with pytest.raises(ValueError, match="version"):
        L.load_profile_product(tmp_path / "stale.npz")


def test_load_mu_profiles_matches_product(fixture_reader, fixture_product):
    """µ(z) from load_mu_profiles at z_deep is the product's mu_deep."""
    p = fixture_product
    for k, zenith in enumerate(L.ZENITHS):
        z, mu = L.load_mu_profiles(zenith, reader=fixture_reader)
        assert mu.shape == (z.size, p.scene.size, 81)
        idx = np.searchsorted(z, p.z_deep)
        picked = np.take_along_axis(mu, idx[None], axis=0)[0]
        np.testing.assert_array_equal(picked, p.mu_deep[k])


def test_profile_reader_refuses_what_it_lacks(fixture_reader):
    with pytest.raises(ValueError, match="asked for"):
        fixture_reader(2, 0, ("Ed_z",))
    with pytest.raises(ValueError, match="no depth"):
        fixture_reader(1, 0, ("Ed_z",), depths=(3.0,))


# ------------------------------------------------------------ full release --
@pytest.fixture(scope="module")
def full_product():
    return L.build_profile_product()


@needs_l23_profiles
def test_gershun_holds_at_depth():
    """Gate: (Ed − Eu)/Eo = a/KEd at 50 m, median relative difference < 1e-3."""
    from ocpy.hydrolight import loisel23

    a = np.asarray(loisel23.load_ds(1, 0).a.data, dtype=float)
    for zenith in L.ZENITHS:
        _, mu = L.load_mu_profiles(zenith, depths=(50.0,))
        raw = L._read_profile(1, zenith, ("KEd_z",), depths=(50.0,))
        ratio = mu[0] / (a / raw["KEd_z"][0].astype(float))
        assert np.nanmedian(np.abs(ratio - 1)) < 1e-3, zenith


@needs_l23_profiles
def test_golden_rows_against_raw_netcdf(full_product):
    """Gate: product values equal those read from the netCDF by plain xarray."""
    import xarray as xr
    from ocpy.hydrolight import loisel23

    p = full_product
    for scene, iw in [(0, 10), (1500, 40), (3319, 70)]:
        for k, zenith in enumerate(L.ZENITHS):
            path = os.path.join(
                loisel23.l23_path, f"Hydrolight1{zenith:02d}_profile.nc"
            )
            with xr.open_dataset(path) as ds:
                zz = float(p.z_deep[scene, iw])
                at = ds.sel(z=zz).isel(IOP_Scenario=scene, Lambda=iw)
                Ed, Eu, Eo = (float(at[v]) for v in ("Ed_z", "Eu_z", "Eo_z"))
                surf = ds.sel(z=0.0).isel(IOP_Scenario=scene, Lambda=iw)
                assert p.mu_deep[k, scene, iw] == (Ed - Eu) / Eo
                assert p.mu_d0[k, scene, iw] == np.float32(surf["md_z"])
                assert p.KLu0[k, scene, iw] == np.float32(surf["KLu_z"])


@needs_l23_profiles
def test_asymptotic_points_are_zenith_independent(full_product):
    """Gate: where flagged asymptotic, the three zeniths agree to < 1%."""
    p = full_product
    ok = p.flag == L.FLAG_ASYMPTOTIC
    assert ok.sum() > 1000  # measured 2.4% of 268,920 = ~6,500
    dev = np.abs(p.mu_deep[:, ok] / p.mu_inf[ok] - 1)
    assert dev.max() < L.ASYMPTOTIC_SPREAD


@needs_l23_profiles
def test_fixture_is_faithful_to_release(full_product, fixture_product):
    """The committed fixture reproduces the release at its own depths."""
    p, f = full_product, fixture_product
    idx = np.asarray(L.PROFILE_FIXTURE_SCENES)
    np.testing.assert_array_equal(f.mu_d0, p.mu_d0[:, idx])
    np.testing.assert_array_equal(f.KLu0, p.KLu0[:, idx])
    same_depth = f.z_deep == p.z_deep[idx]
    np.testing.assert_array_equal(
        f.mu_deep[:, same_depth], p.mu_deep[:, idx][:, same_depth]
    )
