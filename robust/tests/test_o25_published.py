"""
Tests for the published O25 LUTs and the paper's fitting recipe (M8 task 5).

Three groups:

- **The reader's layout**, data-free, on synthetic files whose every value
  encodes its own ``(file, block, row, column)``, and then on the real
  checkout (``needs_o25_checkout``) against a direct ``np.loadtxt`` of the
  same files. Nothing from the GPL-3.0 O25 repository is pasted here: every
  published number in these tests is read from the checkout at run time.
- **The node-polynomial identity** (the task's gate): ``Rrs_o25`` with the
  published table, evaluated exactly at a published node, is the LUT's own
  polynomial ``(G0w + G1w w_bw) w_bw + (G0p + G1p w_bp) w_bp`` to float64
  rounding.
- **``fit_o25_table(rrs_max=...)``**, data-free on the cache miniature and on a
  synthetic round trip: it drops exactly the points it should, and
  ``rrs_max=None`` is bit-identical to the function before the option.

Plus the committed reproduction (``design/ztt_pb24/o25_refit_lut.npz``)
checked against the published files, and a PB24 OLCI positive control for the
azimuth convention.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import baselines as B
from robust.rt.types import Geometry, IOPs
from robust.tests.conftest import (
    needs_o25_checkout,
    needs_pb24_olci,
    pb24_mixed_realisations,
)

ROOT = Path(__file__).resolve().parents[2]
REFIT_LUT = ROOT / "design" / "ztt_pb24" / "o25_refit_lut.npz"

#: Node-identity tolerance, relative: float64 rounding of a four-term sum.
NODE_RTOL = 1e-12


def _published_raw():
    """The four published files as read by ``np.loadtxt``, (Gw0, Gw1, Gp0, Gp1)."""
    data = B.o25_checkout_dir() / "Data"
    return [np.loadtxt(data / name) for name in B.O25_PUBLISHED_FILES]


def _write_coded_files(directory: Path, shape=(130, 10)):
    """Synthetic LUT files whose values encode ``(file, block, row, col)``."""
    data = directory / "Data"
    data.mkdir(parents=True)
    for c, name in enumerate(B.O25_PUBLISHED_FILES):
        rows = np.arange(shape[0])[:, None]
        cols = np.arange(shape[1])[None, :]
        block, row = rows // 10, rows % 10
        values = c * 1e6 + block * 1e4 + row * 1e2 + cols
        np.savetxt(data / name, values, fmt="%.1f")
    return directory


# ----------------------------------------------------------- the reader ----


def test_the_reader_layout_on_coded_files(tmp_path):
    """Block = O25 azimuth (mirrored onto PB24's dphi), row = theta_s, col = theta_v."""
    _write_coded_files(tmp_path)

    table = B.o25_published_table(tmp_path)
    as_coded = B.o25_published_table(tmp_path, azimuth="o25")

    assert table.G.shape == (10, 10, 13, 4)
    for c in range(4):
        for block in (0, 3, 12):
            for row in (0, 4, 9):
                for col in (0, 7, 9):
                    code = c * 1e6 + block * 1e4 + row * 1e2 + col
                    assert table.G[row, col, 12 - block, c] == code
                    assert as_coded.G[row, col, block, c] == code
    np.testing.assert_array_equal(table.theta_s, B.O25_PUBLISHED_THETA_S)
    np.testing.assert_array_equal(table.theta_v, B.O25_PUBLISHED_THETA_V)
    np.testing.assert_array_equal(table.dphi, B.O25_PUBLISHED_DPHI)
    assert "commit unknown" in table.provenance  # tmp_path is not a git checkout
    assert "calibrated on all 5000" in table.provenance


def test_the_reader_refuses_a_wrong_shape(tmp_path):
    _write_coded_files(tmp_path, shape=(120, 10))
    with pytest.raises(ValueError, match="13 azimuth blocks"):
        B.o25_published_table(tmp_path)


def test_the_reader_refuses_an_unknown_azimuth_convention(tmp_path):
    _write_coded_files(tmp_path)
    with pytest.raises(ValueError, match="azimuth"):
        B.o25_published_table(tmp_path, azimuth="sun")


def test_a_missing_checkout_raises_and_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="O25_DIR") as info:
        B.o25_published_table(tmp_path / "nowhere")
    assert str(tmp_path / "nowhere" / "Data") in str(info.value)
    assert "G0w.txt" in str(info.value)


def test_the_checkout_dir_resolution(monkeypatch, tmp_path):
    """Explicit path > ``$O25_DIR`` > ``../O25`` beside the repository."""
    monkeypatch.setenv(B.O25_DIR_ENV, str(tmp_path))
    assert B.o25_checkout_dir() == tmp_path
    assert B.o25_checkout_dir("/elsewhere") == Path("/elsewhere")
    monkeypatch.delenv(B.O25_DIR_ENV)
    assert B.o25_checkout_dir() == ROOT.parent / "O25"


def test_the_published_nodes_are_pb24s():
    """Hard-coded node vectors, so the reader needs no PB24 data."""
    assert B.O25_PUBLISHED_THETA_S[-1] == 87.75
    assert B.O25_PUBLISHED_THETA_V[-1] == 87.5
    assert np.array_equal(B.O25_PUBLISHED_DPHI, np.arange(0.0, 181.0, 15.0))
    assert len(B.O25_PUBLISHED_THETA_S) == len(B.O25_PUBLISHED_THETA_V) == 10


@needs_o25_checkout
def test_the_published_table_layout_against_loadtxt():
    """Spot values at named (file, block, row, col) against a direct read."""
    raw = _published_raw()
    table = B.o25_published_table()
    as_coded = B.o25_published_table(azimuth="o25")

    assert table.G.shape == (10, 10, 13, 4)
    assert np.all(np.isfinite(table.G))
    for c, block, row, col in ((0, 0, 0, 0), (1, 4, 5, 3), (2, 7, 2, 6), (3, 12, 9, 9)):
        assert table.G[row, col, 12 - block, c] == raw[c][10 * block + row, col]
        assert as_coded.G[row, col, block, c] == raw[c][10 * block + row, col]
    # The mirror is a pure re-indexing of the same 5200 numbers.
    np.testing.assert_array_equal(table.G, as_coded.G[:, :, ::-1])
    assert "calibrated on all 5000" in table.provenance
    assert str(B.o25_checkout_dir()) in table.provenance


@needs_o25_checkout
def test_the_nadir_view_column_is_azimuth_free():
    """Identifies the columns as theta_v: a nadir view has no azimuth.

    HydroLight's nadir direction is a polar cap, so every quantity at
    ``theta_v = 0`` is the same for all 13 azimuths, exactly. The published
    files honour that in column 0 and nowhere else (the ``theta_s = 0`` rows
    are not azimuth-free on PB24's wind-roughened surface).
    """
    G = B.o25_published_table().G
    assert np.ptp(G[:, 0, :, :], axis=1).max() == 0.0
    assert np.ptp(G[0, :, :, :], axis=1).max() > 0.0


@needs_o25_checkout
def test_rrs_o25_at_a_published_node_is_the_luts_own_polynomial(jax_x64):
    """**Gate.** At a node, the trilinear lookup must return the file's numbers.

    Reference: NumPy float64, the polynomial written out with the G's taken
    straight from ``np.loadtxt`` (block ``12 - k`` for PB24's ``dphi`` index
    ``k``). Tolerance ``1e-12`` relative: float64 rounding of a four-term sum.
    """
    raw = _published_raw()
    table = B.o25_published_table()
    rng = np.random.default_rng(5)
    nodes = [(0, 0, 0), (3, 5, 7), (7, 7, 12), (9, 9, 0), (8, 2, 4), (9, 0, 12)]
    n, n_wave = 7, 4
    for i, j, k in nodes:
        a = rng.uniform(0.02, 2.0, (n, n_wave))
        bb_w = rng.uniform(2e-4, 3e-3, (n, n_wave))
        bb_p = rng.uniform(1e-4, 0.1, (n, n_wave))
        geometry = Geometry(
            theta_s=jnp.full(n, B.O25_PUBLISHED_THETA_S[i]),
            theta_v=jnp.full(n, B.O25_PUBLISHED_THETA_V[j]),
            dphi=jnp.full(n, B.O25_PUBLISHED_DPHI[k]),
        )
        got = B.Rrs_o25(
            IOPs(a=jnp.asarray(a), bb_w=jnp.asarray(bb_w), bb_p=jnp.asarray(bb_p)),
            None,
            geometry,
            coeffs=table,
        )
        assert got.dtype == jnp.float64
        g0w, g1w, g0p, g1p = (raw[c][10 * (12 - k) + i, j] for c in range(4))
        w_bw = bb_w / (a + bb_w + bb_p)
        w_bp = bb_p / (a + bb_w + bb_p)
        want = (g0w + g1w * w_bw) * w_bw + (g0p + g1p * w_bp) * w_bp
        np.testing.assert_allclose(np.asarray(got), want, rtol=NODE_RTOL, atol=0.0)


@needs_o25_checkout
def test_the_committed_reproduction_matches_the_published_files():
    """The committed paper-recipe refit agrees with the files to < 0.1% in the window.

    The task's prediction was < 1% at every in-window node; the full run
    measured 0.069%. Positive control: read with O25's own azimuth labelling
    (unmirrored) the same refit disagrees by tens of per cent.
    """
    if not REFIT_LUT.is_file():
        pytest.skip(f"{REFIT_LUT} not built")
    lut = np.load(REFIT_LUT)
    table = B.o25_published_table()
    np.testing.assert_array_equal(lut["theta_s"], table.theta_s)
    np.testing.assert_array_equal(lut["theta_v"], table.theta_v)
    np.testing.assert_array_equal(lut["dphi"], table.dphi)
    window = (table.theta_s[:, None] <= 70) & (table.theta_v[None, :] <= 70)

    rel = np.abs(lut["G"] / table.G - 1.0)
    assert rel[window].max() < 1e-3
    assert rel.max() < 2e-3
    unmirrored = np.abs(lut["G"] / B.o25_published_table(azimuth="o25").G - 1.0)
    assert unmirrored[window].max() > 0.3
    assert int(lut["n_points"].min()) > 58_000  # <= 60,000 per node before the cut


@needs_o25_checkout
@needs_pb24_olci
def test_the_published_model_on_pb24_needs_the_mirrored_azimuth():
    """On real OLCI samples the mirrored reading predicts Rrs far better.

    Also checks the reader's hard-coded nodes against the files' grids.
    """
    from ocpy.hydrolight import pb24 as ocpy_pb24

    from robust.rt.data import pb24

    grids = ocpy_pb24.grids("olci")
    np.testing.assert_array_equal(grids["theta_s"], B.O25_PUBLISHED_THETA_S)
    np.testing.assert_array_equal(grids["theta"], B.O25_PUBLISHED_THETA_V)
    np.testing.assert_array_equal(grids["phi"], B.O25_PUBLISHED_DPHI)

    batch = pb24.load_batch(
        realisations=pb24_mixed_realisations(8),
        angles="window",
        resolution="olci",
        extras=(),
        check_fresnel=False,
    )
    true = np.asarray(batch.Rrs)
    errors = {}
    for azimuth in ("pb24", "o25"):
        table = B.o25_published_table(azimuth=azimuth)
        pred = np.asarray(B.Rrs_o25(batch.iops, None, batch.geometry, coeffs=table))
        errors[azimuth] = float(np.median(np.abs(pred / true - 1.0)))
    assert errors["pb24"] < 0.06
    assert errors["pb24"] < 0.7 * errors["o25"]


# --------------------------------------------------- fit_o25_table(rrs_max) -


def _fit(batch, **kwargs):
    return B.fit_o25_table(
        batch.iops,
        batch.Rrs,
        batch.geometry,
        train=np.ones(batch.n_sample, dtype=bool),
        **kwargs,
    )


def test_rrs_max_none_is_bit_identical_to_the_old_call(pb24_small_batch):
    """The new keyword's default changes nothing, bit for bit."""
    batch = pb24_small_batch
    for weighted in (True, False):
        old = _fit(batch, weighted=weighted)
        new = _fit(batch, weighted=weighted, rrs_max=None)
        np.testing.assert_array_equal(new.G, old.G)
        assert new.provenance == old.provenance
    # A ceiling above every value drops nothing either.
    above = _fit(batch, weighted=False, rrs_max=float(np.max(batch.Rrs)) * 2)
    np.testing.assert_array_equal(above.G, _fit(batch, weighted=False).G)
    assert "dropped" in above.provenance


def test_rrs_max_drops_exactly_the_points_above_it(pb24_small_batch):
    """Each node's fit equals a hand-made lstsq over that node's kept points."""
    batch = pb24_small_batch
    # The fixture is float32 and fit_o25_table computes in its inputs' dtype,
    # so the reference does too: that is what makes bit-identity a fair ask.
    Rrs = np.asarray(batch.Rrs)
    threshold = float(np.quantile(Rrs, 0.7))
    table = _fit(batch, weighted=False, rrs_max=threshold, min_samples=10)

    a, bb_w, bb_p = (
        np.asarray(x) for x in (batch.iops.a, batch.iops.bb_w, batch.iops.bb_p)
    )
    w_bw, w_bp = bb_w / (a + bb_w + bb_p), bb_p / (a + bb_w + bb_p)
    ts, tv, ph = (
        np.asarray(x)
        for x in (batch.geometry.theta_s, batch.geometry.theta_v, batch.geometry.dphi)
    )
    checked = 0
    for i, j, k in ((0, 0, 0), (3, 4, 5), (7, 7, 12), (5, 0, 9)):
        cell = (
            (ts == table.theta_s[i]) & (tv == table.theta_v[j]) & (ph == table.dphi[k])
        )
        design = np.stack(
            [w_bw[cell], w_bw[cell] ** 2, w_bp[cell], w_bp[cell] ** 2], axis=-1
        ).reshape(-1, 4)
        target = Rrs[cell].reshape(-1)
        keep = target <= threshold
        assert 0 < keep.sum() < keep.size  # the cut bites in these cells
        beta, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
        np.testing.assert_array_equal(table.G[i, j, k], beta)
        checked += 1
    assert checked == 4


def test_rrs_max_counts_against_min_samples(pb24_small_batch):
    batch = pb24_small_batch
    with pytest.raises(ValueError, match="fewer than"):
        _fit(batch, weighted=False, rrs_max=float(np.min(batch.Rrs)) * 0.5)


def test_rrs_max_recovers_a_known_table_through_corrupted_bright_points():
    """Points above the ceiling are excluded, not down-weighted.

    Rrs is built from a known table, then every true value above the ceiling
    is multiplied by ten. With ``rrs_max`` the fit recovers the table exactly
    (to lstsq rounding); without it, it does not.
    """
    rng = np.random.default_rng(11)
    n = 1200
    geometry = Geometry(
        theta_s=jnp.asarray(rng.choice([0.0, 60.0], n)),
        theta_v=jnp.asarray(rng.choice([0.0, 60.0], n)),
        dphi=jnp.asarray(rng.choice([0.0, 180.0], n)),
    )
    iops = IOPs(
        a=jnp.asarray(rng.uniform(0.02, 0.5, (n, 5))),
        bb_w=jnp.asarray(rng.uniform(1e-4, 3e-3, (n, 5))),
        bb_p=jnp.asarray(rng.uniform(1e-3, 2e-2, (n, 5))),
    )
    G = np.zeros((2, 2, 2, 4))
    G[...] = (0.06, 0.03, 0.04, 0.15)
    G[:, 1, :, 2] = 0.02
    truth = B.O25Table(
        theta_s=np.array([0.0, 60.0]),
        theta_v=np.array([0.0, 60.0]),
        dphi=np.array([0.0, 180.0]),
        G=G,
    )
    clean = np.asarray(B.Rrs_o25(iops, None, geometry, coeffs=truth), dtype=float)
    ceiling = float(np.quantile(clean, 0.8))
    corrupted = np.where(clean > ceiling, clean * 10.0, clean)
    train = np.ones(n, dtype=bool)

    cut = B.fit_o25_table(
        iops, corrupted, geometry, train=train, weighted=False, rrs_max=ceiling
    )
    raw = B.fit_o25_table(iops, corrupted, geometry, train=train, weighted=False)

    np.testing.assert_allclose(cut.G, truth.G, rtol=1e-4, atol=1e-6)
    assert np.max(np.abs(raw.G / truth.G - 1.0)) > 0.1
