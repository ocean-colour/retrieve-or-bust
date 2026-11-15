"""
Tests for M8 task 6: the PB24 baseline runner and its two package helpers.

- :class:`robust.rt.asymptotic.KernelGrid` (FF kernels on a B grid) and
  :mod:`robust.rt.data.pb24_eigen` (the eigen µ∞ product) are additive helpers;
  each is checked on its own, data-free.
- ``design/py/run_pb24_baseline.py`` is run end to end on the committed
  3-realisation cache miniature (no ``$OS_COLOR``), and its artefacts are checked
  for what the task's gate asks of them: headers derived from the same column
  objects as the values, every model named in the registry, the hybrid's
  751-800 nm reported separately, the common table's transfer-error rows, the
  markdown built from the CSVs, and nothing dropped silently.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from robust.rt import asymptotic as A
from robust.rt.data import pb24_eigen
from robust.tests.conftest import PB24_FIXTURE_REALISATIONS

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "design" / "py" / "run_pb24_baseline.py"


def load_script():
    """Import the runner as a module (it is a script, not package code)."""
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_pb24_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pb24_baseline"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ KernelGrid -----
@pytest.fixture(scope="module")
def small_grid():
    """16 streams, coarse azimuth: fast, and enough to test the mechanics."""
    mu, w = A.double_gauss(16)
    kw = {"n_phi": 128}
    grid = A.ff_kernel_grid([0.005, 0.01, 0.02], mu, w, **kw)
    return mu, w, grid, kw


def test_kernel_grid_is_exact_at_its_nodes(small_grid):
    mu, w, grid, kw = small_grid
    exact = A.build_kernel(A.ff_phase_numpy(0.01), mu, w, **kw)
    k = grid.at(0.01)
    np.testing.assert_array_equal(k.matrix, exact.matrix)
    assert k.truncated == exact.truncated


def test_interpolated_kernel_conserves_energy_and_is_close(small_grid):
    mu, w, grid, kw = small_grid
    B = 0.0071
    k = grid.at(B)
    row = 2 * np.pi * (k.matrix * w[None, :]).sum(axis=1)
    np.testing.assert_allclose(row, 1.0, rtol=0, atol=1e-12)
    exact = A.build_kernel(A.ff_phase_numpy(B), mu, w, **kw)
    water = A.build_kernel(A.water_phase(), mu, w, **kw)
    a = np.array([0.05, 0.5])
    b = np.array([[0.002, 0.3, 0.1], [0.001, 0.05, 0.02]])
    m_i = A.mu_infinity_eigen(a, b, mu, w, (water, k, k))
    m_e = A.mu_infinity_eigen(a, b, mu, w, (water, exact, exact))
    # A grid at a factor 2 spacing: still within a few per mille.
    assert np.max(np.abs(m_i / m_e - 1.0)) < 5e-3


def test_kernel_grid_refuses_to_extrapolate(small_grid):
    _, _, grid, _ = small_grid
    with pytest.raises(ValueError, match="outside"):
        grid.at(0.03)
    with pytest.raises(ValueError, match="ascending"):
        A.ff_kernel_grid([0.02, 0.01], *A.double_gauss(4))


# ------------------------------------------------------- the eigen product ---
def test_eigen_product_round_trips_deterministically(tmp_path):
    rng = np.random.default_rng(0)
    content = {
        "realisations": np.array([1, 993, 2500]),
        "wave": np.array([400.0, 500.0]),
        "mu_inf": rng.uniform(0.3, 0.9, (3, 2)),
        "B_ph": np.array([0.01, 0.02, 0.005]),
        "B_nap": np.array([0.015, 0.012, 0.018]),
        "provenance": {"note": "test", "n": 6},
    }
    a = pb24_eigen.save_mu_inf_eigen(tmp_path / "a.npz", **content)
    b = pb24_eigen.save_mu_inf_eigen(tmp_path / "b.npz", **content)
    assert a.read_bytes() == b.read_bytes()
    product = pb24_eigen.load_mu_inf_eigen(a)
    np.testing.assert_array_equal(product.mu_inf, content["mu_inf"])
    assert product.provenance["version"] == pb24_eigen.EIGEN_VERSION
    np.testing.assert_array_equal(product.rows([2500, 1]), [2, 0])
    with pytest.raises(KeyError):
        product.rows([7])
    with pytest.raises(ValueError, match="should be"):
        pb24_eigen.save_mu_inf_eigen(tmp_path / "c.npz", **(content | {"wave": [1.0]}))
    with pytest.raises(FileNotFoundError):
        pb24_eigen.load_mu_inf_eigen(tmp_path / "missing.npz")


def test_the_release_eigen_product_reads_back():
    """Skips without the product (``run_pb24_baseline.py --eigen-only``)."""
    try:
        path = pb24_eigen.default_eigen_path()
    except Exception:  # noqa: BLE001 - no $OS_COLOR / ocpy: no product
        pytest.skip("no $OS_COLOR")
    if not path.is_file():
        pytest.skip(f"eigen µ∞ product not built: {path}")
    product = pb24_eigen.load_mu_inf_eigen(path)
    assert product.mu_inf.shape == (product.realisations.size, product.wave.size)
    assert np.all((product.mu_inf > 0) & (product.mu_inf <= 1))
    assert product.provenance["interpolation_check"]["abs_rel_dev_max"] < 1e-3


# ------------------------------------------------------ the runner itself ----
def test_histogram_quantiles_are_within_half_a_bin():
    R = load_script()
    rng = np.random.default_rng(3)
    e = np.concatenate([rng.lognormal(-3, 1.5, 20000), [0.0, np.inf, -2.0]])
    hb = R.histogram_bin(np.where(np.isfinite(e), e, np.nan))
    h = np.bincount(hb, minlength=R.N_BIN)
    assert h[0] >= 1 and h[-1] >= 1  # zero in the underflow, inf in the overflow
    for p in (0.5, 0.9):
        exact = np.quantile(np.abs(e[np.isfinite(e)]), p)
        tol = 10 ** (R.HIST_WIDTH) - 1 + 1e-3  # a bin, plus the sample-rank slack
        assert abs(R.hist_quantile(h, p) / exact - 1) < tol


@pytest.fixture(scope="module")
def baseline_run(pb24_cache_dir, tmp_path_factory):
    """The runner on the fixture: in-process, no eigen product, no figures."""
    R = load_script()
    out = tmp_path_factory.mktemp("baseline")
    plan = R.make_plan(
        quick=False, cache_dir=pb24_cache_dir, realisations=PB24_FIXTURE_REALISATIONS
    )
    registry = R.build_registry(with_published=False)
    R.run(
        out,
        plan,
        workers=0,
        eigen="skip",
        eigen_path=out / "no_product.npz",
        registry=registry,
        figures=False,
    )
    return R, plan, registry, out


def read(path: Path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], [dict(zip(rows[0], r, strict=True)) for r in rows[1:]]


def test_headers_are_the_column_objects(baseline_run):
    R, _, _, out = baseline_run
    for name, columns in (
        ("scores", R.SCORE_COLUMNS),
        ("per_band", R.BAND_COLUMNS),
        ("breakdowns", R.BREAKDOWN_COLUMNS),
        ("mu_d", R.MU_D_COLUMNS),
        ("klu", R.KLU_COLUMNS),
        ("klu_oracle", R.ORACLE_COLUMNS),
        ("bbw", R.BBW_COLUMNS),
        ("psi", R.PSI_COLUMNS),
        ("o25_refit", R.O25_COLUMNS),
    ):
        header, rows = read(out / f"baseline_{name}.csv")
        assert header == [c.name for c in columns], name
        assert rows, name
    header, _ = read(out / "baseline_mu_d_per_band.csv")
    assert header[3:] == [
        f"median_signed_dev_pct [{label}]" for _, label in R.MU_D_METHODS
    ]


def test_every_model_is_in_the_registry(baseline_run):
    R, plan, registry, out = baseline_run
    labels = {s.label for s in registry}
    for name in ("scores", "per_band", "breakdowns"):
        _, rows = read(out / f"baseline_{name}.csv")
        seen = {r["model"] for r in rows}
        assert seen <= labels, seen - labels
        assert {r["sample"] for r in rows} == {plan.tag}
    _, rows = read(out / "baseline_scores.csv")
    assert {r["model"] for r in rows} == labels


def test_the_hybrids_out_of_domain_band_range_is_separate(baseline_run):
    R, plan, _, out = baseline_run
    assert plan.wave.max() > R.EMULATOR_WAVE_MAX  # the fixture reaches 800 nm
    _, rows = read(out / "baseline_scores.csv")
    hyb = [r for r in rows if r["model"] in (R.HYBRID, R.HYBRID_FB)]
    bands = {r["bands"] for r in hyb}
    assert bands == {R.BRANGE["le750"].label, R.BRANGE["gt750"].label}
    _, rows = read(out / "baseline_per_band.csv")
    for r in rows:
        if r["model"] in (R.HYBRID, R.HYBRID_FB):
            ood = float(r["wavelength_nm"]) > R.EMULATOR_WAVE_MAX
            assert r["domain"] == (
                "out of domain by construction" if ood else "in domain"
            )
        else:
            assert r["domain"] == "in domain"


def test_the_common_table_carries_the_transfer_error(baseline_run):
    R, _, _, out = baseline_run
    _, rows = read(out / "baseline_scores.csv")
    for t in ("dw", "tv"):
        hit = [
            r
            for r in rows
            if r["model"] == R.TRANSFER_ONLY[t]
            and r["tables"] == "common"
            and r["cut"] == R.CUT["window"].label
            and r["side"] == "held out"
        ]
        assert hit and float(hit[0]["median_abs_rel_err_pct"]) > 0
    common = {r["model"] for r in rows if "common" in r["tables"].split("+")}
    assert R.O25_REFIT_RRS["dw"] in common and R.GORDON in common


def test_nothing_is_dropped_without_a_count(baseline_run):
    R, plan, _, out = baseline_run
    _, rows = read(out / "baseline_scores.csv")
    n_w = plan.wave.size
    for r in rows:
        if r["split"] != "none" or r["bands"] != R.BRANGE["all"].label:
            continue
        n_geom = {"window": 832, "shell": 468, "everything": 1300, "in_water": 650}
        for key, g in n_geom.items():
            if r["cut"] == R.CUT[key].label:
                total = int(r["n_values"]) + int(r["n_truth_zero_excluded"])
                assert total == 3 * g * n_w, (r["model"], key)


def test_the_markdown_is_built_from_the_csvs(baseline_run):
    R, _, _, out = baseline_run
    text = (out / "baseline_report.md").read_text()
    _, rows = read(out / "baseline_scores.csv")
    row = R.one(
        rows,
        model=R.GORDON,
        cut=R.CUT["window"].label,
        split="realisation",
        side="held out",
        bands=R.BRANGE["all"].label,
    )
    cells = [R.fnum(row[k]) for k in ("median_abs_rel_err_pct", "q90_abs_rel_err_pct")]
    assert f"| {R.GORDON} | rrs | 350-800 | {cells[0]} | {cells[1]} |" in text
    for prediction in (R.PRED_Q9D, R.PRED_Q10, R.PRED_KLU, R.PRED_Q15):
        assert prediction in text
    # Each prediction precedes its outcome.
    assert text.index(R.PRED_Q10) < text.index("**Outcome.** Span over θs")


def test_the_streamed_o25_fit_is_fit_o25_tables(baseline_run, pb24_cache_dir):
    """The runner's float64 normal equations solve fit_o25_table's problem."""
    from robust.rt import baselines as B
    from robust.rt.data import pb24 as P
    from robust.rt.types import Geometry, IOPs

    R, plan, _, _ = baseline_run
    chunk = R.load_chunk(plan, plan.chunks[0], "air")
    gt = R.geometry_table(chunk, "air")
    ne = R.o25_normal_equations(plan, chunk, gt)
    batch = P.load_batch(
        realisations=PB24_FIXTURE_REALISATIONS,
        angles="all",
        cache_dir=pb24_cache_dir,
        check_fresnel=False,
    )
    iops = IOPs(
        a=np.asarray(batch.iops.a, dtype=float),
        bb_w=np.asarray(batch.iops.bb_w, dtype=float),
        bb_p=np.asarray(batch.iops.bb_p, dtype=float),
    )
    geom = Geometry(theta_s=batch.theta_s, theta_v=batch.theta_v, dphi=batch.dphi)
    Rrs = np.asarray(batch.Rrs, dtype=float)
    for name, kind, weighted, rrs_max in R.O25_FITS:
        mine = R.solve_o25(name, *ne[name], gt)
        train = R.o25_train_mask(plan, chunk, gt, kind)
        ref = B.fit_o25_table(
            iops, Rrs, geom, train=train, weighted=weighted, rrs_max=rrs_max
        )
        np.testing.assert_array_equal(mine.theta_s, ref.theta_s)
        np.testing.assert_allclose(mine.G, ref.G, rtol=1e-7, atol=1e-12)


# ------------------------------------------- the worker-pool policy (task 5b) --


def test_more_than_max_workers_is_refused_before_anything_is_loaded(tmp_path):
    """The guard runs before the plan (no ``$OS_COLOR`` needed) and names the
    override flag; 12 workers logged the box out on 2026-11-14."""
    R = load_script()
    with pytest.raises(SystemExit, match="allow-more-workers"):
        R.main(["--workers", str(R.MAX_WORKERS + 1), "--out-dir", str(tmp_path)])


def test_default_workers_and_chunk_are_the_task_5b_values():
    R = load_script()
    assert R.DEFAULT_WORKERS == 6 and R.MAX_WORKERS == 6
    assert R.CHUNK == 10
    assert R.STAGGER_S > 0 and R.WORKER_PEAK_GB > 0


def test_stagger_delays_are_spread_and_start_at_zero():
    """The n-th worker waits n * STAGGER_S; the first waits nothing."""
    import multiprocessing as mp

    R = load_script()
    counter = mp.get_context("spawn").Value("i", 0)
    delays = [R.stagger_delay(counter, 10.0) for _ in range(4)]
    assert delays == [0.0, 10.0, 20.0, 30.0]
