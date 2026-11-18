"""
Tests for the PB24 fit of ZTT's IOP-side terms (M8 task 8a, ``rt_fit_pb24.md``).

- The committed fit (``design/ztt_pb24/fit_*.json``) is re-evaluated through
  the functions the package ships (``ztt.mu_infinity_pb24``, ``ztt.mu_d_pb24``,
  ``ztt.diffuse_fraction_pb24``), no dataset needed: the task-8a gate.
- The shipped constants equal the committed fit; the leak guard pins the
  training ids; the ``"l23"`` and ``"tt2017"`` variants are untouched (their
  own pins in ``test_ztt_l23.py`` cover that; here only the new option is
  checked to compose as documented).
- The H table is the files' value (on the committed cache miniature), and the
  interpolator's properties and gradients are checked.
- The fit script's building blocks (design matrices, the streamed QR, the
  joint µd fit, the node-to-quadratic step) are checked on synthetic data, and
  the script runs end to end on the miniature with a synthetic eigen product
  whose Equation (8) coefficients it must recover.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import ztt as Z
from robust.rt.types import Geometry, IOPs, PhaseParams
from robust.tests.conftest import PB24_FIXTURE_REALISATIONS

REPO = Path(__file__).resolve().parents[2]
FIT_DIR = REPO / "design" / "ztt_pb24"
SCRIPT = REPO / "design" / "py" / "fit_ztt_pb24.py"


@pytest.fixture(scope="module")
def fit():
    path = FIT_DIR / "fit_coefficients.json"
    if not path.is_file():
        pytest.skip(f"no committed fit at {path}")
    coefficients = json.loads(path.read_text())
    probes = json.loads((FIT_DIR / "fit_probes.json").read_text())
    return coefficients, probes


@pytest.fixture(scope="module")
def script():
    """The fit script as a module (it sets ``JAX_ENABLE_X64`` for its own runs;
    JAX is configured already, so that is a no-op here and is restored)."""
    before = os.environ.get("JAX_ENABLE_X64")
    spec = importlib.util.spec_from_file_location("fit_ztt_pb24", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if before is None:
        os.environ.pop("JAX_ENABLE_X64", None)
    else:
        os.environ["JAX_ENABLE_X64"] = before
    return module


# ------------------------------------------------ the committed fit, shipped --
def test_shipped_constants_match_the_committed_fit(fit):
    coefficients, _ = fit
    assert Z.MU_INF_PB24_COEFFS == tuple(coefficients["mu_inf_coeffs_m1_m16"])
    assert Z.MD_STAR_PB24_COEFFS == tuple(coefficients["md_star_coeffs_m1_m8"])
    assert Z.MD_PLUS_CORR_PB24 == tuple(coefficients["md_plus_corr_c1_c2"])
    assert Z.ZTT_PB24_BOX == tuple(coefficients["clamp_box_X_lo_hi_L_lo_hi"])
    assert Z.MU_INF_PB24_RANGE == tuple(coefficients["mu_inf_out_range"])
    assert Z.MD_STAR_PB24_RANGE == tuple(coefficients["md_star_out_range"])
    assert Z.ZTT_PB24_N_TRAINING == coefficients["fit"]["n_train_realisations"]
    assert Z.ZTT_PB24_TRAINING_DIGEST == coefficients["fit"]["train_digest"]
    assert coefficients["fit"]["quick"] is False
    assert coefficients["fit"]["H_file"] == Z.H_PB24_FILE


def test_probes_reproduce_through_ztt(fit, jax_x64):
    """**Gate**: the committed numbers come back through the shipped
    ``*_pb24`` evaluators (constants, clamps, guards and the H table), to
    float64 rounding, with no dataset."""
    _, probes = fit
    for p in probes:
        wave = jnp.asarray([p["wave_nm"]])
        H = Z.diffuse_fraction_pb24(p["theta_s"], wave)
        assert float(H[0]) == pytest.approx(p["H"], rel=1e-12)
        mu = Z.mu_infinity_pb24(p["bb_over_a"], p["eta_bb"])
        assert float(mu) == pytest.approx(p["mu_inf_pb24"], rel=1e-12)
        mud = Z.mu_d_pb24(p["theta_s"], p["bb_over_a"], p["eta_bb"], wave=wave)
        assert float(mud[0]) == pytest.approx(p["mu_d_pb24"], rel=1e-12)
        # The physical target is on record beside the fit's value.
        assert 0.0 < p["mu_inf_eigen"] <= 1.0


def test_the_fit_was_made_on_training_rows_only():
    """**The leak guard** (as the Fresnel transfer's): the shipped digest is
    that of the realisation split's training ids, and any held-out id in the
    fitting set would change it."""
    from robust.rt.data import pb24 as P

    content = P.read_splits()
    every = np.asarray(content["realisations"], dtype=int)
    test = np.asarray(content["realisation"]["test"], dtype=int)
    train = np.setdiff1d(every, test)
    assert np.intersect1d(train, test).size == 0 and test.size > 0
    assert Z.ZTT_PB24_N_TRAINING == train.size
    assert C.realisation_digest(train) == Z.ZTT_PB24_TRAINING_DIGEST
    assert C.realisation_digest(every) != Z.ZTT_PB24_TRAINING_DIGEST
    assert C.realisation_digest(np.append(train, test[0])) != Z.ZTT_PB24_TRAINING_DIGEST


def test_metrics_record_the_predictions_outcomes(fit):
    """The committed metrics carry the held-out numbers the log quotes."""
    m = json.loads((FIT_DIR / "fit_metrics.json").read_text())
    per = m["mu_d_heldout"]["PB24_fit"]["per_theta_s"]
    assert all(per[f"{z:g}"]["mape"] < 0.011 for z in (0, 10, 20, 30, 40, 50, 60, 70))
    assert m["mu_inf_heldout_vs_eigen"]["PB24_fit"]["all"]["mape"] < 0.015
    assert m["gate_evaluator_agreement"]["mu_inf"] < 1e-5
    assert m["gate_evaluator_agreement"]["mu_d"] < 1e-5


# ------------------------------------------------------- the H table ----------
def test_h_table_is_the_files_value_and_realisation_independent(pb24_cache_dir):
    """On the committed miniature: ``1 − Esdir_Es_ratio`` of every realisation
    equals the shipped table at every node and band, bit for bit (the table
    stores the float32 values)."""
    from robust.rt.data import pb24_cache

    cache = pb24_cache.open_cache(pb24_cache_dir)
    sec_nodes, wave_nodes, table, provenance = Z._h_pb24_table()
    theta_s = np.asarray(cache.grids["theta_s"], dtype=np.float64)
    order = np.argsort(1.0 / np.cos(np.deg2rad(theta_s)))
    np.testing.assert_array_equal(wave_nodes, np.asarray(cache.grids["lambda"], float))
    np.testing.assert_allclose(
        sec_nodes, 1.0 / np.cos(np.deg2rad(theta_s[order])), rtol=1e-15
    )
    for rid in PB24_FIXTURE_REALISATIONS:
        H = 1.0 - np.asarray(cache.table("Esdir_Es_ratio", rid)).astype(np.float64)
        np.testing.assert_array_equal(H[order], table)
    assert provenance["n_training_realisations"] == Z.ZTT_PB24_N_TRAINING
    assert provenance["training_digest"] == Z.ZTT_PB24_TRAINING_DIGEST
    assert "RADTRAN" in provenance["about"]


def test_h_interpolation_is_linear_in_air_mass_and_flat_outside(jax_x64):
    sec_nodes, wave_nodes, table, _ = Z._h_pb24_table()
    wave = jnp.asarray([443.0, 550.0, 700.0])
    at = lambda th: np.asarray(Z.diffuse_fraction_pb24(th, wave))  # noqa: E731
    # Exact at the nodes (0° and 60° are nodes).
    k0 = int(np.argmin(np.abs(sec_nodes - 1.0)))
    k60 = int(np.argmin(np.abs(sec_nodes - 2.0)))
    jw = np.searchsorted(wave_nodes, np.asarray(wave))
    np.testing.assert_allclose(at(0.0), table[k0, jw], rtol=1e-12)
    np.testing.assert_allclose(at(60.0), table[k60, jw], rtol=1e-12)
    # Half-way in air mass between the 50° and 60° nodes: the mean of the two.
    k50 = int(np.argmin(np.abs(sec_nodes - 1.0 / np.cos(np.deg2rad(50.0)))))
    mid = float(np.degrees(np.arccos(2.0 / (sec_nodes[k50] + sec_nodes[k60]))))
    np.testing.assert_allclose(
        at(mid), 0.5 * (table[k50, jw] + table[k60, jw]), rtol=1e-12
    )
    # Half-way in λ between two bands: the mean of the two.
    H_mid = np.asarray(Z.diffuse_fraction_pb24(0.0, jnp.asarray([550.5])))
    np.testing.assert_allclose(H_mid[0], 0.5 * (table[k0, 200] + table[k0, 201]))
    # Flat outside: beyond the last sun node and beyond the band range.
    np.testing.assert_array_equal(at(89.99), at(87.75))
    np.testing.assert_array_equal(
        np.asarray(Z.diffuse_fraction_pb24(30.0, jnp.asarray([340.0, 810.0]))),
        np.asarray(Z.diffuse_fraction_pb24(30.0, jnp.asarray([350.0, 800.0]))),
    )
    # Monotone in the sun angle and in (0, 1).
    h = at(jnp.asarray(np.linspace(0.0, 87.75, 60)))
    assert np.all(np.diff(h, axis=0) >= 0) and np.all((h > 0) & (h < 1))
    # Shapes: (n,) -> (n, W); (n, 1) -> (n, W); scalar -> (W,).
    assert at(jnp.asarray([10.0, 20.0, 30.0])).shape == (3, 3)
    assert at(jnp.asarray([[10.0], [20.0]])).shape == (2, 3)
    assert at(10.0).shape == (3,)


def test_h_gradient_matches_finite_differences(jax_x64):
    wave = jnp.asarray([443.0, 550.0])

    def scalar(th):
        return jnp.sum(Z.diffuse_fraction_pb24(th, wave))

    for th0 in (33.0, 57.0):
        x0, h = jnp.asarray(th0), jnp.asarray(1e-3)
        analytic = float(jax.grad(scalar)(x0))
        numeric = float((scalar(x0 + h) - scalar(x0 - h)) / (2 * h))
        assert analytic == pytest.approx(numeric, rel=1e-6)
        assert analytic > 0


# ------------------------------------------------------- the variant ----------
def fixture_case(dtype, a, bb_p, B_p, w_ph, theta_s, theta_v=20.0, dphi=60.0):
    wave = jnp.asarray([440.0, 550.0, 660.0], dtype=dtype)
    iops = IOPs(
        a=jnp.full(wave.shape, a, dtype=dtype),
        bb_w=jnp.asarray(C.bb_w(wave), dtype=dtype),
        bb_p=jnp.full(wave.shape, bb_p, dtype=dtype),
    )
    w = jnp.stack([jnp.full(wave.shape, w_ph), jnp.full(wave.shape, 1.0 - w_ph)], -1)
    phase = PhaseParams(
        B_p=jnp.full(wave.shape, B_p, dtype=dtype),
        ff_B=jnp.broadcast_to(jnp.asarray([0.006, 0.015], dtype=dtype), w.shape),
        ff_weight=w.astype(dtype),
    )
    geometry = Geometry(
        theta_s=jnp.asarray(theta_s, dtype=dtype),
        theta_v=jnp.asarray(theta_v, dtype=dtype),
        dphi=jnp.asarray(dphi, dtype=dtype),
    )
    return iops, phase, geometry, wave


def test_options_include_pb24_and_compose(jax_x64):
    """``variant="pb24"`` equals the L23 variant with every override set to
    ``"pb24"`` and the floor on (the FF Pbb is shared; task 8b added the
    angular pair and the floor); it differs from the other variants."""
    args = fixture_case(jnp.float64, 0.15, 0.003, 0.0126, 0.4, 30.0)
    assert "pb24" in Z.VARIANTS and "pb24" in Z.MU_INF_MODELS
    assert "pb24" in Z.MU_D_MODELS and Z.DEFAULT_VARIANT == "l23"
    whole = np.asarray(Z.rrs_ZTT(*args, variant="pb24"))
    composed = np.asarray(
        Z.rrs_ZTT(
            *args,
            variant="l23",
            mu_inf_model="pb24",
            mu_d_model="pb24",
            angular_model="pb24",
            floor=True,
        )
    )
    np.testing.assert_array_equal(whole, composed)
    for other in ("l23", "tt2017"):
        assert np.all(np.asarray(Z.rrs_ZTT(*args, variant=other)) != whole)


def test_pb24_default_H_is_the_table_and_a_supplied_H_overrides_it(jax_x64):
    iops, phase, geometry, wave = fixture_case(
        jnp.float64, 0.15, 0.003, 0.0126, 0.4, 45.0
    )
    base = np.asarray(Z.rrs_ZTT(iops, phase, geometry, wave, variant="pb24"))
    table_H = Z.diffuse_fraction_pb24(geometry.theta_s, wave)
    with_table = dataclasses.replace(geometry, diffuse_fraction=table_H)
    np.testing.assert_array_equal(
        np.asarray(Z.rrs_ZTT(iops, phase, with_table, wave, variant="pb24")), base
    )
    tt_H = jnp.broadcast_to(Z.diffuse_fraction(geometry.theta_s), wave.shape)
    with_tt = dataclasses.replace(geometry, diffuse_fraction=tt_H)
    other = np.asarray(Z.rrs_ZTT(iops, phase, with_tt, wave, variant="pb24"))
    assert np.all(other != base)
    # Through the µd ratio only: rrs ∝ 1/µd.
    X, eta = iops.bb / iops.a, iops.bb_w / iops.bb
    ratio = np.asarray(Z.mu_d_pb24(45.0, X, eta, H=table_H)) / np.asarray(
        Z.mu_d_pb24(45.0, X, eta, H=tt_H)
    )
    np.testing.assert_allclose(other / base, ratio, rtol=1e-12)


@pytest.mark.parametrize(
    ("name", "step"),
    [("a", 1e-6), ("bb_p", 1e-9), ("B_p", 1e-8), ("theta_s", 1e-3), ("w_ph", 1e-6)],
)
def test_pb24_variant_gradient_matches_finite_differences(jax_x64, name, step):
    """**Gate.** float64 ``jax.grad`` vs central differences through the PB24
    constants, clamps, guards and the H table (θs = 33°, between two nodes)."""
    dtype = jnp.float64
    base = {"a": 0.15, "bb_p": 0.003, "B_p": 0.0126, "w_ph": 0.4, "theta_s": 33.0}

    def scalar(value):
        p = dict(base, **{name: value})
        return jnp.sum(Z.rrs_ZTT(*fixture_case(dtype, **p), variant="pb24"))

    x0, h = jnp.asarray(base[name], dtype=dtype), jnp.asarray(step, dtype=dtype)
    analytic = float(jax.grad(scalar)(x0))
    numeric = float((scalar(x0 + h) - scalar(x0 - h)) / (2 * h))
    assert analytic == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize(
    ("a", "bb_p"),
    [(0.002, 0.2), (5.0, 1e-5), (0.05, 3e-4), (1e-3, 1.0)],
    ids=["bb/a≫box", "bb/a≪box", "inside", "η→0"],
)
def test_pb24_variant_is_finite_at_and_beyond_the_box(jax_x64, a, bb_p):
    dtype = jnp.float64

    def scalar(a_, bbp_):
        iops, phase, geometry, wave = fixture_case(dtype, a_, bbp_, 0.0126, 0.4, 30.0)
        return jnp.sum(Z.rrs_ZTT(iops, phase, geometry, wave, variant="pb24"))

    value = scalar(a, bb_p)
    grads = jax.grad(scalar, argnums=(0, 1))(
        jnp.asarray(a, dtype=dtype), jnp.asarray(bb_p, dtype=dtype)
    )
    assert np.isfinite(float(value))
    assert all(np.isfinite(float(g)) for g in grads)


def test_pb24_terms_are_physical_over_the_box(jax_x64):
    """µ∞ in (0, 1] and M*_d near 1 across the clamp box: the guards hold."""
    X = np.linspace(Z.ZTT_PB24_BOX[0] - 1, Z.ZTT_PB24_BOX[1] + 1, 41)
    L = np.linspace(Z.ZTT_PB24_BOX[2] - 1, Z.ZTT_PB24_BOX[3] + 1, 41)
    XX, LL = np.meshgrid(X, L)
    mu = np.asarray(Z.mu_infinity_pb24(10.0**XX, 10.0**LL))
    assert np.all((mu > 0.0) & (mu < 1.0))
    md = np.asarray(
        Z.Md_star(
            10.0**XX,
            10.0**LL,
            Z.MD_STAR_PB24_COEFFS,
            box=Z.ZTT_PB24_BOX,
            out_range=Z.MD_STAR_PB24_RANGE,
        )
    )
    assert np.all((md > 0.5) & (md < 1.1))


# ------------------------------------------------------ the fit script --------
def test_design_matrices_match_ztt_order(script, jax_x64):
    rng = np.random.default_rng(1)
    X, Lg = rng.uniform(-4, 1, 9), rng.uniform(-5, -0.1, 9)
    m16, m8 = rng.normal(size=16), rng.normal(size=8)
    np.testing.assert_allclose(
        script.mu_inf_design(X, Lg) @ m16,
        np.asarray(Z.mu_infinity(10**X, 10**Lg, tuple(m16))),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        script.md_star_design(X, Lg) @ m8,
        np.asarray(Z.Md_star(10**X, 10**Lg, tuple(m8))),
        rtol=1e-12,
    )


def test_streaming_lsq_equals_lstsq_and_is_deterministic(script):
    rng = np.random.default_rng(2)
    D = rng.normal(size=(1000, 6)) * np.array([1, 10, 100, 1, 0.1, 1e3])
    b = D @ rng.normal(size=6) + 0.01 * rng.normal(size=1000)
    one = script.StreamingLSQ(6)
    one.add(D, b)
    chunked = script.StreamingLSQ(6)
    for start in range(0, 1000, 70):
        chunked.add(D[start : start + 70], b[start : start + 70])
    again = script.StreamingLSQ(6)
    for start in range(0, 1000, 70):
        again.add(D[start : start + 70], b[start : start + 70])
    ref, *_ = np.linalg.lstsq(D, b, rcond=None)
    x1, r1, c1 = one.solve()
    x2, r2, c2 = chunked.solve()
    x3, _, _ = again.solve()
    np.testing.assert_allclose(x1, ref, rtol=1e-9)
    np.testing.assert_allclose(x2, ref, rtol=1e-9)
    np.testing.assert_array_equal(x2, x3)  # same chunks, same bits
    assert r1 == pytest.approx(np.linalg.norm(D @ ref - b), rel=1e-9)
    assert c1 == pytest.approx(np.linalg.cond(D), rel=1e-6)
    assert one.n_rows == chunked.n_rows == 1000


def test_joint_mu_d_fit_recovers_known_coefficients_at_eight_suns(script, jax_x64):
    rng = np.random.default_rng(3)
    n = 300
    nodes = np.array([0.0, 10, 20, 30, 40, 50, 60, 70])
    X, Lg = rng.uniform(-4, 1, n), rng.uniform(-4.5, -0.05, n)
    m8 = np.asarray(Z.MD_STAR_PB24_COEFFS)
    corr = (-0.14, 0.02)
    theta = np.repeat(nodes, n)
    XX, LL = np.tile(X, nodes.size), np.tile(Lg, nodes.size)
    mud = (
        np.asarray(Z.Md_plus(theta, H=0.2))
        * np.asarray(Z.md_plus_correction(theta, corr))
        * (script.md_star_design(XX, LL) @ m8)
    )
    t = mud / np.asarray(Z.Md_plus(theta, H=0.2))
    md8, got_corr, c_nodes = script.fit_mu_d(XX, LL, theta, t, nodes)
    np.testing.assert_allclose(md8, m8, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(got_corr, corr, rtol=1e-8)
    for z, c in c_nodes.items():
        assert c == pytest.approx(float(Z.md_plus_correction(z, corr)), rel=1e-8)


def test_mu_inf_fit_recovers_known_coefficients(script):
    rng = np.random.default_rng(4)
    X, Lg = rng.uniform(-4, 1, 600), rng.uniform(-4.5, -0.05, 600)
    target = script.mu_inf_design(X, Lg) @ np.asarray(Z.MU_INF_PB24_COEFFS)
    keep = target > 0.05
    got, cond = script.fit_mu_inf(X[keep], Lg[keep], target[keep])
    np.testing.assert_allclose(got, Z.MU_INF_PB24_COEFFS, rtol=1e-7, atol=1e-10)
    assert cond > 1


def test_corr_from_nodes_is_exact_on_a_quadratic(script):
    nodes = np.array([10.0, 20, 30, 40, 50, 60, 70])
    u = 1 - np.cos(np.deg2rad(nodes))
    c = 1 - 0.1 * u - 0.15 * u**2
    corr, at_nodes = script.corr_from_nodes(nodes, c)
    np.testing.assert_allclose(corr, (-0.1, -0.15), rtol=1e-12)
    np.testing.assert_allclose(at_nodes, c, rtol=1e-12)


def test_script_runs_on_the_miniature_and_recovers_a_known_mu_inf(
    script, pb24_cache_dir, tmp_path
):
    """End to end on the committed 3-realisation cache (no ``$OS_COLOR``), with
    a synthetic eigen product whose µ∞ *is* Equation (8) with the shipped
    coefficients: the script must give them back, write the H table that
    equals the shipped one, and pass its own ``--check``."""
    from robust.rt.data import pb24_cache, pb24_eigen

    cache = pb24_cache.open_cache(pb24_cache_dir)
    ids = cache.realisations
    tab = script.read_table(cache, ids, script.IOP_NAMES)
    X, Lg, _ = script.iop_logs(tab)
    mu = script.mu_inf_design(X, Lg) @ np.asarray(Z.MU_INF_PB24_COEFFS)
    assert np.all(mu > 0.05), "synthetic target must be positive on the fixture"
    eigen = tmp_path / "eigen.npz"
    pb24_eigen.save_mu_inf_eigen(
        eigen,
        realisations=ids,
        wave=cache.grids["lambda"],
        mu_inf=mu,
        B_ph=np.full(ids.size, 0.01),
        B_nap=np.full(ids.size, 0.015),
        provenance={"synthetic": True},
    )
    out = tmp_path / "out"
    h_file = tmp_path / "h.npz"
    argv = [
        "--cache-dir", str(pb24_cache_dir), "--eigen", str(eigen),
        "--out-dir", str(out), "--h-file", str(h_file), "--no-figures", "--quiet",
    ]  # fmt: skip
    assert script.main(argv) == 0
    coefficients = json.loads((out / "fit_coefficients.json").read_text())
    assert coefficients["fit"]["n_train_realisations"] == 2  # 1 and 2500 train
    assert coefficients["fit"]["n_test_realisations"] == 1  # 993 held out
    np.testing.assert_allclose(
        coefficients["mu_inf_coeffs_m1_m16"], Z.MU_INF_PB24_COEFFS, rtol=1e-6
    )
    with np.load(h_file) as data:
        _, _, shipped, _ = Z._h_pb24_table()
        theta = np.asarray(data["theta_s"])
        order = np.argsort(1.0 / np.cos(np.deg2rad(theta)))
        np.testing.assert_array_equal(np.asarray(data["H"])[order], shipped)
    probes = json.loads((out / "fit_probes.json").read_text())
    assert len(probes) == 3 and {p["realisation"] for p in probes} == set(ids)
    # Deterministic: a second run compares byte for byte.
    assert script.main([*argv, "--check"]) == 0
