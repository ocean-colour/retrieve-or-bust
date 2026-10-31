"""
Tests for the ZTT-refit evaluators and the committed L23 fit (M7 task 3,
``rt_fit_ztt.md``).

The task-3 gate is that the fitted coefficients reproduce the fit's own numbers
when evaluated **through the functions the model ships** (``ztt.mu_infinity``
and ``ztt.mu_d`` with ``box`` / ``md_plus_corr``). The fit script records a
dozen probe points (``design/ztt_l23/probes.json``); here they are re-evaluated
through ``ztt`` with no dataset needed. The new keywords must also leave the
published model untouched when they are left at their defaults.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import ztt as Z

FIT_DIR = Path(__file__).resolve().parents[2] / "design" / "ztt_l23"
SCRIPT = Path(__file__).resolve().parents[2] / "design" / "py" / "fit_ztt_l23.py"


@pytest.fixture(scope="module")
def fit():
    path = FIT_DIR / "coefficients.json"
    if not path.is_file():
        pytest.skip(f"no committed fit at {path}")
    coefficients = json.loads(path.read_text())
    probes = json.loads((FIT_DIR / "probes.json").read_text())
    return coefficients, probes


@pytest.fixture(scope="module")
def script():
    """The fit script as a module, for its design-matrix helpers.

    It sets ``JAX_ENABLE_X64`` on import (for its own runs); JAX is already
    configured by then, so that has no effect here, and it is restored anyway.
    """
    before = os.environ.get("JAX_ENABLE_X64")
    spec = importlib.util.spec_from_file_location("fit_ztt_l23", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if before is None:
        os.environ.pop("JAX_ENABLE_X64", None)
    else:
        os.environ["JAX_ENABLE_X64"] = before
    return module


# ---------------------------------------------- defaults stay the paper's --
def test_mu_d_defaults_are_the_published_equation():
    theta = jnp.asarray([0.0, 30.0, 60.0])
    x = jnp.asarray([0.003, 0.03, 0.08])
    eta = jnp.asarray([0.8, 0.4, 0.1])
    expect = Z.Md_plus(theta) * Z.Md_star(x, eta)
    np.testing.assert_array_equal(np.asarray(Z.mu_d(theta, x, eta)), np.asarray(expect))


def test_md_plus_correction():
    assert float(Z.md_plus_correction(0.0, (0.3, -0.2))) == 1.0
    u = 1 - np.cos(np.deg2rad(60.0))
    assert float(Z.md_plus_correction(60.0, (0.3, -0.2))) == pytest.approx(
        1 + 0.3 * u - 0.2 * u**2, rel=1e-6
    )


# ------------------------------------------------------------- the clamp --
def test_smooth_clip_identity_inside_bounded_outside(jax_x64):
    x = jnp.linspace(-1.0, 2.0, 301)
    y = Z.smooth_clip(x, 0.0, 1.0)
    inside = (x > 0.2) & (x < 0.8)
    np.testing.assert_allclose(y[inside], x[inside], atol=1e-12)
    assert float(y.min()) >= -Z.CLAMP_WIDTH * np.log(2) - 1e-12
    assert float(y.max()) <= 1.0 + Z.CLAMP_WIDTH * np.log(2) + 1e-12
    # Monotone up to rounding (far outside it saturates to the bound, where
    # float64 jitters by 1 ulp), and strictly increasing across the box.
    assert np.all(np.diff(np.asarray(y)) >= -1e-15)
    assert np.all(np.diff(np.asarray(y)[(x > -0.05) & (x < 1.05)]) > 0)
    g = jax.vmap(jax.grad(lambda v: Z.smooth_clip(v, 0.0, 1.0)))(x)
    # Finite everywhere; positive across the box. Far outside, the analytic
    # gradient (~e^{-d/width}) underflows and rounding can leave ±1e-87.
    assert np.all(np.isfinite(g)) and np.all(np.asarray(g) > -1e-12)
    assert np.all(np.asarray(g)[(x > -0.05) & (x < 1.05)] > 0)


def test_clamped_polynomials_are_flat_far_outside(fit, jax_x64):
    """Beyond the box the inputs saturate, so the output stops changing; without
    the clamp the same polynomial runs away."""
    coefficients, _ = fit
    box = coefficients["clamp_box_X_lo_hi_L_lo_hi"]
    m16 = tuple(coefficients["mu_inf_coeffs_m1_m16"])
    far = Z.mu_infinity(10 ** (box[1] + 3), 10 ** (box[3] + 1), m16, box=box)
    farther = Z.mu_infinity(10 ** (box[1] + 6), 10 ** (box[3] + 2), m16, box=box)
    assert float(far) == pytest.approx(float(farther), abs=1e-12)
    wild = Z.mu_infinity(10 ** (box[1] + 3), 10 ** (box[3] + 1), m16)
    assert abs(float(wild) - float(far)) > 0.5


# ---------------------------------------------------- the committed fit --
def test_probes_reproduce_through_ztt(fit, jax_x64):
    """**Gate**: the committed coefficients, through the shipped evaluators,
    give back the numbers the fit script computed."""
    coefficients, probes = fit
    box = coefficients["clamp_box_X_lo_hi_L_lo_hi"]
    m16 = tuple(coefficients["mu_inf_coeffs_m1_m16"])
    md8 = tuple(coefficients["md_star_coeffs_m1_m8"])
    corr = tuple(coefficients["md_plus_corr_c1_c2"])
    mu_range = tuple(coefficients["mu_inf_out_range"])
    md_range = tuple(coefficients["md_star_out_range"])
    for p in probes:
        got = Z.mu_infinity(
            p["bb_over_a"], p["eta_bb"], m16, box=box, out_range=mu_range
        )
        closure = Z.mu_infinity(
            p["bb_over_a"],
            p["eta_bb"],
            tuple(coefficients["mu_inf_closure_coeffs_m1_m16"]),
            box=box,
            out_range=tuple(coefficients["mu_inf_closure_out_range"]),
        )
        assert float(closure) == pytest.approx(p["mu_inf_L23_closure"], rel=1e-12)
        assert float(got) == pytest.approx(p["mu_inf_L23"], rel=1e-12)
        for zen, value in p["mu_d_L23"].items():
            got = Z.mu_d(
                float(zen),
                p["bb_over_a"],
                p["eta_bb"],
                md_star_coeffs=md8,
                md_plus_corr=corr,
                box=box,
                md_star_range=md_range,
            )
            assert float(got) == pytest.approx(value, rel=1e-12)


def test_fit_is_physical_over_its_box(fit, jax_x64):
    """Over the whole fitted box -- including its empty corners, where L23 has
    no data and the bare polynomial reaches −1.1 and 1.6 -- the guarded
    refits stay physical."""
    coefficients, _ = fit
    box = coefficients["clamp_box_X_lo_hi_L_lo_hi"]
    X, Lg = np.meshgrid(
        np.linspace(box[0] - 1, box[1] + 1, 60), np.linspace(box[2] - 1, box[3] + 1, 60)
    )
    mu_inf = Z.mu_infinity(
        10**X,
        10**Lg,
        tuple(coefficients["mu_inf_coeffs_m1_m16"]),
        box=box,
        out_range=tuple(coefficients["mu_inf_out_range"]),
    )
    assert np.all((np.asarray(mu_inf) > 0.4) & (np.asarray(mu_inf) < 1.0))
    for zen in (0.0, 30.0, 60.0):
        mud = Z.mu_d(
            zen,
            10**X,
            10**Lg,
            md_star_coeffs=tuple(coefficients["md_star_coeffs_m1_m8"]),
            md_plus_corr=tuple(coefficients["md_plus_corr_c1_c2"]),
            box=box,
            md_star_range=tuple(coefficients["md_star_out_range"]),
        )
        assert np.all((np.asarray(mud) > 0.5) & (np.asarray(mud) <= 1.0))


def test_metrics_record_the_headline(fit):
    """The committed metrics say what the record says (held-out, vs targets)."""
    metrics = json.loads((FIT_DIR / "metrics.json").read_text())
    assert metrics["mu_inf_heldout_vs_eigen"]["L23_fit"]["mape"] < 0.005
    for zen in ("0", "30", "60"):
        assert metrics["mu_d_heldout"][zen]["L23_fit"]["mape"] < 0.01
    assert metrics["mu_d_heldout"]["60"]["TT"]["mape"] > 0.05  # the bias fixed
    gate = metrics["gate_evaluator_agreement"]
    assert gate["mu_inf"] < 1e-5 and gate["mu_d"] < 1e-5


# ------------------------------------------------------ the fit's algebra --
def test_mu_inf_design_matches_ztt_order(script, jax_x64):
    rng = np.random.default_rng(1)
    m16 = rng.normal(size=16)
    X, Lg = rng.uniform(-3, -1, 7), rng.uniform(-1.5, -0.1, 7)
    own = script.mu_inf_design(X, Lg) @ m16
    via = np.asarray(Z.mu_infinity(10**X, 10**Lg, tuple(m16)))
    np.testing.assert_allclose(own, via, rtol=1e-12)


def test_md_star_design_matches_ztt_order(script, jax_x64):
    rng = np.random.default_rng(2)
    m8 = rng.normal(size=8)
    X, Lg = rng.uniform(-3, -1, 7), rng.uniform(-1.5, -0.1, 7)
    own = script.md_star_design(X, Lg) @ m8
    via = np.asarray(Z.Md_star(10**X, 10**Lg, tuple(m8)))
    np.testing.assert_allclose(own, via, rtol=1e-12)


def test_joint_mu_d_fit_recovers_known_coefficients(script, jax_x64):
    """Noise-free synthetic µd: the joint LSQ recovers M*_d and (c1, c2)."""
    rng = np.random.default_rng(3)
    n = 400
    X, Lg = rng.uniform(-3.5, -0.5, n), rng.uniform(-1.5, -0.05, n)
    m8 = np.asarray(Z.MD_STAR_COEFFS)
    corr = (-0.1, -0.15)
    theta = np.repeat(np.asarray([0.0, 30.0, 60.0]), n)
    XX, LL = np.tile(X, 3), np.tile(Lg, 3)
    mud = (
        np.asarray(Z.Md_plus(theta))
        * np.asarray(Z.md_plus_correction(theta, corr))
        * (script.md_star_design(XX, LL) @ m8)
    )
    t = mud / np.asarray(Z.Md_plus(theta))
    md8, got_corr, _ = script.fit_mu_d(XX, LL, theta, t)
    np.testing.assert_allclose(md8, m8, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got_corr, corr, rtol=1e-8)


def test_mu_inf_fit_recovers_known_coefficients(script):
    rng = np.random.default_rng(4)
    X, Lg = rng.uniform(-3.5, -0.5, 500), rng.uniform(-1.5, -0.05, 500)
    m16 = rng.normal(scale=0.1, size=16)
    m16[-1] = 0.8  # keep targets positive
    target = script.mu_inf_design(X, Lg) @ m16
    target = np.where(target > 0.05, target, 0.05)
    keep = target > 0.05
    got, _ = script.fit_mu_inf(X[keep], Lg[keep], target[keep])
    np.testing.assert_allclose(
        script.mu_inf_design(X[keep], Lg[keep]) @ got, target[keep], rtol=1e-9
    )


# ================================================== task 4: the variant wired in
from robust.rt import conventions as C  # noqa: E402
from robust.rt.types import Geometry, IOPs, PhaseParams  # noqa: E402

#: SHA-256 of ``rrs_ZTT`` on the committed 50-scene fixture as it was **before
#: M7 task 4** (computed from the committed ``ztt.py`` at 9f3328a).
#: ``variant="tt2017"`` must reproduce it to the bit: the old model is a choice
#: now, not a casualty.
PRE_M7_TT2017_SHA256 = (
    "c353d5bea5bf8cd6d90bf8b7fb0544f52ce31ede7262fe7f8633a933223ab54c"
)


def fixture_args():
    """The committed 50-scene L23 fixture as ``rrs_ZTT`` arguments."""
    from robust.rt.data import l23 as L
    from robust.tests.conftest import L23_SMALL_FIXTURE

    if not L23_SMALL_FIXTURE.is_file():
        pytest.skip("fixture missing")
    b = L.load_batch(reader=L.npz_reader(L23_SMALL_FIXTURE))
    return b.iops, b.phase_params, b.geometry, b.wave


def test_tt2017_variant_is_bit_for_bit_the_pre_m7_model():
    """**Gate.** ``variant="tt2017"`` == the pre-M7 ``rrs_ZTT`` on the fixture."""
    import hashlib

    rrs = np.asarray(Z.rrs_ZTT(*fixture_args(), variant="tt2017"))
    assert hashlib.sha256(rrs.tobytes()).hexdigest() == PRE_M7_TT2017_SHA256


def test_default_variant_is_l23_and_differs():
    args = fixture_args()
    assert Z.DEFAULT_VARIANT == "l23"
    np.testing.assert_array_equal(
        np.asarray(Z.rrs_ZTT(*args)), np.asarray(Z.rrs_ZTT(*args, variant="l23"))
    )
    assert not np.array_equal(
        np.asarray(Z.rrs_ZTT(*args)), np.asarray(Z.rrs_ZTT(*args, variant="tt2017"))
    )


def test_component_switches_compose(jax_x64):
    """``mu_inf_model`` / ``mu_d_model`` swap exactly one term, and the explicit
    overrides take precedence: ``mu_inf`` > ``mu_inf_coeffs`` > ``mu_inf_model``."""
    args = fixture_args()
    iops = args[0]
    x, eta = iops.bb / iops.a, iops.bb_w / iops.bb
    # tt2017 backbone with the L23 µ∞ == tt2017 with that µ∞ passed explicitly.
    np.testing.assert_array_equal(
        np.asarray(Z.rrs_ZTT(*args, variant="tt2017", mu_inf_model="l23")),
        np.asarray(
            Z.rrs_ZTT(*args, variant="tt2017", mu_inf=Z.mu_infinity_l23(x, eta))
        ),
    )
    # Every switch flipped to L23 on the tt2017 variant == the l23 variant, up to
    # the phase function (the fixture carries no FF fields, so both use Sullivan).
    np.testing.assert_array_equal(
        np.asarray(
            Z.rrs_ZTT(*args, variant="tt2017", mu_inf_model="l23", mu_d_model="l23")
        ),
        np.asarray(Z.rrs_ZTT(*args, variant="l23")),
    )
    # mu_inf beats mu_inf_model.
    np.testing.assert_array_equal(
        np.asarray(Z.rrs_ZTT(*args, mu_inf=0.8, mu_inf_model="tt2017")),
        np.asarray(Z.rrs_ZTT(*args, mu_inf=0.8)),
    )


def test_options_are_validated():
    args = fixture_args()
    with pytest.raises(ValueError, match="variant"):
        Z.rrs_ZTT(*args, variant="ztt2018")
    with pytest.raises(ValueError, match="mu_inf_model"):
        Z.rrs_ZTT(*args, mu_inf_model="eigen")
    with pytest.raises(ValueError, match="mu_d_model"):
        Z.rrs_ZTT(*args, mu_d_model="l23_closure")
    # "l23_closure" exists since task 5, and is never what a default selects.
    assert Z.DEFAULT_VARIANT != "l23_closure"
    Z.rrs_ZTT(*args, mu_inf_model="l23_closure")


def l23_case(dtype, a, bb_p, B_p, w_ph, theta_s):
    """One L23-like sample, three bands, with the FF mixture set."""
    wave = jnp.asarray([440.0, 550.0, 660.0], dtype=dtype)
    iops = IOPs(
        a=jnp.full(wave.shape, a, dtype=dtype),
        bb_w=jnp.asarray(C.bb_w(wave), dtype=dtype),
        bb_p=jnp.full(wave.shape, bb_p, dtype=dtype),
    )
    w = jnp.stack([jnp.full(wave.shape, w_ph), jnp.full(wave.shape, 1.0 - w_ph)], -1)
    phase = PhaseParams(
        B_p=jnp.full(wave.shape, B_p, dtype=dtype),
        ff_B=jnp.broadcast_to(jnp.asarray([0.010, 0.018], dtype=dtype), w.shape),
        ff_weight=w.astype(dtype),
    )
    return iops, phase, Geometry.nadir(jnp.asarray(theta_s, dtype=dtype)), wave


@pytest.mark.parametrize(
    ("name", "step"),
    [("a", 1e-6), ("bb_p", 1e-9), ("B_p", 1e-8), ("theta_s", 1e-3), ("w_ph", 1e-6)],
)
def test_l23_variant_gradient_matches_finite_differences(jax_x64, name, step):
    """**Gate.** float64 ``jax.grad`` vs central differences for the L23
    variant -- refit µ∞ and µd, clamps, and the FF phase function all live."""
    dtype = jnp.float64
    base = {"a": 0.15, "bb_p": 0.003, "B_p": 0.0126, "w_ph": 0.4, "theta_s": 30.0}

    def scalar(value):
        p = dict(base, **{name: value})
        return jnp.sum(Z.rrs_ZTT(*l23_case(dtype, **p), variant="l23"))

    x0 = jnp.asarray(base[name], dtype=dtype)
    h = jnp.asarray(step, dtype=dtype)
    analytic = float(jax.grad(scalar)(x0))
    numeric = float((scalar(x0 + h) - scalar(x0 - h)) / (2 * h))
    assert analytic == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize(
    ("a", "bb_p"),
    [(0.002, 0.05), (5.0, 1e-5), (0.05, 3e-4), (1e-3, 1.0)],
    ids=["bb/a≫box", "bb/a≪box", "inside", "η→0"],
)
def test_l23_variant_is_finite_at_and_beyond_the_box(jax_x64, a, bb_p):
    """**Gate.** Far outside the fitted (bb/a, η) box, rrs and its gradients
    stay finite: the clamps hold rather than letting the polynomials run.

    *Finite*, not *positive*. At bb/a ≈ 25 (and in the η → 0 case) ZTT's rrs is
    negative in **both** variants. That is structural, from Eq. (12)'s
    f_L(1 − 1/b̃b) + 1/b̃b once b̃b is small, and no µ∞/µd refit can change it
    (record §7.16 met the same thing on PB24). Inside L23's range it is positive."""
    dtype = jnp.float64

    def scalar(a_, bbp_):
        iops, phase, geometry, wave = l23_case(dtype, a_, bbp_, 0.0126, 0.4, 30.0)
        return jnp.sum(Z.rrs_ZTT(iops, phase, geometry, wave))

    value = scalar(a, bb_p)
    grads = jax.grad(scalar, argnums=(0, 1))(
        jnp.asarray(a, dtype=dtype), jnp.asarray(bb_p, dtype=dtype)
    )
    assert np.isfinite(float(value))
    assert all(np.isfinite(float(g)) for g in grads)
    if 1e-4 < bb_p / a < 0.6:  # inside L23's bb/a range
        assert float(value) > 0


# ------------------------------------------------- the emulator's backbone --
def test_shipped_emulator_is_pinned_to_tt2017():
    from robust.rt import emulator as E

    assert E.load_default().backbone_variant == "tt2017"


def test_emulator_save_load_round_trips_the_variant(tmp_path):
    import dataclasses

    from robust.rt import emulator as E

    shipped = E.load_default()
    for variant in ("tt2017", "l23"):
        path = tmp_path / f"{variant}.npz"
        E.save(dataclasses.replace(shipped, backbone_variant=variant), path)
        assert E.load(path).backbone_variant == variant


def test_a_pre_m7_file_loads_as_tt2017(tmp_path):
    """A weights file without the key -- every file made before M7 -- loads as
    trained on TT2017, which is what it was."""
    from robust.rt import emulator as E

    with np.load(E.DEFAULT_WEIGHTS) as data:
        assert "backbone_variant" not in data.files  # the shipped file is pre-M7
    assert E.load(E.DEFAULT_WEIGHTS).backbone_variant == "tt2017"


def test_hybrid_follows_the_emulators_backbone():
    """The same weights relabelled "l23" give a different hybrid -- the
    correction is applied to the backbone the emulator names."""
    import dataclasses

    from robust.rt import emulator as E
    from robust.rt import hybrid as H

    args = fixture_args()
    shipped = E.load_default()
    relabelled = dataclasses.replace(shipped, backbone_variant="l23")
    a = H.rrs_forward(*args, "hybrid", emulator=shipped, check_domain=False)
    b = H.rrs_forward(*args, "hybrid", emulator=relabelled, check_domain=False)
    assert not np.array_equal(np.asarray(a), np.asarray(b))
    # (ztt + δ) − δ: equal to float32 rounding, not bitwise.
    np.testing.assert_allclose(
        np.asarray(
            b
            - H.rrs_forward(*args, "emulator", emulator=relabelled, check_domain=False)
        ),
        np.asarray(Z.rrs_ZTT(*args, variant="l23")),
        rtol=1e-5,
        atol=1e-12,
    )


# ======================================================= task 5: the closure µ∞
def test_shipped_closure_constants_match_the_committed_fit(fit):
    coefficients, _ = fit
    assert Z.MU_INF_L23_CLOSURE_COEFFS == tuple(
        coefficients["mu_inf_closure_coeffs_m1_m16"]
    )
    assert Z.MU_INF_L23_CLOSURE_RANGE == tuple(coefficients["mu_inf_closure_out_range"])
    # ...and so are the task-3 ones, which a task-5 refit must not have moved.
    assert Z.MU_INF_L23_COEFFS == tuple(coefficients["mu_inf_coeffs_m1_m16"])
    assert Z.MD_STAR_L23_COEFFS == tuple(coefficients["md_star_coeffs_m1_m8"])


def test_closure_reproduces_the_reference_exactly(jax_x64):
    """**Gate (self-consistency).** The closure µ∞, put back into the L23
    variant, gives back the reference rrs (float64 rounding)."""
    iops, phase, geometry, wave = (
        jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64), a)
        for a in fixture_args()
    )
    from robust.rt.data import l23 as L
    from robust.tests.conftest import L23_SMALL_FIXTURE

    batch = L.load_batch(reader=L.npz_reader(L23_SMALL_FIXTURE))
    truth = jnp.asarray(C.Rrs_to_rrs(batch.Rrs), jnp.float64)
    mu = Z.mu_inf_closure(iops, phase, geometry, truth, wave)
    back = Z.rrs_ZTT(iops, phase, geometry, wave, mu_inf=mu)
    np.testing.assert_allclose(np.asarray(back), np.asarray(truth), rtol=1e-12)
    # The fixture has no FF fields, so this closure uses Sullivan's Pbb, and
    # then 8.4% of it exceeds 1. With L23's own FF phase function (the release
    # run, fit_ztt_l23.py) only 2 of 806,760 samples do. The phase function
    # decides whether µ∞ can absorb the rest. Pinned so the contrast stays seen.
    physical = np.mean((np.asarray(mu) > 0) & (np.asarray(mu) <= 1))
    assert 0.88 < physical < 0.95


def test_closure_model_is_the_closure_fit(jax_x64):
    args = fixture_args()
    iops = args[0]
    x, eta = iops.bb / iops.a, iops.bb_w / iops.bb
    np.testing.assert_array_equal(
        np.asarray(Z.rrs_ZTT(*args, mu_inf_model="l23_closure")),
        np.asarray(Z.rrs_ZTT(*args, mu_inf=Z.mu_infinity_l23_closure(x, eta))),
    )


def test_closure_variant_is_closer_to_the_reference_on_the_fixture():
    """The ceiling is a ceiling: on real scenes the closure fit beats the
    physical µ∞ in rrs (the fixture has no FF fields, so both use Sullivan)."""
    from robust.rt import validation as V
    from robust.rt.data import l23 as L
    from robust.tests.conftest import L23_SMALL_FIXTURE

    args = fixture_args()
    batch = L.load_batch(reader=L.npz_reader(L23_SMALL_FIXTURE))
    truth = C.Rrs_to_rrs(batch.Rrs)
    physical = float(V.rrms(truth, Z.rrs_ZTT(*args)))
    closure = float(V.rrms(truth, Z.rrs_ZTT(*args, mu_inf_model="l23_closure")))
    assert closure < physical


# ==================================================== task 6: the acceptance gate
def test_committed_ladder_records_a_pass():
    """The committed ladder (``design/py/run_ztt_l23.py``) says the gate passed,
    at every zenith -- readable without the data."""
    import csv

    path = FIT_DIR / "ladder.csv"
    if not path.is_file():
        pytest.skip("no committed ladder")
    with path.open() as fh:
        rows = {r["model"]: r for r in csv.DictReader(fh)}
    new, old = rows["+ FF Pbb = ZTT (L23 µ∞)"], rows["ZTT (TT2017 µ∞)"]
    for z in (0, 30, 60):
        assert float(new[f"held-out {z}°"]) < float(old[f"held-out {z}°"])


def test_acceptance_gate_on_the_release():
    """**The M7 acceptance gate**, recomputed: on held-out scenes, ZTT (L23 µ∞)
    beats ZTT (TT2017 µ∞) at each of 0°, 30°, 60° (measured 2.91 / 2.56 /
    3.18% vs 4.26 / 4.67 / 8.11%)."""
    from robust.rt import validation as V
    from robust.rt.data import l23 as L
    from robust.tests.conftest import l23_available

    if not l23_available():
        pytest.skip("L23 not available")
    batch = L.load_batch()
    held = L.make_splits(batch).scene_test
    truth = C.Rrs_to_rrs(batch.Rrs)
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    new = Z.rrs_ZTT(*args)
    old = Z.rrs_ZTT(*args, variant="tt2017")
    for z in (0, 30, 60):
        m = held & (batch.zenith == z)
        assert float(V.rrms(truth[m], new[m])) < float(V.rrms(truth[m], old[m])), z
