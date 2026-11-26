"""
Tests for the PB24 fit of ZTT's angular terms (M8 task 8b, ``rt_fit_pb24.md``).

- The committed fit (``design/ztt_pb24/angular_*.json``) is re-evaluated through
  ``rrs_ZTT(variant="pb24")`` (``psi_KLu_pb24``, ``f_L_pb24``, the floor, the
  H table), no dataset needed: the task-8b gate. The floor is bit-invisible on
  the probes.
- The shipped constants equal the committed fit; the leak guard is 8a's.
- On the committed cache miniature: the closure D* returns PB24's ``rrs`` to
  float64 rounding; the script's D-space pieces with the shipped constants
  equal ``rrs_ZTT``'s denominator; the block normal equations equal a direct
  ``lstsq``; the alternating solve recovers a synthetic (F, s, FL_AVE) exactly.
- The new functions' properties (the floor, the interpolation, the Legendre
  → u conversion), the switches (``"l23"`` / ``"tt2017"`` bit for bit; the
  variant composes from its overrides), and gradients through the angular
  inputs.
- The script runs end to end on the miniature and passes its own ``--check``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import ztt as Z
from robust.rt.types import Geometry, IOPs, PhaseParams
from robust.tests.conftest import PB24_FIXTURE_REALISATIONS

REPO = Path(__file__).resolve().parents[2]
FIT_DIR = REPO / "design" / "ztt_pb24"
SCRIPT = REPO / "design" / "py" / "fit_ztt_pb24_angular.py"


@pytest.fixture(scope="module")
def fit():
    path = FIT_DIR / "angular_coefficients.json"
    if not path.is_file():
        pytest.skip(f"no committed fit at {path}")
    coefficients = json.loads(path.read_text())
    probes = json.loads((FIT_DIR / "angular_probes.json").read_text())
    metrics = json.loads((FIT_DIR / "angular_metrics.json").read_text())
    return coefficients, probes, metrics


@pytest.fixture(scope="module")
def script():
    """The fit script as a module (restoring ``JAX_ENABLE_X64`` afterwards)."""
    before = os.environ.get("JAX_ENABLE_X64")
    spec = importlib.util.spec_from_file_location("fit_ztt_pb24_angular", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the script's dataclasses resolve through it
    spec.loader.exec_module(module)
    if before is None:
        os.environ.pop("JAX_ENABLE_X64", None)
    else:
        os.environ["JAX_ENABLE_X64"] = before
    return module


def probe_containers(p, dtype=jnp.float64, with_H=False):
    """One probe as the three containers on its single band."""
    one = lambda v: jnp.asarray([v], dtype=dtype)  # noqa: E731
    wave = one(p["wave_nm"])
    iops = IOPs(a=one(p["a"]), bb_w=one(p["bb_w"]), bb_p=one(p["bb_p"]))
    phase = PhaseParams(
        B_p=one(p["B_p"]),
        ff_B=jnp.asarray([p["ff_B"]], dtype=dtype),
        ff_weight=jnp.asarray([p["ff_weight"]], dtype=dtype),
    )
    geometry = Geometry(
        theta_s=jnp.asarray(p["theta_s"], dtype=dtype),
        theta_v=jnp.asarray(p["theta_v"], dtype=dtype),
        dphi=jnp.asarray(p["dphi"], dtype=dtype),
        diffuse_fraction=one(p["H"]) if with_H else None,
    )
    return iops, phase, geometry, wave


# ------------------------------------------------ the committed fit, shipped --
def test_shipped_constants_match_the_committed_fit(fit):
    coefficients, _, _ = fit
    assert Z.PSI_KLU_PB24_COEFFS == tuple(
        coefficients["psi_klu_coeffs_u_highest_first"]
    )
    assert len(Z.PSI_KLU_PB24_COEFFS) == coefficients["n_f_terms"]
    np.testing.assert_array_equal(Z.FL_AVE_PB24, coefficients["fl_ave_per_band"])
    np.testing.assert_array_equal(Z.FL_AVE_PB24_WAVE, coefficients["fl_ave_wave_nm"])
    assert Z.FL_S_PB24 == tuple(coefficients["fl_s_s0_s1_s2"])
    assert list(Z.D_FLOOR_PB24) == coefficients["d_floor_lo_width"]
    f = coefficients["fit"]
    assert f["quick"] is False and f["n_wave"] == 451
    assert f["train_digest"] == Z.ZTT_PB24_TRAINING_DIGEST
    assert f["n_train_realisations"] == Z.ZTT_PB24_N_TRAINING
    # FL_AVE is shipped rounded and normalised to mean 1.
    decimals = coefficients["fl_ave_decimals"]
    np.testing.assert_array_equal(np.round(Z.FL_AVE_PB24, decimals), Z.FL_AVE_PB24)
    assert float(np.mean(Z.FL_AVE_PB24)) == pytest.approx(1.0, abs=10.0**-decimals)


def test_probes_reproduce_through_rrs_ztt(fit, jax_x64):
    """**Gate**: the committed numbers come back through ``rrs_ZTT`` with the
    shipped constants -- with the H table (no ``diffuse_fraction``) and with
    the data's H supplied -- to float64 rounding, with no dataset; and the
    floor is bit-invisible on every probe."""
    _, probes, _ = fit
    for p in probes:
        for with_H in (False, True):
            iops, phase, geometry, wave = probe_containers(p, with_H=with_H)
            rrs = float(Z.rrs_ZTT(iops, phase, geometry, wave, variant="pb24")[0])
            assert rrs == pytest.approx(p["rrs_pb24_variant"], rel=1e-12)
            raw = float(
                Z.rrs_ZTT(iops, phase, geometry, wave, variant="pb24", floor=False)[0]
            )
            assert raw == rrs
        psi = jnp.asarray(p["psi"])
        assert float(Z.F_psi_pb24(psi)) == pytest.approx(p["F_pb24"], rel=1e-12)
        assert float(Z.psi_KLu_pb24(psi)) == pytest.approx(p["psi_KLu_pb24"], rel=1e-12)
        assert float(Z.f_L_pb24(psi, wave)[0]) == pytest.approx(
            p["f_L_pb24"], rel=1e-12
        )
        assert p["D_floored"] == p["D_no_floor"]
        assert float(Z.denominator_floor_pb24(p["D_no_floor"])) == p["D_no_floor"]
        assert p["rrs_PB24"] > 0


def test_metrics_record_the_predictions_outcomes(fit, jax_x64):
    """The committed metrics carry what the log quotes: the floored variant
    has no non-physical value held out, the floor's footprint is under 1%,
    and the shape check is on record for the ψ grid the test reproduces."""
    coefficients, _, m = fit
    hs = m["heldout_rrs"]["ZTT (PB24), floored"]
    assert hs["nonphysical_fraction"] == 0.0
    assert m["floor_footprint"]["fraction_moved_gt_tol"] < 0.01
    sh = m["psi_shape"]
    grid = np.arange(sh["psi_min_deg"], sh["psi_max_deg"] + 0.125, 0.25)
    Psi = np.asarray(
        Z.psi_KLu_pb24(
            jnp.asarray(grid), tuple(coefficients["psi_klu_coeffs_u_highest_first"])
        )
    )
    assert float(Psi.min()) == pytest.approx(sh["Psi_min"], rel=1e-9)
    assert float(Psi.max()) == pytest.approx(sh["Psi_max"], rel=1e-9)
    d = np.diff(Psi)
    assert bool(np.all(d <= 0) or np.all(d >= 0)) == sh["monotone"]
    assert m["joint_fit"]["chosen_n_f_terms"] == coefficients["n_f_terms"]


# ------------------------------------------------------- the functions -------
def test_floor_is_smooth_monotone_and_invisible_above(jax_x64):
    lo, width = Z.D_FLOOR_PB24
    D = jnp.asarray(np.linspace(-5.0, 10.0, 3001))
    F = np.asarray(Z.denominator_floor_pb24(D))
    assert np.all(np.diff(F) >= -1e-12)  # saturates at lo far below it
    assert np.all(np.diff(F)[np.asarray(D)[1:] > lo - 3 * width] > 0)
    assert np.all(F >= lo - 1e-12)
    far = np.asarray(D) >= lo + 40 * width
    np.testing.assert_array_equal(F[far], np.asarray(D)[far])
    assert F[0] == pytest.approx(lo, abs=1e-12)
    g = np.asarray(jax.vmap(jax.grad(Z.denominator_floor_pb24))(D))
    assert np.all(np.isfinite(g)) and np.all(g >= 0) and np.all(g <= 1 + 1e-12)
    # A custom floor through the keywords.
    assert float(Z.denominator_floor_pb24(-1.0, 2.0, 0.1)) == pytest.approx(
        2.0, abs=1e-12
    )


def test_f_L_pb24_interpolates_on_the_1nm_grid_and_is_flat_outside(jax_x64):
    psi = jnp.asarray([150.0, 180.0])
    at = lambda w: np.asarray(Z.f_L_pb24(psi[:, None], jnp.asarray(w)))  # noqa: E731
    s0, s1, s2 = Z.FL_S_PB24
    scale = (
        s0
        + s1 * np.sin(np.deg2rad([150.0, 180.0]))
        + s2 * np.cos(np.deg2rad([150.0, 180.0]))
    )
    np.testing.assert_allclose(
        at([443.0, 550.0]), scale[:, None] * Z.FL_AVE_PB24[[93, 200]], rtol=1e-14
    )
    np.testing.assert_allclose(
        at([550.5]), scale[:, None] * 0.5 * (Z.FL_AVE_PB24[200] + Z.FL_AVE_PB24[201])
    )
    np.testing.assert_array_equal(at([340.0, 810.0]), at([350.0, 800.0]))
    # The canonical grid is inside the table.
    assert np.asarray(Z.f_L_pb24(psi[:, None])).shape == (2, 81)
    # An override on its own grid.
    custom = np.asarray(
        Z.f_L_pb24(
            jnp.asarray(180.0),
            jnp.asarray([400.0, 500.0]),
            fl_ave=jnp.asarray([2.0, 4.0]),
            fl_wave=jnp.asarray([400.0, 500.0]),
            s=(1.0, 0.0, 0.0),
        )
    )
    np.testing.assert_allclose(custom, [2.0, 4.0], rtol=1e-15)


def test_psi_klu_pb24_is_a_polynomial_in_cos_psi(jax_x64):
    coeffs = (0.5, -0.25, 0.125)  # 0.5 u² − 0.25 u + 0.125, u = 1 + cos ψ
    psi = np.array([91.0, 120.0, 150.0, 180.0])
    u = 1.0 + np.cos(np.deg2rad(psi))
    expect = 1.0 + 0.5 * u**2 - 0.25 * u + 0.125
    np.testing.assert_allclose(
        np.asarray(Z.psi_KLu_pb24(jnp.asarray(psi), coeffs)), expect, rtol=1e-14
    )
    # No cancellation at 180°: Ψ(180°) = 1 + the constant term exactly.
    assert float(Z.psi_KLu_pb24(jnp.asarray(180.0), coeffs)) == 1.125


def test_legendre_to_u_conversion_is_exact(script):
    rng = np.random.default_rng(3)
    p = rng.normal(size=5)
    coeffs = script.legendre_to_u_poly(p)
    psi = np.linspace(85.0, 180.0, 97)
    direct = script.legendre_basis(script.angular_variable(psi), 5) @ p
    via_u = np.polyval(coeffs, 1.0 + np.cos(np.deg2rad(psi)))
    np.testing.assert_allclose(via_u, direct, rtol=1e-13, atol=1e-14)


# ----------------------------------------------------------- the switches ----
def test_l23_and_tt2017_are_untouched_by_the_new_switches(jax_x64):
    from robust.tests.test_ztt_pb24 import fixture_case

    args = fixture_case(jnp.float64, 0.15, 0.003, 0.0126, 0.4, 30.0)
    for variant in ("l23", "tt2017"):
        base = np.asarray(Z.rrs_ZTT(*args, variant=variant))
        explicit = np.asarray(
            Z.rrs_ZTT(*args, variant=variant, angular_model="tt", floor=False)
        )
        np.testing.assert_array_equal(base, explicit)
        # The TT pair is what the denominator uses: swapping it changes bits.
        assert np.all(
            np.asarray(Z.rrs_ZTT(*args, variant=variant, angular_model="pb24")) != base
        )
    assert Z.ANGULAR_MODELS == ("tt", "pb24", "pb24_nadir")
    with pytest.raises(ValueError, match="angular_model"):
        Z.rrs_ZTT(*args, angular_model="nope")


def test_pb24_variant_composes_from_every_override(jax_x64):
    from robust.tests.test_ztt_pb24 import fixture_case

    args = fixture_case(jnp.float64, 0.15, 0.003, 0.0126, 0.4, 30.0)
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
    # Each angular piece alone moves the result (the ladder's rungs exist).
    only_tt = np.asarray(Z.rrs_ZTT(*args, variant="pb24", angular_model="tt"))
    assert np.all(only_tt != whole)


@pytest.mark.parametrize(
    ("name", "step"),
    [("theta_v", 1e-3), ("dphi", 1e-3), ("B_p", 1e-8)],
)
def test_pb24_variant_gradient_through_the_angular_terms(jax_x64, name, step):
    """float64 ``jax.grad`` vs central differences through Ψ_KLu(ψ), f_L(ψ, λ)
    and the floor (θv = 35°, Δφ = 70°: off every node)."""
    from robust.tests.test_ztt_pb24 import fixture_case

    dtype = jnp.float64
    base = {
        "a": 0.15, "bb_p": 0.003, "B_p": 0.0126, "w_ph": 0.4, "theta_s": 33.0,
        "theta_v": 35.0, "dphi": 70.0,
    }  # fmt: skip

    def scalar(value):
        p = dict(base, **{name: value})
        return jnp.sum(Z.rrs_ZTT(*fixture_case(dtype, **p), variant="pb24"))

    x0, h = jnp.asarray(base[name], dtype=dtype), jnp.asarray(step, dtype=dtype)
    analytic = float(jax.grad(scalar)(x0))
    numeric = float((scalar(x0 + h) - scalar(x0 - h)) / (2 * h))
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_pb24_variant_is_finite_where_the_unfloored_denominator_is_not(jax_x64):
    """The floor's purpose: wherever the fitted denominator would be ≤ 0 the
    variant stays finite, positive and differentiable. The search covers the
    ψ < 110° corner and tiny B_p of M8 task 7's sign flips; if the refit
    never crosses zero there, that is recorded by skipping."""
    from robust.tests.test_ztt_pb24 import fixture_case

    dtype = jnp.float64
    found = None
    for theta_v in (60.0, 70.0, 80.0):
        for dphi in (0.0, 20.0):
            for B_p in (0.002, 0.005, 0.0126):
                for a, bb_p in ((0.02, 0.05), (0.05, 0.01), (0.2, 0.002)):
                    args = fixture_case(
                        dtype, a, bb_p, B_p, 0.4, 70.0, theta_v=theta_v, dphi=dphi
                    )
                    raw = np.asarray(Z.rrs_ZTT(*args, variant="pb24", floor=False))
                    if np.any(~np.isfinite(raw) | (raw <= 0)):
                        found = args
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is not None:
            break
    if found is None:
        pytest.skip("the refit's denominator stays positive on the search grid")
    floored = np.asarray(Z.rrs_ZTT(*found, variant="pb24"))
    assert np.all(np.isfinite(floored) & (floored > 0))

    def scalar(a_):
        iops, phase, geometry, wave = found
        iops2 = dataclasses.replace(iops, a=jnp.full(wave.shape, a_, dtype=dtype))
        return jnp.sum(Z.rrs_ZTT(iops2, phase, geometry, wave, variant="pb24"))

    g = float(jax.grad(scalar)(jnp.asarray(float(found[0].a[0]), dtype=dtype)))
    assert np.isfinite(g)


# ------------------------------------------------------- on the miniature ----
@pytest.fixture
def parts(script, pb24_cache_dir, jax_x64):
    """The fixture's window batch and its D-space pieces (float64)."""
    batch = script.load_window(PB24_FIXTURE_REALISATIONS, pb24_cache_dir)
    return batch, script.parts_of(batch)


def test_closure_d_star_returns_rrs_to_rounding(script, parts):
    """**Gate**: ``D* = [β(ψ)/bb]/(µd · rrs)`` inverts exactly."""
    batch, pt = parts
    Ds = pt.D_star
    assert np.all(np.isfinite(Ds) & (Ds > 0))
    np.testing.assert_allclose(pt.rrs_of(Ds), pt.rrs, rtol=1e-13)
    # D* exceeds twice the floor everywhere on the window (the floor's premise).
    assert float(Ds.min()) > 2.0 * Z.D_FLOOR_PB24[0]


def test_d_space_pieces_equal_rrs_ztt(script, parts, fit):
    """**Gate**: the script's ``known + gF·F + gL·f_L`` with the shipped
    constants, floored, gives ``rrs_ZTT(variant="pb24")`` on the fixture to
    float64 rounding (the data's H supplied, as the loader does)."""
    batch, pt = parts
    F, fl = script.pb24_terms(
        pt,
        Z.PSI_KLU_PB24_COEFFS,
        Z.FL_AVE_PB24,
        Z.FL_S_PB24,
        fl_wave=Z.FL_AVE_PB24_WAVE,
    )
    D = script.floored(pt.denominator(F, fl))
    ours = pt.rrs_of(D)
    theirs = np.asarray(
        Z.rrs_ZTT(
            batch.iops, batch.phase_params, batch.geometry, batch.wave, variant="pb24"
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)
    # And the TT pair reproduces the 8a variant (angular_model="tt", no floor).
    F_tt, fl_tt = script.tt_terms(pt)
    eight_a = np.asarray(
        Z.rrs_ZTT(
            batch.iops,
            batch.phase_params,
            batch.geometry,
            batch.wave,
            variant="pb24",
            angular_model="tt",
            floor=False,
        ),
        dtype=np.float64,
    )
    # 1e-9: TT's pair crosses D = 0 on a few fixture rows, where the two
    # evaluation orders cancel differently (the values there are ≤ 0 anyway).
    np.testing.assert_allclose(
        pt.rrs_of(pt.denominator(F_tt, fl_tt)), eight_a, rtol=1e-9
    )


def explicit_design(script, pt, n_f):
    """The un-factorised weighted design: F columns and the per-band f_L
    columns (3 per band), with the weighted target."""
    Ds = pt.D_star
    w = 1.0 / Ds
    y = (Ds - pt.known) * w
    A = (
        pt.gF[..., None]
        * script.legendre_basis(script.angular_variable(pt.psi), n_f)[:, None, :]
    )
    Bm = pt.gL[..., None] * script.fl_basis(pt.psi)[:, None, :]
    return A * w[..., None], Bm * w[..., None], y


def test_block_normal_equations_equal_a_direct_lstsq(script, parts):
    """**Gate**: pass A (``FL`` given) and pass B (``s`` given) solved from
    the block statistics equal ``np.linalg.lstsq`` on the explicit rows; the
    residual from the statistics equals the explicit one."""
    _, pt = parts
    n_f, W = 4, pt.wave.size
    Aw, Bw, yw = explicit_design(script, pt, n_f)
    bl = script.Blocks(n_f, W)
    Ds = pt.D_star
    bl.add(
        pt.gF[..., None]
        * script.legendre_basis(script.angular_variable(pt.psi), n_f)[:, None, :],
        pt.gL[..., None] * script.fl_basis(pt.psi)[:, None, :],
        Ds - pt.known,
        1.0 / Ds,
    )
    assert bl.n_rows == Ds.size
    FL = np.interp(pt.wave, Z.FL_AVE_WAVE, Z.FL_AVE)
    # Pass A: columns (A, Σ_w FL_w B_w) → n_f + 3 unknowns.
    X_a = np.concatenate(
        [Aw.reshape(-1, n_f), (Bw * FL[None, :, None]).reshape(-1, 3)], axis=1
    )
    ref, *_ = np.linalg.lstsq(X_a, yw.ravel(), rcond=None)
    p, s, _ = script.solve_pass_a(bl, FL)
    np.testing.assert_allclose(np.concatenate([p, s]), ref, rtol=1e-8, atol=1e-12)
    # Pass B: columns (A, one per band B_w s) → n_f + W unknowns.
    n = Ds.shape[0]
    band_cols = np.zeros((n, W, W))
    band_cols[:, np.arange(W), np.arange(W)] = Bw @ s
    X_b = np.concatenate([Aw.reshape(-1, n_f), band_cols.reshape(-1, W)], axis=1)
    ref, *_ = np.linalg.lstsq(X_b, yw.ravel(), rcond=None)
    p2, FL2, _ = script.solve_pass_b(bl, s)
    np.testing.assert_allclose(np.concatenate([p2, FL2]), ref, rtol=1e-8, atol=1e-12)
    # The residual from the statistics equals the explicit one.
    explicit = X_b @ np.concatenate([p2, FL2]) - yw.ravel()
    assert bl.residual_ss(p2, s, FL2) == pytest.approx(
        float(explicit @ explicit), rel=1e-8
    )
    # Chunk-order determinism: the same rows in two chunks sum to the same bits.
    half = n // 2
    two = script.Blocks(n_f, W)
    for sl in (slice(0, half), slice(half, None)):
        two.add(
            (
                pt.gF[..., None]
                * script.legendre_basis(script.angular_variable(pt.psi), n_f)[
                    :, None, :
                ]
            )[sl],
            (pt.gL[..., None] * script.fl_basis(pt.psi)[:, None, :])[sl],
            (Ds - pt.known)[sl],
            (1.0 / Ds)[sl],
        )
    np.testing.assert_allclose(two.AtA, bl.AtA, rtol=1e-13)
    np.testing.assert_allclose(two.Bty, bl.Bty, rtol=1e-13)


def test_alternating_solve_recovers_a_synthetic_fit(script, parts):
    """With ``rrs`` made from known (F, s, FL_AVE) on the fixture's IOPs and
    geometry, the fixed-count alternating passes give them back."""
    _, pt = parts
    n_f = 4
    rng = np.random.default_rng(8)
    p_true = np.array([0.05, -0.3, 0.1, 0.02])
    s_true = np.array([0.95, 0.05, -0.02])  # f_L ≤ 1: D > 0 at every b̃b
    FL_true = 1.0 + 0.03 * np.sin(np.linspace(0, 6, pt.wave.size))
    FL_true /= FL_true.mean()
    F = (script.legendre_basis(script.angular_variable(pt.psi), n_f) @ p_true)[:, None]
    fl = FL_true[None, :] * (script.fl_basis(pt.psi) @ s_true)[:, None]
    D_true = pt.denominator(F, fl)
    assert np.all(D_true > 0)
    synthetic = dataclasses.replace(pt, rrs=pt.rrs_of(D_true))
    Ds = synthetic.D_star
    bl = script.Blocks(n_f, pt.wave.size)
    bl.add(
        pt.gF[..., None]
        * script.legendre_basis(script.angular_variable(pt.psi), n_f)[:, None, :],
        pt.gL[..., None] * script.fl_basis(pt.psi)[:, None, :],
        Ds - synthetic.known,
        1.0 / Ds,
    )
    out = script.solve_alternating(
        bl, np.ones(pt.wave.size) + 0.01 * rng.normal(size=pt.wave.size), n_rounds=8
    )
    np.testing.assert_allclose(out["p_legendre"], p_true, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(out["s"], s_true, rtol=1e-8)
    np.testing.assert_allclose(out["FL"], FL_true, rtol=1e-8)
    assert out["history"][-1]["max_rel_change"] < 1e-9
    assert bl.rrms(out["p_legendre"], out["s"], out["FL"]) < 1e-9


def test_script_runs_on_the_miniature_and_passes_its_check(
    script, pb24_cache_dir, tmp_path
):
    """End to end on the committed 3-realisation cache (no ``$OS_COLOR``):
    train {1, 2500}, held out {993}; then ``--check`` byte for byte."""
    out = tmp_path / "out"
    argv = [
        "--cache-dir", str(pb24_cache_dir), "--out-dir", str(out),
        "--workers", "0", "--no-figures", "--quiet",
    ]  # fmt: skip
    assert script.main(argv) == 0
    c = json.loads((out / "angular_coefficients.json").read_text())
    assert c["fit"]["n_train_realisations"] == 2
    assert c["fit"]["n_test_realisations"] == 1
    assert c["fit"]["n_wave"] == 46 and len(c["fl_ave_per_band"]) == 46
    assert len(c["psi_klu_coeffs_u_highest_first"]) == script.N_F_TERMS
    m = json.loads((out / "angular_metrics.json").read_text())
    assert m["heldout_rrs"]["ZTT (PB24), floored"]["nonphysical_fraction"] == 0.0
    assert set(m["D1_shape_heldout"]) >= {
        "ZTT (PB24), rrs",
        "ZTT 8a (TT angular terms), rrs",
    }
    probes = json.loads((out / "angular_probes.json").read_text())
    assert len(probes) == 3
    assert script.main([*argv, "--check"]) == 0
