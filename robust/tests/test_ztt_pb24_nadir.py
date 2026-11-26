"""
Tests for the nadir-only PB24 fit of ZTT (M8 task 9, ``rt_fit_pb24.md`` Q34–Q44).

- The committed fit (``design/ztt_pb24/nadir_*.json``) is re-evaluated through
  ``rrs_ZTT(variant="pb24_nadir")`` with no dataset; the shipped constants
  equal the committed fit; the leak guard is 8a's digest.
- The committed metrics record the gate's outcome (it **fails**), so the
  default variant is still ``"l23"``.
- The variant composes from its overrides, leaves the other variants bit for
  bit, warns (``DomainWarning``) off-nadir on concrete angles only, holds
  Ψ_KLu flat below its ψ range, and is differentiable.
- The script's pieces: the Legendre → u conversion, the f_L alternating solve
  against a synthetic truth, its composition against ``rrs_ZTT``; and an end to
  end run with ``--check`` on the committed cache miniature.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import hybrid as H
from robust.rt import ztt as Z
from robust.rt.types import Geometry, IOPs, PhaseParams
from robust.tests.test_ztt_pb24 import fixture_case

REPO = Path(__file__).resolve().parents[2]
FIT_DIR = REPO / "design" / "ztt_pb24"
SCRIPT = REPO / "design" / "py" / "fit_ztt_pb24_nadir.py"
FINAL = "+ floor = ZTT (PB24_nadir)"


@pytest.fixture(scope="module")
def fit():
    path = FIT_DIR / "nadir_coefficients.json"
    if not path.is_file():
        pytest.skip(f"no committed fit at {path}")
    return tuple(
        json.loads((FIT_DIR / f"nadir_{name}.json").read_text())
        for name in ("coefficients", "probes", "metrics")
    )


@pytest.fixture(scope="module")
def script():
    """The fit script as a module (restoring ``JAX_ENABLE_X64`` afterwards)."""
    before = os.environ.get("JAX_ENABLE_X64")
    spec = importlib.util.spec_from_file_location("fit_ztt_pb24_nadir", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if before is None:
        os.environ.pop("JAX_ENABLE_X64", None)
    else:
        os.environ["JAX_ENABLE_X64"] = before
    return module


def probe_containers(p, with_H: bool):
    """One probe as the three containers on its single band, at nadir."""
    dtype = jnp.float64
    one = lambda v: jnp.asarray([v], dtype=dtype)  # noqa: E731
    iops = IOPs(a=one(p["a"]), bb_w=one(p["bb_w"]), bb_p=one(p["bb_p"]))
    phase = PhaseParams(
        B_p=one(p["B_p"]),
        ff_B=jnp.asarray([p["ff_B"]], dtype=dtype),
        ff_weight=jnp.asarray([p["ff_weight"]], dtype=dtype),
    )
    geometry = Geometry(
        theta_s=jnp.asarray(p["theta_s"], dtype=dtype),
        theta_v=jnp.asarray(0.0, dtype=dtype),
        dphi=jnp.asarray(0.0, dtype=dtype),
        diffuse_fraction=one(p["H"]) if with_H else None,
    )
    return iops, phase, geometry, one(p["wave_nm"])


# ------------------------------------------------ the committed fit, shipped --
def test_shipped_constants_match_the_committed_fit(fit):
    c, _, _ = fit
    assert tuple(c["psi_klu_coeffs_u_highest_first"]) == Z.PSI_KLU_PB24_NADIR_COEFFS
    assert len(Z.PSI_KLU_PB24_NADIR_COEFFS) == c["n_f_terms"]
    assert c["psi_min_deg"] == Z.PSI_KLU_PB24_NADIR_PSI_MIN
    np.testing.assert_array_equal(Z.FL_AVE_PB24_NADIR, c["fl_ave_per_band"])
    np.testing.assert_array_equal(Z.FL_AVE_PB24_WAVE, c["fl_ave_wave_nm"])
    assert tuple(c["fl_s_s0_s1"]) == Z.FL_S_PB24_NADIR
    assert list(Z.D_FLOOR_PB24) == c["d_floor_lo_width"]
    assert c["psi_klu_node_weighting"] == "geometric"
    f = c["fit"]
    assert f["quick"] is False and f["n_wave"] == 451
    # The leak guard: fitted on the realisation split's training side only.
    assert f["train_digest"] == Z.ZTT_PB24_TRAINING_DIGEST
    assert f["n_train_realisations"] == Z.ZTT_PB24_N_TRAINING
    assert f["n_rows_psi"] == 4000 * 8 * 451 == f["n_rows_fl"]
    decimals = c["fl_ave_decimals"]
    np.testing.assert_array_equal(
        np.round(Z.FL_AVE_PB24_NADIR, decimals), Z.FL_AVE_PB24_NADIR
    )
    assert float(np.mean(Z.FL_AVE_PB24_NADIR)) == pytest.approx(1.0, abs=1e-6)


def test_probes_reproduce_through_rrs_ztt(fit, jax_x64):
    """**Gate**: the committed numbers come back through ``rrs_ZTT`` with the
    shipped constants -- with the H table and with the data's H supplied -- to
    float64 rounding, with no dataset."""
    _, probes, _ = fit
    assert len(probes) == 12
    for p in probes:
        for with_H in (False, True):
            args = probe_containers(p, with_H)
            rrs = float(Z.rrs_ZTT(*args, variant="pb24_nadir")[0])
            assert rrs == pytest.approx(p["rrs_variant"], rel=1e-12)
        psi = jnp.asarray(p["psi"])
        assert float(Z.psi_KLu_pb24_nadir(psi)) == pytest.approx(
            p["psi_KLu"], rel=1e-12
        )
        assert float(Z.f_L_pb24_nadir(psi, args[3])[0]) == pytest.approx(
            p["f_L"], rel=1e-12
        )
        assert float(Z.denominator_floor_pb24(p["D_no_floor"])) == pytest.approx(
            p["D_floored"], rel=1e-12
        )
        assert p["rrs_PB24"] > 0


def test_metrics_record_the_gate_and_the_default_follows_it(fit):
    """The acceptance gate was written before the fit (Q42) and its outcome
    is in the committed metrics. It fails, so ``"pb24_nadir"`` is opt-in and
    the default variant has not moved."""
    _, _, m = fit
    g = m["gate"]
    final = m["heldout_rrs"][FINAL]["window"]
    assert g["median_abs"] == final["median_abs"]
    assert g["median_pass"] == (final["median_abs"] < g["median_below"] == 0.04)
    assert g["nonphysical_pass"] is True and final["nonphysical_fraction"] == 0.0
    assert g["l23_pass"] == (
        g["l23_heldout_rrms"] <= g["l23_reference"] + g["l23_tolerance_pp"]
    )
    assert g["pass"] == (g["median_pass"] and g["nonphysical_pass"] and g["l23_pass"])
    assert g["pass"] is False
    assert Z.DEFAULT_VARIANT == "l23"
    # The numbers the docstrings quote.
    gordon = m["heldout_rrs"]["standard Gordon"]["window"]["median_abs"]
    assert 100 * gordon == pytest.approx(6.52, abs=0.005)
    assert 100 * final["median_abs"] == pytest.approx(11.00, abs=0.005)
    d2 = m["diagnostics"]["D2 ceiling: measured KLu/a, per-band f_L(λ), 8a's µd"]
    assert 100 * d2["window"]["median_abs"] == pytest.approx(2.84, abs=0.005)


def test_fitted_psi_klu_meets_its_nodes(fit, jax_x64):
    """The polynomial is within the 1% node rule at the eight fitted suns and
    is the measured K_Lu/K∞ there (0.80 at ψ = 180°, rising with the sun)."""
    _, _, m = fit
    pf = m["psi_fit"]
    ts = np.asarray(pf["theta_s"])
    win = ts <= 70.0
    Psi = np.asarray(Z.psi_KLu_pb24_nadir(jnp.asarray(pf["psi_deg"])))
    np.testing.assert_allclose(Psi, pf["fitted"], rtol=1e-12)
    nodes = np.asarray(pf["node_value"])
    assert np.max(np.abs(Psi[win] / nodes[win] - 1.0)) <= pf["node_tol"]
    assert np.all(np.diff(Psi[win]) > 0) and 0.79 < Psi[0] < 0.82


# ------------------------------------------------------- the variant's API --
def test_variant_composes_and_leaves_the_others_alone(jax_x64):
    args = fixture_case(jnp.float64, 0.15, 0.003, 0.0126, 0.4, 30.0, theta_v=0.0)
    assert Z.VARIANTS == ("l23", "tt2017", "pb24", "pb24_nadir")
    assert Z.ANGULAR_MODELS == ("tt", "pb24", "pb24_nadir")
    whole = np.asarray(Z.rrs_ZTT(*args, variant="pb24_nadir"))
    composed = np.asarray(
        Z.rrs_ZTT(
            *args,
            variant="l23",
            mu_inf_model="pb24",
            mu_d_model="pb24",
            angular_model="pb24_nadir",
            floor=True,
        )
    )
    np.testing.assert_array_equal(whole, composed)
    from_pb24 = np.asarray(Z.rrs_ZTT(*args, variant="pb24", angular_model="pb24_nadir"))
    np.testing.assert_array_equal(whole, from_pb24)
    for other in ("l23", "tt2017", "pb24"):
        assert np.all(whole != np.asarray(Z.rrs_ZTT(*args, variant=other)))
    assert np.all(np.isfinite(whole)) and np.all(whole > 0)


def test_off_nadir_warns_on_concrete_angles_only(jax_x64):
    """Q40 (b): a ``DomainWarning`` beyond ``NADIR_TOL_DEG`` when ``theta_v``
    is concrete; silent at nadir, under ``jit`` and for the other variants."""
    assert H.DomainWarning is C.DomainWarning
    nadir = fixture_case(jnp.float64, 0.15, 0.003, 0.0126, 0.4, 30.0, theta_v=0.0)
    off = fixture_case(jnp.float64, 0.15, 0.003, 0.0126, 0.4, 30.0, theta_v=20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Z.rrs_ZTT(*nadir, variant="pb24_nadir")
        within = Geometry(
            theta_s=nadir[2].theta_s,
            theta_v=jnp.asarray(Z.NADIR_TOL_DEG),
            dphi=nadir[2].dphi,
        )
        Z.rrs_ZTT(nadir[0], nadir[1], within, nadir[3], variant="pb24_nadir")
        Z.rrs_ZTT(*off, variant="pb24")
        Z.rrs_ZTT(*off)

        @jax.jit
        def traced(theta_v):
            g = Geometry(theta_s=off[2].theta_s, theta_v=theta_v, dphi=off[2].dphi)
            return Z.rrs_ZTT(off[0], off[1], g, off[3], variant="pb24_nadir")

        assert np.all(np.isfinite(np.asarray(traced(jnp.asarray(20.0)))))
    with pytest.warns(C.DomainWarning, match="calibrated at nadir only"):
        out = Z.rrs_ZTT(*off, variant="pb24_nadir")
    assert np.all(np.isfinite(np.asarray(out)))


def test_psi_klu_is_flat_below_its_range(jax_x64):
    lo = Z.PSI_KLU_PB24_NADIR_PSI_MIN
    at = float(Z.psi_KLu_pb24_nadir(jnp.asarray(lo)))
    for psi in (lo - 1.0, 110.0, 91.0):
        assert float(Z.psi_KLu_pb24_nadir(jnp.asarray(psi))) == at
    assert float(Z.psi_KLu_pb24_nadir(jnp.asarray(lo + 1.0))) != at
    # f_L is TT's two-term form: no cos ψ term.
    wave = jnp.asarray([440.0, 551.5])
    s0, s1 = Z.FL_S_PB24_NADIR
    shape = np.interp(np.asarray(wave), Z.FL_AVE_PB24_WAVE, Z.FL_AVE_PB24_NADIR)
    np.testing.assert_allclose(
        np.asarray(Z.f_L_pb24_nadir(jnp.asarray(150.0), wave)),
        shape * (s0 + s1 * np.sin(np.deg2rad(150.0))),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("name", "step"), [("theta_s", 1e-3), ("a", 1e-6), ("bb_p", 1e-7), ("B_p", 1e-8)]
)
def test_variant_gradient_matches_finite_differences(jax_x64, name, step):
    """float64 ``jax.grad`` vs central differences through Ψ_KLu(ψ(θs)),
    f_L, the IOP-side terms and the floor, at nadir off every node."""
    base = {
        "a": 0.15, "bb_p": 0.003, "B_p": 0.0126, "w_ph": 0.4, "theta_s": 33.0,
        "theta_v": 0.0, "dphi": 0.0,
    }  # fmt: skip

    def scalar(value):
        p = dict(base, **{name: value})
        return jnp.sum(Z.rrs_ZTT(*fixture_case(jnp.float64, **p), variant="pb24_nadir"))

    x0 = jnp.asarray(base[name], dtype=jnp.float64)
    h = jnp.asarray(step, dtype=jnp.float64)
    analytic = float(jax.grad(scalar)(x0))
    numeric = float((scalar(x0 + h) - scalar(x0 - h)) / (2 * h))
    assert analytic == pytest.approx(numeric, rel=1e-6)


# ------------------------------------------------------ the script's pieces --
def test_legendre_to_u_conversion(script):
    rng = np.random.default_rng(9)
    p = rng.normal(size=5)
    psi = np.linspace(script.Z.PSI_KLU_PB24_NADIR_PSI_MIN, 180.0, 40)
    direct = script.psi_basis(psi, 5) @ p
    via_u = np.polyval(script.legendre_to_u_poly(p), script.u_of(psi))
    np.testing.assert_allclose(via_u, direct, rtol=1e-12, atol=1e-13)
    assert script.U_MAX == pytest.approx(1.0 + np.cos(np.deg2rad(130.0)))


def test_fl_alternating_solve_recovers_a_synthetic_truth(script):
    """The closure statistics and the alternating passes: rows built from a
    known ``FL(λ) · (s0 + s1 sin ψ)`` come back to rounding, and the residual
    computed from the statistics is zero."""
    rng = np.random.default_rng(3)
    n, S, W = 40, 8, 11
    psi = np.linspace(180.0, 135.5, S)
    FL = 1.0 + 0.02 * rng.normal(size=W)
    FL /= FL.mean()
    c = np.array([1.03, 0.02])
    h = np.outer(script.s_basis(psi, 2) @ c, FL)  # (S, W)
    gL = -rng.uniform(1.0, 60.0, size=(n, 1, W))
    w = rng.uniform(0.01, 1.0, size=(n, S, W))
    y = gL * h[None]
    st = {
        "G2": np.sum((w * gL) ** 2, axis=0),
        "Gy": np.sum(w * w * gL * y, axis=0),
        "yy": float(np.sum((w * y) ** 2)),
        "n": n * S * W,
    }
    sol = script.solve_fl(st, psi, 2)
    np.testing.assert_allclose(sol["FL"], FL, rtol=1e-10)
    np.testing.assert_allclose(sol["c"], c, rtol=1e-10)
    assert sol["history"][-1]["max_rel_change"] < 1e-12
    assert script.closure_rrms(st, sol["h"]) < 1e-7
    # Pass A alone equals a direct least-squares solve for (s0, s1) given FL.
    A = (w * gL * FL)[..., None] * script.s_basis(psi, 2)[None, :, None, :]
    direct, *_ = np.linalg.lstsq(A.reshape(-1, 2), (w * y).ravel(), rcond=None)
    np.testing.assert_allclose(direct, c, rtol=1e-10)


def test_script_composition_equals_rrs_ztt(script, fit, jax_x64):
    """The script's own assembly of Equation (12) with the shipped constants
    is ``rrs_ZTT(variant="pb24_nadir")`` -- so what the script scores is what
    the package ships -- and its ladder specs are the package's switches."""
    c, _, _ = fit
    model = {
        "coeffs": tuple(c["psi_klu_coeffs_u_highest_first"]),
        "FL": np.asarray(c["fl_ave_per_band"]),
        "fl_wave": np.asarray(c["fl_ave_wave_nm"]),
        "s": tuple(c["fl_s_s0_s1"]),
    }
    rows = [
        fixture_case(jnp.float64, a, bbp, Bp, 0.4, ts, theta_v=0.0)
        for a, bbp, Bp, ts in (
            (0.15, 0.003, 0.0126, 0.0),
            (0.02, 0.0004, 0.004, 33.0),
            (0.6, 0.2, 0.03, 61.0),
        )
    ]
    stack = lambda xs: jnp.stack(xs)  # noqa: E731
    iops = IOPs(
        *(stack([getattr(r[0], k) for r in rows]) for k in ("a", "bb_w", "bb_p"))
    )
    phase = PhaseParams(
        B_p=stack([r[1].B_p for r in rows]),
        ff_B=stack([r[1].ff_B for r in rows]),
        ff_weight=stack([r[1].ff_weight for r in rows]),
    )
    wave = rows[0][3]
    theta_s = stack([r[2].theta_s for r in rows])
    Hs = Z.diffuse_fraction_pb24(theta_s, wave)
    geometry = Geometry(
        theta_s=theta_s,
        theta_v=jnp.zeros(3),
        dphi=jnp.zeros(3),
        diffuse_fraction=Hs,
    )
    pc = script.pieces(iops, phase, geometry, wave, Hs)
    mine, _ = script.variant_rrs(pc, model)
    shipped = np.asarray(Z.rrs_ZTT(iops, phase, geometry, wave, variant="pb24_nadir"))
    np.testing.assert_allclose(mine, shipped, rtol=1e-12)
    rung_a, _ = script.variant_rrs(pc, model, ("pb24", "pb24", "tt", "tt", False))
    np.testing.assert_allclose(
        rung_a,
        np.asarray(
            Z.rrs_ZTT(
                iops, phase, geometry, wave, variant="pb24",
                angular_model="tt", floor=False,
            )
        ),
        rtol=1e-12,
    )  # fmt: skip
    base, _ = script.variant_rrs(pc, model, ("l23", "l23", "tt", "tt", False))
    np.testing.assert_allclose(
        base,
        np.asarray(Z.rrs_ZTT(iops, phase, geometry, wave, variant="l23")),
        rtol=1e-12,
    )


def test_script_runs_on_the_miniature_and_passes_its_check(
    script, pb24_cache_dir, tmp_path
):
    """End to end on the committed 3-realisation cache (no ``$OS_COLOR``):
    train {1, 2500}, held out {993}; then ``--check`` byte for byte."""
    out = tmp_path / "out"
    argv = [
        "--cache-dir", str(pb24_cache_dir), "--out-dir", str(out),
        "--no-figures", "--no-l23", "--quiet",
    ]  # fmt: skip
    assert script.main(argv) == 0
    c = json.loads((out / "nadir_coefficients.json").read_text())
    assert c["fit"]["n_train_realisations"] == 2
    assert c["fit"]["n_test_realisations"] == 1
    assert c["fit"]["n_wave"] == 46 and len(c["fl_ave_per_band"]) == 46
    assert c["fit"]["n_rows_psi"] == 2 * 8 * 46
    m = json.loads((out / "nadir_metrics.json").read_text())
    assert m["heldout_rrs"][FINAL]["window"]["nonphysical_fraction"] == 0.0
    assert m["gate"]["l23_pass"] is None and m["gate"]["pass"] is False
    assert m["azimuth_spread_at_nadir"] == 0.0
    assert len(json.loads((out / "nadir_probes.json").read_text())) == 12
    assert (out / "nadir_per_band.csv").read_text().count("\n") == 47
    assert script.main([*argv, "--check"]) == 0
