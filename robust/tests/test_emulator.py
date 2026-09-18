"""
Tests for :mod:`robust.rt.emulator` (M3) — the residual emulator and its claims.

The module docstring of ``emulator.py`` makes specific design claims; these tests
pin them, layered by what they need:

**No data at all.** The feature map's shape/order/broadcasting contract, the
scale invariance of the backbone that justifies the feature set, the exact-zero
start, the tanh bound, the ``History`` bookkeeping, config and mask validation,
the linearity of :data:`LINEAR_CONFIG`, and the finite-gradient regression test
for the ``_RMS_EPS`` 0/0 bug. Training tests at this layer run on a 12-sample,
3-band synthetic batch whose "truth" is the backbone times a known smooth
residual, so a fit costs well under a second.

**The committed 50-scene fixture.** ``fit_l23`` with a few hundred steps: the
fit term falls, the hybrid beats the plain backbone on its training split, and
both named held-out curves are recorded and finite. Real numbers, in CI, with no
``$OS_COLOR``.

The full-data accuracy numbers (0.30% etc.) are deliberately **not** asserted
here — they need the full L23 release and live in the implementation record.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import emulator as E
from robust.rt import validation as V
from robust.rt import ztt as Z
from robust.rt.types import Geometry, IOPs, PhaseParams

N_FEATURES = len(E.FEATURES)

#: Small off-canonical wavelength grid for the synthetic tests, chosen so the
#: sample count (12) never equals the band count (3): ``features()`` documents
#: that a per-sample ``B_p`` is ambiguous exactly when they coincide.
SYNTH_WAVE = np.asarray([440.0, 550.0, 660.0])


def synthetic_batch(n: int = 12):
    """A tiny batch with a known, learnable relative residual.

    Truth is ``rrs_ZTT * (1 + delta_true)`` with ``delta_true`` a smooth function
    of wavelength and solar zenith — exactly the structure M2 measured — so a
    short fit must be able to reduce the fit term. The last four samples carry
    the largest absorption, so a train mask excluding them has feature statistics
    that measurably differ from the full batch (used by the standardisation
    test).
    """
    wave = jnp.asarray(SYNTH_WAVE)
    a = jnp.linspace(0.04, 0.6, n)[:, None] * jnp.asarray([1.0, 0.8, 1.3])
    bb_p = jnp.linspace(0.002, 0.02, n)[:, None] * jnp.asarray([1.2, 1.0, 0.9])
    bb_w = jnp.broadcast_to(C.bb_w(wave), a.shape)
    iops = IOPs(a=a, bb_w=bb_w, bb_p=bb_p)
    phase = PhaseParams(B_p=jnp.asarray(0.0126))
    geometry = Geometry.nadir(jnp.asarray(np.tile([0.0, 30.0, 60.0], n // 3)))

    rrs_ztt = Z.rrs_ZTT(iops, phase, geometry, wave)
    cos_s = jnp.cos(jnp.deg2rad(geometry.theta_s))[:, None]
    delta_true = 0.04 * jnp.sin(wave / 60.0) + 0.03 * (cos_s - 0.9)
    truth = rrs_ztt * (1.0 + delta_true)

    train = np.zeros(n, dtype=bool)
    train[: n - 4] = True
    return iops, phase, geometry, wave, truth, rrs_ztt, train


@pytest.fixture(scope="module")
def synthetic_fit():
    """One short fit on the synthetic batch, shared by several tests.

    150 steps with ``eval_every=50`` records exactly 4 history points and takes
    well under a second on the 36-row batch.
    """
    iops, phase, geometry, wave, truth, rrs_ztt, train = synthetic_batch()
    config = E.EmulatorConfig(steps=150, eval_every=50, seed=11)
    emulator, history = E.fit(
        iops,
        phase,
        geometry,
        wave,
        truth,
        train=train,
        eval_masks={"held_out": ~train},
        config=config,
        rrs_ztt=rrs_ztt,
    )
    return emulator, history, config


@pytest.fixture(scope="module")
def l23_fit(l23_small_batch):
    """A cheap ``fit_l23`` run on the committed 50-scene fixture."""
    from robust.rt.data import l23

    splits = l23.make_splits(l23_small_batch)
    config = E.EmulatorConfig(steps=300, eval_every=100)
    emulator, history = E.fit_l23(l23_small_batch, splits, config=config)
    return emulator, history, splits, config


# ------------------------------------------------------------- features() ----


def test_features_shape_and_column_order():
    """Output is ``(..., n_wave, len(FEATURES))`` with columns in FEATURES order.

    Each column is recomputed independently and looked up by
    ``FEATURES.index``, so a silent reordering of the tuple or of the stack
    cannot pass.
    """
    wave = jnp.asarray(SYNTH_WAVE)
    iops = IOPs(
        a=jnp.linspace(0.05, 0.5, 4)[:, None] * jnp.asarray([1.0, 0.8, 1.3]),
        bb_w=jnp.broadcast_to(C.bb_w(wave), (4, 3)),
        bb_p=jnp.linspace(0.002, 0.02, 4)[:, None] * jnp.asarray([1.2, 1.0, 0.9]),
    )
    phase = PhaseParams(B_p=jnp.asarray(0.0126))
    # Off-nadir, per-sample geometry so all three angle columns are distinct.
    geometry = Geometry(
        theta_s=jnp.asarray([10.0, 20.0, 30.0, 40.0]),
        theta_v=jnp.asarray([5.0, 15.0, 25.0, 35.0]),
        dphi=jnp.asarray([0.0, 45.0, 90.0, 135.0]),
    )

    x = E.features(iops, phase, geometry, wave)

    assert x.shape == (4, wave.shape[0], N_FEATURES)
    spectrum = (4, wave.shape[0])
    expected = {
        "log10_u": jnp.log10(iops.u),
        "eta_bb": iops.bb_w / iops.bb,
        "B_p": jnp.broadcast_to(phase.B_p, spectrum),
        "wave_nm": jnp.broadcast_to(wave, spectrum),
        "cos_theta_s": jnp.broadcast_to(
            jnp.cos(jnp.deg2rad(geometry.theta_s))[:, None], spectrum
        ),
        "cos_theta_v": jnp.broadcast_to(
            jnp.cos(jnp.deg2rad(geometry.theta_v))[:, None], spectrum
        ),
        "cos_dphi": jnp.broadcast_to(
            jnp.cos(jnp.deg2rad(geometry.dphi))[:, None], spectrum
        ),
    }
    assert set(expected) == set(E.FEATURES)
    for name, want in expected.items():
        got = np.asarray(x[..., E.FEATURES.index(name)])
        np.testing.assert_allclose(got, np.asarray(want), rtol=1e-6, err_msg=name)


def test_features_single_unbatched_sample():
    """A single spectrum (no batch axis) yields ``(n_wave, len(FEATURES))``."""
    wave = jnp.asarray(SYNTH_WAVE)
    iops = IOPs(
        a=jnp.asarray([0.2, 0.15, 0.4]),
        bb_w=C.bb_w(wave),
        bb_p=jnp.asarray([0.004, 0.003, 0.002]),
    )
    x = E.features(
        iops,
        PhaseParams(B_p=jnp.asarray(0.0126)),
        Geometry.nadir(jnp.asarray(30.0)),
        wave,
    )

    assert x.shape == (wave.shape[0], N_FEATURES)
    assert np.all(np.isfinite(np.asarray(x)))


def test_features_accepts_all_three_B_p_forms():
    """Spectrum, per-sample scalar, and plain scalar ``B_p`` agree exactly.

    All three encode the same constant 0.0126. The batch has 12 samples over 3
    bands, so the documented per-sample/spectrum ambiguity (batch size equal to
    ``n_wave``) cannot bite.
    """
    iops, _, geometry, wave, _, _, _ = synthetic_batch()
    n = iops.a.shape[0]

    forms = (
        jnp.full((n, wave.shape[0]), 0.0126),  # spectrum
        jnp.full((n,), 0.0126),  # per-sample scalar
        jnp.asarray(0.0126),  # plain scalar
    )
    outputs = [
        np.asarray(E.features(iops, PhaseParams(B_p=B_p), geometry, wave))
        for B_p in forms
    ]

    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], outputs[2])


def test_features_default_wave_is_the_canonical_grid():
    """``wave=None`` fills the ``wave_nm`` column with the canonical grid."""
    n_wave = C.N_WAVE
    wave = C.canonical_wave()
    iops = IOPs(
        a=jnp.full((2, n_wave), 0.2),
        bb_w=jnp.broadcast_to(C.bb_w(wave), (2, n_wave)),
        bb_p=jnp.full((2, n_wave), 0.003),
    )
    x = E.features(
        iops,
        PhaseParams(B_p=jnp.asarray(0.0126)),
        Geometry.nadir(jnp.asarray([0.0, 60.0])),
    )

    assert x.shape == (2, n_wave, N_FEATURES)
    column = np.asarray(x[..., E.FEATURES.index("wave_nm")])
    np.testing.assert_array_equal(column, np.broadcast_to(wave, (2, n_wave)))


# ------------------------------------------------------- scale invariance ----


def test_backbone_and_features_are_scale_invariant(jax_x64):
    """Scaling ``(a, bb_w, bb_p)`` by k=10 changes neither ``rrs_ZTT`` nor the features.

    This is the claim that makes the feature set a *complete* description of the
    backbone's input rather than a guess (module docstring, point 2): the
    backbone sees its inputs only through the scale-free ratios ``u`` and
    ``eta_bb``, so those two columns — and with the geometry, B_p, and lambda,
    the whole feature vector — span everything the backbone knew. Verified in
    float64; the module records 8.8e-15 at k=10, so 1e-13 leaves headroom for
    ordinary rounding-order changes without hiding a real leak of an absolute
    magnitude.
    """
    dtype = jnp.float64
    wave = jnp.asarray(SYNTH_WAVE, dtype=dtype)
    a = jnp.asarray([0.3, 0.15, 0.4], dtype=dtype)
    bb_w = jnp.asarray(C.bb_w(wave), dtype=dtype)
    bb_p = jnp.asarray([0.004, 0.003, 0.0015], dtype=dtype)
    phase = PhaseParams(B_p=jnp.asarray(0.0126, dtype=dtype))
    geometry = Geometry.nadir(jnp.asarray(30.0, dtype=dtype))

    k = 10.0
    iops = IOPs(a=a, bb_w=bb_w, bb_p=bb_p)
    scaled = IOPs(a=k * a, bb_w=k * bb_w, bb_p=k * bb_p)

    r = np.asarray(Z.rrs_ZTT(iops, phase, geometry, wave))
    r_scaled = np.asarray(Z.rrs_ZTT(scaled, phase, geometry, wave))
    assert np.max(np.abs(r_scaled / r - 1.0)) < 1e-13

    x = np.asarray(E.features(iops, phase, geometry, wave))
    x_scaled = np.asarray(E.features(scaled, phase, geometry, wave))
    for name in ("log10_u", "eta_bb"):
        j = E.FEATURES.index(name)
        np.testing.assert_allclose(x_scaled[..., j], x[..., j], rtol=1e-13)
    # No column carries an absolute magnitude, so the whole matrix is invariant.
    np.testing.assert_allclose(x_scaled, x, rtol=1e-13)


# ------------------------------------------------- the correction starts at 0 -


def test_network_output_is_exactly_zero_at_init():
    """The output layer is zero-initialised, so ``delta_raw`` is exactly 0.0.

    Checked for both the default MLP and the linear configuration: the hidden
    kernels are random, but they feed a zero output kernel, so the result is
    identically zero — not merely small.
    """
    x = jnp.asarray(np.random.default_rng(0).normal(size=(6, N_FEATURES)))
    for config in (E.EmulatorConfig(), E.LINEAR_CONFIG):
        model = E._network(config)
        params = model.init(jax.random.key(config.seed), x)
        out = np.asarray(model.apply(params, x))
        np.testing.assert_array_equal(out, 0.0)


def test_untrained_emulator_is_the_backbone():
    """A zero-init ``Emulator`` gives ``relative_delta == 0`` and ``delta_rrs == 0``.

    Exactly zero, not approximately: an untrained hybrid *is* the backbone, so
    every reported improvement is an improvement over ZTT rather than an
    initialisation artefact (module docstring).
    """
    iops, phase, geometry, wave, _, rrs_ztt, _ = synthetic_batch()
    config = E.EmulatorConfig()
    model = E._network(config)
    params = model.init(jax.random.key(0), jnp.zeros((1, N_FEATURES)))
    emulator = E.Emulator(
        params=params,
        mean=jnp.zeros(N_FEATURES),
        std=jnp.ones(N_FEATURES),
        config=config,
    )

    delta = np.asarray(emulator.relative_delta(iops, phase, geometry, wave))
    np.testing.assert_array_equal(delta, 0.0)
    d_rrs = np.asarray(emulator.delta_rrs(iops, phase, geometry, wave, rrs_ztt=rrs_ztt))
    np.testing.assert_array_equal(d_rrs, 0.0)


def test_fit_records_the_backbone_at_step_zero():
    """``History`` at step 0 is the plain backbone: fit = rrms(truth, ZTT), delta ~ 0.

    A 1-step fit is the cheapest way to see the step-0 record (0 steps is
    rejected by config validation). ``delta_rms[0]`` is not exactly 0 because of
    the deliberate ``_RMS_EPS`` inside the square root — 100*sqrt(1e-24) =
    1e-10 percent, unmeasurable.
    """
    iops, phase, geometry, wave, truth, rrs_ztt, train = synthetic_batch()
    config = E.EmulatorConfig(steps=1, eval_every=1)

    _, history = E.fit(
        iops,
        phase,
        geometry,
        wave,
        truth,
        train=train,
        config=config,
        rrs_ztt=rrs_ztt,
    )

    backbone = float(V.rrms(truth[train], rrs_ztt[train]))
    assert history.step[0] == 0
    assert history.fit[0] == pytest.approx(backbone, rel=1e-5)
    assert history.loss[0] == pytest.approx(backbone, rel=1e-5)
    assert history.delta_rms[0] < 1e-9


# ------------------------------------------------------------- boundedness ---


def test_delta_is_bounded_for_absurd_inputs(synthetic_fit):
    """``|delta| <= delta_max`` even far outside the training distribution.

    u near 1, a 10-micron "wavelength", the sun on the horizon, an off-nadir
    view: the standardised features are enormous, and the tanh cap is the only
    thing standing between the hybrid and a wild correction. The bound is
    asserted as ``<=`` rather than ``<`` because float32 ``tanh`` saturates to
    exactly 1.0, so the cap can be *attained*; the hard guarantee is that it is
    never exceeded.
    """
    emulator, _, config = synthetic_fit
    wave = jnp.asarray([1.0e4])
    iops = IOPs(
        a=jnp.asarray([1e-4]),
        bb_w=jnp.asarray([1e-4]),
        bb_p=jnp.asarray([10.0]),  # u = 0.99999...
    )
    phase = PhaseParams(B_p=jnp.asarray(0.05))
    geometry = Geometry(
        theta_s=jnp.asarray(89.0), theta_v=jnp.asarray(60.0), dphi=jnp.asarray(180.0)
    )

    delta = np.asarray(emulator.relative_delta(iops, phase, geometry, wave))

    assert np.all(np.isfinite(delta))
    assert np.all(np.abs(delta) <= config.delta_max)


def test_delta_bound_binds_at_saturation(synthetic_fit):
    """With the parameters scaled by 1e6 the cap binds exactly — and holds.

    Multiplying every trained weight by 1e6 drives the raw output deep into
    tanh saturation, the worst case the cap exists for. ``delta`` comes out at
    exactly ``+-delta_max`` and never beyond it.
    """
    emulator, _, config = synthetic_fit
    iops, phase, geometry, wave, _, _, _ = synthetic_batch()
    huge = jax.tree_util.tree_map(lambda leaf: leaf * 1e6, emulator.params)
    saturated = dataclasses.replace(emulator, params=huge)

    delta = np.asarray(saturated.relative_delta(iops, phase, geometry, wave))

    assert np.all(np.abs(delta) <= config.delta_max)
    assert np.max(np.abs(delta)) == pytest.approx(config.delta_max, rel=1e-6)


# ------------------------------------------------------------- determinism ---


def test_fit_is_deterministic_and_seed_sensitive():
    """Same config, same data: bitwise-identical fits. Different seed: different.

    Training is full-batch and unshuffled, so ``seed`` is the only stochastic
    input (config docstring) — reproducibility is a design claim, not luck.
    """
    iops, phase, geometry, wave, truth, rrs_ztt, train = synthetic_batch()
    config = E.EmulatorConfig(steps=25, eval_every=10, seed=3)

    def run(cfg):
        return E.fit(
            iops,
            phase,
            geometry,
            wave,
            truth,
            train=train,
            config=cfg,
            rrs_ztt=rrs_ztt,
        )

    emu_a, hist_a = run(config)
    emu_b, hist_b = run(config)
    emu_c, _ = run(dataclasses.replace(config, seed=4))

    def leaves_equal(p, q):
        pairs = zip(
            jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(q), strict=True
        )
        return [bool(np.array_equal(a, b)) for a, b in pairs]

    assert all(leaves_equal(emu_a.params, emu_b.params))
    np.testing.assert_array_equal(hist_a.loss, hist_b.loss)
    # A different seed re-draws the hidden kernels, so some leaf must differ.
    assert not all(leaves_equal(emu_a.params, emu_c.params))

    # The remainder chunk: 25 steps at eval_every=10 records 0, 10, 20, 25.
    np.testing.assert_array_equal(hist_a.step, [0, 10, 20, 25])


# --------------------------------------------------------------- validation --


def test_fit_rejects_bad_train_masks():
    """Wrong shape, wrong dtype, and all-False masks all fail loudly.

    The mask is required and validated because "trained on everything" is the
    silent mistake that costs a milestone its credibility (fit docstring).
    """
    iops, phase, geometry, wave, truth, rrs_ztt, train = synthetic_batch()
    args = (iops, phase, geometry, wave, truth)
    n = truth.shape[0]

    with pytest.raises(ValueError, match="boolean mask"):
        E.fit(*args, train=train[:-1], rrs_ztt=rrs_ztt)  # wrong shape
    with pytest.raises(ValueError, match="boolean mask"):
        E.fit(*args, train=train.astype(int), rrs_ztt=rrs_ztt)  # wrong dtype
    with pytest.raises(ValueError, match="selects no samples"):
        E.fit(*args, train=np.zeros(n, dtype=bool), rrs_ztt=rrs_ztt)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"hidden": (16, 0)}, "widths must be > 0"),
        ({"hidden": (-4,)}, "widths must be > 0"),
        ({"delta_max": 0.0}, "delta_max"),
        ({"delta_max": -0.5}, "delta_max"),
        ({"penalty": -0.01}, "penalty"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"steps": 0}, "steps"),
        ({"eval_every": 0}, "eval_every"),
    ],
)
def test_config_rejects_untrainable_values(kwargs, match):
    """``EmulatorConfig.__post_init__`` refuses configurations that cannot train."""
    with pytest.raises(ValueError, match=match):
        E.EmulatorConfig(**kwargs)


# ------------------------------------------------------------ trained domain --


def test_domain_is_the_train_range_and_reports_nothing_inside_it(synthetic_fit):
    """A fitted emulator carries its train-split range and is quiet inside it.

    The domain is what makes the extrapolation caveat operational rather than a
    remark in a log: it travels with the weights, so a caller evaluating the
    emulator somewhere it was never trained can be told.
    """
    emulator, _, _ = synthetic_fit
    iops, phase, geometry, wave, _, _, train = synthetic_batch()

    assert emulator.domain is not None
    x = np.asarray(E.features(iops, phase, geometry, wave))[train]
    lo, hi = np.asarray(emulator.domain)
    np.testing.assert_allclose(lo, x.reshape(-1, N_FEATURES).min(axis=0), rtol=1e-6)
    np.testing.assert_allclose(hi, x.reshape(-1, N_FEATURES).max(axis=0), rtol=1e-6)

    # Evaluated on exactly the rows it was trained on: nothing to report. (B_p is a
    # scalar here, so it needs no row selection.)
    trained_rows = (
        IOPs(a=iops.a[train], bb_w=iops.bb_w[train], bb_p=iops.bb_p[train]),
        phase,
        Geometry.nadir(geometry.theta_s[train]),
        wave,
    )
    assert emulator.out_of_domain(*trained_rows) == {}


def test_grazing_the_boundary_is_not_reported_but_leaving_it_is(synthetic_fit):
    """``DOMAIN_TOL`` separates a boundary graze from genuine extrapolation.

    Measured on real data: held-out L23 scenes push a feature 3.7e-4 of a span
    past the trained boundary, while a 75° sun against a ``cos_theta_s`` floor of
    0.5 sits 48% of the span beyond — three orders of magnitude apart. A check
    with no tolerance fires on the first, which teaches the user to silence the
    warning, and then the second passes unnoticed. Here the same contrast is made
    with an explicit shift, and ``tol=0.0`` is shown to report both.
    """
    emulator, _, _ = synthetic_fit
    iops, phase, geometry, wave, *_ = synthetic_batch()
    lo, hi = np.asarray(emulator.domain)
    j = E.FEATURES.index("cos_theta_s")
    span = float(hi[j] - lo[j])

    def report(excess, **kw):
        """Report at the solar zenith whose cosine lies ``excess`` spans below ``lo``.

        The solar zenith is used because its map to the feature is exact and
        affects nothing else: ``cos_theta_s`` is the only column that moves, so
        the tolerance is being tested and not some incidental coupling.
        """
        target = float(lo[j]) - excess * span
        theta = np.degrees(np.arccos(np.clip(target, -1.0, 1.0)))
        moved = Geometry.nadir(jnp.full_like(geometry.theta_s, theta))
        return emulator.out_of_domain(iops, phase, moved, wave, **kw)

    # 0.1% of the span beyond the boundary: a graze, not an excursion.
    assert "cos_theta_s" not in report(0.001)
    assert "cos_theta_s" in report(0.001, tol=0.0)  # tol=0 sees any excursion at all

    # 20% beyond: genuine extrapolation, reported with the magnitude attached
    # rather than as a bare fraction that would round to "0% of values".
    breach = report(0.2)["cos_theta_s"]
    assert breach.excess == pytest.approx(0.2, rel=1e-3)
    assert breach.lo == pytest.approx(float(lo[j]))
    assert breach.worst < breach.lo
    assert breach.fraction == 1.0  # every sample was moved


def test_out_of_domain_requires_a_domain():
    """A hand-built emulator has no training range, so the check refuses to guess."""
    config = E.EmulatorConfig()
    model = E._network(config)
    params = model.init(jax.random.key(0), jnp.zeros((1, N_FEATURES)))
    emulator = E.Emulator(
        params=params,
        mean=jnp.zeros(N_FEATURES),
        std=jnp.ones(N_FEATURES),
        config=config,
    )
    iops, phase, geometry, wave, *_ = synthetic_batch()
    with pytest.raises(ValueError, match="carries no domain"):
        emulator.out_of_domain(iops, phase, geometry, wave)


# ---------------------------------------------------------- standardisation --


def test_standardisation_comes_from_the_train_split_only(synthetic_fit):
    """``Emulator.mean``/``std`` equal the train-row statistics, not the batch's.

    Fitting the statistics on everything would leak the held-out scenes into
    the model in a way no accuracy number reveals (module docstring), so this
    pins that they match the train rows exactly and measurably differ from the
    full-batch values (the held-out samples carry the largest absorption by
    construction).
    """
    emulator, _, _ = synthetic_fit
    iops, phase, geometry, wave, _, _, train = synthetic_batch()

    x = np.asarray(E.features(iops, phase, geometry, wave))
    train_rows = x[train].reshape(-1, N_FEATURES)
    all_rows = x.reshape(-1, N_FEATURES)

    np.testing.assert_allclose(
        np.asarray(emulator.mean), train_rows.mean(axis=0), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(emulator.std),
        np.maximum(train_rows.std(axis=0), E._STD_FLOOR),
        rtol=1e-4,
    )
    j = E.FEATURES.index("log10_u")
    assert abs(train_rows.mean(axis=0)[j] - all_rows.mean(axis=0)[j]) > 1e-3


def test_constant_features_standardise_to_exactly_zero(synthetic_fit):
    """``cos_theta_v`` and ``cos_dphi`` are constant in L23 and standardise to 0.

    Their std over the training split is exactly 0, so without ``_STD_FLOOR``
    the standardisation would be 0/0 = NaN and poison every forward pass. The
    floor turns it into a clean 0.
    """
    emulator, _, _ = synthetic_fit
    iops, phase, geometry, wave, _, _, _ = synthetic_batch()

    x = E.features(iops, phase, geometry, wave)
    x_std = np.asarray(emulator._standardise(x))

    for name in ("cos_theta_v", "cos_dphi"):
        j = E.FEATURES.index(name)
        # The floor is stored in float32, hence approx rather than ==.
        assert float(emulator.std[j]) == pytest.approx(E._STD_FLOOR, rel=1e-6)
        np.testing.assert_array_equal(x_std[..., j], 0.0)
    assert np.all(np.isfinite(x_std))


# --------------------------------------------------- training and History ----


def test_history_arrays_are_consistent(synthetic_fit):
    """All ``History`` arrays share one length; steps run from 0 to ``config.steps``."""
    _, history, config = synthetic_fit

    n = len(history.step)
    assert len(history.loss) == n
    assert len(history.fit) == n
    assert len(history.delta_rms) == n
    assert set(history.eval) == {"held_out"}
    assert len(history.eval["held_out"]) == n
    assert history.step[0] == 0
    assert history.step[-1] == config.steps
    assert np.all(np.isfinite(history.loss))
    assert np.all(np.isfinite(history.eval["held_out"]))


def test_training_reduces_the_fit_term(synthetic_fit):
    """150 steps on a learnable synthetic residual must lower the fit term."""
    _, history, _ = synthetic_fit

    assert history.fit[-1] < history.fit[0]
    assert history.loss[-1] < history.loss[0]


def test_objective_gradient_is_finite_at_zero_init():
    """REGRESSION: the step-0 gradient must be finite when ``penalty > 0``.

    Two deliberate choices collide here: the output layer is zero-initialised
    (so ``delta == 0`` identically at step 0) and the size penalty is an RMS,
    whose derivative ``delta_i / (N*sqrt(mean(delta**2)))`` is 0/0 exactly at
    ``delta == 0``. This NaN-ed the entire training run on the first chunk
    until ``_RMS_EPS`` was added inside the square root. Pinned as a direct
    ``jax.grad`` of ``_objective`` at freshly-initialised parameters, so the
    epsilon can never be "simplified away" without this failing.
    """
    config = E.EmulatorConfig(penalty=0.02)
    model = E._network(config)
    rng = np.random.default_rng(3)
    x_std = jnp.asarray(rng.normal(size=(30, N_FEATURES)))
    rrs_ztt = jnp.asarray(10.0 ** rng.uniform(-5.0, -2.0, size=30))
    rrs_truth = rrs_ztt * 1.05

    params = model.init(jax.random.key(config.seed), x_std)
    grads, _aux = jax.grad(E._objective, has_aux=True)(
        params, model, x_std, rrs_ztt, rrs_truth, config
    )

    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.all(np.isfinite(np.asarray(leaf)))


# ------------------------------------------------------------ LINEAR_CONFIG --


def test_linear_config_has_exactly_one_dense_layer():
    """``hidden=()`` builds a single Dense with a ``(len(FEATURES), 1)`` kernel.

    The linear baseline must be *structurally* linear — same features, same
    loss, same loop, only the depth differs — for the MLP comparison to be
    apples-to-apples (module docstring, point 3).
    """
    model = E._network(E.LINEAR_CONFIG)
    params = model.init(jax.random.key(0), jnp.zeros((2, N_FEATURES)))

    layers = params["params"]
    assert set(layers) == {"Dense_0"}
    assert layers["Dense_0"]["kernel"].shape == (N_FEATURES, 1)
    assert layers["Dense_0"]["bias"].shape == (1,)


def test_linear_config_raw_output_superposes():
    """Before the tanh squash, the linear model is affine: superposition holds.

    ``f(x1) + f(x2) - 2 f(0) == f(x1 + x2) - f(0)`` for an affine ``f`` — a
    property no tanh MLP has. Random (non-zero) parameters are substituted
    because the zero-initialised network satisfies this trivially.
    """
    model = E._network(E.LINEAR_CONFIG)
    params = model.init(jax.random.key(0), jnp.zeros((1, N_FEATURES)))
    rng = np.random.default_rng(7)
    params = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(rng.normal(size=leaf.shape), leaf.dtype), params
    )

    def f(x):
        return np.asarray(model.apply(params, x))

    x1 = jnp.asarray(rng.normal(size=(4, N_FEATURES)))
    x2 = jnp.asarray(rng.normal(size=(4, N_FEATURES)))
    zero = jnp.zeros((4, N_FEATURES))

    np.testing.assert_allclose(
        f(x1) + f(x2) - 2.0 * f(zero), f(x1 + x2) - f(zero), atol=1e-5
    )


# ------------------------------------------- on the committed L23 fixture ----


def test_fit_l23_trains_on_scene_train_and_records_both_curves(
    l23_fit, l23_small_batch
):
    """``fit_l23`` uses ``splits.scene_train`` and records both named hold-outs.

    The training split is pinned through the standardisation statistics: they
    must equal the scene-train rows' mean and differ from the full batch's, so
    training on the wrong mask (or on everything) cannot pass.
    """
    emulator, history, splits, _ = l23_fit
    batch = l23_small_batch

    x = np.asarray(
        E.features(batch.iops, batch.phase_params, batch.geometry, batch.wave),
        dtype=np.float64,
    )
    train_mean = x[splits.scene_train].reshape(-1, N_FEATURES).mean(axis=0)
    full_mean = x.reshape(-1, N_FEATURES).mean(axis=0)

    # rtol 1e-3: the stored mean is a float32 accumulation over ~12k rows, so
    # it carries ~5e-5 relative summation noise -- far below the ~5e-3 gap
    # between the train and full means that this test discriminates on.
    np.testing.assert_allclose(np.asarray(emulator.mean), train_mean, rtol=1e-3)
    j = E.FEATURES.index("log10_u")
    assert abs(train_mean[j] - full_mean[j]) > 1e-3  # measured gap: ~5.1e-3
    assert abs(float(emulator.mean[j]) - full_mean[j]) > 1e-3

    # scene_test_60, not the raw zenith_test mask: ~80% of every-60-degree samples
    # are training scenes in this fit, so that curve would read as held-out while
    # being mostly training error.
    assert set(history.eval) == {"scene_test", "scene_test_60"}
    for curve in history.eval.values():
        assert len(curve) == len(history.step)
        assert np.all(np.isfinite(curve))


def test_hybrid_beats_the_backbone_on_the_fixture(l23_fit, l23_small_batch):
    """After 300 steps the hybrid's train-split rRMS is below plain ZTT's.

    Real L23 numbers, in CI. The fit term should also fall monotonically-ish;
    a small tolerance absorbs the occasional Adam overshoot between recorded
    points. The specific full-data accuracies (0.30% etc.) are deliberately not
    asserted — they need ``$OS_COLOR`` and live in the implementation record.
    """
    emulator, history, splits, _ = l23_fit
    batch = l23_small_batch
    mask = splits.scene_train

    truth = C.Rrs_to_rrs(batch.Rrs)
    rrs_ztt = Z.rrs_ZTT(batch.iops, batch.phase_params, batch.geometry, batch.wave)
    delta = emulator.relative_delta(
        batch.iops, batch.phase_params, batch.geometry, batch.wave
    )
    hybrid = rrs_ztt * (1.0 + delta)

    backbone_rrms = float(V.rrms(truth[mask], rrs_ztt[mask]))
    hybrid_rrms = float(V.rrms(truth[mask], hybrid[mask]))

    assert hybrid_rrms < backbone_rrms
    # The recomputation above must agree with what training recorded.
    assert hybrid_rrms == pytest.approx(history.fit[-1], rel=1e-4)
    # Monotone-ish descent: each recorded fit within 0.1 pp of never rising.
    assert np.all(np.diff(history.fit) <= 0.1)
    # The held-out-scene number exists and is finite at the end of training.
    assert np.isfinite(history.eval["scene_test"][-1])
