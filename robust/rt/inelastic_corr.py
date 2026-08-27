"""
The learned inelastic corrections — δ_R and δ_F (inelastic coding plan, M3).

Two small bounded Flax heads that rescale the M2 analytic terms where those
are *measurably* wrong (record §4.4: Raman increment −38.6 % at θ_s = 0°,
+30 % at 490 nm; fluorescence 685 nm drifting 0.99 → 0.85 with zenith):

- **δ_R** corrects the Raman *increment*, never the baseline:
  ``f_R = 1 + (f_phys − 1) · (1 + δ_R)`` — so ``f_R → 1`` wherever Raman
  vanishes, no matter what the network does, and the bound on δ_R caps how
  far the correction can push (design §4.5).
- **δ_F** rescales the φ_C-linear kernel:
  ``Rrs_fl = φ_C · K_fl · (1 + δ_F)`` — and **its features exclude φ_C**,
  which is what keeps the design's φ_C-linearity promise (§4.4): the yield
  stays a clean multiplicative handle, ``∂Rrs/∂φ_C = K_fl·(1 + δ_F)``.

**Features** (design §4.5), standardized per head at train time; the four
IOP-like columns enter as log10 because they span decades (the elastic
emulator's ``log10(u)`` precedent):

- δ_R, per (scene, λ): ``a(λ), b_b(λ), a(λ′), b_b(λ′), cos θ_s, λ`` — the
  excitation values via the M1 helpers ``conventions.raman_excitation`` +
  ``conventions.interp_spectrum``, the same calls ``raman_factor`` makes.
- δ_F, per (scene, λ_em): ``a_ph(440), a(λ_em), b_b(λ_em), a(490),
  cos θ_s, λ_em``.

**Machinery is the elastic emulator's, reused rather than re-derived**: the
network builder and the tanh-bounded output (`emulator._network` /
`emulator._delta`) are duck-typed on ``config.hidden`` / ``config.delta_max``
and called directly; the save format mirrors ``emulator.save`` including the
refuse-on-feature-mismatch rule (weights against the wrong feature vector
*run* and return plausible nonsense — the failure mode worth designing
against). The output layer is zero-initialised, so a freshly-initialised head
is **exactly the analytic model** (δ ≡ 0) — training starts from the physics,
and task-1 tests pin that identity before any training exists.

**How the corrections reach `forward`.** ``robust.rt.forward(...,
corrections=None)`` resolves ``None`` to :func:`load_default` — the packaged
weights in ``robust/rt/files/{raman,fl}_corr_l23.npz`` once M3 task 2 commits
them. Until then (or if the files are ever missing) the model **falls back to
analytic-only with a single** :class:`MissingCorrectionWarning` — a warning
and not an error because the analytic backbone is a legitimate model (it *is*
the M2 gate), but silence would hide missing physics from a caller who
expected the trained heads. ``corrections=False`` selects analytic-only
explicitly and silently — the M2 characterization tests pin the analytic
terms through exactly that switch — and an explicit :class:`CorrectionHeads`
instance is used as given (training, ablations).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions
from . import emulator as _emulator

__all__ = [  # noqa: RUF022  - grouped by role
    "KINDS",
    "RAMAN_FEATURES",
    "FL_FEATURES",
    "DEFAULT_RAMAN_WEIGHTS",
    "DEFAULT_FL_WEIGHTS",
    "MissingCorrectionWarning",
    "HeadConfig",
    "CorrectionHead",
    "CorrectionHeads",
    "features_raman",
    "features_fl",
    "init_head",
    "save_head",
    "load_head",
    "load_default",
    "corrected_raman_factor",
    "corrected_fluorescence",
]

#: The two correction processes. Order is cosmetic; the kind on a
#: :class:`HeadConfig` is what selects the feature builder.
KINDS = ("raman", "fl")

#: δ_R feature columns, in order (design §4.5; log10 on the IOPs — module
#: docstring). Stored in the weight files; :func:`load_head` refuses a file
#: whose list differs from the code's.
RAMAN_FEATURES = (
    "log10_a_em",
    "log10_bb_em",
    "log10_a_ex",
    "log10_bb_ex",
    "cos_theta_s",
    "wave",
)

#: δ_F feature columns, in order. Deliberately **no φ_C** (design §4.4).
FL_FEATURES = (
    "log10_a_ph440",
    "log10_a_em",
    "log10_bb_em",
    "log10_a_490",
    "cos_theta_s",
    "wave",
)

_FEATURES_BY_KIND = {"raman": RAMAN_FEATURES, "fl": FL_FEATURES}

#: Floor under the log10 features. L23 IOPs sit well above it (min ~3e-4);
#: it exists so a caller's zero (a_ph of pure water, say) yields a finite
#: feature instead of -inf poisoning the network.
_LOG_FLOOR = 1e-10

#: The committed trained weights (M3 task 2). Two files — the heads train,
#: version, and regenerate independently (coding plan M3).
DEFAULT_RAMAN_WEIGHTS = Path(__file__).parent / "files" / "raman_corr_l23.npz"
DEFAULT_FL_WEIGHTS = Path(__file__).parent / "files" / "fl_corr_l23.npz"


class MissingCorrectionWarning(UserWarning):
    """Trained correction weights were requested but not found.

    Its own category so a pipeline that must not silently run analytic-only
    can promote it (``warnings.simplefilter("error",
    MissingCorrectionWarning)``), and a study that means to compare backbones
    can silence it once.
    """


@dataclass(frozen=True)
class HeadConfig:
    """Architecture (and, at task 2, training) hyper-parameters of one head.

    Frozen and hashable so it can be a static ``meta_field`` of
    :class:`CorrectionHead` — the same contract as
    :class:`robust.rt.emulator.EmulatorConfig`, whose ``_network``/``_delta``
    this config is duck-typed against (``hidden``, ``delta_max``).

    Attributes
    ----------
    kind : str
        ``'raman'`` or ``'fl'`` — selects the feature builder and the
        composition form. Stored with the weights.
    hidden : tuple of int
        Hidden widths. The default single 16-wide layer is 129 parameters —
        the low end of the design's O(10²–10³) budget (§4.5); grow only if
        the held-out gate demands it (coding plan M3). ``()`` is the linear
        baseline, as in the elastic effort.
    delta_max : float
        Hard tanh bound on |δ|. Defaults differ by head because the measured
        errors do: δ_R must reach +0.64 to close the −39 % increment gap at
        0° (1/0.61 − 1), so its bound is 1.0; δ_F needs ~+0.18 for the 60°
        drift, so the elastic default 0.5 has ample slack.
    learning_rate, steps, seed : task-2 training knobs, stored now so the
        weight-file format does not change when training lands.
    """

    kind: str
    hidden: tuple[int, ...] = (16,)
    delta_max: float = 0.5
    learning_rate: float = 3e-3
    steps: int = 3000
    seed: int = 23

    def __post_init__(self) -> None:
        """Reject configurations that cannot work, with a reason.

        Raises
        ------
        ValueError
            On an unknown kind or a non-positive width, bound, rate, or
            step count.
        """
        if self.kind not in KINDS:
            raise ValueError(f"HeadConfig.kind: must be one of {KINDS}; {self.kind!r}")
        if any(h <= 0 for h in self.hidden):
            raise ValueError(f"HeadConfig.hidden: widths must be > 0; {self.hidden}")
        if self.delta_max <= 0.0:
            raise ValueError(f"HeadConfig.delta_max: must be > 0; {self.delta_max}")
        if self.learning_rate <= 0.0 or self.steps < 1:
            raise ValueError(
                f"HeadConfig: learning_rate ({self.learning_rate}) must be > 0 "
                f"and steps ({self.steps}) >= 1"
            )

    @property
    def features(self) -> tuple[str, ...]:
        """The feature names this head is defined over."""
        return _FEATURES_BY_KIND[self.kind]


def _log10(x):
    """log10 with the documented floor — finite for a zero IOP."""
    return jnp.log10(jnp.maximum(x, _LOG_FLOOR))


def features_raman(
    iops,
    geometry,
    wave: Float[Array, " wave"] | None = None,
) -> Float[Array, "*batch wave feature"]:
    """Raw δ_R feature vectors, one per (sample, λ) — :data:`RAMAN_FEATURES`.

    The excitation-grid values come from the M1 helpers **by name**
    (`conventions.raman_excitation` + `conventions.interp_spectrum`), so the
    head sees exactly the inputs :func:`robust.rt.inelastic.raman_factor`
    computes with.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
    geometry : robust.rt.types.Geometry
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid.

    Returns
    -------
    Array
        Shape ``(..., n_wave, 6)``.
    """
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)
    a_em, bb_em = iops.a, iops.bb
    wave_ex = conventions.raman_excitation(wave)
    a_ex = conventions.interp_spectrum(wave_ex, wave, a_em)
    bb_ex = conventions.interp_spectrum(wave_ex, wave, bb_em)
    cos_theta = jnp.broadcast_to(
        jnp.cos(jnp.deg2rad(jnp.asarray(geometry.theta_s)))[..., None], a_em.shape
    )
    columns = (
        _log10(a_em),
        _log10(bb_em),
        _log10(a_ex),
        _log10(bb_ex),
        cos_theta,
        jnp.broadcast_to(wave, a_em.shape),
    )
    return jnp.stack(columns, axis=-1)


def features_fl(
    iops,
    geometry,
    wave: Float[Array, " wave"] | None = None,
) -> Float[Array, "*batch wave feature"]:
    """Raw δ_F feature vectors, one per (sample, λ_em) — :data:`FL_FEATURES`.

    **No φ_C column, by design** (§4.4): the head must not be able to break
    the yield's linearity. ``a_ph(440)`` and ``a(490)`` are per-scene scalars
    (via `conventions.interp_spectrum`) broadcast along λ_em.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Must carry ``a_ph`` (the same physical requirement as the kernel).
    geometry : robust.rt.types.Geometry
    wave : Array, optional

    Returns
    -------
    Array
        Shape ``(..., n_wave, 6)``.

    Raises
    ------
    ValueError
        If ``iops.a_ph`` is ``None``.
    """
    if iops.a_ph is None:
        raise ValueError(
            "features_fl: IOPs.a_ph is None — the fluorescence correction is "
            "keyed on the phytoplankton component (a_ph(440) is its leading "
            "feature). Provide a_ph or disable fluorescence"
        )
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)
    a_em, bb_em = iops.a, iops.bb
    aph440 = conventions.interp_spectrum(jnp.asarray([440.0]), wave, iops.a_ph)
    a490 = conventions.interp_spectrum(jnp.asarray([490.0]), wave, a_em)
    cos_theta = jnp.broadcast_to(
        jnp.cos(jnp.deg2rad(jnp.asarray(geometry.theta_s)))[..., None], a_em.shape
    )
    columns = (
        jnp.broadcast_to(_log10(aph440), a_em.shape),
        _log10(a_em),
        _log10(bb_em),
        jnp.broadcast_to(_log10(a490), a_em.shape),
        cos_theta,
        jnp.broadcast_to(wave, a_em.shape),
    )
    return jnp.stack(columns, axis=-1)


_FEATURE_FNS = {"raman": features_raman, "fl": features_fl}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CorrectionHead:
    """One trained (or initialised) correction head: parameters + scaling.

    A registered pytree (parameters and standardisation are leaves, the
    config static), for the same reasons as :class:`robust.rt.emulator
    .Emulator`: ``grad`` w.r.t. a head gives parameter gradients, and
    differentiating the composed forward w.r.t. the *inputs* while carrying
    a head along needs no special handling.

    Attributes
    ----------
    params : dict
        Flax parameter pytree for ``emulator._network(config)``.
    mean, std : Array
        Per-feature standardisation, shape ``(6,)`` — train-split statistics
        at task 2; zeros/ones on a fresh :func:`init_head`.
    config : HeadConfig
        Static.
    """

    params: dict
    mean: Float[Array, " feature"]
    std: Float[Array, " feature"]
    config: HeadConfig = field(metadata={"static": True})

    def delta(
        self,
        iops,
        geometry,
        wave: Float[Array, " wave"] | None = None,
    ) -> Float[Array, "*batch wave"]:
        """The bounded dimensionless correction δ(λ), |δ| < ``delta_max``.

        Parameters
        ----------
        iops, geometry, wave
            As the feature builders.

        Returns
        -------
        Array
            δ, shape ``(..., n_wave)``. Exactly 0 for a freshly-initialised
            head (zero-init output layer).
        """
        x = _FEATURE_FNS[self.config.kind](iops, geometry, wave)
        # Two reorganisations from the M4 speed fallback (record §6), both
        # algebraically identical (ULP-level float differences, ~4e-7 on δ):
        # the standardisation folds into the first layer's weights and bias —
        # ``(x−m)/s @ W = x @ (W/s) − (m/s) @ W`` on (6, 16) arrays, saving a
        # full elementwise pass over the feature tensor — and the (batch...,
        # wave) axes flatten for the two small matmuls, which XLA lowers to
        # its threaded matmul where the N-D dot_general stays in a
        # single-threaded loop. Together: 20 → 10 ms per head, full release.
        p = self.params["params"]
        inv_std = 1.0 / self.std
        first = {
            "kernel": p["Dense_0"]["kernel"] * inv_std[:, None],
            "bias": p["Dense_0"]["bias"]
            - (self.mean * inv_std) @ p["Dense_0"]["kernel"],
        }
        folded = {"params": {**p, "Dense_0": first}}
        flat = x.reshape(-1, x.shape[-1])
        delta = _emulator._delta(
            _emulator._network(self.config), folded, flat, self.config
        )
        return delta.reshape(x.shape[:-1])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CorrectionHeads:
    """The pair `forward` carries: either head may be ``None`` (analytic).

    ``None`` fields contribute no leaves (the ``Geometry.wind`` precedent);
    the treedef changes when one is set, so ``jit`` recompiles once per
    combination — correct and cheap.
    """

    raman: CorrectionHead | None = None
    fl: CorrectionHead | None = None


def init_head(kind: str, config: HeadConfig | None = None) -> CorrectionHead:
    """A freshly-initialised head — δ ≡ 0, i.e. exactly the analytic model.

    Parameters
    ----------
    kind : str
        ``'raman'`` or ``'fl'``.
    config : HeadConfig, optional
        Defaults to ``HeadConfig(kind)`` with the per-kind ``delta_max``
        documented on :class:`HeadConfig` (1.0 for raman, 0.5 for fl).

    Returns
    -------
    CorrectionHead
        Zero-output network (training's starting point), unit scaling.
    """
    if config is None:
        config = HeadConfig(kind, delta_max=1.0 if kind == "raman" else 0.5)
    n = len(config.features)
    model = _emulator._network(config)
    params = model.init(jax.random.PRNGKey(config.seed), jnp.zeros((1, n)))
    return CorrectionHead(
        params=params,
        mean=jnp.zeros(n),
        std=jnp.ones(n),
        config=config,
    )


def corrected_raman_factor(delta_r, f_phys):
    """The corrected factor ``f_R = 1 + (f_phys − 1) · (1 + δ_R)``.

    The design §4.5 increment form, written once so the wiring and the
    training objective cannot drift apart: the correction rescales the Raman
    *increment*, so ``f_R`` equals ``f_phys`` at δ_R = 0 and tends to 1
    wherever the increment vanishes, no matter what the network outputs.

    Parameters
    ----------
    delta_r : Array
        δ_R from :meth:`CorrectionHead.delta`, broadcastable to ``f_phys``.
    f_phys : Array
        The analytic factor from :func:`robust.rt.inelastic.raman_factor`.

    Returns
    -------
    Array
        ``f_R``, same shape as ``f_phys``.
    """
    return 1.0 + (f_phys - 1.0) * (1.0 + delta_r)


def corrected_fluorescence(delta_f, k_fl):
    """The corrected fluorescence kernel ``K_fl · (1 + δ_F)``.

    The Raman rule applied to the additive term: the composition is written
    once so ``hybrid._apply_inelastic``, the M3/M4 gate tests and
    ``run_validation.py`` score literally the same expression — before this
    helper (M4 review finding) each spelled it by hand, and a future change
    to the composition (a clamp, a bounded form) would have left the gates
    certifying a model ``forward`` no longer runs. The caller multiplies by
    ``φ_C`` afterwards, exactly as the design §2 law and ``_apply_inelastic``
    do (float multiplication is not associative, so the order is part of the
    contract).

    Parameters
    ----------
    delta_f : Array
        δ_F from :meth:`CorrectionHead.delta`, broadcastable to ``k_fl``.
    k_fl : Array
        The analytic kernel from
        :func:`robust.rt.inelastic.fluorescence_kernel`.

    Returns
    -------
    Array
        The corrected kernel, same shape as ``k_fl``.
    """
    return k_fl * (1.0 + delta_f)


def save_head(head: CorrectionHead, path) -> None:
    """Write a head to ``.npz`` — the ``emulator.save`` format, per head.

    Everything needed to reproduce a prediction in one file: parameters,
    standardisation, **feature names** (refused on mismatch at load), and
    the config. ~3 kB for the default architecture.

    Parameters
    ----------
    head : CorrectionHead
    path : str or pathlib.Path
    """
    from flax.traverse_util import flatten_dict

    arrays = {
        f"param/{'/'.join(k)}": np.asarray(v)
        for k, v in flatten_dict(head.params).items()
    }
    arrays["mean"] = np.asarray(head.mean)
    arrays["std"] = np.asarray(head.std)
    arrays["features"] = np.asarray(head.config.features)
    arrays["config/kind"] = np.asarray(head.config.kind)
    arrays["config/hidden"] = np.asarray(head.config.hidden, dtype=np.int64)
    for name in ("delta_max", "learning_rate", "steps", "seed"):
        arrays[f"config/{name}"] = np.asarray(getattr(head.config, name))
    np.savez(path, **arrays)


def load_head(path) -> CorrectionHead:
    """Read a head written by :func:`save_head`.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    CorrectionHead

    Raises
    ------
    ValueError
        If the file's feature list differs from the current code's for its
        kind — the weights would run against the wrong vector and return
        plausible nonsense, so this is a refusal, not a warning (the
        ``emulator.load`` rule).
    """
    from flax.traverse_util import unflatten_dict

    with np.load(path, allow_pickle=False) as data:
        kind = str(data["config/kind"])
        expected = _FEATURES_BY_KIND.get(kind)
        stored = tuple(str(f) for f in data["features"])
        if expected is None or stored != expected:
            raise ValueError(
                f"inelastic_corr.load_head: {path} holds kind {kind!r} with "
                f"features {stored}, but this code defines "
                f"{expected} — retrain rather than reinterpret the weights"
            )
        params = unflatten_dict(
            {
                tuple(key.split("/")[1:]): jnp.asarray(data[key])
                for key in data.files
                if key.startswith("param/")
            }
        )
        config = HeadConfig(
            kind=kind,
            hidden=tuple(int(h) for h in data["config/hidden"]),
            delta_max=float(data["config/delta_max"]),
            learning_rate=float(data["config/learning_rate"]),
            steps=int(data["config/steps"]),
            seed=int(data["config/seed"]),
        )
        return CorrectionHead(
            params=params,
            mean=jnp.asarray(data["mean"]),
            std=jnp.asarray(data["std"]),
            config=config,
        )


@cache
def load_default() -> CorrectionHeads:
    """The packaged trained heads, read once — with the documented fallback.

    Unlike ``emulator.load_default`` (which *raises* on missing weights,
    because the hybrid without its emulator is a different model), a missing
    correction file degrades to the analytic term — a legitimate model, the
    M2 gate itself — behind a single :class:`MissingCorrectionWarning` per
    process. Silence would hide missing physics; an error would make the
    package unusable between M2 and M3 task 2.

    Returns
    -------
    CorrectionHeads
        With ``None`` for each head whose weight file is absent.
    """
    missing = []
    heads = {}
    for kind, path in (("raman", DEFAULT_RAMAN_WEIGHTS), ("fl", DEFAULT_FL_WEIGHTS)):
        if path.exists():
            heads[kind] = load_head(path)
        else:
            heads[kind] = None
            missing.append(path.name)
    if missing:
        warnings.warn(
            f"trained inelastic correction weights not found ({', '.join(missing)} "
            f"in {DEFAULT_RAMAN_WEIGHTS.parent}); running the ANALYTIC-ONLY "
            "inelastic terms, whose known errors are the M2 error table (record "
            "§4.4: Raman increment −39% at zenith 0°, fluorescence −15% at 60°). "
            "Train and commit the heads with design/py/train_inelastic_corr.py "
            "(M3 task 2), or pass corrections=False to choose the analytic model "
            "explicitly and silence this",
            MissingCorrectionWarning,
            stacklevel=2,
        )
    return CorrectionHeads(**heads)
