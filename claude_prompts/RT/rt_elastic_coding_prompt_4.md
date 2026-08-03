# Elastic RT Coding — Prompt 4 (M3: Emulator + hybrid)

## Goals

Implement **Milestone M3**: the **residual emulator** `ΔRrs` and the **hybrid**
`forward()` = `Rrs_ZTT + ΔRrs`. This produces the first trained, end-to-end
differentiable forward model — the core deliverable of the Week-1 prototype.

M3 is the milestone where the package stops being physics-only. Everything it needs
exists: M1's data and types, M2's backbone and baseline, and a shared differentiable
`rrms`. What M3 adds is the *learned* half — and the interesting question is not
whether it can fit, but whether it earns its place.

## Claude

### Skills

Consider `.claude/skills/` — `critical-partner` (M3 is where over-claiming is
easiest: a network can always reduce training error), `code-review`, `dataviz`.

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).

The rhythm M0–M2 settled: **ask in Q&A** (numbered, phrased so work continues without
an answer; check for answers before each task), then after the code a **notebook**, a
**PR-review pass**, and a **hand-off edit to the next prompt doc**.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M3.
- **Design** — `design/rt_elastic_model.md` §4.4–4.5 (emulator + hybrid assembly) and
  §2 (the residual is small and smooth — the point of the hybrid over a black box).
  §2's claim is now **measured**, see below.
- **The BING weighting lesson** — relative (∝ `rrs`) weighting; unweighted fits let
  the red-λ terms run away (`context/RT/rt_elastic_model.md` §4 note). Note that
  `validation.rrms` is *already* the relatively-weighted metric and is
  differentiable, so it can serve directly as the loss.
- **Implementation record** — `design/rt_elastic_implementation.md` (currently
  **v0.13**; §3 is M1, §4 is M2 including the results and the one bug, §8 the
  cross-cutting conventions, §10 CI). At close: bump the version, add a **§5 for M3**
  with *Modules added / Tests / Results / Notebook*, flip M3 in §1, refresh the
  module index.
- **The three notebooks** — `notebooks/RT/rt_elastic_coding_{1,2,3}.ipynb` (M0: JAX,
  autodiff, float64; M1: `rrs` vs `Rrs`, the water/particle split, `B_p`, the splits;
  M2: the explicit VSF, the transcription, ZTT vs Gordon, the gradient gate).
  **Do not re-explain any of them**; link.

## Status entering M3

M2 is complete: **171 tests pass** (151 + 20 skipped without `$OS_COLOR`), `ruff
check` and `ruff format --check` clean, CI green on Python 3.12 and 3.14.

### The API, by name

```python
from robust.rt import conventions as C   # A_RRS/B_RRS, Rrs_to_rrs, rrs_to_Rrs
                                         # canonical_wave(), bb_w(wave), check_*
from robust.rt import ztt as Z           # rrs_ZTT, Rrs_ZTT, P_bb_sullivan
                                         # mu_infinity_tt2017, mu_infinity
from robust.rt import baselines as B     # rrs_gordon, Rrs_gordon, G1/G2_GORDON
from robust.rt import validation as V    # rrms(truth, pred, axis=None)  -> percent
from robust.rt.data import l23 as L      # load_batch, make_splits, select, npz_reader
from robust.rt.types import IOPs, PhaseParams, Geometry
```

- `Z.rrs_ZTT(iops, phase_params, geometry, wave, *, P_bb=None, mu_inf=None,
  mu_inf_coeffs=None, visibility_km=15.0)`. `P_bb=None` uses
  `Z.P_bb_sullivan(ψ)` — the paper's own best-performing choice. `mu_inf*=None` uses
  the TT2017 µ∞ (see *Outstanding* below).
- `V.rrms` is pure JAX and differentiable. **Use it as the loss**; it is already the
  relative weighting the BING lesson calls for, so there is no second definition to
  keep in sync.
- Splits: `L.make_splits(batch)` → `scene_train/scene_test` (by **scene**, so no
  water body is in both) and `zenith_train/zenith_test` (train 0°/30°, test 60°).
  On the full batch: **7968 train / 1992 held-out scenes / 3320 held-out 60°**.

### The residual you are about to model — measured, not assumed

`ΔRrs = rrs_L23 − rrs_ZTT` on all 9960 samples, in `rrs` space:

- **Relative:** mean **+2.20%**, sd **5.52%**, |max| 27.8%.
- **Absolute:** sd 4.98e-4 sr⁻¹ against a mean `rrs` of 5.27e-3.

Its structure is the part that matters. Mean relative residual (%) by zenith and band:

| sun | 400 nm | 500 | 550 | 600 | 700 |
|---|---|---|---|---|---|
| 0° | −3.54 | −1.74 | +1.03 | −2.61 | −3.98 |
| 30° | −1.25 | +1.98 | +5.86 | +3.23 | +2.67 |
| 60° | **+6.07** | **+8.59** | **+11.15** | **+7.49** | **+6.07** |

Two clean structures: a **monotone offset in solar zenith** (≈ −2%, +2%, +8%) and a
**spectral hump peaking near 550 nm** that grows with zenith. The zenith offset is
ZTT over-predicting the geometry effect (§4.4 of the record: 60°/0° ratio 0.855 vs
L23's 0.949).

**It is very smooth spectrally**, which is the design's central claim about why a
*small* network suffices — now quantified. Fraction of relative-residual variance
explained by a polynomial in λ:

| degree | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| variance explained | 83.9% | 91.1% | 91.6% | **96.0%** |

A straight line in λ already captures 84%. Start small; if a wide network is needed,
something is wrong.

### **The M3 gate as written is already satisfied — please strengthen it**

The coding plan's M3 gate is "hybrid **beats standard Gordon** rRMS on the train
split at all three solar zeniths". ZTT alone already does:

| split | ZTT | Gordon |
|---|---|---|
| train | **5.95%** | 7.21% |
| held-out scenes | **5.93%** | 7.21% |
| held-out 60° | **8.09%** | 9.01% |

So an emulator that outputs **exactly zero** passes that gate. It has to be
tightened, or M3 proves nothing about the learned half. The natural fix: **the hybrid
must beat `mode="ztt"` on the held-out splits, not just Gordon** — and the emulator's
contribution should be reported as a number (rRMS reduction over the backbone), not
implied. Raised as **Q5** below; the recommendation is in there and does not block
task 1.

### Throughput baseline for the "must not collapse" check

Jitted, full 9960 × 81 batch, CPU: **ZTT 3.4 ms** ≈ 235 M sample·λ/s; Gordon 0.3 ms
(12.8× cheaper). Measure the hybrid the same way and report the ratio.

### Gotchas carried forward

1. **`pytest` from the repo root** (`robust` may not be pip-installed).
2. **Flax and optax are installed but no package module imports them yet** — only
   `test_env.py`'s import smoke test does. Keep them
   *inside* `emulator.py`'s functions, per the M0 convention, so the analytic-only
   path never pays for the ML stack. `types.py` deliberately uses
   `jax.tree_util.register_dataclass` rather than `flax.struct` for the same reason
   (record §3.2).
3. **The gradient gate needs a per-variable step.** M2 measured that no single
   finite-difference step clears 1e-6 for all four inputs: `theta_s` is O(30) and
   wants h ≈ 1e-3, the IOP-like variables want ≈ 1e-7. Run under `jax_x64` and pin
   the dtype on the *arrays*. See notebook 3 §5.
4. **A step larger than the variable can leave the physical domain**: for `bb_p`,
   h ≳ 3e-3 drives it negative and the model returns NaN. Mask non-finite results
   rather than letting `argmin` pick one as the "best" step (it did, once).
5. **float32 floor of ~5e-5.** ZTT's Equation (4) is a quartic in degrees that
   cancels by a factor 78,000, costing ~5e-5 relative in float32. Do not assert
   hybrid agreement tighter than that in float32; the gradient gate is unaffected
   (float64).
6. **`B_p` covers only a 1.75× slice** of the design's 7.5× band (notebook 2), so a
   good fit here is **not** evidence of phase-function generalisation. Say so in the
   notebook rather than letting the reader infer otherwise.
7. **Test layering** (`test_ztt.py`/`test_l23.py` are the models): pure logic with no
   data; the committed 50-scene fixture via `L.npz_reader` so real numbers run in CI;
   `needs_l23` only for claims that need all 3320 scenes. Prefer the middle layer.
8. **Training must be deterministic and cheap enough for CI.** Seed everything;
   keep the committed test at toy size (a few hundred steps on the fixture) and
   leave the real training run to a script, as PAB does with its MCMC.

### Outstanding

- **Equation (8)'s `m1..m16`** — JXP has emailed the authors. Until they arrive µ∞
  comes from Twardowski & Tonizzo (2017); `mu_inf_coeffs=` swaps in the published
  2018 model in one line. Report as *ZTT with the TT2017 µ∞*.
- **A PR review of M2** — no PR was open when M2's task 5 ran, so Bugbot has not seen
  that diff. If it flags anything after JXP pushes, fold it into M3's task 5.

## Prompts

1. Read this doc. Execute the 1st task in the "M3" section below. If you have any
   questions, ask me in the Q&A section below. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task. Check my answers in Q&A. Use Fable if you
   can. Log your work.
3. Read this doc. Execute the 3rd task — the notebook. Use Fable if you can. Log your
   work.
4. Read this doc. Execute the 4th task — the PR review. Log your work.
5. Read this doc. Execute the 5th task — the hand-off to prompt 5. Log your work.

## M3

### Tasks

1. **Emulator.** `robust/rt/emulator.py`: a **small Flax MLP** for `ΔRrs`, trained
   with **Optax** on `rrs_L23 − rrs_ZTT` over the M1 **train split only**.

   Features: the design suggests `u` (or the `(ω_bw, ω_bp)` split), `B_p`, geometry,
   and λ. M2's measurements argue for making **λ and `theta_s` first-class** — they
   carry the two clean structures in the residual — and for including `η_bb =
   bb_w/bb`, which is already what ZTT's own µ∞ and µd depend on. Normalise inputs;
   an unnormalised λ in nanometres alongside a `B_p` of 0.012 will dominate the first
   layer.

   Loss: `V.rrms` directly (relatively weighted by construction). Regularise so the
   correction stays *bounded* — the hybrid's whole argument is that ΔRrs is small, so
   report its magnitude, not just the loss curve.

   Start with a genuinely small network and record what a linear-in-λ baseline
   achieves first: 84% of the residual variance is a straight line, so that is the
   number a net must beat to justify itself.

2. **Hybrid + gate.** `robust/rt/hybrid.py`: `forward()` = `Rrs_ZTT + ΔRrs` with the
   `mode ∈ {"ztt", "emulator", "hybrid"}` flag already declared as `MODES` (so the
   three design-doc options compare on identical data). The signature is pinned and
   currently raises. Tests in `robust/tests/test_hybrid.py`:

   - the hybrid **beats `mode="ztt"`** at all three solar zeniths on the train split
     — and beats Gordon, which is the weaker plan-level statement (see Q5);
   - **`jax.grad` finite-difference check on the full `forward`**, per-variable step,
     under `jax_x64`;
   - **throughput** recorded against the 3.4 ms ZTT baseline;
   - `mode="ztt"` reproduces `Z.rrs_ZTT` exactly, so the flag cannot silently change
     the physics;
   - the emulator's contribution reported as a number.

   Update the implementation record; note the branch for JXP.

3. **Notebook.** `notebooks/RT/rt_elastic_coding_4.ipynb` — the M3 explainer,
   following the conventions in the record §2.6/§3.4/§4.5 and §8: **executed** with
   outputs, data cells degrading without `$OS_COLOR` (prefer the committed fixture),
   `sys.path` bootstrap, house figure style, and the CVD-checked categorical palette
   (blue `#0072B2`, vermillion `#D55E00`, sky `#56B4E9`, near-black `#1a1a1a` — worst
   case ΔE 17.9; **validate any new colour, a green I tried failed at 5.4**).
   **Render and look at every figure** — that has caught a label collision, a
   truncated axis, a mis-rendered label, and a wrong colour encoding so far.

   Explain what M3 *decided*: which features the emulator sees and why; what a linear
   baseline already achieves; where the hybrid improves on the backbone and where it
   does not; and how much of the gain is generalisation versus fit. A figure of
   residual-before/after per λ and zenith earns its place; so does the learning curve
   with the held-out splits on it.

4. **PR Review.** Fetch the review comments on the open PR and address them. `gh` is
   not authenticated here; the public REST API works:
   `curl -s https://api.github.com/repos/ocean-colour/retrieve-or-bust/pulls/<n>/comments`
   and `.../reviews`. Fix the *class* of each defect, not the single instance, and
   demonstrate the fix catches what was reported. If no PR is open, review the diff
   yourself and say so.

5. **Finally.** Modify `rt_elastic_coding_prompt_5.md` (M4: validation — prototype
   done) given what M3 established. Use Fable if you can. Log your work.

### Q&A

**Q5 (M3 hand-off, Claude → JXP). The M3 gate needs strengthening before task 2.**

The coding plan's M3 acceptance is "hybrid beats standard Gordon rRMS on the train
split at all three solar zeniths". M2 measured that **ZTT alone already beats Gordon
on every split** (train 5.95% vs 7.21%; held-out scenes 5.93% vs 7.21%; held-out 60°
8.09% vs 9.01%). So an emulator returning exactly zero satisfies the gate, and M3
would prove nothing about the learned half.

My recommendation: **keep the Gordon comparison as a floor, and add "the hybrid must
beat `mode="ztt"` on the held-out splits"** as the real gate, with the emulator's
contribution reported as an rRMS reduction over the backbone. That preserves the
plan's relative-not-absolute philosophy while making the gate load-bearing.

This is a plan-level acceptance criterion, so it is yours to change — I have written
task 2 to test both, with the ZTT comparison as the one that matters. Say if you
would rather I only test the plan's original wording.
— *Not blocking task 1.*

## Next

→ `rt_elastic_coding_prompt_5.md` (M4: Validation — prototype done).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
