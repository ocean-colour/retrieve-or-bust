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
4. Please modify the Notebook to have a section describing in detail the Hybrid model.  Include a figure if you can.  
5. Read this doc. Execute the 4th task — the PR review. Log your work.
6. Read this doc. Execute the 5th task — the hand-off to prompt 5. Log your work.

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

>A. Yes, keep Gordon as a floor.  

**Q6 (M3 task 1, Claude → JXP). Geometry extrapolation is not reproducible, and it
is M4's second gate.**

The emulator interpolates superbly — on the scene split it takes ZTT's 5.93% to
**0.30%** on held-out scenes. But trained on 0°/30° and asked for the **unseen 60°**,
MLP(16,16) gives, over seeds {23, 1, 7, 101, 2024}: **4.7 / 8.4 / 7.8 / 5.4 / 12.2%**.
The backbone alone is 8.09% there and Gordon is 9.01%, so *whether the hybrid passes
M4's zenith gate depends on the seed*. The linear model (`hidden=()`) gives up a lot
in sample (2.40%) but is stable at **6.16%**, beating both. The cause is that
`cos θ_s` ∈ [0.866, 1.0] in training while 60° needs 0.5, so every `tanh` is outside
its fitted range.

Three ways forward, and the choice is a project one rather than a coding one:

1. **Report and defer.** M3's deliverable is the scene-split result; state plainly
   that geometry extrapolation is unresolved and let M5's HydroLight runs (which vary
   geometry) fix it with data. Cheapest, and honest.
2. **Early stopping on a geometry-held-out curve.** The unseen-60° error is best
   early and degrades with training. This works but tunes on the test split unless a
   third geometry is carved out — and L23 has only three angles.
3. **Constrain the geometry direction** so it cannot bend (e.g. geometry enters
   linearly, nonlinearity only in the IOP/λ features). Principled, but it is a real
   modelling commitment and I would not want to make it on one unseen angle.

My recommendation is **1 for M3, with the numbers above in the notebook**, and
raising it as M4's main open risk. Unless you say otherwise I will write task 2's
gate on the **scene** split (where the result is solid and reproducible) and *report*
the zenith numbers as a spread rather than gating on them.
— *Not blocking task 2.*

>A. Ok, go with 1.  I must add that trying to extrapolate to 60deg is a real stretch goal.  If we have outputs we trust at that angle, we can use them.  And we won't use the emulator at larger angles (or will warn the user).

## Next

→ `rt_elastic_coding_prompt_5.md` (M4: Validation — prototype done).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-05 (new 4th prompt — a section specifying the hybrid model, with two figures)

**Added §1 "The hybrid model, end to end"** to `notebooks/RT/rt_elastic_coding_4.ipynb`
and renumbered the rest 2–8, fixing the internal cross-references. Now 30 cells,
**6 figures**, still executed end to end with no errors and still degrading to the
committed fixture without `$OS_COLOR`. Record §5.8 updated.

The notebook explained M3's *decisions* but never wrote the model down. §1 now does,
stage by stage: the unfitted ZTT backbone; the seven-feature map evaluated **per
wavelength** (mirroring the backbone's own locality, which is what frees the emulator
from L23's 81-band grid); train-split standardisation; the 7→16→16→1 `tanh` MLP with its
zero-initialised output; `δ = 0.5·tanh(·)` as a hard bound; and
`rrs = rrs_ZTT(1 + δ)` followed by the non-linear interface. The loss is written out,
including why `rrms` is used directly rather than redefined.

**The parameter inventory is counted, not asserted** — the cell prints each layer's
shape from the *packaged* weights and sums them (112+16+256+16+16+1 = 417), and names the
three arrays that travel untrained but are just as load-bearing: `mean`, `std`, and
`domain`. §1 is also the only place that exercises `load_default()`, i.e. the path a user
of the package actually takes, and it re-derives the additivity claim on the spot:
exactly 0 in `rrs` space, 6.6e-4 sr⁻¹ in `Rrs`.

**Two figures.** A dataflow schematic (matplotlib boxes/arrows) separating the physics
path from the learned half and marking where each `mode` stops; and a term-by-term
decomposition of a single water body — the backbone's residual, the correction applied,
and what is left — on a **representative** scene rather than a flattering one, chosen as
the *median* backbone error among the 3320 samples at 60° and labelled as such in the
output.

**One thing I got wrong and caught by looking at the figure.** The decomposition panel
first plotted the network's raw `δ` against two residuals measured relative to the truth.
All three looked like the same quantity, but `δ` divides by `rrs_ZTT` while the residuals
divide by the L23 reference, so the correction curve sat ~2 points above the residual it
cancels — inviting exactly the wrong reading, that the model overshoots. Now all three
share the project's rRMS denominator, so the correction lands *on* the residual and the
leftover is visibly near zero; the prose states the distinction and gives `δ`'s own
definition rather than hiding it.

Also: the schematic's `Δrrs = δ·rrs_ZTT` label was cramped against a box edge on the
first render and was moved. Fourth milestone in a row where rendering the figure caught
something the code could not.

*Incidental, since I had begun the PR review before spotting the renumbered prompt list:*
there is **no open PR**; #10 (M2) was reviewed by Bugbot at commit `4d6c628` and found
**no new issues** — though that is not M2's final commit, so the last few M2 commits
remain unreviewed; and #9's one finding (a `bb_p` zenith check that skipped 30°) was
already fixed during M1, with the fix's docstring citing PR #9. Full pass belongs to
prompt 5.

### 2026-08-04 (M3 task 3 — the notebook, which caught four wrong numbers of mine)

**Built** `notebooks/RT/rt_elastic_coding_4.ipynb` — 23 cells, executed, 4 figures,
7 sections, ~5 min to run on the full batch. Q5 and Q6 answers read and applied: the
gate keeps Gordon as a floor, and geometry extrapolation is reported and deferred.

Sections: why the correction is relative (measured, not asserted); the features as the
backbone's *complete* state; the linear baseline before the MLP; fit vs generalisation;
**where it fails** (the seed spread at an unseen zenith); the gradient gate and cost;
and what M3 leaves open. Figures: residual before/after per λ and zenith on one shared
axis; the four-model per-λ ladder; the learning curve with both held-out splits; and the
five-seed extrapolation fan, which is the notebook's real contribution — five identical
configurations differing only in initialisation, landing between 4.7% and 12.2% at the
unseen 60° while all five sit at ~0.24% in sample. Seed replicates are drawn in one
de-emphasised grey precisely because they are *not* four things to tell apart; the two
reference lines get the categorical colours.

**Writing an honest notebook found four errors in what I had already logged.** The
notebook computes everything it claims, which is exactly why it caught them:

1. **The hybrid is ~5× ZTT, not 3.3×.** I had divided the emulator's 11.2 ms by ZTT's
   3.4 ms and reported that ratio as the *hybrid's*. Measured end to end: 16 ms against
   3.2 ms. Corrected in the record §5.4, `test_hybrid.py`'s comment, and the task-2 log
   above.
2. **The scale-invariance demo was measuring float32.** I enabled `jax_enable_x64` in
   the cell but the slices were already float32 arrays, so the invariance read 2e-7 —
   float32 epsilon — under prose claiming 1e-15. Pinning the dtype on the arrays gives
   6.7e-16. Gotcha 3 in this very doc says to pin the dtype on the arrays; I wrote the
   cell anyway.
3. **The "four decades" argument was measuring the mean spectrum** (a factor 419, so
   2.6 decades). The claim is true of the per-sample range, 4300, which the cell now
   prints.
4. **The domain check fired on in-range data**, which is how I found that the check
   itself was wrong — see below.

A Fable review pass over the executed notebook caught four more: the finite differences
are ~2e-7, not ~1e-9 (two of the four variables are worse than 1e-7); the title quoted
the linear model's *train* number in a held-out sentence; `DOMAIN_TOL`'s headroom is
27× below and 48× above, not "~25 on each side"; and "in-sample error ~0.24%
throughout" was unsupported — it is 0.24% *at the end*, which the notebook now prints.
Also fixed: the per-λ figure claimed the MLP sits "a decade" below the linear model and
that the improvement was "spectrally uniform". Neither survives the curves: the gap
narrows to ~3× where the linear model dips, the MLP's per-λ error spans 0.08–0.64%, and
the **blue retains the largest residual**. The honest claim — no wavelength region is
abandoned — is the one that matters anyway.

**A real design fix, prompted by the notebook.** The domain check compared against the
raw training min/max, so it warned on the full batch: held-out scenes legitimately
*graze* the boundary, by 3.7e-4 of a span, for 3 values in 800,000. A warning that fires
on that gets silenced, and then the case JXP actually asked to be warned about — a 75°
sun sitting **48%** of a span past the `cos_theta_s` floor, 1300× further out — passes
unnoticed. `out_of_domain` now measures *how far* outside, with a 1% tolerance
(`DOMAIN_TOL`), and returns a `DomainBreach` record instead of a tuple whose fraction
printed as "0% of values outside" — a number that reads as "nothing wrong". Three tests
pin the graze/breach distinction, and `tol=0.0` still reports any excursion.

**Also corrected:** `fit_l23` recorded a `zenith_test` curve that was ~80% training
data, i.e. it would have read as held-out while being mostly training error. It now
records `scene_test_60` — held-out scenes *at* 60° — which is what the learning-curve
figure plots.

Suite **225 passed**, ruff clean, record at v0.14 with the counts corrected. Figures
were rendered and inspected at every iteration: three rounds of layout fixes (a legend
sitting on the MLP curve, a label clipped off the axis, unreadable log minor ticks),
which is the third milestone running where looking at the figure caught something the
code could not.

### 2026-08-04 (M3 task 2 — the hybrid, its gate, and a bug in my own guard)

**Built** `robust/rt/hybrid.py` (`forward` → `Rrs`, `rrs_forward` → `rrs`, `MODES`,
`DomainWarning`), `robust/tests/test_hybrid.py` (22 tests), `design/py/train_emulator.py`,
and the shipped weights `robust/rt/files/emulator_l23.npz` (6.5 KB). Record updated to
**v0.14** with a full §5. Suite **222 passed** (202 + 20 skipped without `$OS_COLOR`,
which is what CI sees), ruff clean.

**The gate, per solar zenith on the train split** — a deterministic 400-step fit on the
committed fixture, so it runs in CI:

| sun | ZTT | hybrid | Gordon |
|---|---|---|---|
| 0° | 4.154% | **0.527%** | 5.268% |
| 30° | 3.899% | **0.525%** | 5.533% |
| 60° | 7.409% | **0.542%** | 8.623% |

Gated on beating **`mode="ztt"`** at every zenith, with the reduction asserted positive
and printed as a number; beating Gordon is kept as the weaker floor. That is Q5, still
unanswered — I implemented the stronger test, so an answer either way costs nothing.

**Also gated:** `mode="ztt"` reproduces `Z.rrs_ZTT` **bitwise**, so the flag cannot
silently change the physics; additivity is exact in `rrs` and violated by 1.28e-4 sr⁻¹
in `Rrs` (the interface is non-linear — a positive test that we score in the right
space); `jax.grad` vs central differences through the emulator under float64 agrees to
2.7e-9 (`a`), 6.8e-11 (`bb_p`), 1.7e-9 (`B_p`), 5.4e-10 (`theta_s`) with per-variable
steps. Throughput: the hybrid is **4.8×** ZTT — 15.0 ms against 3.1 ms on the full
9960×81 batch — so the speed advantage survives. (An earlier draft of this log said
3.3×, which was the *emulator-alone* ratio, 11.2 ms / 3.4 ms, misapplied to the total.
The notebook measured the hybrid end to end and caught it.)

**Q6 implemented, not just recorded.** Your "we won't use the emulator at larger angles
(or will warn the user)" is now enforced: `Emulator.domain` carries the per-feature
training range *with the weights*, and any emulator-using mode warns
(`hybrid.DomainWarning`, its own category so a pipeline can promote it to an error).
A sun at 75° warns; in-range inputs do not; `check_domain=False` opts out. It is a
boundary check, so it is skipped under `jit`/`grad` and says so.

**I shipped trained weights** (6.5 KB) plus `design/py/train_emulator.py` to regenerate
them, because "trained differentiable `forward()`" is the milestone's deliverable and
without weights `forward()` would raise until every caller reproduced a 50-second fit.
`save`/`load` store the feature list and `load` **refuses** a file whose `FEATURES` no
longer match — the weights would still run and return plausible nonsense otherwise.
`setup.py` gained `package_data`, without which an installed copy would import fine and
fail at the first `mode="hybrid"`.

**A bug in code I had just written, found by the test author I delegated to.** My
traced-input guard checked only `iops.a` and `geometry.theta_s`. But `jax.grad` traces
*only* the variable it differentiates, so `grad` w.r.t. `bb_p` or `B_p` alone left those
two concrete, the guard declared "not traced", and the domain check died in `np.asarray`
with a `TracerArrayConversionError` — on the default path, and specifically for
backscattering, which is exactly what the inversion this API exists for will
differentiate. Now every leaf is inspected via `tree_leaves`. I verified the fix the way
this project verifies fixes: reverting it makes the new parametrised regression test fail
on precisely `bb_p` and `B_p`, and only those.

Two of my own stale numbers also got caught in review (the weights described as "4 KB"
when they are 6.5, and a "57 s" fit that measures 50) — both written before I measured,
both now corrected. Same failure mode as the three in M2: assert after measuring, not
before.

**One interface question I did not settle**, in §5.7 of the record: `mode="emulator"`
returns the learned **correction term** `Δrrs`, not a standalone learned model, because
the emulator is parameterised as a relative correction. The design's "learned-only"
option therefore needs a differently trained network predicting `rrs` across four
decades — that belongs beside PR05 and O25 in M4's protocol, not as a flag on `forward`.

### 2026-08-03 (M3 task 1 — the emulator, and a result I first got wrong)

**Built** `robust/rt/emulator.py` (+`robust/tests/test_emulator.py`, 29 tests; suite
now **200 passed**, ruff clean). Public surface: `FEATURES`, `EmulatorConfig`,
`LINEAR_CONFIG`, `Emulator` (a registered pytree), `History`, `features()`, `fit()`,
`fit_l23()`.

**The headline, on the scene split (full 9960-sample batch):**

| rRMS % | train | held-out scenes | held-out @60° |
|---|---|---|---|
| Gordon | 7.21 | 7.21 | 9.01 |
| ZTT backbone | 5.95 | 5.93 | 8.11 |
| hybrid, linear (8 params) | 2.57 | 2.54 | 2.48 |
| **hybrid, MLP(16,16) (417 params)** | **0.30** | **0.30** | **0.32** |

Held-out equals train to two decimals, which is what 417 parameters against 645k
training rows should look like. (32,32) reached 0.27% — diminishing enough to confirm
the design's "start small". Fit takes 57 s for 3000 full-batch Adam steps; the
emulator alone is 11.2 ms jitted on 9960×81 (72 M sample·λ/s), so the hybrid will be
~4.8× dearer than ZTT's 3.1 ms. **The linear baseline is the honest yardstick and it
matters**: 2.57% means the MLP's nonlinearity earns its place by ~8×, not by 20× over
the backbone as a linear-baseline-free presentation would imply.

**Four design decisions, each measured rather than assumed** (all documented in the
module docstring):

1. *The correction is relative.* The net emits a dimensionless `δ(λ)` and
   `Δrrs = δ · rrs_ZTT`. Still additive as the design specifies, but the target is
   O(1) instead of spanning four decades — and `δ` is directly the "keep the
   correction small" quantity. Measured `|δ|` rms 6.44%, against the residual's own
   5.52% sd: the emulator is correcting the residual, not adding a large correction
   that partly cancels.
2. *The features are provably complete, not a guess.* I verified `rrs_ZTT` is
   **scale-invariant** — scaling `(a, bb_w, bb_p)` by k=10 moves it by 8.8e-15
   relative — so the backbone sees its inputs only through ratios, and `(u, η_bb)`
   invert back to `(a : bb_w : bb_p)` exactly. With `B_p`, geometry and λ that is the
   whole input state, so no absolute magnitude is a candidate feature. A test pins
   the invariance.
3. *λ and `cos θ_s` are first-class*, per M2's measured structure; standardisation
   comes from the **train split only** (leaking it is silent), and the stats live
   inside the `Emulator` so train and inference cannot disagree.
4. *Bounded by construction*: `δ = delta_max·tanh(·)`, plus a soft penalty in the
   same percent units as the loss, plus a zero-initialised output layer so an
   untrained hybrid **is** the backbone and every gain is a gain. `tanh` not `relu`
   because M3's gate is a finite-difference check and a relu kink breaks it.

**A bug worth recording.** Training NaN-ed on the first chunk. Two deliberate choices
collided: the output layer starts at zero, so `δ ≡ 0`, and the size penalty is an
RMS, whose derivative `δ/(N·√mean δ²)` is 0/0 exactly there. `_RMS_EPS` inside the
square root fixes it; a regression test asserts `jax.grad` of the objective is finite
at init, with the explanation, so it cannot come back silently.

**And a wrong inference I caught, which is the real lesson of the task.** Geometry
extrapolation is bad: trained on 0°/30°, the MLP scored 11.57% at the unseen 60° —
worse than ZTT (8.09%) *and* Gordon (9.01%). I diagnosed tanh saturation outside the
fitted `cos θ_s` range, tried a linear skip path, measured 5.40%, and was ready to
adopt it as the architecture on that comparison. Then the same no-skip config
reproduced at 4.74% — because moving from `nn.Sequential` to a module class had
changed the Flax parameter *names*, which changes PRNG folding and hence the
initialisation. Architecture and seed had moved together. A seed sweep with the
architecture fixed showed the skip is **not** better (median 9.20% vs 7.75%, worst
case 25%), so I removed it rather than ship a knob whose justification was a fluke.

What the sweep does establish is worse than the original claim and more useful: the
MLP's unseen-60° error is **4.7 / 8.4 / 7.8 / 5.4 / 12.2%** across five seeds, so
whether the hybrid passes M4's zenith gate is decided by initialisation. The linear
model is stable at 6.16% and beats both references. That is **Q6** above, with a
recommendation. This is the second time in this project that a plausible mechanism
plus one supporting number produced a wrong diagnosis (M2's `Pbb(ψ)` was the first);
both times the fix was to vary one thing at a time.

Not done here, deliberately: the record and notebook updates (tasks 2–3), and
persisting trained weights — the real fit is a 57 s script, so tests train toy-size on
the committed fixture per the CI gotcha.
