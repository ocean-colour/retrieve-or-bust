# Elastic RT Coding — Prompt 5 (M4: Validation — Week-1 prototype complete)

## Goals

Implement **Milestone M4**: the **validation protocol** and the acceptance gate that
declares the Week-1 prototype **done**. Run the design §6 protocol on the held-out
splits, compare against Gordon / PR05 / O25, and ship a metrics table + figures.

M4 is the milestone that decides what the prototype may claim. Every module it needs
exists and is tested; what M4 adds is *judgement* — which numbers are the headline,
which are caveated, which comparison is fair. Two of its required inputs are not in the
repo at all (see **PR05 and O25**), and its acceptance gate has a half that M3 measured
as **seed-dependent** (see **The M4 gate**). Both are flagged below with numbers rather
than left to be discovered.

## Claude

### Skills

Consider `.claude/skills/` — `critical-partner` (M4 declares the prototype "done": the
pressure is to round a fragile result up), `grill-me`, plus the `dataviz` and
`code-review` conventions the earlier milestones used.

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).

The rhythm M0–M3 settled: **ask in Q&A** (numbered, phrased so work continues without
an answer; check for answers before each task), then after the code a **notebook**, a
**PR-review pass**, and a **hand-off edit to the next prompt doc**.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M4 (the acceptance gate)
  and the Testing strategy.
- **Design** — `design/rt_elastic_model.md` §6 (validation protocol: accuracy per
  λ / geometry / `B_p`; held-out generalisation; speed; the gradient gate).
- **Split policy (CQ6)** — random 20% of scenes; **hold out 60° solar zenith**
  (train 0°/30°). Both are built already: `L.make_splits(batch)`.
- **Implementation record** — `design/rt_elastic_implementation.md` (currently
  **v0.14**; §5 is all of M3 — §5.4 results, §5.5 the open problem, §5.6/§5.6b the bugs
  and the PR review, §5.8 the notebook; §8 cross-cutting conventions; §9 module index;
  §10 CI). §6 is an M4 stub. At close: bump the version, fill §6, flip M4 in §1, refresh
  the module index.
- **The four notebooks** — `notebooks/RT/rt_elastic_coding_{1,2,3,4}.ipynb`. Notebook 4
  §1 specifies the hybrid stage by stage and is the fastest way to understand what you
  are validating. **Do not re-explain any of them**; link.
- **The synthesis material** — `context/RT/rt_elastic_model.md` transcribes the PR05 and
  O25 equations (~lines 125–129 and 170–173); both PDFs are in `context/RT/`
  (`park2005.pdf`, `pitarch2025.pdf`).

## Status entering M4

M3 is complete: **227 tests pass** (206 + 21 skipped without `$OS_COLOR`), `ruff check`
and `ruff format --check` clean, CI green on Python 3.12 and 3.14. PR #11 is open and its
two Bugbot findings are fixed (record §5.6b).

### The API, by name

```python
from robust.rt import conventions as C   # A_RRS/B_RRS, Rrs_to_rrs, rrs_to_Rrs
                                         # canonical_wave(), bb_w(wave), check_*
from robust.rt import ztt as Z           # rrs_ZTT, Rrs_ZTT, P_bb_sullivan, mu_infinity*
from robust.rt import baselines as B     # rrs_gordon, Rrs_gordon, G1/G2_GORDON
from robust.rt import emulator as E      # fit, fit_l23, features, FEATURES, Emulator,
                                         # EmulatorConfig, LINEAR_CONFIG, DomainBreach,
                                         # DOMAIN_TOL, save, load, load_default
from robust.rt import hybrid as H        # forward, rrs_forward, MODES, DomainWarning
from robust.rt import validation as V    # rrms(truth, pred, axis=None)  -> percent
from robust.rt.data import l23 as L      # load_batch, make_splits, select, npz_reader
from robust.rt.types import IOPs, PhaseParams, Geometry
```

- `H.rrs_forward(iops, phase_params, geometry, wave, mode="hybrid", *, emulator=None,
  check_domain=True)`, and `H.forward(...)` for the same in `Rrs`.
  `mode ∈ {"ztt", "emulator", "hybrid"}`. `emulator=None` loads the **packaged weights**
  (`robust/rt/files/emulator_l23.npz`, 6.5 KB, committed), so `forward` is a *trained*
  model out of the box and needs no `$OS_COLOR`.
- **Additivity holds in `rrs`, not `Rrs`** (the interface is non-linear), so score in
  `rrs`. `mode="ztt"` reproduces `Z.rrs_ZTT` **bitwise**.
- `E.fit(...)` requires an explicit `train=` mask; `E.fit_l23(batch, splits)` trains on
  `scene_train` and records `scene_test` and `scene_test_60`.
- Regenerate weights with `python design/py/train_emulator.py` (~60 s; writes to a temp
  file and verifies a round-trip before replacing anything). Output is **bit-identical**
  run to run — measured, not assumed.

### What M3 measured — the numbers M4 has to contextualise

Full 9960-sample L23 batch, rRMS % in `rrs` space, **scene** split:

| model | train | held-out scenes | held-out @60° |
|---|---|---|---|
| standard Gordon | 7.21 | 7.21 | 9.01 |
| ZTT backbone | 5.95 | 5.93 | 8.11 |
| hybrid, linear (8 par) | 2.57 | 2.54 | 2.48 |
| **hybrid, MLP (417 par)** | **0.30** | **0.30** | **0.32** |

Held-out equals train to two decimals — 417 parameters against ~645k training rows.
**Keep the linear row in every table you produce**: it is what makes the MLP's gain
legible as ~8×, rather than the ~20× that quoting only ZTT implies.

Throughput, jitted, 9960×81, CPU: the hybrid is **≈4.8–5.0× the backbone** (≈16 ms vs
≈3.2 ms; Gordon 0.3 ms). Wall-clock wanders ~20% between runs, so report the *ratio*.

### **The M4 gate has a half that M3 measured as seed-dependent**

The plan's acceptance gate: *hybrid beats standard Gordon on **both** held-out splits
(random 20% scenes; unseen 60° zenith) **and** passes the gradient gate*.

- **Scene split: passes by 24×.** 0.30% against Gordon's 7.21%. Not in doubt.
- **Zenith split: depends on the seed.** Trained on 0°/30° only and scored at the unseen
  60°, MLP(16,16) gives **4.74 / 8.37 / 7.75 / 5.40 / 12.24%** over seeds
  {23, 1, 7, 101, 2024}, against Gordon's **9.01%** there. Four of five pass; one fails.
  The median (7.75%) passes, barely. The committed default (seed 23) passes comfortably
  at 4.74% — which is exactly the trap: *reporting that pass without the spread would be
  the over-claim this milestone exists to prevent.*
- For reference on the same split: **ZTT alone is 8.09%**, beating Gordon with no seed to
  worry about; the **linear** emulator is stable at **6.16%** and beats both.

JXP's standing decision (Q6, prompt 4): 60° extrapolation is a *stretch goal*, and the
emulator will not be used at larger angles without warning the user. That is already
implemented as `Emulator.domain` + `hybrid.DomainWarning`. What M4 must settle is what
the *gate* does with it — **Q7** below, with a recommendation. Do not quietly gate on the
seed that passes.

### PR05 and O25 are not in the repo — and this doc used to be wrong about that

The plan requires scoring "alongside Gordon, PR05, O25". Gordon is implemented; the other
two are **not**. An earlier version of this doc said the O25 form could be reused from
`context/RT/make_rt_elastic_figures.py`. **It cannot — that script contains the Gordon
ladder only**, verified by grep over the whole repo. What is actually available:

**O25 — Pitarch et al. (2025)**, *Remote Sens. Environ.* **329**, 114920. Form (repo
transcription `context/RT/rt_elastic_model.md` ~line 170; paper Eqs. 3–4):

```
Rrs = (Gw0 + Gw1 · ω_bw) · ω_bw  +  (Gp0 + Gp1 · ω_bp) · ω_bp
ω_bw = bb_w/(a+bb),   ω_bp = bb_p/(a+bb)
```

A bivariate quadratic in the **water/particle-split** pseudo-albedos — which maps exactly
onto `IOPs(a, bb_w, bb_p)`, the split this project already keeps explicit. The four `G`s
are **geometry-only** LUTs over (θs, θv, Δφ), wavelength- and IOP-agnostic; phase-function
dependence is implicit in its calibration set's Fournier-Forand functions (PB24). Models
**`Rrs`, not `rrs`**. Stated validity ceiling `Rrs ≤ 0.06 sr⁻¹`.
*Coefficients are **not** in the repo and **not tabulated in the paper** (plots only).*
Sources: `github.com/jaipipor/O25`, NASA HyperCP, EUMETSAT ThoMaS — or re-fit per
geometry, a 4-parameter unweighted least squares, which **the paper itself does on L23**
(its Fig. 3), finding `Gw0`/`Gw1` close to PB24's.
*Transcription trap:* the paper's §2.1 writes `ηb = ω_bw/ω_bp`, contradicting its own
Fig. 2 caption (`ηb = bb_w/bb`, the standard definition and the one consistent with
PR05's `γb = 1 − ηb`). Use the latter.

**PR05 — Park & Ruddick (2005)**, *Appl. Opt.* **44**(7), 1236–1249. Form (repo
transcription ~line 125; paper Eq. 6):

```
Rrs = Σ_{i=1..4} g_i(θs, θv, Δφ, γb) · ω_b^i,     γb = bb_p/bb
```

A 4th-order polynomial in `ω_b` whose coefficients form a **4-D lookup table**. Also in
`Rrs`. *Coefficients are neither in the repo nor printed in the paper* — only ranges
(`g1 ≈ 0.03–0.07`, `g2 ≈ 0–0.3`, `g3 ≈ −0.8–0.2`, `g4 ≈ 0.2–1.0`). The paper points at a
2005 MUMM URL we do not have; POLYMER embeds the LUT. **And L23 is nadir-only with three
solar zeniths, so a re-fit here can populate the θs and γb axes but not θv/Δφ** — a
re-fitted PR05 is a different object from the published one and must be labelled so.
See **Q8**.

Neither is importable from `bing` or `ocpy` — checked, including git history.

### What *is* reusable

`context/RT/make_rt_elastic_figures.py` (246 lines, numpy/scipy, **not** JAX): the L23
locator/loader, four Gordon-ladder fitters (fixed; per-λ `(G1,G2)`; `+G0`; `+Gb·bb_p`), an
`rrms` identical to `validation.rrms`, and three figures. Caveats: it reads **only the 0°
file** (no geometry axis) and its fits are **unweighted**, which the plan warns against.
`context/RT/fig_rrms_ladder.csv` holds its output (`lam,std,quad,const,bbp`) — e.g. at
700 nm 9.04 / 5.74 / 0.35 / 0.37. Gordon-ladder rows only; no PR05/O25 row exists
anywhere in `context/RT/`. BING's L23-fitted Gordon tables
(`bing/data/RT/gordon_coefficients*.csv`, per-λ with errors) are useful as **extra
comparison rows**, not as PR05/O25.

### Gotchas carried forward

1. **`pytest` from the repo root** (`robust` may not be pip-installed).
2. **Score in `rrs`, not `Rrs`** (design §6). PR05 and O25 are defined in `Rrs`, so they
   must be converted with `C.Rrs_to_rrs` before scoring — and say so in the table
   caption, since their coefficients were fitted in `Rrs` space.
3. **`splits.zenith_test` is not a held-out mask for a scene-split fit.** ~80% of its
   samples are training scenes there, so it reads as held-out while being mostly training
   error. `fit_l23` therefore records `scene_test_60`; for true zenith extrapolation,
   train on `splits.zenith_train` via `E.fit`. This bit once.
4. **Per-`B_p`-bin metrics will have little dynamic range.** L23 spans only 1.75× in
   `B_p` (0.0103–0.0180) against the design's ~7× band, so report the bins but do not
   read phase-function generalisation into them.
5. **The gradient gate needs a per-variable step**: `a` 1e-6, `bb_p` 1e-9, `B_p` 1e-8,
   `theta_s` 1e-3, under `jax_x64`, dtype pinned on the **arrays**. Achieved through the
   full hybrid: 2.7e-9, 6.8e-11, 1.6e-7, 2.1e-7. A step larger than the variable can
   leave the physical domain (NaN) — mask non-finite differences rather than letting them
   into a comparison.
6. **`check_domain` is a boundary check**, skipped whenever any input is traced (`jit`,
   or `grad` of a single input). Pass `check_domain=False` in hot loops, and expect a
   `DomainWarning` when scoring a full batch with an emulator trained on a subset — that
   is correct behaviour, not noise (`DOMAIN_TOL` is 1% of the trained span).
7. **`mode="emulator"` is the correction term `Δrrs`, not a standalone model.** The
   design's "learned-only" option needs a *differently trained* network predicting `rrs`
   across four decades — an M4 model to add beside PR05/O25 if it is wanted at all
   (record §5.7).
8. **float32 floor of ~5e-5** from ZTT's quartic (record §4); do not assert tighter
   agreement in float32.
9. **Validate before overwriting a committed artifact.** Both write paths in the repo
   (`train_emulator.py`, `l23.write_fixture`) write to a temp file, verify, then
   `os.replace`. Any new artifact writer in M4 — the metrics table, the figures — should
   follow it.

### Outstanding

- **Equation (8)'s `m1..m16`** — still not received; µ∞ comes from Twardowski & Tonizzo
  (2017). Report as *ZTT with the TT2017 µ∞*; `mu_inf_coeffs=` swaps them in.
- **M2's last commits were never reviewed** — Bugbot's clean pass on PR #10 was at
  `4d6c628`, not M2's final commit.
- **PR #11** is open; if Bugbot adds findings after JXP pushes, fold them into task 5.

> JXP says -- ignore all of these outstanding items

## Prompts

1. Read this doc. Execute the 1st task in the "M4" section below. If you have any
   questions, ask me in the Q&A section below. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task. Check my answers in Q&A. Use Fable if you can.
   Log your work.
3. Read this doc. Execute the 3rd task — the acceptance gate. Use Fable if you can. Log
   your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can. Log your
   work.
5. Read this doc. Execute the 5th task — the PR review. Log your work.
6. Read this doc. Execute the 6th task — the prototype hand-off. Log your work.

## M4

### Tasks

1. **Comparison models.** Add **O25** to `robust/rt/baselines.py` (or a sibling module if
   it grows), matching the established convention: the same signature as `forward`,
   ignoring nothing it can legitimately see, and documenting what it is blind to. O25's
   form is fully specified above and its `(ω_bw, ω_bp)` split already lives in `IOPs`.
   Settle the coefficient question first (**Q8**): fetch the published `G` LUTs, or fit
   them per solar zenith on L23's **training** split and label the model *"O25 form,
   refit on L23"* everywhere it appears. A refit must use the train split only — fitting
   a comparison model on the test data would flatter the hybrid's rival, which is the one
   direction of bias nobody would think to check.

   Also, note that JXP has cleared all 3 Outstanding items.

   **PR05** is a judgement call rather than a coding task: see Q8. If it goes in, the same
   labelling rule applies.

   Tests: each model reproduces a hand-checked value; each is differentiable (the same
   harness scores them); and any refit is deterministic and demonstrably trained on the
   train split only.

2. **Validation module.** Fill out `robust/rt/validation.py` (currently `rrms` alone) with
   the design §6 protocol: rRMS **per λ**, **per solar zenith**, **per `B_p` bin**;
   metrics on **both** held-out splits; **throughput**; and the **gradient-correctness**
   check — every model on identical data, in `rrs` space. Then
   `design/py/run_validation.py` regenerates the metrics table and the figures.

   Report the **seed spread** for anything trained, not a single fit: M3 measured that one
   number hides a 2.6× range at an unseen geometry. A `seeds=` argument defaulting to
   several, and a table carrying a median and a range, is the honest shape.

3. **Acceptance gate.** `robust/tests/test_validation.py`. The plan's gate is the hybrid
   beating standard Gordon on **both** held-out splits plus the gradient gate; write it
   per the Q7 answer, and make the *reason* each assertion holds visible in its docstring.
   Keep the committed test cheap (the fixture plus a toy fit) and leave the real run to
   `run_validation.py`. Commit the metrics table and the figures. Update the record: fill
   §6, flip M4 in §1, bump the version.

4. **Notebook.** `notebooks/RT/rt_elastic_coding_5.ipynb` — the M4 explainer, following
   record §8 and the four existing notebooks: **executed** with outputs, degrading
   without `$OS_COLOR`, `sys.path` bootstrap, the house figure style, and the
   CVD-validated categorical set (blue `#0072B2`, vermillion `#D55E00`, sky `#56B4E9`,
   near-black `#1a1a1a`; validate any new colour — a green failed at ΔE 5.4).
   **Render and look at every figure**: doing so has caught a label collision, a clipped
   label, unreadable log ticks, a wrong colour encoding, and two different denominators
   on one axis — once each, in four notebooks.

   The story is the *protocol and its verdict*, not another accuracy boast: which model
   wins where, what the seed spread does to the zenith claim, and what the prototype may
   and may not say.

5. **PR Review.** Fetch the review comments on the open PR and address them. `gh` is not
   authenticated; the public REST API works:
   `curl -sS https://api.github.com/repos/ocean-colour/retrieve-or-bust/pulls/<n>/comments`
   and `.../reviews` — write the response to a file, since `conda run` does not forward
   stdin. Fix the *class* of each defect, not the single instance, and demonstrate the fix
   catches what was reported. If no PR is open, review the diff yourself and say so.

6. **Finally — the prototype hand-off.** The Week-1 prototype is complete. Prepare it
   (JXP issues and runs all git):
   - a short summary of what M0–M4 delivered, the validation numbers **with their
     caveats**, and the known gaps (nadir-only, fixed-FF L23; the 1.75× `B_p` slice; the
     unseen-zenith instability; the missing Equation (8) coefficients; PR05/O25
     provenance);
   - confirm `pytest` is green and the branch is clean;
   - then modify `rt_elastic_coding_prompt_6.md` (M5: beyond the prototype) given what M4
     established.

### Q&A

**Q7 (M3 → M4 hand-off, Claude → JXP). What should the zenith half of the M4 gate
assert?**

The plan wants the hybrid to beat standard Gordon on the unseen-60° split. M3 measured
that outcome as **seed-dependent**: 4.74 / 8.37 / 7.75 / 5.40 / 12.24% over five seeds
against Gordon's 9.01%, so four seeds pass and one fails. Meanwhile `mode="ztt"` beats
Gordon there deterministically (8.09%), and the linear emulator does better still
(6.16%), stably.

You have already said (Q6) that 60° extrapolation is a stretch goal and that we will not
use the emulator at larger angles without warning. Three ways to write the gate:

1. **Gate on the scene split; report the zenith split as a measured limitation**, with
   the full seed spread. Honest, and consistent with your Q6 answer — but it drops half
   of a plan-level criterion, so it is yours to approve.
2. **Gate the zenith split on the median over a fixed seed set** (say five). Keeps both
   halves, makes the claim reproducible, and forces the spread into the report. The
   median passes at 7.75% against 9.01% — with little margin.
3. **Add an out-of-domain policy to `forward` and gate on that.** Your Q6 phrasing was
   "we won't use the emulator at larger angles (**or** will warn the user)" — the first
   clause is a *fallback*, which is not yet implemented. An
   `on_out_of_domain ∈ {"warn", "ztt"}` option would make the hybrid degrade to the
   backbone exactly where M3 showed the emulator to be unreliable, and the gate would
   then pass deterministically at 8.09%, because that is the model actually being used.
   This is the option I would pick if you want both halves kept *and* a defensible claim;
   it is ~20 lines plus tests.

My recommendation: **3, with 1's reporting.** Unless you say otherwise I will implement
the fallback as an *option* (default unchanged, so nothing silently changes), gate the
scene split, and report the zenith split both ways — hybrid-with-fallback and
hybrid-without — with the seed spread. — *Not blocking task 1.*

**Q8 (M3 → M4 hand-off, Claude → JXP). PR05 and O25: fetch the published coefficients,
or re-fit their forms on L23?**

Neither model's coefficients are in the repo, and neither paper prints them (see above).

- **O25** is cheap either way. Its `G` LUTs are published as code at
  `github.com/jaipipor/O25` (also inside NASA HyperCP and EUMETSAT ThoMaS). Fetching
  needs network access and a licence check; re-fitting is a 4-parameter least squares per
  solar zenith, and the paper itself re-fits on L23. Re-fitting gives a *fair* comparison
  on our data but is no longer the published model, so it must be labelled.
- **PR05** is harder: a 4-D `(θs, θv, Δφ, γb)` LUT that exists only behind a 2005 MUMM
  URL or inside POLYMER. L23 is nadir-only, so we could never populate θv/Δφ ourselves.

My recommendation: **implement O25's form and re-fit it on the train split, labelled
"O25 form, refit on L23"**, and **drop PR05 from the gate**, saying in the report why
(coefficients unobtainable; a nadir-only refit of a BRDF model would be a different
object). If you would rather have the published numbers, say so — and tell me whether I
may fetch from GitHub, since I will not pull external code without your go-ahead.
— *Not blocking task 1: O25's form can be written before the coefficients are settled.*

## Next

→ `rt_elastic_coding_prompt_6.md` (M5: beyond the prototype).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
