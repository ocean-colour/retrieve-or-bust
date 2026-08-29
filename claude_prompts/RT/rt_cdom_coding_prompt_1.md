# CDOM Fluorescence Coding — Prompt 1 (M5: analytic term & interface)

## Goals

Implement **Milestone M5** of the CDOM-fluorescence design
(`design/rt_cdom_fluorescence_model.md`): the analytic CDOM-fluorescence
emission term on the Hawes et al. (1992) quantum-efficiency basis, the full
interface (`IOPs.a_cdom`, `CDOMFl(scale=1.0)`, the `Inelastic.cdom_fl` slot
retyped from reserved to live), and the correction head δ_C **defined but
untrained** — gated by the truth-less v1 acceptance criteria (design §5:
off-state bit-identity, correctness pins, literature-plausibility band,
gradients, speed). No head training in M5: no CDOM-fl truth exists anywhere in
hand (M6 is deferred until the HydroLight runs of design §7 land).

Work happens on a **fresh branch off `main`** (JXP creates it once
`inelastic-rt` is merged; the branch name is JXP's call — do not hardcode
one). The default model must remain provably CDOM-fl-free:
`Inelastic(..., cdom_fl=None)` (the default) stays **bit-identical** to the
shipped inelastic output, because the X4 truth and the reported 0.34 % gate
omit CDOM fluorescence.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`,
`code-review`) as helpful. Note: the `inelastic-rrs` skill documents **BING's**
Raman/Chl-fl wiring — BING has no CDOM-fluorescence implementation at all, so
that skill is background only here (there is no BING reference to cross-check
this term against).

### Working agreements (hold for the whole M5 prompt doc)

- **Git is handled by JXP** (per `CLAUDE.md`). Work on the fresh branch off
  `main` that JXP creates; the milestone is a reviewable commit/PR. Do **not**
  run state-changing git commands; read-only inspection is fine.
- **Python only**, in the `ocean14` conda env; **CPU-only JAX**.
- **Reuse, don't reinvent.** Build on the shipped `robust/rt` modules — the
  Ed module, the two-flow emission transport and quanta/energy bookkeeping in
  the Chl-fl kernel, the excitation-grid helpers, the pytree/validator
  patterns in `types.py`, and the two-tier (strict-hash / ULP-closeness)
  regression pattern. Do **not** install `requirements.txt` wholesale on a dev
  machine (its `git+` lines clobber editable `bing`/`ocpy` checkouts — prompt
  1 Q&A Q1 of the inelastic effort).
- **Every task is `pytest`-gated**, and from task 1 onward **both** existing
  hash-regressions — elastic (`inelastic=None`) *and* inelastic
  (`cdom_fl=None`, pinned in task 5) — must stay green.
  Use Fable if you can. Log your work.

## Context

Read before coding:

- **Design** — `design/rt_cdom_fluorescence_model.md` (all of it — it is
  short; §2 architecture, §3 interface, §5 the v1 gate, §8 risks).
- **Companion design** — `design/rt_inelastic_model.md` (§2 composition, §4.2
  Ed, §4.4 the Chl-fl kernel this term's transport mirrors, §4.5 the bounded
  head pattern).
- **Implementation record** — `design/rt_inelastic_implementation.md` (the
  conventions, environment, and gotchas the inelastic milestones established).

Where this sits: the inelastic prototype (Raman + Chl-a fluorescence) is
gate-passed and reported (`reports/report_rt_inelastic_model.md`, v1.0,
2026-08-27 — held-out 0.34 % rRMS vs X4, per-process ≤ 1.03 %, bit-identical
elastic off-state, 1.59× elastic runtime). This milestone adds the **third**
inelastic term, CDOM fluorescence, per the companion design: analytic-only,
default-off, head defined but untrained — because L23 omits CDOM-fl and BING
never implemented it, there is no truth channel and no cross-check reference;
correctness pins and a literature-plausibility band stand in (design §5) until
the design-§7 HydroLight runs unblock M6.

## Status entering M5

- The inelastic milestones M0–M4 are complete on `inelastic-rt`; the report
  refers to `main` (JXP merges before/as this effort starts). Suite state at
  the end of M4: **431 passed, 1 skipped** from the repo root in `ocean14`.
- `robust/rt/types.py` today: `IOPs` has `a_ph` (optional, validated ≤ a) but
  **no `a_cdom`**; `Inelastic.cdom_fl` is typed `Scalar | None = None` and its
  validator **rejects** any non-None value ("reserved hook") — task 1 retypes
  it to `CDOMFl | None`.
- `robust/rt/hybrid.py` composes `(Rrs_ZTT + ΔRrs) × f_R + Rrs_fl`; there is
  no CDOM branch. The two-tier regression pattern (strict SHA-256 pins on the
  dev machine, `skipif $CI`; ULP-closeness vs committed reference arrays
  everywhere) is established for the elastic off-state.
- The Ed module (`robust/rt/ed.py`), excitation-grid helpers
  (`conventions.py`), and the L23 loaders (`robust/rt/data/l23.py`, with
  `a_ph` extraction) all exist and are reused, not rebuilt. L23 stores a_g
  separately from a_nap, so `a_cdom` extraction mirrors the `a_ph` pattern.
- *(Fill in / correct on entry: branch name JXP created, exact test count on
  that branch, anything the merge changed.)*

## Prompts

1. Read this doc. Execute task 1 in the "M5" section below. If you have any
   questions, ask me in the Q&A section below. Use Fable if you can. Log your
   work.
2. Read this doc. Execute tasks 2–3. Use Fable if you can. Log your work.
3. Read this doc. Execute tasks 4–5. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Fable if you can. Log your work.
4. Read this doc. Execute tasks 6–7. Use Fable if you can. Log your work.
5. Read this doc. Execute task 8 — the notebook and record. Use Fable if you
   can. Log your work.
6. Read this doc. Execute task 9 — reviewing the pull request. Use Fable if
   you can. Log your work.

## M5

### Tasks

1. **Extend the types.** In `robust/rt/types.py`: `IOPs` gains optional
   `a_cdom` (default `None`; validated like `a_ph` — non-negative, ≤ a,
   shape-broadcast in `from_total_bb`, preserved by `select()`); new
   registered pytree `CDOMFl(scale=1.0)` (`scale` a differentiable leaf;
   frozen; validated finite and > 0); `Inelastic.cdom_fl` retyped
   `CDOMFl | None = None` — **`None` stays the default**, an instance is now
   accepted, and setting it is validated against the process being usable
   (clear error at `forward()` time if `iops.a_cdom is None`).

   **Gate.** `test_cdom_types.py`: pytree mechanics (flatten/unflatten,
   leaf/static split, jit/vmap/grad traversal of `scale`), validators, and
   bitwise indifference of the elastic and inelastic paths to a set-but-unused
   `a_cdom`. Both existing hash-regressions green; full suite passes.

2. **Loader wiring.** `robust/rt/data/l23.py`: extract `a_cdom` (= a_g) from
   L23 alongside `a_ph`, mirroring that pattern (marker-guarded on data-less
   machines). Pin the decomposition bookkeeping in tests: `a_cdom ≥ 0`,
   `a_ph + a_cdom ≤ a` on real scenes — the a_dg double-counting foot-gun of
   design §8.

   **Gate.** Loader tests green on this machine (data present) and skipping
   cleanly without `$OS_COLOR`; golden values for a few scenes.

3. **The Hawes kernel.** New `robust/rt/cdom_fl.py`: the Hawes et al. (1992)
   spectral fluorescence quantum-efficiency function η(λ′, λ) with its
   constants recorded (which published function/variant — HydroLight's
   default choice — and source, per design §7's matching requirement); the
   analytic kernel `K_cdom(IOPs, Ω, λ)` — source ∝ `a_cdom(λ′)`, `Ed(λ′)`
   weighting via the existing Ed module, excitation quadrature, quanta→energy
   factor, per-λ_em attenuation, two-flow transport with **L_u = E_u/π**, and
   A·rrs/(1−B·rrs), reusing the Chl-fl kernel's machinery wherever it
   transfers.

   **Gate.** Correctness pins (design §5.2): η reproduced against its
   published values; energy/quanta bookkeeping unit tests (e.g. emission
   red-shifted from excitation everywhere, quantum bookkeeping consistent
   with the tabulated efficiency); quadrature convergence under grid
   refinement (result stable to a stated rtol between the native grid and a
   2× refined grid).

4. **The 350 nm clamp + truncated-fraction diagnostic.** Impose the hard
   350 nm lower excitation limit (design §2); implement a committed diagnostic
   that quantifies, from the Hawes function itself, the fraction of emission
   truncated by the clamp per emission wavelength, and write the numbers into
   the implementation record as the documented caveat.

   **Gate.** Diagnostic test pins the truncated fraction (banded — it
   *characterizes*, it doesn't gate to zero); the kernel provably never reads
   IOPs or Ed below 350 nm.

5. **Composition + the extended bit-identity regression.** Wire
   `Rrs_total = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl + Rrs_cdom` into
   `hybrid.forward()` — the CDOM branch **unreachable** when `cdom_fl is None`
   (no-op by construction, not by arithmetic). Pin the new regression
   **before** wiring: `forward(..., inelastic=Inelastic(..., cdom_fl=None))`
   on the CI fixture, two-tier (strict hash on this machine, ULP-closeness
   fixture for CI), alongside the elastic pin.

   **Gate.** Both off-state regressions bit-identical/green; with
   `cdom_fl=CDOMFl()` set, output changes only additively at the kernel's
   value (spot-checked); full suite passes.

6. **δ_C head defined, untrained.** In `robust/rt/inelastic_corr.py`: the
   bounded (tanh-scaled) δ_C head per the δ_F pattern, **zero-initialized** so
   the untrained head is exactly the analytic backbone; a stubbed training
   entry point that raises with a message naming M6 and the missing truth.
   No weights file is committed (nothing to train on).

   **Gate.** Test: zero-init head ⇒ `(1 + δ_C) ≡ 1` bitwise; the stub raises
   informatively; composition with the head present equals composition
   without it.

7. **Plausibility, gradients, speed.** (i) The literature-plausibility band
   (design §5.3) on L23 IOPs: CDOM-fl contribution a few % of Rrs in the
   blue-green for CDOM-rich scenes, ≲ 1 % oligotrophic, monotone in a_g(440)
   — loose banded gates plus a reported table. (ii) `jax.grad` vs central
   differences for all inputs **including `scale`** and `a_cdom`. (iii) Speed:
   full-batch forward with CDOM-fl on ≤ **2×** the elastic hybrid (the
   existing budget, now including this term).

   **Gate.** All three as pytest tests (speed banded, per the established
   throughput-test pattern); this closes design §5 items 3–5.

8. **Notebook + record + docs note.** Executed notebook
   `notebooks/RT/rt_cdom_coding_1.ipynb` (the established `ocean14`
   kernelspec/nbconvert recipe; degrade without `$OS_COLOR`): what M5
   *decided* — the Hawes basis and its recorded variant, the clamp and its
   measured truncated fraction, why default-off is load-bearing, the
   plausibility table. Update `design/rt_inelastic_implementation.md` with an
   M5 section (or start a sibling record if cleaner — say which and why), and
   add a short changelog/docs note stating the term exists, is analytic-only,
   default-off, and **unvalidated until M6**.

   **Gate.** Notebook committed with outputs; record updated; the
   unvalidated-until-M6 language present verbatim.

9. **Pull request.** JXP will create the PR. Review it (multi-angle, per the
   inelastic M0 task-6 precedent), verify findings against live code, and fix
   what JXP triages in. Use Fable if you can. Log your work.

### Q&A

## Next

→ M6 (δ_C training + quantitative gate) is **deferred**: it opens only when
the HydroLight "X4 vs X4 + CDOM-fl" runs of
`design/rt_cdom_fluorescence_model.md` §7 exist. No prompt doc until then.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
