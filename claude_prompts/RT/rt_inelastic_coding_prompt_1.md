# Inelastic RT Coding — Prompt 1 (M0: Environment & API extension)

## Goals

Implement **Milestone M0** of the coding plan
(`design/rt_inelastic_model_coding_plan.md`): install the JAX stack into
`ocean14` **on this machine** (it is absent here — the elastic install record
describes the laptop), extend the `robust/rt` API with the inelastic types,
and prove the elastic behavior is untouched. Nothing scientific yet — the
point is a green base whose `inelastic=None` path is **bit-identical** to the
current elastic hybrid.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`,
`code-review`) as helpful.

### Working agreements (hold for every M-prompt)

- **Git is handled by JXP** (per `CLAUDE.md`). Work on branch
  **`rt-inelastic-prototype`** (JXP creates it); each milestone is a
  reviewable commit/PR. Do **not** run state-changing git commands;
  read-only inspection is fine.
- **Python only**, in the `ocean14` conda env on **this machine (tank
  server)**; **CPU-only JAX**.
- **Reuse, don't reinvent.** Build on the elastic `robust/rt` modules, the
  installed `bing` package (the *fixed* inelastic physics lives on its
  `inelastic-fixes` branch / PR), and `ocpy`'s L23 loader. Follow the
  conventions already recorded in `design/rt_elastic_implementation.md`
  (§2.6, §8): tests in `robust/tests/`, `ruff.toml` rules, executed
  notebooks, run `pytest` from the repo root.
- **Every milestone is `pytest`-gated**, and from M0 onward the **elastic
  hash-regression** (`inelastic=None` bit-identical) must stay green.
  Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — `design/rt_inelastic_model_coding_plan.md` (Ground rules,
  Package layout, M0).
- **Design** — `design/rt_inelastic_model.md` (§3 interface: `IOPs.a_ph`,
  `Inelastic`, `Geometry.Ed`; §1 the bit-identical guarantee).
- **Elastic implementation record** — `design/rt_elastic_implementation.md`
  (§2.3 the install procedure and its verification gate; §2 the M0 gotchas:
  `pytest` from repo root, the `jax_x64` fixture, float32 tolerances).

## Status entering M0

The elastic Week-1 prototype is complete and merged (see
`design/prototype_summary.md`): `forward(iops, phase_params, geometry, wave)`
is pinned in `robust/rt/hybrid.py`, 279 tests pass, CI runs on GitHub with a
committed 50-scene fixture, `ruff.toml` and `ruff format` are adopted.
`ocean14` **on this machine** has no `jax` (verified 2026-08-20); the stack
is declared in `requirements.txt` already.

## Prompts

1. Read this doc. Execute the 1st task in the "M0" section below. If you have
   any questions, ask me in the Q&A section below. Use Fable if you can. Log
   your work.
2. Read this doc. Execute the 2nd task. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can.
   Log your work.
5. Read this doc. Execute the 5th task — modifying the next prompt doc,
   `rt_inelastic_coding_prompt_2.md`, given what we have done here. Use Fable
   if you can. Log your work.

## M0

### Tasks

1. **Create the implementation record.** New file
   `design/rt_inelastic_implementation.md`, mirroring the elastic record: a
   milestone/status table (M0–M4 from the coding plan) and a per-milestone
   section for modules added, environment, tests, and results. Seed it with
   M0-in-progress.

2. **Install the JAX stack on this machine.** `pip install --dry-run -r
   requirements.txt` in `ocean14` first — verify the install is **purely
   additive** (nothing uninstalled/upgraded; the elastic M0 procedure).
   Then install and verify: `jax.default_backend() == "cpu"`; float64
   available via the x64 flag; `jax.grad` smoke test; and the pre-existing
   stack (`numpy, scipy, xarray, pandas, matplotlib, emcee, bing, ocpy`
   incl. `ocpy.hydrolight.loisel23`) still imports. Record exact versions
   in the implementation record. If the dry run is *not* additive, stop and
   ask in Q&A (fallback per the coding plan: a dedicated env).

3. **Extend the API.** In `robust/rt/types.py`: `IOPs` gains optional
   `a_ph` (default `None`; elastic path ignores it); new
   `Inelastic(phi_C=0.02, raman=True, fluorescence=True,
   emission_shape='single', cdom_fl=None)` pytree; `Geometry` gains optional
   `Ed` override (`(wave_Ed, Ed)`). In `robust/rt/hybrid.py`:
   `forward(..., inelastic=None)` — when `None`, take the *existing code
   route* (no-op by construction, not by arithmetic). Register the new
   pytree the same way the existing ones are registered.

   **Gate.** `robust/tests/test_inelastic_types.py`: pytree
   flatten/unflatten, defaults, `jit`/`vmap` traversal. **Elastic
   hash-regression**: `forward(..., inelastic=None)` on the elastic CI
   fixture is bit-identical (hash the output array) to the pre-change
   result — pin the hash now, it guards every later milestone. The full
   existing elastic suite passes unmodified. Update the implementation
   record; note the branch state for JXP.

4. **Notebook.** `notebooks/RT/rt_inelastic_coding_1.ipynb` — executed and
   committed with outputs, following the elastic notebook conventions
   (degrade gracefully without `$OS_COLOR`; `sys.path` bootstrap; figures
   per the recorded style). Explain what M0 *decided*: the `Inelastic`
   pytree shape and why `inelastic=None` must be bit-identical rather than
   merely close; the environment verification. Link to elastic notebook 1
   rather than re-explaining JAX.

5. **Finally.** Modify the next prompt doc,
   `rt_inelastic_coding_prompt_2.md`, given what M0 actually established
   (fill its "Status entering M1" section; adjust tasks if reality
   disagreed with the plan). Log your work.

### Q&A

## Next

→ `rt_inelastic_coding_prompt_2.md` (M1: Ed, excitation grid, X2/X4 data).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-20 (M0 task 1 — implementation record created)

Created `design/rt_inelastic_implementation.md` (v0.1), seeded with
M0-in-progress. Model: Fable 5.

Structure mirrors the elastic record (`rt_elastic_implementation.md`):
the same front matter (version/date/status/companions block, versioning
convention, the pointer to prompt Logs for chronology), a "Status at a
glance" table — here M0–M4 from the coding plan, with the planned package
surface per milestone — and per-milestone sections. M0's section is fully
scaffolded (task-status table for this prompt doc's five tasks, plus
Modules / Environment / Tests / Results / Notebook subsections carrying the
*planned* content from the coding plan as placeholders to be replaced by
actuals); M1–M4 get goal-only stub sections to be expanded when each opens.
A "Module index" section is stubbed at the end, mirroring the elastic
record's §9.

Choices worth noting:

- The record states up front the two things that differ from the elastic
  effort and are easy to get wrong later: **this machine (tank server), not
  the laptop** — `ocean14` here has no `jax` as of today, and the elastic
  record's §2.3 install table describes the *laptop* env — and the
  **absolute** (≤ 5 %) per-process gates vs the elastic effort's
  relative-only philosophy (DQ6/CQ3 decisions), including the bing
  `skipif`-on-CI behavior.
- The elastic hash-regression ("bit-identical by construction, not by
  arithmetic") is written into §1 as a recurring gate banner, since it
  guards every milestone from M0 on.
- Task 1 marked ✅ in the M0 task table; tasks 2–5 ⬜. The "Verification
  (current)" line is an explicit placeholder — no pytest run is claimed,
  since nothing was built yet.

Read before writing: this prompt doc, the coding plan (Ground rules, layout,
M0–M4 gates), the inelastic design header, and the elastic record (front
matter, §1, §2 in full; section skeleton of §3–§10) to mirror faithfully.
No code, environment, or git state was touched.
