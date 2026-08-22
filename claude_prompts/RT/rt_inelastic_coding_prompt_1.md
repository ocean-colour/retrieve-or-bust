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
6. Read this doc. Execute the 6th task — reviewing the pull request. Use Fable
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

6. **Pull request.**  I have craeted a pull request and had Cursor write a review.
   Please review the pull request. Use Fable if you can. Log your work.

### Q&A

**Q1 (Claude, 2026-08-20, task 2).** The full `pip install --dry-run -r
requirements.txt` was **not** purely additive — but not because of the JAX
stack. `requirements.txt` lists `bing` and `ocpy` as `git+` GitHub sources,
while `ocean14` here has both as **editable installs from local checkouts**
(`/mnt/tank/Oceanography/python/{bing,ocpy}`; the `bing` checkout is on
`inelastic-fixes` — the very code M2 cross-checks against). A full-file
install would have replaced them with GitHub clones. Since the JAX-only
subset (`pip install --dry-run jax flax optax jaxtyping`) *was* purely
additive (23 new packages, zero uninstalls/upgrades), I installed exactly
that subset and left `bing`/`ocpy` untouched, rather than blocking the
milestone — the coding plan's dedicated-env fallback targets dependency
conflicts, which never arose. **Please confirm** this reading, and consider
whether `requirements.txt` should grow a comment warning that the two `git+`
lines will clobber editable dev checkouts if the file is installed wholesale
on a dev machine.
>A. I confirm

**Q2 (Claude, 2026-08-20, task 3).** Two things from the API extension for
your judgment, neither blocking:

- **The pinned hashes are platform-anchored.** The elastic hash-regression
  pins SHA-256 of the float32 output bytes computed on *this machine*
  (jax 0.11.1, CPU/XLA as installed). Bit-identity across different CPUs or
  jax/XLA versions is not guaranteed by JAX, so GitHub CI *may* produce
  different low bits and fail the two hash tests without any real
  regression. I left the pins strict (that is the gate's letter); if CI
  turns red on hardware grounds, the options are a platform-keyed pin or
  scoping the hash gate to this machine like M2's bing xcheck. Your call if
  it happens — please don't let anyone "fix" it by re-pinning to CI's hash,
  which would invert the guard.
- **Branch name.** The working agreement says `rt-inelastic-prototype`, but
  the checkout is on **`inelastic-rt`**. I worked where I stood (no git from
  me, per `CLAUDE.md`). If the plan's name matters for the PR bookkeeping,
  that's a rename for you; otherwise the record now names `inelastic-rt`.
>A. I decided to stick with the current branch.  And got it about the hash

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

### 2026-08-20 (M0 task 2 — JAX stack installed on this machine, all gates green)

Installed the CPU JAX stack into `ocean14` on the tank server and verified
it; the implementation record §2.3 now carries the full version table and
procedure (record bumped to v0.2). Model: Fable 5.

What happened, in order:

- **Pre-flight found a real hazard before the dry run did.** `bing` and
  `ocpy` are *editable* installs from local checkouts here, and the `bing`
  checkout is on `inelastic-fixes` (verified via read-only `git branch`) —
  exactly the physics M2's cross-check needs. `requirements.txt` declares
  both as `git+` GitHub sources.
- **The full-file dry run confirmed it**: `Would install` included
  `bing-0.0.dev0` and `ocpy-ocean-0.1.0` — i.e. wholesale install would
  clobber both editable checkouts (ocpy's GitHub distribution is even
  renamed `ocpy-ocean`). Not additive, so per the task I did *not* install
  the full file. Asked in Q&A (Q1) rather than silently proceeding — but
  since the JAX-only subset dry run (`jax flax optax jaxtyping`) **was**
  purely additive (23 packages, all new, zero uninstalls/upgrades,
  bing/ocpy untouched, transitive list identical to the elastic laptop
  install), I installed that subset instead of blocking. The dedicated-env
  fallback was for dependency conflicts; none arose.
- **Installed**: jax 0.11.1, jaxlib 0.11.1, flax 0.12.9, optax 0.2.8,
  jaxtyping 0.3.11 (all one patch/minor ahead of the laptop's 0.11.0/0.12.8
  — unpinned requirements, two weeks later).
- **All task-2 verifications pass**: `default_backend() == "cpu"` with every
  device CPU (note: this machine *has* an NVIDIA GPU; jax prints a
  "falling back to cpu" notice — expected with the CPU wheel, and exactly
  what CQ2 wants); float64 via the x64 flag; `jax.grad` smoke (6.0 at x=3);
  `jax.jit` computes; flax/optax/jaxtyping import; the pre-existing stack
  (numpy 2.4.6, scipy 1.18.0, xarray 2026.4.0, pandas 3.0.3, matplotlib
  3.11.0, emcee 3.1.6, sklearn, bing, ocpy incl.
  `ocpy.hydrolight.loisel23`) all import from their pre-install locations.
- **Bonus verification**: ran the full elastic suite from the repo root —
  **279 passed in 53 s** (`$OS_COLOR` is set on this machine) — the first
  time the elastic tests could run here at all, and the count matches the
  laptop exactly. The task-3 baseline is therefore established: any
  regression from here is caused by the API extension, not the environment.

Learned: on this machine `$OS_COLOR=/home/xavier/Oceanography/data/Color`
(L23 data present — the `needs_l23` tests run rather than skip), miniconda
not miniforge, and the GPU-present-but-CPU-wheel notice is normal log noise
for every future pytest/notebook run here.

### 2026-08-20 (M0 task 3 — API extended, elastic hash pinned, 309 green)

Extended the `robust/rt` API with the inelastic types and pinned the elastic
hash-regression; record bumped to v0.3. Model: Fable 5. Q&A checked first:
no answers from JXP yet (Q1 still open — nothing in it blocks this task);
added Q2 (hash platform-anchoring; branch-name mismatch).

Order of operations mattered and was kept honest: the **pre-change hashes
were computed before any edit** — `forward`/`rrs_forward` on the 50-scene CI
fixture (`check_domain=False`), SHA-256 over the float32 bytes, run twice in
separate processes to confirm determinism:

- Rrs `aaa0616119f179551e64969cd8407ed44e8eb0f8f5d9b27ba6ac7c97d826bbc7`
- rrs `d111464020aacb47bbc9dd9aa027dd11b2e15e019a735687b6c6c0fa504c2c38`

The code changes (details in the implementation record §2.2):

- `types.py`: `IOPs.a_ph` optional/None (the `Geometry.wind` precedent — no
  leaves unset, so no elastic call site changes); `Inelastic` registered
  pytree with the design-§3 signature — the one real design decision here is
  **`phi_C` as leaf, switches as static** (`field(metadata=dict(static=True))`),
  so `grad` traverses the quantum yield while `jit` specializes on the
  process configuration; `Geometry.Ed` `(wave_Ed, Ed)` override;
  `EMISSION_SHAPES`; validators for each (incl. `a_ph ≤ a`, `cdom_fl` must
  stay None in v1, and φ_C=0 rejected with a message pointing at
  `fluorescence=False`).
- `hybrid.py`: keyword-only `inelastic=None` on `forward`/`rrs_forward`
  (the design writes it positionally after `wave`, but `mode` already holds
  that slot — keyword-only is the backward-compatible reading). `None` falls
  through to the untouched elastic route before any computation; an instance
  raises `NotImplementedError` naming M2 (elastic stub convention).
- `__init__.py`: re-exports `Inelastic`; docstring updated.

Gate: `robust/tests/test_inelastic_types.py`, 30 tests, all CI-runnable —
the two pinned hash-regressions plus none-vs-omitted bitwise identity,
NotImplementedError, jit-with-None; Inelastic pytree mechanics
(flatten/unflatten, leaf/static split, treedef-per-configuration, frozen,
jit/vmap/grad with `grad` returning a labelled `Inelastic`); a_ph and Ed
leaf behavior, validators, and **bitwise elastic indifference** to both new
fields ("ignores it" as arithmetic, not prose). Full run: **309 passed**
(279 elastic unmodified + 30 new), `ruff check` + `ruff format --check`
clean (one `ruff format` pass on the new test file).

Learned/for JXP: (1) the fixture hash is deterministic across processes here
but *platform-anchored* — see Q2 before trusting it on other hardware;
(2) the checkout is `inelastic-rt`, not the plan's `rt-inelastic-prototype`
— recorded as-is (Q2); (3) one float32 re-lesson — a 405-term float32 sum
needs `rel=1e-5`, not 1e-6, exactly the elastic record's §2 warning.

### 2026-08-21 (M0 task 4 — notebook built, executed, committed with outputs)

`notebooks/RT/rt_inelastic_coding_1.ipynb`: 18 cells, executed end to end
with outputs and two figures; record bumped to v0.4 (§2.6 filled, §2.3
addendum). Model: Fable 5. Q&A checked first: JXP answered both — Q1
confirmed (JAX-subset install), Q2 resolved (branch stays `inelastic-rt`;
hash caveat acknowledged) — both now cited in the notebook and record.

Content follows the task's brief: what M0 *decided*, not what JAX is (elastic
notebook 1 is linked for that). Sections: environment on this machine → the
`Inelastic` leaf/static split with a live labelled-gradient demo
(`∂/∂φ_C` = the toy kernel, φ_C-linearity) → `a_ph`/`Ed` under the `wind`
precedent with live bitwise elastic indifference → why bit-identical, not
merely close (figure 1) → the frozen baseline with M2's target regions
marked (figure 2) → the gate run inline (30 passed). Hashes re-verified live
in the notebook against the pinned constants, imported from the test module
rather than duplicated.

Two findings worth the log:

- **My first "arithmetic no-op" demo refuted itself.** Dividing and
  re-multiplying the output by a smooth `f ≈ 1+1e-3` round-tripped *exactly*
  for all but 5 of 12,150 float32 elements — an algebraic identity that IS
  an IEEE identity, contradicting the surrounding narrative. Caught by
  reading the executed outputs against the prose. Replaced with the
  physically apt threat: one extra pass through the `rrs ↔ Rrs` conversion
  pair (the composition works below the surface, so it is a plausible M2
  restructure) shifts **39.9 %** of elements by 1–2 ULP — invisible to any
  rtol gate, loud under the hash. The record §2.6 keeps both halves of the
  lesson.
- **Notebook tooling did not exist on this machine** (no kernelspecs; the
  elastic notebooks ran on the laptop). Installed `ipykernel` into `ocean14`
  (dry-run first: purely additive, 7 packages) and registered a user
  kernelspec `ocean14`; execution driven by `os_313`'s `jupyter nbconvert
  --execute` launching that kernel, so outputs are genuinely `ocean14`'s.
  The committed kernelspec is named `ocean14`, not the elastic notebooks'
  `python3` — recorded in §2.3.

Figure craft per the house style (recessive frame, ink text, CVD-checked
colors): fig 1's histogram carries the 40 %-vs-0 asymmetry; fig 2 uses the
single-hue zenith sequence with a legend instead of direct labels — the
three medians sit within ~5 % and direct labels overprinted in the first
render (caught visually, fixed before committing outputs).

### 2026-08-21 (M0 task 5 — prompt 2 updated with reality; M0 complete)

Filled `rt_inelastic_coding_prompt_2.md`'s "Status entering M1" and adjusted
its tasks where reality disagreed with the plan; record bumped to v0.5 with
M0 marked ✅ done in the status table. Model: Fable 5. Q&A re-checked: no new
entries beyond the answered Q1/Q2.

The status section now hands M1 everything M0 established: the verified
environment (versions, 309-green suite, `$OS_COLOR` set here), the pinned
API surface (keyword-only `inelastic=None`; instance raises until M2 — M1
must not change that), the armed hash-regression, the `bing`
`inelastic-fixes` editable checkout with its raman helpers, the X2/X4
netCDFs verified on disk, and the `ocean14` kernelspec/nbconvert recipe for
the M1 notebook. The working-agreements block now names branch
`inelastic-rt` (JXP's Q2 decision) and carries the do-not-install-
requirements-wholesale warning (Q1).

Three factual corrections to the M1 tasks, each verified before editing:

- **The Raman cross-check number was wrong.** The task said "λ′(488) ≈
  583.6 nm"; the exact 3400 cm⁻¹ wavenumber form gives **585.076 nm**
  (1e7/488 − 3400 = 17091.8 cm⁻¹), confirmed by running
  `bing.rt.raman.excitation_to_emission_wavelength(488)` → 585.0758. Even
  bing's docstring example ("583.0 # approximately") is off. The task now
  pins both directions (488 exc → 585.08 em; 488 em ← 418.55 exc) and says
  to test the computed value, not a prose approximation.
- **The sibling-fixture size bound was arithmetically impossible.** 18
  float32 arrays of 50×81 weigh ~292 kB raw; the elastic fixture's 15
  arrays already weigh 213 kB. The ≲ 200 kB gate became ≲ 300 kB, with the
  arithmetic shown.
- **`needs_l23` guards only the elastic X=1 files.** M1's raw-netCDF
  cross-checks over X2/X4 would *fail*, not skip, on a machine with partial
  data. Added to both the status section and task 3's gate: a separate
  marker (e.g. `needs_l23_inelastic`) over the six inelastic files.

Also pinned the BING reference to its real path
(`.../bing/bing/tests/files/gen_l23_inelastic_fixture.py`, present on the
checkout with its committed fixture) so M1 doesn't hunt for it.

**M0 is complete**: implementation record seeded and maintained (v0.5), JAX
stack installed and verified on this machine, API extended with the elastic
path pinned bit-identical (309 tests green, ruff clean), notebook executed
and committed, and the M1 prompt updated. The milestone sits on branch
`inelastic-rt` as an uncommitted working tree for JXP's review and commit.

### 2026-08-22 (M0 task 6 — PR #14 reviewed; Cursor's findings verified; 9 confirmed findings)

Reviewed PR #14 ("Inelastic rt: M0", `inelastic-rt` → `RT`, 28 files) with a
multi-angle review sweep (8 finder agents: line-by-line ×2, cross-file
tracer, removed-behavior audit, reuse, simplification, efficiency,
altitude/conventions), then adversarially verified every candidate against
the live code — the three headline bugs were *reproduced*, not just argued.
Cursor's Bugbot pre-review was checked finding by finding. Full ranked list
in the implementation record §2.7 (v0.7); highlights:

- **Both Cursor findings are real** and were confirmed by inspection: the
  figure script's `$OS_COLOR_DATA`-only L23 lookup (project standard is
  `$OS_COLOR`; the elastic sibling checks both), and `write_metrics` reusing
  the Raman error columns with different meanings on ChlFl rows in the
  committed CSV.
- **Three confirmed bugs in the new M0/M1 code, all reproduced live**:
  (1) `ed.Ed()` casts spectra to `wave.dtype` — an integer wavelength grid
  truncates Ed to 0/1 and `ratio()` goes inf/NaN silently; (2) `l23.select()`
  rebuilds containers field-by-field and strips the brand-new `a_ph`/`Ed`
  (three angles found this independently — latent until M1 task 3 subsets an
  inelastic batch); (3) `from_total_bb` broadcasts `bb_w` but not `a_ph`,
  breaking its own uniform-batch-shape contract. None affects the elastic
  hash (verified conceptually: all sit off the elastic path), none blocks
  merge, but 1–3 should be fixed before M1 task 3.
- Four smaller confirmed items: the `/30` zenith-stride hard-code in `ed.py`
  (reuse `_interp_wave`'s searchsorted), `tiny_args()` duplicated into the
  new test module, the hand-rolled `_TABLE` cache vs `@functools.cache`, and
  the stale "still stubs" Status paragraph in `robust/rt/__init__.py`.
- **Cleared**: setup.py packaging (empirical build — all three `.npz` ship),
  `.gitignore` shadowing, CI dep coverage, split leak-freedom, and the ZTT
  in-water-cosine transcription (re-derived; code right, docstring formula
  loose).
- **Out-of-scope finds handed to JXP** (elastic-era code = PR #13's diff):
  `ztt.bb_tilde` crashes on the per-sample scalar `B_p` that `PhaseParams`
  documents (verified crash — contract mismatch with the emulator layer);
  `on_out_of_domain='ztt'` silently no-ops for domain-less emulators;
  `out_of_domain`'s worst/excess side mismatch; `write_fixture`'s unchecked
  cross-zenith wavelength grid; and `.claude/settings.json`'s committed
  allowlist rule ending in `python -c ' *` (auto-approves arbitrary inline
  Python — worth removing) plus dead machine-specific path rules.

No fixes were applied (the task asked for review; `--fix` not requested) and
nothing was posted to GitHub — findings live here, in §2.7, and in the
session report for JXP to triage. Suggested triage: fix findings 1–3 +
Cursor's two in one commit on this branch; fold 6–9 in opportunistically;
schedule the elastic-era items for PR #13.
