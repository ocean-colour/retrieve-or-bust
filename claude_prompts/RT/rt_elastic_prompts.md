# Elastic Radiative Transfer (RT)

This prompt doc will guide our efforts to develop elastic radiative transfer (RT) models for ocean color.

## Context

- Read the `context/context_summary.md` file.
- Examine the code and Notebooks in the `Oceanography/python/bing/dev/Gordon` folder

The standard approach in the community is to adopt the so-called "Gordon approximation" for the RT.  This is a simple model that relates the water-leaving radiance to the inherent optical properties (IOPs) of the water.  The Gordon approximation is given by:

```
Rrs(λ) ≈ f/Q · bb(λ) / [a(λ) + bb(λ)]
       = Σ_i  G_i · u(λ)^i        with   u(λ) = bb(λ) / [a(λ) + bb(λ)]
```
where `Rrs(λ)` is the water-leaving radiance at wavelength `λ`, `f/Q` is a factor that depends on the illumination/viewing geometry and the volume scattering function (VSF), `bb(λ)` is the backscattering coefficient at wavelength `λ`, `a(λ)` is the absorption coefficient at wavelength `λ`, `u(λ)` is the single-scattering albedo at wavelength `λ`, and `G_i` are the geometric factors. The `G_i` are the geometric factors that depend on the illumination/viewing geometry and the volume scattering function (VSF).

JXP has found after doing a rather deep-dive that: (1) there is a wavelength dependence to the `G_i` and (ii) they are (not surprisingly) sensitive to `bb`.  

Read the papers (PDFs) in `context/RT/`. These are:

- `hansen1971.pdf` : An RT paper by James Hansen in the context of planetary atmospheres.
- `gordon1988b.pdf` : The original Gordon approximation paper.
- `park2005.pdf` : A 4th order polynomial fit to the Gordon approximation and also includes Gordon coefficients that are a function of `gamma_b = bbp/bb`.
- `tan2018.pdf` : A recent examination of the `park2005.pdf` approach
- `pitarch2025.pdf` : A recent examination of bidirectional reflectance distributions (BRDF)

## Code

If you need to run Python use the `ocean14` conda environment. 

## Prompts

### Setup

1.  Let's start from the work done by JXP in the BING repo.  Please review that work. We will write a report named `context/rt_bing_summary.md`.  Before doing so, please have a conversation with me in Q&A/Setup.  Use Fable if you can.  Log your work. 

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

3. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

4. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Now proceed to write the report.  Be sure to include figures.  Generate them with Python and save the script(s) to the `context/RT/` folder. Use Fable if you can.  Log your work.  Note that I have put the L23 data on this laptop now in `$OS_COLOR_DATA/Loisel2023/`.

5. I decided to move the report into the `context/RT/` folder.  Please update it so the figures render properly.  Use Fable if you can.  Log your work.

### Edits

1. I've pasted in comments from Robert Frouin on the RT.  Please review them and revise the report accordingly.  Use Fable if you can.  Log your work.

### Design

1. Ok, let's generate a plan together for how to proceed.  We are going to write a design document for this elastic RT effort.  We will name it `design/rt_elastic_model.md`.  Please ask me a set of questions in Q&A/Design before writing the document.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

3. The design document looks great!  Please generate a separate coding plan with a staged implementation plan including milestones.  Name it `design/rt_elastic_model_coding_plan.md`.  Please ask me a set of questions in Q&A/Coding before writing the document.  Use Fable if you can.  Log your work.

4. I have answered your questions in the Q&A section below.  Please review them and proceed to write the coding plan.  Use Fable if you can.  Log your work.

5. The coding plan looks great!  Please generate a series of prompt docs to execute the coding plan.  Name them `claude_prompts/RT/rt_elastic_coding_prompt_1.md`, `claude_prompts/RT/rt_elastic_coding_prompt_2.md`, etc.  Model them after the ones in the PAB repository (in that `claude_prompts` folder). Use Fable if you can.  Log your work.

### Report

1. We have worked through all of the prompt docs.  Please write a report summarizing the work we have done.  Name it `report/rt_elastic_model.md`.  Please ask me a set of questions in Q&A/Report before writing the document.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them.  Then proceed to write the report.  Use Fable if you can.  Log your work.

### PR

1. As you will note, I am on the `rt-eleastic-prototype` branch of the RoB repo.
I have just pulled in `origin/main` and with it a slew of conflicts.  Please examine
where things have diverged and why.  Write a report in Forensics below and offer 
suggestions on how to resolve this.  If you have any questions, put them in the
Prompts/PR/Q&A section.  
Use Fable if you can.  Log your work.

2. I have answered the questions in the Q&A section below.  Please review my
answers and ask another round to continue the Discussion.
Use Fable if you can.  Log your work.

3. I have answered the questions in the Q&A section below.  Please review my
answers and proceed to resolve the conflicts.
Use Fable if you can.  Log your work.

4. The merge is committed and the tree is clean.  Now do the code-and-numbers
half of the punch list, in this order because the order matters:

   a. Fix the three `:func:`robust.rt.forward`` cross-reference targets that the
   docs sweep renamed to `:func:`~robust.rt.hybrid.forward`` everywhere else —
   `robust/tests/conftest.py:267`, `robust/rt/conventions.py:757`,
   `robust/rt/types.py:277`.  The `-W`/nitpicky Sphinx job is red until they go.

   b. Fix the PQ10 contamination.  `robust/rt/files/surface_pb24.npz` was fitted
   under a 400-realisation split while the benchmark splits 200 and the gate 800,
   and `make_splits` permutes whatever set it is given — so `SPLIT_SEED = 23`
   does not name a partition (`design/m5_report.md` §6).  Refit the transfer on
   each consumer's own train mask, regenerate the artefact, and say in
   `design/m5_report.md` §6 that it is fixed and how.  Do this **before** (c):
   it moves the PB24 numbers.

   c. Re-run and regenerate, per PQ11 — `pytest robust/tests`, then
   `design/py/run_validation.py` (elastic and `--inelastic`) into
   `design/validation/`, then `design/py/run_pb24_validation.py` into
   `design/validation_pb24/`.  **Commit only what moved.**  Report every number
   that changed and why, and say plainly if one moved that you cannot explain.
   Note that `design/py/train_emulator_pb24.py` exits 1 by design when the gate
   fails — that is M5 working, not a regression.

   Use Fable if you can.  Log your work.

5. Now the documentation half, and Route C with it.

   a. PQ7, prose only: `docs/model/forward.md` and `docs/using/limitations.md`
   present `on_out_of_domain="ztt"` as a safe degradation.  Add the measurement
   that qualifies it — off-nadir the backbone it falls back to is non-physical on
   22.3% of PB24 (`design/m5_report.md` §2.1).  Do not change the policy's
   behaviour.

   b. Give M5 its documentation: an `automodule` for `robust.rt.data.pb24` in
   `docs/api.rst`, and a `docs/model/surface.md` for the geometry-aware surface
   transfer, the OLCI grid, and the per-model `Envelope`.

   c. Execute Q18 Route C, which I have approved: state the envelope where the
   backbone is valid (ψ ≳ 134°, `bb/a` ≤ 0.1), report the coverage honestly, and
   per Q19 make the current model's failing clear in `design/prototype_summary.md`.
   Scope the claim to (a) from PQ8 — all processes, nadir, L23-like water.

   d. Confirm `cd docs && make html` passes with `-W`, then draft the PR
   description: what the branch brings, what it does not claim, and what M6 is
   now that Route C is done and the HydroLight runs are the next major task.

   Use Fable if you can.  Log your work.

#### Forensics

**Report of 2026-09-19. Working tree is mid-merge** (`.git/MERGE_HEAD` present,
`Merge remote-tracking branch 'origin/main' into rt-elastic-prototype`), 9 files
unmerged. Nothing below has been resolved or committed — this is diagnosis only.

##### 1. The short version

Nothing is lost and nothing is duplicated. This is a **stale-fork** merge, not a
collision of two implementations. The branch was merged away on 2026-08-06,
kept going for another week (M5), and was never merged again; everything else
moved on without it for five weeks. The two sides touched 12 of the same files,
9 of which conflict, and **every conflict is "both sides added different things
in the same place"** — union resolutions, except for exactly one hunk that
needs a two-word edit (§5.1). There is no case where the two sides disagree
about what the code should do.

##### 2. Where the fork is

```
12cbcbf  2026-08-06  "mo"   <-- merge base, and the tip that PR #12 merged
   |
   |  OURS: rt-elastic-prototype, 10 commits, 2026-08-08 .. 2026-08-13
   |        39 files, +10,679 / -175   (M5: PB24 + off-nadir)
   |        8295fec setting up the tasks ... 69d577f "M5 done"
   |
   |  THEIRS: origin/main, 84 non-merge commits, 2026-08-21 .. 2026-09-18
   |          167 files, +49,352 / -567 (inelastic, CDOM, Ed/solar, docs, proposals)
   v
3863b3f  2026-09-18  Merge pull request #24 from vicc-proposal  <-- origin/main
```

`git cherry origin/main rt-elastic-prototype` reports all 10 local commits as
`+` — **none of the M5 work is on main in any form.** Conversely none of
main's 84 commits are here.

##### 3. Why it diverged

The PR chain tells the story. PRs #9–#12 merged `rt-elastic-prototype` into the
long-lived **`RT`** branch, not into `main`:

| merge | date | what it merged |
|---|---|---|
| PR #9, #10, #11 | 2026-08-01 … 08-04 | `rt-elastic-prototype` → `RT` |
| **PR #12** | **2026-08-06** | `rt-elastic-prototype` @ `12cbcbf` → `RT` |
| PR #14 | 2026-08-23 | `inelastic-rt` → `RT` |
| PR #21 | 2026-09-11 | `cdom-rt` → `RT` |
| PR #13 | 2026-09-18 | **`RT` → `main`** (the whole five weeks, at once) |
| PR #22, #23, #24 | 2026-09-18 | `profxj-patch-1`, `inelastic-rt`, `vicc-proposal` → `main` |

So:

1. **2026-08-06** — PR #12 takes the prototype at `12cbcbf` into `RT`. At that
   point the two lines are identical. This commit is the merge base.
2. **2026-08-08 → 08-13** — M5 is developed *here*, on `rt-elastic-prototype`,
   and never PR'd. `inelastic-rt` and `cdom-rt` were cut from `RT` at or after
   PR #12, i.e. **without M5**.
3. **2026-08-21 onwards** — the inelastic and CDOM work edits the very same
   modules M5 was editing (`types.py`, `validation.py`, `conventions.py`,
   `conftest.py`), unaware of it. First `RT`-side commit to touch `robust/rt/`
   after the base is `1d85f48`, 2026-08-21 — eight days after M5 finished.
4. **2026-09-11 → 09-18** — the docs effort (`rt_docs_prompt_1.md`) sweeps
   every module for Sphinx cleanliness, touching another two (`ztt.py`,
   `baselines.py`).
5. **2026-09-18** — PR #13 lands five weeks of `RT` on `main` in one merge, and
   `git pull` now confronts a branch whose last common ancestor is 08-06.

There is no rebase, no force-push, no rewritten history, and no duplicated
implementation. **One fact explains all nine conflicts: M5 never went back.**

##### 4. What each side actually added

| file | ours (M5) | theirs (inelastic / CDOM / docs) | conflicts |
|---|---|---|---|
| `robust/rt/types.py` | `bb_w_mode` kwarg on `IOPs.from_total_bb`; `PhaseParams` extension | `a_ph`, `a_cdom` fields + kwargs | 3 |
| `robust/rt/validation.py` | `FD_STEPS_EXTRA`, `default_steps`, masked `rrms`, `group_rrms(expected=)` | `_grad_vs_fd` refactor, `inelastic_gradient_report`, `cdom_gradient_report`, gate constants | 3 |
| `robust/rt/conventions.py` | `SurfaceTransfer`, `SURFACE_TABLE`, `OLCI_GRID`, `wave_grid`, `bb_w(mode=)` | Raman excitation grid, `interp_spectrum` | 1 (`__all__`) |
| `robust/rt/emulator.py` | per-model `Envelope` replacing the module constant | docstring/Sphinx polish | 2 (docstrings) |
| `robust/rt/hybrid.py` | envelope wording | inelastic composition + docstring polish | 1 (comment) |
| `robust/tests/conftest.py` | PB24 fixture + `needs_pb24` | `needs_l23_inelastic`, `needs_weights` | 2 |
| `robust/tests/test_{conventions,emulator,validation}.py` | new M5 blocks appended | new inelastic blocks appended | 1 each |
| `robust/rt/ztt.py` | `P_bb_from_phase`, `P_BB_PIVOT_DEG` | docstring polish only | auto-merged |
| `robust/rt/baselines.py` | `O25Table`, `fit_o25_table` | docstring polish only | auto-merged |
| `design/py/run_validation.py` | named-group tables (25 lines) | `inelastic_main`, figures (+684) | auto-merged |

Everything else is disjoint: M5 alone owns `robust/rt/data/pb24.py` (1,479
lines), `robust/rt/files/surface_pb24.npz`, `robust/tests/test_pb24.py`,
`design/validation_pb24/`, `design/m5_report.md`,
`claude_prompts/RT/rt_elastic_coding_prompt_7.md`; main alone owns
`inelastic.py`, `inelastic_corr.py`, `cdom_fl.py`, `ed.py`, `solar.py`, all of
`docs/`, `reports/`, and `proposals/`.

Note that main *anticipated* M5 without implementing it — `emulator.py:738`
("This keeps the M5 seam intact"), `validation.py:643` ("…the moment M5 goes
off-nadir"), `test_validation.py:714`, and the `pb24` references throughout
`design/` and `docs/using/limitations.md` are all placeholders written against
the plan. M5 fills those seams; it does not contradict them.

##### 5. Conflict-by-conflict resolution

**Take the union** for eight of the nine files — the two sides added different
kwargs, different `__all__` entries, different fixtures, or appended different
test blocks at the same end-of-file:

- `types.py` ×3 — keep `bb_w_mode` **and** `a_ph`/`a_cdom` in the signature and
  in the numpydoc block. The *bodies* already auto-merged with both (`mode=bb_w_mode`
  and `if a_ph is not None:` both survive below the marker), so dropping either
  side of the signature yields a `NameError` at import. For the `PhaseParams`
  docstring take HEAD's past tense ("M5 task 14 added…") — it is the same claim,
  now true.
- `conventions.py` ×1 — both `__all__` blocks, in order.
- `conftest.py` ×2 — both fixture sets; they share no names.
- `test_conventions.py`, `test_emulator.py`, `test_validation.py` ×1 each — both
  appended blocks; no shared test names, no shared helper names
  (`_in_domain`/`_wide_envelope` are M5-only).
- `emulator.py` ×2, `hybrid.py` ×1 — prose only. **Take HEAD**: M5 genuinely
  replaced `SUPPORTED_THETA_S` with the per-model `Envelope`, so main's text is
  now stale. The constant itself survives as `DEFAULT_ENVELOPE`'s default
  (`emulator.py:204,248`, asserted by `test_emulator.py:805`), so main's two
  code-level references (`test_validation.py:295,579`) keep working.

**5.1 The one hunk with teeth — `validation.py:846`.** Main refactored
`gradient_report`'s comparison loop into a shared `_grad_vs_fd` helper so the
three reports cannot drift; M5 rewrote the *setup* above it (the `perturbable`
dispatch that made `theta_v`/`dphi`/`bb_w` expressible). Both changes are
wanted, but main's replacement line reads

```python
dtype = jnp.asarray(a0).dtype
```

and `a0` is a local that only exists in **main's** version of the function —
M5's rewrite removed that binding. A literal `--theirs` on this hunk gives a
`NameError` the moment any gate runs. Resolve as:

```python
    dtype = jnp.asarray(iops.a).dtype
    return {
        name: _grad_vs_fd(scalar, name, step, dtype) for name, step in steps.items()
    }
```

I diffed `_grad_vs_fd` (`validation.py:729`) against HEAD's inline loop: it is
line-for-line the same arithmetic and the same two edge cases (`inf` for a step
that left the domain, `0.0` for a genuinely-ignored variable), so this loses no
M5 behaviour. After it, `partial` and `np` both remain used.

##### 6. What will still be broken after the markers are gone

The merge will be *clean* before it is *green*. Three items, in the order they
will bite:

1. **The docs build will fail, and it is a required job.** Main added a
   `sphinx (-W)` CI job (`ci.yml:96`) and `conf.py` sets `nitpicky = True`
   (`conf.py:197`), so a dangling cross-reference is an error. M5 was written
   five weeks before that existed and still spells the old target:
   `robust/rt/types.py:284`, `robust/rt/conventions.py:760` and
   `robust/tests/conftest.py:267` say `:func:`robust.rt.forward``, which the
   docs sweep renamed to `:func:`~robust.rt.hybrid.forward`` everywhere else.
   Three edits.
2. **`docs/api.rst` has no page for M5.** It carries an `automodule` for
   `robust.rt.data.l23` but none for `robust.rt.data.pb24`, and nothing under
   `docs/model/` mentions PB24, the OLCI grid, `SurfaceTransfer` or the
   `Envelope`. Not a build failure, but the published API would silently omit
   the largest module on this branch. Roughly a `pb24` section in `api.rst`
   plus a `docs/model/surface.md`.
3. **Cross-checks nobody has run.** `types.IOPs` now carries five optional
   leaves from two independent authors; `hybrid.forward` composes inelastic
   processes that have never seen an off-nadir geometry or a 12-band OLCI grid;
   `run_validation.py` has an elastic path (ours) and an `inelastic_main`
   (theirs) that were merged by text alone. The full suite plus both
   validation drivers is the only thing that will tell us.

##### 7. Recommendation

**Resolve in place on this branch and PR it into `main`** — do not rebase, do
not cherry-pick, do not restart M5 on a fresh branch off `main`.

Rebasing 10 commits over 84 would replay these same nine conflicts up to ten
times with no benefit; the merge commit is also the honest record of what
happened. Cherry-picking M5 onto a fresh branch is the same work minus the
history. The conflicts are small and all of them are additive.

Suggested order:

1. Resolve the eight union files; do `validation.py:846` per §5.1.
2. `ruff check robust/ && ruff format --check robust/` (main's CI runs both).
3. `pytest robust/tests` under `ocean14` — expect the PB24 tests to run from
   the committed fixture and the inelastic ones from the committed weights, so
   a green run needs no `$OS_COLOR`.
4. Fix the three `:func:` targets, then `cd docs && make html` (that Makefile
   target already passes `-W --keep-going`).
5. Re-run both validation drivers and diff the committed tables — `design/py/run_validation.py`
   (elastic + `--inelastic`) and `design/py/run_pb24_validation.py` — to confirm
   the merge moved no numbers.
6. Add the `api.rst`/`docs/model` pages for M5, then open the PR.

Items 1–3 are the merge. Items 4–6 are the price of arriving five weeks late
into a repo that grew a docs gate while we were away, and they are worth doing
inside this PR rather than after it.

**And the process fix, which matters more than the merge:** the prototype
branch was treated as long-lived while `RT` was the actual integration branch.
The Q&A/Report round in this very document (2026-08-15) and the report it
produced (`reports/report_rt_elastic_model.md`) were written on the `RT` line —
i.e. the elastic *narrative* moved on without the elastic *code*. Whatever M6
looks like, cutting it from `main` after this PR lands, and PR'ing it back
within days rather than weeks, avoids a repeat.

#### Q&A

**PQ1.** §7 recommends resolving in place and PR'ing this branch into `main`.
Is that what you want, or would you rather I produce the resolution as a patch
for you to inspect before anything touches the working tree? (You own all git
commands; I would resolve files and stop before `git add`/`git commit` either
way — the question is whether you want the resolved files written into the tree
now.)

>A. Yes, a PR into `main` sounds sensible.  We will do that.

**PQ2.** §5 says "take HEAD" for the `emulator.py`/`hybrid.py` prose conflicts,
on the grounds that M5's per-model `Envelope` superseded the `SUPPORTED_THETA_S`
narrative. That silently retires wording that shipped in the published docs.
Confirm that is the intent — or would you rather keep main's sentence and have
M5's envelope described in an added paragraph?

>A. I will follow your recommendation here.

**PQ3.** How much of §6 belongs in this PR? My instinct is items 1 (the three
`:func:` fixes, mandatory — CI is red without them) and 2 (the `pb24` API page,
because an undocumented 1,479-line module is how modules stay undocumented), and
to defer a `docs/model/surface.md` narrative page to its own prompt. Your call.

>A. Let us do all of the work on this branch with 1 PR into `main`

**PQ4.** Do you want the merged state *validated numerically* before the PR —
i.e. re-run `run_validation.py` and `run_pb24_validation.py` and diff the
committed tables in `design/validation/` and `design/validation_pb24/` — or is a
green `pytest` enough for the PR, with the regeneration left to a follow-up?
Re-running needs `$OS_COLOR/SD/v5` (PB24) and the L23 netCDFs; confirm both are
on this laptop.

>A. yes, we will re-run.  If the files aren't on this laptop, I will run on 
the `profx` worksation when the time comes.

**PQ5.** The inelastic/CDOM work has never been exercised off-nadir or on the
12-band OLCI grid, and M5 has never seen a Raman or fluorescence term. Nothing
about the merge forces those to meet — but they now live in one `hybrid.forward`.
Do you want a cross-check (even a smoke test: inelastic on, `theta_v = 40°`)
added as part of this PR, or is that M6's problem?

>A. Is there actually an M6 waiting for us?  As you will note, I have already merged
in the inelastic processes.  Our goal now is to provide a complete forward model
with all of the processes.

##### Round 2 (2026-09-19)

**Where your answers leave us.** A1/A3 settle the shape: one branch, one PR into
`main`, carrying the merge *and* the docs debt. A2 keeps the "take HEAD" prose
resolutions. A4 says re-run — and I can report that nothing needs `profx`:

| | |
|---|---|
| `$OS_COLOR` | `/Users/xavier/Projects/Oceanography/data/Color/` |
| PB24 | `SD/v5/` — **10,001 files**, `SD_OLCI_no_R_0001.nc` onward ✅ |
| L23 elastic (X=1) | `Loisel2023/Hydrolight1{00,30,60}.nc` ✅ |
| L23 inelastic (X=2/X=4) | `Hydrolight2{00,30,60}.nc`, `Hydrolight4{00,30,60}.nc` ✅ |
| `ocpy` resolution | `loisel23.l23_path` resolves under `ocean14` and all six inelastic files test `True` ✅ |

So both validation drivers and the full test suite can run here. (`$OS_COLOR_DATA`,
which the Setup prompts referenced, is unset — `ocpy` finds L23 through `$OS_COLOR`,
so nothing is broken, but the prompt doc's path is stale.)

**A5 is the one that changes the plan, so I want to answer it before asking
anything else.**

*Yes — there is an M6, and it is already written.* `claude_prompts/RT/rt_elastic_coding_prompt_7.md`
exists **only on this branch** (main has prompts 1–6), so this PR is what would
carry it to `main`. It is a scoping prompt: its task 0 is "decide what M6 is",
and it carries two questions to you — **Q18** (which route, and are HydroLight
runs on the table) and **Q19** (does the prototype's headline claim need
restating) — posed 2026-08-13 and never answered, because the work moved to the
`RT` line the following week.

And the reason M6 needs scoping is the part I should have led with in the
Forensics report. I read `design/m5_report.md` in full for this round:

> **M5's gate FAILED, on both splits, at every seed. No PB24 weights exist.
> `load_default()` is still the L23 nadir model.**

The cause is not the network. `Ψ_KLu(ψ)` — ZTT's quartic, fitted for ψ ≳ 134° —
crosses zero at **110.4°**, so in PB24's own sanctioned window 42% of geometries
extrapolate, 16% are sign-flipped, and **22.3% of ZTT's predicted `rrs` are zero
or negative**. The hybrid's correction is *bounded and relative* (`|δ| ≤ 0.5`), so
`1 + δ` can never reach a negative backbone. The oracle correction — chosen with
the truth in hand — scores 5324% where the trained model scores 5484%: the model
is within 3% of the best its functional form permits. L23's minimum ψ is 139.7°,
which is why nadir-only Week-1 could not have found this and why none of the
0.30% numbers are affected.

So the merge brings to `main` a large, gated **infrastructure** delivery (PB24
loader and splits, the geometry-aware surface transfer — 7.2× better than Lee's
nadir constants at θv = 60°, `O25Table` over the full grid, per-model `Envelope`,
`rrms` masking, 279 → 416 tests) **plus a negative result**, not a BRDF-capable
forward model.

**Which is why I want to check A5's goal against it.** "A complete forward model
with all of the processes" is true today along the *process* axis — elastic +
Raman + chlorophyll fluorescence + CDOM fluorescence all compose in one
`hybrid.forward` — but every one of those processes is **nadir-only**. I checked:
`inelastic.py`, `inelastic_corr.py`, `cdom_fl.py` and `ed.py` contain no reference
to `theta_v` or `dphi` at all; they interpolate over three solar-zenith anchors and
default to the L23 grid. The BRDF axis is the one M5 went after and did not get,
and it is blocked on the backbone, not on effort.

**One concrete contradiction the merge creates.** `docs/model/forward.md:173–180`
tells readers that `on_out_of_domain="ztt"` makes the model "degrade to the
analytic backbone exactly where the emulator was measured to be unreliable", and
the elastic report's §5 item 5 says "the domain check correctly flags any
off-nadir view". Both were written 2026-08-15, on the `RT` line, two days *after*
M5 measured that the thing it degrades *to* returns non-physical `rrs` on 22.3%
of off-nadir geometries. Neither statement was wrong when written and neither
author could have known. After this merge they sit in one tree with the
measurement that contradicts them, and the published site is the one place a
reader outside the project would look.

---

**PQ6 — Q18 and Q19 have been open since 2026-08-13. Answer them in this PR's
discussion, or after it lands?**

My recommendation is **now, at least provisionally**, because Q18's Route C
("restrict the claim and ship what works") is largely a *documentation* action —
and this PR is already opening `docs/`. Doing C here costs almost nothing extra
and makes the merged state honestly describable; B (change the correction's form
from bounded-relative to additive or unbounded) is a few days of experiment and
is a separate PR; A (replace or repair the backbone) is the real answer and is
not ours alone to make. If you would rather keep the PR purely mechanical, say so
and I will carry Q18/Q19 forward untouched.

>A. I have answered Q18 and Q19 in the coding prompt.  Proceed accordingly

**PQ7 — Does this PR correct the `on_out_of_domain="ztt"` guidance?**

Smallest honest fix: a note on `docs/model/forward.md` and in
`docs/using/limitations.md` saying the `"ztt"` policy is a safe degradation
*within the nadir domain the site documents*, and that off-nadir the backbone it
falls back to is measured non-physical on 22.3% of PB24 — citing
`design/m5_report.md` §2.1. Alternative, if you would rather the code carry it
than the prose: make the policy refuse (or warn separately) when
`backbone_is_usable` is False, which M5 already built. **Prose, code, or both?**
I lean prose-only in this PR — changing a shipped policy's behaviour inside a
merge PR is how a merge PR stops being reviewable.

>A. Prose only for now

**PQ8 — Which reading of "complete forward model with all of the processes" is
the goal we are working to?**

(a) **All processes, nadir, L23-like water** — essentially what `main` has today,
plus M5's infrastructure and an honest envelope. Reachable in this PR.
(b) **All processes across the BRDF** — needs the backbone question settled
(Q18 Route A or B) before any of the inelastic heads can be asked for an
off-nadir geometry they have never seen.

I have been assuming the goal is (b) and the PR delivers (a) as a way-station.
If (a) *is* the goal, then M6 is much smaller than prompt 7 assumes and Q18
collapses to Route C. Worth being explicit, because it decides whether M6 is a
documentation milestone or a physics one.

>A. Yes, let us make (a) the goal for now.

**PQ9 — Are HydroLight runs on the table?** (This is Q18's blocking half, and it
now has a second sponsor.)

Robert Frouin's comment in this document says it directly: *"use a full
radiative-transfer solver (most naturally HydroLight) as the reference forward
model, with particle phase-function parameters explicitly varied"*, with a fast
differentiable emulator on top — which is the architecture we have. M5 arrived at
the same requirement from the other end: Route A needs reference data that
tabulates an **asymptotic** K (PB24 tabulates seven K's and all seven vary ~1.4×
with solar zenith, so they are surface K's — µ∞ cannot be refit from it), and it
needs the VSF *family* varied rather than only the Fournier–Forand parameter,
which is the one headline gap M5 never reached. Two independent routes to the
same ask is the strongest argument this project has for commissioning them. **Is
that possible — and if so, on what timescale?**

>A. Yes, it is possible and will be the next major development task.

**PQ10 — Two known contaminations ship with this PR unless we fix them.**
`design/m5_report.md` §6 records that the shipped `robust/rt/files/surface_pb24.npz`
was fitted on a 400-realisation split while the benchmark split 200 and the gate
800 — and `make_splits` permutes whatever set it is given, so `SPLIT_SEED = 23`
does not name a partition. 31 of the benchmark's 40 held-out realisations were in
the transfer's training set; residual ~1.8% median, and it biases *toward* O25, so
every hybrid FAIL is conservative. The fix is to refit the transfer on each
consumer's own train mask. **Fix in this PR, or merge it as recorded debt with an
issue?** I lean fix-it-here — that `.npz` is a shipped artefact of the package,
and "known-contaminated" is a bad property for a file to acquire by being merged
to `main`.

>A. Fix it here

**PQ11 — What exactly gets re-run, and do the regenerated tables get committed?**

Proposed, in order, all under `ocean14`:
1. `pytest robust/tests` — expect ~416 + main's inelastic/CDOM tests, now with
   `$OS_COLOR` present so the 44 previously-skipped ones run too.
2. `python design/py/run_validation.py` (elastic) and `--inelastic` → diff
   `design/validation/`.
3. `python design/py/run_pb24_validation.py` → diff `design/validation_pb24/`.
4. `cd docs && make html` (that target already passes `-W --keep-going`).

Two flags on this. **`train_emulator_pb24.py` exits 1 by design** when the gate
fails — that is M5's correct behaviour, not a regression, and whoever reviews the
PR needs to know before they see a red exit code. And `design/py/fit_surface.py`
is only re-run if PQ10 says fix. **Do we commit regenerated tables even where the
numbers are unchanged**, or only where they move? I lean commit-only-if-moved, so
the PR diff stays readable and any movement is visible rather than buried in
timestamp churn.

>A. Commit-only-if-moved

## Comments

### Robert's

Most analytical models other than ZTT (Twardowski and Tonizzo, 2018) do not treat the particle phase function, or parameters describing its shape, as explicit and independently adjustable inputs. Instead, phase-function effects are incorporated implicitly into coefficients or LUTs derived from RT simulations using prescribed phase functions. Consequently, the models cannot explicitly represent independent variability in particle phase-function shape. Because the phase function is a primary determinant of the angular distribution of water-leaving radiance, this limitation can introduce geometry-dependent forward-model errors and associated biases in retrieved IOPs.  ZTT largely addresses this limitation by introducing the backward VSF and related phase-function parameters explicitly into the analytical forward model.

My recommendation would be to use a full radiative-transfer solver (most naturally HydroLight) as the reference forward model, with particle phase-function parameters explicitly varied. For computational retrieval, construct a fast differentiable emulator of that solver. You may use ZTT as the principal analytical benchmark and possibly as the physical backbone of a hybrid emulator. Retain O25 as a BRDF/retrieval comparison model, not as the final physical reference.

A hybrid architecture would be particularly attractive:

Rrs(model) = Rrs(ZTT) + deltaRrs(simulator)

where ZTT supplies the physically interpretable backbone and a small emulator learns the remaining multiple-scattering and phase-function effects. This preserves physical scaling and geometry while avoiding the unrestricted behavior of a wholly black-box model.

## Q&A

### Report

#### Round 1 (2026-08-15)

The prototype is complete: M0–M4 delivered `robust/rt/` (ZTT backbone in JAX,
residual MLP emulator, hybrid `forward()`, O25/Gordon comparisons, the validation
protocol), 279 tests + CI, five executed notebooks, and the headline table —
hybrid **0.30%** rRMS on held-out scenes vs O25-refit **0.69%**, ZTT 5.93%, Gordon
7.21%; unseen-60° reported (seed-dependent 4.7–12.2%) rather than gated. Before
writing `report/rt_elastic_model.md`, five questions:

**RQ1 — Audience & register.** Who is this report for, and at what length?
  (a) Science-facing progress report for you + advisors (e.g. share with Robert
      Frouin): results-led, the architecture and validation story, light on code
      mechanics; ~4–6 pages. [Claude's lean]
  (b) Project-record capstone: the full arc (synthesis → design → coding plan →
      M0–M4) with per-milestone narrative, for future contributors; ~10+ pages.
  (c) Seed of a publication: methods/results structured like a paper draft.
>A. For the moment, it is for the Team.  But, it will then evolve into a "publication".  I will not submit to a refereed journal, but I will share it with the world.  So (a) and (c)

**RQ2 — Relationship to the existing docs, and a name collision.** The effort
already has `context/RT/rt_elastic_model.md` (literature synthesis),
`design/rt_elastic_model.md` (design), `design/rt_elastic_model_coding_plan.md`,
`design/prototype_summary.md` (the reviewer-facing M0–M4 summary), and
`design/rt_elastic_implementation.md` (the current-state record). The new file
would be the **third** named `rt_elastic_model.md`. Should the report:
  (a) Be the top-level synthesis of the whole effort — what we set out to do, what
      was built, what the numbers say, what's next — linking to those docs rather
      than duplicating them; keep the requested name. [lean]
  (b) Absorb/supersede `design/prototype_summary.md` as the single summary (and we
      retire that file).
  (c) Take a distinct name (e.g. `report/rt_elastic_prototype.md`) to avoid the
      collision.
>A. Let's use the name `report/report_rt_elastic_model.md`.

**RQ3 — Figures.** The Setup-report prompt required Python-generated figures saved
beside the doc; this prompt doesn't mention them. Should I:
  (a) Generate report-specific figures with a script saved in `report/` (per-λ
      five-model ladder, the unseen-60° seed-spread plot, and a hybrid architecture
      /dataflow schematic), regenerated from `design/validation/*.csv`. [lean]
  (b) Reuse the committed `design/validation/*.png` and notebook figures by
      reference only.
  (c) No figures.

>A. (a), but draw from (b)

**RQ4 — Headline framing & a roadmap section.** `prototype_summary.md` leads
caveats-forward ("what it may not claim": 2.3× over O25 not 24×, O25's 0.69% is
its best case, geometry extrapolation unresolved, phase-function generalisation
untested on L23's narrow B_p range). Should the report:
  (a) Keep that caveats-forward stance and close with a recommended-priorities
      section for M5 (HydroLight runs with the phase function varied, PB24
      geometry, Eq. (8) µ∞ coefficients, promoting P_bb into `PhaseParams`). [lean]
  (b) Conventional results-then-limitations structure, no forward-looking
      recommendations (strictly a summary of work done).

>A. (a)

**RQ5 — Open threads.** A few items are formally unfinished: the ZTT Eq. (8) µ∞
coefficients (TT2017 stand-in; you emailed the authors — any reply?), the M3
task-5 hand-off edit, and prompt doc 6 (M5) is deliberately a placeholder. Treat
the prototype as complete per the amended M4 gate and list these as open items in
the report [lean], or close any of them first?

>A.  Write the report now, but have a Section on open items. I'll then deal with them.

### Design

#### Round 1 (2026-07-31)

We have the synthesis + roadmap (`context/RT/rt_elastic_model.md`) and Robert's steer:
a full-RT (HydroLight) **reference** with the phase function varied → a fast
**differentiable emulator** for retrieval → a **hybrid** `Rrs = Rrs(ZTT) + ΔRrs(emulator)`,
with O25/PR05 as comparisons; data L23-first then HydroLight/PB24; elastic-only;
geometry + phase-function shape first-class. The design doc `design/rt_elastic_model.md`
should turn that into a buildable plan. Six questions to scope it:

**DQ1 — Design-doc horizon & code home.** Which should the doc be?
  (a) Near-term buildable plan (~6–12 mo): stand up the reference-data generator, ZTT,
      a first emulator, and the hybrid on L23/PB24 — at module/API detail.
  (b) Full multi-year architecture through operational PACE retrieval + priors —
      conceptual + milestones.
  (c) Both, layered: a concrete near-term build section inside a longer architecture
      map. [Claude's lean]
  Also: should the forward model live as a new `robust/rt/` subpackage exposing a
  single `forward(iops, phase_params, geometry) → Rrs` API (my proposal)?

DQ1: (a) near-term buildable plan.  I anticipate it only taking one week to get our first high quality prototype.

**DQ2 — RT reference solver & access.** Robert says HydroLight "most naturally." Do we
have a HydroLight license/machine to run batches, or should the design target an
**open** solver as the reference of record (OSOAA, a polarized Monte Carlo, PlanarRad)?
  (a) HydroLight is available — make it the reference.
  (b) Design to a **solver-agnostic interface** with an open MC/OSOAA as the
      reproducible default, HydroLight as a cross-check where available. [lean]
  (c) Other / you'll provide access details.

DQ2: (a); and we have in hand a significant set of outputs from Loisel+2023

**DQ3 — Autodiff stack.** For the differentiable emulator + hybrid (and eventual
gradient-based inversion): **JAX** or **PyTorch**? And do we implement ZTT analytically
in the same framework so `Rrs = Rrs(ZTT) + ΔRrs(emulator)` is differentiable
end-to-end? (Lean: **JAX** — natural for differentiable physics, analytic ZTT, and
`vmap` over geometry; happy to use PyTorch if the team is PyTorch-native.)

DQ3:  JAX; and yes, we should implement ZTT analytically in the same framework so `Rrs = Rrs(ZTT) + ΔRrs(emulator)` is differentiable end-to-end.

**DQ4 — Phase-function parameterization (the explicit axis).** How should the particle
phase function enter as an explicit, adjustable input?
  (a) Fournier–Forand via (backscatter ratio, Junge slope) — matches L23/PB24.
  (b) ZTT-style **backward-VSF parameters** (β(π)/bb + a backward-shape term) as the
      primary axis, with FF (a) as the training-data generator so it ties to L23/PB24.
      [lean]
  (c) Henyey–Greenstein / two-term-HG mixtures.
  (d) A low-dimensional basis learned from measured VSFs (Petzold + field VSFs).

DQ4:  I am not sure about this one.  Let's start with a simple approach for now.  Your pick

**DQ5 — Inversion coupling / scope of this doc.** Is `design/rt_elastic_model.md`:
  (a) **Forward-model only** (reference + emulator + hybrid + validation); inversion
      deferred to a companion design doc. 
  (b) Forward **+ inversion**, and if so — reuse BING's MCMC, or move to
      **gradient-based / amortized simulation-based inference (SBI)** that exploits the
      differentiable forward model and the project's learned priors?
  (Lean: forward model in depth **+ a clear inversion interface and recommended path**
  — differentiable → gradient/SBI + priors — with the full inversion design deferred.)

DQ5:  Forward-model only, but differentiable.

**DQ6 — Acceptance criteria / validation protocol.** What defines "done" for the
forward model? Candidates to bake in: forward rRMS vs the reference (per λ, geometry,
phase-function shape); **held-out generalization** across geometry & phase function;
and the decisive one — **component retrieval MAPE** (a_ph, a_dg, bb_p) vs L23/PB24
truth. Hard target numbers, or **protocol-only** (report spreads, no blind target —
consistent with the "unbiased view" stance from the biomass report)? (Lean:
protocol-only with fixed held-out splits.)

DQ6:  This is difficult to answer.  We want something that is fast and accurate and differentiable.

### Coding

#### Round 1 (2026-07-31)

Questions to shape `design/rt_elastic_model_coding_plan.md` — the staged implementation
plan with milestones. (Locked by the design doc: JAX/Flax/Optax; `robust/rt/` with a
`forward(iops, phase_params, geometry)→Rrs` API; ZTT analytic backbone + residual
emulator hybrid; L23 elastic (3 solar zeniths) as the in-hand reference; `B_p` phase
param; forward-model-only, differentiable; ~1-week first prototype.)

**CQ1 — Execution model (most important).** Should Claude **implement** the prototype
now — scaffolding `robust/rt/` and building the stages on a branch, with JXP running all
git (commits/PRs to review), per CLAUDE.md — or is this coding plan a **spec/handoff**
for you (or others) to implement? (Lean: Claude implements on a branch; each milestone a
reviewable commit/PR; JXP does the git.)

CQ1: Claude will implement the prototype now.  I will run git and review the commits.

**CQ2 — Milestone structure & cadence.** How to stage it?
  (a) **Milestone-gated** (M1 scaffold+data → M2 ZTT-in-JAX → M3 emulator+hybrid →
      M4 validation → M5 HydroLight/BRDF+phase-function), each with an explicit
      acceptance gate; fine-grained (task-level) for the Week-1 milestones, coarser
      after. [lean]
  (b) A flat ordered task list.
  (c) Time-boxed (day-by-day for week 1).

CQ2: (a)

**CQ3 — ZTT equations source (a real dependency).** ZTT (Twardowski & Tonizzo 2018) is
the analytic backbone, but the paper is **not** in `context/RT/`. To implement
`Rrs_ZTT` in JAX we need its equations. Should I:
  (a) Source the paper / equations myself (web) and transcribe them;
  (b) You will provide the ZTT equations / a PDF;
  (c) Start with an **analytic placeholder backbone** (Gordon/O25 in JAX) so the
      pipeline is end-to-end first, then swap in true ZTT once its equations are in
      hand. [lean: (c) in parallel with (a)]

CQ3: I have provided the paper in `context/RT/` and named it twardowski2018.pdf 

**CQ4 — Testing, tooling & conventions.** Gate each milestone with **pytest** (unit +
the gradient-correctness check)? Formatting/linting (ruff/black), type hints
(jaxtyping)? Should I follow any **existing retrieve-or-bust / BING conventions** (test
layout, CI), or set them up fresh here? (Lean: pytest gates + ruff + light jaxtyping;
match BING's layout if present.)

CQ4: Yes, gate each milestone with pytest.  I will follow the BING conventions.

**CQ5 — Environment & compute.** `ocean14` almost certainly lacks `jax/flax/optax`.
Add them to `ocean14`, or create a dedicated env (e.g. `rt-jax`)? And is the Week-1
prototype **CPU-only** (fine for L23-scale), or is a **GPU** available if we want it?
(Lean: CPU-only JAX added to a dedicated env to avoid disturbing `ocean14`.)

CQ5: CPU-only for now.  Feel free to update the requirements file as needed and add packages to `ocean14` 

**CQ6 — Validation split policy (the acceptance gate).** For "held-out generalization,"
hold out (a) a **seeded random 20% of scenes**, and (b) **one solar-zenith angle** (e.g.
train 0°/30°, test 60°) to probe geometry generalization — with the M4 gate = beats
standard Gordon on both splits and passes the gradient check. Agree, or a different
split?

CQ6: Agree

### Setup

Claude has reviewed JXP's Gordon work in the BING repo (`prompts/gordon.md`
logs, `dev/Gordon/calc_gordon.py`, `dev/Gordon/plot_gordon.py`, the four
`gordon_coefficients*.csv` files, and `bing/rt/rrs.py`). Before writing
`context/rt_bing_summary.md`, the following questions:

**Q1 — Report purpose.** What should `rt_bing_summary.md` primarily be?
  (a) Orienting synthesis — a technical summary of the BING Gordon findings +
      open threads to orient the retrieve-or-bust RT work (narrative + result
      tables, no new analysis).
  (b) Decision doc — same synthesis, ending in a concrete recommendation on
      which recipe RT should adopt as its baseline.
  (c) Synthesis + roadmap — synthesis plus a prioritized list of the next RT
      experiments (validation, retrieval impact, cross-dataset), framed as this
      project's starting plan.

A1: (c)

**Q2 — Baseline recipe.** Which recipe should the report treat as the current
canonical / recommended baseline?
  (a) Two-stage 4-param (G0, Gb fit to residuals vs bbp(700), G1/G2 held fixed)
      — what currently ships in `gordon_coefficients_with_G0_Gb.csv`;
      interpretable but ~2.8% rRMS at red.
  (b) Joint 4-param (G0 + G1·u + G2·u² + Gb·bbp fit simultaneously) — best
      forward-model rRMS everywhere (0.3% red, 0.76% at 550 nm), but G1/G2
      absorb bbp structure.
  (c) Report both, present the trade-off, defer the choice to the pending
      retrieval-impact study.

A2: The Park & Ruddick (2005) approach.

**Q3 — Open threads.** How much emphasis on the unclosed validation gaps
(held-out set, IOP-retrieval impact, cross-dataset stability of G0/Gb)?
  (a) Foreground them as the central to-do for retrieve-or-bust — the
      forward-model rRMS wins are an optimistic in-sample ceiling until
      validated.
  (b) Note briefly as caveats; keep the focus on documenting what was done.

A3: Ignore those for now.

**Q4 — Plumbing depth.** How deep should the report go on the package wiring?
  (a) Findings-focused — science/results; mention flags and CSVs only enough to
      reproduce, not the file-by-file plumbing.
  (b) Include a plumbing map — document how it's wired (`variable_Gordon_G0` /
      `variable_Gordon_bbp` flags, the four CSVs, and the
      `rrs.py`/`anw.py`/`l23.py`/`evaluate.py` code paths) so RT can build on
      the code directly.

A4:  The report should not worry about the BING plumbing.

**Q5 — Anything else.** Is there context I'm missing — e.g. the deeper JXP
"deep-dive" finding on wavelength dependence and bb-sensitivity of the G_i that
rt_prompts.md alludes to, or a direction you already want RT to take beyond the
Gordon approximation (full RT, learned forward model, etc.)?

A5: I have just put a set of PDFs in the `context/RT/` folder.  Please read them as they are relevant to the report

### Setup — Round 2 (2026-07-19)

Claude has now read the five `context/RT/` PDFs (Gordon 1988, Park & Ruddick 2005,
Tan 2018, Pitarch 2025, Hansen 1971) and re-reviewed the BING Gordon work now that
the BING repo is local (`Oceanography/python/bing`). Note: `hansen1971.pdf` is a
scanned image with no extractable text layer (pypdf returns only the download
watermark) — see Q9.

Reading A1–A5 against the papers, one substantive tension surfaced, plus three
scoping questions.

**Q6 — Report identity / name.** A2 makes Park & Ruddick (2005) the baseline, A4
drops the BING plumbing, and A5 adds the RT literature. The deliverable is
therefore no longer really a "BING summary." Which should it be?
  (a) Keep the name `context/rt_bing_summary.md`; the BING findings are the spine,
      literature is supporting context.
  (b) Rename to a broader synthesis (e.g. `context/rt_forward_model.md`): the
      Gordon → PR05 → Lee2011/Pitarch-O25 lineage + the BING G0/Gb finding +
      roadmap, with PR05 named as the baseline.
  (c) Two files: a short BING-findings note + a separate literature/roadmap doc.

Q6:  (b)

**Q7 — Baseline forward model (the substantive one).** You chose PR05 (A2). But the
very papers you added argue PR05 is the *weakest* of the modern schemes:
  - Tan (2018): PR05 reconstructs Rrs well, but its retrieved parameters (esp. Chl)
    do not represent true values, and PR05-reconstructed Rrs fed to QAA gives badly
    biased IOPs — a caution given our component-separation goal.
  - Pitarch (2025) / D'Alimonte (2025): rank Lee-2011 (L11) > Morel-2002 > PR05;
    PR05 blows up the LUT dimension per γb and provides *no mechanism to retrieve
    γb = bbp/bb from Rrs*.
  - Pitarch's O25 replaces PR05's 4th-order polynomial in ωb (γb-indexed
    coefficients) with a bivariate quadratic that splits ωb into a water part
    ωbw = bbw/(a+bb) and a particle part ωbp = bbp/(a+bb), with coefficients that
    depend on geometry ONLY. This "Rrs is not univocal in u" insight is exactly what
    your BING G0/Gb terms captured empirically — and because retrieve-or-bust
    already separates bb_w (known) from bb_p, the O25 split maps onto our model for
    free (no γb iteration needed).
  Which does RT adopt as its forward-model baseline?
  (a) PR05 as written (4th-order in ωb, γb-indexed coefficients).
  (b) The L11 / Pitarch-O25 bivariate (ωbw, ωbp) form — modern successor,
      structurally aligned with BING's G0/Gb.
  (c) Report PR05 as the baseline per A2, but explicitly recommend O25/L11 as the
      evolution to adopt (reconcile A2 with the newer papers). [Claude's lean]

Q7: (c) and also emphasize that we are likely to construct our own new approach

**Q8 — BRDF / geometry scope.** PR05 and O25 are fundamentally bidirectional
(θs, θv, Δφ); the BING Gordon fits were at a fixed geometry. For the RT roadmap:
  (a) Fixed nominal geometry first (e.g. nadir view); defer full BRDF and treat the
      geometry-indexed LUTs as future work.
  (b) Build variable geometry / BRDF in from the start, making the geometry-indexed
      coefficients first-class.

Q8: (b)

**Q9 — Hansen (1971).** The PDF is a scan with no text layer, so I could not read
its content (only the "planetary atmospheres / doubling method" role you noted).
Should I (a) source a text/OCR'd copy so I can fold in its specifics, or (b) treat
it only as background (the multiple-scattering / doubling-method lineage behind the
u-polynomial) and proceed?

Q9: (b)

### Setup — Round 3 (2026-07-19)

Q6–Q9 fix the report scope: a broad synthesis named `context/rt_forward_model.md`;
PR05 as the named baseline with O25/L11 recommended as the evolution and our own new
approach flagged; BRDF/variable geometry first-class; Hansen as background. Before
writing, three questions remain that materially shape the roadmap.

**Q10 — "Our own new approach" (from Q7).** You said retrieve-or-bust is likely to
construct its own forward model. Which flavor should the roadmap foreground?
  (a) Extend the analytic / physically-structured line — more terms or a better
      parameterization in the BING-G0/Gb ↔ O25-(ωbw, ωbp) family (interpretable,
      few parameters, keeps the inversion analytic).
  (b) Learned forward model — a neural emulator of RT mapping (IOPs, geometry) → Rrs,
      replacing the polynomial (max accuracy, differentiable for retrieval, but a
      black box).
  (c) Hybrid — an analytic backbone (O25-style water/particle split) plus a learned
      residual / coefficient network (keeps physics, learns what the polynomial
      misses). [Claude's lean]
  (d) Undecided — lay all three out as options in the roadmap and choose later.

Q10: (d)

**Q11 — Elastic-only scope.** The doc is titled *Elastic* RT, and Gordon/PR05/O25 are
all elastic (no Raman / fluorescence); BING already carries `raman.py` and
`chl_fl.py` separately. Should the report:
  (a) Stay strictly elastic — treat inelastic (Raman, CDOM/chl fluorescence) as out
      of scope, noted only as a boundary.
  (b) Cover elastic as the core but include an explicit inelastic-extension section
      in the roadmap.

Q11: (a).  Indeed, let's name it `context/rt_elastic_model.md` 

**Q12 — RT data / truth for a BRDF-capable forward model.** Q8(b) wants variable
geometry first-class, but the BING fits used L23 (Loisel 2023), which is effectively
a single nominal geometry — a multi-angular truth set is needed. Which source should
the roadmap build on?
  (a) Pitarch & Brando 2025 (PB24) — the open multi-angular synthetic set O25 was
      built on (5000 IOPs × 1300 geometries, Fournier–Forand); reuse it.
  (b) Generate our own RT runs (Hydrolight or an open RT code) over a chosen
      IOP × geometry grid — full control, more work.
  (c) Start on L23 at its fixed geometry to reproduce/extend the BING result, then
      add geometry via PB24 or new runs as a second phase. [Claude's lean]

Q12: (c)

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-07-16 (Review BING Gordon work; open Q&A/Setup before writing rt_bing_summary.md)

Executed the 1st Setup prompt: reviewed JXP's wavelength-dependent Gordon work
in the BING repo and posed setup questions (Q1–Q5 above) rather than writing the
report yet.

**Sources reviewed.**
- `bing/prompts/gordon.md` — the driving prompt doc + full Logs (2026-05-31 →
  2026-06-03), which is the authoritative record of the development arc.
- `bing/dev/Gordon/calc_gordon.py` (1726 lines) — all fit recipes.
- `bing/dev/Gordon/plot_gordon.py` — assessment figures (re-exported).
- `bing/dev/Gordon/gordon_coefficients*.csv` (4 files: 2-param, +G0, +Gb, +G0+Gb)
  and the deployed copies under `bing/bing/data/RT/`.
- `bing/bing/rt/rrs.py` — `wave_dependent_gordon*`, `calc_Rrs`,
  `calc_elastic_Rrs` (how the coefficients are consumed).

**What I learned (to be written up in rt_bing_summary.md after Q&A).**
- The physical crux: standard Gordon `rrs = G1·u + G2·u²` makes `rrs` a function
  of `u = bb/(a+bb)` alone, but the Hydrolight (L23) truth depends on `(a, bb)`
  separately. The wavelength-dependent coefficients + the extra G0 (constant
  offset) and Gb (bbp slope) terms capture that residual (a,bb) structure —
  matching the rt_prompts.md note that the G_i are λ-dependent and bb-sensitive.
- Two original bugs fixed: (1) `B=1.17` vs `1.7` Rrs↔rrs convention mismatch;
  (2) constant-σ (unweighted) fit letting G2 run away at red λ. Relative
  weighting + bounds + Tikhonov smoothness on G1(λ),G2(λ) made variable-Gordon
  beat standard everywhere.
- Recipe ladder, rRMS vs Hydrolight (3320 L23 scenes): standard → 2-param smooth
  → +G0 → +Gb → G0+Gb. G0 wins red (700 nm 9.05%→0.35%), Gb wins blue (400 nm
  2.54%→1.84%), joint 4-param wins/ties everywhere (550 nm 0.76%). All plumbed
  into the package (opt-in flags `variable_Gordon_G0`, `variable_Gordon_bbp`);
  80 tests pass.
- Key tension: the joint 4-param is best but lets G1/G2 absorb bbp structure;
  JXP then adopted a two-stage fit (Stage 1: G1,G2; Stage 2: G0,Gb vs bbp(700))
  for interpretability at the cost of red-λ rRMS (~2.8%). The two-stage recipe is
  what currently ships in `gordon_coefficients_with_G0_Gb.csv`.
- Recurring un-closed recommendations across every log entry: no held-out /
  out-of-sample validation, no retrieval-impact (IOP MAPE) study, no
  cross-dataset (PACE/MOBY) test of whether G0/Gb are structural or L23-specific.

Next: await answers to Q1–Q5, then write `context/rt_bing_summary.md`.

### 2026-07-19 (Read the five context/RT PDFs; BING repo now local; posed Round 2 Q&A before writing the report)

Executed the 2nd Setup prompt: reviewed answers A1–A5, read the five RT papers, and
posed a second round of questions (Q6–Q9 in Q&A above) driven by a tension between
A2 (PR05 baseline) and the newer papers JXP added.

**Sources read.**
- Extracted text from all five PDFs with `pypdf` (ocean14) — `pdftoppm`/poppler is
  not installed, so the Read tool's page-render path is unavailable; text extraction
  was sufficient for four of the five.
- `park2005.pdf` (PR05) — read in full. QSSA-derived Rrs model; 4th-order polynomial
  Rrs = Σ_{i=1..4} g_i(θo,θ,Δφ,γb)·ωb^i with ωb = bb/(a+bb) (their "backscattering
  albedo"); coefficients tabulated over Sun/sensor angles and γb = bbp/bb (particle
  fraction of backscatter, 0.2–1). Hydrolight 4.2, Fournier–Forand phase functions,
  case-1+case-2 IOPs. Model uncertainty ~2% (rms ~1%), dominated by residual
  phase-function variability. γb must be estimated (iteratively) before inversion;
  ~2% Rrs accuracy needs γb to ~0.05 (≈20–30% bbp at low γb, worse at high γb).
- `gordon1988b.pdf` — read (agent). Canonical R/Q = l1·u + l2·u², l1=0.0949,
  l2=0.0794; θo>20°, u≤~0.2; the l_i are treated wavelength-independent (λ-dependence
  parked in a, bb, and Q≈4–5). R→Rrs interface factors: (1−ρ)/m² ≈ 0.54, (1−rR),
  r=0.48. Max rms error of the scheme ~±20%; i>1 term matters at high radiance.
- `tan2018.pdf` — read in full. Evaluates PR05 in POLYMER vs Hydrolight (IOCCG L23)
  + AERONET-OC. PR05 reconstructs Rrs and band ratios well (RMS <15%), BUT retrieved
  Chl is biased (a fit parameter, not physical) and PR05-reconstructed Rrs → QAA
  gives badly biased IOPs. Recommendation: use reconstructed reflectance, not
  retrieved parameters. Two-param PR05 preferred over three-param (robustness).
- `pitarch2025.pdf` (O25) — read (agent). BRDF-normalization successor in the
  Gordon→Lee-QAA→L11 line. Bivariate quadratic Rrs = (Gw0+Gw1·ωbw)·ωbw +
  (Gp0+Gp1·ωbp)·ωbp with ωbw=bbw/(a+bb), ωbp=bbp/(a+bb); the four G-coeffs depend on
  geometry ONLY. Calibrated on PB24 (Fournier–Forand, 5000 IOPs × 1300 geometries).
  Ranks L11 > M02 > PR05. Physical crux: Rrs is *not* univocal in ωb — two branches
  split by ηb = bbw/bb, because at single scattering Rrs ∝ β(π)/bb (0.23 sr⁻¹ water
  vs 0.12–0.16 particles), so water vs particle backscatter must be separated. Code
  open-source (github.com/jaipipor/O25); operational in EUMETSAT OLCI Collection 4.
- `hansen1971.pdf` — **could not read**: scanned image, no text layer (pypdf returns
  only the "Unauthenticated | Downloaded …" watermark). Role per JXP: multiple-
  scattering RT / doubling method in planetary atmospheres. Flagged as Q9.
- Re-confirmed the BING Gordon artifacts now that the repo is local:
  `bing/dev/Gordon/{calc,plot}_gordon.py`, four `gordon_coefficients*.csv`, deployed
  copies under `bing/bing/data/RT/`, and `bing/bing/rt/rrs.py`.

**Key realization to carry into the report.** The central physical point is the same
across the BING G0/Gb finding and Pitarch's O25: **Rrs is not a function of
u = bb/(a+bb) alone** — the residual (a, bb) / water-vs-particle structure must be
represented. BING captured this with extra G0 (offset) + Gb (bbp slope) terms; O25
captures it structurally by splitting u into ωbw (water) and ωbp (particle) with
geometry-only coefficients. Because retrieve-or-bust separates bb_w (known) from
bb_p, the O25 split is essentially free for us — which is why Q7 asks whether to
keep PR05 (A2) as the literal baseline or evolve to the L11/O25 bivariate form.

Next: await answers to Q6–Q9, then write the report (name per Q6).

### 2026-07-19 (Reviewed Q6–Q9; held off on the report to pose Round 3 on the roadmap)

Executed the 3rd Setup prompt: reviewed the Round-2 answers (Q6=b broad synthesis
`rt_forward_model.md`; Q7=c PR05 baseline + O25/L11 recommendation + our-own-approach;
Q8=b BRDF first-class; Q9=b Hansen as background). Grounded the BING numbers directly
from the now-local repo — the four `gordon_coefficients*.csv` headers/recipes and the
per-wavelength rRMS ladder (standard → smooth 2-param → +G0 → +Gb → joint 4-param) and
two-stage shipped fit — so the synthesis tables will be exact.

At the user's request, paused before writing to pose Round 3 (Q10–Q12 above): the
direction of "our own new approach", elastic-only vs inelastic-extension scope, and
the RT truth dataset for a BRDF-capable model (L23 is single-geometry; a multi-angular
set like PB24 is needed for Q8=b). These three shape the roadmap section.

Next: await answers to Q10–Q12, then write `context/rt_forward_model.md`.

### 2026-07-19 (Wrote context/rt_elastic_model.md with three L23-derived figures)

Executed the 4th Setup prompt. Reviewed the Round-3 answers (Q10=d lay out all three
forward-model options; Q11=a strictly elastic, renamed the report to
`context/rt_elastic_model.md`; Q12=c L23 first, PB24 phase 2) and wrote the report.

**Figures — generated with Python, saved to `context/RT/`.** New script
`context/RT/make_rt_elastic_figures.py` (ocean14). It reads the L23 **elastic** set
directly (`$OS_COLOR_DATA/Loisel2023/Hydrolight100.nc`, i.e. `loisel23.load_ds(1,0)`,
X=1 = no inelastic — the same scenario the BING Gordon fits used, confirmed in
`calc_gordon.py`) and the BING coefficient CSVs. Convention `rrs = Rrs/(A+B·Rrs)`,
A=0.52, B=1.7. Three figures:
- `fig_rrs_vs_u.png` — the central point: at 440/550/665 nm, `rrs` vs `u` colored by
  `bbp` with the standard Gordon curve overlaid. Shows the `bbp` fan (blue) and the
  near-constant red-λ offset — `rrs` is not univocal in `u`.
- `fig_rrms_ladder.png` — per-λ rRMS recomputed from L23: standard → per-λ 2-param →
  `+G0` → `+Gb`. **Recomputation reproduces the BING logs** (standard 700 nm 9.04%
  vs logged 9.05%; `+G0` 550 nm 1.12% vs 1.11%; `+G0` 700 nm 0.35%), so the ladder is
  authentic, not transcribed. `+G0` wins red, `+Gb` wins blue — complementary.
- `fig_G_lambda.png` — `G0(λ)` (sign change ~510 nm) and `Gb(λ)` from the shipped
  BING CSVs.

**Report `context/rt_elastic_model.md`** (per Q6=b broad synthesis, Q11 rename;
A1=c synthesis+roadmap; A4 no plumbing; A3 validation gaps not foregrounded).
Sections: (1) why the forward model gates the retrieval; (2) the organizing fact —
`rrs ≠ f(u)` — with the L23 figure and the `β(π)/bb` = 0.23 (water) vs 0.12–0.16
(particle) physics from Pitarch; (3) the elastic lineage Gordon 1988 → PR05
(baseline) → Tan 2018 → L11/O25 (recommended evolution) → Hansen (background);
(4) the BING `G0`/`Gb` deep-dive with the two data figures, and the explicit
convergence BING-`G0/Gb` ↔ O25-`ωbw/ωbp`; (5) a one-table synthesis; (6) baseline =
PR05, recommend evolving to O25/L11, and our-own-approach kept open as three options
(analytic / learned / hybrid, per Q10=d); (7) a 5-step roadmap with BRDF first-class
(Q8=b) and L23→PB24 truth data (Q12=c), inelastic explicitly out of scope (Q11=a);
(8) references.

Learned/confirmed: L23 elastic is 3320 scenes × 81 λ (350–750 nm); `bbnw` is the
particle (non-water) backscatter, `bb_w = bb − bbnw`. The recomputed ladder is
independent corroboration of the BING result from the raw Hydrolight data.

Next: await review of `context/rt_elastic_model.md` (and any request to add the O25
bivariate curve to the ladder figure, per roadmap step 1).

### 2026-07-19 (Report moved to context/RT/; fixed figure paths)

Executed the 5th Setup prompt. JXP had already moved the report to
`context/RT/rt_elastic_model.md` (co-located with the figures and the figure
script). Updated the three image links from `](RT/fig_*.png)` to `](fig_*.png)` so
they resolve now that the report sits in `context/RT/`. Verified all three links
(`fig_rrs_vs_u.png`, `fig_rrms_ladder.png`, `fig_G_lambda.png`) point at the
co-located PNGs. The remaining `RT/`-prefixed strings in the report are prose repo
paths (Hansen PDF, BING CSVs, the figure script), not image links, and stay correct.
No content changes. The figure script's `HERE`-based output path already writes into
`context/RT/`, so re-running it stays consistent with the new location.

### 2026-07-31 (Revised context/RT/rt_elastic_model.md per Robert Frouin's RT comments)

Executed the 1st Edits prompt: reviewed R. Frouin's pasted comments and revised the
report. His two thrusts — (1) most analytical models bury the particle phase function
in coefficients/LUTs from a *prescribed* phase function and so cannot represent
independent phase-function-shape variability, causing geometry-dependent forward-model
error and IOP bias, with **ZTT (Twardowski & Tonizzo 2018)** the exception; (2) target
a **full-RT (HydroLight) reference** with phase-function parameters varied, a **fast
differentiable emulator** for retrieval, **ZTT** as analytical benchmark/backbone, the
**hybrid** `Rrs = Rrs(ZTT) + ΔRrs(emulator)`, and **O25 demoted to a comparison model**.

**Changes made.**
- Header: added an "Advisory input (R. Frouin)" note; broadened scope to
  `Rrs(λ; a, bb, phase function, geometry)`; added ZTT + HydroLight to the arc.
- §2: new paragraph making the **phase-function shape** the deeper axis behind
  `rrs ≠ f(u)` — most models hide it; only ZTT exposes it; it drives geometry-dependent
  bias. Framed as the single most important structural gap for our goal.
- §3.4: retitled O25 from "the recommended evolution" to "the modern semi-analytical
  **benchmark**"; noted its phase function is implicit (PB24 FF), so it is a
  BRDF/retrieval *comparison*, not the physical reference.
- §3.5 (new): **ZTT** — backward VSF / phase-function parameters explicit; analytical
  benchmark and candidate physical backbone. Hansen renumbered to §3.6.
- §5 table: added **ZTT** and **HydroLight (reference)** rows and a "Phase function"
  column (prescribed/implicit vs explicit/adjustable); rewrote the through-line to end
  on the phase-function axis.
- §6: restructured into three roles — **physical reference = HydroLight** (PF varied);
  **analytical benchmark/backbone = ZTT**; **comparison = PR05, O25/L11**; and the
  retrieval operator as our-own-approach with the **hybrid (c) now advisor-recommended**
  (`Rrs = Rrs(ZTT) + ΔRrs(emulator)`), keeping Q10=d's three options.
- §7 roadmap: now 6 steps — reproduce on L23 (+ overlay O25 & ZTT) → retrieval-impact
  MAPE → **build the HydroLight reference with the phase function varied** (PB24 in
  parallel) → **stand up ZTT** benchmark → **differentiable emulator + hybrid**, O25/PR05
  as comparisons → conventions. Geometry **and phase-function shape** are first-class.
- §8: added Twardowski & Tonizzo (2018) and Mobley/HydroLight references.

Note: reconciled with prior answers — Q7 already flagged "our own new approach" and
Q10=d left it open, so Robert's steer *fills in* the concrete architecture (hybrid,
ZTT backbone, HydroLight reference) and reframes O25 from "evolution to adopt" to
"comparison model" without contradicting JXP. Did not read the ZTT PDF (not in the
folder); represented it from Robert's description + the β(π)/bb physics already cited.

Next: await review; if wanted, add a ZTT/O25 overlay to the L23 ladder figure
(roadmap step 1) and/or source the ZTT paper to deepen §3.5.

### 2026-07-31 (Posed Round-1 Design Q&A before writing design/rt_elastic_model.md)

Executed the 1st Design prompt: posed six questions (DQ1–DQ6 in Q&A/Design) to scope
the design document, which will turn the synthesis roadmap + Robert's architecture
(HydroLight reference → differentiable emulator → ZTT-backbone hybrid; O25/PR05 as
comparisons; L23→HydroLight/PB24; elastic-only) into a buildable engineering plan.

The questions target the concrete build decisions the synthesis left open: DQ1
doc horizon (near-term buildable vs full multi-year vs layered — lean layered) and
code home (a `robust/rt/` subpackage with a single `forward(iops, phase_params,
geometry)→Rrs` API); DQ2 the RT reference solver & whether HydroLight is actually
available vs a solver-agnostic interface with an open MC/OSOAA default; DQ3 the autodiff
stack (JAX vs PyTorch; ZTT analytic in-framework for end-to-end differentiability —
lean JAX); DQ4 the phase-function parameterization (FF / ZTT backward-VSF / HG / learned
VSF basis — lean ZTT backward-VSF as primary, FF as the training-data generator); DQ5
inversion coupling / scope (forward-only vs forward+inversion; reuse BING MCMC vs
gradient/amortized SBI with priors — lean forward-in-depth + inversion interface,
full inversion deferred); DQ6 acceptance criteria (protocol-only with held-out
geometry/phase-function splits + component-retrieval MAPE, no blind target — consistent
with the biomass report's stance). Gave my lean on each.

Next: await DQ1–DQ6, then write `design/rt_elastic_model.md`.

### 2026-07-31 (Reviewed DQ answers; wrote design/rt_elastic_model.md)

Executed the 2nd Design prompt. The DQ answers were complete enough to write the design
doc without another Q&A round (DQ4 was explicitly delegated to me; DQ6 gave clear
acceptance themes), so I proceeded rather than asking again.

**DQ answers applied.** DQ1=(a) near-term buildable, ~1-week first prototype → the doc
centers a concrete Week-1 milestone. DQ2=(a) HydroLight available + L23 outputs in hand
→ prototype uses L23 as the in-hand reference; confirmed L23 elastic (X=1) exists at 3
solar-zenith angles (`Hydrolight100/130/160.nc` = 0/30/60°, nadir, fixed FF), so the
solar-zenith geometry axis is exercisable now. DQ3=JAX + analytic ZTT in-framework,
end-to-end differentiable. DQ4="your pick, keep it simple" → I chose the **particulate
backscattering ratio B_p = bb_p/b_p** (Fournier–Forand, one scalar; directly available
in L23 as bbnw/bnw), with the ZTT backward-VSF parameterization documented as the
planned generalization. DQ5=forward-model only but differentiable → inversion out of
scope, API kept inversion-ready. DQ6="fast, accurate, differentiable" → a protocol-only
validation section (no blind targets): accuracy (rRMS vs reference per λ/geometry/B_p +
held-out generalization, alongside Gordon/PR05/O25), speed (throughput/latency), and a
gradient-correctness gate (jax.grad vs finite differences).

**Doc `design/rt_elastic_model.md`** (new `design/` dir): goals/non-goals; the hybrid
architecture diagram; the `robust/rt/` subpackage layout + `forward(iops, phase_params,
geometry) → Rrs` API; components (reference data, phase-function param, ZTT-in-JAX,
residual emulator, hybrid); JAX/Flax/Optax stack + conventions (A=0.52, B=1.7); the §6
validation protocol; the §7 one-week prototype definition and sequence; the beyond-week-1
forward-model track; and a §9 "decisions I made on your behalf" list flagging the DQ4
pick, the prototype target, and the protocol-only acceptance for correction.

Next: await review of `design/rt_elastic_model.md`; on approval, scaffold `robust/rt/`
and build the Week-1 prototype.

### 2026-07-31 (Posed Round-1 Coding Q&A before writing the coding plan)

Executed the 3rd Design prompt: posed six questions (CQ1–CQ6 in Q&A/Coding) to scope
`design/rt_elastic_model_coding_plan.md` (the staged implementation plan + milestones).
The design doc already fixed architecture/stack, so these target the process/
implementation decisions still open: CQ1 execution model (Claude implements on a branch
with JXP running git, vs a spec/handoff — lean Claude implements); CQ2 milestone
structure (milestone-gated M1–M5 with acceptance gates, fine-grained for week 1);
CQ3 the **ZTT equations dependency** — the paper isn't in `context/RT/`, so source it /
JXP provides / start with a Gordon-O25 analytic placeholder backbone and swap ZTT in
(lean: placeholder now + source in parallel); CQ4 testing/tooling (pytest gates incl.
the gradient check, ruff, jaxtyping; match BING conventions); CQ5 environment (add
jax/flax/optax to a dedicated env vs ocean14; CPU vs GPU); CQ6 the validation split
policy / M4 gate (random-20% scenes + hold out one solar-zenith angle). Gave my lean on
each. CQ3 and CQ1 are the load-bearing ones.

Next: await CQ1–CQ6, then write `design/rt_elastic_model_coding_plan.md`.

### 2026-07-31 (Reviewed CQ answers; wrote design/rt_elastic_model_coding_plan.md)

Executed the 4th Design prompt. CQ answers: CQ1 Claude implements now on a branch, JXP
runs git & reviews; CQ2 milestone-gated; CQ3 the ZTT paper is now in the repo
(`context/RT/twardowski2018.pdf`, confirmed 30 pp, extractable); CQ4 pytest-gate each
milestone, follow BING conventions; CQ5 CPU-only, may update requirements + ocean14;
CQ6 agreed split (random 20% scenes + hold out 60° zenith).

Recon before writing: confirmed retrieve-or-bust layout (`robust/` package, one
`__init__.py`; requirements.txt already has pytest/xarray/ocpy/bing; no tests yet), and
BING's test convention (`bing/bing/tests/test_*.py` + `conftest.py` + `files/`) → mirror
as `robust/tests/`. Confirmed the ZTT PDF is Twardowski & Tonizzo 2018 (*Applied
Sciences*, VSF-explicit ocean-color model) and extractable for M2.

**Doc `design/rt_elastic_model_coding_plan.md`.** Ground rules from the CQ answers;
`robust/rt/` + `robust/tests/` layout mirroring BING; a milestone table and detailed
**M0–M5**: M0 env+scaffold (add jax/flax/optax CPU, update requirements) → M1 data+
conventions (L23 loader via ocpy, A/B, `B_p`, golden-value gate) → M2 ZTT-in-JAX
(transcribe twardowski2018; gates = reproduce a paper reference case + grad-vs-FD) →
M3 emulator+hybrid (Flax MLP ΔRrs, Optax, relative weighting; gate = beat standard
Gordon + grad check) → **M4 validation = prototype done** (CQ6 held-out splits; gate =
beat Gordon on both splits + grad gate; table+figures) → M5 beyond-week-1 (HydroLight
PF/BRDF, ZTT backward-VSF params, freeze API). Each milestone pytest-gated; the
gradient-correctness and golden-value tests are first-class; accuracy gates are
*relative* (no blind targets). Risks section keeps the Gordon/O25-in-JAX placeholder
backbone as ZTT de-risking.

Next: on approval, create the `rt-elastic-prototype` branch and start M0.

### 2026-07-31 (Generated the M0–M5 coding prompt docs, modeled on PAB)

Executed the 5th Design prompt: created six execution prompt docs
`claude_prompts/RT/rt_elastic_coding_prompt_{1..6}.md`, one per milestone (M0→1 … M5→6),
modeled after the PAB repo's `claude_prompts/coding_stage*.md` / `initial_coding.md`
style.

Studied PAB's pattern first (`PAB/claude_prompts/{start_up,initial_coding,coding_stage4}.md`):
each stage doc has Goals → Claude (Skills + Working agreements) → Context (files to read
+ existing seams) → Prompts (numbered, "Execute the Nth task in section X") → the
Stage/Tasks section with explicit gates → Q&A → pointer to next → Logging/Logs; PAB also
keeps a running `*_implementation.md` record.

Each RT prompt doc mirrors that: shared Working agreements (git by JXP on the
`rt-elastic-prototype` branch; ocean14; CPU-only JAX; reuse ocpy/bing; BING test layout;
pytest-gated; Fable; log), milestone-specific Context, numbered Prompts, a gated Tasks
section, an empty Q&A, and a Next pointer. Mapping: **1**=M0 env+scaffold (+ creates the
running record `design/rt_elastic_implementation.md`); **2**=M1 data+conventions (L23
loader, B_p, splits, golden-value gate); **3**=M2 ZTT-in-JAX (transcribe
`twardowski2018.pdf`; reference-case + grad-vs-FD gates; Gordon/O25 placeholder de-risk);
**4**=M3 emulator+hybrid (beat-Gordon + grad + throughput gates); **5**=M4 validation =
prototype done (CQ6 held-out splits; acceptance gate; a PR section like PAB's);
**6**=M5 beyond-week-1, deliberately a coarse placeholder to be detailed from M4 results
(HydroLight PF/BRDF, PB24, ZTT backward-VSF params, freeze API).

Next: on approval, execute `rt_elastic_coding_prompt_1.md` (M0).

### 2026-08-15 (Posed Round-1 Report Q&A before writing report/rt_elastic_model.md)

Executed the 1st Report prompt: surveyed the completed work, then posed five
questions (RQ1–RQ5 in Q&A/Report) rather than writing the report yet.

**State surveyed.** `design/rt_elastic_implementation.md` (v0.16, read in full),
`design/prototype_summary.md`, and `design/validation/` (metrics + figures).
The Week-1 prototype is complete per the amended M4 gate: `robust/rt/` holds the
ZTT backbone (TT2017 µ∞ stand-in), the 417-parameter residual MLP emulator, the
hybrid `forward()`, Gordon and O25-refit comparisons, and the §6 validation
protocol; 279 tests pass (256+23 skips without `$OS_COLOR` = CI), ruff clean,
five executed notebooks. Headline (rRMS in `rrs`, held-out scenes): hybrid-MLP
**0.30%** vs O25-refit 0.69%, hybrid-linear 2.54%, ZTT 5.93%, Gordon 7.21%;
unseen-60° is reported not gated (hybrid seed-dependent 4.7–12.2% vs O25's
deterministic 4.63%). The `report/` directory does not exist yet.

**Why these questions.** RQ1 audience/length and RQ4 framing decide whether the
report inherits `prototype_summary.md`'s caveats-forward stance and adds an M5
recommendations section. RQ2 flags a real collision — `report/rt_elastic_model.md`
would be the third file with that name (after `context/RT/` and `design/`) — and
asks whether the report links to or supersedes `prototype_summary.md`. RQ3 asks
whether to generate fresh figures with a script in `report/` (the Setup-report
convention) or reuse the committed validation artefacts. RQ5 lists the formally
unfinished threads (Eq. (8) µ∞ coefficients, M3 task-5 hand-off edit, the M5
placeholder prompt) and asks whether any should close before the report calls
the prototype done.

Next: await RQ1–RQ5, then write `report/rt_elastic_model.md`.

### 2026-08-15 (Wrote report/report_rt_elastic_model.md with three script-generated figures)

Executed the 2nd Report prompt. Reviewed the RQ answers — RQ1 = (a)+(c), a team
report that will evolve into a public (non-refereed) write-up; RQ2 = name it
`report/report_rt_elastic_model.md`; RQ3 = (a) script-generated figures in
`report/`, drawing from the committed validation artefacts; RQ4 = (a)
caveats-forward with an M5 recommended-priorities section; RQ5 = write now, with
an Open Items section for JXP to dispatch — and wrote the report.

**Figures — new script `report/make_report_figures.py`** (ocean14), three PNGs
beside the report:
- `fig_architecture.png` — the hybrid dataflow: inputs → ZTT backbone (unfitted
  physics, TT2017 µ∞ stand-in) and the 7-feature → MLP(16,16) → δ path, combined
  as `rrs = rrs_ZTT(1+δ)` → the air–water interface; annotated with the gradient
  gate and the DomainWarning/fallback behaviour.
- `fig_rrms_ladder.png` — the per-λ five-model ladder on held-out scenes, read
  from `design/validation/rrms_per_wavelength.csv` (the "MLP + fallback" column
  dropped as a bitwise duplicate on L23). Log axis; every series direct-labelled.
- `fig_unseen_zenith.png` — the unseen-60° dot-and-range comparison; the five MLP
  seed values (transcribed from `design/validation/metrics.md`, provenance noted
  in the script) as open circles with the median filled; analytic models as
  deterministic dots.
  The five-color palette (Okabe–Ito subset, one color per model across all
  figures) was validated for CVD safety in stacking order: worst adjacent
  protan/deutan ΔE 11.4, normal-vision floor 18.4; the two sub-3:1-contrast hues
  carry direct labels. Figures were rendered and visually checked; one layout
  pass fixed box-overflow in the schematic, a label collision at the ladder's
  right edge, and a truncated title.

**Report `report/report_rt_elastic_model.md`** (~7 pp). Structure per the RQ
answers: executive summary (0.30% held-out rRMS, the honest 2.3×-over-O25 margin
stated in the same breath); §1 motivation (rrs ≠ f(u), the phase-function axis,
Robert's architecture); §2 the model (backbone, emulator, guard rails); §3 data +
protocol (scene/zenith splits, rrs-space rRMS, the fairness-to-O25 refit); §4
results (headline table, the three figures, speed, gradients, 279 tests); §5
may-claim/may-not-claim (six items, carried from `prototype_summary.md`); §6 Open
Items for JXP (Eq. (8) coefficients, geometry extrapolation as an M5 design
decision, M3 task-5 closable as overtaken, prompt 6 placeholder, merge/archival
calls); §7 M5 priorities in leverage order (vary the phase function in the
reference data first, then retrieval-impact MAPE — the decisive metric — then
geometry data, PhaseParams promotion, Eq. (8)); §8 reproducibility commands; §9
document map; references. Links to the sibling docs are relative
(`../design/...`, `../context/RT/...`) so they resolve from `report/`.

Next: JXP reviews the report and dispatches §6's open items; prompt doc 6 (M5)
gets detailed from §7.

### 2026-09-19 (Merge forensics: why `origin/main` conflicts with `rt-elastic-prototype`)

Executed the 1st PR prompt. Read-only git throughout — the working tree was left
exactly as found, mid-merge with `MERGE_HEAD` present and nine files unmerged.
No file was staged, committed, or resolved.

**What I examined.** `git merge-base` → `12cbcbf` (2026-08-06), the tip PR #12
merged. `git cherry origin/main rt-elastic-prototype` → all ten local commits
marked `+`, so none of M5 exists on main in any form. Walked the PR chain's
parents (`git log --format='%h %P'` on #9–#14, #21–#24) to establish that #9–#12
targeted the long-lived `RT` branch rather than `main`, and that `RT` reached
`main` only on 2026-09-18 via PR #13. Compared the three `robust/` trees at base,
HEAD and `MERGE_HEAD`. Listed the twelve files touched on both sides (`comm -12`
over the two name-only diffs) and read every conflict hunk, plus the full
base→main diffs of the three files that auto-merged (`ztt.py`, `baselines.py`,
`run_validation.py`) to check for silent semantic conflicts.

**Finding.** A stale fork, not a collision: M5 was developed here 08-08 → 08-13
and never PR'd back, while `inelastic-rt`, `cdom-rt` and the docs sweep edited
the same four modules from 08-21 onwards, unaware of it. Ours +10,679 over 10
commits; theirs +49,352 over 84. Eight of the nine conflicts are pure unions
(different kwargs, different `__all__` entries, different fixtures, test blocks
appended at the same EOF). The two auto-merged code files were docstring-only on
main's side, so nothing silent got through there.

**The one hunk with teeth**, worth recording because a literal `--theirs` would
import fine and fail at runtime: `validation.py:846`. Main refactored
`gradient_report`'s comparison loop into a shared `_grad_vs_fd`; M5 rewrote the
setup above it. Both are wanted, but main's replacement reads
`dtype = jnp.asarray(a0).dtype` and `a0` is a binding M5's rewrite removed —
`NameError` on the first gate. Resolution is `jnp.asarray(iops.a).dtype`. I
diffed `_grad_vs_fd` against HEAD's inline loop first: identical arithmetic and
identical edge cases, so taking main's helper loses no M5 behaviour.

**Also flagged**, because clean ≠ green: main added a `sphinx (-W)` CI job with
`nitpicky = True`, and M5 predates it — three surviving `:func:`robust.rt.forward``
targets (`types.py:284`, `conventions.py:760`, `conftest.py:267`) that the docs
sweep renamed everywhere else will turn that required job red. `docs/api.rst`
also has no `automodule` for the 1,479-line `robust/rt/data/pb24.py`.

Recommended resolving in place and PR'ing this branch into `main` — a rebase
would replay the same nine conflicts up to ten times for nothing — and noted the
process cause: this branch was treated as long-lived while `RT` was the real
integration branch, which is why the elastic *report* (2026-08-15) advanced on a
line that did not carry the elastic *code*.

Wrote Forensics above (timeline, per-file table, per-conflict resolutions, the
post-merge punch list, a six-step order of work) and posed PQ1–PQ5 in
Prompts/PR/Q&A: whether to write resolutions into the tree now, how to treat the
superseded `SUPPORTED_THETA_S` prose, how much of the docs debt belongs in this
PR, whether to re-run both validation drivers before it, and whether an
inelastic × off-nadir smoke test is this PR's job or M6's.

Next: JXP answers PQ1–PQ5; then resolve, lint, test, fix the docs targets, and PR.

### 2026-09-19 (Reviewed PA1–PA5; posed Round 2 before touching the conflicts)

Executed the 2nd PR prompt. No conflict was resolved and no file outside this
prompt doc was edited — the tree is still mid-merge exactly as in the previous
entry.

**Checked the data claim rather than asserting it** (PQ4). `$OS_COLOR` is
`/Users/xavier/Projects/Oceanography/data/Color/`; `SD/v5/` holds 10,001 PB24
files; `Loisel2023/` holds the elastic X=1 set (`Hydrolight1{00,30,60}.nc`) and
all six X=2/X=4 files `conftest.L23_INELASTIC_FILES` wants. Imported `ocpy` under
`ocean14` and confirmed `loisel23.l23_path` resolves to that directory with all
six files present. **Nothing needs `profx`.** Noted in passing that
`$OS_COLOR_DATA`, which the Setup prompts cite, is unset — harmless, `ocpy` goes
through `$OS_COLOR`, but the prompt doc's path is stale.

**A5 sent me to read `design/m5_report.md` and `rt_elastic_coding_prompt_7.md`,
and that is the substance of this round.** Yes, M6 exists — prompt 7 is on this
branch only (main has 1–6), it is a *scoping* prompt whose task 0 is "decide what
M6 is", and it carries Q18 (route A/B/C; are HydroLight runs on the table) and
Q19 (restate the headline claim?) unanswered since 2026-08-13.

The thing I should have put in the Forensics report and did not: **M5's gate
failed.** Both splits, every seed; no PB24 weights exist; `load_default()` is
still the L23 nadir model. `Ψ_KLu(ψ)` is fitted for ψ ≳ 134° and crosses zero at
110.4°, so 42% of PB24's sanctioned geometries extrapolate, 16% are sign-flipped,
and 22.3% of ZTT's `rrs` come out zero or negative. The hybrid's bounded
*relative* correction (`|δ| ≤ 0.5`) cannot reach a negative backbone, and the
oracle proves the form rather than the network is the limit — 5324% oracle vs
5484% trained, within 3%. L23's minimum ψ is 139.7°, so none of the Week-1
numbers are touched, and nadir-only work could not have found it. Consequence for
the merge: what goes to `main` is gated *infrastructure* (PB24 loader/splits,
surface transfer 7.2× better off-nadir, full-grid `O25Table`, per-model
`Envelope`, `rrms` masking, 416 tests) **plus a negative result** — not a
BRDF-capable model.

**Tested A5's goal against the code instead of taking it at its word.**
"Complete forward model with all of the processes" holds on the *process* axis
today, but `grep` shows `inelastic.py`, `inelastic_corr.py`, `cdom_fl.py` and
`ed.py` contain **no occurrence of `theta_v` or `dphi`** — every process is
nadir-only, interpolating three solar-zenith anchors. So PQ8 asks which reading
is the goal: all-processes-at-nadir (reachable in this PR) or all-processes
across the BRDF (blocked on Q18).

**Found one concrete contradiction the merge creates.** `docs/model/forward.md:173–180`
presents `on_out_of_domain="ztt"` as degrading "to the analytic backbone exactly
where the emulator was measured to be unreliable", and the elastic report §5 item
5 says the domain check "correctly flags any off-nadir view". Both written
2026-08-15 on the `RT` line — two days *after* M5 measured that the fallback
target is non-physical on 22.3% of off-nadir geometries. Neither was wrong when
written; after this merge they share a tree with the measurement that refutes
them, on the published site.

**Posed PQ6–PQ11**: answer Q18/Q19 now or after the merge (I recommend now, since
Route C is mostly a docs action and this PR already opens `docs/`); fix the
`on_out_of_domain` guidance in prose, in code, or both (I lean prose — changing a
shipped policy inside a merge PR makes it unreviewable); which reading of
"complete forward model"; whether HydroLight runs are on the table (Q18's
blocking half, and Robert Frouin's comment in this doc asks for exactly the same
thing from the other direction — two independent routes to one ask); whether to
fix the two `m5_report.md` §6 contaminations before `surface_pb24.npz` ships to
`main` (I lean fix); and the exact re-run list, flagging that
`train_emulator_pb24.py` exits 1 *by design* so a reviewer does not read the gate
working as a regression.

Next: JXP answers PQ6–PQ11; then resolve the nine conflicts, lint, test, docs,
re-run, PR.

### 2026-09-20 (Resolved the nine merge conflicts; 674 tests pass, ruff clean)

Executed the 3rd PR prompt. All nine conflicted files are resolved **in the
working tree**; nothing was staged or committed — `git status` still shows them
`UU`, and `git add` is yours.

**Resolutions.** Eight went the way the Forensics report predicted:

| file | resolution |
|---|---|
| `types.py` ×3 | union on the signature and the numpydoc block (`bb_w_mode` **and** `a_ph`/`a_cdom` — the *bodies* had already auto-merged with both, so dropping either half was a `NameError`); HEAD's past-tense `PhaseParams` docstring |
| `conventions.py` ×1 | union of both `__all__` groups |
| `emulator.py` ×2, `hybrid.py` ×1 | HEAD — per PA2, M5's per-model `Envelope` supersedes the `SUPPORTED_THETA_S` prose; the constant itself survives as `DEFAULT_ENVELOPE`'s default, so main's two code references still resolve |
| `test_{conventions,emulator,validation}.py` ×1 each | union. Checked first for name collisions between the two appended blocks and against the rest of each file: **none**, 0 of 66 names |
| `validation.py` ×2 | union of `__all__` and of the constants block |
| `validation.py` ×1 | the custom resolution from Forensics §5.1 |

**`conftest.py` was not a union, and a union would have been wrong.** The two
conflict hunks *interleave* two different functions that happen to share a
`Returns/-------/bool` docstring block between them — concatenating ours+theirs
yields `def pb24_available():` immediately followed by `def l23_inelastic_available():`
with one body between them. Hand-wrote the region instead, emitting both function
families complete: PB24 (`PB24_SMALL_FIXTURE`, `pb24_available`, `needs_pb24`,
`pb24_reader`, `pb24_small_batch`) then L23-inelastic (`l23_inelastic_available`,
`needs_l23_inelastic`, `_correction_weights_committed`, `needs_weights`). This is
the one conflict a mechanical resolver would have silently broken.

**Two things the mechanical pass got wrong, both caught by ruff, not by me.**
Recording them because the pattern is the point:

1. `validation.py`'s third hunk carried more than it looked like. Main's side of
   the marker was not just the four-line tail of `gradient_report` — it was that
   tail **plus the whole of `inelastic_gradient_report` and `cdom_gradient_report`**,
   ~200 lines. Replacing the block with the corrected tail deleted both functions;
   `F822 Undefined name 'cdom_gradient_report' in __all__` is what said so. Restored
   them verbatim from `MERGE_HEAD`.
2. They then wanted `from . import types`, which main added and M5 removed (M5's
   rewrite uses `dataclasses.replace` and constructs no containers). Git's
   auto-merge of the import block took M5's. Re-added; `types` imports
   `conventions` only, so no cycle.

The §5.1 fix itself held: `dtype = jnp.asarray(iops.a).dtype` in place of main's
`a0`, which M5's rewrite had removed.

**The freeze fired, which is the merge's only real decision.** `test_env.py` —
M5 task 15, and on this branch only, so main's inelastic work never saw it —
failed three ways: `forward`/`rrs_forward` gained `inelastic` and `corrections`,
`IOPs` gained `a_ph`/`a_cdom`, `Geometry` gained `Ed`. Checked each against what
§8 permits before touching anything: all five are keyword-only or appended, all
default to `None`, the five positional parameters are untouched in name, order
and default, and nothing was renamed, reordered or removed. So this is a
permitted mutation and the constants needed re-baselining, not the merge
rethinking. Updated `FROZEN_FORWARD` and `FROZEN_CONTAINERS` with a comment
recording the delta and why it qualifies — a freeze that is re-baselined without
a written reason is not a freeze.

**Verification** (all under `ocean14`): `ruff check robust/` clean;
`ruff format` reformatted three files (union joins wanted blank lines) and now
passes `--check`; `pytest robust/tests` → **674 passed, 5 skipped, 0 failed**,
1m46s. The 5 skips are all principled — three machine-anchored bitwise hash pins
(`ROBUST_HASH_ANCHOR` is `tank`/`mac`, not this laptop), one removed fallback
path, one undefined azimuth at the poles. **No data skips**, confirming both PB24
and L23 are genuinely in play rather than quietly absent.

Still to do, from the answers already given and untouched by this prompt: the
three `:func:`robust.rt.forward`` targets (`conftest.py:267`,
`conventions.py:757`, `types.py:277`) that the `-W`/nitpicky docs job will reject;
the `pb24` API page; the PQ7 prose note on the `on_out_of_domain="ztt"` guidance;
the PQ10 surface-transfer refit; and the PQ11 re-runs.

Next: `git add` the nine files, then the docs and re-run work.
