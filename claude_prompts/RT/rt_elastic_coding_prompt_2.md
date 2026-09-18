# Elastic RT Coding — Prompt 2 (M1: Data & conventions)

## Goals

Implement **Milestone M1**: the data layer and shared conventions — the `Rrs↔rrs`
convention, the IOP/phase/geometry types, and a one-call **L23 loader** that returns JAX
arrays with the `B_p` phase parameter and the held-out splits. This is the foundation
every later milestone consumes.

M1 is where the package stops being a scaffold: three of M0's stub modules
(`conventions.py`, `types.py`, `data/l23.py`) get real bodies, and the first numbers
enter the repo.

## Claude

### Skills

Consider using `.claude/skills/` (`code-review`, `verify`) as helpful. If a figure is
involved, the `dataviz` skill applies (M0's notebook established the figure conventions —
see *Status entering M1*).

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).

Two additions M0 settled:

- **Ask in Q&A.** Judgment calls that are JXP's to make go in the *Q&A* section of this
  doc as a numbered question, stated so work can continue without an answer. Check for
  answers before the next task.
- **Each milestone ships a notebook** (task 4 below) — the pattern M0 set.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M1.
- **Design** — `design/rt_elastic_model.md` §3 (interface/data model), §4.1 (reference
  data), §4.2 (`B_p`), and the A=0.52, B=1.7 convention.
- **Implementation record** — `design/rt_elastic_implementation.md` (currently **v0.4**;
  §2 is M0 as built, §8 the cross-cutting conventions). Update at close: bump to v0.5,
  add a §3 for M1 with *Modules added / Tests / Results / Notebook* subsections, flip M1
  to ✅ in §1, and refresh the module index.
- **M0's notebook** — `notebooks/RT/rt_elastic_coding_1.ipynb`, for the house figure
  style and what has already been explained (do not re-explain it).
- **L23 loader** — `ocpy.hydrolight.loisel23`:
  - `load_ds(X, Y)` → `Hydrolight{X}{Y:02d}.nc`; use **X=1** (elastic) with
    **Y∈{0,30,60}** (solar zenith 0°/30°/60°).
  - **Resolve the directory through `ocpy`, never a hardcoded path**: `loisel23.l23_path`
    is `$OS_COLOR/Loisel2023`. On this laptop `$OS_COLOR` =
    `/Users/xavier/Projects/Oceanography/data/Color/`, so `l23_path` =
    `/Users/xavier/Projects/Oceanography/data/Color/Loisel2023`. (An older draft of this
    doc said `$OS_COLOR_DATA/…`, which is **unset**; `/Users/xavier/data/Color/Loisel2023`
    is the same directory reached through a symlink, not a second copy.)
  - Dataset anatomy, verified in M0: dims **`IOP_Scenario` = 3320** × **`Lambda` = 81**
    (350–750 nm, 5 nm), coordinate `Lambda`; variables `Rrs`, `Ed_0+`, `Lw`, `Lu_0+`,
    `a`, `anw`, `aph`, `ag`, `ad`, `b`, `bnw`, `bph`, `bd`, `bb`, `bbnw`, `bbph`, `bbd`.
    All three elastic files are on disk.

## Status entering M1

M0 is complete (commit `ccbc0cc`). What M1 inherits — build on it rather than rebuilding:

**The stubs to fill.** `conventions.py`, `types.py`, and `data/l23.py` are
docstring-only, and each docstring already lists its planned contents from the design.
Fill the bodies and keep those docstrings current — they are the module-level
documentation, not scaffolding to delete.

**Test machinery that already exists** in `robust/tests/conftest.py`:

- `needs_l23` — the skip marker for data-dependent tests, plus `l23_available()` and
  `L23_ELASTIC_FILES`. **Use it** rather than writing new skip logic; the suite must stay
  green on a machine with no `$OS_COLOR` mount (verified in M0).
- `jax_x64` — a fixture enabling float64 for one test and restoring the flag after.
- `robust/tests/files/` — empty, `.gitkeep`-ed, and intended for a small cached L23 batch
  so the data tests do not re-read the ~17 MB netCDFs. M1 is the milestone that uses it.

**Environment.** jax/jaxlib 0.11.0, flax 0.12.8, optax 0.2.8, jaxtyping 0.3.11, on
Python 3.14.6, CPU backend. `jax.experimental.enable_x64` **does not exist** in JAX 0.11
— use the `jax_x64` fixture.

**Two gotchas M0 paid for:**

1. **`robust` may not be pip-installed** in `ocean14`. `pip install -e .` was in fact
   *broken* until CI setup exposed why (`setup.py` declared an illegal `provides`
   value; now fixed), so nothing was ever installed and `pytest` has only ever been
   run **from the repo root**. Keep doing that, and keep the `sys.path` bootstrap in
   the notebook (M0's notebook shows the pattern) so it works either way.
2. **JAX defaults to float32.** A "round-trips to 1e-6" test has barely one digit of
   headroom at float32 (eps ≈ 1.2e-7). Either run it under `jax_x64` or state the
   tolerance as explicitly relative and justify it — do not let a tolerance silently
   become a test of the dtype. M0's notebook (§4) has the measurement behind this.

**Open, not blocking:** Q2 in prompt 1 (whether to pin a `ruff.toml`, and whether to
adopt `ruff format`). Until JXP rules, keep new code clean under `ruff check robust/`
and use a commented `noqa` where a default rule is wrong for the code.

## Prompts

1. Read this doc. Execute the 1st task in the "M1" section below. See my answers on ruff in the 1st prompt doc.  If you have any questions, ask me in the Q&A section below.  Use Fable if you can.
2. Read this doc. Execute the 2nd task in the "M1" section below. If you have any
   questions, ask me in the Q&A section below.  Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task in the "M1" section below. Check my answers in
   Q&A. If you have any additional questions, ask me in the Q&A section below.  Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can. Log your work.
5. Read this doc. Execute the 5th task — responding to the PR review. Use Fable if you can. Log your work.
6. Read this doc. Execute the 6th task — modifying the next prompt doc `rt_elastic_coding_prompt_3.md`. Use Fable if you can. Log your work.

## M1

### Tasks

1. **Conventions.** `robust/rt/conventions.py`: `A_RRS=0.52, B_RRS=1.7`;
   `Rrs_to_rrs`/`rrs_to_Rrs`; the canonical wavelength grid (L23 350–750, 81 bands);
   pure-water `bb_w(λ)`; load-time asserts.

   Reuse over reinvention: `bing.rt` exports `A_Rrs`/`B_Rrs` — a test asserting our
   constants equal BING's is worth more than a comment, since the two packages sharing
   `rrs` is the point of fixing them at all. For `bb_w(λ)`, check whether BING already
   has the pure-water model (`bbNWModel.init_bbw` reads `Hydrolight400.nc`) before
   writing one.

   **Test:** `Rrs→rrs→Rrs` round-trips to ~1e-6 (mind the float32 caveat above); asserts
   fire on bad input (wrong grid, negative IOPs).

2. **Types.** `robust/rt/types.py`: `IOPs(a, bb_w, bb_p)`, `PhaseParams(B_p, …)`,
   `Geometry(theta_s, theta_v, dphi, wind)` as JAX pytrees with `jaxtyping` shapes.

   Keep `bb_w` and `bb_p` separate (design §3 — the water/particle split is load-bearing,
   and free for us). Register the pytrees so `jit`/`vmap`/`grad` traverse them —
   `flax.struct.dataclass` and `jax.tree_util.register_dataclass` are both available;
   pick one and say why in the record. `PhaseParams` must be shaped so the ZTT
   backward-VSF parameters can join it at M5 **without changing the `forward`
   signature** (that signature is already pinned in `hybrid.py`).

3. **L23 loader + splits.** `robust/rt/data/l23.py`: load the elastic set via `ocpy`
   for Y∈{0,30,60}; assemble `(IOPs, Geometry, Rrs)` JAX batches; compute
   `B_p = bbnw / bnw`; expose the **seeded splits** (random 20% of scenes; and the
   solar-zenith hold-out: train 0°/30°, test 60°).  

   Measured in M0, so the range assert has a real reference: at **440 nm**, `B_p` has
   median **0.0126** and 1st–99th percentiles **0.0105–0.0180** — comfortably inside the
   design's ~[0.004, 0.03]. That was one wavelength only. **Check the range across all
   81 bands and all three zeniths, and report (do not silently clip) if it fails in the
   UV or the far red** — an assert tuned to 440 nm that fires at 350 nm would be
   discovered at M3, when it is expensive.

   **Gate.** `test_conventions.py` + `test_l23.py`: shapes `(3320, 81)`; `a, bb ≥ 0`;
   `B_p` within ~[0.004, 0.03]; a **golden-value** row cross-checked against the raw
   netCDF. Data tests carry `needs_l23`, so `pytest -q` stays green without the dataset.
   Update the implementation record; note the branch for JXP.

4. **Notebook.** `notebooks/RT/rt_elastic_coding_2.ipynb` — the M1 explainer, following
   the conventions recorded in the implementation record §2.6 and §8:

   - **Executed**, committed with outputs (`jupyter nbconvert --to notebook --execute
     --inplace`), so it reads without a kernel.
   - Data-dependent cells **degrade to a message** when `$OS_COLOR` is absent; bootstrap
     `sys.path` to the repo root (M0's notebook shows both patterns).
   - Figures: recessive grid/frame, text in ink colours not series colours, legend plus
     direct labels, one hue light→dark for sequential magnitude, and the CVD-checked
     categorical pair `#0072B2`/`#D55E00`. **Render and look at** each figure before
     calling it done — M0 caught two label collisions that way.
   - Explain what M1 *decided*, not just what it calls: why `rrs` rather than `Rrs` is
     the fitting space, why `bb_w` stays separate from `bb_p`, what `B_p` is and how it
     varies across λ / zenith / scene, and what the two held-out splits protect against.
     Do not re-explain M0 (JAX, autodiff, the float64 argument) — link to notebook 1.
   - Worth a figure: the `B_p` distribution across λ (it justifies or corrects the range
     assert), and the reference `Rrs` spectra grouped by split so the hold-out is visible
     rather than asserted.

Also read my answer to Q3 in the Q&A section below.

5. **PR Review** I have had Cursor perform a review of our PR.  Please address its comments.  Log your work.

6. **Finally** Modify the next prompt doc `rt_elastic_coding_prompt_3.md` given what we have done here.  Use Fable if you can. Log your work.

### Q&A

**Q3 (M1 task 3, Claude → JXP).** This doc said `robust/tests/files/` was
"intended for a small cached L23 batch so the data tests do not re-read the ~17 MB
netCDFs", and that M1 would use it. **I did not put anything there**, for two
reasons. Reading a file turns out to cost only ~0.27 s, so a session-scoped
`l23_batch` fixture (loads once per run) already removes the repetition — the full
suite is 4.1 s. And the coverage motive is better served another way: the split
logic is now tested against a *synthetic* batch, so it runs in CI, with no data at
all.

What a committed fixture would still buy is CI exercising the **loader itself**
against real numbers, rather than skipping all 16 data tests. A 50-scene ×
3-zenith `.npz` would be ~150–250 KB.

I did not do it unilaterally because it means **committing data derived from
someone else's dataset** (L23, Dryad) into a public repo — a licensing and
repo-hygiene call that is yours, not mine. Say the word and it is a small change:
a `write_test_fixture()` helper plus a `files/`-backed branch in the fixture.
— *No answer needed; task 4 is unaffected.*

>A. Yes, please do generate a small L23 batch

## Next

→ `rt_elastic_coding_prompt_3.md` (M2: ZTT-in-JAX backbone).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-01 (M1 task 6 — handed M2 off in prompt 3)

Rewrote `rt_elastic_coding_prompt_3.md` so M2 starts from what M1 actually built.
The three M2 tasks keep their substance; the work was folding in the API, the
measured numbers, and the traps — plus three additions of my own, below. No code
changed, so the suite stands at 117 passed.

**The biggest addition: build Gordon-in-JAX *first*, and stop calling it a
fallback.** The coding plan frames a Gordon/O25 backbone as the de-risking option
*if* ZTT proves ambiguous. But M3's gate is literally "hybrid beats standard
Gordon" and M4 scores against Gordon/PR05/O25 — so **Gordon is a required artifact
either way**. Building it first therefore costs nothing, gives an immediately
end-to-end path, and makes the de-risk branch free rather than a detour. I also
flagged that the plan's package layout has no home for comparison models and
suggested `robust/rt/baselines.py`, as a Q&A question rather than a decision I
should make alone.

**Second: there is already a Gordon baseline in the repo, and M2 should reproduce
rather than re-derive it.** `context/RT/make_rt_elastic_figures.py` computed
per-λ rRMS for standard Gordon on Hydrolight100.nc, and `fig_rrms_ladder.csv` holds
the answer (**2.49% at 400 nm rising to 9.04% at 700 nm**, Y = 0). I read the script
to extract the *exact* definition it used — `100·sqrt(mean(((pred−truth)/truth)²))`
in `rrs` space, percent, with `G1 = 0.0949`, `G2 = 0.0794` fixed, not fitted — and
put both the formula and the table in the prompt. Reusing that definition verbatim
keeps every number in the project comparable, and reproducing the table is a cheap
cross-check that the new JAX code and the old NumPy figure code agree. Much cheaper
to discover a discrepancy at M2 than at M4.

**Third, the connection I had not previously drawn: M1's water/particle split is
exactly what ZTT needs.** The synthesis doc records `β(π)/bb` ≈ **0.23 sr⁻¹ for pure
water** vs **0.12–0.16 for particles** — so an explicit-VSF model weights the two
backscatter components with *different* backward-VSF values, which is only
expressible if `bb_w` and `bb_p` arrive separately. M1 kept them apart on the
design's general instruction; M2 is where that pays off concretely. I put it in the
facts table so the transcription does not quietly collapse them into `bb`.

**A "Status entering M2" section** lists the M1 API by name (so M2 does not
re-derive `u`, re-implement `bb_w`, or hand-roll a skip marker) and tabulates the six
measured facts that bear on the transcription — with the consequence of each spelled
out rather than left as trivia. The two that constrain M2 most: `B_p` is a
**spectrum**, shape `(n_sample, 81)`, so `Rrs_ZTT` must not reduce it to a scalar;
and `Rrs(60°)/Rrs(0°) = 0.949` is the geometry signal Gordon *structurally cannot*
express, which is why task 3 now asks for rRMS broken out **per zenith** rather than
pooled — pooling would hide the one asymmetry the whole M3/M4 comparison rests on.

**Gotchas carried forward with their evidence, not as folklore.** The gradient gate
must run under `jax_x64`, and I gave M0's measurement as the reason (at 1e-6, float32
meets the check at 0 of 33 step sizes; float64 at 21 of 33) so nobody is tempted to
loosen the tolerance instead. Likewise the finite-difference dtype trap (pin the
dtype on the arrays, or the check silently computes in float64 and proves nothing),
the float32 ceiling on any L23 comparison (`rtol=1e-5`, not 1e-6), validators being
boundary-only, and M1's three-layer test structure — with the note to prefer the
**fixture** layer, since that is what makes CI meaningful.

**Verified every factual claim before writing it**, having previously shipped a
stale path in a prompt doc: the M1 export lists came from each module's `__all__`;
the rRMS numbers and formula from the CSV and script; the `β(π)/bb` figures from
synthesis line 70 (citing Zhang 2009 and Twardowski & Tonizzo 2018); `ztt.py` still
raising its M2 `NotImplementedError`; and the ZTT PDF present on disk but
**gitignored** (`.gitignore:14`, `context/*/*.pdf`) — worth stating, since a reader
without the file cannot check the transcription, which is why I asked for equation
numbers in the docstrings.

I also extended the Prompts list to six to match the tasks, and added tasks 5 (PR
review, with the unauthenticated-`gh` workaround written down) and 6 (hand off to
prompt 4), since those are now the established rhythm rather than one-offs.

Modified: `claude_prompts/RT/rt_elastic_coding_prompt_3.md`. **M1 is fully closed
out** — tasks 1–6 done. Branch `rt-elastic-prototype` for JXP to commit.

### 2026-08-01 (M1 task 5 — addressed the Cursor Bugbot review on PR #9)

`pytest -q` → **117 passed**; without `$OS_COLOR`, **100 passed + 17 skipped**.
`ruff` clean. Record at v0.9.1.

`gh` is not authenticated in this environment, so I read the review through the
public REST API (`/repos/ocean-colour/retrieve-or-bust/pulls/9/{comments,reviews}`).
Bugbot raised **one** issue, medium severity, and it was a fair catch. I also
checked PR #7 (same branch, M0): a review exists but with no inline comments, so
nothing outstanding there.

**The finding.** In `test_iops_are_identical_across_zeniths`, absorption was
compared at all three zeniths but `bb_p` only between 0° and 60°:

```python
np.testing.assert_array_equal(a[0], a[1])
np.testing.assert_array_equal(a[0], a[2])
np.testing.assert_array_equal(bb[0], bb[2])   # 30 deg never checked
```

So a 30° `bb_p` mismatch could not have failed the test, even though its docstring
claims the IOP *fields* are identical across the zenith files. Exactly the kind of
gap I have been warning about in these logs — a test whose name and docstring
promise more than its assertions deliver — and I wrote it anyway.

**The fix goes after the class, not the instance.** Patching in the one missing
line would have left the same hand-written-pair structure that produced the
omission. Instead the test now loops over every field the batch carries (`a`,
`bb_w`, `bb_p`, **and** `B_p`, which the earlier version did not check at all) ×
every non-zero zenith, with an `err_msg` naming the field and angle that differ. So
the assertions now cover the docstring's claim exactly, and adding a field to the
batch cannot silently escape the check.

**Verified the fix catches the specific gap**, rather than just re-running green: I
perturbed `bb_p` at 30° only and confirmed the strengthened loop fails with
`bb_p differs 0 vs 30`. A fix for a missing assertion should be demonstrated to
assert.

I then swept for the same pattern elsewhere in the module — the only other
zenith-indexed tests are the `Rrs`-ratio ones, which already use all three angles.

**One extension beyond the letter of the review.** The full-release test carries
`needs_l23`, so the claim it defends was unchecked in CI. Since the review's real
point was "a claim is under-covered", I added the same loop against the committed
50-scene fixture (`test_fixture_iops_are_identical_across_zeniths`), which runs
without the dataset. That is the 117th test, and it is why the no-data count went
from 99 to 100.

I could not reply on the PR or resolve the thread — no authenticated `gh`, and
posting to GitHub is an outward-facing action I would not take unasked. The commit
message or a reply from you can close it out.

Modified: `robust/tests/test_l23.py`, `design/rt_elastic_implementation.md`
(v0.9.1). Branch `rt-elastic-prototype` for JXP to commit.

### 2026-08-01 (M1 task 4 — the explainer notebook, plus the cached fixture from Q3)

`pytest -q` → **116 passed**; with `$OS_COLOR` unset, **99 passed + 17 skipped** (was
90 + 17). `ruff` clean. Record at v0.9, **M1 complete**.

**First, Q3, which you answered yes to.** `robust/tests/files/l23_small.npz` now
exists: 50 scenes × 3 zeniths, **213 kB** compressed. The design choice that makes
it worth having is what it stores. A snapshot of `load_batch`'s *output* could only
ever be checked for staleness — the loader itself would not run. So the fixture
stores the loader's **input**: the raw per-file fields, behind a new injectable
`reader` seam (`load_batch(..., reader=...)`, `write_fixture`, `npz_reader`, the
same pattern PAB uses for its cloud reader). The real `load_batch` therefore
executes against real L23 numbers wherever the fixture is present. Concretely, CI
went from **90 to 99** passing tests: shapes, the `B_p` band per wavelength, the
`bb_w`-vs-`conventions` cross-check, the zenith ratios, the golden value, and
splits/`select` over genuine scene labels all now run with no dataset. `npz_reader`
refuses a scenario or zenith it does not hold, because silently serving the wrong
sun angle would be far worse than failing. One test also asserts the fixture stays
under 512 kB, so it cannot quietly grow into a data dump in git history.

**The notebook** (`notebooks/RT/rt_elastic_coding_2.ipynb`, 23 cells, executed,
three figures) is organised around M1's four *decisions*, not its call signatures,
and deliberately does not re-explain M0's JAX material.

**Two things writing it taught me that the code alone had not said.**

(1) **The water/particle split matters more than I had been claiming.** I had
written that keeping `bb_w` separate was "load-bearing" on the design's authority.
Actually measuring the share: for the median L23 scene pure water is **~72% of
total backscatter at 400 nm**, ~50% at 550, and still **~29% at 750** (37–87% at
400 nm across scenes). So `bb_w` is not a small correction one could fold into
`bb` — over much of the spectrum it is the *larger* term. My first draft of that
paragraph guessed "roughly half at 400 nm, a few per cent by 750" and was wrong in
both directions; the printed table caught it before it shipped.

(2) **The `rrs` fitting space is not cosmetic.** If `Rrs → rrs` were a constant
scaling, the choice would be presentational — a relative error in one space would
be the same in the other. It is not: the true conversion sits **6% below a linear
rescaling at `Rrs` = 0.02 and 14% below at 0.05**. That is the concrete reason the
design's protocol specifies rRMS *in `rrs` space*.

**Figures — and looking at them earned its keep again.** The first render had three
defects no assertion would have caught. The `Rrs↔rrs` annotation arrow pointed at
the wrong x (I had used the value at the array's last element, not at the `Rrs` =
0.02 it claimed), and the accompanying prose said "drifts ~6%" when the plotted
range reaches 14%. The pole panel was truncated at `ylim=0.09`, so the divergence
it existed to show was off-screen — fixed with a log y-axis, which makes the
asymptote and the decade of headroom above ocean `rrs` both visible. And the
zenith-ratio y-label rendered as literal `R_rs(0u00b0)`, because `°` inside an
**r-string** is not an escape sequence; the surrounding f-strings interpreted theirs
fine, which is exactly why it slipped through.

Figure 2 does the most work: `B_p` percentiles per wavelength against the design
band. It answers prompt 2's question (the range holds at every one of the 81 bands)
and simultaneously shows the caveat — L23 fills a **1.75×** slice of a band
spanning **7.5×** — in one picture. Figure 3 draws the zenith hold-out as
`Rrs(θ)/Rrs(0°)` with a dashed line at 1.0, which *is* standard Gordon's
assumption; that makes the M3/M4 comparison legible rather than something the
reader has to take on faith.

New: `notebooks/RT/rt_elastic_coding_2.ipynb`, `robust/tests/files/l23_small.npz`.
Modified: `robust/rt/data/l23.py` (the `reader` seam, `write_fixture`,
`npz_reader`), `robust/tests/conftest.py` (`l23_small_batch` fixture, `FILES`,
`L23_SMALL_FIXTURE`), `robust/tests/test_l23.py` (9 fixture-backed tests; the module
docstring now describes three layers rather than two),
`design/rt_elastic_implementation.md` (v0.9 — M1 ✅, new §3.4 Notebook, §3.5
Results). Branch `rt-elastic-prototype` for JXP to commit. **M1 is done**; next is
`rt_elastic_coding_prompt_3.md` (M2: the ZTT backbone in JAX).

### 2026-08-01 (M1 task 3 — the L23 loader, seeded splits, and what the data says)

`pytest -q` → **107 passed** (12 M0 + 27 conventions + 32 types + 36 L23); with
`$OS_COLOR` unset, 90 passed + 17 skipped. `ruff check` and `ruff format --check`
clean. Record at v0.8. `load_batch`, `make_splits`, and `select` are implemented.

**First, the measurement this doc asked for — and the answer is reassuring.** The
worry was that a `B_p` range assert tuned to 440 nm might fire in the UV or the far
red. It does not: across **all 81 bands, all 3320 scenes, and all three zeniths**
(268,920 values) `B_p` lies in **[0.01026, 0.01800]**, entirely inside the design's
nominal ~[0.004, 0.03]. So the assert needs no wavelength-dependent escape hatch.
I test it **per band** rather than only globally, since a narrow band-specific
excursion would hide inside a global min/max. Also checked the division: `bnw`
never gets near zero (minimum 6.1e-3), so `B_p = bbnw/bnw` is safe.

**But the same measurement carries an honest limitation, which I would rather
record now than have someone discover at M5.** L23 spans a factor of only **~1.75**
in `B_p` where the design's nominal band spans ~7. The prototype therefore trains
on a *narrow slice* of phase-function space, so "explicit phase-function
dependence" — the headline argument for the ZTT backbone over Gordon — is only
weakly exercised until M5's HydroLight runs vary the phase function properly. That
is a real caveat on what M4 can claim.

**Three more properties of the release, each now a test.** (1) The IOP fields are
**bit-identical** across the three zenith files — the same 3320 water bodies
illuminated three ways, with only `Rrs` differing. (2) `Rrs` falls with solar
zenith: median ratios **0.990** at 30° and **0.949** at 60°. That ~5% is the only
geometry signal in hand, and it is exactly what standard Gordon *cannot* express
(it has no solar-zenith dependence at all) — so it is the lever the M3/M4
comparison pulls. (3) `B_p` **varies with wavelength** within a scene (0.0134 at
350 nm → 0.0125 at 750 nm), so it is carried as a spectrum, not collapsed to a
per-scene scalar. Good thing `types.py` left that open.

On (1): I deliberately did **not** exploit the identity to store one copy of the
IOPs and tile it. The saving is ~13 MB; the cost would be silent breakage if a
future release ever varied them per zenith. The loader reads each file's own IOPs
and concatenates, so the identity is an *observation with a test*, not a
dependency.

**The split is by scene, and that is the single most consequential line in the
module.** Each water body appears three times, once per zenith. A per-*sample*
split would put the same IOPs in both train and test at different sun angles, and
every held-out number afterwards would be quietly optimistic — the M4 gate would
be measuring memorisation. `make_splits` draws *scenes* and expands to a sample
mask, and a test asserts the two scene sets are disjoint and jointly complete.
Because this is the property the whole acceptance gate rests on, I also test it
against a **synthetic** batch, so the split logic runs in CI where there is no
dataset. `make_splits` also refuses to build a split whose zenith hold-out would be
empty: a vacuous gate is worse than a failing one.

**Design choices worth recording.** One flat sample axis (9960 samples), so every
leaf shares the batch shape and `vmap(in_axes=0)` needs no per-field spec.
`L23Batch` is deliberately **not** a registered pytree — it is an analysis
container, and `scene` is host-side integer metadata that must never be traced or
differentiated; a test pins that. And `bb_w`/`bb_p` come from the **file**
(`bb − bbnw`, `bbnw`) rather than from `conventions`, so a batch is exactly what
L23 says, with a test tying the loader's `bb_w` back to the embedded table.

**That last test taught me something, after failing.** I first asserted agreement
at `rtol=1e-6` and it failed on 0.1% of elements, by up to **3.4e-6 relative**. Not
a bug in either module: L23 stores float32, and the two paths differ only in *where*
the subtraction happens — the embedded table was extracted with float32 arithmetic
(what the file's own dtype gives), while the loader upcasts to float64 first, which
is slightly *more* accurate. The disagreement is worst in the red tail, where
`bb_w ≈ 3e-4` and the cancellation is relatively largest. So the tolerance was
simply tighter than the reference data's own precision. I set it to 1e-5 and
documented why, rather than nudging a number until green — the fix is knowing which
of the two values is "right" (neither; they bracket float32 noise) and saying so.

I also raised **Q3**: this doc expected M1 to put a cached batch in
`robust/tests/files/`, and I did not. Loading costs only 0.27 s per file, so a
session-scoped fixture already removes the repetition, and the coverage motive is
better served by the synthetic batch that runs in CI. What a committed fixture
would still buy is CI exercising the *loader* against real numbers instead of
skipping 16 tests — but that means committing data derived from someone else's
dataset into a public repo, which is your call, not mine.

New: `robust/rt/data/l23.py` (implemented), `robust/tests/test_l23.py`. Modified:
`robust/tests/conftest.py` (session-scoped `l23_batch` fixture that skips rather
than errors when the data is absent), `design/rt_elastic_implementation.md` (v0.8 —
§3.2 loader decisions, a new §3.2.1 table of measured properties, §3.3 tests, §3.4
results). Branch `rt-elastic-prototype` for JXP to commit. **M1 tasks 1–3 are
done**; next is task 4, the notebook.

### 2026-07-31 (M1 task 2 — types.py: the three forward() pytrees)

`pytest -q` → **71 passed** (12 M0 + 27 conventions + 32 types); with `$OS_COLOR`
unset, 68 passed + 3 skipped. `ruff check` and `ruff format --check` clean. Record
at v0.7. `IOPs`, `PhaseParams`, and `Geometry` are implemented and re-exported
from `robust.rt`, per the coding plan's "`__init__.py` exports `forward()`, public
types".

**The pytree-registration decision, and a rationale I had to discard.** The task
said to pick between `flax.struct.dataclass` and `jax.tree_util.register_dataclass`
and say why. My first instinct was import cost — M0 recorded a convention of
keeping Flax off the analytic path — so I measured it: **`flax` adds only ~0.08 s
once `jax` is loaded** (it lazy-imports; 116 modules appear in `sys.modules` but
little is actually executed). That killed the argument I was about to make, so I
went with the one that survives: **dependency direction**. These types sit on the
analytic path — M2's ZTT backbone needs them and needs nothing from Flax — so
having the core data model import a neural-network library to describe a container
is backwards. JAX's own mechanism is more primitive and stable, and stdlib
`dataclasses` gives `replace()` and a sane `repr` for free. Flax arrives at M3
inside `emulator.py`, where it earns its place. The measured 0.08 s is in the
record so the convention rests on structure rather than on a speed claim that
would not have survived scrutiny.

I prototyped the choice before writing the module and confirmed all of it:
field inference works with no explicit `data_fields`; `jax.grad` of a scalar of an
`IOPs` returns **an `IOPs`** with per-field derivatives (the shape the future
inversion wants, and the whole reason these are containers); `vmap`, `tree_map`,
`jit`, `dataclasses.replace`, and an extra optional field all behave.

**A subtlety worth the memory: `bb_w` is broadcast to the batch shape.** Storing
it as a bare `(81,)` spectrum is more honest about the physics — water is the same
in every scene — but it makes every leaf *not* share a batch axis, so
`jax.vmap(f, in_axes=0)` would fail and each caller would have to spell out
`in_axes=IOPs(a=0, bb_w=None, bb_p=0)`. Broadcasting costs ~1 MB for a full L23
batch. I took the convenience and documented the trade, with a test asserting the
payoff (plain `vmap` over a batched `IOPs` works).

**Validation stays out of `__post_init__`, and a test enforces that.** Under `jit`
or `vmap` the fields are tracers with no concrete value, so a constructor-time
check would either crash or pass vacuously — the second being much worse. So each
type has an explicit `validate()`, and one test asserts it raises
`jax.errors.TracerArrayConversionError` inside `jit`. That pins the contract: if
someone later "improves" the class by validating on construction, the suite says
no.

**`PhaseParams` is the extension point, and the M5 promise is now tested rather
than asserted.** A local variant with an extra optional field goes through `jit`
and `grad` in the test suite. Two things that emerged: an unset optional field
contributes **no leaves** (so gradients and `tree_map` ignore it), but the
*treedef* does change once set, so `jit` recompiles once per variant — correct and
cheap, but worth knowing before someone sees it in a profile. I deliberately did
**not** invent names for the M5 backward-VSF fields; the contract is documented and
demonstrated, the naming waits for the physics.

I also kept the tight `B_p ∈ ~[0.004, 0.03]` range **out** of the type.
`PhaseParams.validate()` checks only the definitional bound `(0, 1]` — it is a
ratio, not a coefficient. The narrow band is the loader's business (task 3),
because M2/M3 need to sweep `B_p` outside the L23 range to probe the model, and a
type-level invariant would fight that.

**Two things I wrote badly and fixed rather than shipped.** (1) A test named
`test_geometry_validate_rejects_radians` did nothing of the sort — its own comment
admitted the radians case passes, and the assertion it actually made (>90° is
rejected) duplicated another test. The honest fact is that 30° in radians is 0.52,
which sits happily inside `[0, 90]`, so the range check **cannot** catch that
mix-up; it would surface at M3 as a poor fit. I replaced it with
`test_validate_cannot_detect_radians_a_known_limitation`, which pins the blind
spot, and corrected the `validate()` docstring, which had claimed the reverse.
A validator's gaps matter as much as its coverage, and a test whose name overstates
its coverage is worse than no test. (2) A `pytest.raises(Exception)` with a
`noqa: B017` became the specific `jax.errors.TracerArrayConversionError` once I
checked which error JAX actually raises.

New: `robust/rt/types.py` (implemented), `robust/tests/test_types.py`. Modified:
`robust/rt/__init__.py` (status docstring; re-export the three types and add them
to `__all__`), `design/rt_elastic_implementation.md` (v0.7). Branch
`rt-elastic-prototype` for JXP to commit. Next: task 3, the L23 loader and the
seeded splits — where the `B_p` range gets checked across all 81 bands.

### 2026-07-31 (M1 task 1 — conventions.py, plus the ruff config JXP approved)

`pytest -q` → **39 passed** (12 M0 + 27 new). With `$OS_COLOR` unset: **36 passed,
3 skipped** — exactly the three `needs_l23` golden tests, so CI stays green
without the reference data. `ruff check robust/` and `ruff format --check robust/`
clean. Implementation record at v0.6 with a new §3.

**First, Q2 from prompt 1** (JXP: yes to both). Added `ruff.toml` — `select =
E/F/I/W/UP/B`, line length 88, `target-version = py312`, formatter
`quote-style = "double"` — matching PAB so the two repos lint alike. Then ran
`ruff format robust/` (6 files reformatted) *before* writing new code, so
`conventions.py` was authored in the final style rather than reformatted after.
Also added `ruff format --check` to the CI lint job and rewrote its now-stale
comment about there being no config.

Two rules are ignored, both **because of `jaxtyping`**: `F722` (pyflakes tries to
parse a shape string like `" 81"` as a forward reference and calls it a syntax
error) and `UP037` (pyupgrade offers to delete quotes that *are* the shape
specification). The coding plan asks for light jaxtyping on public signatures, so
the rules go rather than the annotations. Worth knowing: `ruff check .` across the
whole repo reports **48** findings, all in pre-existing non-package scripts
(`setup.py`, `reports/py/`, `context/RT/`) — mostly `E702` from `import glob, os`
and long lines. I left them alone and scoped CI to `robust/`; say the word if you
want that swept.

**`conventions.py`.** `A_RRS`/`B_RRS`, `Rrs_to_rrs`/`rrs_to_Rrs`, the canonical
grid, `bb_w(λ)`, and three boundary validators. Decisions worth recording:

- **`WAVE` is NumPy, not `jnp`.** A device array built at import would freeze its
  dtype before a caller can enable float64; `canonical_wave()` converts on demand
  so it follows `jax_enable_x64`. The values are exact multiples of 5, so float32
  holds them without error either way. A test asserts the x64 behaviour.
- **Validators raise `ValueError`, not `assert`.** `python -O` strips `assert`,
  and a convention check that silently vanishes is worse than none. They are
  documented as *boundary* checks — they read concrete values, so they cannot run
  under `jit`, and are for where data enters the package, leaving `forward` clean.
- **`RRS_POLE` is named and checked.** `rrs_to_Rrs` diverges at `rrs = 1/B ≈
  0.588` and goes *negative* past it. Ocean `rrs` is ~1e-3–5e-2, so the only way
  to get there is a unit error — exactly the kind that otherwise surfaces as an
  inexplicable negative Rrs at M3. `check_rrs` looks for it and says "check the
  units".

**On `bb_w`, the one number that came from data — I followed the "reuse" rule and
it led somewhere useful.** Both `bing.bbNWModel.init_bbw` and
`ocpy.water.scattering.bbw_from_l23` compute pure-water backscattering as
`bb − bbnw` from an L23 file, and **both carry a TODO** saying to replace it with
a proper calculation. Chasing that: `ocpy.water.scattering.betasw_ZHH2009` (Zhang,
Hu & He 2009, the T/S-dependent physical model both TODOs point at) **raises
`ValueError("THIS IS NOT SUCCESFULLY CONVERTED YET")` on its first line** — an
unfinished MATLAB port. So the physical path does not exist yet in either package,
which is *why* both fall back to the difference. Recorded in the module and the
record so nobody re-discovers it; it matters at M5, when new HydroLight runs may
not share L23's water column.

For our purposes the L23 difference is not a fallback but the *correct* choice:
the model is trained against L23, so any other `bb_w` would put a bias straight
into `bb_p = bb − bb_w`. But both existing implementations take that difference at
an **arbitrary scene index** (bing: 0; ocpy: 170, commented "Random choie")
without checking that the choice is immaterial. So I checked: `bb − bbnw` is
constant to **1.6e-7 relative** (float32 storage noise) across all 3320 scenes,
all three solar zeniths, and both X=1 and X=4 — X=1 vs X=4 are bit-identical.
That makes a single 81-value table legitimate, so I embedded it: no data
dependency at import, which is what lets CI exercise the module, and two
`needs_l23` tests re-derive it from the netCDF and re-assert the
scene-independence so neither claim can rot.

**Tests (27).** The round trip is asserted in **both** dtype regimes — float32 to
1e-6 (measured 2.0e-7, so ~5× headroom) and float64 to 1e-12 (measured 2.6e-16) —
specifically so the float32 tolerance cannot later be tightened into a test of the
dtype, which is the trap M0's notebook §4 measured. `A_RRS`/`B_RRS` are asserted
equal to `bing.rt`'s under `importorskip` (CI has no bing): fixing the constants
only buys something if the package we share `rrs` with agrees. Both conversions
and `bb_w` are checked under `jit` and `grad` with derivative *signs* verified,
since they sit on the `forward` path. `bb_w`'s slope is fitted at λ^-4.2 against a
sanity band around Morel's -4.32 — a band, not a gate, because L23's water is
close to but not identical with pure molecular scattering. And the shifted-grid
test asserts the error message reports the offset, because a validator that says
only "bad grid" costs more time than it saves.

One consequence to note: `conventions.py` imports `jax.numpy` at module scope, so
`from robust import rt` now pulls JAX. M0 recorded that it pulled nothing and
predicted this would change at M2; it changed at M1. Expected, not a regression.

New: `ruff.toml`, `robust/rt/conventions.py` (implemented), `robust/tests/
test_conventions.py`. Modified: `robust/rt/__init__.py` (its docstring claimed
every module was a stub), `.github/workflows/ci.yml` (format check + stale
comment), `design/rt_elastic_implementation.md` (v0.6, new §3, lint conventions,
module index), and the 6 files `ruff format` touched. Branch
`rt-elastic-prototype` for JXP to commit. Next: task 2, the `types.py` pytrees.
