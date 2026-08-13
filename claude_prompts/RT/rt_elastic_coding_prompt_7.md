# Elastic RT Coding — Prompt 7 (M6: after the backbone broke)

## Goals

M5 tried to extend the forward model to the particle phase function and the full BRDF. It
did not get there, and the reason is the most useful thing this project has learned:

> **The ZTT backbone is being evaluated far outside the range its authors fitted, and the
> hybrid's bounded relative correction cannot repair that — not with a better network, not
> with a wider bound.**

M6's job is to act on that. **Read [`design/m5_report.md`](../../design/m5_report.md)
first** — one page, and it says what may and may not be claimed. This doc assumes it.

## Claude

### Skills

`.claude/skills/` — `critical-partner`, `grill-me`, plus the `dataviz` and `code-review`
conventions M0–M5 settled.

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).
Still **forward-model only** and **differentiable**; the inversion remains a separate
design.

The rhythm: **ask in Q&A** (numbered, phrased so work continues without an answer; check
before each task), then after the code a **notebook**, a **PR-review pass**, and a
**hand-off edit to the next prompt doc**.

## Status entering M6

**416 tests pass** (372 + 44 skipped without `$OS_COLOR`), ruff clean. `forward` is
**frozen** (record §8.0). `load_default()` is unchanged — the L23 model — and **no PB24
weights exist**, because M5's gate failed and `train_emulator_pb24.py` refuses to ship on a
failure.

### The three numbers M6 has to design around

| | |
|---|---|
| `Ψ_KLu(ψ)` crosses zero at | **110.4°** — ZTT's quartic is fitted for ψ ≳ 134° |
| PB24's sanctioned window below 134° / 110.4° | **42% / 16%** of geometries |
| ZTT's `rrs` that is zero or negative on PB24 | **22.3%** |

L23's minimum ψ is 139.7°, which is why none of this touches the Week-1 numbers and why
none of the Week-1 numbers say anything about off-nadir use.

### What is built and gated (M5's actual delivery)

```python
from robust.rt import conventions as C   # WaveGrid/GRIDS, SurfaceTransfer, interp_geometry,
                                         # fit_surface_transfer, default_transfer
from robust.rt import baselines as B     # O25Table, fit_o25_table (full geometry)
from robust.rt import emulator as E      # Envelope, fit_pb24, backbone_is_usable
from robust.rt import validation as V    # rrms(where=), group_rrms(expected=), default_steps
from robust.rt import ztt as Z           # P_bb_from_phase
from robust.rt.data import pb24 as P     # load_batch, make_splits, confound_reference
```

`PhaseParams` now carries `beta_tilde_pi` and `backward_slope` (**uncalibrated**).

### Gotchas carried forward

The M5 list still applies (see `rt_elastic_coding_prompt_6.md` → *Gotchas*). The four that
bit hardest:

1. **Score on everything, then explain.** Excluding the samples only *our* model cannot
   represent flatters us — O25 scores 9.46% on exactly the ones M5's first draft dropped.
2. **Refit per split.** M5's first version trained once and scored two splits; 75% of the
   second split's held-out realisations were in the first's training set.
3. **`seed 23` does not name a partition.** `make_splits` permutes whatever realisation set
   it is given, so the same seed on 200 and 400 realisations gives different held-out sets.
   The shipped surface transfer was fitted under one and is used under another.
4. **A causal story arrives with the observation and feels like part of it.** Twice in M5 a
   plausible mechanism was wrong (`bb/a` for 1% of an effect; "compensation" at r = −0.03).
   Write the mechanism down as a prediction, then test the prediction.

## Prompts

0. Read this doc and `design/m5_report.md`. Execute the 0th task — scoping. Ask your
   questions in the Q&A first. Use Fable if you can. Log your work.

## M6

### Tasks

0. **Decide what M6 is.** M5 closed off two of Q17's four options by measurement and left
   three live routes. They are different projects, and the choice is JXP's — so this task
   is scoping, and its deliverable is a task list in this doc plus a filled §9 of the
   record.

   **Route A — replace or repair the backbone.** The failing terms are `Ψ_KLu` (68% of the
   non-physical predictions) and, secondarily, `Md_star`/µ∞ (`bb/a` beyond 0.1 on 36% of
   PB24). Neither can be refit from PB24, because both are defined against the asymptotic
   `K∞` and **PB24 tabulates only surface K's** (report §3). So this route needs either a
   different analytic backbone whose validity domain covers the BRDF, or reference data
   that tabulates an asymptotic K. Commissioning HydroLight runs would settle both this and
   the VSF-family question M5 never reached.

   **Route B — change the correction's form (Q17 option 4).** The oracle says a bounded
   *relative* correction cannot pass; an additive or unbounded one might. This is the
   cheapest route and the one M5's data most directly supports, but `delta_max` exists to
   stop the emulator swamping the physics, and removing it means the "hybrid" becomes
   mostly emulator wherever the backbone is bad — which is a different scientific claim,
   not a tuning change.

   **Route C — restrict the claim and ship what works.** Declare the model valid where the
   backbone is (ψ ≳ 134°, `bb/a` ≤ 0.1), report the coverage honestly, and spend M6 on the
   inversion instead. M5 already built the machinery to express that envelope.

   Sequence by what each unlocks, gate each with a test as M0–M5 did, and say plainly which
   tasks are blocked and on what.

### Q&A

**Q18 (M5 → M6 hand-off, Claude → JXP). Which route, and is commissioning HydroLight runs
on the table?**

Route A is the only one that makes the model *right* rather than *narrower*, and it is the
one that needs a decision outside this repository: it wants either a different backbone or
new reference data. Routes B and C can start immediately.

My recommendation: **C now, B as an experiment, A as the project decision.** Restricting the
claim costs nothing and makes the current model honestly usable; the additive-correction
experiment is a few days and would settle whether the form is the whole story; and the
backbone question is where the real answer lives but is not ours alone to make.

The specific thing I would need for A: reference runs that tabulate an **asymptotic**
diffuse attenuation (or the asymptotic mean cosine directly), and that vary the VSF
*family* rather than only the Fournier-Forand parameter — which would close the one
headline gap M5 left untouched. — *Blocking: task 0's output depends on the answer.*

**Q19 (M5 → M6 hand-off, Claude → JXP). Does the prototype's headline claim need
restating?**

`prototype_summary.md` says the hybrid reaches 0.30% on held-out L23 water bodies. That is
still true and still reproducible. But M5 established that the fallback the summary
recommends off-nadir (`on_out_of_domain="ztt"`) degrades to a backbone that is non-physical
on 22% of a multi-angular dataset — so the summary's item 5 now understates the limitation.
I have added an update note there rather than rewriting the claim.

Worth your view on whether the prototype summary should be revised more strongly before it
goes to anyone outside the project. — *Not blocking.*

## Next

→ M6's own hand-off, when it closes.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
