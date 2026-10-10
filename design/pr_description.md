# PR: `rt-elastic-prototype` → `main` — M5 (PB24, BRDF) and Route C

M5 took the elastic forward model to a multi-angular reference dataset to close
the two axes the Week-1 prototype could not speak to: the **particle phase
function** and the **full BRDF**.

**It did not close them, and that is the headline.** The milestone's central
result is negative, it is load-bearing, and everything else in this PR follows
from it:

> The ZTT backbone is evaluated far outside the range its own authors fitted as
> soon as the sensor leaves nadir, and no bounded *relative* correction to it can
> pass an honest gate on that data.

`Ψ_KLu(ψ)` is a quartic fitted for ψ ≳ 134°; it crosses zero at **110.4°** and is
negative below. Inside PB24's own sanctioned 0–70° window, 42 % of geometries are
extrapolation, 16 % are sign-flipped, and **22.3 % of ZTT's predicted `rrs` are
zero or negative**. The hybrid is `rrs_ZTT · (1 + δ)` with `|δ| ≤ 0.5`, so
`1 + δ` can never reach a negative backbone. The oracle correction — chosen *with
the truth in hand*, then clipped — scores 5324 % where the trained model scores
5484 %: the trained model is within 3 % of the best its functional form permits.
**The network is not the limitation.** No PB24 weights exist;
`train_emulator_pb24.py` refuses to write on a failed gate, and it still exits 1.
`load_default()` is unchanged and remains the L23 nadir model.

L23's minimum ψ is 139.7°, because nadir viewing pins the scattering angle near
backscatter. **No number `main` already publishes moves in this PR.**

---

## What the branch brings

**Nine pieces of gated infrastructure**, none of which depended on the hybrid
passing:

| | |
|---|---|
| `robust/rt/data/pb24.py` (1551 lines) | loader, three held-out splits (realisation / `B_p` band / geometry), per-axis geometry subsampling, `LoadReport`, `fit_transfer` |
| `conventions` | a second wavelength grid (`WaveGrid`/`GRIDS`, the 12 OLCI bands), `bb_w` extrapolation modes, and a **geometry-aware surface transfer** — Lee's nadir constants are wrong by a median ~33 % at θ_v = 60°, and the fitted table is **7.2× better** there on held-out realisations |
| `emulator` | a per-model `Envelope` carried with the weights, `fit_pb24`, `backbone_is_usable` |
| `baselines` | `O25Table` over the full `(θ_s, θ_v, Δφ)` grid — **1.67× better** than the θ_s-only refit, so the benchmark is no longer one we crippled |
| `validation` | `rrms` masking (double-`where`, NaN-safe gradients), `group_rrms(expected=)`, `gradient_report` generalised to any field |
| `ztt` / `types` | the backward-VSF axis (`beta_tilde_pi`, `backward_slope`) — **uncalibrated**, an axis to sweep |
| API | `forward` **frozen** (`test_env.py`), re-baselined once here to admit `main`'s keyword-only `inelastic` / `corrections` and the `None`-defaulted `a_ph` / `a_cdom` / `Ed` fields — all of which the freeze's own rules permit |
| artefacts | `design/validation_pb24/`, `robust/rt/files/surface_pb24.npz` |
| docs | a `pb24` API page, `docs/model/surface.md`, and the two limitation sections below |

**A merge of five weeks of `main`.** The branch forked at PR #12 (2026-08-06) and
M5 was never PR'd back, while `inelastic-rt`, `cdom-rt` and the docs sweep edited
the same four modules. Nine files conflicted; all were additive. One deserves a
reviewer's eye: `validation.gradient_report` — `main` refactored its comparison
loop into a shared `_grad_vs_fd`, M5 rewrote the setup above it, and both were
kept.

**A contamination fixed, not just recorded.** `surface_pb24.npz` was fitted under
one 400-realisation split while its consumers split 200 and 800 — and
`make_splits` permutes whatever set it is given, so `SPLIT_SEED` names a
*procedure*, not a partition. 31 of one consumer's 40 held-out realisations had
been in the table's training set. `pb24.fit_transfer(batch, train)` is the fix and
both consumers now call it. The bias favoured the rival, so the conclusions
survived — but "survived" is not "measured", and the numbers are now measured:
the held-out score moves 6.53 → 6.48 and the **geometry split 22.46 → 24.64**,
the leak closing. The packaged coefficients themselves were never wrong —
regenerating reproduced `A` and `B` bit-identically, and only the provenance
string changed.

## What it does not claim

1. **No accuracy claim on PB24.** The gate failed and no weights exist.
2. **The L23 numbers are untouched but narrower than they looked.** 0.30 % on
   L23 stands and still reproduces. It says nothing about off-nadir use — and M5
   shows the `on_out_of_domain="ztt"` fallback is *unsafe* there, because the
   backbone it degrades to is non-physical on 22 % of PB24.
3. **`B_p` generalisation is still untested end to end.** The split exists; the
   model that would have been scored on it failed for unrelated reasons.
4. **The backward-VSF parameters are uncalibrated.** No dataset here constrains
   them.
5. **O25's PB24 numbers are refits** on its own calibration set, and must never
   be set beside its 0.69 % on L23 — that would compare datasets and call it
   models.

## Route C, executed

JXP's answer to Q18 was "C now, B as an experiment, A as the project decision",
and C is done here: the model declares where it is valid and reports the coverage
without flattering it (`design/prototype_summary.md`, `docs/using/limitations.md`).

The measurement that shaped the wording is worth surfacing, because the literal
version of Route C would have backfired:

| | L23 (nadir) | PB24 (0–70°) |
|---|---|---|
| ψ ≥ 134° | **100 %** of samples | 58.1 % |
| `bb/a` ≤ 0.1 at every band | 25.0 % of samples | 6.7 % |
| **backbone physical** | **100 %** of samples | 75.0 % |

"Valid where ψ ≳ 134° **and** `bb/a` ≤ 0.1" would declare **75 % of L23 out of
envelope**, including most of the data the 0.30 % is measured on — and it would
be wrong, since the backbone is physical on 100 % of L23 regardless. So ψ is
stated as a boundary and `bb/a` as a disclosure, with the asymmetry argued rather
than asserted.

Scope, per PQ8: **all processes, nadir, L23-like water.** Elastic scattering,
Raman, and chlorophyll fluorescence compose in one differentiable call — with the
sensor at nadir. This is not a BRDF model.

## What M6 is now

Route C is closed, so M6 is **Route A: the reference data**. M5 established what
it has to tabulate, and Robert Frouin's review asks for the same thing from the
other direction — *"use a full radiative-transfer solver (most naturally
HydroLight) as the reference forward model, with particle phase-function
parameters explicitly varied."*

Two requirements, both measured rather than assumed:

- **An asymptotic K**, or the asymptotic mean cosine directly. µ∞ is `a/K∞` and
  `K∞` is θ_s-independent by definition; PB24 tabulates seven diffuse-attenuation
  coefficients and **all seven vary ~1.4× across solar zenith**, so none is `K∞`
  and µ∞ cannot be refit from it. `F(ψ) = K_Lu/K∞ − 1`, so repairing the quartic
  needs `K∞` too.
- **The VSF *family* varied**, not only the Fournier–Forand parameter — the one
  headline gap M5 never reached.

Route B — changing the correction's form from bounded-relative to additive or
unbounded — remains available as a short experiment, and is the only route M5's
own data directly supports. It is deliberately not in this PR: removing `δ_max`
makes the "hybrid" mostly emulator wherever the backbone is bad, which is a
different scientific claim, not a tuning change.

## Verification

```bash
pytest robust/tests              # 677 passed, 5 skipped
ruff check robust/ design/py/    # clean
ruff format --check robust/      # clean
cd docs && make html             # build succeeded, -W --keep-going, nitpicky
```

The 5 skips are principled: three machine-anchored bitwise hash pins
(`ROBUST_HASH_ANCHOR` is `tank`/`mac`), one removed fallback path, one undefined
azimuth at the poles. **No data skips** — both PB24 and L23 were present, so the
suite genuinely exercised them.

Regenerating the artefacts:

```bash
python design/py/fit_surface.py            # the shipped surface table
python design/py/run_pb24_validation.py    # design/validation_pb24/
python design/py/train_emulator_pb24.py    # the gate — exits 1, ships nothing
python design/py/cross_dataset.py          # PB24-trained model on L23
```

`design/validation/` (the elastic and inelastic tables) is **unchanged in this
PR** and deliberately so: those numbers were generated on the `tank` server, the
accuracy metrics reproduce here to the 4th decimal, and the only quantities that
actually move are wall-clock timings. Re-running on this laptop would replace
tank's numbers with a MacBook's for no informational gain.

## Reviewer's note

One judgment call worth a second opinion. The breakdown figures in
`run_pb24_validation.py` now fit the surface transfer in-window — matching O25's
coefficients, which were *already* fitted `train & in_window` — and then score
across the full angle range. O25 therefore looks worse past 70°: 17.52 → 24.25 at
θ_v = 80°. The mechanism was tested rather than assumed: restricted to the
sanctioned 0–70° window the old and new tables agree to **≤ 1 % at every
wavelength**, and the entire divergence lives in the shell. The previous state
was incoherent (in-window coefficients, all-geometry transfer), but this is a
rival's number moving, and `m5_report.md` §7's third recorded defect was exactly
that — so it is flagged rather than buried.
