# HydroLight runs — a HOWTO for the operator

**For:** Henry Houskeeper · **From:** J. Xavier Prochaska · **Date:** 2026-09-23

*This is the short version. The full specification is
[`design/hydrolight_runs.md`](hydrolight_runs.md); you should not need it unless
something here is ambiguous.*

---

## What this is, in one paragraph

We are building a fast, differentiable forward model that predicts
remote-sensing reflectance `Rrs` from inherent optical properties, so it can be
inverted to retrieve IOPs from ocean colour. The two HydroLight datasets we have
— Loisel et al. 2023 and Pitarch & Brando 2024 — between them cover **none** of
the combinations we actually need: L23 has the inelastic processes but only
nadir viewing, three solar zeniths and one fixed phase function; PB24 has the
full viewing geometry but no Raman, no fluorescence, and only one phase-function
family. We need one dataset that crosses those axes, and that is what we are
asking you to produce.

**You will not have to write any input decks.** We generate them. What we need
from you is your HydroLight, some setup advice, and the compute.

---

## Where we are: three stages

| | | your effort | status |
|---|---|---|---|
| **Stage 0** | Answer four questions. No compute. | ~30 minutes | **← we are here** |
| **Stage 1** | The pilot: **172 runs** | ~a day of wall-clock | waiting on stage 0 |
| **Stage 2** | The campaign: **~156,000 runs** | weeks of wall-clock | waiting on stage 1 |

**Please do not run anything yet.** Until we have your answers to stage 0 we
cannot write decks your HydroLight will accept, and one of the four answers
could change the size of the whole campaign by a factor of 130.

---

## Stage 0 — four answers we need before anything runs

These four block us. The other six at the end are useful but can wait.

### 1. Does one run give you the whole radiance distribution?

Our entire sizing assumes that **one HydroLight run returns the upwelling
radiance at every view direction at once**, so that only the *solar* zenith and
the inelastic switches multiply the number of runs:

> runs = water bodies × solar zeniths × scenarios

If instead your workflow needs one execution per view direction, the campaign is
about **130× larger** than we have specified and we need to rethink it before
committing you to anything. A one-line answer is enough.

### 2. Can we send you one example input deck — yours, from a working run?

We do not know your version's exact deck syntax, so we have deliberately **not**
guessed at it. Everything we are shipping is format-neutral: tabulated IOPs and
phase functions, plus a plain-text description of each run. One real deck from
your setup, with whatever it produced, lets us write the translator in an
afternoon.

Related: **will you accept user-supplied IOP files** — tabulated `a(λ)`, `b(λ)`
and a discretised `β̃(ψ)` per water body — rather than concentration-driven
inputs? Every water body in this campaign is specified that way. If that is not
how your setup runs, tell us now, because it changes the shape of everything.

### 3. Your Raman and fluorescence data files, verbatim

Three internal parameterisations live in HydroLight's own data files rather than
in any deck, and we need the actual files (or their contents) to make our
analytic terms match your truth:

- the Raman scattering coefficient and its wavelength-redistribution function;
- the chlorophyll fluorescence quantum efficiency and emission shape — band
  centre, width, and whether any PS I shoulder is included;
- **specifically, which published Hawes variant your CDOM-fluorescence option
  uses, and its constants.**

The last one matters more than it looks. Published Hawes constants exist in
several variants (fulvic vs humic, plus HydroLight's own default). If ours and
yours differ, every CDOM-fluorescence number we derive is quietly wrong in a way
no test would catch.

### 4. Your exact version and build string, and any local modifications

One line, recorded permanently with the data.

---

## Stage 1 — the pilot: 172 runs

Once stage 0 is answered we will send real decks. Here is what the pilot is, so
you can see where it is going.

**Why a pilot at all:** it answers three things that cannot be answered after
the compute is spent — whether our decks reproduce a dataset we already have,
what HydroLight's own numerical error is at your chosen settings, and whether the
file format works end to end. 172 runs is cheap insurance against finding a
convention mismatch after 156,000.

### The three groups

| group | water bodies | runs | what it is for |
|---|---|---|---|
| **P1** | 10 | **90** | **The reproduction test.** These are L23 water bodies, rebuilt from L23's own published IOPs. We compare your `Rrs` against L23's published `Rrs` for the same scenes. If they agree, every convention below is validated at once. If they do not, we learn *which* one is wrong while it still costs 90 runs to find out. |
| **P2** | 5 | **70** | **The convergence sweep.** The same run repeated under seven solver configurations — a baseline, plus quad resolution, depth-grid resolution and inelastic-source iteration count each moved coarser and finer. **Please choose the actual settings.** Nobody has ever recorded HydroLight's own numerical error for this project, and we quote accuracy figures of 0.3 % against it. |
| **P3** | 6 | **12** | **The corners.** The extremes of our IOP grid, at solar zenith 0° and 80°, to exercise the user-supplied-IOP path at its limits and the grazing shell. |

### Conventions

| | |
|---|---|
| **wavelengths** | 330–750 nm, 5 nm bands (85 bands). The 350–750 part matches L23's grid exactly, which is what makes P1 a like-for-like test. |
| **solar zenith** | listed per water body; the campaign grid is 0–80° in 10° steps. 0–70° is the sanctioned window; 80° is a deliberate extrapolation shell. |
| **view geometry** | **the standard quad layout — no custom quad file.** Please output the **full** upwelling radiance distribution. |
| **water column** | homogeneous, optically deep, no bottom. |
| **sky** | clear, HydroLight's semi-empirical model, wind **5 m s⁻¹**. If you would set this up differently, say so — we would rather match your normal practice. |
| **precision** | **float64 on radiometric fields, please.** Single precision underflows `rrs` to exactly zero at grazing geometries, and our error metric divides by it. |

### Scenarios

We name these by switch rather than by L23's "X" label, so nothing depends on a
convention someone has to remember:

| | Raman | Chl fluorescence | CDOM fluorescence | = L23 |
|---|---|---|---|---|
| **S1** | off | off | off | X = 1 |
| **S2** | **on** | off | off | X = 2 |
| **S4** | **on** | **on** | off | X = 4 |
| **S5** | **on** | **on** | **on** | *(new)* |

Chlorophyll fluorescence quantum yield is **φ_C = 0.02** everywhere except one
later subset that varies it deliberately.

### What we need back from each run

- `Rrs` (above water) **and** `rrs` (just below the surface) — both, and please
  state explicitly which is which rather than leaving us to infer it from
  magnitudes;
- `Lu(0⁻)` at all quads — the full radiance distribution;
- `Ed(0⁺)`, `Ed(0⁻)`, `Eu(0⁻)`;
- `K_d`, `K_u`, `K_Lu`, and the mean cosines `μ_d`, `μ_u`, `μ_tot`;
- **the IOPs as you received them**, echoed back. Not redundancy — it is how we
  verify that the deck we wrote is the deck that ran.

For the **P3** bodies only, additionally: `K_d(z)`, `K_u(z)`, `K_Lu(z)`, `μ̄(z)`
on a depth grid reaching **at least 25 optical depths**, and the **asymptotic
radiance distribution and `K∞`** if your build reports them. (That last one is
question 6 below — we would like to know either way.)

---

## Stage 2 — the campaign, so you can price your time

| batch | runs | what it is |
|---|---|---|
| **A** | 16,092 | An IOP grid crossed with 26 phase-function designs, elastic only. Homogeneous water, with the depth profiles above. |
| **B** | 130,140 | 4,820 water bodies × 9 solar zeniths × S1/S2/S4. This is the bulk. |
| **C** | 9,720 | Varied fluorescence yield, CDOM fluorescence on/off pairs, and a small vertically structured set. |
| | **~156,100** | about three times the compute that produced PB24 |

Our guess at the wall-clock is **7 to 34 days on 16 cores**, depending entirely
on what a run costs you — which is question 5 below, and the number that turns
this table into a date. Delivered volume is **310–780 GB**.

We would like it **staged**, not delivered all at once: pilot → batch A →
batch B's first block → batch B's second block → batch C. Each is self-contained,
and we can start work on each while the next runs.

---

## The other six questions

Useful, but they do not block you.

5. **What does a run cost** in wall-clock — one 85-band elastic run, and one
   with the inelastic sources on — and how many can you run in parallel?
6. **Does your build report the asymptotic radiance distribution and `K∞`** for
   homogeneous, optically deep water, and under which switch? If not we will read
   the asymptote off the depth profile instead, which we are requesting anyway.
7. **Can the band set start at 330 nm**, and does your Raman implementation
   handle excitation falling below the band-set floor by clipping or by
   extrapolating?
8. **Which sky model** would you normally use, and is a fixed 5 m s⁻¹ wind
   acceptable?
9. **Disk and transfer** — what can you produce, and how do we take delivery?
10. **Anything in this document you would do differently.** You run HydroLight and
    we do not; if a convention here is unusual or a request is awkward, we would
    much rather change it now.

---

## What we have sent you

`HH26_batch0.tar.gz` — 161 files, 134 kB. It contains the pilot as we currently
have it, so you can see exactly what is coming:

| | |
|---|---|
| `README.txt` | the same conventions as above, in plain text |
| `manifest.json` | every water body, every convention, and the metadata we still need from you, marked as pending |
| `ir/*.json`, `ir/*.npz` | one run description and one IOP table per water body — `a`, `b`, `bb`, `b_p` on the 85-band grid, float64 |
| `vsf/*.txt` | tabulated `β̃(ψ)` on 740 angles, normalised |
| `provisional/*.txt` | a readable description of each run — **these are not decks**, and are labelled as such |

There are no input decks in it yet, on purpose: see question 2.

---

## Summary — what to do now

1. Read this page.
2. Answer questions **1–4**. A short email is fine.
3. Send **one example deck and its output** from a run you have already done.
4. Do not run anything yet.

We will turn that around into real decks quickly, and the pilot can start.
