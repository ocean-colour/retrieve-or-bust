# Radiative Transfer (RT)

This prompt doc will guide our efforts to develop radiative transfer (RT) models for ocean color.

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

## Code

If you need to run Python use the `ocean14` conda environment. 

## Prompts

### Setup

1.  Let's start from the work done by JXP in the BING repo.  Please review that work. We will write a report named `context/rt_bing_summary.md`.  Before doing so, please have a conversation with me in Q&A/Setup.  Use Fable if you can.  Log your work. 


## Q&A

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

A1:

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

A2:

**Q3 — Open threads.** How much emphasis on the unclosed validation gaps
(held-out set, IOP-retrieval impact, cross-dataset stability of G0/Gb)?
  (a) Foreground them as the central to-do for retrieve-or-bust — the
      forward-model rRMS wins are an optimistic in-sample ceiling until
      validated.
  (b) Note briefly as caveats; keep the focus on documenting what was done.

A3:

**Q4 — Plumbing depth.** How deep should the report go on the package wiring?
  (a) Findings-focused — science/results; mention flags and CSVs only enough to
      reproduce, not the file-by-file plumbing.
  (b) Include a plumbing map — document how it's wired (`variable_Gordon_G0` /
      `variable_Gordon_bbp` flags, the four CSVs, and the
      `rrs.py`/`anw.py`/`l23.py`/`evaluate.py` code paths) so RT can build on
      the code directly.

A4:

**Q5 — Anything else.** Is there context I'm missing — e.g. the deeper JXP
"deep-dive" finding on wavelength dependence and bb-sensitivity of the G_i that
rt_prompts.md alludes to, or a direction you already want RT to take beyond the
Gordon approximation (full RT, learned forward model, etc.)?

A5:

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
