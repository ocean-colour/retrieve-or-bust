# Inelastic RT Coding — Prompt 4 (M3: Correction heads δ_R and δ_F)

## Goals

Implement **Milestone M3** of the coding plan
(`design/rt_inelastic_model_coding_plan.md`): the two bounded learned
corrections — δ_R on the Raman increment, δ_F on the φ_C-linear fluorescence
kernel — trained on the L23 X-difference truth channels, hitting the
per-process **5 % held-out gates at every zenith including 0°**. This is the
milestone that earns the design's accuracy: the analytic backbone alone fails
the Raman gate at high sun by −39 %.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`,
`code-review`) as helpful.

### Working agreements

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git (branch
`rt-inelastic-prototype`); `ocean14` on this machine, CPU JAX; every task
`pytest`-gated; the **elastic hash-regression stays green**. Use Fable if
you can. Log your work.

## Context

Read before coding:

- **Coding plan** — M3 section + the δ_R-at-0° and eutrophic-tail risk
  entries (they carry the escalation paths).
- **Design** — `design/rt_inelastic_model.md` §4.5 (features, bounded
  forms, size budget), §2 (how the corrections compose), §4.4
  (φ_C-linearity: δ_F must not see φ_C).
- **Elastic emulator** — `robust/rt/emulator.py` and the training pipeline
  conventions (relative weighting, seeds, committed weights) plus
  `design/py/train_emulator.py`.

## Status entering M3

*(Filled by M2's final task.)*

## Prompts

1. Read this doc. Execute the 1st task in the "M3" section below. If you have
   any questions, ask me in the Q&A section below. Use Fable if you can. Log
   your work.
2. Read this doc. Execute the 2nd task. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can.
   Log your work.
5. Read this doc. Execute the 5th task — modifying the next prompt doc,
   `rt_inelastic_coding_prompt_5.md`. Use Fable if you can. Log your work.

## M3

### Tasks

1. **Heads.** `robust/rt/inelastic_corr.py`: two small Flax MLPs with
   bounded (`tanh`-scaled) outputs, per design §4.5 —
   `f_R = 1 + (f_phys − 1)(1 + δ_R)` with per-(scene, λ) features
   `a(λ), bb(λ), a(λ′), bb(λ′), cos θ_s, λ`;
   `Rrs_fl = φ_C · K_fl · (1 + δ_F)` with features
   `a_ph(440), a(λ_em), bb(λ_em), a(490), cos θ_s, λ_em` and **no φ_C
   input**. Start minimal (O(10²) params); grow only if the held-out gate
   demands it. Wire into `forward` (weights loaded lazily from
   `robust/rt/files/`, absent-weights → analytic-only with a warning).

2. **Training.** `design/py/train_inelastic_corr.py`: Optax, fixed seeds,
   **train split only, all three zeniths**; targets = the X2/X1 and X4−X2
   channels; **relatively weighted** losses (the BING/elastic lesson —
   unweighted lets the red run away); monitor the a_ph(440)-decile
   diagnostic for the fluorescence tail during training. Train both heads;
   commit `robust/rt/files/raman_corr_l23.npz` and `fl_corr_l23.npz`.
   Record architectures, sizes, and training curves in the implementation
   record. As a *reported* diagnostic (not gated), also train a
   zenith-holdout variant (train 0°/30°, test 60°) to measure geometry
   generalization, mirroring the elastic CQ6 split.

3. **Held-out gates.** `robust/tests/test_inelastic_corr.py` on the
   **held-out scenes**, loading the *committed* weights (no
   train-at-test-time): Raman increment median |error| ≤ **5 %** over
   550–700 nm at each zenith **including 0°**; fluorescence 685 nm peak
   median |error| ≤ **5 %** per zenith; bounded-output property
   (corrections cannot exceed their tanh bound); elastic hash-regression
   still green. If the 0° Raman gate cannot be reached, follow the coding
   plan's escalation path and ask JXP in Q&A **before** relaxing anything.
   Update the implementation record.

4. **Notebook.** `notebooks/RT/rt_inelastic_coding_4.ipynb` — executed.
   Show: before/after error spectra per zenith for both processes (the
   −39 % @ 0° closing); the a_ph(440)-decile diagnostic; the zenith-holdout
   result with an honest caption; head sizes vs the elastic emulator's 417
   params.

5. **Finally.** Modify the next prompt doc,
   `rt_inelastic_coding_prompt_5.md`, given what M3 actually established.
   Log your work.

### Q&A

## Next

→ `rt_inelastic_coding_prompt_5.md` (M4: validation — prototype complete).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
