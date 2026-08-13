# M5 — what PB24 revealed, and why the hybrid could not pass

**Branch `rt-elastic-prototype` · milestone M5 · 2026-08-13**

M5 set out to close the two axes the Week-1 prototype could not speak to — the **particle
phase function** and the **full BRDF** — using PB24, a 5000-realisation HydroLight release
with 1300 geometries per realisation.

It did not close them. **The milestone's central result is a negative one**, and it is
worth more than the positive one would have been: the analytic backbone the whole hybrid is
built on is being evaluated far outside the range its own authors fitted, and no bounded
relative correction to it can pass any honest gate on this data. This document sets out
that chain in full, as JXP asked (Q17), including the options *not* taken.

The full account is [`rt_elastic_implementation.md`](rt_elastic_implementation.md) §7
(v0.30); the Week-1 context is [`prototype_summary.md`](prototype_summary.md).

---

## 1. The headline

| | |
|---|---|
| **Gate** | Hybrid retrained on PB24 beats O25 refit on PB24, on the held-out-realisation *and* held-out-`B_p` splits (Q15) |
| **Result** | ❌ **FAILED**, on both splits, at every seed and both subsample densities |
| **Weights shipped** | **None.** `train_emulator_pb24.py` refuses to write when the gate fails |
| **`load_default()`** | Unchanged — still the L23 model (Q13's promotion rule, evaluated) |

Held out, full test set, 300 realisations, 3 seeds:

| δ_max | split | hybrid | **oracle** | O25 | gate |
|---|---|---|---|---|---|
| 0.5 | realisation | 5484.65% | **5324.10%** | 5.43% | FAIL |
| 0.5 | `B_p` band | 4052.36% | **3954.03%** | 5.37% | FAIL |
| 1.0 | realisation | 892.21% | **34.89%** | 5.43% | FAIL |

**The oracle column is the finding.** It is the correction chosen *with the truth in hand*
and then clipped to ±δ_max — a bound no emulator can beat. At the shipped δ_max = 0.5 the
trained hybrid sits **within 3% of it**. The network is not the limitation; the functional
form is.

## 2. The chain, in order

### 2.1 The backbone is outside its validity domain — this is the root cause

ZTT's `Ψ_KLu(ψ) = 1 + F(ψ)` uses a quartic in the in-water scattering angle. **`F_psi`'s
own docstring, written at M2, records the paper's fitted range: ψ ≳ 134°.**

- `Ψ_KLu` **crosses zero at ψ = 110.4°** and is negative below it, flipping the sign of the
  `(a/bb)(1 − cos θv · Ψ_KLu / µ∞)` term and hence of the whole ZTT denominator.
- In PB24's sanctioned window (Q14: θs, θv ≤ 70°), **42% of geometries have ψ < 134°**
  (extrapolation) and **16% have ψ < 110.4°** (sign-flipped). The full grid reaches
  ψ = 44.3°, where `Ψ_KLu = −60.6`.
- **22.3% of ZTT's predicted `rrs` on PB24 are zero or negative** — non-physical, not merely
  inaccurate.
- **L23's minimum ψ is 139.7°.** Nadir viewing pins the scattering angle near backscatter,
  so the Week-1 prototype could not have found this at any level of care, and its numbers
  are unaffected.

A second axis compounds it: `Md_star` and the TT2017 µ∞ are fitted for `bb/a` ≤ 0.1, which
**36% of PB24 exceeds** (the release reaches 20.1). There ZTT's `mu_d` reaches **−1.020** —
an average cosine, negative.

**This is not a transcription error.** Gershun's law (`a/Knet = mu_tot`) holds in PB24 to a
median **0.9999**, so the reference data and our IOPs are mutually consistent. The
disagreement is ZTT's, and it is a domain problem.

### 2.2 The hybrid's form cannot absorb it

The hybrid is `rrs = rrs_ZTT · (1 + δ)` with `|δ| ≤ δ_max = 0.5` — a **bounded relative**
correction. A negative backbone requires `1 + δ < 0`, so **no bounded relative correction
of any size can repair it**. The oracle quantifies the rest: 5324% at δ_max = 0.5, and
34.89% even at 1.0 — still 6× worse than O25.

### 2.3 The emulator learned nothing transferable

Trained on PB24 and applied to L23 — where the backbone is healthy (5.93%) — the correction
makes things **four to five times worse** (27.01% against 5.93%). It flags 100% of L23 as
out of domain and applies a median +21.6% where the backbone needs +2.4%.

The obvious reading, "it learned to compensate for a broken backbone", is **not supported**:
the correlation between its correction and what L23's backbone actually needs is
**−0.028**, against **+0.999** for the L23-trained model on the same data. There is no
relationship. Nor is it simple feature extrapolation — only `wave_nm` leaves the trained
range, and restricting to the 71 overlapping bands changes nothing.

## 3. Q17: the four options, and where each now stands

JXP chose option 1 now, option 3 as task 13's output, and kept 4 live. Measurement has
since moved all four.

| # | Option | Status after M5 |
|---|---|---|
| **1** | Restrict the sanctioned envelope where the backbone is unusable | **Implemented, and insufficient.** Training excludes non-physical samples and the coverage is reported. But the exclusion is *geometric* — the samples cluster at large θv and Δφ — so it narrows the geometry range too, and **O25 scores 9.46% on exactly the samples it drops**. Excluding them to make a gate pass would be self-flattery, so the benchmark scores everything. |
| **2** | Clamp µ∞ to a physical floor | **Would address ~1% of the problem.** µ∞ accounts for 1% of the non-physical predictions; `Ψ_KLu` for 68%. Not worth inventing physics for. |
| **3** | Refit µ∞ ourselves from PB24 | **Not possible.** µ∞ is the *asymptotic* mean cosine, `a/K∞`, and `K∞` is θs-independent by definition. PB24 tabulates seven K's and **all seven vary ~1.4× across solar zenith** — they are surface K's. The best proxy, `a/Kd`, is 100% physical and within 13.7% of TT2017, but it is a downwelling *surface* attenuation standing in for an asymptotic quantity; adopting it swaps a published parameterization for one whose error we cannot characterise. And `F(ψ)` is defined as `K_Lu/K∞ − 1`, so **refitting it needs `K∞` too**. |
| **4** | Change the correction's form | **Now the only route that M5's own data supports**, alongside replacing the backbone or obtaining reference data with an asymptotic K. |

## 4. What M5 did deliver

Nine pieces of infrastructure, each gated, none of which depends on the hybrid passing:

| | what landed |
|---|---|
| **conventions** | a second wavelength grid (`WaveGrid`, `GRIDS`), `bb_w` extrapolation modes, and a **geometry-aware surface transfer** — Lee's nadir constants are wrong by a median 33.6% at θv = 60°, and the fitted table is **7.2× better** there on held-out data |
| **data.pb24** | loader, three splits (realisation / `B_p` band / geometry), per-axis geometry subsampling, `LoadReport` |
| **baselines** | `O25Table` over the full `(θs, θv, Δφ)` grid — **1.67× better** than the θs-only refit, so the benchmark is no longer one we crippled |
| **validation** | `rrms` masking (double-`where`, NaN-safe gradients), `group_rrms(expected=)`, `gradient_report` generalised to any field |
| **emulator** | per-model `Envelope` carried with the weights, `fit_pb24`, `backbone_is_usable` |
| **ztt / types** | the backward-VSF axis (`beta_tilde_pi`, `backward_slope`) — **uncalibrated**, an axis to sweep |
| **API** | `forward` **frozen** (§8.0), with the numbers pinned by digest and golden value |
| **artefacts** | `design/validation_pb24/`, `robust/rt/files/surface_pb24.npz` |
| **tests** | 279 → **416** |

## 5. What may and may not be claimed

**May:** the surface transfer is 7.2× better off-nadir than the nadir constants, on held-out
realisations; O25's geometry table is 1.67× better than the zenith-only refit; PB24 varies
`B_p` by 12× against L23's 1.7×, so a held-out-phase-function split is now constructible.

**May not:**

1. **No accuracy claim on PB24.** The hybrid failed its gate, and no PB24 weights exist.
2. **The prototype's L23 numbers are untouched but narrower than they looked.** 0.30% on
   L23 stands. It says nothing about off-nadir use — and M5 now shows the
   `on_out_of_domain="ztt"` fallback is *unsafe* off-nadir, since the backbone it falls back
   to is non-physical on 22% of PB24.
3. **`B_p` generalisation is still untested in the end-to-end sense.** The split exists; the
   model that would have been scored on it failed for unrelated reasons.
4. **The backward-VSF parameters are uncalibrated.** No dataset here constrains them.
5. **O25's PB24 numbers are refits**, on its own calibration set, and must never be set
   beside its 0.69% on L23 — that would compare datasets and call it models.

## 6. Two contaminations noted and not fixed

- **The shipped surface transfer overlaps this milestone's held-out sets.** It was fitted on
  a 400-realisation split under `SPLIT_SEED = 23`; the PB24 benchmark splits 200 and the
  gate 800, and `make_splits` permutes whatever set it is given — so **"seed 23" does not
  name one partition**. 31 of the benchmark's 40 held-out realisations were in the
  transfer's training set. The transfer sits only in O25's scoring path, so the bias
  favours the rival and every hybrid FAIL is conservative; the residual is ~1.8% median.
  Unquantified rather than harmless, and it should be fixed by refitting the transfer on
  each consumer's own train mask.
- **`B_p` span is quoted at three sample sizes** across the milestone (6.2× from ~50 files,
  9.7× from 200, 12.4× from 600). Each is honest at its own size; none said so. The
  dataset-level figure is the largest sample's.

## 7. Three defects an audit found in my own work

Recorded because the pattern matters more than the individual fixes.

1. **A leak.** The emulator was fit once on the realisation split and scored on the `B_p`
   split too — **75% of whose held-out realisations sit in that training set**. It would
   have been training error presented as the milestone's headline generalisation result.
2. **A comparison that flattered us** — scoring both models only where *our* backbone is
   physical (§3, option 1).
3. **A handicap on the rival** — O25 fitted in `Rrs` and converted, paying a surface-transfer
   residual the emulator never pays (5.43% vs 5.74% fitted directly in `rrs`).

And two causal stories asserted before being tested, both wrong: ZTT's collapse blamed on
`bb/a` (1% of it) rather than `Ψ_KLu` (68%), and the cross-dataset failure called
"compensation" when the correlation is −0.03. **A plausible mechanism arrives with the
observation and feels like part of it.** Both were caught by writing the mechanism down as a
prediction and testing it.

## 8. Where the reproducible numbers come from

```bash
pytest -q                                       # 416 tests; 44 skip without $OS_COLOR
python design/py/fit_surface.py                 # the surface transfer table
python design/py/run_pb24_validation.py         # design/validation_pb24/
python design/py/train_emulator_pb24.py         # the gate — exits 1, ships nothing
python design/py/cross_dataset.py               # PB24-trained model on L23
```
