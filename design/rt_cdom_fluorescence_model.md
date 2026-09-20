# Design — CDOM Fluorescence Term (companion to the Inelastic RT Forward Model)

*A short, buildable plan for adding the third inelastic process — CDOM
fluorescence — to retrieve-or-bust's forward model as an additive, analytic,
differentiable emission term with a defined-but-untrained correction head.*

**Companion document:** [`design/rt_inelastic_model.md`](rt_inelastic_model.md),
the finished, gate-passed inelastic design whose architecture, conventions,
Ed module, composition law, and correction-head machinery this document
inherits by reference. This is a **separate** document by decision (CFQ1,
`claude_prompts/RT/rt_inelastic_prompts.md`, Q&A/CDOM, 2026-08-29): the
inelastic design is a finished decision record — its §6 gate has been declared
PASSED and the shipped report cites it as-is — and amending it retroactively
would muddy that record. Its §8 reserved this term's landing zone
(`Inelastic.cdom_fl` + the additive-term pattern of its §4.4); this document
fills it.

**Date:** 2026-08-29. **Authors:** J. Xavier Prochaska and Claude (Fable 5).

Decisions locked in Q&A/CDOM (CFQ1–CFQ8, 2026-08-29): standalone doc (CFQ1);
**analytic term + interface now, head training gated behind truth arrival**
(CFQ2 — HydroLight CDOM-fl runs will eventually exist, but not before this
lands on `main`); the Chl-fl term's exact shape — additive, physics backbone
plus bounded head, head *defined* now but shipped untrained (CFQ3); the
**Hawes et al. (1992)** quantum-efficiency functions as the physics basis,
source ∝ a_CDOM(λ′), a new optional `IOPs.a_cdom` field, hard 350 nm
excitation clamp (CFQ4); `cdom_fl=None` default **even inside `Inelastic()`**,
a small `CDOMFl(scale=1.0)` pytree when set, extended bit-identity regression
(CFQ5); the truth-less v1 acceptance gate of §5 (CFQ6); a ~1–2 day **M5** plus
a deferred **M6**, executed by a single prompt doc (CFQ7); the HydroLight run
request of §7, commissioned jointly with the geometry runs (CFQ8).

---

## 1. Goals and non-goals

**Goals (v1 = milestone M5).**
- Implement the **analytic CDOM-fluorescence emission term** — a JAX,
  differentiable, additive `Rrs_cdom(λ)` built on the Hawes et al. (1992)
  spectral quantum-efficiency functions (the HydroLight-native
  parameterization, so any future truth is Hawes-consistent by construction).
- Ship the **complete interface**: `IOPs.a_cdom`, the `CDOMFl` pytree, the
  composition into `forward()` — so that when truth arrives, only training
  remains.
- **Default off, provably.** The shipped X4 truth *omits* CDOM fluorescence,
  so the report's 0.34 % gate and every claim built on it remain valid only if
  the default model is CDOM-fl-free. `cdom_fl=None` stays the default even
  inside `Inelastic()`, and `Inelastic(..., cdom_fl=None)` must be
  **bit-identical** to the current shipped inelastic output.
- Preserve every inherited property: differentiable end-to-end (now including
  ∂Rrs/∂scale), batched, within the speed budget (originally the inherited 2×
  elastic; rescoped to a machine-anchored 2.6× at M5 — see §5 item 5).

**Non-goals (v1).**
- **No correction-head training.** No CDOM-fl truth exists anywhere in hand —
  not in L23 (omitted by design) and not in BING (which has no CDOM-fl
  implementation at all, so there is also no fixed-BING reference to
  cross-check against). The head δ_C is *defined* (zero-initialized, so
  untrained head ≡ analytic backbone) but ships untrained; inventing
  pseudo-truth would defeat the point. Training is M6, blocked on the §7 runs.
- No quantitative rRMS gate in v1 (impossible without truth; written in §5 as
  M6's gate, explicitly conditional).
- No sub-350 nm excitation (§3, the UV clamp); no EEM shape retrieval; no
  changes to the Raman/Chl-fl terms or their trained heads.

## 2. Architecture

CDOM fluorescence takes exactly the Chl-fl term's shape (inelastic design
§2/§4.4): an **additive** emission term with a physics backbone and a bounded
correction head,

```
Rrs_cdom(λ) = s_C · K_cdom(IOPs, Ω, λ) · (1 + δ_C)
```

composed as

```
Rrs_total(λ) = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl + Rrs_cdom
```

- **`K_cdom`** is the analytic kernel: excitation integral of the CDOM source
  `b_C(λ′) ∝ a_cdom(λ′)` weighted by `Ed(λ′)` and the Hawes spectral
  fluorescence quantum-efficiency function `η(λ′, λ)`, with the same two-flow
  emission transport, per-λ_em attenuation, quanta→energy bookkeeping,
  **L_u = E_u/π** normalization, and A·rrs/(1−B·rrs) conversion the validated
  Chl-fl kernel uses (inelastic design §4.4). The Ed module (§4.2 there) is
  reused unchanged.
- **`s_C`** is a differentiable amplitude (`CDOMFl.scale`, default 1.0) on the
  fixed Hawes reference kernel — the φ_C-analogue handle for the eventual
  inversion.
- **`δ_C`** is a bounded (tanh-scaled) correction head, defined now with
  zero-initialized weights so the untrained head is exactly the analytic
  backbone — the same decay-to-physics property the δ_R/δ_F heads have. It
  trains only in M6.

**Physics basis (CFQ4): Hawes et al. (1992).** CDOM emission is broad and
featureless — no 685 nm-style line — so the Chl-fl single-Gaussian machinery
does not transfer. The Hawes functions parameterize the full excitation →
emission redistribution `η(λ′, λ)` (for each excitation wavelength, an
emission distribution that is approximately Gaussian in wavenumber with
center, width, and amplitude depending on λ′), and they are what HydroLight
itself implements — so the §7 truth runs and this kernel share constants by
construction, the same controlled-experiment property that made the X2/X4
channels clean. v1 fixes the kernel to a single published Hawes function
(HydroLight's default choice; the exact function/version is recorded in the
code and in the §7 run request so model and truth match).

**The `a_cdom` input.** The source term is proportional to **a_CDOM(λ′)**,
not a_ph — a physical requirement. `IOPs` grows an optional `a_cdom` field,
mirroring the `a_ph` pattern exactly (optional, elastic path ignores it,
required when the process is on). L23 stores a_g separately from a_nap, so
the existing loaders can populate it.

**The 350 nm UV clamp.** CDOM excitation extends into the UV below the 350 nm
edge of the L23/IOP grid — a sharper version of the Raman excitation clamp.
v1 imposes a **hard 350 nm lower limit** on the excitation integral. The
fraction of emission thereby truncated is **quantified from the Hawes
functions at implementation time** (a committed diagnostic, reported per
emission wavelength) and documented as a caveat wherever the term's output is
quoted. Removing it requires excitation-side IOPs below 350 nm (inelastic
design §8, wishlist item 6).

## 3. Interface

Extensions to the pinned API (all backward compatible; the `a_ph`/`Inelastic`
precedents apply verbatim):

- **`IOPs.a_cdom`** — optional `Spectrum`, default `None`. Validated like
  `a_ph` (non-negative, ≤ a, shape-matched). Elastic, Raman, and Chl-fl paths
  ignore it; CDOM-fl requires it (clear error naming the field if absent).
- **`CDOMFl` pytree** — `CDOMFl(scale=1.0)`: `scale` a differentiable leaf
  (the amplitude s_C on the Hawes reference kernel), with room for shape
  metadata (static fields) once truth exists. A pytree, not a bare scalar, so
  M6 can grow it without an API break.
- **`Inelastic.cdom_fl`** — the reserved slot, currently typed
  `Scalar | None` with a validator that rejects non-None. M5 retypes it to
  `CDOMFl | None` and accepts an instance. **`None` stays the default even
  inside `Inelastic()`** — load-bearing (§1 goals). Setting it with
  `iops.a_cdom is None` is an error.
- **Bit-identity regression (extended).** Alongside the elastic pin,
  `forward(..., inelastic=Inelastic(..., cdom_fl=None))` must be
  **bit-identical** to the current shipped inelastic output (hash-level on the
  dev machine, ULP-closeness tier on CI, per the established two-tier
  pattern). The CDOM-fl branch must be unreachable — no-op by construction,
  not by arithmetic — when the slot is `None`.

## 4. Components and layout

Same stack and conventions as the inelastic effort (JAX, `robust/rt/`,
committed fixtures, ruff). Planned surface:

| module | contents |
|---|---|
| `robust/rt/types.py` (extend) | `IOPs.a_cdom`; `CDOMFl` pytree; `Inelastic.cdom_fl` retyped |
| `robust/rt/cdom_fl.py` (new) | Hawes η(λ′, λ) kernel + constants; `K_cdom`; truncated-fraction diagnostic |
| `robust/rt/hybrid.py` (extend) | `+ Rrs_cdom` composition when `cdom_fl` is set |
| `robust/rt/inelastic_corr.py` (extend) | δ_C head definition (zero-init; training entry point stubbed for M6) |
| `robust/rt/data/l23.py` (extend) | `a_cdom` (a_g) extraction alongside `a_ph` |

The Hawes constants live in the code with their provenance (function/version,
source table) recorded — the same constants the §7 run request names.

## 5. Validation and acceptance gate

**v1 gate (M5 — truth-less by necessity; CFQ6).** The DQ6-style held-out rRMS
gate is impossible (no truth), and so is an M2-style rtol ≤ 1e-6 BING
cross-check (no BING implementation exists). M5 passes when all of:

1. **Off-state bit-identity**: `cdom_fl=None` (the default) ⇒ output
   bit-identical to the current shipped inelastic model, alongside the
   existing elastic pin (§3).
2. **Implementation-correctness pins**: the Hawes η(λ′, λ) function reproduced
   against its published values; energy/quanta bookkeeping unit tests;
   excitation-quadrature convergence under grid refinement.
3. **Literature-plausibility band** (reported and gated loosely): the CDOM-fl
   contribution on L23 IOPs lands in the published range — a few % of Rrs in
   the blue-green for CDOM-rich scenes, ≲ 1 % oligotrophic — and is monotone
   in a_g(440).
4. **Gradients**: central-difference checks pass for all inputs including the
   new `scale` amplitude (and `a_cdom`).
5. **Speed**: the composed forward with CDOM-fl on stays within the runtime
   budget — **rescoped at M5 task 8 from the original 2× elastic to a
   machine-anchored 2.6× elastic** (Q&A CQ3, 2026-08-30, JXP: "Go ahead and
   rescope the budget and make note that it is machine-anchored"). The
   measured facts behind the rescope: on JXP's Mac the everything-on forward
   reproducibly measures 2.26–2.34× (2.45× under load), with the CDOM
   *marginal* only ~0.3–0.4× elastic — most of the overage is baseline drift
   on that machine (the shipped Raman+Chl-fl model measures ~1.9× there vs
   its M4-recorded 1.59×). The 2.6 bound is **machine-anchored** in exactly
   the strict-SHA-256-hash-pin sense: it characterizes *that machine's*
   measured behavior with headroom, not a portable physical requirement — a
   different machine (e.g. the tank server that anchored the M4 speed
   record) may reproduce a different, possibly tighter, ratio. The shipped
   M4 model's own 2× gate (`validation.INELASTIC_GATE_SPEED`) is untouched;
   the rescoped constant lives separately as
   `test_cdom_validation.CDOM_GATE_SPEED_MACHINE_ANCHORED`.

**M6 gate (deferred — conditional on truth arrival).** Written now, armed
later: with HydroLight "X4 vs X4+CDOM-fl" pairs in hand (§7), train δ_C on the
difference channel and gate at the established per-process bar — **median
|error| of the CDOM-fl delta ≤ 5 %** on held-out scenes at every zenith, plus
the total-Rrs rRMS gate re-verified with the term on. Until those runs exist,
no quantitative accuracy claim is made for this term; the §5.3 plausibility
band is explicitly *not* a validation.

## 6. Milestones

Scope is ~1–2 days (CFQ7), not another week-long arc — one prompt doc,
`claude_prompts/RT/rt_cdom_coding_prompt_1.md`, executes M5. Work happens on a
fresh branch off `main` once `inelastic-rt` merges; JXP runs all git.

| milestone | delivers | gate |
|---|---|---|
| **M5** | `IOPs.a_cdom` + loader wiring; Hawes kernel `K_cdom` in JAX with the 350 nm clamp + truncated-fraction diagnostic; `CDOMFl(scale=1.0)` wired into `Inelastic`/`forward()` with `None` default; δ_C defined zero-init/untrained; extended bit-identity regression; plausibility, gradient, and speed checks; docs note | the five-item v1 gate of §5 |
| **M6** (deferred) | δ_C trained on the §7 truth channel; quantitative validation | the M6 gate of §5 — **blocked on HydroLight CDOM-fl truth**; no prompt doc until the runs exist |

M5 task breakdown (the prompt doc carries the gates per task): (1) types —
`a_cdom`, `CDOMFl`, `cdom_fl` retype + validators; (2) L23 loader `a_cdom`
extraction; (3) the Hawes kernel + constants + correctness pins; (4) the clamp
+ truncated-fraction diagnostic; (5) composition into `forward()` + the
extended bit-identity pin; (6) δ_C definition (zero-init, stubbed training
entry point); (7) plausibility/gradient/speed checks; (8) docs/changelog and
the implementation-record update.

## 7. HydroLight run request (refines wishlist item 3)

This section expands the inelastic design §8's one-line wishlist item 3 into
a commissioning spec (CFQ8); it refines that item rather than replacing the
wishlist. **Requested jointly with the geometry runs** (wishlist item 1 — the
inelastic report's priority-2 ask) to amortize the HydroLight setup.

- **Paired runs "X4 vs X4 + CDOM-fl"** — an X5-style scenario on the same
  3,320-scene L23 ensemble, so the pair difference isolates CDOM-fl on top of
  the realistic all-processes-on ocean, exactly as X4−X2 isolated Chl-fl.
- **A CDOM-stratified scene subset** spanning the full a_g(440) range with the
  **CDOM-rich tail oversampled** (the sparse-tail lesson from δ_F's eutrophic
  drift).
- **All three zeniths** (0/30/60°) — or the denser zenith grid if wishlist
  item 1 runs in the same batch.
- **Full 350–750 nm output** on the L23 grid.
- **The exact Hawes quantum-efficiency function/version recorded** (which
  published function, any HydroLight-side modifications, all constants) so the
  model kernel of §2 matches the truth's constants exactly.

## 8. Risks and open issues

- **No truth to validate against.** The §5 correctness pins and plausibility
  band verify the *implementation*, not the *physics accuracy* — they are a
  weak substitute for real validation, and the design says so. Until M6, the
  term is "Hawes-consistent and plausible", never "validated". Mitigation:
  default-off (§1), and the §7 runs are already specified.
- **The 350 nm truncation** cuts real excitation (Hawes excitation extends
  well below 350 nm); its magnitude is unquantified until the kernel exists.
  If the truncated fraction turns out large in the blue emission bands, the
  caveat hardens into a wishlist-item-6 dependency.
- **a_cdom / a_ph / a_dg bookkeeping.** The loaders must populate `a_cdom`
  (= a_g) consistently with the existing `a_ph` split and the total-`a`
  validators (`a_ph + a_cdom ≤ a` etc.); silent double-counting against a_dg
  conventions elsewhere in the ecosystem (BING's a_dg models) is a real
  foot-gun — the loader tests must pin the decomposition.
- **Hawes function provenance.** Published Hawes constants exist in several
  variants (fulvic vs humic samples; HydroLight's default choice). Picking a
  different variant than the eventual truth runs would re-introduce exactly
  the constants mismatch the CFQ4 decision avoids — hence the §7 requirement
  to record the version on both sides.
- **Interface drift before M6.** `CDOMFl` may need shape metadata once truth
  exists; it is a pytree precisely so that growth is non-breaking, but M6
  should expect a small types revision.

## 9. References

- **Hawes, S. K., K. L. Carder, and G. R. Harvey (1992)** — *Quantum
  fluorescence efficiencies of fulvic and humic acids: effects on ocean color
  and fluorometric detection*, Ocean Optics XI, Proc. SPIE 1750, 212–223.
  The spectral fluorescence quantum-efficiency parameterization HydroLight
  implements for CDOM fluorescence. *(Bibliographic note: authors, title,
  venue, and volume are as cited by Mobley's Light and Water / the Ocean
  Optics Web Book; the page range should be confirmed against the SPIE record
  when the kernel is implemented.)*
- [`design/rt_inelastic_model.md`](rt_inelastic_model.md) — the companion
  design: architecture, Ed module, composition law, correction-head machinery,
  conventions — all inherited here by reference.
- `context/RT/rt_inelastic_bing_summary.md` §4.3 — the assessment that
  flagged CDOM fluorescence as the untreated third inelastic process with no
  in-hand truth (the origin of this document).
- Loisel et al. (2023), *ESSD* 15, 3711 — the L23 database (CDOM fluorescence
  omitted by design; a_g stored separately from a_nap).
