# The surface, the second grid, and the envelope

Three pieces of machinery that M5 added while taking the model to a
multi-angular dataset. They share a theme: **each replaces a package-wide
constant with something that varies**, because a constant that was right for
one nadir dataset stops being right the moment there are two datasets or two
geometries.

They are documented together, and separately from the model chapters, because
none of them changes any number this site claims. The elastic and inelastic
results are nadir, on the L23 grid, from the shipped model — and all three
defaults below are exactly what that model had before.

*Sources for this page: the {mod}`robust.rt.conventions` and
{mod}`robust.rt.emulator` module docstrings;
[`design/m5_report.md`](gh:design/m5_report.md) §4 (what M5 delivered) and §6
(the contamination and its fix);
[`design/rt_elastic_implementation.md`](gh:design/rt_elastic_implementation.md)
§7. The tables, grids and envelopes printed here were re-measured in this
environment when the page was written; the three findings under "All three angles
earn their place" are quoted from the
{class}`~robust.rt.conventions.SurfaceTransfer` docstring, where they were
measured at M5.*

## The geometry-aware surface transfer

{doc}`conventions` introduces the Lee et al. (2002) conversion
$R_{rs} = A\,r_{rs}/(1 - B\,r_{rs})$ with $A = 0.52$, $B = 1.7$. Those are
**nadir** constants, and the page says so. What it could not say, before there
was off-nadir truth to measure against, is how badly they fail elsewhere.

The true $A$ tracks the Fresnel transmittance through the interface, and that
falls as the view angle grows. Measured on PB24 and read out of the shipped
table at $\theta_s = 30°$, $\Delta\varphi = 0°$:

```text
theta_v =  0 deg   A = 0.5298   B = 1.714      (Lee: A = 0.52, B = 1.7)
theta_v = 30 deg   A = 0.5099   B = 1.811
theta_v = 60 deg   A = 0.4343   B = 2.032
theta_v = 70 deg   A = 0.3764   B = 2.128
```

At nadir Lee's constants are essentially exact — which is the reassuring half of
the result, since every number on this site was computed with them. By 70° the
true $A$ has fallen 29 %.

{class}`~robust.rt.conventions.SurfaceTransfer` is the response: $A$ and $B$
tabulated against all three angles and interpolated trilinearly. The fit is
**linear in both coefficients** — $R_{rs} = A\,r_{rs} + B\,(R_{rs}\,r_{rs})$ —
so it is one `lstsq` per grid cell, with no seed, no learning rate and no
stopping rule. It is bit-reproducible by construction.

The shipped table, {func}`~robust.rt.conventions.default_transfer`:

```text
shape       (10, 10, 13)          theta_s x theta_v x dphi
theta_s     0 10 20 30 40 50 60 70 80 87.75  deg
theta_v     0 10 20 30 40 50 60 70 80 87.5   deg
dphi        0 15 ... 180                     deg, 15 deg steps
A           0.0636 - 0.5311
B           1.668 - 2.476
provenance  PB24 OLCI, 320 training realisations (1-400), 1300 geometry cells,
            415982 samples; one lstsq per cell
```

Scored on **held-out realisations**, against Lee's constants:

| $\theta_v$ | nadir constants | fitted table | gain |
|---|---|---|---|
| 0° | 1.84 % | 1.56 % | 1.2× |
| 30° | 4.32 % | 2.04 % | 2.1× |
| 60° | 33.15 % | 4.57 % | **7.2×** |
| 70° | 65.63 % | 6.43 % | 10.2× |
| 0–70° window | 6.81 % | 2.05 % | 3.3× |

Three findings are worth carrying out of that table.

**All three angles earn their place.** At $\theta_v = 60°$ the per-geometry $A$
still spans 0.28–0.46 across solar zenith and azimuth, so an $A(\theta_v)$ table
alone leaves a median 3.4 % and up to 70 % error against the full one.

**$Q$ does not.** Lee's $B = 1.7$ is really $\bar{r}Q$ with $Q$ assumed $\approx
3.5$, and PB24 tabulates the real $Q$ (0.9–6.0). Refitting with it in place —
$1 - \bar{r}Q r_{rs}$ — scores 1.71 % against 1.74 % for simply fitting $B$ per
geometry. Not worth carrying, which is fortunate:
{func}`~robust.rt.hybrid.forward` has no $Q$ to offer.

**The residual does not go to zero.** Even fitting both coefficients at every
geometry leaves a median **1.8 %**. The Lee *form* is the floor here, not the
coefficients.

:::{important}
**Fit the table on your own training rows.** The packaged table exists for a
caller who has no PB24 split of their own. Anyone scoring a model on a held-out
PB24 set should call {func}`~robust.rt.data.pb24.fit_transfer` with that split's
own `train` mask instead.

This is not hypothetical tidiness. The packaged table was fitted under one
400-realisation split while the two analysis scripts split 200 and 800 — and
`make_splits` permutes whatever realisation set it is given, so `SPLIT_SEED`
names a *procedure*, not a partition. 31 of one script's 40 held-out
realisations turned out to have been in the table's training set. The table sits
only in the rival model's scoring path, so the bias favoured the rival and the
conclusions survived; "survived" is not "measured". Both scripts now refit.
[`design/m5_report.md`](gh:design/m5_report.md) §6 has the full account,
including the measurement that the packaged coefficients themselves were never
wrong — regenerating them reproduced $A$ and $B$ bit-identically, and only the
provenance string changed.
:::

## A second wavelength grid

{doc}`conventions` describes the canonical 81-point grid, 350–750 nm in 5 nm
steps, which is L23's. PB24 is on the 12 OLCI bands, so `check_wave` had to stop
meaning "the grid" and start meaning "*a* grid".

```text
GRIDS        canonical, l23, olci
l23          81 bands, 350-750 nm, 5 nm steps
olci         12 bands, 400-753 nm
             400 412 443 490 510 560 620 665 673 681 709 753
```

Half the OLCI bands — 412, 443, 673, 681, 709, 753 — do **not** fall on 5 nm
nodes, so this is a genuinely different grid rather than a subsample of the
canonical one.

The extension was made without weakening the check it generalises.
{func}`~robust.rt.conventions.wave_grid` resolves `None` to L23, so every
pre-M5 call site keeps its exact meaning; an unknown name raises rather than
falling back; and {func}`~robust.rt.conventions.check_wave` still refuses a
mismatch — it now refuses *per grid*, rejecting a 12-band array that is not
OLCI's as firmly as it rejects an 81-band one.

{func}`~robust.rt.conventions.bb_w` gained the same treatment from the other
direction: `mode="clamp"` is the default and is what M0–M4 used, so no number
moves, but a grid reaching past 750 nm can now ask for `"extrapolate"` (the
fitted red tail) or `"raise"` rather than being silently clamped.

## The envelope is per model, not per package

Until M5 the sanctioned solar-zenith span was a single module constant,
{data}`~robust.rt.emulator.SUPPORTED_THETA_S`, consulted by every emulator's
domain check. That is safe with one model and unsafe with two: widening it for a
PB24-trained network would have widened the shipped L23 network's envelope at
the same time, silently.

{class}`~robust.rt.emulator.Envelope` is now a field on
{class}`~robust.rt.emulator.Emulator`, carried with the weights:

```text
DEFAULT_ENVELOPE        theta_s = (0.0, 60.0)   theta_v = None   dphi = None
PB24_ENVELOPE           theta_s = (0.0, 70.0)   theta_v = (0.0, 70.0)
load_default().envelope theta_s = (0.0, 60.0)   theta_v = None   dphi = None
```

`None` means "judge this angle by what the fit was actually trained on", which
for an L23 model is the single value $\theta_v = 0$ — so any off-nadir view is
flagged, which is the correct answer for a nadir-only fit. The shipped model's
envelope is `DEFAULT_ENVELOPE`, identical to the old constant, and
`SUPPORTED_THETA_S` survives as its default. Nothing about the domain guard
described in {doc}`forward` behaves differently.

The distinction the envelope preserves is the one {doc}`emulator` draws: for the
IOP and wavelength features the trained range is the right bound, because
outside it the network is genuinely unconstrained; for the **angles** the bound
is a *project decision*, so a fit trained on a subset of angles may still be
used across the whole sanctioned span. Making it per model means two models can
hold different decisions at once.

## What none of this bought

An honest closing, because this page describes infrastructure that was built for
a milestone that failed.

M5 assembled all three pieces above in order to train an emulator on PB24 and
claim the BRDF axis. **That gate failed**, on both splits and at every seed, and
no PB24 weights exist — `load_default()` is still the L23 nadir model. The cause
was not the network: the ZTT backbone is evaluated outside its own fitted range
off-nadir, and the hybrid's bounded relative correction cannot repair a backbone
that has gone negative. {doc}`../using/limitations` states the measurement, and
[`design/m5_report.md`](gh:design/m5_report.md) sets out the chain in full.

So the surface transfer, the OLCI grid and the per-model envelope are real,
gated, and used — by the analysis scripts, and by whatever addresses the
backbone question next. They are not evidence that this model works off-nadir.
It does not.
