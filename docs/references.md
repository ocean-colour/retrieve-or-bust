# References

The bibliography the model rests on. Every entry below is taken from the
References section of one or both of the forward model's two reports —
`reports/report_rt_elastic_model.md` and
`reports/report_rt_inelastic_model.md` — rather than composed for this page,
and each was checked against its source before being copied here. The
annotation after each entry is the report's own, saying what the work
contributes *to this model*; only the reports' HTML subscripts have been
rewritten for the web (`b<sub>b</sub>` → `b_b` and so on).

The **Source** column names which report's bibliography an entry comes from:
**E** for the elastic report, **I** for the inelastic one.

## The elastic backbone

| Reference | Source |
| --- | --- |
| **Twardowski, M., & Tonizzo, A. (2018)**, *Appl. Sci.* 8, 2684 — the ZTT analytic model with the explicit backward VSF. | E |
| **Twardowski, M., & Tonizzo, A. (2017)**, *Opt. Express* 25, 18122 — the `μ_∞(b_b/a, η_bb)` surface used here. | E |
| **Gordon, H. R., et al. (1988)**, *J. Geophys. Res.* 93, 10909 — the canonical `rrs(u)` polynomial. | E |
| **Lee, Z.-P., et al. (2002)** — the `Rrs` ↔ `rrs` interface conversion (A = 0.52, B = 1.7). | E, I |
| **Sullivan, J. M., & Twardowski, M. S. (2009)**, *Appl. Opt.* 48, 6811 — the measured particulate backward phase function. | E |

## The bidirectional baselines

| Reference | Source |
| --- | --- |
| **Park, Y.-J., & Ruddick, K. (2005)**, *Appl. Opt.* 44, 1236 — the 4th-order bidirectional polynomial. | E |
| **Pitarch, J., et al. (2025)** — the O25 bivariate (`ω_bw`, `ω_bp`) BRDF model; the modern benchmark. | E |

These are the two baselines the hybrid model is scored against, referred to
throughout the reports and the code as **PR05** and **O25**.

## Raman scattering

| Reference | Source |
| --- | --- |
| **Sathyendranath, S., & Platt, T. (1998)**, *Appl. Opt.* 37, 2216 — the two-flow Raman reflectance formulation (the analytic backbone). | I |
| **Bartlett, J. S., et al. (1998)**, *Appl. Opt.* 37, 3324 — the Raman scattering coefficient of water and its spectral dependence. | I |
| **Ge, Y., Gordon, H. R., & Voss, K. J. (1993)**, *Appl. Opt.* 32, 4028 — the 3400 cm⁻¹ Raman wavenumber shift. | I |
| **Walrafen, G. E. (1967)**, *J. Chem. Phys.* 47, 114 — the Raman wavenumber redistribution of water. | I |

## Chlorophyll-a fluorescence

| Reference | Source |
| --- | --- |
| **Gordon, H. R. (1979)**, *Appl. Opt.* 18, 1161 — chlorophyll fluorescence in ocean color. | I |
| **Maritorena, S., et al. (2000)**, *Appl. Opt.* 39, 6725 — the fluorescence quantum yield of natural populations. | I |
| **Behrenfeld, M. J., et al. (2009)**, *Biogeosciences* 6, 779 — satellite fluorescence and phytoplankton physiology (the case for retrieving `phi_C`). | I |

## Data and reference implementations

| Reference | Source |
| --- | --- |
| **Loisel, H., et al. (2023)**, *Earth Syst. Sci. Data* — the L23 HydroLight ensemble, elastic and inelastic releases (via `ocpy`). | E, I |
| **Mobley, C. D.** — *The Ocean Optics Web Book*, Raman scattering chapter; and the HydroLight default inelastic settings L23 used. | I |

The elastic report's entry for Loisel et al. is shorter — "the L23 HydroLight
synthetic ensemble (via `ocpy`)" — and names no journal; the fuller inelastic
entry is the one reproduced above.

## CDOM fluorescence

The one section on this page that does **not** come from a report's
bibliography: the CDOM-fluorescence term postdates both reports, so its two
references are taken from
[`design/rt_cdom_fluorescence_model.md`](gh:design/rt_cdom_fluorescence_model.md)
§9 and from the {mod}`robust.rt.cdom_fl` module docstring, which is where the
kernel's provenance is recorded in full. The **Source** column names those
instead.

| Reference | Source |
| --- | --- |
| **Hawes, S. K., Carder, K. L., & Harvey, G. R. (1992)** — *Quantum fluorescence efficiencies of fulvic and humic acids: effects on ocean color and fluorometric detection*, Ocean Optics XI, Proc. SPIE 1750, 212–223. The spectral fluorescence quantum-efficiency parameterization HydroLight implements, and the Station FA7 constants the kernel uses. | design §9 |
| **Zhai, Hu, Lee et al. (2017)**, *Opt. Express* 25(8), A213–A235 — Eqs. (5)–(8), the functional form the kernel implements literally, and the same 350 nm excitation floor imposed for the same stated reason. | module docstring |
| **Mobley, C. D.** — *The Ocean Optics Web Book*, CDOM-fluorescence page: where the FA7 numeric constants were sourced. | module docstring |

Two provenance notes travel with these, both recorded in the code rather than
resolved: the FA7 constants were **accepted from the Ocean Optics Web Book
without independent primary-source verification**, and the Hawes page range
above is as cited by *Light and Water* / the Web Book rather than confirmed
against the SPIE record. The middle author's initials also differ between
sources — Zhai et al.'s reference list prints "C. K. Carder" where the
ocean-optics literature has Kendall L. Carder. See
{doc}`model/cdom_fluorescence`.

## A name with no entry

`robust/rt/inelastic.py` attributes an alternative Raman coefficient
(2.4 × 10⁻⁴ m⁻¹ at 488 nm) to **Desiderio**, in the comment above the
`B_RAMAN_488` constant. Neither report's bibliography carries a matching
entry, so none is invented here: the attribution stands where it is, in the
source, and this page records the gap rather than papering over it with a
guessed citation.
