# Reports

Two team reports stand behind this site. They are where the headline accuracy,
speed and gradient numbers were measured, and both are reproduced here **in full
and verbatim** rather than summarised — the point of a Reports section is that a
reader can check a claim against the document that made it, without leaving the
site or trusting a paraphrase. Numbers that are *not* theirs — a quantity
re-measured on the committed fixture, say, which is smaller than the release the
reports scored — are labelled as such wherever a chapter prints one, and several
chapters say in as many words that a fixture-scale figure must not be quoted as
a report result.

They are dated, versioned documents, written at the end of the effort they
describe. They are not tutorials and they are not maintained as the API moves;
the chapters under *The model* and *Using it* are the maintained prose, and they
cite these reports by section. Where the two disagree, the report is the record
of what was measured and the chapter is the current description of the code.

## What each one measured

| | [The elastic model](report_rt_elastic_model.md) | [The inelastic completion](report_rt_inelastic_model.md) |
|---|---|---|
| Version, date | 1.0, 2026-08-15 | 1.0, 2026-08-27 |
| Subject | the elastic hybrid: the ZTT analytic backbone plus a 417-parameter learned residual | the two inelastic emission terms composed on top: Raman scattering and chlorophyll-a fluorescence, each with a bounded learned correction |
| Truth it was scored against | Loisel et al. (2023), elastic scattering only | Loisel et al. (2023) with all processes on — the "X4" release |
| Headline | **0.30 % rRMS** on held-out water bodies, **2.3×** better than the O25 benchmark of Pitarch et al. (2025) refit on the same data and 24× better than standard Gordon (1988) | **0.34 % rRMS** on held-out water bodies at every solar zenith over 400–700 nm, where the elastic model alone scores 16–19 % against the same truth and 48 % at the 685 nm fluorescence peak |
| Gradients | agree with finite differences to ≤ 5 × 10⁻⁹ | ≤ 5.9 × 10⁻⁹, including in $\varphi_C$, at **1.59×** the elastic runtime, with `inelastic=None` bit-identical (SHA-256 pinned) |
| Length | 313 lines, 9 sections, 3 figures | 432 lines, 9 sections, 4 figures |

Both share a structure, which is worth knowing before you open one: §1
motivation, §2 the model, §3 the data and validation protocol, §4 the results,
**§5 what the prototype may claim and what it may not**, §6 open items, §7
recommended priorities, §8 reproducibility, §9 a map of the documents behind the
report.

§5 is the section to read first if you are deciding whether to use the model.
Both are quoted verbatim on the {doc}`../using/limitations` page — including the
−74 % unseen-zenith cliff — so that the site's limitations cannot drift from the
reports' own wording.

## How these two pages are produced

They are **generated**, not committed. `docs/figures/make_docs_figures.py`
copies `reports/report_rt_elastic_model.md` and
`reports/report_rt_inelastic_model.md` into this directory on every
documentation build (from `docs/conf.py`, so Read the Docs and CI do it
themselves from a bare checkout), changing exactly three things and nothing
else:

- a generated-file banner, so nobody edits the copy;
- the image paths, repointed at the seven figures the same script copies into
  `docs/_static/` — the same PNGs the reports carry, not redrawn;
- the repo-relative links (`design/…`, `context/…`, the plotting scripts),
  rewritten to absolute GitHub URLs on the one `github_url_base` the rest of
  the site's outbound links use.

No wording, number, table or section is touched. The generated copies are
gitignored, so there is no second version of either report in the repository to
drift from the first. The sources are
[`reports/report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md)
and
[`reports/report_rt_inelastic_model.md`](gh:reports/report_rt_inelastic_model.md);
the documents *they* stand on — the design docs, the coding plans and the two
implementation records — are indexed on the {doc}`../development_record` page.

:::{note}
The GitHub links on this page and in the two reports point at `main`, and
`main` does not yet carry `reports/`, `design/`, `context/RT/` or `notebooks/` —
that work lives on an unmerged branch, and the links become correct the moment
it lands. Same dial, same reason, as the {doc}`../development_record` page.
:::

```{toctree}
:maxdepth: 2
:hidden:

report_rt_elastic_model
report_rt_inelastic_model
```
