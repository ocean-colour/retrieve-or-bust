# Development record

`robust.rt` — the forward model, and retrieve-or-bust's first component — was
built in ten milestones, each written up as a notebook in the repository. Those notebooks are a **chronological build record, not a
tutorial**: they show what was decided, what was measured and what was rejected
on the day, in the order it happened, and they are not maintained against the
current API. For a runnable introduction use the
[Quickstart](quickstart.md) or the [Quickstart notebook](quickstart_nb.ipynb);
for what the model does today use [The model in one page](model/overview.md)
and the [API reference](api.rst).

:::{note}
Everything on this page links out to GitHub rather than into the site. The
notebooks, the design documents and the implementation records are deliberately
**not rendered here** — they are working documents with their own repo-relative
links and their own audience.

The links point at the `main` branch. The documentation work has not been merged
yet, and `main` does not currently carry the `notebooks/`, `design/` or
`reports/` directories, so these links **will 404 until that merge**. This is
known and accepted: the branch name is deliberately not hardcoded, and the whole
site's outbound links move together through the single `github_url_base` setting
in `docs/conf.py`.
:::

## The elastic backbone — five notebooks

Built against the Loisel et al. (2023) synthetic archive, from an empty
environment to a validated hybrid of the Twardowski & Tonizzo analytic model and
a learned residual.

| | Notebook | What the milestone settled |
|---|---|---|
| M0 | [`rt_elastic_coding_1.ipynb`](gh:notebooks/RT/rt_elastic_coding_1.ipynb) | Environment and scaffold: the JAX stack, the package skeleton, the green base everything else is measured against. |
| M1 | [`rt_elastic_coding_2.ipynb`](gh:notebooks/RT/rt_elastic_coding_2.ipynb) | Data and conventions: the L23 loader, the canonical 81-point grid, the $R_{rs} \leftrightarrow r_{rs}$ convention. |
| M2 | [`rt_elastic_coding_3.ipynb`](gh:notebooks/RT/rt_elastic_coding_3.ipynb) | The ZTT analytic backbone, with the phase function carried explicitly. |
| M3 | [`rt_elastic_coding_4.ipynb`](gh:notebooks/RT/rt_elastic_coding_4.ipynb) | The residual emulator and the composition into `mode='hybrid'`. |
| M4 | [`rt_elastic_coding_5.ipynb`](gh:notebooks/RT/rt_elastic_coding_5.ipynb) | Validation: what the elastic prototype may and may not claim. |

## The inelastic terms — five notebooks

Raman scattering by water and chlorophyll-a fluorescence added on top of the
elastic hybrid, each as analytic physics plus a small bounded learned
correction, without disturbing the elastic result.

| | Notebook | What the milestone settled |
|---|---|---|
| M0 | [`rt_inelastic_coding_1.ipynb`](gh:notebooks/RT/rt_inelastic_coding_1.ipynb) | API extension: the `Inelastic` pytree, and the `inelastic=None` bit-identity guarantee. |
| M1 | [`rt_inelastic_coding_2.ipynb`](gh:notebooks/RT/rt_inelastic_coding_2.ipynb) | The solar spectrum, the excitation grid, and the inelastic truth data. |
| M2 | [`rt_inelastic_coding_3.ipynb`](gh:notebooks/RT/rt_inelastic_coding_3.ipynb) | The analytic terms — the Raman factor $f_{\rm phys}$, the fluorescence kernel $K_{fl}$, and the composition law. |
| M3 | [`rt_inelastic_coding_4.ipynb`](gh:notebooks/RT/rt_inelastic_coding_4.ipynb) | The two learned correction heads, $\delta_R$ and $\delta_F$. |
| M4 | [`rt_inelastic_coding_5.ipynb`](gh:notebooks/RT/rt_inelastic_coding_5.ipynb) | Validation: the inelastic prototype against its acceptance gate. |

## The documents behind them

Each half of the effort has a design document (what was to be built and why), a
coding plan (the milestone breakdown the notebooks follow), and an
implementation record (what was actually built, with the measured numbers). The
two reports are the site's evidence base; they will also be rendered in full in
the site's Reports section, which arrives later in this milestone.

| | Elastic | Inelastic |
|---|---|---|
| Design | [`rt_elastic_model.md`](gh:design/rt_elastic_model.md) | [`rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) |
| Coding plan | [`rt_elastic_model_coding_plan.md`](gh:design/rt_elastic_model_coding_plan.md) | [`rt_inelastic_model_coding_plan.md`](gh:design/rt_inelastic_model_coding_plan.md) |
| Implementation record | [`rt_elastic_implementation.md`](gh:design/rt_elastic_implementation.md) | [`rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md) |
| Report | [`report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md) | [`report_rt_inelastic_model.md`](gh:reports/report_rt_inelastic_model.md) |
