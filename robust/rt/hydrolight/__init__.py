"""
HydroLight deck generation -- the input files the operator runs from.

The campaign is specified in :doc:`design/hydrolight_runs.md`; this package
*builds* it. Because the operator will run any input files we give them (Q&A
A5), deck generation is our engineering rather than theirs, and it lives in the
package under test so the loader can later cross-check every delivered file
against the deck that produced it.

Modules
-------
:mod:`~robust.rt.hydrolight.vsf`
    The volume-scattering-function design set, and the deliverability criterion
    that decides which Fournier-Forand designs can be shipped at all.
:mod:`~robust.rt.hydrolight.grid`
    Batch A's ``(bb/a, eta_bb)`` grid, its physical realizability mask, and the
    fill set.
:mod:`~robust.rt.hydrolight.l23_recon`
    Batch B's L23 block: L23's water bodies rebuilt from their own published
    IOPs, because their input decks are unobtainable and unnecessary.
:mod:`~robust.rt.hydrolight.ensemble`
    Batch B's design block and batch C's three subsets.
:mod:`~robust.rt.hydrolight.deck`
    The format-neutral run IR, its deterministic serialiser, and the adapter
    seam that waits on the operator's template.
:mod:`~robust.rt.hydrolight.manifest`
    The per-batch manifest and the ten-item metadata contract.
"""

from . import deck, ensemble, grid, l23_recon, manifest, vsf

__all__ = ["deck", "ensemble", "grid", "l23_recon", "manifest", "vsf"]
