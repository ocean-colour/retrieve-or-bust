"""
Reference-data loaders for the elastic-RT forward model.

The RT reference is HydroLight. Week 1 used the Loisel+2023 (L23) *elastic*
outputs already in hand (:mod:`robust.rt.data.l23`); M5 adds **PB24**
(:mod:`robust.rt.data.pb24`), the Pitarch & Brando multi-angular release, which
is the first reference here to vary the particle phase function and the full
BRDF rather than fixing both.

The two loaders share a shape -- ``load_batch``, ``write_fixture``,
``npz_reader``, a committed fixture holding the loader's *input* so CI runs real
numbers -- but deliberately not a field list: L23 ships ``Rrs`` only, while PB24
tabulates ``rrs``, ``Q`` and the average cosines as well.
"""

from . import l23, pb24

__all__ = ["l23", "pb24"]
