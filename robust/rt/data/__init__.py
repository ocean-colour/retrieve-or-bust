"""
Reference-data loaders for the elastic-RT forward model.

The RT reference is HydroLight. Week 1 uses the Loisel+2023 (L23) *elastic*
outputs already in hand (:mod:`robust.rt.data.l23`); new HydroLight runs that
vary the particle phase function and the sensor angles arrive at M5 and get
their own module here.
"""

from . import l23

__all__ = ['l23']
