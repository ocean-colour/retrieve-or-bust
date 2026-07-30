"""
Loisel+2023 elastic reference batches — **stub (lands in M1)**.

Wraps ``ocpy.hydrolight.loisel23`` (reuse, not reimplementation) to turn the L23
netCDFs into the ``(IOPs, Geometry, Rrs)`` batches the model trains and scores
against.

The elastic set is ``Hydrolight1{Y:02d}.nc`` — X=1 means no inelastic processes —
at three solar-zenith angles, ``Y ∈ {0, 30, 60}``, i.e. ``Hydrolight100.nc``,
``Hydrolight130.nc``, ``Hydrolight160.nc``. Each holds 3320 scenes × 81 λ
(350–750 nm), nadir view, fixed Fournier–Forand phase function. Solar zenith is
therefore the one geometry axis the prototype can genuinely exercise; the
X=2 / X=4 files (Raman, fluorescence) are out of scope for the elastic model.

Planned contents (design §4.1, coding plan M1)
----------------------------------------------
- A one-call batch loader returning JAX arrays for the three zenith angles.
- ``B_p = bbnw / bnw`` computed per scene from the L23 fields.
- The seeded validation splits (coding-plan CQ6): a random 20% of scenes held
  out, and a held-out solar zenith (train on 0°/30°, test on 60°).

The files live outside the repo (~17 MB each); ``ocpy`` resolves the directory
from ``$OS_COLOR``. Tests that need them skip when they are absent — see
``robust/tests/conftest.py``.
"""
