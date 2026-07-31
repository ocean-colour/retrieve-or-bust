"""
Fixed conventions for the elastic-RT forward model — **stub (lands in M1)**.

One place for the choices that must be identical across every run, model, and
figure, asserted at load so results stay comparable.

Planned contents (design §3, coding plan M1)
--------------------------------------------
- ``A_RRS = 0.52``, ``B_RRS = 1.7`` (Lee et al. 2002) and the pair
  ``Rrs_to_rrs`` / ``rrs_to_Rrs`` for the subsurface conversion
  ``rrs = Rrs / (A + B * Rrs)``. These match ``bing.rt.rrs``, deliberately: the
  two packages must not disagree about what ``rrs`` means.
- The canonical wavelength grid: L23's 350–750 nm, 81 bands.
- The pure-water backscattering model ``bb_w(λ)``, kept separate from ``bb_p``
  because the water/particle split is load-bearing for the physics.
- Load-time asserts that fire on out-of-range IOPs or a mismatched grid.
"""
