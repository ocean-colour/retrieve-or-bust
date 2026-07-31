"""
Input pytrees for the forward model — **stub (lands in M1)**.

The three arguments of :func:`robust.rt.forward`, as registered JAX pytrees so
they can be ``jit``/``vmap``-ed and differentiated through, with light
``jaxtyping`` annotations on the public signatures.

Planned contents (design §3–4.2, coding plan M1)
------------------------------------------------
- ``IOPs(a, bb_w, bb_p)`` — all m⁻¹, on the canonical grid; ``bb = bb_w + bb_p``.
  The water/particle split is kept explicit rather than summed away.
- ``PhaseParams(B_p, ...)`` — the *explicit* phase-function descriptor. Week 1
  carries the single particulate backscattering ratio ``B_p = bb_p / b_p``
  (~0.005–0.03, realized through Fournier–Forand); the container is shaped so
  the fuller ZTT backward-VSF parameters slot in at M5 **without** changing the
  ``forward`` signature.
- ``Geometry(theta_s, theta_v, dphi, wind)`` — solar zenith, sensor zenith,
  relative azimuth (degrees), and optional wind speed. L23 fixes the view at
  nadir, but the API carries the full geometry from day one so the M5 BRDF runs
  need no interface change.
"""
