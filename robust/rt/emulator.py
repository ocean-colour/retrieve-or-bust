"""
Residual emulator ΔRrs — **stub (lands in M3)**.

A small Flax MLP trained (Optax) on ``ΔRrs = Rrs_reference − Rrs_ZTT``. It learns
only what the analytic backbone misses, which is why it can stay small and why
its extrapolation is bounded — the whole argument for the hybrid over a wholly
learned model.

Planned contents (design §4.4, coding plan M3)
----------------------------------------------
- The MLP: features ``(u = bb/(a+bb)`` or the ``(ω_bw, ω_bp)`` split, ``B_p``,
  geometry, λ) → ``ΔRrs(λ)``. Modest width and depth; the target is small.
- An Optax training loop minimizing **relatively weighted** rRMS. The weighting
  is not incidental — the BING lesson is that an unweighted objective lets the
  red-λ terms, where ``Rrs`` is tiny, run away.
- Regularization (L2 / size) keeping the correction a *bounded* one.

``flax`` and ``optax`` are imported inside the functions that need them, not at
module scope, so ``import robust.rt`` stays cheap for the analytic-only path.
"""
