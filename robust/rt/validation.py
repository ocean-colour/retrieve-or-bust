"""
Validation protocol — **stub (lands in M4)**.

Scores the forward model on the three axes that define acceptance: *accurate*,
*fast*, *differentiable*. Deliberately a **protocol, not a target** — no blind
absolute numbers, consistent with the project's unbiased-uncertainty stance.
Absolute rRMS and latency are reported; only the *relative* comparison is gated.

Planned contents (design §6, coding plan M4)
--------------------------------------------
- **Accurate.** rRMS in ``rrs`` space, relatively weighted, broken out per λ, per
  solar zenith, and per ``B_p`` bin — always alongside Gordon, PR05, and O25 on
  the same splits, plus the two held-out splits (random 20% of scenes; the unseen
  60° zenith).
- **Fast.** Throughput (scenes·λ/s, batched) and single-call latency, so it is
  visible if the emulator erases the speed advantage over calling an RT solver.
- **Differentiable.** ``jax.grad`` versus central finite differences w.r.t. ``a``,
  ``bb_p``, ``B_p``, and geometry, across a random batch. A hard gate: it is the
  property the future inversion depends on.

The M4 acceptance gate — hybrid beats standard Gordon on **both** held-out splits
*and* passes the gradient check — is what makes the week-1 prototype "done".
Run-and-figure scripts live in ``design/py/``, outside the package.
"""
