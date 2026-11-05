#!/usr/bin/env python
"""
Write the pre-M8-task-4 nadir reference for ``rrs_ZTT`` (M8 task 4).

M8 task 4 (``claude_prompts/RT/rt_fit_pb24.md``) refracts ``Geometry.theta_v``
inside ``ztt.geometry_to_paper_angles`` and lets ``Geometry.diffuse_fraction``
replace TT's H in µd. Both must leave nadir viewing with no diffuse fraction
**bit-identical**. This script records what the code computed *before* the
change, on synthetic inputs, so ``robust/tests/test_ztt.py`` can assert the
identity rather than argue it.

It must be run against the **pre-change** package. The committed fixture was
made from commit ``fb253f9`` ("3 is done") with a read-only extract::

    git archive fb253f9 robust | tar -x -C /tmp/head
    PYTHONPATH=/tmp/head python design/py/gen_ztt_nadir_reference.py

Run against the current tree it reproduces the same bytes (that is the test).
Output: ``robust/tests/files/ztt_nadir_reference.npz`` (float32 outputs, default
JAX dtype; inputs stored alongside).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "robust" / "tests" / "files"
OUT = OUT / "ztt_nadir_reference.npz"


def inputs():
    """The synthetic nadir batch: 6 sun zeniths x 81 canonical bands."""
    from robust.rt import conventions as C

    wave = np.asarray(C.WAVE, dtype=float)
    n = 6
    theta_s = np.array([0.0, 10.0, 30.0, 45.0, 60.0, 75.0])
    dphi = np.array([0.0, 30.0, 90.0, 135.0, 180.0, 60.0])
    scale = np.linspace(0.5, 2.0, n)[:, None]
    a = 0.02 + 0.3 * np.exp(-0.015 * (wave - 350.0))[None, :] * scale
    a = a + 0.4 * np.exp(-(((wave - 700.0) / 60.0) ** 2))[None, :]
    bb_w = np.broadcast_to(np.asarray(C.bb_w(wave), dtype=float), (n, wave.size))
    bb_p = 0.004 * scale * (550.0 / wave)[None, :]
    B_p = np.broadcast_to(np.linspace(0.006, 0.03, n)[:, None], (n, wave.size))
    ff_B = np.broadcast_to(np.array([0.008, 0.015]), (n, wave.size, 2))
    ff_weight = np.broadcast_to(np.array([0.7, 0.3]), (n, wave.size, 2))
    return {
        "wave": wave,
        "theta_s": theta_s,
        "dphi": dphi,
        "a": a,
        "bb_w": np.array(bb_w),
        "bb_p": bb_p,
        "B_p": np.array(B_p),
        "ff_B": np.array(ff_B),
        "ff_weight": np.array(ff_weight),
    }


def build(arrays):
    """Containers from the stored inputs (nadir: ``theta_v = 0``)."""
    from robust.rt.types import Geometry, IOPs, PhaseParams

    iops = IOPs(
        a=jnp.asarray(arrays["a"]),
        bb_w=jnp.asarray(arrays["bb_w"]),
        bb_p=jnp.asarray(arrays["bb_p"]),
    )
    phase = PhaseParams(
        B_p=jnp.asarray(arrays["B_p"]),
        ff_B=jnp.asarray(arrays["ff_B"]),
        ff_weight=jnp.asarray(arrays["ff_weight"]),
    )
    geometry = Geometry(
        theta_s=jnp.asarray(arrays["theta_s"]),
        theta_v=jnp.zeros(arrays["theta_s"].shape),
        dphi=jnp.asarray(arrays["dphi"]),
    )
    return iops, phase, geometry, jnp.asarray(arrays["wave"])


def outputs(arrays):
    """Every nadir output the reference pins."""
    from robust.rt import hybrid, ztt

    iops, phase, geometry, wave = build(arrays)
    rrs_l23 = ztt.rrs_ZTT(iops, phase, geometry, wave)
    return {
        "rrs_l23": rrs_l23,
        "rrs_tt2017": ztt.rrs_ZTT(iops, phase, geometry, wave, variant="tt2017"),
        "Rrs_l23": ztt.Rrs_ZTT(iops, phase, geometry, wave),
        "mu_inf_closure": ztt.mu_inf_closure(iops, phase, geometry, rrs_l23, wave),
        "psi": ztt.geometry_to_paper_angles(geometry)[3],
        "forward_ztt": hybrid.forward(iops, phase, geometry, wave, "ztt"),
        "forward_hybrid": hybrid.forward(
            iops, phase, geometry, wave, check_domain=False
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    arrays = inputs()
    result = {f"out_{k}": np.asarray(v) for k, v in outputs(arrays).items()}
    np.savez(args.out, **arrays, **result)
    print(
        f"wrote {args.out}: "
        + ", ".join(f"{k} {v.dtype}{v.shape}" for k, v in result.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
