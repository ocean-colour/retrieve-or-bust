#!/usr/bin/env python
"""Train the inelastic correction heads (M3 task 2) and write the weights.

Two independent fits, one per head, on the full L23 release:

- **delta_R** — target ``Rrs_X2 / Rrs_X1``. The loss is the *relative
  increment error* ``(f_R - 1)/(f_truth - 1) - 1`` (RMS, in percent), which
  is intrinsically relatively weighted: a clear-water scene at 700 nm counts
  the same as a turbid one at 550 (the BING/elastic lesson — an unweighted
  loss lets the red run away). Evaluated over the official band
  (lambda >= 400 nm; below it the single-shift machinery clamps — M1).
- **delta_F** — target ``Rrs_X4 - Rrs_X2``. Loss over the emission window
  (655-715 nm), each residual normalized by the scene's *own* 685 nm truth,
  so trophic states weigh equally and the near-zero tails cannot blow up a
  pointwise relative error.

Both losses carry the elastic effort's size penalty (``+ penalty * 100 *
rms(delta)``): corrections must earn their magnitude.

Splits are the elastic ones verbatim (``make_splits`` — proved mask-for-mask
at M1), **train split, all three zeniths**; the zenith-holdout variant
(train 0/30 deg, evaluate 60 deg) is trained and *reported* as the geometry
generalization diagnostic (elastic CQ6), never shipped or gated.

Usage
-----
    python design/py/train_inelastic_corr.py             # write both .npz
    python design/py/train_inelastic_corr.py --dry-run   # train, report only

Requires ``$OS_COLOR`` (the six L23 netCDFs) and the ``ocean14`` environment.
Regenerate after any change to the head features, architecture, or loss —
``inelastic_corr.load_head`` refuses a file whose feature list no longer
matches the code.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from robust.rt import inelastic as I  # noqa: E402
from robust.rt import inelastic_corr as IC  # noqa: E402
from robust.rt.data import l23 as L  # noqa: E402

#: Loss-band edges. Raman: the official band (clamp caveat below 400 nm).
#: Fluorescence: the emission window the line actually occupies.
RAMAN_BAND = (400.0, 750.0)
FL_WINDOW = (655.0, 715.0)

#: Reporting bands (the M2 characterization / M3 gate bands).
GATE_BAND = (550.0, 700.0)

#: Size-penalty weight, the elastic convention: percentage points of fit
#: error paid per percentage point of correction RMS.
PENALTY = 0.02

#: Guard inside the RMS square roots (the elastic ``_RMS_EPS`` lesson: the
#: output layer is zero-initialised, so step 0 is exactly 0/0 without it).
EPS = 1e-24

#: The shipped configurations. delta_max per the record §5.2 sizing.
RAMAN_CONFIG = IC.HeadConfig("raman", hidden=(16,), delta_max=1.0)
FL_CONFIG = IC.HeadConfig("fl", hidden=(16,), delta_max=0.5)


def standardize_stats(x, rows):
    """Per-feature mean/std over the training rows (all wavelengths)."""
    flat = np.asarray(x[rows]).reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0.0] = 1.0
    return jnp.asarray(mean), jnp.asarray(std)


def fit_head(config, x_raw, rows_train, loss_of_delta, steps, lr=3e-3):
    """Full-batch Adam on one head; returns the trained CorrectionHead.

    ``loss_of_delta(delta_train) -> (fit_percent,)`` maps the head output on
    the training rows to the fit term; the size penalty is added here so
    both heads share it.
    """
    import optax

    mean, std = standardize_stats(x_raw, rows_train)
    head = IC.init_head(config.kind, config)
    head = IC.CorrectionHead(params=head.params, mean=mean, std=std, config=config)
    x_std = jnp.asarray(
        (np.asarray(x_raw[rows_train]) - np.asarray(mean)) / np.asarray(std)
    )
    model = IC._emulator._network(config)

    def objective(params):
        delta = IC._emulator._delta(model, params, x_std, config)
        fit = loss_of_delta(delta)
        size = 100.0 * jnp.sqrt(jnp.mean(delta**2) + EPS)
        return fit + PENALTY * size, (fit, size)

    tx = optax.adam(lr)
    state = tx.init(head.params)
    grad_fn = jax.jit(jax.value_and_grad(objective, has_aux=True))

    @jax.jit
    def step(params, state):
        (loss, aux), grads = grad_fn(params)
        updates, state = tx.update(grads, state)
        return optax.apply_updates(params, updates), state, loss, aux

    params = head.params
    t0 = time.perf_counter()
    for k in range(steps):
        params, state, loss, (fit, size) = step(params, state)
        if k % 500 == 0 or k == steps - 1:
            print(
                f"    step {k:5d}  loss {float(loss):7.3f}  "
                f"fit {float(fit):7.3f}%  |delta|rms {float(size):6.3f}%"
            )
    print(f"    {time.perf_counter() - t0:.0f} s for {steps} steps")
    return IC.CorrectionHead(params=params, mean=mean, std=std, config=config)


def raman_metrics(f_model, f_truth, batch, wave, mask, label):
    """Median increment error per zenith, gate band + 490 nm. Returns worst."""
    inc_err = (f_model - 1.0) / (f_truth - 1.0) - 1.0
    band = (wave >= GATE_BAND[0]) & (wave <= GATE_BAND[1])
    i490 = int(np.abs(wave - 490.0).argmin())
    worst = 0.0
    for z in (0.0, 30.0, 60.0):
        rows = mask & (batch.zenith == z)
        med = np.median(inc_err[rows][:, band])
        med490 = np.median(inc_err[rows, i490])
        worst = max(worst, abs(med))
        print(
            f"    {label:28s} zenith {z:2.0f}:  550-700 {100 * med:+6.2f}%   "
            f"490 nm {100 * med490:+6.2f}%"
        )
    return worst


def fl_metrics(fl_model, fl_truth, batch, wave, mask, label):
    """Median 685 nm model/truth - 1 per zenith. Returns worst |median|."""
    i685 = int(np.abs(wave - 685.0).argmin())
    worst = 0.0
    for z in (0.0, 30.0, 60.0):
        rows = mask & (batch.zenith == z)
        med = np.median(fl_model[rows, i685] / fl_truth[rows, i685]) - 1.0
        worst = max(worst, abs(med))
        print(f"    {label:28s} zenith {z:2.0f}:  685 nm {100 * med:+6.2f}%")
    return worst


def aph_decile_table(fl_model, fl_truth, batch, wave, mask):
    """Median 685 nm error by a_ph(440) decile — the eutrophic-tail watch."""
    i685 = int(np.abs(wave - 685.0).argmin())
    i440 = int(np.abs(wave - 440.0).argmin())
    aph = np.asarray(batch.iops.a_ph)[:, i440]
    err = fl_model[:, i685] / fl_truth[:, i685] - 1.0
    edges = np.quantile(aph[mask], np.linspace(0, 1, 11))
    print("    a_ph(440) decile | median 685 nm error (held-out scenes)")
    for d in range(10):
        rows = mask & (aph >= edges[d]) & (aph <= edges[d + 1])
        print(
            f"      {d + 1:2d} [{edges[d]:8.4f}, {edges[d + 1]:8.4f}]  "
            f"{100 * np.median(err[rows]):+6.2f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="write nothing")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--out-dir", type=Path, default=IC.DEFAULT_RAMAN_WEIGHTS.parent)
    args = parser.parse_args()

    batch = L.load_inelastic_batch()
    splits = L.make_splits(batch)
    wave = np.asarray(batch.wave)
    print(
        f"L23 inelastic: {batch.n_sample} samples x {len(wave)} lambda; "
        f"{int(splits.scene_train.sum())} train / "
        f"{int(splits.scene_test.sum())} held-out samples"
    )

    # The analytic terms and truth channels, once, full batch.
    f_phys = np.asarray(I.raman_factor(batch.iops, batch.geometry, batch.wave))
    k_fl = np.asarray(I.fluorescence_kernel(batch.iops, batch.geometry, batch.wave))
    fl_base = L.PHI_C_L23 * k_fl
    f_truth = np.asarray(batch.truth_raman_factor)
    fl_truth = np.asarray(batch.truth_fluorescence)

    x_raman = np.asarray(IC.features_raman(batch.iops, batch.geometry, batch.wave))
    x_fl = np.asarray(IC.features_fl(batch.iops, batch.geometry, batch.wave))

    results = {}
    for tag, train_mask in (
        ("shipped (all zeniths)", splits.scene_train),
        ("zenith-holdout 0/30->60", splits.scene_train & splits.zenith_train),
    ):
        print(f"\n=== {tag}: delta_R ===")
        band = (wave >= RAMAN_BAND[0]) & (wave <= RAMAN_BAND[1])
        inc_truth = jnp.asarray(f_truth[train_mask][:, band] - 1.0)
        inc_phys = jnp.asarray(f_phys[train_mask][:, band] - 1.0)

        def raman_loss(delta, inc_truth=inc_truth, inc_phys=inc_phys, band=band):
            rel = inc_phys * (1.0 + delta[:, band]) / inc_truth - 1.0
            return 100.0 * jnp.sqrt(jnp.mean(rel**2) + EPS)

        head_r = fit_head(RAMAN_CONFIG, x_raman, train_mask, raman_loss, args.steps)

        print(f"=== {tag}: delta_F ===")
        window = (wave >= FL_WINDOW[0]) & (wave <= FL_WINDOW[1])
        i685 = int(np.abs(wave - 685.0).argmin())
        base_w = jnp.asarray(fl_base[train_mask][:, window])
        truth_w = jnp.asarray(fl_truth[train_mask][:, window])
        peak = jnp.asarray(fl_truth[train_mask][:, i685])[:, None]

        def fl_loss(delta, base_w=base_w, truth_w=truth_w, peak=peak, window=window):
            rel = (base_w * (1.0 + delta[:, window]) - truth_w) / peak
            return 100.0 * jnp.sqrt(jnp.mean(rel**2) + EPS)

        head_f = fit_head(FL_CONFIG, x_fl, train_mask, fl_loss, args.steps)
        results[tag] = (head_r, head_f)

        # ---- report on held-out scenes (all zeniths — incl. the unseen 60
        # deg for the holdout variant, which is the point of it) ----
        d_r = np.asarray(head_r.delta(batch.iops, batch.geometry, batch.wave))
        f_corr = np.asarray(IC.corrected_raman_factor(d_r, f_phys))
        d_f = np.asarray(head_f.delta(batch.iops, batch.geometry, batch.wave))
        fl_corr = fl_base * (1.0 + d_f)

        print(f"  --- {tag}: held-out scenes ---")
        raman_metrics(f_phys, f_truth, batch, wave, splits.scene_test, "Raman analytic")
        worst_r = raman_metrics(
            f_corr, f_truth, batch, wave, splits.scene_test, "Raman corrected"
        )
        fl_metrics(fl_base, fl_truth, batch, wave, splits.scene_test, "Fl analytic")
        worst_f = fl_metrics(
            fl_corr, fl_truth, batch, wave, splits.scene_test, "Fl corrected"
        )
        if tag.startswith("shipped"):
            print(
                f"  gate check (<= 5.00%): Raman worst {100 * worst_r:.2f}%, "
                f"fluorescence worst {100 * worst_f:.2f}%"
            )
            aph_decile_table(fl_corr, fl_truth, batch, wave, splits.scene_test)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    head_r, head_f = results["shipped (all zeniths)"]
    out_r = args.out_dir / IC.DEFAULT_RAMAN_WEIGHTS.name
    out_f = args.out_dir / IC.DEFAULT_FL_WEIGHTS.name
    IC.save_head(head_r, out_r)
    IC.save_head(head_f, out_f)
    for path in (out_r, out_f):
        IC.load_head(path)  # refuse-on-mismatch runs now, not at first use
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
