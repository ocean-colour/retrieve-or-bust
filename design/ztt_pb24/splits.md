# PB24 splits (M8 task 3)

Written by `design/py/make_pb24_splits.py` with
`design/ztt_pb24/splits.json`; do not edit by hand. Loaded by
`robust.rt.data.pb24.make_splits`.

Source: PB24 hyperspectral cache pb24_v1 table (reader version 1); B_p_mean = mean over 451 bands of (bbph + bbNAP) / (bph + bNAP).

| split | definition | held out | train |
|---|---|---|---|
| realisation | seed 23, test fraction 0.2 | 1000 realisations | 4000 |
| bp_band | mean B_p in its (0.4, 0.6) quantile band, [0.01322, 0.01522] | 1000 realisations | 4000 |
| geometry | held out: theta_v_air >= 80 or theta_s >= 80 (degrees, above-water angles); everything else trains | 468 of 1300 air geometries per realisation | 832 (the window) |

Realisations: 5000 (1-5000). Mean B_p spans 0.00144-0.04299 (29.9x). 194 realisations are held out by both the realisation and the B_p-band split.

## The confound

corr(log B_p_mean, log C) = **-0.469** over all 5000 realisations.

Test/train ratio of medians, against what a random 20% hold-out does (median [min, max] over seeds 0-11):

| label | realisation | bp_band | random reference |
|---|---|---|---|
| C | 0.991 | 0.913 | 0.987 [0.780, 1.125] |
| N | 0.953 | 1.308 | 0.958 [0.847, 1.205] |
| Y | 1.010 | 1.053 | 0.971 [0.918, 1.054] |
| a_mean | 1.002 | 1.001 | 0.999 [0.995, 1.003] |
| bb_p_mean | 0.974 | 1.168 | 0.978 [0.830, 1.101] |
| B_p_mean | 0.992 | 0.999 | 1.000 [0.973, 1.019] |

The B_p band is interior, so its train side is both tails and its
`B_p_mean` ratio sits near 1 by construction; the separation lives in the
band edges. Read every other entry against the random reference.
