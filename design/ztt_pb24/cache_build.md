# PB24 consolidation cache — build record (2026-11-02)

*M8 task 2. Built by `design/py/build_pb24_cache.py` with
`robust.rt.data.pb24_cache.build`; cache version 1, `ocpy.hydrolight.pb24`
READER_VERSION 1.*

**Where:** `$OS_COLOR/SD/derived/pb24_v1/` (not committed). **What:** all
5000 hyperspectral realisations (`SD_hyp_no_R_0001–5000.nc`), each opened
once; one float32 `.npy` per field plus `sidecar.json`. Layout: rows are
realisations 1–5000; λ is the last axis; `(λ, θs)` fields stored
`(realisation, θs, λ)`; 4-D fields stored `(realisation, θs, θ, φ, λ)`.
The OLCI set is not cached.

## Sizes

| files | shape (each) | bytes (each) | count | bytes |
|---|---|---:|---:|---:|
| `rrs`, `Rrs`, `Q` | (5000, 10, 10, 13, 451) | 11,726,000,128 | 3 | 35,178,000,384 |
| 11 AOPs + `Esdir_Es_ratio` | (5000, 10, 451) | 90,200,128 | 12 | 1,082,401,536 |
| 10 IOP components | (5000, 451) | 9,020,128 | 10 | 90,201,280 |
| `C`, `N`, `Y` | (5000,) | 20,128 | 3 | 60,384 |
| `sidecar.json` | | 367,984 | 1 | 367,984 |
| **total** | | | 29 | **36,351,031,568** (36.35 GB) |

## Wall time

- Build, 8 reader processes: **286.9 s** (the read-and-stream loop over
  5000 files 241 s, the final fsync and renames the rest).
- Second build (`--rebuild-check`), 4 reader processes: 268.6 s; with the
  SHA-256 + byte comparison of every file and the delete, 6 min 05 s.
- `--check`, 8 processes: 51.1 s.

## Verification

- **`--check`** (`pb24_cache.verify`, seed 24): 5000 realisations re-read
  through the ocpy reader; **293,165,000 table values compared (all of
  them)**; **181,061,166 of 8,794,500,000 4-D values compared (2.06%)**:
  realisations 1, 2500 and 5000 in full, a random 2% of each of the others'
  `rrs`, `Rrs` and `Q`. **All bit-identical** (equality of the raw 32-bit
  words, which implies `np.array_equal(..., equal_nan=True)`, also checked).
- **`--rebuild-check`**: a second build into
  `$OS_COLOR/SD/derived/.pb24_v1_rebuild_check` (same volume, different
  worker count), every one of the 29 files identical by SHA-256 and by
  `filecmp` byte comparison, **sidecar included**; the copy was then deleted.

## Fixture

`robust/tests/files/pb24_cache_small.zip`: realisations 1, 993, 2500; every
table field at 451 bands; the 4-D fields at every 10th band (46 bands,
350–800 nm by 10; `four_d_bands` in the sidecar). 29 members, **2,871,457
bytes** unpacked, **2,190,735 bytes** as one deterministic zip (`--pack`,
2026-11-03). Built by the same `build()`; `--check` and `--rebuild-check` on
the unpacked directory pass, and rebuilding through `--pack` reproduces the
committed zip byte for byte.
