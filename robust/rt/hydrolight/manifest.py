"""
The per-batch manifest (spec Section 2.7) and the metadata contract (Section 3).

**The manifest is the file the loader's tests read.** Anything not in it is not
recorded, so the ten items of the metadata contract appear here explicitly --
including the ones we cannot yet fill, which are marked ``pending`` with the
reason rather than omitted. An absent field and an unanswered question look
identical in a file; they must not.

Because we author the decks (Q&A A5, A7), **seven of the ten items are ours by
construction** and are therefore testable rather than asserted in an email. The
three that are not -- the Raman redistribution function, the chlorophyll
fluorescence efficiency and emission shape, and which published Hawes variant
the CDOM fluorescence option uses -- live in HydroLight's own data files and are
requested from the operator verbatim.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib

from . import deck

__all__ = [
    "RELEASE",
    "VERSION",
    "METADATA_CONTRACT",
    "sha256",
    "build",
    "write",
]

#: Release name. Placeholder pending the operator's preference; frozen before
#: batch 0 ships (spec Section 2.1).
RELEASE = "HH26"

#: Release version directory.
VERSION = "v1"

#: The ten items of spec Section 3, with who supplies each. ``ours`` items are
#: filled by :func:`build`; ``operator`` items are emitted as ``pending``.
METADATA_CONTRACT = {
    "1_hydrolight_version_and_switches": "operator",
    "2_phase_function_family_params_and_table": "ours",
    "3_raman_coefficient_and_redistribution": "operator",
    "4_chl_fluorescence_efficiency_and_shape": "operator",
    "5_cdom_fluorescence_hawes_variant_and_constants": "operator",
    "6_solar_spectrum_and_sky_model": "ours_and_operator",
    "7_depth_grid_and_optical_depth_reached": "ours",
    "8_output_convention_Rrs_vs_rrs_above_vs_below": "operator",
    "9_iop_decomposition": "ours",
    "10_iops_identical_across_scenarios_and_zeniths": "ours",
}


def sha256(path):
    """SHA-256 of a file, as a hex string."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(batch, bodies, files, *, generator_version, designs=(), created=None):
    """Assemble one batch's manifest.

    Parameters
    ----------
    batch : str
        Batch label, e.g. ``"pilot"``.
    bodies : sequence of WaterBody
    files : sequence of path-like
        Every input file shipped with the batch; each is hashed.
    generator_version : str
        Identifies the code that produced this batch. A commit hash when one is
        available; otherwise a caller-supplied string. **Not** read from git here
        -- this package does not run git commands.
    designs : sequence of VSFDesign, optional
        Recorded so metadata item 2 carries the family, the parameters *and* a
        pointer to the table actually used, not just a name.
    created : str, optional
        ISO date. Defaults to today; pass a fixed value for byte-identical
        regeneration.

    Returns
    -------
    dict
    """
    created = created or datetime.date.today().isoformat()
    scenarios = sorted({s for b in bodies for s in b.scenarios})
    zeniths = sorted({z for b in bodies for z in b.theta_s})
    return {
        "release": RELEASE,
        "version": VERSION,
        "batch": batch,
        "created": created,
        "generator": {
            "package": "robust.rt.hydrolight",
            "version": generator_version,
        },
        "conventions": {
            "wave_nm": {
                "min": float(deck.WAVE_HL[0]),
                "max": float(deck.WAVE_HL[-1]),
                "step": 5.0,
                "n": int(deck.WAVE_HL.size),
                "l23_subset_is_bit_identical_above_nm": 350.0,
            },
            "theta_s_deg": zeniths,
            "view_geometry": "standard HydroLight quad layout; no custom quad file",
            "scenarios": {s: deck.SCENARIOS[s] for s in scenarios},
            "sky": deck.SKY,
            "precision": "float64 on all radiometric fields",
            "file_layout": "one netCDF per (water body, scenario), all zeniths inside",
        },
        "run_count_model": (
            "runs = water bodies x solar zeniths x scenarios; the view geometry "
            "is free because one full-HydroLight run returns the whole upwelling "
            "radiance distribution. CONFIRM WITH THE OPERATOR (spec Section 10, Q1)."
        ),
        "totals": {
            "water_bodies": len(bodies),
            "runs": sum(b.n_runs() for b in bodies),
            "blocks": sorted({b.block for b in bodies}),
        },
        "vsf_designs": [
            {
                "name": d.name,
                "kind": d.kind,
                "target_B_p": d.target_bp,
                "params": d.params,
                "table": f"vsf/{d.name}.txt" if d.tabulated else None,
                "builtin": d.builtin,
                "optional": d.optional,
                "delivered_B_p": d.realised_bp(),
            }
            for d in designs
        ],
        "water_bodies": [
            {
                "wbid": b.wbid,
                "block": b.block,
                "vsf_design": b.vsf_design,
                "scenarios": list(b.scenarios),
                "theta_s_deg": list(b.theta_s),
                "configs": list(b.configs),
                "phi_C": b.phi_C,
                "homogeneous": b.depth_profile is None,
                "want_depth_output": b.want_depth_output,
                "n_runs": b.n_runs(),
                "provenance": b.provenance,
            }
            for b in bodies
        ],
        "files": [
            {"path": str(pathlib.Path(f).name), "sha256": sha256(f)}
            for f in sorted(map(str, files))
        ],
        "metadata_contract": {
            key: (
                {"supplied_by": who, "status": "recorded"}
                if who == "ours"
                else {
                    "supplied_by": who,
                    "status": "pending",
                    "reason": (
                        "lives in HydroLight's own data files or is a property of "
                        "the run; requested from the operator with batch 0 "
                        "(spec Section 10)"
                    ),
                }
            )
            for key, who in METADATA_CONTRACT.items()
        },
        "operator_questions": (
            "See design/hydrolight_runs.md Section 10 -- ten questions, ordered "
            "by how much of the specification each answer moves."
        ),
    }


def write(path, manifest):
    """Write a manifest as deterministic JSON (sorted keys, trailing newline)."""
    path = pathlib.Path(path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
