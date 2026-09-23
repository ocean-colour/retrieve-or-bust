"""
The run intermediate representation, its serialiser, and the format adapter.

**Why an IR at all.** We do not yet know HydroLight 6's deck syntax -- that is
spec Section 10 question 7, "one example deck and its outputs from the working
setup", and it will not be answered until the operator replies. Blocking the
whole generator on it would be a choice to do nothing. So everything physical
lives in a **format-neutral** record here, and a thin adapter renders it into
whatever the operator actually wants, written last and replaceable in an
afternoon.

If the adapter is never written, the IR plus the ``beta_tilde`` and IOP tables
are still shippable: they are the physics, and the physics is format-light.

**Determinism.** ``np.savez`` stores a zip whose entries carry the wall-clock
time, so two runs of the same generator produce different bytes and every hash
in the manifest becomes decorative. :func:`write_arrays` writes the same zip
container with a fixed timestamp instead, so regeneration is byte-identical and
the manifest's SHA-256s mean something.
"""

from __future__ import annotations

import dataclasses
import io
import json
import pathlib
import zipfile

import numpy as np

__all__ = [
    "WaterBody",
    "SCENARIOS",
    "THETA_S",
    "WAVE_HL",
    "SKY",
    "write_arrays",
    "write_water_body",
    "render_provisional_deck",
    "render_deck",
]

#: The 85-band delivery grid, nm (spec Section 2.2). Its 350-750 subset is
#: bit-identical to ``robust.rt.conventions.WAVE``, which is what makes the L23
#: reproduction test of spec Section 4 a like-for-like comparison; the extension
#: to 330 nm is run R6 and supplies Raman excitation for emission below 400 nm.
WAVE_HL = np.arange(330.0, 750.0 + 1e-9, 5.0)

#: Solar zeniths, degrees (spec Section 2.3). 0-70 is the sanctioned window; 80
#: is retained as a deliberate extrapolation shell, as PB24 holds 80/87.75.
THETA_S = (0, 10, 20, 30, 40, 50, 60, 70, 80)

#: Scenario switches, named rather than X-labelled (A10). The X mapping is
#: recorded so the loader pins switches and the label is a convenience.
SCENARIOS = {
    "S1": {"raman": False, "chl_fl": False, "cdom_fl": False, "l23_X": 1},
    "S2": {"raman": True, "chl_fl": False, "cdom_fl": False, "l23_X": 2},
    "S4": {"raman": True, "chl_fl": True, "cdom_fl": False, "l23_X": 4},
    "S5": {"raman": True, "chl_fl": True, "cdom_fl": True, "l23_X": None},
}

#: Sky and surface, fixed across the campaign (spec Section 2.5) and validated
#: rather than assumed: batch 0 compares the delivered ``Ed(0+)`` against
#: ``robust/rt/data/ed_l23.npz`` at 0/30/60 deg.
SKY = {
    "model": "hydrolight_semi_empirical_clear_sky",
    "wind_speed_m_s": 5.0,
    "cloud_fraction": 0.0,
    "bottom": "none_infinitely_deep",
}


@dataclasses.dataclass
class WaterBody:
    """One water body, in SI, on :data:`WAVE_HL` -- the unit of the IR.

    Attributes
    ----------
    wbid : str
        Stable identifier; becomes part of every filename.
    block : str
        Which block of the campaign it belongs to (``"A-grid"``, ``"A-fill"``,
        ``"B-L23"``, ``"B-design"``, ``"C-R4"``, ``"C-R5"``, ``"C-R8"``,
        ``"pilot-*"``).
    a, b, bb : ndarray
        Total absorption, scattering and backscattering, m^-1, on
        :data:`WAVE_HL`.
    a_ph, a_cdom, a_nap : ndarray or None
        The absorption decomposition (metadata item 9). None where the water
        body is a synthetic IOP-grid node with no component interpretation.
    b_p : ndarray
        Particle scattering, m^-1.
    vsf_design : str
        Name of the :mod:`robust.rt.hydrolight.vsf` design carried.
    scenarios : tuple of str
        Which of :data:`SCENARIOS` to run.
    theta_s : tuple of int
        Solar zeniths, degrees.
    configs : tuple of str
        Solver configurations to repeat the run under. ``("baseline",)`` for
        everything except R0's convergence sweep, which is the only place the
        *same* water body is run more than once at the same sky and scenario.
    phi_C : float
        Chlorophyll fluorescence quantum yield.
    depth_profile : dict or None
        None for a homogeneous column (everything but R8).
    want_depth_output : bool
        Whether to request ``K(z)`` profiles to >= 25 optical depths -- batch A
        only, and most of the difference between 310 GB and 780 GB delivered.
    provenance : dict
        Free-form record: how this water body was constructed, and every caveat
        that attaches to it. Carried as **data**, not as a comment, so the
        loader can read it and a test can assert it is present.
    """

    wbid: str
    block: str
    a: np.ndarray
    b: np.ndarray
    bb: np.ndarray
    b_p: np.ndarray
    vsf_design: str
    scenarios: tuple = ("S1",)
    theta_s: tuple = THETA_S
    configs: tuple = ("baseline",)
    phi_C: float = 0.02
    a_ph: np.ndarray | None = None
    a_cdom: np.ndarray | None = None
    a_nap: np.ndarray | None = None
    depth_profile: dict | None = None
    want_depth_output: bool = False
    provenance: dict = dataclasses.field(default_factory=dict)

    def n_runs(self) -> int:
        """Runs this water body costs: zeniths x scenarios x configurations."""
        return len(self.theta_s) * len(self.scenarios) * len(self.configs)

    def arrays(self) -> dict:
        """The numeric payload, as a name -> float64 array mapping."""
        out = {
            "wave": WAVE_HL,
            "a": self.a,
            "b": self.b,
            "bb": self.bb,
            "b_p": self.b_p,
        }
        for name in ("a_ph", "a_cdom", "a_nap"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

    def metadata(self) -> dict:
        """The JSON sidecar -- everything that is not an array."""
        return {
            "wbid": self.wbid,
            "block": self.block,
            "vsf_design": self.vsf_design,
            "scenarios": list(self.scenarios),
            "theta_s_deg": list(self.theta_s),
            "configs": list(self.configs),
            "phi_C": self.phi_C,
            "depth_profile": self.depth_profile,
            "want_depth_output": self.want_depth_output,
            "homogeneous": self.depth_profile is None,
            "sky": SKY,
            "wave_nm": {"min": 330.0, "max": 750.0, "step": 5.0, "n": len(WAVE_HL)},
            "scenario_switches": {k: SCENARIOS[k] for k in self.scenarios},
            "n_runs": self.n_runs(),
            "provenance": self.provenance,
        }


#: Fixed zip timestamp so regeneration is byte-identical (see module docstring).
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def write_arrays(path, arrays):
    """Write a ``.npz``-compatible archive with **deterministic bytes**.

    Parameters
    ----------
    path : str or Path
        Destination, ``.npz``.
    arrays : dict
        Name -> array.
    """
    path = pathlib.Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())


def write_water_body(directory, wb):
    """Serialise one :class:`WaterBody` as a JSON sidecar plus an NPZ.

    Returns
    -------
    list of Path
        The files written, in a stable order.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    js = directory / f"{wb.wbid}.json"
    npz = directory / f"{wb.wbid}.npz"
    js.write_text(json.dumps(wb.metadata(), indent=2, sort_keys=True) + "\n")
    write_arrays(npz, wb.arrays())
    return [js, npz]


def render_provisional_deck(wb, theta_s, scenario):
    """A readable, **provisional** rendering of one run -- not HydroLight syntax.

    Exists so the operator can read what we are asking for before the real
    adapter exists, and so a reviewer can see that the IR carries everything a
    deck needs. It is deliberately not called a deck anywhere it could be
    mistaken for one.

    Returns
    -------
    str
    """
    sw = SCENARIOS[scenario]
    lines = [
        "# PROVISIONAL run description -- NOT a HydroLight input deck.",
        "# Generated by robust.rt.hydrolight; awaiting the operator's template",
        "# (spec Section 10, question 7).",
        f"water_body        {wb.wbid}",
        f"block             {wb.block}",
        f"scenario          {scenario}  (L23 X = {sw['l23_X']})",
        f"  raman           {sw['raman']}",
        f"  chl_fluor       {sw['chl_fl']}   phi_C = {wb.phi_C}",
        f"  cdom_fluor      {sw['cdom_fl']}",
        f"solar_zenith_deg  {theta_s}",
        f"sky_model         {SKY['model']}",
        f"wind_speed_m_s    {SKY['wind_speed_m_s']}",
        f"bottom            {SKY['bottom']}",
        "water_column      "
        + ("homogeneous" if wb.depth_profile is None else "profiled"),
        f"wavelengths_nm    330:750:5  ({len(WAVE_HL)} bands)",
        f"vsf_design        {wb.vsf_design}",
        f"iop_table         {wb.wbid}.npz  (a, b, bb, b_p on the 85-band grid)",
        f"beta_tilde_table  vsf/{wb.vsf_design}.txt",
        "output            Rrs(above), rrs(below), Lu(0-) at all quads,",
        "                  Ed(0+), Ed(0-), Eu(0-), K_d, K_u, K_Lu, mu_d, mu_u, mu_tot,",
        "                  and the IOPs as received",
    ]
    if wb.want_depth_output:
        lines.append(
            "                  PLUS K_d(z), K_u(z), K_Lu(z), mu_bar(z) to >= 25 "
            "optical depths,"
        )
        lines.append(
            "                  and the asymptotic radiance distribution and K_inf "
            "if the build reports them"
        )
    return "\n".join(lines) + "\n"


def render_deck(wb, theta_s, scenario, template=None):
    """Render one run as a real HydroLight input deck.

    Not implemented: the template is spec Section 10 question 7 and has not
    arrived. This raises rather than guessing, because a deck that is *nearly*
    right is worse than none -- it would run, and quietly answer a different
    question.

    Raises
    ------
    NotImplementedError
        Always, until ``template`` is a real operator-supplied deck.
    """
    if template is None:
        raise NotImplementedError(
            "HydroLight deck syntax is not known yet (spec Section 10, Q7: "
            "'one example deck and its outputs from the working setup'). "
            "Use render_provisional_deck() for a readable description, and ship "
            "the IR plus the beta_tilde and IOP tables, which are format-light."
        )
    raise NotImplementedError(
        "adapter stub: fill this in against the operator's template"
    )
