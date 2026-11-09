"""PIP-FUCCI phase calling on synthetic tracks with known boundaries."""

import numpy as np
import pandas as pd
import pytest
from omero_screen_track.phases import (
    PhaseParams,
    annotate_phases,
    fit_reference,
)

PIP_HIGH, GEM_FLOOR, GEM_HIGH, BLEED = 1500.0, 30.0, 1200.0, 0.17


def _signals(phases: list[tuple[str, int]], rng, noise: float = 0.06):
    """PIP and geminin per frame for a sequence of (phase, frames).

    G1: PIP high, geminin floor. S: PIP low, geminin rising linearly to
    ~70 % of high. G2: PIP back up, geminin high. X: like G1 (after an exit).
    Geminin carries BLEED × PIP; multiplicative noise on both.
    """
    pip, gem = [], []
    for name, n in phases:
        for i in range(n):
            if name == "G1":
                p, g = PIP_HIGH, GEM_FLOOR
            elif name == "X":
                # Exit straight from S: geminin decays over ~10 frames while
                # PIP is still degraded; PIP returns only once geminin is gone.
                f = min(1.0, (i + 1) / 10)
                p = (
                    0.12 if i < 10 else 0.12 + 0.88 * min(1.0, (i - 9) / 6)
                ) * PIP_HIGH
                g = 0.7 * GEM_HIGH * (1 - f) + GEM_FLOOR * f
            elif name == "XG2":
                # Exit after G2 entry: PIP already back, geminin decays.
                f = min(1.0, (i + 1) / 10)
                p = 0.8 * PIP_HIGH
                g = GEM_HIGH * (1 - f) + GEM_FLOOR * f
            elif name == "S":
                p = 0.12 * PIP_HIGH
                g = GEM_FLOOR + (0.7 * GEM_HIGH - GEM_FLOOR) * (i + 1) / n
            else:  # G2
                p, g = 0.8 * PIP_HIGH, GEM_HIGH
            pip.append(p * (1 + noise * rng.standard_normal()))
            gem.append((g + BLEED * p) * (1 + noise * rng.standard_normal()))
    return np.array(pip), np.array(gem)


def _well(
    tracks: dict[int, list[tuple[str, int]]], start: int = 72, seed: int = 0
):
    """Cells and states tables for a set of synthetic tracks."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid, phases in tracks.items():
        pip, gem = _signals(phases, rng)
        for i, (p, g) in enumerate(zip(pip, gem, strict=True)):
            rows.append(
                {"cell": cid, "timepoint": start + i, "pip": p, "geminin": g}
            )
    states = pd.DataFrame(rows)
    cells = pd.DataFrame({"cell": sorted(tracks)})
    return cells, states


def _cycling_well(
    extra: dict[int, list[tuple[str, int]]] | None = None, n: int = 12
):
    """A well of complete G1 → S → G2 cycles of varied length, plus extras."""
    tracks = {}
    for c in range(1, n + 1):
        tracks[c] = [("G1", 15 + c), ("S", 22 + c % 5), ("G2", 10 + c % 3)]
    tracks.update(extra or {})
    return _well(tracks)


def test_bleed_through_and_scales_are_recovered() -> None:
    """The well's geminin-from-PIP factor is fitted from G1-like frames."""
    _, states = _cycling_well()
    ref = fit_reference(states, PhaseParams())
    assert ref.bleed == pytest.approx(BLEED, abs=0.04)
    assert ref.pip_high == pytest.approx(PIP_HIGH, rel=0.15)


def test_complete_cycle_boundaries_inside_their_intervals() -> None:
    """S entry and G2 entry land within a frame or two of the truth."""
    cells, states = _cycling_well()
    out, st, _ = annotate_phases(cells, states)
    for c in range(1, 13):
        row = out.set_index("cell").loc[c]
        s_true = 72 + 15 + c
        g2_true = s_true + 22 + c % 5
        assert row.s_entry_state == "inside" and row.g2_entry_state == "inside"
        assert abs(row.s_entry - s_true) <= 2
        assert abs(row.g2_entry - g2_true) <= 3
        assert row.g2_entry_lo <= row.g2_entry <= row.g2_entry_hi
        assert row.start_phase == "G1" and row.end_phase == "G2"


def test_called_phases_follow_the_cycle_order() -> None:
    """Per-frame calls never go backwards along a track."""
    cells, states = _cycling_well()
    _, st, _ = annotate_phases(cells, states)
    order = {"G1": 0, "S": 1, "G2": 2, "E": 3, "X": 4}
    for _, g in st.groupby("cell"):
        seq = g.sort_values("timepoint")["phase"].map(order).to_numpy()
        assert (np.diff(seq) >= 0).all()


def test_p_g2_or_later_only_rises() -> None:
    """In the left-to-right model P(≥G2) is a CDF along the track."""
    cells, states = _cycling_well()
    _, st, _ = annotate_phases(cells, states)
    for _, g in st.groupby("cell"):
        cdf = (
            g.sort_values("timepoint")[["p_g2", "p_e", "p_x"]].sum(axis=1)
        ).to_numpy()
        assert (np.diff(cdf) >= -1e-9).all()


def test_g1_arrested_cell_stays_g1() -> None:
    """A cell held in G1 for the whole window is not given an S phase.

    Well-level normalisation; per-track min-max scaling would invent one.
    """
    cells, states = _cycling_well({99: [("G1", 96)]})
    out, st, _ = annotate_phases(cells, states)
    row = out.set_index("cell").loc[99]
    assert row.s_entry_state == "after" and row.start_phase == "G1"
    assert set(st[st.cell == 99].phase) == {"G1"}


def test_track_starting_in_s_is_left_censored() -> None:
    """A cell already in S at the window start has no S entry but a G2 entry."""
    cells, states = _cycling_well({98: [("S", 14), ("G2", 12)]})
    out, _, _ = annotate_phases(cells, states)
    row = out.set_index("cell").loc[98]
    assert row.s_entry_state == "before"
    assert (
        row.g2_entry_state == "inside" and abs(row.g2_entry - (72 + 14)) <= 3
    )
    # S started before the track, so it has no full S length.
    assert np.isnan(row.s_hours)


def test_exit_from_s_is_x_not_g2() -> None:
    """Geminin lost while PIP is still degraded is an exit from S, not G2.

    The exit is timed at the onset of the geminin decay.
    """
    cells, states = _cycling_well({97: [("G1", 15), ("S", 24), ("X", 30)]})
    out, st, _ = annotate_phases(cells, states)
    row = out.set_index("cell").loc[97]
    onset = 72 + 15 + 24
    assert row.exit_state == "inside" and abs(row.exit - onset) <= 3
    assert row.g2_entry_state != "inside"
    assert "G2" not in set(st[st.cell == 97].phase)


def test_exit_after_pip_returns_passes_through_g2() -> None:
    """Under Rob's convention, PIP back before geminin is lost means G2 first."""
    cells, states = _cycling_well(
        {95: [("G1", 15), ("S", 24), ("G2", 12), ("XG2", 30)]}
    )
    out, _, _ = annotate_phases(cells, states)
    row = out.set_index("cell").loc[95]
    assert (
        row.g2_entry_state == "inside" and abs(row.g2_entry - (72 + 39)) <= 3
    )
    assert row.exit_state == "inside" and abs(row.exit - (72 + 51)) <= 3


def test_one_frame_spike_does_not_make_an_s_phase() -> None:
    """A single PIP dip (a compaction or segmentation artefact) is not S."""
    cells, states = _cycling_well({96: [("G1", 60)]})
    i = states.index[(states.cell == 96) & (states.timepoint == 100)][0]
    states.loc[i, "pip"] *= 0.1
    out, st, _ = annotate_phases(cells, states)
    assert set(st[st.cell == 96].phase) == {"G1"}


def test_gaps_are_bridged_not_skipped() -> None:
    """Frames without a mask get posteriors but no features."""
    cells, states = _cycling_well()
    states = states[~((states.cell == 3) & states.timepoint.between(95, 97))]
    out, st, _ = annotate_phases(cells, states)
    assert out.set_index("cell").loc[3].s_entry_state == "inside"
    assert len(st[st.cell == 3]) == len(states[states.cell == 3])


def test_durations_carry_the_truth() -> None:
    """Sampled S durations bracket the true S length."""
    cells, states = _cycling_well()
    out, _, _ = annotate_phases(cells, states)
    hours = PhaseParams().interval_minutes / 60
    for c in (2, 7, 11):
        row = out.set_index("cell").loc[c]
        true = (22 + c % 5) * hours
        assert row.s_hours_lo - 1.0 <= true <= row.s_hours_hi + 1.0


def test_rob_rule_agrees_on_clean_cycles() -> None:
    """Rob's derivative rule sits near the model's S/G2 median on clean data."""
    cells, states = _cycling_well()
    out, _, _ = annotate_phases(cells, states)
    diff = (out["g2_entry_rob"] - out["g2_entry"]).abs()
    assert diff.median() <= 4
