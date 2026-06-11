"""Unit tests for cell-cycle phase thresholding.

Focus: the H3P-first mitosis rule. Mitotic chromatin condensation and cell
rounding inflate the integrated DNA and spill into the EdU channel, so a
genuine mitotic cell's DNA/EdU read out of the interphase windows. H3P
positivity must therefore take precedence over the DNA/EdU gates, otherwise
mitotic cells are mis-called Polyploid or S (glaring on MG132-arrest plates).
"""

import pandas as pd

from omero_screen.cellcycle_analysis import _thresholdingH3

_DNA = "integrated_int_DAPI_norm"
_EDU = "intensity_mean_EdU_nucleus_norm"
_H3P = "intensity_mean_H3P_nucleus_norm"


def _row(dna: float, edu: float, h3p: float) -> pd.Series:
    return pd.Series({_DNA: dna, _EDU: edu, _H3P: h3p})


class TestH3PFirstMitosis:
    def test_h3p_positive_with_inflated_dna_is_M_not_polyploid(self) -> None:
        """The core fix: a mitotic cell whose DNA over-reads (>5.5) is M."""
        assert _thresholdingH3(_row(dna=6.4, edu=4.0, h3p=31.0)) == "M"

    def test_h3p_positive_with_inflated_edu_is_M_not_S(self) -> None:
        """A mitotic cell spilling into EdU (>3) at 4N is M, not Late S."""
        assert _thresholdingH3(_row(dna=4.0, edu=4.0, h3p=31.0)) == "M"

    def test_h3p_positive_clean_4n_is_M(self) -> None:
        assert _thresholdingH3(_row(dna=4.0, edu=1.0, h3p=31.0)) == "M"

    def test_h3p_positive_below_dna_floor_is_not_M(self) -> None:
        """DAPI < 3 floor excludes debris / Sub-G1 with a stray bright pixel."""
        assert _thresholdingH3(_row(dna=2.0, edu=1.0, h3p=31.0)) == "G1"

    def test_sub_g1_wins_over_h3p(self) -> None:
        assert _thresholdingH3(_row(dna=1.0, edu=1.0, h3p=31.0)) == "Sub-G1"


class TestH3PNegativeUnchanged:
    """H3P-negative cells must classify exactly as before the reordering."""

    def test_g1(self) -> None:
        assert _thresholdingH3(_row(dna=2.0, edu=1.0, h3p=1.0)) == "G1"

    def test_g2_not_mistaken_for_m(self) -> None:
        assert _thresholdingH3(_row(dna=4.0, edu=1.0, h3p=1.0)) == "G2"

    def test_early_s(self) -> None:
        assert _thresholdingH3(_row(dna=2.0, edu=5.0, h3p=1.0)) == "Early S"

    def test_late_s(self) -> None:
        assert _thresholdingH3(_row(dna=4.0, edu=5.0, h3p=1.0)) == "Late S"

    def test_polyploid_non_replicating(self) -> None:
        assert (
            _thresholdingH3(_row(dna=6.0, edu=1.0, h3p=1.0))
            == "Polyploid (non-replicating)"
        )

    def test_polyploid_replicating(self) -> None:
        assert (
            _thresholdingH3(_row(dna=6.0, edu=5.0, h3p=1.0))
            == "Polyploid (replicating)"
        )
