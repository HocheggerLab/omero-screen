"""Tests for cellview.utils.nucleus_channel helpers.

Covers the pure detection / validation / rename helpers. The OMERO plate
lookup and interactive prompt paths are exercised via mocked inputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cellview.utils.error_classes import DataError, StateError
from cellview.utils.nucleus_channel import (
    detect_nucleus_candidates,
    prompt_nucleus_channel,
    rename_nucleus_to_dapi,
    resolve_nucleus_channel_from_plate,
    validate_nucleus_channel_in_df,
)


# ------------------------------------------------------------------
# detect_nucleus_candidates
# ------------------------------------------------------------------


class TestDetectNucleusCandidates:
    def test_dapi_only(self):
        df = pd.DataFrame(
            columns=[
                "intensity_mean_DAPI_nucleus",
                "intensity_min_DAPI_nucleus",
                "intensity_max_DAPI_nucleus",
                "intensity_mean_EdU_nucleus",
                "area_nucleus",
            ]
        )
        # EdU also appears in a nucleus column — it's a valid candidate at
        # this stage (the resolver / user picks the actual nucleus channel).
        assert detect_nucleus_candidates(df) == ["DAPI", "EdU"]

    def test_hoechst(self):
        df = pd.DataFrame(
            columns=[
                "intensity_mean_Hoechst_nucleus",
                "intensity_min_Hoechst_nucleus",
            ]
        )
        assert detect_nucleus_candidates(df) == ["Hoechst"]

    def test_h2b_rfp_underscore(self):
        df = pd.DataFrame(columns=["intensity_mean_H2B_RFP_nucleus"])
        assert detect_nucleus_candidates(df) == ["H2B_RFP"]

    def test_no_nucleus_columns(self):
        df = pd.DataFrame(
            columns=["area_nucleus", "intensity_mean_p21_cell", "label"]
        )
        assert detect_nucleus_candidates(df) == []

    def test_preserves_order_no_duplicates(self):
        df = pd.DataFrame(
            columns=[
                "intensity_mean_EdU_nucleus",
                "intensity_mean_DAPI_nucleus",
                "intensity_min_DAPI_nucleus",
            ]
        )
        assert detect_nucleus_candidates(df) == ["EdU", "DAPI"]


# ------------------------------------------------------------------
# resolve_nucleus_channel_from_plate (E1, E2, E4)
# ------------------------------------------------------------------


class TestResolveNucleusChannelFromPlate:
    def _make_plate(self, plate_id: int = 42) -> MagicMock:
        plate = MagicMock()
        plate.getId.return_value = plate_id
        return plate

    def test_resolves_dapi(self):
        plate = self._make_plate()
        with patch(
            "omero_utils.map_anns.parse_annotations",
            return_value={"DAPI": "0", "Tub": "1", "EdU": "2"},
        ):
            assert resolve_nucleus_channel_from_plate(plate) == "DAPI"

    def test_resolves_hoechst(self):
        plate = self._make_plate()
        with patch(
            "omero_utils.map_anns.parse_annotations",
            return_value={"Hoechst": "0", "EdU": "1"},
        ):
            assert resolve_nucleus_channel_from_plate(plate) == "Hoechst"

    def test_resolves_h2b_rfp(self):
        plate = self._make_plate()
        with patch(
            "omero_utils.map_anns.parse_annotations",
            return_value={"H2B_RFP": "0", "Tub": "1"},
        ):
            assert resolve_nucleus_channel_from_plate(plate) == "H2B_RFP"

    def test_strips_nucleus_suffix_from_annotation(self):
        """Annotation uses _nucleus suffix convention; resolver strips it.

        Mirrors the column-token rule applied by omero_screen.image_analysis,
        so the returned name matches the actual feature column name
        (``intensity_mean_SirDNA_nucleus``, not the doubled-up variant).
        """
        plate = self._make_plate()
        with patch(
            "omero_utils.map_anns.parse_annotations",
            return_value={"SirDNA_nucleus": "0", "EdU": "1"},
        ):
            assert (
                resolve_nucleus_channel_from_plate(plate) == "SirDNA"
            )

    def test_strips_cell_suffix_case_insensitive(self):
        plate = self._make_plate()
        with patch(
            "omero_utils.map_anns.parse_annotations",
            return_value={"Hoechst_NUCLEUS": "0", "Tub": "1"},
        ):
            assert resolve_nucleus_channel_from_plate(plate) == "Hoechst"

    def test_no_annotation_raises_data_error(self):
        """E1: plate has no channel annotation → hard fail."""
        plate = self._make_plate(plate_id=99)
        with patch(
            "omero_utils.map_anns.parse_annotations",
            return_value={},
        ):
            with pytest.raises(DataError, match="no channel map annotations"):
                resolve_nucleus_channel_from_plate(plate)

    def test_no_nucleus_role_raises_data_error(self):
        """E2: annotation exists but no channel resolves to nucleus role."""
        plate = self._make_plate(plate_id=7)
        with patch(
            "omero_utils.map_anns.parse_annotations",
            return_value={"GFP": "0", "RFP": "1"},
        ):
            with pytest.raises(
                DataError, match="does not resolve to a nucleus role"
            ):
                resolve_nucleus_channel_from_plate(plate)

    def test_duplicate_nucleus_propagates(self):
        """E4: two channels claim the nucleus role — resolver-side error."""
        plate = self._make_plate()
        with patch(
            "omero_utils.map_anns.parse_annotations",
            return_value={"DAPI": "0", "Hoechst": "1"},
        ):
            with pytest.raises(
                DataError, match="does not resolve to a nucleus role"
            ):
                resolve_nucleus_channel_from_plate(plate)


# ------------------------------------------------------------------
# prompt_nucleus_channel (E5, E6, E7, E9, E10)
# ------------------------------------------------------------------


class TestPromptNucleusChannel:
    def test_empty_candidates_raises(self):
        """E5: CSV with no nucleus columns at all."""
        with pytest.raises(
            StateError, match="no nucleus-segmentation columns"
        ):
            prompt_nucleus_channel([])

    def test_cli_flag_wins(self):
        assert (
            prompt_nucleus_channel(["DAPI", "Hoechst"], cli_flag="Hoechst")
            == "Hoechst"
        )

    def test_cli_flag_validated_against_candidates(self):
        """E10: flag doesn't match any discovered candidate."""
        with pytest.raises(StateError, match="not found in CSV"):
            prompt_nucleus_channel(["DAPI"], cli_flag="Hoechst")

    def test_non_interactive_without_flag_raises(self):
        """E9: scripted environment must pass --nucleus-channel."""
        with patch(
            "cellview.utils.nucleus_channel.sys.stdin.isatty",
            return_value=False,
        ):
            with pytest.raises(StateError, match="non-interactive"):
                prompt_nucleus_channel(["DAPI", "Hoechst"])

    def test_interactive_single_candidate_uses_default(self):
        """E7: one candidate → it's the default for the prompt."""
        with patch(
            "cellview.utils.nucleus_channel.sys.stdin.isatty",
            return_value=True,
        ):
            with patch(
                "cellview.utils.nucleus_channel.Prompt.ask",
                return_value="Hoechst",
            ) as mock_ask:
                result = prompt_nucleus_channel(["Hoechst"])
                assert result == "Hoechst"
                _, kwargs = mock_ask.call_args
                assert kwargs["default"] == "Hoechst"

    def test_interactive_multiple_candidates_no_default(self):
        """E6: multiple candidates — no default, force the user to pick."""
        with patch(
            "cellview.utils.nucleus_channel.sys.stdin.isatty",
            return_value=True,
        ):
            with patch(
                "cellview.utils.nucleus_channel.Prompt.ask",
                return_value="DAPI",
            ) as mock_ask:
                result = prompt_nucleus_channel(["DAPI", "Hoechst"])
                assert result == "DAPI"
                _, kwargs = mock_ask.call_args
                # No default passed → user must explicitly pick.
                assert "default" not in kwargs
                assert kwargs["choices"] == ["DAPI", "Hoechst"]


# ------------------------------------------------------------------
# validate_nucleus_channel_in_df (E3)
# ------------------------------------------------------------------


class TestValidateNucleusChannelInDf:
    def test_passes_when_column_present(self):
        df = pd.DataFrame(columns=["intensity_mean_DAPI_nucleus"])
        validate_nucleus_channel_in_df("DAPI", df)  # does not raise

    def test_raises_when_annotation_mismatches_csv(self):
        """E3: plate says Hoechst, CSV has DAPI columns."""
        df = pd.DataFrame(columns=["intensity_mean_DAPI_nucleus"])
        with pytest.raises(DataError, match="not found in the CSV"):
            validate_nucleus_channel_in_df("Hoechst", df)


# ------------------------------------------------------------------
# rename_nucleus_to_dapi (E12, E13, normal path)
# ------------------------------------------------------------------


class TestRenameNucleusToDapi:
    def test_dapi_is_noop(self):
        """E12: legacy DAPI plate — no rename needed."""
        df = pd.DataFrame({
            "intensity_mean_DAPI_nucleus": [1.0],
            "intensity_max_DAPI_cell": [2.0],
        })
        out = rename_nucleus_to_dapi(df, "DAPI")
        assert list(out.columns) == list(df.columns)

    def test_renames_hoechst_columns(self):
        df = pd.DataFrame({
            "intensity_min_Hoechst_nucleus": [1.0],
            "intensity_mean_Hoechst_nucleus": [1.0],
            "intensity_max_Hoechst_nucleus": [1.0],
            "intensity_min_Hoechst_cell": [1.0],
            "intensity_mean_Hoechst_cell": [1.0],
            "intensity_max_Hoechst_cell": [1.0],
            "intensity_min_Hoechst_cyto": [1.0],
            "intensity_mean_Hoechst_cyto": [1.0],
            "intensity_max_Hoechst_cyto": [1.0],
            "integrated_int_Hoechst": [10.0],
            "integrated_int_Hoechst_norm": [2.0],
        })
        out = rename_nucleus_to_dapi(df, "Hoechst")
        expected = {
            "intensity_min_DAPI_nucleus",
            "intensity_mean_DAPI_nucleus",
            "intensity_max_DAPI_nucleus",
            "intensity_min_DAPI_cell",
            "intensity_mean_DAPI_cell",
            "intensity_max_DAPI_cell",
            "intensity_min_DAPI_cyto",
            "intensity_mean_DAPI_cyto",
            "intensity_max_DAPI_cyto",
            "integrated_int_DAPI",
            "integrated_int_DAPI_norm",
        }
        assert set(out.columns) == expected

    def test_leaves_unrelated_columns_alone(self):
        df = pd.DataFrame({
            "intensity_mean_Hoechst_nucleus": [1.0],
            "intensity_mean_EdU_nucleus": [1.0],
            "intensity_mean_p21_cell": [1.0],
            "area_nucleus": [42.0],
        })
        out = rename_nucleus_to_dapi(df, "Hoechst")
        assert "intensity_mean_DAPI_nucleus" in out.columns
        assert "intensity_mean_EdU_nucleus" in out.columns
        assert "intensity_mean_p21_cell" in out.columns
        assert "area_nucleus" in out.columns

    def test_underscore_channel_name(self):
        df = pd.DataFrame({"intensity_mean_H2B_RFP_nucleus": [1.0]})
        out = rename_nucleus_to_dapi(df, "H2B_RFP")
        assert "intensity_mean_DAPI_nucleus" in out.columns
        assert "intensity_mean_H2B_RFP_nucleus" not in out.columns

    def test_clobber_detection_raises(self):
        """E13: CSV mixes DAPI and non-DAPI nucleus columns."""
        df = pd.DataFrame({
            "intensity_mean_Hoechst_nucleus": [1.0],
            "intensity_mean_DAPI_nucleus": [2.0],
        })
        with pytest.raises(DataError, match="already contains columns"):
            rename_nucleus_to_dapi(df, "Hoechst")

    def test_returns_copy_not_inplace(self):
        df = pd.DataFrame({"intensity_mean_Hoechst_nucleus": [1.0]})
        original_cols = list(df.columns)
        rename_nucleus_to_dapi(df, "Hoechst")
        assert list(df.columns) == original_cols
