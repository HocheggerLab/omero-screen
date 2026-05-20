"""Tests for user-friendly error messages across the omero-screen pipeline.

Validates that common metadata mistakes (wrong channel indices, missing columns,
missing well annotations) produce clear, actionable error messages rather than
cryptic Python exceptions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from omero.gateway import PlateWrapper

from omero_screen.cellcycle_analysis import cellcycle_analysis
from omero_screen.metadata_parser import MetadataParser

# -------------------- MockParser (reused from test_metadata.py) --------------------


class MockParser(MetadataParser):
    """Mock parser class that skips OMERO connection."""

    def __init__(self, channel_data=None, well_data=None, plate_wells=None):  # noqa: D107
        self.channel_data = channel_data if channel_data is not None else {}
        self.well_data = well_data if well_data is not None else {}
        self._empty_well_positions: list[str] = []

        class MockWell:
            def __init__(self, pos):
                self.pos = pos

            def getWellPos(self):
                return self.pos

        class MockPlate:
            def __init__(self, wells):
                self.wells = wells

            def listChildren(self):
                return [MockWell(pos) for pos in wells]

        wells = plate_wells or []
        self.plate = MockPlate(wells)

    def _get_plate(self) -> PlateWrapper:
        return self.plate


# -------------------- Channel Index Validation --------------------


class TestChannelIndexValidation:
    """Tests for channel index validation in metadata_parser."""

    def test_non_consecutive_indices_detected(self):
        """Test that non-consecutive indices like 0, 2, 3, 4 are caught."""
        parser = MockParser({"DAPI": "0", "EdU": "2", "H3P": "3", "Tub": "4"})
        errors = parser._validate_channel_data()
        assert any("must be 0 to 3" in e for e in errors)
        assert any("0, 2, 3, 4" in e for e in errors)

    def test_indices_starting_from_one_detected(self):
        """Test that 1-based indices (1, 2, 3) are caught for 3 channels."""
        parser = MockParser({"DAPI": "1", "Tub": "2", "EdU": "3"})
        errors = parser._validate_channel_data()
        assert any("must be 0 to 2" in e for e in errors)
        assert any("1, 2, 3" in e for e in errors)

    def test_non_integer_index_detected(self):
        """Test that non-integer indices produce a clear error."""
        parser = MockParser({"DAPI": "zero", "GFP": "1"})
        errors = parser._validate_channel_data()
        assert any("non-integer index" in e for e in errors)
        assert any("'zero'" in e for e in errors)

    def test_valid_consecutive_indices_pass(self):
        """Test that valid consecutive indices 0, 1, 2 pass validation."""
        parser = MockParser({"DAPI": "0", "Tub": "1", "EdU": "2"})
        errors = parser._validate_channel_data()
        assert not errors

    def test_valid_two_channels_pass(self):
        """Test that valid two-channel setup passes."""
        parser = MockParser({"DAPI": "0", "GFP": "1"})
        errors = parser._validate_channel_data()
        assert not errors

    def test_single_channel_valid(self):
        """Test that a single channel with index 0 passes."""
        parser = MockParser({"DAPI": "0"})
        errors = parser._validate_channel_data()
        assert not errors

    def test_single_channel_wrong_index(self):
        """Test that a single channel with index 1 fails."""
        parser = MockParser({"DAPI": "1"})
        errors = parser._validate_channel_data()
        assert any("must be 0 to 0" in e for e in errors)

    def test_duplicate_indices_detected(self):
        """Test that duplicate indices (e.g. 0, 0, 1) are caught."""
        parser = MockParser({"DAPI": "0", "Tub": "0", "EdU": "1"})
        errors = parser._validate_channel_data()
        assert any("must be 0 to 2" in e for e in errors)

    def test_error_message_mentions_excel(self):
        """Test that error messages guide user to check Excel metadata."""
        parser = MockParser({"DAPI": "0", "EdU": "2", "H3P": "3", "Tub": "4"})
        errors = parser._validate_channel_data()
        assert any("Excel" in e or "consecutive" in e for e in errors)


# -------------------- Channel Index vs Image Validation --------------------


class TestChannelIndexVsImage:
    """Tests for validating channel indices against actual image channel count."""

    def test_index_exceeds_image_channels(self):
        """Test that an index beyond actual image channels is caught."""
        parser = MockParser({"DAPI": "0", "Tub": "1", "EdU": "2", "H3P": "3"})

        # Mock _get_first_image to return an image with only 3 channels
        mock_image = MagicMock()
        mock_image.getSizeC.return_value = 3
        with patch.object(parser, "_get_first_image", return_value=mock_image):
            errors = parser._validate_channel_indices_against_image()

        assert len(errors) == 1
        assert "H3P" in errors[0]
        assert "index 3" in errors[0]
        assert "3 channels" in errors[0]
        assert "0-2" in errors[0]

    def test_all_indices_within_range(self):
        """Test that valid indices produce no errors."""
        parser = MockParser({"DAPI": "0", "Tub": "1", "EdU": "2"})

        mock_image = MagicMock()
        mock_image.getSizeC.return_value = 4  # 4 channels, indices 0-3 valid
        with patch.object(parser, "_get_first_image", return_value=mock_image):
            errors = parser._validate_channel_indices_against_image()

        assert not errors

    def test_negative_index_detected(self):
        """Test that negative indices are caught."""
        parser = MockParser({"DAPI": "-1", "GFP": "0"})

        mock_image = MagicMock()
        mock_image.getSizeC.return_value = 2
        with patch.object(parser, "_get_first_image", return_value=mock_image):
            errors = parser._validate_channel_indices_against_image()

        assert len(errors) == 1
        assert "DAPI" in errors[0]
        assert "index -1" in errors[0]

    def test_graceful_when_no_image_available(self):
        """Test that validation is skipped when image cannot be retrieved."""
        parser = MockParser({"DAPI": "0", "GFP": "5"})
        # _get_first_image will raise because MockParser has no real wells
        errors = parser._validate_channel_indices_against_image()
        # Should not raise, just return empty (validation skipped)
        assert not errors


# -------------------- Well Conditions Lookup --------------------


class TestWellConditionsErrors:
    """Tests for well_conditions error messages."""

    def test_missing_well_raises_with_available_wells(self):
        """Test that looking up a non-existent well lists available wells."""
        parser = MockParser(
            well_data={
                "Well": ["A1", "A2", "B1"],
                "cell_line": ["RPE", "RPE", "HeLa"],
                "condition": ["ctrl", "drug", "ctrl"],
            }
        )
        with pytest.raises(ValueError, match="Well 'C3' not found"):
            parser.well_conditions("C3")

    def test_error_lists_available_wells(self):
        """Test that the error message includes available well positions."""
        parser = MockParser(
            well_data={
                "Well": ["A1", "B2"],
                "cell_line": ["RPE", "RPE"],
                "condition": ["ctrl", "drug"],
            }
        )
        with pytest.raises(ValueError, match="Available wells.*A1.*B2"):
            parser.well_conditions("Z9")

    def test_valid_well_returns_conditions(self):
        """Test that a valid well lookup works correctly."""
        parser = MockParser(
            well_data={
                "Well": ["A1", "A2"],
                "cell_line": ["RPE", "HeLa"],
                "condition": ["ctrl", "drug"],
            }
        )
        result = parser.well_conditions("A2")
        assert result["cell_line"] == "HeLa"
        assert result["condition"] == "drug"


# -------------------- Flatfield Channel Bounds Check --------------------


class TestFlatfieldChannelBounds:
    """Tests for channel index bounds checking in flatfield_corr."""

    def test_channel_out_of_range_error_message(self):
        """Test that out-of-range channel index produces a helpful error."""
        from omero_screen.flatfield_corr import aggregate_imgs

        mock_conn = MagicMock()
        mock_image = MagicMock()
        mock_image.getSizeX.return_value = 512
        mock_image.getSizeY.return_value = 512
        mock_image.getSizeZ.return_value = 1
        mock_image.getSizeC.return_value = 3  # Only 3 channels (0, 1, 2)
        mock_image.getSizeT.return_value = 1
        mock_conn.getObject.return_value = mock_image

        with pytest.raises(IndexError, match="Channel index 4 is out of range"):
            aggregate_imgs(mock_conn, [100], channel=4)

    def test_channel_error_mentions_metadata(self):
        """Test that the error message points user to check metadata."""
        from omero_screen.flatfield_corr import aggregate_imgs

        mock_conn = MagicMock()
        mock_image = MagicMock()
        mock_image.getSizeX.return_value = 256
        mock_image.getSizeY.return_value = 256
        mock_image.getSizeZ.return_value = 1
        mock_image.getSizeC.return_value = 2
        mock_image.getSizeT.return_value = 1
        mock_conn.getObject.return_value = mock_image

        with pytest.raises(IndexError, match="channel metadata"):
            aggregate_imgs(mock_conn, [100], channel=3)

    def test_valid_channel_index_proceeds(self):
        """Test that a valid channel index does not raise bounds error."""
        from omero_screen.flatfield_corr import aggregate_imgs

        mock_conn = MagicMock()
        mock_image = MagicMock()
        mock_image.getSizeX.return_value = 64
        mock_image.getSizeY.return_value = 64
        mock_image.getSizeZ.return_value = 1
        mock_image.getSizeC.return_value = 3
        mock_image.getSizeT.return_value = 1
        mock_conn.getObject.return_value = mock_image

        # This will fail later (get_image is real), but should NOT raise IndexError
        # about channel bounds
        with patch("omero_screen.flatfield_corr.get_image") as mock_get:
            mock_get.return_value = (
                None,
                np.random.default_rng(42).uniform(100, 1000, (1, 1, 64, 64, 1)),
            )
            # channel 2 is valid for 3 channels
            result = aggregate_imgs(mock_conn, [100], channel=2)
            assert result is not None


# -------------------- Image Analysis Channel Errors --------------------


class TestImageAnalysisChannelErrors:
    """Tests for channel access error messages in image_analysis."""

    @pytest.fixture
    def mock_image_setup(self):
        """Create common mocks for Image class tests."""
        mock_conn = MagicMock()
        mock_well = MagicMock()
        mock_well.getWellPos.return_value = "A1"
        mock_well.getId.return_value = 1

        mock_image_obj = MagicMock()
        mock_image_obj.getId.return_value = 100
        mock_image_obj.getSizeZ.return_value = 1

        mock_metadata = MagicMock()
        mock_metadata.plate_id = 1
        mock_metadata.well_conditions.return_value = {
            "cell_line": "RPE-1",
            "condition": "ctrl",
        }

        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = []
        mock_conn.getObject.return_value = mock_dataset

        return mock_conn, mock_well, mock_image_obj, mock_metadata

    @patch("omero_screen.image_analysis.upload_masks")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.get_image")
    def test_channel_index_out_of_range(
        self, mock_get_image, mock_seg_model, mock_upload, mock_image_setup
    ):
        """Test that out-of-range channel index in _get_img_dict gives clear error."""
        mock_conn, mock_well, mock_image_obj, mock_metadata = mock_image_setup
        # Image has only 2 channels but metadata references index 3
        mock_metadata.channel_data = {"DAPI": "0", "EdU": "3"}
        synthetic_data = np.random.default_rng(42).uniform(
            100, 1000, (1, 1, 64, 64, 2)
        )
        mock_get_image.return_value = (None, synthetic_data)

        from omero_screen.image_analysis import Image

        with pytest.raises(IndexError, match="only has 2 channels"):
            Image(
                mock_conn, mock_well, mock_image_obj, mock_metadata,
                dataset_id=1,
                flatfield_dict={"DAPI": np.ones((64, 64)), "EdU": np.ones((64, 64))},
            )

    @patch("omero_screen.image_analysis.upload_masks")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.get_image")
    def test_flatfield_channel_mismatch(
        self, mock_get_image, mock_seg_model, mock_upload, mock_image_setup
    ):
        """Test that mismatched flatfield keys give clear error."""
        mock_conn, mock_well, mock_image_obj, mock_metadata = mock_image_setup
        mock_metadata.channel_data = {"DAPI": "0", "EdU": "1"}
        synthetic_data = np.random.default_rng(42).uniform(
            100, 1000, (1, 1, 64, 64, 2)
        )
        mock_get_image.return_value = (None, synthetic_data)

        from omero_screen.image_analysis import Image

        # Flatfield dict is missing 'EdU'
        with pytest.raises(KeyError, match="'EdU' not found in flatfield"):
            Image(
                mock_conn, mock_well, mock_image_obj, mock_metadata,
                dataset_id=1,
                flatfield_dict={"DAPI": np.ones((64, 64))},
            )


# -------------------- Cell Cycle Analysis Column Validation --------------------


class TestCellCycleColumnValidation:
    """Tests for column validation in cellcycle_analysis."""

    @pytest.fixture
    def minimal_df(self):
        """Create a minimal DataFrame for cell cycle analysis."""
        rng = np.random.default_rng(42)
        n = 50
        return pd.DataFrame({
            "integrated_int_DAPI": rng.uniform(1000, 5000, n),
            "intensity_mean_EdU_nucleus": rng.uniform(100, 2000, n),
            "intensity_min_EdU_nucleus": rng.uniform(10, 50, n),
            "cell_line": ["RPE-1"] * n,
            "image_id": [1] * n,
            "Cyto_ID": range(n),
            "label": range(n),
            "area_nucleus": rng.uniform(100, 500, n),
            "well": ["A1"] * n,
            "plate_id": [1] * n,
            "experiment": ["test"] * n,
        })

    def test_missing_edu_column(self):
        """Test that missing EdU column produces a clear error."""
        df = pd.DataFrame({
            "integrated_int_DAPI": [1000, 2000],
            "cell_line": ["RPE-1", "RPE-1"],
        })
        with pytest.raises(KeyError, match="intensity_mean_EdU_nucleus"):
            cellcycle_analysis(df)

    def test_missing_dapi_column(self):
        """Test that missing DAPI column produces a clear error."""
        df = pd.DataFrame({
            "intensity_mean_EdU_nucleus": [100, 200],
            "intensity_min_EdU_nucleus": [10, 20],
            "cell_line": ["RPE-1", "RPE-1"],
        })
        with pytest.raises(KeyError, match="integrated_int_DAPI"):
            cellcycle_analysis(df)

    def test_missing_h3p_when_h3_true(self):
        """Test that missing H3P columns are caught when H3=True."""
        df = pd.DataFrame({
            "integrated_int_DAPI": [1000, 2000],
            "intensity_mean_EdU_nucleus": [100, 200],
            "intensity_min_EdU_nucleus": [10, 20],
            "cell_line": ["RPE-1", "RPE-1"],
        })
        with pytest.raises(KeyError, match="intensity_mean_H3P_nucleus"):
            cellcycle_analysis(df, H3=True)

    def test_missing_cyto_id_when_cyto_true(self):
        """Test that missing Cyto_ID column is caught when cyto=True."""
        df = pd.DataFrame({
            "integrated_int_DAPI": [1000, 2000],
            "intensity_mean_EdU_nucleus": [100, 200],
            "intensity_min_EdU_nucleus": [10, 20],
            "cell_line": ["RPE-1", "RPE-1"],
        })
        with pytest.raises(KeyError, match="Cyto_ID"):
            cellcycle_analysis(df, cyto=True)

    def test_error_message_mentions_channel(self):
        """Test that error messages guide user to check channels."""
        df = pd.DataFrame({
            "cell_line": ["RPE-1"],
        })
        with pytest.raises(KeyError, match="channel"):
            cellcycle_analysis(df)

    def test_valid_dataframe_no_error(self, minimal_df):
        """Test that a valid DataFrame passes validation."""
        # Should not raise
        result = cellcycle_analysis(minimal_df)
        assert "cell_cycle" in result.columns


# -------------------- Loops Well Annotation Errors --------------------


class TestLoopsWellAnnotationErrors:
    """Tests for well annotation error handling in loops.py."""

    @patch("omero_screen.loops.parse_annotations")
    def test_missing_cell_line_annotation_treated_as_empty(self, mock_parse):
        """Wells missing the cell_line annotation are skipped as Empty.

        Previously this raised ``WellAnnotationError``; the loop now
        treats absent ``cell_line`` the same as ``cell_line == "Empty"``
        because the metadata parser deliberately writes no annotations
        to Empty wells.
        """
        from omero_screen.loops import process_wells

        mock_parse.return_value = {"condition": "ctrl"}  # No cell_line

        mock_conn = MagicMock()
        mock_well = MagicMock()
        mock_well.getWellPos.return_value = "A1"
        mock_plate = MagicMock()
        mock_plate.listChildren.return_value = [mock_well]
        mock_conn.getObject.return_value = mock_plate

        mock_metadata = MagicMock()
        mock_metadata.plate_id = 1

        df, qc, gallery = process_wells(mock_conn, mock_metadata, 1, {})
        assert df.empty

    @patch("omero_screen.loops.parse_annotations")
    def test_case_insensitive_cell_line(self, mock_parse):
        """Test that Cell_Line (capital L) is accepted via case-insensitive lookup."""
        from omero_screen.loops import process_wells

        mock_parse.return_value = {"Cell_Line": "Empty", "condition": "ctrl"}

        mock_conn = MagicMock()
        mock_well = MagicMock()
        mock_well.getWellPos.return_value = "A1"
        mock_plate = MagicMock()
        mock_plate.listChildren.return_value = [mock_well]
        mock_conn.getObject.return_value = mock_plate

        mock_metadata = MagicMock()
        mock_metadata.plate_id = 1

        # Should not raise — "Empty" wells are skipped
        df, qc, gallery = process_wells(mock_conn, mock_metadata, 1, {})
        assert df.empty

    @patch("omero_screen.loops.parse_annotations")
    def test_wells_with_other_annotations_but_no_cell_line_are_skipped(
        self, mock_parse
    ):
        """Wells that have unrelated annotations but no cell_line are skipped.

        Mirrors the "Empty well" path: annotations exist but the
        ``cell_line`` key is missing, so the well contributes nothing.
        """
        from omero_screen.loops import process_wells

        mock_parse.return_value = {"condition": "ctrl", "timepoint": "24h"}

        mock_conn = MagicMock()
        mock_well = MagicMock()
        mock_well.getWellPos.return_value = "B3"
        mock_plate = MagicMock()
        mock_plate.listChildren.return_value = [mock_well]
        mock_conn.getObject.return_value = mock_plate

        mock_metadata = MagicMock()
        mock_metadata.plate_id = 1

        df, qc, gallery = process_wells(mock_conn, mock_metadata, 1, {})
        assert df.empty
