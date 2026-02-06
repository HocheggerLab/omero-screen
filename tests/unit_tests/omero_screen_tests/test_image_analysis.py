"""Unit tests for image_analysis module with mocked Cellpose and synthetic data."""

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch

from omero_screen.image_analysis import (
    Image,
    get_cell_model,
)


class TestGetCellModel:
    """Test the get_cell_model function for model selection."""

    def test_get_cell_model_40x(self):
        """Test model selection for 40X magnification."""
        result = get_cell_model("RPE-1 40X")
        assert result == "40x_Tub_H2B"

    def test_get_cell_model_40x_lowercase(self):
        """Test model selection with lowercase 40x."""
        result = get_cell_model("u2os 40x")
        assert result == "40x_Tub_H2B"

    def test_get_cell_model_20x(self):
        """Test model selection for 20X magnification."""
        result = get_cell_model("HeLa 20X")
        assert result == "cyto"

    def test_get_cell_model_20x_lowercase(self):
        """Test model selection with lowercase 20x."""
        result = get_cell_model("rpe-1 20x")
        assert result == "cyto"

    def test_get_cell_model_rpe(self):
        """Test model selection for RPE cell line."""
        result = get_cell_model("RPE-1")
        assert result == "RPE-1_Tub_Hoechst"

    def test_get_cell_model_rpe_substring(self):
        """Test model selection with RPE substring matching."""
        result = get_cell_model("RPE-1 treated")
        assert result == "RPE-1_Tub_Hoechst"

    def test_get_cell_model_hela(self):
        """Test model selection for HeLa cell line."""
        result = get_cell_model("HELA")
        assert result == "HeLa_Tub_Hoechst"

    def test_get_cell_model_u2os(self):
        """Test model selection for U2OS cell line."""
        result = get_cell_model("U2OS")
        assert result == "U2OS_Tub_Hoechst"

    def test_get_cell_model_unknown_uses_default(self):
        """Test that unknown cell line uses default model."""
        result = get_cell_model("UnknownCellLine")
        assert result == "U2OS_Tub_Hoechst"  # default

    def test_get_cell_model_custom_default(self):
        """Test custom default model."""
        result = get_cell_model(
            "UnknownCellLine", default_model="custom_model"
        )
        assert result == "custom_model"

    def test_get_cell_model_removes_spaces(self):
        """Test that spaces are removed from cell line name."""
        result = get_cell_model("R P E - 1")
        assert result == "RPE-1_Tub_Hoechst"

    def test_get_cell_model_case_insensitive(self):
        """Test that cell line matching is case insensitive."""
        result = get_cell_model("hela")
        assert result == "HeLa_Tub_Hoechst"


class TestImageInitialization:
    """Test the Image class initialization with mocked dependencies."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock OMERO connection."""
        conn = MagicMock()
        return conn

    @pytest.fixture
    def mock_well(self):
        """Create a mock OMERO well."""
        well = MagicMock()
        well.getWellPos.return_value = "A1"
        well.getId.return_value = 123
        return well

    @pytest.fixture
    def mock_image_obj(self):
        """Create a mock OMERO image object."""
        image = MagicMock()
        image.getId.return_value = 456
        image.getSizeZ.return_value = 1  # Single z-plane
        image.getSizeX.return_value = 512
        image.getSizeY.return_value = 512
        image.getSizeC.return_value = 2
        image.getSizeT.return_value = 1
        return image

    @pytest.fixture
    def mock_metadata(self):
        """Create a mock metadata parser."""
        metadata = MagicMock()
        metadata.channel_data = {"DAPI": "0", "Tub": "1"}
        metadata.plate_id = 789
        metadata.well_conditions.return_value = {
            "cell_line": "RPE-1",
            "condition": "control",
        }
        return metadata

    @pytest.fixture
    def flatfield_dict(self):
        """Create a flatfield correction dictionary."""
        return {
            "DAPI": np.ones((512, 512), dtype=np.float64),
            "Tub": np.ones((512, 512), dtype=np.float64),
        }

    @pytest.fixture
    def synthetic_image_data(self):
        """Create synthetic image data (TZCYX format)."""
        rng = np.random.default_rng(seed=42)
        # Create TZCYX array: 1 timepoint, 1 z-plane, 512x512, 2 channels
        data = rng.uniform(100, 1000, (1, 1, 512, 512, 2)).astype(np.float32)
        return data

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_image_initialization_basic(
        self,
        mock_upload,
        mock_segmentation,
        mock_get_image,
        mock_conn,
        mock_well,
        mock_image_obj,
        mock_metadata,
        flatfield_dict,
        synthetic_image_data,
    ):
        """Test basic Image initialization without segmentation."""
        # Setup mock returns
        mock_get_image.return_value = (None, synthetic_image_data)

        # Mock dataset to prevent segmentation from running
        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = []
        mock_conn.getObject.return_value = mock_dataset

        # Mock Cellpose model
        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = (
            np.zeros((512, 512), dtype=np.uint32)
        )
        mock_segmentation.return_value = mock_model_instance

        # Create Image instance
        img = Image(
            conn=mock_conn,
            well=mock_well,
            image_obj=mock_image_obj,
            metadata=mock_metadata,
            dataset_id=999,
            flatfield_dict=flatfield_dict,
        )

        # Verify initialization
        assert img.channels == {"DAPI": "0", "Tub": "1"}
        assert img.well_pos == "A1"
        assert img.cell_line == "RPE-1"
        assert img.nuc_diameter == 10  # default for non-20x/40x
        assert "DAPI" in img.img_dict
        assert "Tub" in img.img_dict

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_image_initialization_with_40x(
        self,
        mock_upload,
        mock_segmentation,
        mock_get_image,
        mock_conn,
        mock_well,
        mock_image_obj,
        mock_metadata,
        flatfield_dict,
        synthetic_image_data,
    ):
        """Test Image initialization with 40X magnification."""
        # Modify metadata for 40X
        mock_metadata.well_conditions.return_value = {
            "cell_line": "RPE-1 40X",
            "condition": "control",
        }

        mock_get_image.return_value = (None, synthetic_image_data)
        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = []
        mock_conn.getObject.return_value = mock_dataset

        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = (
            np.zeros((512, 512), dtype=np.uint32)
        )
        mock_segmentation.return_value = mock_model_instance

        img = Image(
            conn=mock_conn,
            well=mock_well,
            image_obj=mock_image_obj,
            metadata=mock_metadata,
            dataset_id=999,
            flatfield_dict=flatfield_dict,
        )

        # Verify 40X diameter
        assert img.nuc_diameter == 100

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_image_initialization_with_20x(
        self,
        mock_upload,
        mock_segmentation,
        mock_get_image,
        mock_conn,
        mock_well,
        mock_image_obj,
        mock_metadata,
        flatfield_dict,
        synthetic_image_data,
    ):
        """Test Image initialization with 20X magnification."""
        # Modify metadata for 20X
        mock_metadata.well_conditions.return_value = {
            "cell_line": "U2OS 20X",
            "condition": "control",
        }

        mock_get_image.return_value = (None, synthetic_image_data)
        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = []
        mock_conn.getObject.return_value = mock_dataset

        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = (
            np.zeros((512, 512), dtype=np.uint32)
        )
        mock_segmentation.return_value = mock_model_instance

        img = Image(
            conn=mock_conn,
            well=mock_well,
            image_obj=mock_image_obj,
            metadata=mock_metadata,
            dataset_id=999,
            flatfield_dict=flatfield_dict,
        )

        # Verify 20X diameter
        assert img.nuc_diameter == 25

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_image_flatfield_correction(
        self,
        mock_upload,
        mock_segmentation,
        mock_get_image,
        mock_conn,
        mock_well,
        mock_image_obj,
        mock_metadata,
        synthetic_image_data,
    ):
        """Test that flatfield correction is applied correctly."""
        # Create non-uniform flatfield
        flatfield_dict_gradient = {
            "DAPI": np.linspace(0.5, 1.5, 512 * 512).reshape(512, 512),
            "Tub": np.ones((512, 512), dtype=np.float64),
        }

        mock_get_image.return_value = (None, synthetic_image_data)
        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = []
        mock_conn.getObject.return_value = mock_dataset

        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = (
            np.zeros((512, 512), dtype=np.uint32)
        )
        mock_segmentation.return_value = mock_model_instance

        img = Image(
            conn=mock_conn,
            well=mock_well,
            image_obj=mock_image_obj,
            metadata=mock_metadata,
            dataset_id=999,
            flatfield_dict=flatfield_dict_gradient,
        )

        # Verify flatfield correction was applied
        assert img.img_dict["DAPI"].shape == (1, 512, 512)  # TYX format
        # The corrected image should be different from original
        # due to division by gradient flatfield


class TestImageSegmentation:
    """Test Image segmentation methods with mocked Cellpose."""

    @pytest.fixture
    def mock_setup(self):
        """Create comprehensive mock setup for segmentation tests."""
        # Mock connection
        conn = MagicMock()

        # Mock well
        well = MagicMock()
        well.getWellPos.return_value = "B2"
        well.getId.return_value = 111

        # Mock image
        image_obj = MagicMock()
        image_obj.getId.return_value = 222
        image_obj.getSizeZ.return_value = 1

        # Mock metadata
        metadata = MagicMock()
        metadata.channel_data = {"DAPI": "0"}
        metadata.well_conditions.return_value = {
            "cell_line": "HeLa",
            "condition": "test",
        }

        # Flatfield dict
        flatfield_dict = {"DAPI": np.ones((256, 256))}

        # Synthetic data
        synthetic_data = (
            np.random.default_rng(42)
            .uniform(100, 1000, (1, 1, 256, 256, 1))
            .astype(np.float32)
        )

        # Mock dataset (no existing masks)
        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = []
        conn.getObject.return_value = mock_dataset

        return {
            "conn": conn,
            "well": well,
            "image_obj": image_obj,
            "metadata": metadata,
            "flatfield_dict": flatfield_dict,
            "synthetic_data": synthetic_data,
        }

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_nucleus_segmentation_called(
        self, mock_upload, mock_segmentation, mock_get_image, mock_setup
    ):
        """Test that Cellpose is called for nucleus segmentation."""
        mock_get_image.return_value = (None, mock_setup["synthetic_data"])

        # Create synthetic segmentation mask
        synthetic_mask = np.zeros((256, 256), dtype=np.uint32)
        # Add some "cells"
        rng = np.random.default_rng(42)
        for i in range(10):
            x, y = rng.integers(20, 236, 2)
            synthetic_mask[y : y + 20, x : x + 20] = i + 1

        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = (
            synthetic_mask
        )
        mock_segmentation.return_value = mock_model_instance

        img = Image(
            conn=mock_setup["conn"],
            well=mock_setup["well"],
            image_obj=mock_setup["image_obj"],
            metadata=mock_setup["metadata"],
            dataset_id=333,
            flatfield_dict=mock_setup["flatfield_dict"],
        )

        # Verify Cellpose was called
        assert mock_segmentation.called
        assert mock_model_instance.eval.called

        # Verify mask was created
        assert img.n_mask is not None
        assert img.n_mask.shape == (1, 256, 256)  # TYX format

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_cell_segmentation_with_tubulin(
        self, mock_upload, mock_segmentation, mock_get_image, mock_setup
    ):
        """Test that cell segmentation is called when Tub channel exists."""
        # Modify metadata to include Tub channel
        mock_setup["metadata"].channel_data = {"DAPI": "0", "Tub": "1"}
        mock_setup["flatfield_dict"]["Tub"] = np.ones((256, 256))

        # Modify synthetic data to have 2 channels
        synthetic_data_2ch = (
            np.random.default_rng(42)
            .uniform(100, 1000, (1, 1, 256, 256, 2))
            .astype(np.float32)
        )
        mock_get_image.return_value = (None, synthetic_data_2ch)

        # Create synthetic masks
        nuc_mask = np.zeros((256, 256), dtype=np.uint32)
        cell_mask = np.zeros((256, 256), dtype=np.uint32)
        rng = np.random.default_rng(42)
        for i in range(5):
            x, y = rng.integers(30, 226, 2)
            # Nucleus (smaller)
            nuc_mask[y : y + 10, x : x + 10] = i + 1
            # Cell (larger)
            cell_mask[y - 5 : y + 15, x - 5 : x + 15] = i + 1

        mock_model_instance = MagicMock()
        # First call returns nucleus mask, second call returns cell mask
        mock_model_instance.eval.side_effect = [
            nuc_mask,
            cell_mask,
        ]
        mock_segmentation.return_value = mock_model_instance

        img = Image(
            conn=mock_setup["conn"],
            well=mock_setup["well"],
            image_obj=mock_setup["image_obj"],
            metadata=mock_setup["metadata"],
            dataset_id=333,
            flatfield_dict=mock_setup["flatfield_dict"],
        )

        # Verify both segmentations were performed
        assert mock_model_instance.eval.call_count == 2

        # Verify all masks exist
        assert img.n_mask is not None
        assert img.c_mask is not None
        assert img.cyto_mask is not None

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_segmentation_with_existing_masks(
        self, mock_upload, mock_segmentation, mock_get_image, mock_setup
    ):
        """Test that existing masks are loaded instead of re-segmenting."""
        # Create existing mask data
        existing_mask_data = np.zeros((1, 1, 256, 256, 1), dtype=np.uint32)
        existing_mask_data[0, 0, 50:150, 50:150, 0] = 1  # One cell

        # Mock existing mask image
        mock_mask_image = MagicMock()
        mock_mask_image.getName.return_value = "222_segmentation"
        mock_mask_image.getId.return_value = 999

        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = [mock_mask_image]
        mock_setup["conn"].getObject.return_value = mock_dataset

        # Setup get_image to return different data for original vs mask
        def get_image_side_effect(conn, image_id, **kwargs):
            if image_id == 222:  # Original image
                return (None, mock_setup["synthetic_data"])
            elif image_id == 999:  # Mask image
                return (None, existing_mask_data)

        mock_get_image.side_effect = get_image_side_effect

        img = Image(
            conn=mock_setup["conn"],
            well=mock_setup["well"],
            image_obj=mock_setup["image_obj"],
            metadata=mock_setup["metadata"],
            dataset_id=333,
            flatfield_dict=mock_setup["flatfield_dict"],
        )

        # Verify Cellpose was NOT called (masks already existed)
        assert not mock_segmentation.called

        # Verify masks were loaded
        assert img.n_mask is not None
        assert np.max(img.n_mask) == 1  # The cell we created

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_segmentation_uploads_masks(
        self, mock_upload, mock_segmentation, mock_get_image, mock_setup
    ):
        """Test that newly created masks are uploaded to OMERO."""
        mock_get_image.return_value = (None, mock_setup["synthetic_data"])

        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = (
            np.zeros((256, 256), dtype=np.uint32)
        )
        mock_segmentation.return_value = mock_model_instance

        img = Image(
            conn=mock_setup["conn"],
            well=mock_setup["well"],
            image_obj=mock_setup["image_obj"],
            metadata=mock_setup["metadata"],
            dataset_id=333,
            flatfield_dict=mock_setup["flatfield_dict"],
        )

        # Verify upload_masks was called
        assert mock_upload.called
        # Check that it was called with the correct arguments
        call_args = mock_upload.call_args
        assert call_args[0][0] == mock_setup["conn"]
        assert call_args[0][1] == 333  # dataset_id


class TestImageUtilityMethods:
    """Test utility methods of the Image class."""

    def test_get_cyto_basic(self):
        """Test cytoplasm mask generation."""
        # Create mock Image instance
        img = MagicMock()

        # Create synthetic masks
        n_mask = np.zeros((100, 100), dtype=np.uint32)
        c_mask = np.zeros((100, 100), dtype=np.uint32)

        # Add a cell: nucleus is 10x10, cell is 20x20
        n_mask[45:55, 45:55] = 1
        c_mask[40:60, 40:60] = 1

        # Call the actual _get_cyto method
        from omero_screen.image_analysis import Image

        cyto_mask = Image._get_cyto(img, n_mask, c_mask)

        # Verify cytoplasm is cell minus nucleus
        assert cyto_mask is not None
        # The nucleus region should be 0 in cytoplasm
        assert np.all(cyto_mask[45:55, 45:55] == 0)
        # The cell border should be non-zero
        assert np.any(cyto_mask[40:60, 40:60] != 0)

    def test_compact_mask_uint8(self):
        """Test mask compaction to uint8."""
        img = MagicMock()

        # Create mask with small values (fits in uint8)
        mask = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

        from omero_screen.image_analysis import Image

        compacted = Image._compact_mask(img, mask)

        assert compacted.dtype == np.uint8
        np.testing.assert_array_equal(compacted, mask)

    def test_compact_mask_uint16(self):
        """Test mask compaction to uint16."""
        img = MagicMock()

        # Create mask with values that need uint16
        mask = np.zeros((100, 100), dtype=np.uint32)
        mask[0, 0] = 300  # Requires uint16

        from omero_screen.image_analysis import Image

        compacted = Image._compact_mask(img, mask)

        assert compacted.dtype == np.uint16

    def test_compact_mask_stays_uint32(self):
        """Test mask stays uint32 when needed."""
        img = MagicMock()

        # Create mask with values that need uint32
        mask = np.zeros((100, 100), dtype=np.uint32)
        mask[0, 0] = 70000  # Requires uint32

        from omero_screen.image_analysis import Image

        compacted = Image._compact_mask(img, mask)

        assert compacted.dtype == np.uint32


class TestMultiTimepoint:
    """Test Image class with multi-timepoint data."""

    @patch("omero_screen.image_analysis.get_image")
    @patch("omero_screen.image_analysis._get_segmentation_model")
    @patch("omero_screen.image_analysis.upload_masks")
    def test_multi_timepoint_segmentation(
        self, mock_upload, mock_segmentation, mock_get_image
    ):
        """Test segmentation with multiple timepoints."""
        # Setup mocks
        conn = MagicMock()
        well = MagicMock()
        well.getWellPos.return_value = "C3"

        image_obj = MagicMock()
        image_obj.getId.return_value = 555
        image_obj.getSizeZ.return_value = 1

        metadata = MagicMock()
        metadata.channel_data = {"DAPI": "0"}
        metadata.well_conditions.return_value = {
            "cell_line": "U2OS",
            "condition": "timelapse",
        }

        flatfield_dict = {"DAPI": np.ones((128, 128))}

        # Create multi-timepoint data: 3 timepoints
        synthetic_data = (
            np.random.default_rng(42)
            .uniform(100, 1000, (3, 1, 128, 128, 1))
            .astype(np.float32)
        )

        mock_get_image.return_value = (None, synthetic_data)

        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = []
        conn.getObject.return_value = mock_dataset

        # Mock Cellpose to return different masks for each timepoint
        mock_model_instance = MagicMock()

        def segmentation_eval_side_effect(img, **kwargs):
            # Return different masks based on call count
            t = mock_model_instance.eval.call_count - 1
            mask = np.zeros((128, 128), dtype=np.uint32)
            # Add cells that change position over time
            mask[20 + t * 10 : 40 + t * 10, 20:40] = 1
            return mask

        mock_model_instance.eval.side_effect = segmentation_eval_side_effect
        mock_segmentation.return_value = mock_model_instance

        img = Image(
            conn=conn,
            well=well,
            image_obj=image_obj,
            metadata=metadata,
            dataset_id=444,
            flatfield_dict=flatfield_dict,
        )

        # Verify segmentation was called for each timepoint
        assert mock_model_instance.eval.call_count == 3

        # Verify mask has correct shape (T, Y, X)
        assert img.n_mask.shape == (3, 128, 128)
