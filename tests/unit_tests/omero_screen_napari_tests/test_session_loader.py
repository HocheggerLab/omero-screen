"""Unit tests for session loading functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.session_data_loader import SessionDataLoader
from omero_screen_napari.session_utils import apply_masks_to_crops, parse_npy_file


class TestSessionUtils:
    """Test session utility functions."""

    def test_apply_masks_to_crops_valid(self):
        """Test applying masks to crops with valid data."""
        omero_data = OmeroData()

        # Create test data: 2 images (3x3x3) and 2 masks (3x3)
        crops = [
            np.ones((3, 3, 3), dtype=np.uint8) * 100,
            np.ones((3, 3, 3), dtype=np.uint8) * 200,
        ]
        labels = [
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
            np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8),
        ]

        omero_data.selected_crops = crops
        omero_data.selected_labels = labels

        masked = apply_masks_to_crops(omero_data)

        assert len(masked) == 2
        assert masked[0].shape == (3, 3, 3)

        # Check that masked regions are zeroed
        assert masked[0][2, 2, 0] == 0  # Outside mask
        assert masked[0][0, 0, 0] == 100  # Inside mask

    def test_apply_masks_to_crops_empty(self):
        """Test applying masks with no data returns empty list."""
        omero_data = OmeroData()
        omero_data.selected_crops = None
        omero_data.selected_labels = None

        masked = apply_masks_to_crops(omero_data)

        assert masked == []

    def test_parse_npy_file_valid(self, tmp_path):
        """Test parsing valid NPY file."""
        # Create test NPY file
        npy_path = tmp_path / "test_session.npy"

        crops = [np.ones((10, 10, 3), dtype=np.uint8)]
        labels = [np.ones((10, 10), dtype=np.uint8)]
        classes = [0]

        data = {"data": (crops, labels), "target": classes}
        np.save(npy_path, data)

        # Create OmeroData and UserData
        omero_data = OmeroData()
        user_data = UserData()
        user_data.channels = [0, 1, 2]
        user_data.contour = False

        # Parse file
        parse_npy_file(npy_path, omero_data, user_data)

        # Verify data was loaded
        assert omero_data.selected_crops is not None
        assert len(omero_data.selected_crops) == 1
        assert omero_data.selected_classes is not None
        assert len(omero_data.selected_classes) == 1
        assert omero_data.selected_images is not None
        assert len(omero_data.selected_images) == 1

    def test_parse_npy_file_channels_as_strings(self, tmp_path):
        """Test parsing NPY file with channels stored as strings."""
        # Create test NPY file
        npy_path = tmp_path / "test_session.npy"

        crops = [np.ones((10, 10, 3), dtype=np.uint8)]
        labels = [np.ones((10, 10), dtype=np.uint8)]
        classes = [0]

        data = {"data": (crops, labels), "target": classes}
        np.save(npy_path, data)

        # Create OmeroData and UserData with string channels
        omero_data = OmeroData()
        user_data = UserData()
        user_data.channels = ["0", "1", "2"]  # Strings, not ints
        user_data.contour = False

        # Parse file - should handle string channels
        parse_npy_file(npy_path, omero_data, user_data)

        # Verify data was loaded
        assert omero_data.selected_crops is not None
        assert len(omero_data.selected_crops) == 1
        assert omero_data.selected_images is not None
        assert len(omero_data.selected_images) == 1

    def test_parse_npy_file_channel_names(self, tmp_path):
        """Test parsing NPY file with channel names instead of indices."""
        # Create test NPY file
        npy_path = tmp_path / "test_session.npy"

        crops = [np.ones((10, 10, 3), dtype=np.uint8)]
        labels = [np.ones((10, 10), dtype=np.uint8)]
        classes = [0]

        data = {"data": (crops, labels), "target": classes}
        np.save(npy_path, data)

        # Create OmeroData and UserData with channel names
        omero_data = OmeroData()
        user_data = UserData()
        user_data.channels = ["DAPI", "EdU"]  # Channel names, not indices
        user_data.contour = False

        # Parse file - should fall back to using all channels
        parse_npy_file(npy_path, omero_data, user_data)

        # Verify data was loaded and fallback to all channels worked
        assert omero_data.selected_crops is not None
        assert len(omero_data.selected_crops) == 1
        assert omero_data.selected_images is not None
        assert len(omero_data.selected_images) == 1
        # Should use all 3 channels since names can't be converted
        assert omero_data.selected_images[0].shape == (10, 10, 3)

    def test_parse_npy_file_missing_file(self, tmp_path):
        """Test parsing non-existent NPY file raises FileNotFoundError."""
        npy_path = tmp_path / "nonexistent.npy"

        omero_data = OmeroData()
        user_data = UserData()

        with pytest.raises(FileNotFoundError):
            parse_npy_file(npy_path, omero_data, user_data)

    def test_parse_npy_file_invalid_structure(self, tmp_path):
        """Test parsing NPY with invalid structure raises ValueError."""
        npy_path = tmp_path / "invalid.npy"

        # Save NPY with wrong structure
        data = {"wrong_key": [1, 2, 3]}
        np.save(npy_path, data)

        omero_data = OmeroData()
        user_data = UserData()

        with pytest.raises(ValueError, match="Invalid NPY structure"):
            parse_npy_file(npy_path, omero_data, user_data)

    def test_parse_npy_file_without_user_data(self, tmp_path):
        """Test parsing NPY file without user_data only loads raw data."""
        npy_path = tmp_path / "test_session.npy"

        crops = [np.ones((10, 10, 3), dtype=np.uint8)]
        labels = [np.ones((10, 10), dtype=np.uint8)]
        classes = [0]

        data = {"data": (crops, labels), "target": classes}
        np.save(npy_path, data)

        omero_data = OmeroData()

        # Parse without user_data
        parse_npy_file(npy_path, omero_data, user_data=None)

        # Verify only raw data was loaded, not processed images
        assert omero_data.selected_crops is not None
        # When user_data is None, selected_images won't be populated
        # The function only loads crops, labels, and classes


class TestSessionDataLoader:
    """Test SessionDataLoader class."""

    def test_load_session_success(self, tmp_path):
        """Test successful session loading."""
        # Create test NPY file
        npy_path = tmp_path / "test_session.npy"

        crops = [np.ones((10, 10, 3), dtype=np.uint8)]
        labels = [np.ones((10, 10), dtype=np.uint8)]
        classes = [0]

        data = {"data": (crops, labels), "target": classes}
        np.save(npy_path, data)

        # Mock database
        mock_db = MagicMock()
        session_data = {
            "id": 123,
            "plate_id": 456,
            "well_id": "A1",
            "image_id": 789,
            "file_path": str(npy_path),
            "metadata": {
                "user_data": {
                    "channels": [0, 1, 2],
                    "contour": False,
                }
            },
        }
        mock_db.get_session_by_id.return_value = session_data

        # Create OmeroData and UserData
        omero_data = OmeroData()
        user_data = UserData()

        # Load session
        success, message = SessionDataLoader.load_session(
            session_id=123,
            db=mock_db,
            omero_data=omero_data,
            user_data=user_data,
        )

        assert success is True
        assert "cells from session" in message
        assert omero_data.plate_id == 456
        assert omero_data.selected_crops is not None
        assert len(omero_data.selected_crops) == 1

    def test_load_session_not_found(self):
        """Test loading non-existent session."""
        mock_db = MagicMock()
        mock_db.get_session_by_id.return_value = None

        omero_data = OmeroData()
        user_data = UserData()

        success, message = SessionDataLoader.load_session(
            session_id=999,
            db=mock_db,
            omero_data=omero_data,
            user_data=user_data,
        )

        assert success is False
        assert "not found in database" in message

    def test_load_session_file_not_found(self):
        """Test loading session with missing NPY file."""
        mock_db = MagicMock()
        session_data = {
            "id": 123,
            "plate_id": 456,
            "file_path": "/nonexistent/path/session.npy",
            "metadata": None,
        }
        mock_db.get_session_by_id.return_value = session_data

        omero_data = OmeroData()
        user_data = UserData()

        success, message = SessionDataLoader.load_session(
            session_id=123,
            db=mock_db,
            omero_data=omero_data,
            user_data=user_data,
        )

        assert success is False
        assert "NPY file not found" in message

    def test_load_session_restores_user_data(self, tmp_path):
        """Test that session loading restores UserData from metadata."""
        npy_path = tmp_path / "test_session.npy"

        crops = [np.ones((10, 10, 3), dtype=np.uint8)]
        labels = [np.ones((10, 10), dtype=np.uint8)]
        classes = [0]

        data = {"data": (crops, labels), "target": classes}
        np.save(npy_path, data)

        mock_db = MagicMock()
        session_data = {
            "id": 123,
            "plate_id": 456,
            "file_path": str(npy_path),
            "metadata": {
                "user_data": {
                    "channels": [1, 2, 3],
                    "contour": True,
                    "crop_size": 50,
                }
            },
        }
        mock_db.get_session_by_id.return_value = session_data

        omero_data = OmeroData()
        user_data = UserData()

        success, message = SessionDataLoader.load_session(
            session_id=123,
            db=mock_db,
            omero_data=omero_data,
            user_data=user_data,
        )

        assert success is True
        assert user_data.channels == [1, 2, 3]
        assert user_data.contour is True
        assert user_data.crop_size == 50

    def test_validate_session_file_valid(self, tmp_path):
        """Test validation of valid session file."""
        npy_path = tmp_path / "test.npy"
        npy_path.write_bytes(b"dummy")

        session = {"file_path": str(npy_path)}

        valid, message = SessionDataLoader.validate_session_file(session)

        assert valid is True
        assert message == "File valid"

    def test_validate_session_file_missing(self, tmp_path):
        """Test validation of missing file."""
        npy_path = tmp_path / "missing.npy"

        session = {"file_path": str(npy_path)}

        valid, message = SessionDataLoader.validate_session_file(session)

        assert valid is False
        assert "File not found" in message

    def test_validate_session_file_wrong_extension(self, tmp_path):
        """Test validation rejects non-NPY files."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("dummy")

        session = {"file_path": str(txt_path)}

        valid, message = SessionDataLoader.validate_session_file(session)

        assert valid is False
        assert "Not a .npy file" in message

    def test_validate_session_file_no_path(self):
        """Test validation fails when file_path missing."""
        session = {}

        valid, message = SessionDataLoader.validate_session_file(session)

        assert valid is False
        assert "missing file_path" in message
