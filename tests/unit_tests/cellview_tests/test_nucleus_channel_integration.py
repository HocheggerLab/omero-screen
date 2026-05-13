"""Integration tests for nucleus-channel resolution in CellView import flows.

Covers the wiring from CLI flag / OMERO annotation / interactive prompt
through ``CellViewStateCore.create_from_args`` and the rename-to-DAPI step
performed by ``prepare_for_measurements``.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pandas as pd

from cellview.utils.state import (
    CellViewStateCore,
    create_cellview_state,
)


# ------------------------------------------------------------------
# CSV import path
# ------------------------------------------------------------------


class TestCsvImportWithFlag:
    """CSV path: --nucleus-channel flag wins."""

    def test_legacy_dapi_csv_with_flag(self, sample_data_path):
        args = argparse.Namespace(
            csv=sample_data_path, plate_id=None, nucleus_channel="DAPI"
        )
        state = create_cellview_state(args)
        assert state.nucleus_channel == "DAPI"
        # channel_0..3 record actual fluorophores; legacy plate uses DAPI.
        assert state.channel_0 == "DAPI"

    def test_hoechst_csv_with_flag(self, sample_data_hoechst_path):
        args = argparse.Namespace(
            csv=sample_data_hoechst_path,
            plate_id=None,
            nucleus_channel="Hoechst",
        )
        state = create_cellview_state(args)
        assert state.nucleus_channel == "Hoechst"
        # channel_0..3 preserve the actual fluorophore name.
        assert state.channel_0 == "Hoechst"
        # Raw CSV still has Hoechst columns — rename happens later in
        # prepare_for_measurements.
        assert "intensity_mean_Hoechst_nucleus" in state.df.columns


class TestCsvImportRenameToDapi:
    """After _apply_nucleus_rename, the DataFrame uses DAPI columns."""

    def test_hoechst_csv_renamed(self, sample_data_hoechst_path):
        args = argparse.Namespace(
            csv=sample_data_hoechst_path,
            plate_id=None,
            nucleus_channel="Hoechst",
        )
        state = create_cellview_state(args)
        # Exercise the rename step directly — the full prepare_for_measurements
        # also requires timepoint/image_id which the test fixture does not
        # carry. The rename step is the focus of this integration test.
        state._apply_nucleus_rename()
        cols = set(state.df.columns)
        assert "intensity_mean_Hoechst_nucleus" not in cols
        assert "intensity_mean_DAPI_nucleus" in cols
        assert "intensity_max_DAPI_nucleus" in cols
        # Real fluorophore name retained on state and channel_0.
        assert state.nucleus_channel == "Hoechst"
        assert state.channel_0 == "Hoechst"

    def test_legacy_dapi_csv_is_noop_rename(self, sample_data_path):
        args = argparse.Namespace(
            csv=sample_data_path,
            plate_id=None,
            nucleus_channel="DAPI",
        )
        state = create_cellview_state(args)
        before = set(state.df.columns)
        state._apply_nucleus_rename()
        after = set(state.df.columns)
        # DAPI plate — rename is a no-op; column set unchanged.
        assert before == after
        assert "intensity_mean_DAPI_nucleus" in after


class TestCsvImportNonInteractive:
    """E9: non-interactive session without --nucleus-channel raises."""

    def test_non_interactive_without_flag_raises(self, sample_data_path):
        from cellview.utils.error_classes import StateError

        args = argparse.Namespace(
            csv=sample_data_path, plate_id=None, nucleus_channel=None
        )
        # Force non-TTY so the prompt is unavailable (pytest is non-TTY anyway,
        # but this makes the intent explicit).
        with patch(
            "cellview.utils.nucleus_channel.sys.stdin.isatty",
            return_value=False,
        ):
            import pytest

            with pytest.raises(StateError, match="non-interactive"):
                create_cellview_state(args)


# ------------------------------------------------------------------
# OMERO import path
# ------------------------------------------------------------------


class TestOmeroImportNucleusChannel:
    """OMERO route reads nucleus_channel from plate annotation."""

    def test_omero_nucleus_channel_from_plate(self):
        args = argparse.Namespace(
            csv=None, plate_id=12345, nucleus_channel=None
        )
        with patch.object(
            CellViewStateCore, "parse_omero_data"
        ) as mock_parse:
            mock_parse.return_value = (
                pd.DataFrame({"intensity_mean_DAPI_nucleus": [1.0]}),
                "Project",
                "Experiment",
                "2024-01-01",
                "Owner",
                "Hoechst",  # plate annotation says Hoechst
            )
            state = create_cellview_state(args)
        assert state.nucleus_channel == "Hoechst"

    def test_cli_flag_overrides_omero_annotation(self):
        """--nucleus-channel wins over plate annotation."""
        args = argparse.Namespace(
            csv=None, plate_id=12345, nucleus_channel="DAPI"
        )
        with patch.object(
            CellViewStateCore, "parse_omero_data"
        ) as mock_parse:
            mock_parse.return_value = (
                pd.DataFrame({"intensity_mean_DAPI_nucleus": [1.0]}),
                "Project",
                "Experiment",
                "2024-01-01",
                "Owner",
                "Hoechst",
            )
            state = create_cellview_state(args)
        assert state.nucleus_channel == "DAPI"
