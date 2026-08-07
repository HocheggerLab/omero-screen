"""Unit tests for remembered direct-load dialog state."""

import json

import pytest
from omero_screen_napari.direct_load_state import (
    DirectLoadState,
    load_state,
    save_state,
    state_path,
)


@pytest.fixture
def base_dir(tmp_path):
    """Stand-in for ~/omeroscreen_trainingdata."""
    return tmp_path


class TestRoundTrip:
    """Tests for saving and restoring state."""

    def test_saved_state_is_restored(self, base_dir):
        """Every field survives a save/load cycle."""
        state = DirectLoadState(
            plate_id=4053,
            well="C7",
            image_input="0, 1",
            timepoint=3,
            cellcycle="G2/M",
            classifier_column="classifier_mitosis",
            classifier_class="metaphase",
            n_crops=250,
        )

        assert save_state("mit-release", state, base_dir) is True
        assert load_state("mit-release", base_dir) == state

    def test_state_is_per_classifier(self, base_dir):
        """One classifier's state never leaks into another's."""
        save_state("clf-a", DirectLoadState(plate_id=1, well="A1"), base_dir)
        save_state("clf-b", DirectLoadState(plate_id=2, well="B2"), base_dir)

        assert load_state("clf-a", base_dir).well == "A1"
        assert load_state("clf-b", base_dir).well == "B2"

    def test_saving_twice_keeps_the_latest(self, base_dir):
        """A later load overwrites the remembered values."""
        save_state("clf", DirectLoadState(plate_id=1), base_dir)
        save_state("clf", DirectLoadState(plate_id=99), base_dir)

        assert load_state("clf", base_dir).plate_id == 99

    def test_save_creates_the_classifier_directory(self, base_dir):
        """State can be written before the classifier dir exists."""
        assert save_state("brand-new", DirectLoadState(), base_dir) is True
        assert state_path("brand-new", base_dir).exists()

    def test_state_file_sits_beside_metadata_not_inside_it(self, base_dir):
        """State is a separate file — metadata.json must stay untouched."""
        metadata = base_dir / "clf" / "metadata.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text('{"user_data": {"crop_size": 100}}')

        save_state("clf", DirectLoadState(plate_id=7), base_dir)

        assert json.loads(metadata.read_text()) == {
            "user_data": {"crop_size": 100}
        }
        assert state_path("clf", base_dir).name == "direct_load_state.json"


class TestDefaults:
    """Tests for the no-state and bad-state paths."""

    def test_missing_file_gives_defaults(self, base_dir):
        """A classifier never loaded before opens with defaults."""
        assert load_state("never-used", base_dir) == DirectLoadState()

    def test_corrupt_file_gives_defaults(self, base_dir):
        """Unparseable JSON degrades to defaults instead of raising."""
        path = state_path("clf", base_dir)
        path.parent.mkdir(parents=True)
        path.write_text("{not json at all")

        assert load_state("clf", base_dir) == DirectLoadState()

    def test_non_dict_json_gives_defaults(self, base_dir):
        """Valid JSON of the wrong shape degrades to defaults."""
        path = state_path("clf", base_dir)
        path.parent.mkdir(parents=True)
        path.write_text('["not", "a", "dict"]')

        assert load_state("clf", base_dir) == DirectLoadState()


class TestFromDict:
    """Tests for tolerant deserialisation."""

    def test_missing_keys_fall_back_to_defaults(self):
        """State written by an older version still loads."""
        state = DirectLoadState.from_dict({"plate_id": 42})

        assert state.plate_id == 42
        assert state.well == DirectLoadState().well
        assert state.n_crops == DirectLoadState().n_crops

    def test_unknown_keys_are_ignored(self):
        """State written by a newer version doesn't raise."""
        state = DirectLoadState.from_dict(
            {"plate_id": 42, "future_field": "whatever"}
        )

        assert state.plate_id == 42

    def test_wrong_types_fall_back_to_defaults(self):
        """A garbled value defaults that field rather than the whole form."""
        state = DirectLoadState.from_dict(
            {"plate_id": "not-a-number", "well": 17, "timepoint": None}
        )

        assert state.plate_id == DirectLoadState().plate_id
        assert state.well == DirectLoadState().well
        assert state.timepoint == DirectLoadState().timepoint

    def test_numeric_strings_are_coerced(self):
        """JSON round-trips of ints as strings still restore."""
        state = DirectLoadState.from_dict({"plate_id": "4053"})

        assert state.plate_id == 4053

    def test_empty_dict_is_all_defaults(self):
        """An empty state file is equivalent to no state file."""
        assert DirectLoadState.from_dict({}) == DirectLoadState()
