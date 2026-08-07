"""Unit tests for Dataset Information panel refreshing.

The panel is filled only when the classifier dropdown changes, so without
an explicit refresh its session and annotation totals go stale as soon as
annotations are saved. These tests pin the refresh contract without
standing up a Qt application: ``refresh`` is called as an unbound method
against a stub carrying just the attributes it reads.
"""

from unittest.mock import MagicMock

from omero_screen_napari._classifier_selector import ClassifierInfoPanel


class _PanelStub:
    """Minimal stand-in exposing what ``refresh`` touches."""

    def __init__(self, classifier=None, db=None, omero_data=None, user_data=None):
        self._current_classifier = classifier
        self._current_db = db
        self._omero_data = omero_data
        self._user_data = user_data
        self.update_info = MagicMock()


class TestRefresh:
    """Tests for re-querying the displayed classifier."""

    def test_requeries_the_displayed_classifier(self):
        """Refresh re-runs the same lookup that first filled the panel."""
        db = MagicMock()
        stub = _PanelStub(classifier="mit-rel", db=db)

        ClassifierInfoPanel.refresh(stub)

        stub.update_info.assert_called_once_with(
            "mit-rel", db, omero_data=None, user_data=None
        )

    def test_passes_through_omero_and_user_data(self):
        """The session-manager context survives a refresh."""
        db, omero_data, user_data = MagicMock(), MagicMock(), MagicMock()
        stub = _PanelStub("clf", db, omero_data, user_data)

        ClassifierInfoPanel.refresh(stub)

        stub.update_info.assert_called_once_with(
            "clf", db, omero_data=omero_data, user_data=user_data
        )

    def test_noop_when_no_classifier_displayed(self):
        """Refreshing an empty panel does nothing rather than erroring."""
        stub = _PanelStub(classifier=None, db=MagicMock())

        ClassifierInfoPanel.refresh(stub)

        stub.update_info.assert_not_called()

    def test_noop_without_a_database(self):
        """A panel with no DB handle can't be refreshed."""
        stub = _PanelStub(classifier="clf", db=None)

        ClassifierInfoPanel.refresh(stub)

        stub.update_info.assert_not_called()

    def test_refresh_is_repeatable(self):
        """Each save triggers its own re-query; nothing is cached."""
        stub = _PanelStub("clf", MagicMock())

        ClassifierInfoPanel.refresh(stub)
        ClassifierInfoPanel.refresh(stub)

        assert stub.update_info.call_count == 2
