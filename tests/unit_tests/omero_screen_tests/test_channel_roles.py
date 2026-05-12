"""Unit tests for the pure channel-role resolver in metadata_parser.

These tests cover :func:`resolve_channel_roles` and :func:`strip_role_suffix` in
isolation — no OMERO connection or fixtures required.
"""

import warnings

import pytest

from omero_utils.message import ChannelAnnotationError

from omero_screen.metadata_parser import (
    MetadataParser,
    resolve_channel_roles,
    strip_role_suffix,
)


def _make_parser(channel_data: dict[str, str]) -> MetadataParser:
    """Construct a MetadataParser bypassing __init__ for isolated validation tests."""
    parser = object.__new__(MetadataParser)
    parser.channel_data = dict(channel_data)
    parser.channel_roles = {}
    parser.plate_id = 0  # used by error messages
    return parser


def _filter_non_image_errors(errors: list[str]) -> list[str]:
    """Drop image-validation errors which require an OMERO connection."""
    return [
        e for e in errors
        if "image" not in e.lower() and "plate" not in e.lower()
    ]


class TestResolveChannelRoles:
    def test_suffix_nuclei_wins(self):
        roles = resolve_channel_roles({"Hoechst_nuclei": 0, "Tub": 1})
        assert roles == {"nuclei": "Hoechst_nuclei", "cell": "Tub"}

    def test_suffix_cell(self):
        roles = resolve_channel_roles({"DAPI": 0, "CellMask_cell": 1})
        assert roles == {"nuclei": "DAPI", "cell": "CellMask_cell"}

    def test_suffix_case_insensitive(self):
        roles = resolve_channel_roles(
            {"Hoechst_NUCLEI": 0, "Phalloidin_Cell": 1}
        )
        assert roles == {
            "nuclei": "Hoechst_NUCLEI",
            "cell": "Phalloidin_Cell",
        }

    def test_alias_nuclei(self):
        for alias in ("DAPI", "Hoechst", "DNA", "H2B_RFP"):
            roles = resolve_channel_roles({alias: 0})
            assert roles == {"nuclei": alias}

    def test_legacy_tub_substring(self):
        roles = resolve_channel_roles({"DAPI": 0, "aTub": 1})
        assert roles == {"nuclei": "DAPI", "cell": "aTub"}

    def test_feature_channels_have_no_role(self):
        roles = resolve_channel_roles(
            {"Hoechst": 0, "Tub": 1, "p21": 2, "EdU": 3}
        )
        assert roles == {"nuclei": "Hoechst", "cell": "Tub"}

    def test_rfp_no_longer_nuclei_alias(self):
        with pytest.raises(ChannelAnnotationError, match="No nuclei channel"):
            resolve_channel_roles({"RFP": 0, "GFP": 1})

    def test_h2b_rfp_is_nuclei(self):
        roles = resolve_channel_roles({"H2B_RFP": 0})
        assert roles["nuclei"] == "H2B_RFP"

    def test_missing_nuclei_raises(self):
        with pytest.raises(ChannelAnnotationError, match="No nuclei channel"):
            resolve_channel_roles({"GFP": 0, "RFP": 1})

    def test_duplicate_nuclei_raises(self):
        with pytest.raises(
            ChannelAnnotationError, match="Multiple channels resolve to role 'nuclei'"
        ):
            resolve_channel_roles({"DAPI": 0, "Hoechst": 1})

    def test_duplicate_cell_raises(self):
        with pytest.raises(
            ChannelAnnotationError, match="Multiple channels resolve to role 'cell'"
        ):
            resolve_channel_roles(
                {"DAPI": 0, "Tub": 1, "CellMask_cell": 2}
            )

    def test_suffix_overrides_alias_collision(self):
        roles = resolve_channel_roles(
            {"DAPI_nuclei": 0, "Tub": 1}
        )
        assert roles["nuclei"] == "DAPI_nuclei"

    def test_nucleus_only_segmentation(self):
        roles = resolve_channel_roles({"DAPI": 0, "EdU": 1})
        assert roles == {"nuclei": "DAPI"}
        assert "cell" not in roles

    def test_does_not_mutate_input(self):
        channel_data = {"Hoechst": 0, "Tub": 1}
        resolve_channel_roles(channel_data)
        assert channel_data == {"Hoechst": 0, "Tub": 1}


class TestStripRoleSuffix:
    def test_strips_nuclei(self):
        assert strip_role_suffix("Hoechst_nuclei") == "Hoechst"

    def test_strips_cell(self):
        assert strip_role_suffix("CellMask_cell") == "CellMask"

    def test_case_insensitive_match_preserves_case(self):
        assert strip_role_suffix("Hoechst_NUCLEI") == "Hoechst"

    def test_no_suffix_returns_input(self):
        assert strip_role_suffix("DAPI") == "DAPI"
        assert strip_role_suffix("Tub") == "Tub"

    def test_only_strips_trailing_suffix(self):
        assert strip_role_suffix("cellMask_nuclei") == "cellMask"
        assert strip_role_suffix("nuclei_marker") == "nuclei_marker"

    def test_idempotent(self):
        stripped = strip_role_suffix("Hoechst_nuclei")
        assert strip_role_suffix(stripped) == stripped


class TestMetadataParserChannelRoles:
    """Integration of resolve_channel_roles into MetadataParser._validate_channel_data."""

    def test_roles_populated_alongside_legacy_rename(self):
        parser = _make_parser({"Hoechst": "0", "Tub": "1", "EdU": "2"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            errors = parser._validate_channel_data()
        assert _filter_non_image_errors(errors) == []
        assert parser.channel_data == {"DAPI": "0", "Tub": "1", "EdU": "2"}
        assert parser.channel_roles == {"nuclei": "DAPI", "cell": "Tub"}

    def test_dapi_user_name_no_rename_no_warning(self):
        parser = _make_parser({"DAPI": "0", "Tub": "1"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parser._validate_channel_data()
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations == []
        assert parser.channel_roles == {"nuclei": "DAPI", "cell": "Tub"}

    def test_hoechst_emits_deprecation_warning(self):
        parser = _make_parser({"Hoechst": "0", "Tub": "1"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parser._validate_channel_data()
        deprecations = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecations) == 1
        assert "Hoechst" in str(deprecations[0].message)
        assert "channel_roles" in str(deprecations[0].message)

    def test_suffix_marked_channel_populates_role(self):
        parser = _make_parser({"Hoechst_nuclei": "0", "CellMask_cell": "1"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            errors = parser._validate_channel_data()
        # The legacy rename does NOT fire on Hoechst_nuclei (it's not a literal alias).
        assert "Hoechst_nuclei" in parser.channel_data
        assert parser.channel_roles["cell"] == "CellMask_cell"
        # No nuclei alias matched, so the legacy "no nuclei" error fires —
        # acceptable during the transition since channel_roles still resolves
        # the nuclei role from the suffix.
        assert parser.channel_roles.get("nuclei") == "Hoechst_nuclei"

    def test_nucleus_only_segmentation_legacy(self):
        parser = _make_parser({"DAPI": "0", "EdU": "1"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            errors = parser._validate_channel_data()
        assert _filter_non_image_errors(errors) == []
        assert parser.channel_roles == {"nuclei": "DAPI"}
        assert "cell" not in parser.channel_roles

    def test_missing_nuclei_reports_error_and_empty_roles(self):
        parser = _make_parser({"GFP": "0", "RFP": "1"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            errors = parser._validate_channel_data()
        non_image_errors = _filter_non_image_errors(errors)
        assert any("nuclei" in e.lower() for e in non_image_errors)
        assert parser.channel_roles == {}


class TestFeatureChannelToken:
    """The nuclei role always emits a canonical 'DAPI' token in feature columns
    (required by cellcycle_analysis); other roles use the suffix-stripped name.
    """

    def _make_image_properties(self, nuclei_channel: str):
        """Build a minimal ImageProperties stand-in for testing token logic."""
        from types import SimpleNamespace

        from omero_screen.image_analysis import ImageProperties

        instance = object.__new__(ImageProperties)
        instance._image = SimpleNamespace(_nuclei_channel=nuclei_channel)  # type: ignore[attr-defined]
        return instance

    def test_nuclei_role_emits_canonical_dapi(self):
        # User's nuclei channel is "Hoechst" (legacy rename would force "DAPI"
        # during the transition; this still emits the canonical token).
        props = self._make_image_properties(nuclei_channel="Hoechst")
        assert props._feature_channel_token("Hoechst") == "DAPI"

    def test_nuclei_role_emits_dapi_when_already_dapi(self):
        props = self._make_image_properties(nuclei_channel="DAPI")
        assert props._feature_channel_token("DAPI") == "DAPI"

    def test_cell_role_strips_suffix(self):
        props = self._make_image_properties(nuclei_channel="DAPI")
        assert props._feature_channel_token("CellMask_cell") == "CellMask"

    def test_cell_role_legacy_tub_unchanged(self):
        props = self._make_image_properties(nuclei_channel="DAPI")
        assert props._feature_channel_token("Tub") == "Tub"

    def test_feature_channel_unchanged(self):
        props = self._make_image_properties(nuclei_channel="DAPI")
        assert props._feature_channel_token("EdU") == "EdU"
        assert props._feature_channel_token("p21") == "p21"


class TestClassifierNucleiFallback:
    """ImageClassifier resolves a model trained on 'DAPI' against a plate using
    a different nuclei-channel name (e.g. 'Hoechst' post-step-7)."""

    def test_dapi_classifier_resolves_against_hoechst_plate(self):
        import numpy as np

        from omero_screen.image_classifier import _resolve_nuclei_fallback

        image_data = {"Hoechst": np.zeros((1, 4, 4)), "EdU": np.ones((1, 4, 4))}
        resolved = _resolve_nuclei_fallback("DAPI", image_data)
        assert resolved is not None
        assert resolved.shape == (1, 4, 4)
        assert resolved.sum() == 0  # picked Hoechst (zeros), not EdU (ones)

    def test_suffix_marked_classifier_channel_resolves(self):
        import numpy as np

        from omero_screen.image_classifier import _resolve_nuclei_fallback

        image_data = {"DAPI": np.zeros((1, 4, 4))}
        resolved = _resolve_nuclei_fallback("MyMarker_nuclei", image_data)
        assert resolved is not None

    def test_non_nuclei_channel_returns_none(self):
        import numpy as np

        from omero_screen.image_classifier import _resolve_nuclei_fallback

        image_data = {"Hoechst": np.zeros((1, 4, 4))}
        # EdU is not a nuclei alias — no fallback
        resolved = _resolve_nuclei_fallback("EdU", image_data)
        assert resolved is None

    def test_no_nuclei_in_image_data_returns_none(self):
        import numpy as np

        from omero_screen.image_classifier import _resolve_nuclei_fallback

        image_data = {"EdU": np.ones((1, 4, 4)), "p21": np.ones((1, 4, 4))}
        resolved = _resolve_nuclei_fallback("DAPI", image_data)
        assert resolved is None


class TestNapariRoleBasedColors:
    """napari display colours are assigned by role, not literal name.

    Mirrors the resolution rules in resolve_channel_roles so that nuclei stays
    blue and cell stays green regardless of user-chosen channel names.
    """

    def _color_map(self, channel_names: list[str]) -> dict[str, str]:
        import sys
        pkg = "/Users/hh65/code/omero-screen/packages/omero-screen-napari/src"
        if pkg not in sys.path:
            sys.path.insert(0, pkg)
        from omero_screen_napari._welldata_widget import (
            _role_based_color_map,
        )

        return _role_based_color_map(channel_names)

    def test_legacy_plate(self):
        colors = self._color_map(["DAPI", "Tub", "EdU"])
        assert colors["DAPI"] == "blue"
        assert colors["Tub"] == "green"

    def test_hoechst_plate(self):
        colors = self._color_map(["Hoechst", "CellMask_cell"])
        assert colors["Hoechst"] == "blue"
        assert colors["CellMask_cell"] == "green"

    def test_suffix_marked_plate(self):
        colors = self._color_map(
            ["H2B_RFP_nuclei", "Phalloidin_cell", "p21"]
        )
        assert colors["H2B_RFP_nuclei"] == "blue"
        assert colors["Phalloidin_cell"] == "green"
        assert "p21" not in colors

    def test_no_nuclei_or_cell_yields_empty_map(self):
        colors = self._color_map(["EdU", "p21", "GFP"])
        assert colors == {}


class TestEditPropertiesNaming:
    """Feature column name construction uses the channel token, not the raw name."""

    def test_emits_canonical_dapi_columns(self):
        from omero_screen.image_analysis import ImageProperties

        feature_dict = ImageProperties._edit_properties(
            "DAPI", "nucleus", ["label", "area", "intensity_mean"]
        )
        assert feature_dict["intensity_mean"] == "intensity_mean_DAPI_nucleus"
        assert feature_dict["area"] == "area_nucleus"

    def test_emits_stripped_cell_columns(self):
        from omero_screen.image_analysis import ImageProperties

        feature_dict = ImageProperties._edit_properties(
            "CellMask", "cell", ["label", "area", "intensity_mean"]
        )
        assert feature_dict["intensity_mean"] == "intensity_mean_CellMask_cell"
        # No suffix duplication (the bug the refactor prevents).
        assert "_cell_cell" not in feature_dict["intensity_mean"]
