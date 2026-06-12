"""Unit tests for the pure channel-role resolver in metadata_parser.

These tests cover :func:`resolve_channel_roles` and :func:`strip_role_suffix` in
isolation — no OMERO connection or fixtures required.
"""

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
    def test_suffix_nucleus_wins(self):
        roles = resolve_channel_roles({"Hoechst_nucleus": 0, "Tub": 1})
        assert roles == {"nucleus": "Hoechst_nucleus", "cell": "Tub"}

    def test_suffix_cell(self):
        roles = resolve_channel_roles({"DAPI": 0, "CellMask_cell": 1})
        assert roles == {"nucleus": "DAPI", "cell": "CellMask_cell"}

    def test_suffix_case_insensitive(self):
        roles = resolve_channel_roles(
            {"Hoechst_NUCLEUS": 0, "Phalloidin_Cell": 1}
        )
        assert roles == {
            "nucleus": "Hoechst_NUCLEUS",
            "cell": "Phalloidin_Cell",
        }

    def test_alias_nucleus(self):
        for alias in ("DAPI", "Hoechst", "DNA", "H2B_RFP"):
            roles = resolve_channel_roles({alias: 0})
            assert roles == {"nucleus": alias}

    def test_tub_only_as_separate_token(self):
        """Cell-role auto-detection requires ``tub`` as a separate token.

        Older naive substring matching also classified things like
        ``aTub``, ``Tubulin``, ``stub`` as cell. The new rule splits on
        ``_`` / ``-`` and requires an exact token match — so ``Tub_GFP``
        works but ``aTub`` does not. Users with such legacy names can
        either rename (``a_Tub``) or use the explicit ``_cell`` suffix.
        """
        # Recognised: token "tub" present after splitting
        for name in ("Tub", "tub", "Tub_GFP", "tub_gfp", "a_Tub", "Tub-AF488"):
            roles = resolve_channel_roles({"DAPI": 0, name: 1})
            assert roles == {"nucleus": "DAPI", "cell": name}, name

        # Not recognised: "tub" only appears as part of another token
        for name in ("aTub", "Tubulin", "stub", "tube"):
            roles = resolve_channel_roles(
                {"DAPI": 0, name: 1, f"{name}_cell": 2}
            )
            # Only the explicit suffix wins; the bare name has no role
            assert roles == {"nucleus": "DAPI", "cell": f"{name}_cell"}, name

    def test_gapdh_is_cell_alias(self):
        """``gapdh`` triggers the cell role, case-insensitively and as a token."""
        for name in ("gapdh", "GAPDH", "Gapdh", "GAPDH_488", "cell_gapdh"):
            roles = resolve_channel_roles({"DAPI": 0, name: 1})
            assert roles == {"nucleus": "DAPI", "cell": name}, name

    def test_feature_channels_have_no_role(self):
        roles = resolve_channel_roles(
            {"Hoechst": 0, "Tub": 1, "p21": 2, "EdU": 3}
        )
        assert roles == {"nucleus": "Hoechst", "cell": "Tub"}

    def test_rfp_no_longer_nucleus_alias(self):
        with pytest.raises(ChannelAnnotationError, match="No nucleus channel"):
            resolve_channel_roles({"RFP": 0, "GFP": 1})

    def test_h2b_rfp_is_nucleus(self):
        roles = resolve_channel_roles({"H2B_RFP": 0})
        assert roles["nucleus"] == "H2B_RFP"

    def test_missing_nucleus_raises(self):
        with pytest.raises(ChannelAnnotationError, match="No nucleus channel"):
            resolve_channel_roles({"GFP": 0, "RFP": 1})

    def test_duplicate_nucleus_raises(self):
        with pytest.raises(
            ChannelAnnotationError, match="Multiple channels resolve to role 'nucleus'"
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
            {"DAPI_nucleus": 0, "Tub": 1}
        )
        assert roles["nucleus"] == "DAPI_nucleus"

    def test_nucleus_only_segmentation(self):
        roles = resolve_channel_roles({"DAPI": 0, "EdU": 1})
        assert roles == {"nucleus": "DAPI"}
        assert "cell" not in roles

    def test_does_not_mutate_input(self):
        channel_data = {"Hoechst": 0, "Tub": 1}
        resolve_channel_roles(channel_data)
        assert channel_data == {"Hoechst": 0, "Tub": 1}


class TestStripRoleSuffix:
    def test_strips_nucleus(self):
        assert strip_role_suffix("Hoechst_nucleus") == "Hoechst"

    def test_strips_cell(self):
        assert strip_role_suffix("CellMask_cell") == "CellMask"

    def test_case_insensitive_match_preserves_case(self):
        assert strip_role_suffix("Hoechst_NUCLEUS") == "Hoechst"

    def test_no_suffix_returns_input(self):
        assert strip_role_suffix("DAPI") == "DAPI"
        assert strip_role_suffix("Tub") == "Tub"

    def test_only_strips_trailing_suffix(self):
        assert strip_role_suffix("cellMask_nucleus") == "cellMask"
        assert strip_role_suffix("nuclei_marker") == "nuclei_marker"

    def test_idempotent(self):
        stripped = strip_role_suffix("Hoechst_nucleus")
        assert strip_role_suffix(stripped) == stripped


class TestMetadataParserChannelRoles:
    """Integration of resolve_channel_roles into MetadataParser._validate_channel_data.

    Post-refactor: channel_data is preserved verbatim — no DAPI rename. The actual
    fluorophore name flows through to feature columns; channel_roles records which
    name plays which role.
    """

    def test_hoechst_preserved_with_nucleus_role(self):
        parser = _make_parser({"Hoechst": "0", "Tub": "1", "EdU": "2"})
        errors = parser._validate_channel_data()
        assert _filter_non_image_errors(errors) == []
        assert parser.channel_data == {"Hoechst": "0", "Tub": "1", "EdU": "2"}
        assert parser.channel_roles == {"nucleus": "Hoechst", "cell": "Tub"}

    def test_dapi_user_name_preserved(self):
        parser = _make_parser({"DAPI": "0", "Tub": "1"})
        errors = parser._validate_channel_data()
        assert _filter_non_image_errors(errors) == []
        assert parser.channel_data == {"DAPI": "0", "Tub": "1"}
        assert parser.channel_roles == {"nucleus": "DAPI", "cell": "Tub"}

    def test_h2b_rfp_preserved(self):
        parser = _make_parser({"H2B_RFP": "0", "Tub": "1"})
        errors = parser._validate_channel_data()
        assert _filter_non_image_errors(errors) == []
        assert parser.channel_data == {"H2B_RFP": "0", "Tub": "1"}
        assert parser.channel_roles == {"nucleus": "H2B_RFP", "cell": "Tub"}

    def test_suffix_marked_channel_populates_role(self):
        parser = _make_parser({"Hoechst_nucleus": "0", "CellMask_cell": "1"})
        errors = parser._validate_channel_data()
        assert _filter_non_image_errors(errors) == []
        assert "Hoechst_nucleus" in parser.channel_data
        assert parser.channel_roles == {
            "nucleus": "Hoechst_nucleus",
            "cell": "CellMask_cell",
        }

    def test_nucleus_only_segmentation(self):
        parser = _make_parser({"DAPI": "0", "EdU": "1"})
        errors = parser._validate_channel_data()
        assert _filter_non_image_errors(errors) == []
        assert parser.channel_roles == {"nucleus": "DAPI"}
        assert "cell" not in parser.channel_roles

    def test_missing_nucleus_reports_error_and_empty_roles(self):
        parser = _make_parser({"GFP": "0", "RFP": "1"})
        errors = parser._validate_channel_data()
        non_image_errors = _filter_non_image_errors(errors)
        assert any("nucleus" in e.lower() for e in non_image_errors)
        assert parser.channel_roles == {}


class TestFeatureChannelToken:
    """Feature column tokens preserve the actual channel name (suffix-stripped).

    The nucleus channel is no longer renamed to "DAPI"; cellcycle_analysis
    receives the actual nucleus channel name and constructs column names
    dynamically.
    """

    def _make_image_properties(self, nuclei_channel: str):
        """Build a minimal ImageProperties stand-in for testing token logic."""
        from types import SimpleNamespace

        from omero_screen.image_analysis import ImageProperties

        instance = object.__new__(ImageProperties)
        instance._image = SimpleNamespace(_nucleus_channel=nuclei_channel)  # type: ignore[attr-defined]
        return instance

    def test_nucleus_role_preserves_hoechst(self):
        props = self._make_image_properties(nuclei_channel="Hoechst")
        assert props._feature_channel_token("Hoechst") == "Hoechst"

    def test_nucleus_role_emits_dapi_when_dapi(self):
        props = self._make_image_properties(nuclei_channel="DAPI")
        assert props._feature_channel_token("DAPI") == "DAPI"

    def test_nucleus_role_strips_suffix(self):
        props = self._make_image_properties(nuclei_channel="H2B_RFP_nucleus")
        assert props._feature_channel_token("H2B_RFP_nucleus") == "H2B_RFP"

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
    a different nuclei-channel name (e.g. 'Hoechst' post-step-7).
    """

    def test_dapi_classifier_resolves_against_hoechst_plate(self):
        import numpy as np

        from omero_screen.image_classifier import _resolve_nucleus_fallback

        image_data = {"Hoechst": np.zeros((1, 4, 4)), "EdU": np.ones((1, 4, 4))}
        resolved = _resolve_nucleus_fallback("DAPI", image_data)
        assert resolved is not None
        assert resolved.shape == (1, 4, 4)
        assert resolved.sum() == 0  # picked Hoechst (zeros), not EdU (ones)

    def test_suffix_marked_classifier_channel_resolves(self):
        import numpy as np

        from omero_screen.image_classifier import _resolve_nucleus_fallback

        image_data = {"DAPI": np.zeros((1, 4, 4))}
        resolved = _resolve_nucleus_fallback("MyMarker_nucleus", image_data)
        assert resolved is not None

    def test_non_nucleus_channel_returns_none(self):
        import numpy as np

        from omero_screen.image_classifier import _resolve_nucleus_fallback

        image_data = {"Hoechst": np.zeros((1, 4, 4))}
        # EdU is not a nuclei alias — no fallback
        resolved = _resolve_nucleus_fallback("EdU", image_data)
        assert resolved is None

    def test_no_nucleus_in_image_data_returns_none(self):
        import numpy as np

        from omero_screen.image_classifier import _resolve_nucleus_fallback

        image_data = {"EdU": np.ones((1, 4, 4)), "p21": np.ones((1, 4, 4))}
        resolved = _resolve_nucleus_fallback("DAPI", image_data)
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
            ["H2B_RFP_nucleus", "Phalloidin_cell", "p21"]
        )
        assert colors["H2B_RFP_nucleus"] == "blue"
        assert colors["Phalloidin_cell"] == "green"
        assert "p21" not in colors

    def test_gapdh_plate(self):
        colors = self._color_map(["DAPI", "GAPDH", "EdU"])
        assert colors["DAPI"] == "blue"
        assert colors["GAPDH"] == "green"

    def test_no_nucleus_or_cell_yields_empty_map(self):
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
