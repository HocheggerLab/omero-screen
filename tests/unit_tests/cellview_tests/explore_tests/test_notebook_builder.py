"""Tests for template-related CLI arguments."""

from cellview.cli import get_parser


class TestTemplateCLIArgs:
    """Tests for template-related CLI arguments."""

    def test_template_default(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "12345"])
        assert args.template == "cellcycle"

    def test_template_custom(self) -> None:
        parser = get_parser()
        args = parser.parse_args(
            ["explore", "12345", "--template", "drug_screen"]
        )
        assert args.template == "drug_screen"
