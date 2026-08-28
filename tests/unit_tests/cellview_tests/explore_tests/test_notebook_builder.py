"""Tests for the --template option on the explore command."""

from unittest.mock import patch

from click.testing import CliRunner

from cellview.cli import cli


def _template_for(argv):
    with patch("cellview.main.handle_explore") as mock:
        result = CliRunner().invoke(cli, ["explore", *argv])
    assert result.exit_code == 0, result.output
    return mock.call_args.kwargs["template"]


class TestTemplateCLIArgs:
    """Tests for template-related CLI arguments."""

    def test_template_default(self) -> None:
        assert _template_for(["12345"]) == "cellcycle"

    def test_template_custom(self) -> None:
        assert _template_for(["12345", "--template", "drug_screen"]) == (
            "drug_screen"
        )
