from typer.testing import CliRunner

from pr_review.cli import app


def test_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "review" in result.output and "doctor" in result.output
