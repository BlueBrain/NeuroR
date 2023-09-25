'''Tests in this file are to be run only with the [plotly] extra installed'''
from pathlib import Path

from click.testing import CliRunner

from neuror.cli import cli

DATA = Path(__file__).parent / 'data'


def test_report(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    runner = CliRunner()
    result = runner.invoke(cli, ['report', str(DATA.resolve()), str(tmpdir)], catch_exceptions=False)
    assert result.exit_code == 0
