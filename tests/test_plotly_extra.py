'''Tests in this file are to be run only with the [plotly] extra installed'''
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
from numpy.testing import assert_equal

from neuror.cli import cli


DATA = Path(__file__).parent / 'data'


def test_report():
    with TemporaryDirectory('test-report') as folder:
        runner = CliRunner()
        result = runner.invoke(cli, ['report', str(DATA), folder])
        assert_equal(result.exit_code, 0)
