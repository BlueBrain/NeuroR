from os.path import dirname, join as joinp
from nose.tools import assert_equal
from click.testing import CliRunner
from repair.cli import cli

PATH = joinp(dirname(__file__), 'data')


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ['report', PATH, PATH])
    assert_equal(result.exit_code, 0)

    result = runner.invoke(cli, ['file',
                                 joinp(PATH, 'real.asc'),
                                 '/tmp/test_repair_cli.asc'])
    assert_equal(result.exit_code, 0)

    result = runner.invoke(cli, ['folder', PATH, '/tmp/'])
    assert_equal(result.exit_code, 0)
