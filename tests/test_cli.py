import os


from nose.tools import assert_equal
PATH = os.path.dirname(__file__)

from click.testing import CliRunner

from repair.cli import cli

PATH = os.path.dirname(__file__)


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ['report',
                                 os.path.join(PATH, 'data'),
                                 os.path.join(PATH, 'data')])
    assert_equal(result.exit_code, 0)

    result = runner.invoke(cli, ['cell',
                                 os.path.join(PATH, 'data', 'real.asc'),
                                 '/tmp/test_repair_cli.asc'])
    assert_equal(result.exit_code, 0)

    result = runner.invoke(cli, ['folder',
                                 os.path.join(PATH, 'data'),
                                 '/tmp/'])
    assert_equal(result.exit_code, 0)
