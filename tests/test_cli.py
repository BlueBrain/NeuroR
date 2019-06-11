import shutil
from os.path import dirname, join as joinp
from nose.tools import assert_equal
from click.testing import CliRunner
from repair.cli import cli

from .utils import setup_tempdir

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


def test_cli_full():
    runner = CliRunner()
    with setup_tempdir('test-cli-full', cleanup=False) as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(PATH, 'test-full-repair'), test_folder)
        result = runner.invoke(cli, ['full', test_folder])
        assert_equal(result.exit_code, 0)
