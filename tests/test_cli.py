import shutil
import os
from os.path import dirname
from nose.tools import assert_equal
from click.testing import CliRunner
from morph_repair.cli import cli

from .utils import setup_tempdir

PATH = os.path.join(dirname(__file__), 'data')


def test_cli():
    with setup_tempdir('test-report') as folder:
        runner = CliRunner()
        result = runner.invoke(cli, ['report', PATH, folder])
        assert_equal(result.exit_code, 0)

    with setup_tempdir('test-file') as folder:
        result = runner.invoke(cli, ['file',
                                     os.path.join(PATH, 'real.asc'),
                                     '/tmp/test_repair_cli.asc'])
        assert_equal(result.exit_code, 0)

    with setup_tempdir('test-cli-folder') as folder:
        result = runner.invoke(cli, ['folder', PATH, folder])
        assert_equal(result.exit_code, 0)


def test_cli_full():
    runner = CliRunner()
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = os.path.join(tmp_folder, 'test-full-repair')
        shutil.copytree(os.path.join(PATH, 'test-full-repair'), test_folder)
        result = runner.invoke(cli, ['full', test_folder])
        assert_equal(result.exit_code, 0)


def test_cli_axon():
    runner = CliRunner()
    with setup_tempdir('test-cli-axon') as tmp_folder:
        result = runner.invoke(cli, ['file', '-a', os.path.join(PATH, 'real-with-axon.asc'),
                                     os.path.join(PATH, 'real-with-axon.asc'),
                                     os.path.join(tmp_folder, 'output.asc')])
        assert_equal(result.exit_code, 0)
