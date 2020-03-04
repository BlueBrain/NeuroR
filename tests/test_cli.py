import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from nose.tools import assert_equal
from click.testing import CliRunner

from neuror.cli import cli

DATA = Path(__file__).parent / 'data'


def test_cli():
    runner = CliRunner()

    with TemporaryDirectory('test-file') as folder:
        result = runner.invoke(cli, ['cut-plane', 'repair', 'file',
                                     str(DATA / 'real.asc'),
                                     '/tmp/test_repair_cli.asc'])
        assert_equal(result.exit_code, 0)

    with TemporaryDirectory('test-cli-folder') as folder:
        result = runner.invoke(cli, ['cut-plane', 'repair', 'folder', str(DATA), folder])
        assert_equal(result.exit_code, 0)


def test_cli_full():
    runner = CliRunner()
    with TemporaryDirectory('test-cli-full') as tmp_folder:
        test_folder = str(Path(tmp_folder, 'test-full-repair'))
        shutil.copytree(DATA / 'test-full-repair', test_folder)
        result = runner.invoke(cli, ['cut-plane', 'repair', 'full', test_folder])
        assert_equal(result.exit_code, 0)


def test_cli_axon():
    runner = CliRunner()
    with TemporaryDirectory('test-cli-axon') as tmp_folder:
        tmp_folder = Path(tmp_folder)
        result = runner.invoke(cli, ['cut-plane', 'repair', 'file',
                                     '-a', str(DATA / 'real-with-axon.asc'),
                                     str(DATA / 'real-with-axon.asc'),
                                     str(tmp_folder / 'output.asc')])
        assert_equal(result.exit_code, 0)


def test_sanitize():
    runner = CliRunner()
    with TemporaryDirectory('test-cli-axon') as tmp_folder:
        tmp_folder = Path(tmp_folder)
        result = runner.invoke(cli, ['sanitize', 'file',
                                     str(DATA / 'simple-with-duplicates.asc'),
                                     str(tmp_folder / 'output.asc')])
        assert_equal(result.exit_code, 0)

        result = runner.invoke(cli, ['sanitize', 'folder',
                                    str(DATA), str(tmp_folder)])
        assert_equal(result.exit_code, 0)
