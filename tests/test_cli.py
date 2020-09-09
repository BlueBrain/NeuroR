import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from nose.tools import assert_equal
from click.testing import CliRunner

from neuror.cli import cli

DATA = Path(__file__).parent / 'data'


def test_repair_file():
    runner = CliRunner()
    with TemporaryDirectory('test-file') as folder:
        result = runner.invoke(cli, ['cut-plane', 'repair', 'file',
                                     str(DATA / 'real.asc'),
                                     str(Path(folder, 'out.asc'))])
        assert_equal(result.exit_code, 0, result.exception)


def test_repair_folder():
    runner = CliRunner()
    with TemporaryDirectory('test-cli-folder') as folder:
        result = runner.invoke(cli, ['cut-plane', 'repair', 'folder',
                                     str(DATA / 'input-repair-all'),
                                     folder])
        assert_equal(result.exit_code, 0, result.exception)
        assert_equal(set(str(path.relative_to(folder)) for path in Path(folder).rglob('*')),
                     {'simple.asc', 'simple2.asc'})


def test_repair_with_plane():
    runner = CliRunner()
    input_path = DATA / 'input-repair-all'
    with TemporaryDirectory('test-cli-folder') as folder:
        result = runner.invoke(cli, ['cut-plane', 'repair', 'folder',
                                     str(input_path),
                                     folder,
                                     '--cut-file-dir', str(input_path / 'planes')])
        assert_equal(result.exit_code, 0, result.exc_info)


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
