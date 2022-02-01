from pathlib import Path

from click.testing import CliRunner
from neuror.cli import cli

DATA = Path(__file__).parent / 'data'


def assert_cli_runs(cmd):
    runner = CliRunner()
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0, result.exception


def test_error_annotation_file(tmpdir):
    assert_cli_runs(
        [
            'error-annotation', 'file',
            str(DATA / 'test-error-detection/error-morph.asc'),
            str(tmpdir / 'out.asc')
        ]
    )


def test_error_annotation_folder(tmpdir):
    assert_cli_runs(
        [
            'error-annotation', 'folder',
            str(DATA / 'test-error-detection'),
            str(tmpdir)
        ]
    )
    assert (set(str(p.relative_to(tmpdir)) for p in Path(tmpdir).rglob('*')) ==
                 {'simple.asc', 'error-morph.asc'})


def test_repair_file(tmpdir):
    assert_cli_runs(
        [
            'cut-plane', 'repair', 'file',
            str(DATA / 'real.asc'),
            str(tmpdir / 'out.asc')
        ]
    )


def test_repair_folder(tmpdir):
    assert_cli_runs(
        [
            'cut-plane', 'repair', 'folder',
            str(DATA / 'input-repair-all'),
            str(tmpdir)
        ]
    )
    assert (set(str(p.relative_to(tmpdir)) for p in Path(tmpdir).rglob('*')) ==
                 {'simple.asc', 'simple2.asc'})


def test_repair_with_plane(tmpdir):
    input_path = DATA / 'input-repair-all'

    assert_cli_runs(
        [
            'cut-plane', 'repair', 'folder',
            str(input_path),
            str(tmpdir),
            '--cut-file-dir', str(input_path / 'planes')

        ]
    )


def test_cli_axon(tmpdir):
    assert_cli_runs(
        [
            'cut-plane', 'repair', 'file',
            '-a', str(DATA / 'real-with-axon.asc'),
            str(DATA / 'real-with-axon.asc'),
            str(tmpdir / 'output.asc')

        ]
    )


def test_sanitize(tmpdir):

    assert_cli_runs(
        [
            'sanitize', 'file',
            str(DATA / 'simple-with-duplicates.asc'),
            str(tmpdir / 'output.asc')
        ]
    )
    assert Path(tmpdir, 'output.asc').exists()

    assert_cli_runs(
        [
            'sanitize', 'file',
            str(DATA / 'neurite-with-multiple-types.swc'),
            str(tmpdir / 'file-neurite-with-multiple-types.swc'),
            '--allow-inhomogeneous-trees',
        ]
    )
    assert Path(tmpdir, 'file-neurite-with-multiple-types.swc').exists()

    assert_cli_runs(
        [
            'sanitize', 'folder',
            str(DATA), str(tmpdir)
        ]
    )
    # the inhomogeneous cell will not be sanitized
    assert not Path(tmpdir, 'neurite-with-multiple-types.swc').exists()

    assert_cli_runs(
        [
            'sanitize', 'folder',
            str(DATA), str(tmpdir),
            '--allow-inhomogeneous-trees',
        ]
    )
    # the inhomogeneous cell will be sanitized
    assert Path(tmpdir, 'neurite-with-multiple-types.swc').exists()
