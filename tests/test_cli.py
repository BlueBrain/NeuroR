from pathlib import Path

from click.testing import CliRunner
from neuror.cli import cli

DATA = Path(__file__).parent / 'data'


def _run(command):
    return CliRunner().invoke(cli, command, catch_exceptions=False)


def test_error_annotation_file(tmpdir):
    result = _run([
        'error-annotation', 'file',
        str(DATA / 'test-error-detection/error-morph.asc'),
        str(tmpdir / 'out.asc'),
    ])
    assert result.exit_code == 0, result.exception


def test_error_annotation_folder(tmpdir):
    result = _run([
        'error-annotation', 'folder',
        str(DATA / 'test-error-detection'),
        str(tmpdir)
    ])
    assert result.exit_code == 0, result.exception
    assert (set(str(p.relative_to(tmpdir)) for p in Path(tmpdir).rglob('*')) ==
                 {'simple.asc', 'error-morph.asc'})


def test_repair_file(tmpdir):
    result = _run([
        'cut-plane', 'repair', 'file', str(DATA / 'real.asc'), str(tmpdir / 'out.asc'),
    ])
    assert result.exit_code == 0, result.exception


def test_repair_folder(tmpdir):
    result = _run([
        'cut-plane', 'repair', 'folder', str(DATA / 'input-repair-all'), str(tmpdir)
    ])
    assert result.exit_code == 0, result.exception
    assert (set(str(p.relative_to(tmpdir)) for p in Path(tmpdir).rglob('*')) ==
                 {'simple.asc', 'simple2.asc'})


def test_repair_with_plane(tmpdir):
    input_path = DATA / 'input-repair-all'
    result = _run([
        'cut-plane', 'repair', 'folder',
        str(input_path), str(tmpdir), '--cut-file-dir', str(input_path / 'planes'),
    ])
    assert result.exit_code == 0, result.exc_info


def test_cli_axon(tmpdir):
    result = _run([
        'cut-plane', 'repair', 'file',
        '-a', str(DATA / 'real-with-axon.asc'),
        str(DATA / 'real-with-axon.asc'),
        str(tmpdir / 'output.asc')
    ])
    assert result.exit_code == 0


def test_sanitize(tmpdir):
    result = _run([
        'sanitize', 'file', str(DATA / 'simple-with-duplicates.asc'), str(tmpdir / 'output.asc')
    ])
    assert result.exit_code == 0

    result = _run([
        'sanitize', 'folder', str(DATA), str(tmpdir)
    ])
    assert result.exit_code == 0
