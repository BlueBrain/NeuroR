import json
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
from numpy.testing import assert_array_almost_equal

from neuror.cut_plane.cli import cli

DATA = Path(__file__).parent.parent / 'data'

NEURON_SLICE = str(DATA / 'Neuron_slice.h5')


def test_cli_compute_file():
    runner = CliRunner()

    # vanilla test
    result = runner.invoke(cli, ['compute', 'file', NEURON_SLICE])
    if result.exit_code:
        raise AssertionError(result.exception)

    # test writing output
    with TemporaryDirectory(prefix='test_cli_compute_file') as temp_dir:
        json_file = str(Path(temp_dir, 'plane.json'))
        result = runner.invoke(cli, ['compute', 'file', NEURON_SLICE,
                                     '-o', json_file])
        if result.exit_code:
            raise AssertionError(result.exception)

        with open(json_file) as f:
            data = json.load(f)
            assert_array_almost_equal(data['cut-plane']['d'], -48.68020515427703)

    # test forcing plane
    with TemporaryDirectory(prefix='test_cli_compute_file') as temp_dir:
        json_file = str(Path(temp_dir, 'plane.json'))
        result = runner.invoke(cli, ['compute', 'file', NEURON_SLICE,
                                     '-o', json_file,
                                     '--plane', 'y'])
        if result.exit_code:
            raise AssertionError(result.exception)

        with open(json_file) as f:
            data = json.load(f)
            assert_array_almost_equal([data['cut-plane']['a'],
                                       data['cut-plane']['b'],
                                       data['cut-plane']['c']],
                                      [0, 1, 0])

    # test forcing plane and position
    with TemporaryDirectory(prefix='test_cli_compute_file') as temp_dir:
        json_file = str(Path(temp_dir, 'plane.json'))
        result = runner.invoke(cli, ['compute', 'file', NEURON_SLICE,
                                     '-o', json_file,
                                     '--plane', 'y',
                                     '--position', 40])
        if result.exit_code:
            raise AssertionError(result.exception)

        with open(json_file) as f:
            data = json.load(f)
            assert_array_almost_equal([data['cut-plane']['a'],
                                       data['cut-plane']['b'],
                                       data['cut-plane']['c'],
                                       data['cut-plane']['d']],
                                      [0, 1, 0, -40])


def test_cli_compute_folder():
    runner = CliRunner()

    with TemporaryDirectory(prefix='cut-plane-compute-folder') as temp_dir:
        result = runner.invoke(cli, ['compute', 'folder', str(DATA), temp_dir])
    if result.exit_code:
        raise AssertionError(result.exception)
