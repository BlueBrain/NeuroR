import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
from nose.tools import ok_
from numpy.testing import assert_equal

from neuror.cli import cli
from neuror.full import full

from .test_full import assert_output_exists

DATA = Path(__file__).parent / 'data'


def test_full_custom_plots_dir():
    try:
        from neuror.view import plot_repaired_neuron
    except ImportError:
        print('Skipping this test as [plotly] extra is not installed here')
        return

    with TemporaryDirectory('test-cli-full') as tmp_folder:
        test_folder = Path(tmp_folder, 'test-full-repair')
        shutil.copytree(DATA / 'test-full-repair', test_folder)

        custom_path = test_folder / 'plots_custom'
        full(test_folder, plots_dir=custom_path)
        assert_output_exists(test_folder)
        assert_equal(len(list(custom_path.iterdir())), 3)
        ok_((custom_path / 'report.pdf').exists())


def test_report():
    try:
        from neuror.view import plot_repaired_neuron
    except ImportError:
        print('Skipping this test as [plotly] extra is not installed here')
        return

    with TemporaryDirectory('test-report') as folder:
        runner = CliRunner()
        result = runner.invoke(cli, ['report', str(DATA), folder])
        assert_equal(result.exit_code, 0)
