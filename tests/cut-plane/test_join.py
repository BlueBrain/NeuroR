import json
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
from nose.tools import assert_dict_equal, assert_equal

from neuror.cut_plane.cli import cli

DATA = Path(__file__).parent.parent / 'data'


def test_join():
    with TemporaryDirectory(prefix='cut-plane-join') as temp_dir:
        out_filename = str(Path(temp_dir, 'plane.json'))

        runner = CliRunner()
        args = ['join',
                out_filename,
                str(DATA / 'plane1.json'),
                str(DATA / 'plane2.json'),
                str(DATA / 'plane3.json')]

        result = runner.invoke(cli, args)
        assert result.exit_code == 0

        with open(out_filename) as f:
            actual = json.load(f)

        with open(out_filename) as f:
            expected = json.load(f)

        assert_equal(len(actual), len(expected))
        for act, exp in zip(actual, expected):
            assert_dict_equal(act, exp)
