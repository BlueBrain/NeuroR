import json
from pathlib import Path

from click.testing import CliRunner

from neuror.cli import cut_plane

DATA = Path(__file__).parent.parent / 'data'


def test_join(tmpdir):
    out_filename = str(tmpdir / 'plane.json')

    runner = CliRunner()
    args = ['join',
            out_filename,
            str(DATA / 'plane1.json'),
            str(DATA / 'plane2.json'),
            str(DATA / 'plane3.json')]

    result = runner.invoke(cut_plane, args)
    assert result.exit_code == 0

    with open(out_filename) as f:
        actual = json.load(f)

    with open(out_filename) as f:
        expected = json.load(f)

    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        assert act == exp
