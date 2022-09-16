import json
from pathlib import Path

from click.testing import CliRunner
from numpy.testing import assert_array_almost_equal

from neuror.cli import compute

DATA = Path(__file__).parent.parent / "data"

NEURON_SLICE = str(DATA / "Neuron_slice.h5")


def test_cli_compute_file_invalid():
    runner = CliRunner()
    result = runner.invoke(compute, ["file", NEURON_SLICE])
    if result.exit_code:
        raise AssertionError(result.exception)


def test_cli_compute_file(tmpdir):
    runner = CliRunner()
    json_file = str(tmpdir / "plane.json")
    result = runner.invoke(compute, ["file", NEURON_SLICE, "-o", json_file])
    if result.exit_code:
        raise AssertionError(result.exception)

    with open(json_file) as f:
        data = json.load(f)
        assert_array_almost_equal(data["cut-plane"]["d"], -48.68020515427703)


def test_cli_compute_file_plane(tmpdir):
    runner = CliRunner()
    json_file = str(tmpdir / "plane.json")
    result = runner.invoke(compute, ["file", NEURON_SLICE, "-o", json_file, "--plane", "y"])
    if result.exit_code:
        raise AssertionError(result.exception)

    with open(json_file) as f:
        data = json.load(f)
        assert_array_almost_equal(
            [data["cut-plane"]["a"], data["cut-plane"]["b"], data["cut-plane"]["c"]],
            [0, 1, 0],
        )


def test_cli_compute_file_plane_pos(tmpdir):
    runner = CliRunner()
    json_file = str(tmpdir / "plane.json")
    result = runner.invoke(
        compute,
        ["file", NEURON_SLICE, "-o", json_file, "--plane", "y", "--position", 40],
    )
    if result.exit_code:
        raise AssertionError(result.exception)

    with open(json_file) as f:
        data = json.load(f)
        assert_array_almost_equal(
            [
                data["cut-plane"]["a"],
                data["cut-plane"]["b"],
                data["cut-plane"]["c"],
                data["cut-plane"]["d"],
            ],
            [0, 1, 0, -40],
        )


def test_cli_compute_folder(tmpdir):
    runner = CliRunner()

    result = runner.invoke(compute, ["folder", str(DATA), str(tmpdir)])
    if result.exit_code:
        raise AssertionError(result.exception)
