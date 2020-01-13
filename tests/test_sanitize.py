import os
from os.path import dirname
from pathlib import Path
from morph_repair.sanitize import fix_non_zero_segments, sanitize, sanitize_all

from morphio import Morphology
from numpy.testing import assert_equal, assert_array_equal

from .utils import setup_tempdir

PATH = Path(dirname(__file__), 'data')

def test_fix_non_zero_segments():
    neuron = fix_non_zero_segments(Path(PATH, 'simple-with-duplicates.asc'))
    assert_equal(len(neuron.root_sections), 1)
    assert_array_equal(neuron.section(0).points,
                       [[0., 0., 0.],
                        [1., 1., 0.],
                        [2., 0., 0.],
                        [3., 0., 0.]])


def test_sanitize():
    with setup_tempdir('test-sanitize') as tmp_folder:
        output_path = Path(tmp_folder, 'sanitized.asc')
        sanitize(Path(PATH, 'simple-with-duplicates.asc'), output_path)
        neuron = Morphology(output_path)
        assert_equal(len(neuron.root_sections), 1)
        assert_array_equal(neuron.section(0).points,
                           [[0., 0., 0.],
                            [1., 1., 0.],
                            [2., 0., 0.],
                            [3., 0., 0.]])


def test_sanitize_all():
    with setup_tempdir('test-sanitize') as tmp_folder:
        sanitize_all(Path(PATH), tmp_folder)
        assert_equal(len(os.listdir(tmp_folder)), 14)
