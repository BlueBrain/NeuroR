from pathlib import Path

from neurom import load_neuron
from nose.tools import assert_raises
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)

import neuror.cut_plane.legacy_detection as tested

DATA_PATH = Path(__file__).parent / 'data'

SIMPLE = load_neuron(DATA_PATH / 'simple.swc')


def test_legacy():
    points, sign = tested.cut_detect(SIMPLE, 'y')
    assert_array_equal(points,
                       [[ 6., -4., 0.],
                        [-5., -4., 0.]])
    assert_equal(sign, -1)

    assert_raises(Exception, tested.cut_detect, SIMPLE, 'z')
