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


from neuror.cut_plane import CutPlane
def test_connard():
    filename = '/gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_new_code-2020-09-02-cut-plane-25/out-legacy/02_ZeroDiameterFix/sm080523a1-5_idB.h5'
    plane = CutPlane.find_legacy(filename, 'z')
