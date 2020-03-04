from pathlib import Path

import numpy as np
from neurom import load_neuron
from nose.tools import assert_raises, ok_
from numpy.testing import assert_array_equal, assert_equal

from morphio.mut import Morphology
from neuror.zero_diameter_fixer import fix_zero_diameters, fix_in_between, Point, fix_from_downstream, fix_from_upstream, fix_neurite

DATA_PATH = Path(__file__).parent / 'data'


def test_fix_in_between():
    neuron = Morphology(DATA_PATH / 'zero-diameter-middle-of-neurite.asc')
    root = neuron.root_sections[0]
    point = Point(root, 0)

    fix_in_between(point, stack=list())
    assert_array_equal(root.diameters, np.array([2., 2., 2.2, 2.4, 2.8, 3.], dtype=np.float32))

    assert_array_equal(root.children[0].diameters, np.array([3, 3, 3, 2.8], dtype=np.float32))
    assert_array_equal(root.children[0].children[0].diameters,
                       np.array([2.8, 2.6, 2.4, 2.2, 2. ], dtype=np.float32))



def test_fix_from_downstream():
    neuron = Morphology(DATA_PATH / 'zero-diameter-middle-of-neurite.asc')
    root = neuron.root_sections[0]
    point = Point(root, 0)

    fix_from_downstream(point)
    assert_array_equal(neuron.section(0).diameters,
                       np.array([2., 2., 1.e-6, 1.e-6, 1.e-6, 3.], dtype=np.float32),
                       "The diameters should not have been modified")


    neuron = Morphology(DATA_PATH / 'zero-diameter-beginning-of-neurite.swc')
    root = neuron.root_sections[0]
    point = Point(root, 0)

    assert_array_equal(root.diameters, [0., 0.])
    fix_from_downstream(point)
    assert_array_equal(root.diameters, [0., 0.],
                       'Diameters should not have been changed '
                       'since the whole neurite has a null diameter')

    neuron = Morphology(DATA_PATH / 'zero-diameter-beginning-of-neurite.swc')
    root = neuron.root_sections[1]
    point = Point(root, 0)

    assert_array_equal(root.diameters, [0, 0, 4])
    fix_from_downstream(point)
    assert_array_equal(root.diameters, [4, 4, 4])


def test_fix_from_upstream():
    neuron = Morphology(DATA_PATH / 'zero-diameter-end-of-neurite.swc')
    root = neuron.root_sections[0]
    point = Point(root, 0)
    leaf = root.children[0]

    fix_from_upstream(point, 12)
    assert_array_equal(leaf.diameters, np.array([2, 2, 2, 2, 2], dtype=np.float32))


def test_fix_zero_diameters():
    neuron = Morphology(DATA_PATH / 'zero-diameter-end-of-neurite.swc')
    fix_zero_diameters(neuron)
    leaf = neuron.root_sections[0].children[0]
    assert_array_equal(leaf.diameters, np.array([2, 2, 2, 2, 2], dtype=np.float32))
