from pathlib import Path

import numpy as np
from morphio.mut import Morphology
from numpy.testing import assert_array_equal, assert_array_almost_equal
from morph_tool import diff

from neuror.zero_diameter_fixer import (_fix_downstream, _fix_upstream,
                                        _fix_in_between, fix_zero_diameters)

DATA_PATH = Path(__file__).parent / 'data'


def test_fix_in_between():
    neuron = Morphology(DATA_PATH / 'zero-diameter-middle-of-neurite.asc')
    root = neuron.root_sections[0]

    _fix_in_between(root, list(), False)
    assert_array_equal(root.diameters, np.array([2., 2., 2.2, 2.4, 2.8, 3.], dtype=np.float32))
    assert_array_equal(root.children[0].diameters, np.array([3, 3, 3, 2.8], dtype=np.float32))
    assert_array_equal(root.children[0].children[0].diameters,
                       np.array([2.8, 2.6, 2.4, 2.2, 2.], dtype=np.float32))


def test_fix_in_between_legacy():
    neuron = Morphology(DATA_PATH / 'zero-diameter-middle-of-neurite.asc')
    root = neuron.root_sections[0]

    _fix_in_between(root, list(), True)
    assert_array_equal(root.diameters,
                       np.array([2., 2., 2.25, 2.5, 2.75, 3.], dtype=np.float32))
    assert_array_almost_equal(root.children[0].diameters,
                              np.array([3., 3., 3., 2.833333], dtype=np.float32))
    assert_array_almost_equal(root.children[0].children[0].diameters,
                              np.array([2.666667, 2.5, 2.333333, 2.166667, 2.], dtype=np.float32))


def test_fix_from_downstream():
    neuron = Morphology(DATA_PATH / 'zero-diameter-middle-of-neurite.asc')
    root = neuron.root_sections[0]

    _fix_downstream(root)
    assert_array_equal(neuron.section(0).diameters,
                       np.array([2., 2., 1.e-6, 1.e-6, 1.e-6, 3.], dtype=np.float32),
                       "The diameters should not have been modified")

    neuron = Morphology(DATA_PATH / 'zero-diameter-beginning-of-neurite.swc')
    root = neuron.root_sections[0]

    assert_array_equal(root.diameters, [0., 0.])
    _fix_downstream(root)
    assert_array_equal(root.diameters, [0., 0.],
                       'Diameters should not have been changed '
                       'since the whole neurite has a null diameter')

    neuron = Morphology(DATA_PATH / 'zero-diameter-beginning-of-neurite.swc')
    root = neuron.root_sections[1]

    assert_array_equal(root.diameters, [0, 0, 4])
    _fix_downstream(root)
    assert_array_equal(root.diameters, [4, 4, 4])


def test_fix_from_upstream():
    neuron = Morphology(DATA_PATH / 'zero-diameter-end-of-neurite.swc')
    root = neuron.root_sections[0]
    leaf = root.children[0]

    _fix_upstream(root, 12)
    assert_array_equal(leaf.diameters, np.array([2, 2, 2, 2, 2], dtype=np.float32))


def test_fix_zero_diameters():
    neuron = Morphology(DATA_PATH / 'zero-diameter-end-of-neurite.swc')
    fix_zero_diameters(neuron)
    leaf = neuron.root_sections[0].children[0]
    assert_array_equal(leaf.diameters, np.array([2, 2, 2, 2, 2], dtype=np.float32))


def test_functional():
    functional_neuron = Morphology(DATA_PATH / 'compare-zero-diameter/original.h5')
    fix_zero_diameters(functional_neuron, legacy=True)
    expected = Morphology(DATA_PATH / 'compare-zero-diameter/cpp-fixed.h5')
    assert not diff(functional_neuron, expected)
