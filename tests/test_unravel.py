import os

from mock import patch
import numpy as np
from neurom import COLS, load_neuron
from neurom.geom import bounding_box
from nose.tools import assert_dict_equal, ok_
from numpy.testing import assert_array_almost_equal, assert_equal, assert_array_equal

import repair.main as test_module
from morphio import SectionType, Morphology

from utils import setup_tempdir

import repair.unravel as test_module
PATH = os.path.dirname(__file__)

SIMPLE = load_neuron(os.path.join(PATH, 'data', 'simple.swc'))

def test_get_principal_direction():
    assert_array_almost_equal(test_module._get_principal_direction([[0.,0,0], [1,1,2]]),
                              np.array([-0.408248, -0.408248, -0.816497]))

    assert_array_almost_equal(test_module._get_principal_direction([[0., 0, 0],
                                                                    [10, -1, 0],
                                                                    [10, 1, 0]]),
                              np.array([1, 0, 0]))

def test_unravel():
    neuron = test_module.unravel(os.path.join(PATH, 'data', 'simple.asc'))
    assert_array_almost_equal(neuron.root_sections[0].points,
                              np.array([[ 0.      ,  0.      ,  0.      ],
                                        [ 1.404784, -0.163042,  0.      ],
                                        [ 2.809567, -0.326085,  0.      ],
                                        [ 3.8029  , -0.441373,  0.      ]]))

    assert_array_almost_equal(neuron.root_sections[0].children[0].points[0],
                              np.array([ 3.8029  , -0.441373,  0.      ]))
