import os

from mock import patch
import numpy as np
from neurom import COLS, load_neuron
from neurom.geom import bounding_box
from nose.tools import assert_dict_equal, ok_
from numpy.testing import assert_array_almost_equal, assert_equal, assert_array_equal

import repair.main as test_module
from morphio import SectionType
from repair.main import Action
from repair.utils import read_apical_points
from .utils import setup_tempdir

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_read_apical_points():
    neuron = load_neuron(os.path.join(DATA_PATH, 'with-apical-attribute.h5'))
    filename = os.path.join(DATA_PATH, 'with-apical-attribute.h5')
    assert_array_equal(read_apical_points(filename, neuron).id, 1)

    filename = os.path.join(DATA_PATH, 'valid.h5')
    assert_array_equal(read_apical_points(filename, neuron), None)
