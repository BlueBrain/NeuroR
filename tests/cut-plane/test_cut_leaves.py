from pathlib import Path
import numpy as np

from morphio import Morphology
from nose.tools import assert_almost_equal, assert_equal, assert_not_equal, assert_raises, ok_
from numpy.testing import assert_array_almost_equal, assert_array_equal

import neuror.cut_plane.cut_leaves as test_module

DATA = Path(__file__).parent.parent / "data"


def test_find_cut_leaves():
    filename = DATA / "Neuron_slice.h5"
    neuron = Morphology(filename)
    leaves, qualities = test_module.find_cut_leaves(neuron, bin_width=10)
    expected_leaves = np.array(
        [
            [63.97578, 61.525646, 44.460205],
            [70.55578, 91.74565, 44.070206],
            [-99.83422, -2.2443538, 48.600204],
            [-141.51422, 20.185646, 48.680206],
            [-53.534218, 97.40565, 46.410206],
            [56.855785, -71.19435, 43.360207],
            [36.015785, 4.5756464, 44.460205],
            [34.875782, 4.8656464, 39.140205],
            [16.155783, -22.084354, 45.860207],
            [34.845783, 3.395646, 39.690205],
            [61.365784, -80.39435, 40.550205],
            [85.11578, -43.264355, 44.380207],
            [39.885784, -15.054354, 45.240204],
            [88.63578, 11.385646, 45.080204],
            [132.03578, 48.625645, 40.160206],
            [-14.654218, -9.674354, 47.270206],
            [-30.674217, -16.844355, 45.710205],
            [-35.614216, -15.954353, 46.570206],
            [-24.964218, -0.52435374, 46.640205],
            [-16.084217, 19.265646, 46.490204],
            [-6.4742174, 13.075646, 46.180206],
            [-7.8942175, 29.275646, 45.390205],
            [28.885782, 36.645645, 42.660206],
            [27.375782, 49.955647, 45.860207],
            [-3.6142175, 44.925648, 46.020206],
            [-65.55422, 55.615646, 39.140205],
            [37.635784, 43.825645, 43.990204],
            [48.415783, 65.95565, 42.500206],
            [21.215782, 49.985645, 44.140205],
            [35.525784, 70.56565, 42.890205],
            [5.3857822, 61.325645, 44.930206],
        ],
        dtype=float,
    )
    assert_array_almost_equal(np.array(leaves, dtype=float), expected_leaves, decimal=5)
    assert_equal(qualities,  [{'axis': 'Z', 'side': -1, 'quality': 25.0}])
