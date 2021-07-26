import sys
import json
from pathlib import Path
from packaging import  version

import neurom as nm
import numpy as np
import pytest
from mock import MagicMock
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
import scipy

import neuror.cut_plane.detection as test_module
from neuror.cut_plane import PlaneEquation
from neuror.cut_plane.detection import _minimize, _success_function

DATA = Path(__file__).parent.parent / 'data'


def _get_points():
    '''Utility function to get all neuron points'''
    neuron = nm.load_neuron(DATA / 'rotated.h5')
    return np.array([point
                     for neurite in neuron.neurites
                     for section in nm.iter_sections(neurite)
                     for point in section.points])


def test_project_normal():
    plane = PlaneEquation(0, 0, 1, 0)
    assert_array_equal(plane.project_on_normal(np.array([[0,0,2], [1,2,3]])),
                       [2, 3])

    plane = PlaneEquation(0, 1, 0, -9)
    assert_array_equal(plane.project_on_normal(np.array([[0,0,0], [1,2,3]])),
                       [-9, -7])


def test_projection():
    points = _get_points()
    plane = PlaneEquation.from_rotations_translations([4, 45, -21, 0, 0, 61])
    bin_width = 10
    n_left, n_right = plane.count_near_plane(points, bin_width)

    assert n_left == 1830
    assert n_right == 513

    projected = plane.project_on_normal(points)
    binning = [-160., -150., -140., -130., -120., -110., -100.,  -90.,  -80.,
     -70.,  -60.,  -50.,  -40.,  -30.,  -20.,  -10.,    0.,   10.]
    val, _ = np.histogram(projected, bins=binning)
    assert_array_equal(val,
                       [13,   65,  129,  247,  347,  581,  848, 1285, 1624, 2084, 1557,
                        1646, 1300, 1359, 1698, 1830,  513])


def test_success_function():
    rot_x, rot_y, rot_z = 4, 45, -21
    transl_x, transl_y, transl_z = 0, 0, 61
    bin_width = 10
    params = rot_x, rot_y, rot_z, transl_x, transl_y, transl_z
    res = _success_function(params, _get_points(), bin_width=bin_width)
    assert res == 513-1830


@pytest.mark.skipif(version.parse(scipy.__version__) >= version.parse('1.6'),
                    reason="requires scipy < 1.6")
def test_minimize():
    params = 4, 45, -21, 0, 0, 61

    result = _minimize(params, _get_points(), bin_width=10)
    assert_allclose(result,
                    [4.11038409e+00, 4.65181441e+01, -2.05934568e+01, -2.44713344e-04,
                     -2.71528635e-04, 6.88986409e+01])


@pytest.mark.skipif(version.parse(scipy.__version__) < version.parse('1.6'),
                    reason="requires scipy >= 1.6")
def test_minimize():
    params = 4, 45, -21, 0, 0, 61

    result = _minimize(params, _get_points(), bin_width=10)
    assert_allclose(result,
                    [4.1510e+00, 4.6763e+01, -2.0966e+01, -2.7207e-04, -4.5070e-04, 6.8804e+01],
                    rtol=1e-4)

def test__compute_probabilities():
    plane = test_module.CutPlane((1, 0, 0, 4), None, None, None)
    plane.histogram = MagicMock(return_value=[np.array([10, 2, 2])])
    plane._compute_probabilities()
    assert plane.minus_log_prob == 10.0
    assert (plane.status ==
                 'The probability that there is in fact NO cut plane is high: -log(p) = 10.0 !')

    plane.histogram = MagicMock(return_value=[np.array([])])
    plane._compute_probabilities()
    assert np.isnan(plane.minus_log_prob)
    assert plane.status == 'The proba is NaN, something went wrong'


def test_plane_equation():
    # test invalid plane
    with pytest.raises(ValueError):
        PlaneEquation(0, 0, 0, 4)

    equation = PlaneEquation.from_rotations_translations([0, 0, 2, 3, 4, 5])
    assert_array_equal(equation.coefs, [0, 0, 100, -500])
    assert str(equation) == '(0.0) * X + (0.0) * Y + (100.0) * Z + (-500.0) = 0'

    assert_array_equal(equation.distance([[2,3,4], [2,3,7]]),
                       [1, 2])

    equation = PlaneEquation.from_rotations_translations([45, 0, 0, 3, 4, 5])
    assert equation.coefs[0] == 0
    assert_almost_equal(equation.coefs[1], -equation.coefs[2])
    assert_almost_equal(equation.coefs[3], -70.7106781186547)

    equation = PlaneEquation.from_rotations_translations([0, 0, 0, 0, 0, 0])
    assert equation.coefs[0] == 0
    assert equation.coefs[1] == 0
    assert equation.coefs[2] != 0
    assert equation.coefs[3] == 0


def test_from_json():
    filename = DATA / 'plane.json'

    def _assert_expected_plane(plane):
        assert plane.bin_width == 8
        assert_array_equal(plane.coefs, [4, 7, 8, 1])
        assert_array_equal(plane.cut_leaves_coordinates,
                           [[10, 10, 10], [1, 0, 20]])
        assert plane.status == 'ok'
        assert plane.minus_log_prob == 70

    _assert_expected_plane(
        test_module.CutPlane.from_json(str(filename))
    )

    _assert_expected_plane(
        test_module.CutPlane.from_json(filename)
    )

    with filename.open() as file_:
        dictionnary = json.load(file_)
        _assert_expected_plane(
            test_module.CutPlane.from_json(dictionnary)
        )


def test_find():
    filename = DATA / 'rotated.h5'
    neuron = nm.load_neuron(filename)
    result = test_module.CutPlane.find(neuron, bin_width=10).to_json()
    assert (set(result.keys()) ==
                 {'details', 'cut-plane', 'cut-leaves', 'status'})

    assert result['status'] == 'ok'
