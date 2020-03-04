import os
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from neuror.cut_plane import CutPlane
from neuror.cut_plane.viewer import _get_displaced_pos
from neurom import COLS, iter_sections, load_neuron
from nose.tools import assert_not_equal, ok_
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_equal, assert_allclose)
from pyquaternion import Quaternion

matplotlib.use('Agg')

DATA = Path(__file__).parent.parent / 'data'


def test_cut_plane_from_rotations_translations():
    filename = DATA / 'Neuron_slice.h5'
    equation = CutPlane.from_rotations_translations(
        [0, 21, -21, 0, 0, 50], morphology=filename, bin_width=10)
    assert_array_almost_equal(equation.coefs,
                              [35.836795, 0., 93.358043, -4667.902132])


def test_cut_neuron_simple():
    filename = DATA / 'simple2.asc'
    result = CutPlane.find(filename, bin_width=0.2).to_json()
    ok_('The probability that there is in fact NO cut plane is high: -log(p)'
        in result['status'])
    assert_almost_equal(result['cut-plane']['a'], 0)
    assert_almost_equal(result['cut-plane']['b'], 0)
    assert_almost_equal(result['cut-plane']['c'], 1)
    assert_allclose(result['cut-plane']['d'], -2, rtol=0.2)


def test_cut_real_neuron():
    filename = DATA / 'Neuron_slice.h5'
    result = CutPlane.find(filename, bin_width=10).to_json()
    assert_equal(result['status'], 'ok')
    assert_almost_equal(result['cut-plane']['a'], 0)
    assert_almost_equal(result['cut-plane']['b'], 0)
    assert_almost_equal(result['cut-plane']['c'], 1)
    assert_almost_equal(result['cut-plane']['d'], -48.68020515427703,
                        decimal=5)
    assert_equal(result['cut-plane']['comment'],
                 'Equation: a*X + b*Y + c*Z + d < 0')

    leaves_coord = [[63.97577896,   61.52564626,   44.46020393],
                    [70.55578079,   91.74564748,   44.07020454],
                    [-99.8342186,   -2.24435372,   48.60020332],
                    [-141.51421891,   20.18564611,   48.68020515],
                    [-53.53421936,   97.40564351,   46.4102047],
                    [56.85578384,  -71.19435496,   43.36020546],
                    [36.01578369,    4.57564645,   44.46020393],
                    [34.87578048,    4.86564641,   39.14020424],
                    [16.15578308,  -22.08435435,   45.86020546],
                    [34.8457817,    3.39564615,   39.69020348],
                    [61.36578216,  -80.39435191,   40.55020409],
                    [85.11578598,  -43.26435465,   44.38020592],
                    [39.88578262,  -15.05435366,   45.24020271],
                    [88.63578262,   11.38564592,   45.08020287],
                    [132.03578415,   48.62564474,   40.1602047],
                    [-14.65421734,   -9.67435355,   47.27020531],
                    [-30.67421685,  -16.84435458,   45.71020393],
                    [-35.61421738,  -15.95435328,   46.57020454],
                    [-24.96421776,   -0.52435374,   46.64020424],
                    [-16.08421765,   19.26564603,   46.49020271],
                    [-6.47421751,   13.07564645,   46.18020515],
                    [-7.89421759,   29.27564626,   45.39020424],
                    [28.88578262,   36.64564519,   42.6602047],
                    [27.37578239,   49.95564657,   45.86020546],
                    [-3.61421762,   44.92564779,   46.02020531],
                    [-65.55421982,   55.61564641,   39.14020424],
                    [37.63578262,   43.8256455,   43.99020271],
                    [48.4157814,   65.95564657,   42.50020485],
                    [21.21578254,   49.98564535,   44.14020424],
                    [35.52578201,   70.56564718,   42.89020424],
                    [5.38578214,   61.3256455,   44.93020515]]
    assert_array_almost_equal(result['cut-leaves'], leaves_coord, decimal=5)

    plane = CutPlane.from_json(result, filename)
    assert_array_almost_equal([sec.points[-1, COLS.XYZ] for sec in plane.cut_sections],
                              leaves_coord,
                              decimal=5)


def test_repaired_neuron():
    result = CutPlane.find(DATA / 'bio_neuron-000.h5', bin_width=10).to_json()
    assert_not_equal(result['status'], 'ok')
