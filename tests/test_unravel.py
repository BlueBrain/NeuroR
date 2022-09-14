from pathlib import Path

import numpy as np
import pandas as pd
from morph_tool import diff
from neurom import load_neuron
from numpy.testing import assert_array_almost_equal, assert_array_equal

import neuror.unravel as test_module
from neuror.cut_plane.detection import CutPlane

DATA = Path(__file__).parent / 'data'

SIMPLE = load_neuron(DATA / 'simple.swc')


def test_get_principal_direction():
    direction = test_module._get_principal_direction([[0.,0,0], [1,1,2]])

    # Trick since the PCA can return a vector in either of the 2 orientations
    if direction[0] < 0:
        direction *= -1
    assert_array_almost_equal(direction, np.array([0.408248, 0.408248, 0.816497]))

    assert_array_almost_equal(test_module._get_principal_direction([[0., 0, 0],
                                                                    [10, -1, 0],
                                                                    [10, 1, 0]]),
                              np.array([1, 0, 0]))


def test_unravel():
    neuron, mapping = test_module.unravel(DATA / 'simple.asc', window_half_length=0.1)
    assert_array_almost_equal(neuron.root_sections[0].points,
                              np.array([[ 0.      ,  0.      ,  0.      ],
                                        [1.      , 1.      , 0.      ],
                                        [2.      , 0.      , 0.      ],
                                        [3.      , 0.      , 0.      ]]))

    assert_array_almost_equal(neuron.root_sections[0].children[0].points[0],
                              np.array([3.0    , 0.    , 0.    ]))

    vals = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 4.0, 2.0],
        ]

    assert_array_almost_equal(mapping[["x0", "y0", "z0"]].values, vals)
    assert_array_almost_equal(mapping[["x1", "y1", "z1"]].values, vals)


def test_unravel_no_path_length():
    neuron, mapping = test_module.unravel(DATA / 'simple.asc', use_path_length=False)
    assert_array_almost_equal(neuron.root_sections[0].points,
                              np.array([[ 0.      ,  0.      ,  0.      ],
                                        [ 1.404784, -0.163042,  0.      ],
                                        [ 2.809567, -0.326085,  0.      ],
                                        [ 3.8029  , -0.441373,  0.      ]]))

    assert_array_almost_equal(neuron.root_sections[0].children[0].points[0],
                              np.array([ 3.8029  , -0.441373,  0.      ]))

    assert_array_almost_equal(mapping[['x0', 'y0', 'z0']].values,
                              [[0., 0., 0.],
                               [1., 1., 0.],
                               [2., 0., 0.],
                               [3., 0., 0.],
                               [3., 0., 0.],
                               [4., 1., 0.],
                               [3., 0., 0.],
                               [6., 4., 2.]])

    assert_array_almost_equal(mapping[['x1', 'y1', 'z1']].values,
                              [[0.        ,  0.        , 0.],
                               [1.40478373, -0.16304241, 0.],
                               [2.80956745, -0.32608482, 0.],
                               [3.8028996 , -0.44137323, 0.],
                               [3.8028996 , -0.44137323, 0.],
                               [4.80289936,  0.55862677, 0.],
                               [3.8028996 , -0.44137323, 0.],
                               [6.80289936,  3.55862665, 2.]])


def test_unravel_with_backward_segment():
    '''Test the fix to the  unravel issue
    which was not working when there was a segment going backward wrt to the window direction
    (direction from the window first to the window last point)
    '''
    neuron, mapping = test_module.unravel(DATA / 'simple-with-backward-segment.asc',
                                          window_half_length=0.5)
    assert_array_almost_equal(
        neuron.root_sections[0].points,
        np.array([
            [0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
            [9.9999994e-01,  9.9999994e-01,  0.0000000e+00],
            [0.0000000e+00,  1.9999999e+00,  0.0000000e+00],
            [2.0000000e+00,  1.9999999e+00,  0.0000000e+00],
            [2.0000000e+00, -1.1920929e-07,  0.0000000e+00],
            [3.0000000e+00, -1.1920929e-07,  0.0000000e+00]],
            dtype=np.float32,
        ),
    )


def test_unravel_plane():
    mapping = pd.read_csv(DATA / 'mapping.csv')
    input_plane = CutPlane.from_json(DATA / 'neuron-slice-plane.json')
    plane = test_module.unravel_plane(CutPlane.from_json(DATA / 'neuron-slice-plane.json'), mapping)
    assert_array_almost_equal(plane.cut_leaves_coordinates,
                              [[-111.24885559,   -1.29032707,   55.46524429],
                               [-156.59031677,   23.12454224,   53.51153946],
                               [ -60.68390274,  111.23972321,   54.0868721 ],
                               [  15.63226223,  -25.58386421,   51.45604706],
                               [ -14.81918049,  -10.20301151,   52.09701157],
                               [ -32.34645462,  -19.15251732,   50.921875  ],
                               [ -40.98423767,  -20.00574303,   51.760952  ],
                               [ -26.68880844,   -1.51575243,   51.23664093],
                               [ -18.93479538,   22.15104103,   50.69124985],
                               [  -7.5544219 ,   15.30504322,   51.06955719],
                               [  32.2554512 ,   56.86440277,   49.46508026],
                               [  -4.24387503,   47.21520996,   52.44573212]])

    input_plane.cut_leaves_coordinates = []
    assert_array_almost_equal(test_module.unravel_plane(input_plane, mapping),
                              [])

    input_plane.cut_leaves_coordinates = None
    assert_array_almost_equal(test_module.unravel_plane(input_plane, mapping),
                              [])


def test_unravel_all(tmpdir):
    tmpdir = Path(tmpdir)

    input = DATA / 'input-unravel-all'
    raw_planes = input / 'raw-planes'
    unravel_planes = tmpdir / 'unravelled-planes'
    unravel_planes.mkdir()

    test_module.unravel_all(input, tmpdir, raw_planes, unravel_planes, window_half_length=0.1)
    assert_array_equal(list(tmpdir.rglob('*.h5')), [tmpdir / 'Neuron_slice.h5'])


def test_legacy():
    actual, _ = test_module.unravel(
        DATA / "legacy-unravel/1-pt-soma.swc",
        legacy_behavior=True,
        use_path_length=False,
    )
    assert not diff(actual, DATA / "legacy-unravel/expected-1-pt-soma.h5")

    actual, _ = test_module.unravel(
        DATA / "legacy-unravel/3-pts-soma.swc",
        legacy_behavior=True,
        use_path_length=False,
                                    )
    assert not diff(actual, DATA / "legacy-unravel/expected-3-pts-soma.h5")
