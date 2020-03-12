from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from neurom import load_neuron
from numpy.testing import assert_array_almost_equal, assert_array_equal

import neuror.unravel as test_module

DATA = Path(__file__).parent / 'data'

SIMPLE = load_neuron(DATA / 'simple.swc')

def test_get_principal_direction():
    assert_array_almost_equal(test_module._get_principal_direction([[0.,0,0], [1,1,2]]),
                              np.array([-0.408248, -0.408248, -0.816497]))

    assert_array_almost_equal(test_module._get_principal_direction([[0., 0, 0],
                                                                    [10, -1, 0],
                                                                    [10, 1, 0]]),
                              np.array([1, 0, 0]))

def test_unravel():
    neuron, mapping = test_module.unravel(DATA / 'simple.asc')
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


def test_unravel_plane():
    with TemporaryDirectory('test-unravel-plane'):
        mapping = pd.read_csv(DATA / 'mapping.csv')
        plane = test_module.unravel_plane(str(DATA / 'neuron-slice-plane.json'), mapping)
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

def test_unravel_plane():
    with TemporaryDirectory('test-unravel-plane') as output:
        output = Path(output)

        input = DATA / 'input-unravel-all'
        raw_planes = input / 'raw-planes'
        unravel_planes = output / 'unravelled-planes'
        unravel_planes.mkdir()

        test_module.unravel_all(input, output, raw_planes, unravel_planes)
        assert_array_equal(list(output.rglob('*.h5')), [output / 'Neuron_slice.h5'])
