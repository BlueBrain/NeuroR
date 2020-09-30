from pathlib import Path

from neurom import load_neuron
from nose.tools import assert_raises
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
from morph_tool.spatial import point_to_section_segment

import neuror.cut_plane.legacy_detection as test_module

DATA_PATH = Path(__file__).parent / 'data'

SIMPLE = load_neuron(DATA_PATH / 'simple.swc')


def test_legacy():
    points, sign = test_module.cut_detect(SIMPLE, 'y')
    assert_array_equal(points,
                       [[ 6., -4., 0.],
                        [-5., -4., 0.]])
    assert_equal(sign, -1)

    assert_raises(Exception, test_module.cut_detect, SIMPLE, 'z')

def test_legacy_compare_with_legacy_result():
    '''Comparing results with the old repair launch with the following commands:

    repair --dounravel 0 --inputdir /gpfs/bbp.cscs.ch/project/proj83/home/bcoste/release/out-new/01_ConvertMorphologies --input rp120430_P-2_idA --overlap=true --incremental=false --restrict=true --distmethod=mirror

    The arguments are the one used in the legacy morphology workflow.
    '''
    neuron = load_neuron(DATA_PATH / 'compare-bbpsdk/rp120430_P-2_idA.h5')
    points, sign = test_module.cut_detect(neuron, 'z')
    assert_equal(sign, 1)
    cut_sections = [point_to_section_segment(neuron, point)[0]
                    for point in points]

    legacy_cut_sections = [13,14,17,18,38,39,40,45,58,67,68,69,73,75,76,93,94,101,102,103,105,106,109,110,111,120,124,125,148,149,150,156,157,158,162,163,164,166,167,168,169,192,202,203,205,206,208]
    assert_array_equal(cut_sections, legacy_cut_sections)
