from pathlib import Path

from morph_tool import apical_point_section_segment
from morph_tool.spatial import point_to_section_segment
from neurom import load_neuron
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_almost_equal, assert_array_equal

import neuror.cut_plane.legacy_detection as test_module
from neuror.utils import repair_type_map

DATA_PATH = Path(__file__).parent / 'data'

SIMPLE = load_neuron(DATA_PATH / 'simple.swc')

# OFFSET to go from BBPSDK section IDs to neurom ones
OFFSET = 1

def test_legacy():
    points, sign = test_module.internal_cut_detection(SIMPLE, 'y')
    assert_array_equal(points,
                       [[ 6., -4., 0.],
                        [-5., -4., 0.]])
    assert_equal(sign, -1)

    assert_raises(Exception, test_module.cut_detect, SIMPLE, 'z')

def test_get_obliques():
    '''Comparing results with the old repair launch with the following commands:

    repair --dounravel 0 --inputdir /gpfs/bbp.cscs.ch/project/proj83/home/bcoste/release/out-new/01_ConvertMorphologies --input rp120430_P-2_idA --overlap=true --incremental=false --restrict=true --distmethod=mirror

    The arguments are the one used in the legacy morphology workflow.
    '''
    neuron = load_neuron(DATA_PATH / 'compare-bbpsdk/rp120430_P-2_idA.h5')

    apical_section = neuron.sections[apical_point_section_segment(neuron)[0]]
    extended_types = repair_type_map(neuron, apical_section)
    assert_equal([sec.id + OFFSET for sec in test_module.get_obliques(neuron, extended_types)],
                 [217])


def test_children_ids():
    apical_section_id = 133
    neuron = load_neuron(DATA_PATH / 'compare-bbpsdk/rp120430_P-2_idA.h5')
    section = neuron.sections[apical_section_id]
    assert_equal([sec.id + OFFSET for sec in test_module.children_ids(section)],
                 [134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, ])

def test_legacy_compare_with_legacy_result():
    '''Comparing results with the old repair launch with the following commands:

    repair --dounravel 0 --inputdir /gpfs/bbp.cscs.ch/project/proj83/home/bcoste/release/out-new/01_ConvertMorphologies --input rp120430_P-2_idA --overlap=true --incremental=false --restrict=true --distmethod=mirror

    The arguments are the one used in the legacy morphology workflow.
    '''
    neuron = load_neuron(DATA_PATH / 'compare-bbpsdk/rp120430_P-2_idA.h5')
    points, sign = test_module.internal_cut_detection(neuron, 'z')
    assert_equal(sign, 1)
    cut_sections = {point_to_section_segment(neuron, point)[0]
                    for point in points}

    legacy_cut_sections = {13,14,17,18,38,39,40,45,58,67,68,69,73,75,76,93,94,101,102,103,105,106,109,110,111,120,124,125,148,149,150,156,157,158,162,163,164,166,167,168,169,192,201,202,203,205,206,208}
    assert_equal(cut_sections, legacy_cut_sections)
