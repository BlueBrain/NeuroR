import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from mock import patch
from morph_tool.spatial import point_to_section_segment
from morphio import SectionType
from neurom import COLS, NeuriteType, load_neuron
from nose.tools import assert_dict_equal, assert_equal, ok_
from numpy.testing import assert_array_almost_equal, assert_array_equal

import neuror.main as test_module
from neuror.main import Action, Repair, _PARAMS
from neuror.utils import RepairType

from .expected_sholl_stats import SHOLL_STATS

DATA_PATH = Path(__file__).parent / 'data'

SIMPLE_PATH = DATA_PATH / 'simple.swc'
SLICE_PATH = DATA_PATH / 'neuron-slice.h5'
SIMPLE = load_neuron(SIMPLE_PATH)
SLICE = load_neuron(SLICE_PATH)


class DummySection:
    def __init__(self, points, children=None):
        self.points = np.array(points)
        self.children = children or []


def test_is_cut_section():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module.is_cut_section(section, np.array([[2, 2, 2]])),
                 False)

    assert_equal(test_module.is_cut_section(section, np.array([[0, 0, 0]])),
                 True)


def test_is_branch_intact():
    neurite = SIMPLE.neurites[0]
    assert_equal(test_module.is_branch_intact(neurite.root_node, np.array([[2, 2, 2]])),
                 True)

    assert_equal(test_module.is_branch_intact(neurite.root_node, np.array([[0, 0, 0]])),
                 False)


def test__find_intact_sub_trees():
    obj = Repair(SIMPLE_PATH)
    obj.cut_leaves = np.array([[2, 2, 2]])
    obj._fill_repair_type_map()

    assert_equal(len(obj._find_intact_sub_trees()), 2)

    points = test_module.CutPlane.find(SLICE, bin_width=15).cut_leaves_coordinates
    obj = Repair(SLICE_PATH,cut_leaves_coordinates=points)
    obj._fill_repair_type_map()
    intact_sub_trees = obj._find_intact_sub_trees()
    assert_array_equal([section.id for section in intact_sub_trees],
                       [0, 34])

    obj = Repair(DATA_PATH / 'test-cut-apical' / 'simple-apical-cut.swc',
                 cut_leaves_coordinates=[[1.0, 30.0, 0.0]])
    obj._fill_repair_type_map()
    intact_sub_trees = obj._find_intact_sub_trees()
    assert_array_equal([section.id for section in intact_sub_trees],
                       [1, 0, 3])
    assert_array_equal([obj.repair_type_map[section] for section in intact_sub_trees],
                       [RepairType.basal, RepairType.axon, RepairType.tuft])

    filename = DATA_PATH / 'no-intact-basals.h5'
    points = test_module.CutPlane.find(filename, bin_width=15).cut_leaves_coordinates
    obj = Repair(filename, cut_leaves_coordinates=points)
    obj._fill_repair_type_map()
    intact_sub_trees = obj._find_intact_sub_trees()
    basals = [section for section in intact_sub_trees
              if section.type == NeuriteType.basal_dendrite]
    assert_equal(len(basals), 78)


def test_section_length():
    assert_equal(test_module.section_length(SIMPLE.neurites[0].root_node), 5)


def test_branching_angles():
    assert_equal(test_module._branching_angles(SIMPLE.neurites[0].root_node),
                 [(0, 1.5707963267948966), (0, 1.5707963267948966)])

    # Test skip too short sections
    assert_equal(test_module._branching_angles(DummySection([[0, 0, 0], [0, 0, 1e-9]])),
                 [])

    # Test skip too short child sections
    tiny_child = DummySection([[0, 0, 0], [0, 0, 1e-9]])
    parent = DummySection([[0, 0, 0], [0, 0, 1]], children=[tiny_child])
    with patch('neuror.main.branch_order'):
        assert_equal(test_module._branching_angles(parent),
                     [])


def test_best_case_angle_data():
    obj = Repair(SIMPLE_PATH)
    obj.info = {'intact_branching_angles': {SectionType.axon: {
        0: [1, 2, 3],
        1: [2, 2],
        2: []
    }}}

    # Use exact data for this specific branching angle
    assert_array_equal(obj._best_case_angle_data(SectionType.axon, 0),
                       [1, 2, 3])

    # No info for this branching angle, use aggregate data
    assert_array_equal(obj._best_case_angle_data(SectionType.axon, 2),
                       [1, 2, 3, 2, 2])


def test_intact_branching_angles():
    obj = Repair(SIMPLE_PATH)
    obj._fill_repair_type_map()
    branches = [neurite.root_node for neurite in obj.neuron.neurites]
    angles = obj._intact_branching_angles(branches)
    assert_array_almost_equal(angles[test_module.RepairType.basal][0],
                              [1.5707963267948966, 1.5707963267948966])


def test__get_sholl_proba():
    sholl_data = {RepairType.axon: {0: {1: {Action.TERMINATION: 2,
                                             Action.BIFURCATION: 2,
                                             Action.CONTINUATION: 4
                                             }}}}

    assert_dict_equal(test_module._get_sholl_proba(sholl_data, RepairType.axon, 0, 1),
                      {Action.TERMINATION: 0.25,
                       Action.BIFURCATION: 0.25,
                       Action.CONTINUATION: 0.5})

    # No info for pseudo_order == 2, re-use data from pseudo_order == 1
    assert_dict_equal(test_module._get_sholl_proba(sholl_data, RepairType.axon, 0, 2),
                      {Action.TERMINATION: 0.25,
                       Action.BIFURCATION: 0.25,
                       Action.CONTINUATION: 0.5})

    # No info for sholl_layer == 1, use default value
    assert_dict_equal(test_module._get_sholl_proba(sholl_data, RepairType.axon, 1, 2),
                      {Action.TERMINATION: 1,
                       Action.BIFURCATION: 0,
                       Action.CONTINUATION: 0})

    # No data at all, use default value
    assert_dict_equal(test_module._get_sholl_proba({RepairType.axon: {1: {}}}, RepairType.axon, 1, 2),
                      {Action.TERMINATION: 1,
                       Action.BIFURCATION: 0,
                       Action.CONTINUATION: 0})


def test_continuation():

    test_module._continuation(
        DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]]), [0, 0, 0], _PARAMS, 1., 0.1
    )
    test_module._continuation(
        DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]]), [1, 1, 1], _PARAMS, 1., 0.1
    )


def test_get_origin():
    obj = Repair(SIMPLE)
    section = obj.neuron.section(1)
    obj.repair_type_map = {section: RepairType.basal}
    assert_array_equal(obj._get_origin(section), [0, 0, 0])

    obj.repair_type_map = {section: RepairType.oblique}
    assert_array_equal(obj._get_origin(section), [0, 5, 0])

def test_get_order_offset():
    obj = Repair(SIMPLE_PATH)
    section = obj.neuron.neurites[0].root_node.children[0]
    obj.repair_type_map = {section: RepairType.basal}
    assert_equal(obj._get_order_offset(section), 0)

    obj.repair_type_map = {section: RepairType.oblique}
    assert_equal(obj._get_order_offset(section), 1)


def test__get_sholl_layer():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module._get_sholl_layer(
        section, SIMPLE.soma.center, _PARAMS['sholl_layer_size']), 0)


def test_last_segment_vector():
    section = SIMPLE.neurites[0].root_node
    assert_array_equal(test_module._last_segment_vector(section, True),
                       [0, 1, 0])

    assert_array_equal(test_module._last_segment_vector(section, False),
                       [0, 5, 0])


def test__grow_until_sholl_sphere():
    np.random.seed(0)
    neuron = load_neuron(DATA_PATH / 'simple.swc')
    section = neuron.neurites[0].root_node
    test_module._grow_until_sholl_sphere(section, SIMPLE.soma.center, 0, _PARAMS,
                                         lambda diam:diam, 0.1, 1.0)
    assert_array_almost_equal(section.points[:, COLS.XYZ],
                              np.array([[ 0.      ,  0.      ,  0.      ],
                                        [ 0.      ,  5.      ,  0.      ],
                                        [ 0.302276,  9.950119,  0.636358],
                                        [ 1.172574, 14.19536 ,  3.130453]], dtype=np.float32))


def test__compute_sholl_data():
    obj = Repair(SIMPLE_PATH)
    obj._fill_repair_type_map()
    branches = [neurite.root_node for neurite in obj.neuron.neurites]
    sholl_data = obj._compute_sholl_data(branches)
    assert_dict_equal(sholl_data[RepairType.basal][0][0],
                      {Action.TERMINATION: 0, Action.CONTINUATION: 0, Action.BIFURCATION: 1})
    assert_dict_equal(sholl_data[RepairType.basal][0][1],
                      {Action.TERMINATION: 2, Action.CONTINUATION: 0, Action.BIFURCATION: 0})
    assert_dict_equal(sholl_data[RepairType.oblique], {})
    assert_dict_equal(sholl_data[RepairType.axon],
                      {0: {0: {Action.BIFURCATION: 1,
                               Action.CONTINUATION: 0,
                               Action.TERMINATION: 0},
                           1: {Action.BIFURCATION: 0,
                               Action.CONTINUATION: 0,
                               Action.TERMINATION: 2}}})


def test__external_apical_point():
    obj = Repair(DATA_PATH / 'repair_type.asc', apical_point=[1.0, 0.0, 0.0])
    assert_array_almost_equal(obj.apical_section.points,
                              np.array([[0., 0., 4., 1.],
                                        [0., 0., 0., 0.5],
                                        [1., 0., 0., 0.5]], dtype=np.float32))


def test__fill_repair_type_map():
    obj = Repair(DATA_PATH / 'repair_type.asc')
    obj.apical_section = obj.neuron.sections[3]
    # Pre check to be sure that the apical point is the one it should be
    assert_array_almost_equal(obj.apical_section.points,
                              np.array([[0., 0., 4., 1.],
                                        [0., 0., 0., 0.5],
                                        [1., 0., 0., 0.5]], dtype=np.float32))

    obj._fill_repair_type_map()
    assert_dict_equal({sec.id: v for sec, v in obj.repair_type_map.items()},
                      {
                          0: test_module.RepairType.axon,
                          1: test_module.RepairType.trunk,
                          2: test_module.RepairType.oblique,
                          3: test_module.RepairType.trunk,
                          4: test_module.RepairType.tuft,
                          5: test_module.RepairType.tuft,
                          6: test_module.RepairType.basal,
    })


def json_compatible_dict(dict_):
    '''Remap the dict keys so that it can be saved to JSON'''
    if not isinstance(dict_, dict):
        return dict_

    result = dict()
    for k, v in dict_.items():
        if isinstance(k, (SectionType, Action, int)):
            k = str(k)
        result[k] = json_compatible_dict(v)

    return result

def test__compute_statistics_for_intact_subtrees():
    input_file = DATA_PATH / 'neuron-slice.h5'
    points = test_module.CutPlane.find(input_file, bin_width=15).cut_leaves_coordinates
    obj = Repair(input_file, cut_leaves_coordinates=points)
    obj._fill_repair_type_map()
    obj._fill_statistics_for_intact_subtrees()

    with open(DATA_PATH / 'neuron-slice-sholl-data.json') as f:
        expected = json.load(f)

    basal_data = expected['SectionType.basal_dendrite']

    actual = json_compatible_dict(obj.info['sholl'][RepairType.basal])
    for layer in basal_data:
        for order in basal_data[layer]:
            for action in basal_data[layer][order]:
                assert_equal(basal_data[layer][order][action],
                             actual[layer][order][action])

def test__grow():
    obj = Repair(SIMPLE_PATH, seed=1)
    leaf = obj.neuron.sections[2]
    obj.repair_type_map = {section: RepairType.basal for section in obj.neuron.sections}
    obj.info = {
        'sholl': {RepairType.basal: {0: {1: {Action.BIFURCATION: 0.5, Action.TERMINATION: 0, Action.CONTINUATION: 0.5}}}},
        'dendritic_sections': obj.neuron.sections,
        'intact_branching_angles': {RepairType.basal: {1: [0.2]}}
    }
    obj.current_trunk_radius = 1.0
    obj._grow(leaf, 0, obj.neuron.soma.center)
    assert_equal(len(obj.neuron.sections), 8)


def test_repair_axon():
    filename = DATA_PATH / 'real-with-axon.asc'
    with TemporaryDirectory('test-cli-axon') as tmp_folder:
        outfilename = Path(tmp_folder, 'out.asc')
        test_module.repair(filename, outfilename, axons=[filename])
        neuron_in = load_neuron(filename)
        neuron_out = load_neuron(outfilename)
        axon = neuron_out.section(41)
        ok_(axon.type == NeuriteType.axon)
        assert_array_equal(neuron_in.section(40).points[0],
                           neuron_out.section(40).points[0])
        ok_(len(neuron_out.section(41).points) > len(neuron_in.section(41).points))

        # Test disactivating the axon repair
        repair_flags = {RepairType.axon: False}
        test_module.repair(filename, outfilename, axons=[filename], repair_flags=repair_flags)
        neuron_out = load_neuron(outfilename)
        axon = neuron_out.section(41)
        ok_(axon.type == NeuriteType.axon)
        assert_array_equal(neuron_in.section(41).points[0],
                           neuron_out.section(41).points[0])

        ok_(len(neuron_out.section(41).points) == len(neuron_in.section(41).points),
            'The section should not have been regrown')


def test_repair_no_intact_axon():
    filename = DATA_PATH / 'no-intact-basals.h5'
    with TemporaryDirectory('test-cli-axon') as tmp_folder:
        outfilename = Path(tmp_folder, 'out.asc')
        test_module.repair(filename, outfilename, axons=[filename])


def test_repair_no_trunk():
    '''Test repair when the morph has oblique sections but no trunk'''
    filename = DATA_PATH / 'Fluo42_right.h5'
    with TemporaryDirectory('test-no-trunk') as tmp_folder:
        outfilename = Path(tmp_folder, 'out.asc')
        test_module.repair(filename, outfilename, legacy_detection=True)


def test_legacy_compare_with_legacy_result():
    '''Comparing results with the old repair launch with the following commands:

    repair --dounravel 0 --inputdir /gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_old_code-2020-07-27/output/04_ZeroDiameterFix --input rp120430_P-2_idA --overlap=true --incremental=false --restrict=true --distmethod=mirror

    The arguments are the one used in the legacy morphology workflow.
    '''
    neuron = load_neuron(DATA_PATH / 'compare-bbpsdk/rp120430_P-2_idA.h5')
    obj = test_module.Repair(inputfile=DATA_PATH / 'compare-bbpsdk/rp120430_P-2_idA.h5', legacy_detection=True)

    cut_sections = {point_to_section_segment(neuron, point)[0]
                    for point in obj.cut_leaves}

    legacy_cut_sections = {
        13,14,17,18,38,39,40,45,58,67,68,69,73,75,76,93,94,101,102,103,105,106,
        109,110,111,120,124,125,148,149,150,156,157,158,162,163,164,166,167,168,
        169,192,201,202,203,205,206,208
    }
    assert_equal(cut_sections, legacy_cut_sections)

    obj._fill_repair_type_map()

    types = defaultdict(list)
    for k, v in obj.repair_type_map.items():
        types[v].append(k)


    # offset due to the first section id in the old soft being the soma
    offset = 1

    # These numbers come from the attribute 'apical' from the h5py group 'neuron1'
    section_id, segment_id = 134, 8


    assert_equal(obj.apical_section.id + offset, section_id)

    assert_equal(len(obj.apical_section.points) - 1, segment_id)

    assert_array_equal([section.id + offset for section in types[RepairType.basal]],
                       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                        107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                        122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132])

    assert_array_equal([0] + [section.id + offset for section in types[RepairType.axon]],
                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                        78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89])
    assert_array_equal([section.id + offset for section in types[RepairType.oblique]],
                       [217, 218, 219])
    assert_array_equal([section.id + offset for section in types[RepairType.trunk]],
                       [133, 134])

    expected_tufts = {135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                      150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                      165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                      180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
                      195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                      210, 211, 212, 213, 214, 215, 216}
    actual_tufts = {section.id + offset for section in types[RepairType.tuft]}
    assert_equal(actual_tufts, expected_tufts)


    expected_axons = {1, 2, 77, 3, 70, 4, 59, 5, 46, 6, 41, 7, 40, 8, 39, 9, 38, 10, 19, 11, 16, 12,
                      15, 13, 14, 17, 18, 20, 37, 21, 34, 22, 31, 23, 28, 24, 27, 25, 26, 29, 30,
                      32, 33, 35, 36, 42, 45, 43, 44, 47, 58, 48, 53, 49, 52, 50, 51, 54, 55, 56,
                      57, 60, 69, 61, 66, 62, 63, 64, 65, 67, 68, 71, 76, 72, 75, 73, 74, 78, 83,
                      79, 82, 80, 81, 84, 85, 86, 89, 87, 88}
    actual_axons = {section.id + offset for section in types[RepairType.axon]}
    assert_equal(actual_axons, expected_axons)


    intacts = defaultdict(list)

    for sec in obj._find_intact_sub_trees():
        intacts[obj.repair_type_map[sec]].append(sec)

    assert_equal([sec.id + offset for sec in intacts[RepairType.trunk]],
                 [])
    assert_equal([sec.id + offset for sec in intacts[RepairType.oblique]],
                 [217])
    assert_equal({sec.id + offset for sec in intacts[RepairType.tuft]},
                 {135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 151, 152, 153,
                  154, 155, 159, 160, 161, 165, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                  180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196,
                  197, 198, 199, 200, 204, 207, 209, 210, 211, 212, 213, 214, 215, 216})


def test_legacy_sholl_data():
    '''Comparing results with the old repair launch with the following commands:

    repair --dounravel 0 --inputdir /gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_old_code-2020-07-27/output/04_ZeroDiameterFix --input rp120430_P-2_idA --overlap=true --incremental=false --restrict=true --distmethod=mirror

    The arguments are the one used in the legacy morphology workflow.
    '''
    neuron = load_neuron(DATA_PATH / 'compare-bbpsdk/rp120430_P-2_idA.h5')
    obj = test_module.Repair(inputfile=DATA_PATH / 'compare-bbpsdk/rp120430_P-2_idA.h5', legacy_detection=True)
    obj._fill_repair_type_map()
    obj._fill_statistics_for_intact_subtrees()

    key_mapping = {
        'basal': RepairType.basal,
        'oblique': RepairType.oblique,
        'tuft': RepairType.tuft
    }
    expected_stats = {key_mapping[k]: v for k,v in SHOLL_STATS.items()}

    flattened = {(i, j): [obj.info['sholl'][RepairType.oblique][i][j][action] for action in
                          [Action.BIFURCATION, Action.CONTINUATION, Action.TERMINATION]]
                 for i in obj.info['sholl'][RepairType.oblique].keys()
                 for j in obj.info['sholl'][RepairType.oblique][i].keys()}

    assert_dict_equal(flattened, expected_stats[RepairType.oblique])

def test_repair_only_axon():
    '''Test repairing a morph which has only an axon (but no dendrites) does not crash'''
    test_module.Repair(DATA_PATH / 'just-axon.asc')
    ok_(True)

def test_legacy_compare_with_legacy_result2():
    '''Comparing results with the old repair launch with the following commands:

    repair --dounravel 0 --inputdir /gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_old_code-2020-07-27/output/04_ZeroDiameterFix --input vd100714B_idB --overlap=true --incremental=false --restrict=true --distmethod=mirror

    The arguments are the one used in the legacy morphology workflow.
    '''
    neuron = load_neuron(DATA_PATH / 'compare-bbpsdk/vd100714B_idB.h5')
    obj = test_module.Repair(inputfile=DATA_PATH / 'compare-bbpsdk/vd100714B_idB.h5', legacy_detection=True)

    cut_sections = {point_to_section_segment(neuron, point)[0]
                    for point in obj.cut_leaves}

    legacy_cut_sections = {62,64,65,69,73,77,78,85,87,88,89,91,93,94,115,116,119,120,125,126,130,
                           133,136,137,138,140,142,144,145,147,150,151,152,159,165,171,172,175,177
                           ,179,180,182,184,188,191,200,202,204,205,207,208,209,211,215,217,218,219,
                           220,238,239,241,247,248,250,251,252,253,256,257,258,261,262,264,266,267,
                           283,288,289,290,291,293,294,295,316,318,320,322,324,326,328,330,331,337,
                           338,339,340,343,344,345,351,357,359,362,363,371,372,375,377,378,384,385,
                           386,387,388,390,391,394,416,426,427,429,430,431,438,439,440,441,453,466,
                           468,470,471,481,486,487,488,489,527,528,529,533,534,538,540,541,543,545,
                           548,549,551,572,573,574,576,577,581,583,584,588,595,596,598,599,602,607,
                           608,609,610,613,614,615,617,620,622,623,624,626,637,639,640,645,647,648,
                           649,650,653,654,665,666,667,670,677,678,679,680,689,691,693,694,703,716,
                           717,721,723,725,726}
    assert_equal(cut_sections, legacy_cut_sections)

    obj._fill_repair_type_map()

    types = defaultdict(list)
    for k, v in obj.repair_type_map.items():
        types[v].append(k)


    # offset due to the first section id in the old soft being the soma
    offset = 1

    assert_equal(obj.apical_section, None)

    assert_equal({section.id + offset for section in types[RepairType.basal]},
                 {650, 651, 668, 671, 702, 719, 652, 655, 653, 654, 656, 663, 657, 662, 658, 661,
                  659, 660, 664, 667, 665, 666, 669, 670, 672, 681, 673, 680, 674, 679, 675, 678,
                  676, 677, 682, 695, 683, 684, 685, 686, 687, 690, 688, 689, 691, 692, 693, 694,
                  696, 699, 697, 698, 700, 701, 703, 704, 705, 706, 707, 710, 708, 709, 711, 718,
                  712, 713, 714, 715, 716, 717, 720, 727, 721, 722, 723, 724, 725, 726, 728, 735,
                  729, 730, 731, 734, 732, 733, 736, 737, 738})

    assert_array_equal([section.id + offset for section in types[RepairType.oblique]],
                       [])
    assert_array_equal([section.id + offset for section in types[RepairType.trunk]],
                       [])

    assert_equal({section.id + offset for section in types[RepairType.tuft]},
                 set())

    intacts = defaultdict(list)

    for sec in obj._find_intact_sub_trees():
        intacts[obj.repair_type_map[sec]].append(sec)

    # Since there is no apical dendrite, all of those are empty
    for extended_type in [RepairType.trunk, RepairType.oblique, RepairType.tuft]:
        assert_equal(intacts[extended_type], [])

    assert_equal({sec.id + offset for sec in intacts[RepairType.basal]},
                 {651, 668, 671, 702, 719, 652, 655, 656, 663, 657, 662, 658, 661, 659, 660, 664,
                  669, 672, 681, 673, 674, 675, 676, 682, 695, 683, 684, 685, 686, 687, 690, 688,
                  692, 696, 699, 697, 698, 700, 701, 704, 705, 706, 707, 710, 708, 709, 711, 718,
                  712, 713, 714, 715, 720, 727, 722, 724, 728, 735, 729, 730, 731, 734, 732, 733,
                  736, 737, 738})
