import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from neurom import COLS, NeuriteType, load_neuron
from nose.tools import assert_dict_equal, assert_raises, ok_
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)

import neuror.main as test_module
from mock import patch
from neuror.main import Action, Repair, RepairType
from morphio import SectionType

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

    obj = Repair(SIMPLE_PATH)
    obj.cut_leaves = [[0, 0, 0]]
    obj._fill_repair_type_map()
    assert_raises(Exception, obj._find_intact_sub_trees)


    obj = Repair(SLICE_PATH,
                 plane=test_module.CutPlane.find(SLICE, bin_width=15))
    obj._fill_repair_type_map()
    intact_sub_trees = obj._find_intact_sub_trees()
    assert_array_equal([section.id for section in intact_sub_trees],
                       [0, 34])

    obj = Repair(DATA_PATH / 'test-cut-apical' / 'simple-apical-cut.swc',
                 plane=str(DATA_PATH / 'test-cut-apical' / 'cut-plane.json'))
    obj._fill_repair_type_map()
    intact_sub_trees = obj._find_intact_sub_trees()
    assert_array_equal([section.id for section in intact_sub_trees],
                       [1, 0, 3])
    assert_array_equal([obj.repair_type_map[section] for section in intact_sub_trees],
                       [RepairType.basal, RepairType.axon, RepairType.tuft])


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
    test_module._continuation(DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]]), [0, 0, 0])
    test_module._continuation(DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]]), [1, 1, 1])


def test_get_similar_child_diameters():
    children = [DummySection([[1.0, 1, 1, 10]]),
                DummySection([[1.0, 1, 1, 12]])]
    sections = [DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]], children=children)]

    similar_section = DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]], children=[])
    assert_array_equal(test_module._get_similar_child_diameters(sections, similar_section),
                       [20, 24])

    different_section = DummySection([[1.0, 1, 1, 42]],
                                     children=[DummySection([[1.0, 1, 1, 1]]),
                                               DummySection([[1.0, 1, 1, 2]])])
    assert_array_equal(test_module._get_similar_child_diameters(sections, different_section),
                       [84, 84])


def test_get_origin():
    obj = Repair(SIMPLE)
    section = obj.neuron.section(1)
    obj.repair_type_map = {section: RepairType.basal}
    assert_equal(obj._get_origin(section), [0, 0, 0])

    obj.repair_type_map = {section: RepairType.oblique}
    assert_equal(obj._get_origin(section), [0, 5, 0])

def test_get_order_offset():
    obj = Repair(SIMPLE_PATH)
    section = obj.neuron.neurites[0].root_node.children[0]
    obj.repair_type_map = {section: RepairType.basal}
    assert_equal(obj._get_order_offset(section), 0)

    obj.repair_type_map = {section: RepairType.oblique}
    assert_equal(obj._get_order_offset(section), 1)


def test__get_sholl_layer():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module._get_sholl_layer(section, SIMPLE.soma.center), 0)


def test_last_segment_vector():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module._last_segment_vector(section, True),
                 [0, 1, 0])

    assert_equal(test_module._last_segment_vector(section, False),
                 [0, 5, 0])


def test__grow_until_sholl_sphere():
    np.random.seed(0)
    neuron = load_neuron(DATA_PATH / 'simple.swc')
    section = neuron.neurites[0].root_node
    test_module._grow_until_sholl_sphere(section, SIMPLE.soma.center, 0)
    assert_array_almost_equal(section.points[:, COLS.XYZ],
                              np.array([[0., 0., 0.],
                                        [0., 5., 0.],
                                        [0.170201, 9.98424, 0.358311],
                                        [0.469826, 14.901759, 1.211672]], dtype=np.float32))


def test__compute_sholl_data():
    obj = Repair(SIMPLE_PATH)
    obj._fill_repair_type_map()
    branches = [neurite.root_node for neurite in obj.neuron.neurites]
    sholl_data = obj._compute_sholl_data(branches)
    assert_dict_equal(sholl_data[RepairType.basal][0][0],
                      {Action.TERMINATION: 0, Action.CONTINUATION: 1, Action.BIFURCATION: 1})
    assert_dict_equal(sholl_data[RepairType.basal][0][1],
                      {Action.TERMINATION: 2, Action.CONTINUATION: 2, Action.BIFURCATION: 0})
    assert_dict_equal(sholl_data[RepairType.oblique], {})
    assert_dict_equal(sholl_data[RepairType.axon],
                      {0: {0: {Action.BIFURCATION: 1,
                               Action.CONTINUATION: 1,
                               Action.TERMINATION: 0},
                           1: {Action.BIFURCATION: 0,
                               Action.CONTINUATION: 2,
                               Action.TERMINATION: 2}}})


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
    obj = Repair(input_file,
                 plane=test_module.CutPlane.find(input_file, bin_width=15))
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

    obj._grow(leaf, 0, obj.neuron.soma.center)
    assert_equal(len(obj.neuron.sections), 8)

def test_repair_axon():
    filename = DATA_PATH / 'real-with-axon.asc'
    with TemporaryDirectory('test-cli-axon') as tmp_folder:
        outfilename = Path(tmp_folder, 'out.asc')
        test_module.repair(filename, outfilename, axons=[filename])
        neuron_in = load_neuron(filename)
        neuron_out = load_neuron(outfilename)
        axon = neuron_out.section(40)
        ok_(axon.type == NeuriteType.axon)
        assert_array_equal(neuron_in.section(40).points[0],
                           neuron_out.section(40).points[0])
        ok_(len(neuron_out.section(40).points) > len(neuron_in.section(40).points))
