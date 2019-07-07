import json
import logging
import os
from os.path import join as joinp
import shutil

from mock import patch
import numpy as np
from neurom import COLS, load_neuron
from neurom.geom import bounding_box
from morphio import Morphology, diff
from nose.tools import assert_dict_equal, ok_, assert_raises
from numpy.testing import assert_array_almost_equal, assert_equal, assert_array_equal

import repair.main as test_module
from morphio import SectionType
from repair.main import Action
from .utils import setup_tempdir
from repair.main import RepairType, full

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

SIMPLE = load_neuron(os.path.join(DATA_PATH, 'simple.swc'))
SLICE = load_neuron(os.path.join(DATA_PATH, 'neuron-slice.h5'))
REPAIR_TYPE_MAP = test_module.subtree_classification(SIMPLE, None)


class DummySection:
    def __init__(self, points, children=None):
        self.points = np.array(points)
        self.children = children or []


def test_is_cut_section():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module.is_cut_section(section, [[2, 2, 2]]),
                 False)

    assert_equal(test_module.is_cut_section(section, [[0, 0, 0]]),
                 True)


def test_is_branch_intact():
    neurite = SIMPLE.neurites[0]
    assert_equal(test_module.is_branch_intact(neurite.root_node, [[2, 2, 2]]),
                 True)

    assert_equal(test_module.is_branch_intact(neurite.root_node, [[0, 0, 0]]),
                 False)


def test_find_intact_sub_trees():
    assert_equal(len(test_module.find_intact_sub_trees(SIMPLE, [[2, 2, 2]], REPAIR_TYPE_MAP)),
                 1)

    assert_raises(Exception, test_module.find_intact_sub_trees, SIMPLE, [[0, 0, 0]], REPAIR_TYPE_MAP)

    repair_type_map = test_module.subtree_classification(SLICE, apical_section=None)
    cut_plane = test_module.CutPlane.find(SLICE, bin_width=15)
    intact_sub_trees = test_module.find_intact_sub_trees(SLICE, cut_plane.cut_leaves_coordinates, repair_type_map)
    assert_array_equal([section.id for section in intact_sub_trees],
                       [0, 34])


def test_section_length():
    assert_equal(test_module._section_length(SIMPLE.neurites[0].root_node), 5)


def test_branching_angles():
    assert_equal(test_module.branching_angles(SIMPLE.neurites[0].root_node),
                 [(0, 1.5707963267948966), (0, 1.5707963267948966)])

    # Test skip too short sections
    assert_equal(test_module.branching_angles(DummySection([[0, 0, 0], [0, 0, 1e-9]])),
                 [])

    # Test skip too short child sections
    tiny_child = DummySection([[0, 0, 0], [0, 0, 1e-9]])
    parent = DummySection([[0, 0, 0], [0, 0, 1]], children=[tiny_child])
    with patch('repair.main.branch_order'):
        assert_equal(test_module.branching_angles(parent),
                     [])


def test_best_case_angle_data():
    info = {'intact_branching_angles': {SectionType.axon: {
        0: [1, 2, 3],
        1: [2, 2],
        2: []
    }}}

    # Use exact data for this specific branching angle
    assert_array_equal(test_module.best_case_angle_data(info, SectionType.axon, 0),
                       [1, 2, 3])

    # No info for this branching angle, use aggregate data
    assert_array_equal(test_module.best_case_angle_data(info, SectionType.axon, 2),
                       [1, 2, 3, 2, 2])


def test_intact_branching_angles():
    branches = [neurite.root_node for neurite in SIMPLE.neurites]
    angles = test_module.intact_branching_angles(branches, REPAIR_TYPE_MAP)
    assert_array_almost_equal(angles[test_module.RepairType.basal][0],
                              [1.5707963267948966, 1.5707963267948966])


def test_get_sholl_proba():
    sholl_data = {RepairType.axon: {0: {1: {Action.TERMINATION: 2,
                                             Action.BIFURCATION: 2,
                                             Action.CONTINUATION: 4
                                             }}}}

    assert_dict_equal(test_module.get_sholl_proba(sholl_data, RepairType.axon, 0, 1),
                      {Action.TERMINATION: 0.25,
                       Action.BIFURCATION: 0.25,
                       Action.CONTINUATION: 0.5})

    # No info for pseudo_order == 2, re-use data from pseudo_order == 1
    assert_dict_equal(test_module.get_sholl_proba(sholl_data, RepairType.axon, 0, 2),
                      {Action.TERMINATION: 0.25,
                       Action.BIFURCATION: 0.25,
                       Action.CONTINUATION: 0.5})

    # No info for sholl_layer == 1, use default value
    assert_dict_equal(test_module.get_sholl_proba(sholl_data, RepairType.axon, 1, 2),
                      {Action.TERMINATION: 1,
                       Action.BIFURCATION: 0,
                       Action.CONTINUATION: 0})

    # No data at all, use default value
    assert_dict_equal(test_module.get_sholl_proba({RepairType.axon: {1: {}}}, RepairType.axon, 1, 2),
                      {Action.TERMINATION: 1,
                       Action.BIFURCATION: 0,
                       Action.CONTINUATION: 0})


def test_continuation():
    test_module.continuation(DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]]), [0, 0, 0])
    test_module.continuation(DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]]), [1, 1, 1])


def test_get_similar_child_diameters():
    children = [DummySection([[1.0, 1, 1, 10]]),
                DummySection([[1.0, 1, 1, 12]])]
    sections = [DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]], children=children)]

    similar_section = DummySection([[1.0, 1, 1, 1], [1, 1, 2, 1]], children=[])
    assert_array_equal(test_module.get_similar_child_diameters(sections, similar_section),
                       [20, 24])

    different_section = DummySection([[1.0, 1, 1, 42]],
                                     children=[DummySection([[1.0, 1, 1, 1]]),
                                               DummySection([[1.0, 1, 1, 2]])])
    assert_array_equal(test_module.get_similar_child_diameters(sections, different_section),
                       [84, 84])


def test_get_origin():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module.get_origin(section, [1,1,1], {section: RepairType.basal}),
                 [1,1,1])

    assert_equal(test_module.get_origin(section, [1,1,1], {section: RepairType.oblique}),
                 [0,0,0])

def test_get_order_offset():
    section = SIMPLE.neurites[0].root_node.children[0]
    assert_equal(test_module.get_order_offset(section, {section: RepairType.basal}),
                 0)

    assert_equal(test_module.get_order_offset(section, {section: RepairType.oblique}),
                 1)


def test_get_sholl_layer():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module.get_sholl_layer(section, SIMPLE.soma.center), 0)


def test_last_segment_vector():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module.last_segment_vector(section, True),
                 [0, 1, 0])

    assert_equal(test_module.last_segment_vector(section, False),
                 [0, 5, 0])


def test_grow_until_sholl_sphere():
    np.random.seed(0)
    neuron = load_neuron(os.path.join(DATA_PATH, 'simple.swc'))
    section = neuron.neurites[0].root_node
    test_module.grow_until_sholl_sphere(section, SIMPLE.soma.center, 0)
    assert_array_almost_equal(section.points[:, COLS.XYZ],
                              np.array([[0., 0., 0.],
                                        [0., 5., 0.],
                                        [0.170201, 9.98424, 0.358311],
                                        [0.469826, 14.901759, 1.211672]], dtype=np.float32))


def test_compute_sholl_data():
    branches = [neurite.root_node for neurite in SIMPLE.neurites]
    sholl_data = test_module.compute_sholl_data(branches,
                                                soma_center=SIMPLE.soma.center, repair_type_map=REPAIR_TYPE_MAP)
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


def test_subtree_classification():
    neuron = load_neuron(os.path.join(DATA_PATH, 'repair_type.asc'))

    apical_section = neuron.sections[3]
    # Pre check to be sure that the apical point is the one it should be
    assert_array_almost_equal(apical_section.points,
                              np.array([[0., 0., 4., 1.],
                                        [0., 0., 0., 0.5],
                                        [1., 0., 0., 0.5]], dtype=np.float32))

    mapping = test_module.subtree_classification(neuron, apical_section)
    assert_dict_equal({sec.id: v for sec, v in mapping.items()},
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

def test_compute_statistics_for_intact_subtrees():
    input_file = os.path.join(DATA_PATH, 'neuron-slice.h5')
    cut_plane = test_module.CutPlane.find(input_file, bin_width=15)
    cut_leaves = cut_plane.cut_leaves_coordinates

    neuron = load_neuron(input_file)
    repair_type_map = test_module.subtree_classification(neuron, apical_section=None)
    info = test_module.compute_statistics_for_intact_subtrees(neuron, cut_leaves, repair_type_map)

    with open(os.path.join(DATA_PATH, 'neuron-slice-sholl-data.json')) as f:
        expected = json.load(f)

    basal_data = expected['SectionType.basal_dendrite']

    actual = json_compatible_dict(info['sholl'][RepairType.basal])
    for layer in basal_data:
        for order in basal_data[layer]:
            for action in basal_data[layer][order]:
                assert_equal(basal_data[layer][order][action],
                             actual[layer][order][action])

def test_grow():
    np.random.seed(1)
    leaf = SIMPLE.sections[2]
    info = {
        'sholl': {RepairType.basal: {0: {1: {Action.BIFURCATION: 0.5, Action.TERMINATION: 0, Action.CONTINUATION: 0.5}}}},
        'dendritic_sections': SIMPLE.sections,
        'intact_branching_angles': {RepairType.basal: {1: [0.2]}}
    }
    repair_type_map = {section: RepairType.basal for section in SIMPLE.sections}
    test_module.grow(leaf, info, 0, SIMPLE.soma.center, repair_type_map)
    assert_equal(len(SIMPLE.sections), 8)

def assert_output_exists(root_dir,
                         raw_dir=None,
                         raw_planes_dir=None,
                         unravelled_dir=None,
                         unravelled_planes_dir=None,
                         repaired_dir=None):
    raw_dir = raw_dir or joinp(root_dir, 'raw')
    raw_planes_dir = raw_planes_dir or joinp(raw_dir, 'planes')
    unravelled_dir = unravelled_dir or joinp(root_dir, 'unravelled')
    unravelled_planes_dir = unravelled_planes_dir or joinp(unravelled_dir, 'planes')
    repaired_dir = repaired_dir or joinp(root_dir, 'repaired')

    for folder in [raw_dir, raw_planes_dir, unravelled_dir, unravelled_planes_dir, repaired_dir]:
        ok_(os.path.exists(folder), '{} does not exists'.format(folder))
        ok_(os.listdir(folder), '{} is empty !'.format(folder))


# Patch this to speed up test
@patch('repair.main.plot_repaired_neuron')
@patch('repair.main.view_all')
def test_full(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)
        full(test_folder)
        assert_output_exists(test_folder)


# Patch this to speed up test
@patch('repair.main.plot_repaired_neuron')
@patch('repair.main.view_all')
def test_full_custom_raw_dir(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        raw_dir_custom_path = joinp(test_folder, 'raw_custom')
        # Should raise because raw_custom dir does not exist
        assert_raises(Exception, full, test_folder, raw_dir=raw_dir_custom_path)

        shutil.move(joinp(test_folder, 'raw'), raw_dir_custom_path)
        full(test_folder, raw_dir=raw_dir_custom_path)
        assert_output_exists(test_folder, raw_dir=raw_dir_custom_path)


# Patching this to speed up test
@patch('repair.main.plot_repaired_neuron')
@patch('repair.main.view_all')
def test_full_custom_unravel_dir(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        custom_path = joinp(test_folder, 'unravel_custom')
        full(test_folder, unravelled_dir=custom_path)
        assert_output_exists(test_folder, unravelled_dir=custom_path)


# Patching this to speed up test
@patch('repair.main.plot_repaired_neuron')
@patch('repair.main.view_all')
def test_full_custom_unravelled_planes_dir(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        custom_path = joinp(test_folder, 'unravelled_planes_custom')
        full(test_folder, unravelled_planes_dir=custom_path)
        assert_output_exists(test_folder, unravelled_planes_dir=custom_path)


# Patching this to speed up test
@patch('repair.main.plot_repaired_neuron')
@patch('repair.main.view_all')
def test_full_custom_repaired_dir(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        custom_path = joinp(test_folder, 'repaired_planes_custom')
        full(test_folder, repaired_dir=custom_path)
        assert_output_exists(test_folder, repaired_dir=custom_path)


def test_full_custom_plots_dir():
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        custom_path = joinp(test_folder, 'plots_custom')
        full(test_folder, plots_dir=custom_path)
        assert_output_exists(test_folder)
        assert_equal(len(os.listdir(custom_path)), 3)
        ok_(os.path.exists(os.path.join(custom_path, 'report.pdf')))
