import os

from mock import patch
import numpy as np
from neurom import COLS, load_neuron
from neurom.geom import bounding_box
from nose.tools import assert_dict_equal, ok_
from numpy.testing import assert_array_almost_equal, assert_equal, assert_array_equal

import repair.main as test_module
from morphio import SectionType
from repair.main import Action
from .utils import setup_tempdir

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

SIMPLE = load_neuron(os.path.join(DATA_PATH, 'simple.swc'))

class DummySection:
    def __init__(self, points, children=None):
        self.points = np.array(points)
        self.children = children or []


def test_is_cut_section():
    section = SIMPLE.neurites[0].root_node
    assert_equal(test_module.is_cut_section(section, [[2,2,2]]),
                 False)

    assert_equal(test_module.is_cut_section(section, [[0,0,0]]),
                 True)


def test_is_neurite_intact():
    neurite = SIMPLE.neurites[0]
    assert_equal(test_module.is_neurite_intact(neurite, [[2,2,2]]),
                 True)

    assert_equal(test_module.is_neurite_intact(neurite, [[0,0,0]]),
                 False)


def test_find_intact_sub_trees():
    assert_equal(len(test_module.find_intact_sub_trees(SIMPLE, [[2,2,2]])),
                 2)

    assert_equal(test_module.find_intact_sub_trees(SIMPLE, [[0,0,0]]),
                 [])


def test_section_length():
    assert_equal(test_module._section_length(SIMPLE.neurites[0].root_node), 5)


def test_branching_angles():
    assert_equal(test_module.branching_angles(SIMPLE.neurites[0].root_node),
                 [(0, 1.5707963267948966), (0, 1.5707963267948966)])

    # Test skip too short sections
    assert_equal(test_module.branching_angles(DummySection([[0,0,0], [0,0,1e-9]])),
                 [])

    # Test skip too short child sections
    tiny_child = DummySection([[0,0,0], [0,0,1e-9]])
    parent = DummySection([[0,0,0], [0,0,1]], children=[tiny_child])
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
    angles = test_module.intact_branching_angles(SIMPLE.neurites)
    assert_array_almost_equal(angles[SectionType.basal_dendrite][0],
                              [1.5707963267948966, 1.5707963267948966])


def test_compute_sholl_data():
    data = test_module.compute_sholl_data(SIMPLE.neurites, SIMPLE.soma.center)
    assert_dict_equal(data[SectionType.basal_dendrite],
                      {0: {0: {Action.TERMINATION: 0, Action.CONTINUATION: 1, Action.BIFURCATION: 1},
                           1: {Action.TERMINATION: 2, Action.CONTINUATION: 2, Action.BIFURCATION: 0}}})


def test_get_sholl_proba():
    sholl_data = {SectionType.axon: {0: {1: {Action.TERMINATION: 2,
                                             Action.BIFURCATION: 2,
                                             Action.CONTINUATION: 4
    }}}}

    assert_dict_equal(test_module.get_sholl_proba(sholl_data, SectionType.axon, 0, 1),
                      {Action.TERMINATION: 0.25,
                       Action.BIFURCATION: 0.25,
                       Action.CONTINUATION: 0.5})

    # No info for pseudo_order == 2, re-use data from pseudo_order == 1
    assert_dict_equal(test_module.get_sholl_proba(sholl_data, SectionType.axon, 0, 2),
                      {Action.TERMINATION: 0.25,
                       Action.BIFURCATION: 0.25,
                       Action.CONTINUATION: 0.5})

    # No info for sholl_layer == 1, use default value
    assert_dict_equal(test_module.get_sholl_proba(sholl_data, SectionType.axon, 1, 2),
                      {Action.TERMINATION: 1,
                       Action.BIFURCATION: 0,
                       Action.CONTINUATION: 0})

    # No data at all, use default value
    assert_dict_equal(test_module.get_sholl_proba({SectionType.axon: {1: {}}}, SectionType.axon, 1, 2),
                      {Action.TERMINATION: 1,
                       Action.BIFURCATION: 0,
                       Action.CONTINUATION: 0})


def test_continuation():
    test_module.continuation(DummySection([[1.0,1,1,1], [1,1,2,1]]), [0,0,0])
    test_module.continuation(DummySection([[1.0,1,1,1], [1,1,2,1]]), [1,1,1])


def test_get_similar_child_diameters():
    children = [DummySection([[1.0,1,1,10]]),
                DummySection([[1.0,1,1,12]])]
    sections = [DummySection([[1.0,1,1,1], [1,1,2,1]], children=children)]

    similar_section = DummySection([[1.0,1,1,1], [1,1,2,1]], children=[])
    assert_array_equal(test_module.get_similar_child_diameters(sections, similar_section),
                       [20, 24])

    different_section = DummySection([[1.0,1,1, 42]],
                                     children=[DummySection([[1.0,1,1,1]]),
                                               DummySection([[1.0,1,1,2]])])
    assert_array_equal(test_module.get_similar_child_diameters(sections, different_section),
                       [84, 84])


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
                              np.array([[ 0.      ,  0.      ,  0.      ],
                                        [ 0.      ,  5.      ,  0.      ],
                                        [ 0.170201,  9.98424 ,  0.358311],
                                        [ 0.469826, 14.901759,  1.211672]], dtype=np.float32))


def test_compute_sholl_data():
    sholl_data = test_module.compute_sholl_data(SIMPLE.neurites, SIMPLE.soma.center)
    assert_dict_equal(sholl_data[SectionType.basal_dendrite][0][0],
                      {Action.TERMINATION: 0, Action.CONTINUATION: 1, Action.BIFURCATION: 1})
    assert_dict_equal(sholl_data[SectionType.basal_dendrite][0][1],
                      {Action.TERMINATION: 2, Action.CONTINUATION: 2, Action.BIFURCATION: 0})
    assert_dict_equal(sholl_data[SectionType.apical_dendrite], {})
    assert_dict_equal(sholl_data[SectionType.axon],
                      {0: {0: {Action.BIFURCATION: 1,
                               Action.CONTINUATION: 1,
                               Action.TERMINATION: 0},
                           1: {Action.BIFURCATION: 0,
                               Action.CONTINUATION: 2,
                               Action.TERMINATION: 2}}})

def test_subtree_classification():
    neuron = load_neuron(os.path.join(DATA_PATH, 'repair_type.asc'))

    APICAL_POINT_ID = 3
    # Pre check to be sure that the apical point is the one it should be
    assert_array_almost_equal(neuron.sections[APICAL_POINT_ID].points,
                              np.array([[0., 0., 4., 1. ],
                                        [0., 0., 0., 0.5],
                                        [1., 0., 0., 0.5]], dtype=np.float32))

    mapping = test_module.subtree_classification(neuron, APICAL_POINT_ID)
    assert_dict_equal({sec.id: v for sec, v in mapping.items()},
                      {
                          1: test_module.RepairType.trunk,
                          2: test_module.RepairType.oblique,
                          3: test_module.RepairType.trunk,
                          4: test_module.RepairType.tuft,
                          5: test_module.RepairType.tuft,
                          6: test_module.RepairType.basal,
                      })
