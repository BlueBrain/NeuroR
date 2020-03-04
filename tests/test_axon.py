from pathlib import Path

from neurom import load_neuron
from nose.tools import assert_raises, ok_
from numpy.testing import assert_array_equal, assert_equal

from morph_tool import diff
from morphio import SectionType

import neuror.axon as test_module

DATA_PATH = Path(__file__).parent / 'data'

SIMPLE_PATH = DATA_PATH / 'simple.swc'
SLICE_PATH = DATA_PATH / 'neuron-slice.h5'
SIMPLE = load_neuron(SIMPLE_PATH)
SLICE = load_neuron(SLICE_PATH)


def test_tree_distance():
    child = SLICE.neurites[0].root_node.children[0]
    assert_equal(test_module._tree_distance(child, SLICE.neurites[0].root_node), 1)

    child1, child2 = SLICE.neurites[0].root_node.children
    assert_equal(test_module._tree_distance(child1, child2), 1)
    assert_equal(test_module._tree_distance(child2, child1), 1)

    assert_equal(test_module._tree_distance(child1.children[0], child2.children[0]), 3)


    assert_raises(Exception, test_module._tree_distance, SLICE.neurites[0].root_node,
                  SLICE.neurites[1].root_node)

def test__downstream_pathlength():
    root = SIMPLE.root_sections[0]
    assert_equal(test_module._downstream_pathlength(root), 16)
    assert_equal(test_module._downstream_pathlength(root.children[0]), 5)


def test__similar_section():
    root = SIMPLE.root_sections[0]
    similar = test_module._similar_section(SIMPLE.root_sections, root)
    assert_equal(similar.id, root.id)

def test__sort_intact_sections_by_score():
    root = SIMPLE.root_sections[0]

    res = test_module._sort_intact_sections_by_score(root, root, SIMPLE.root_sections)
    assert_equal(res[0].id, root.id)


def test__repair():
    neuron = load_neuron(Path(DATA_PATH, 'valid.h5'))
    axon = neuron.root_sections[0]
    assert_equal(axon.type, SectionType.axon)
    test_module.repair(neuron, axon, [axon], [axon], y_extent=10000)
    assert_equal(len(axon.children), 1)
    assert_array_equal(axon.children[0].points[0], axon.points[-1])


def test__repair_no_intact_axon():
    filename = Path(DATA_PATH, 'valid.h5')
    neuron = load_neuron(filename)
    axon = neuron.root_sections[0]
    test_module.repair(neuron, axon, [], [axon], y_extent=10000)
    # There should not be any repair
    ok_(not diff(neuron, filename))
