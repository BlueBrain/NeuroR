from pathlib import Path

from morph_tool import diff
from morphio import SectionType, Morphology
from neurom import COLS
from neurom import iter_neurites
from neurom import load_morphology
import pytest
from numpy.testing import assert_array_equal

import neuror.axon as test_module

DATA_PATH = Path(__file__).parent / 'data'

SIMPLE_PATH = DATA_PATH / 'simple.swc'
SLICE_PATH = DATA_PATH / 'neuron-slice.h5'


@pytest.fixture
def neuron_slice():
    return load_morphology(SLICE_PATH)


@pytest.fixture
def neuron_simple():
    return load_morphology(SIMPLE_PATH)


def test_tree_distance(neuron_slice):

    neurites = neuron_slice.neurites
    root = neurites[0].root_node
    child1, child2 = root.children

    assert test_module._tree_distance(child1, root) == 1
    assert test_module._tree_distance(child1, child2) == 1
    assert test_module._tree_distance(child2, child1) == 1

    assert test_module._tree_distance(child1.children[0], child2.children[0]) == 3

    with pytest.raises(test_module.NeuroRError):
        test_module._tree_distance(root, neurites[1].root_node)


def test__downstream_pathlength(neuron_simple):
    root = neuron_simple.to_morphio().root_sections[0]
    assert test_module._downstream_pathlength(root) == 16
    assert test_module._downstream_pathlength(root.children[0]) == 5


def test__similar_section(neuron_simple):
    root = neuron_simple.to_morphio().root_sections[0]
    similar = test_module._similar_section(neuron_simple.to_morphio().root_sections, root)
    assert similar.id == root.id


def test__sort_intact_sections_by_score(neuron_simple):
    root = neuron_simple.to_morphio().root_sections[0]

    res = test_module._sort_intact_sections_by_score(root, root, neuron_simple.to_morphio().root_sections)
    assert res[0].id == root.id


def test__repair():
    neuron = load_morphology(Path(DATA_PATH, 'valid.h5'), mutable=True)
    axon = neuron.neurites[0].root_node
    assert axon.type == SectionType.axon
    test_module.repair(neuron, axon, [axon], [axon], set(), y_extent=10000)
    assert len(axon.children) == 1
    assert_array_equal(axon.children[0].points[0, COLS.XYZ], axon.points[-1, COLS.XYZ])
    assert not diff(neuron.to_morphio(), DATA_PATH / 'axon-repair.h5')


def test__repair_no_intact_axon():
    filename = Path(DATA_PATH, 'valid.h5')
    neuron = load_morphology(filename, mutable=True)
    axon = neuron.neurites[0].root_node
    used_axon_branches = set()
    test_module.repair(neuron, axon, [], [axon], used_axon_branches, y_extent=10000)
    # There should not be any repair
    assert not diff(neuron.to_morphio(), filename)
