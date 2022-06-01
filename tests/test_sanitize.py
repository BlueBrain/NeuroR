import tempfile
from pathlib import Path

import numpy as np
import pytest
from morphio import Morphology
from numpy.testing import assert_array_equal, assert_equal, assert_array_almost_equal

from morph_tool.utils import iter_morphology_files
from neurom import load_neuron
from neuror.sanitize import CorruptedMorphology, fix_non_zero_segments, sanitize, sanitize_all
from neuror.sanitize import annotate_neurolucida, annotate_neurolucida_all
from neuror.sanitize import fix_points_in_soma

DATA = Path(__file__).parent / 'data'


def test_fix_non_zero_segments():
    neuron = fix_non_zero_segments(Path(DATA, 'simple-with-duplicates.asc'))
    assert len(neuron.root_sections) == 1
    assert_array_equal(neuron.section(0).points,
                       [[0., 0., 0.],
                        [1., 1., 0.],
                        [2., 0., 0.],
                        [3., 0., 0.]])


def test_fix_non_zero_segments__check_downstream_tree_is_not_removed():

    with tempfile.NamedTemporaryFile(suffix=".asc") as tfile:
        filepath = tfile.name
        with open(filepath, "w") as f:
            f.write(
                """
                ("CellBody"
                  (Color Red)
                  (CellBody)
                  ( 0.1  0.1 0.0 0.1)
                  (-0.1  0.1 0.0 0.1)
                  (-0.1 -0.1 0.0 0.1)
                  ( 0.1 -0.1 0.0 0.1)
                )

                ( (Color Cyan)
                  (Axon)
                  (0.0  0.0 0.0 2.0)
                  (0.0 -4.0 0.0 2.0)
                  (
                    (0.0 -4.0 0.0 4.0)
                    (0.0 -4.0 0.0 4.0)
                    (0.0 -4.0 0.0 4.0)
                    (
                        (6.0 -4.0 0.0 4.0)
                        (7.0 -5.0 0.0 4.0)
                    |
                        (6.0 -4.0 0.0 4.0)
                        (8.0 -4.0 0.0 4.0)
                    )
                  |
                    ( 0.0 -4.0 0.0 4.0)
                    (-5.0 -4.0 0.0 4.0)
                  )
                )
                """
            )

        morph = fix_non_zero_segments(filepath)

        assert len(morph.sections) == 4

        axon = morph.root_sections[0]

        # the zero length section should be removed
        # but the tree downstream should be joined upstream
        #assert_array_equal(
        #    axon.children[0].points,
        #    [[6.0, -4.0, 0.0], [7.0, -5.0, 0.0]],
        #)
        #assert_array_equal(
        #    axon.children[0].points,
        #    [[6.0, -4.0, 0.0], [8.0, -4.0, 0.0]]
        #)



def test_sanitize(tmpdir):
    output_path = Path(tmpdir, 'sanitized.asc')
    sanitize(Path(DATA, 'simple-with-duplicates.asc'), output_path)
    neuron = Morphology(output_path)
    assert len(neuron.root_sections) == 1
    assert_array_equal(neuron.section(0).points,
                       [[0., 0., 0.],
                        [1., 1., 0.],
                        [2., 0., 0.],
                        [3., 0., 0.]])

    with pytest.raises(CorruptedMorphology,
                       match=f'{DATA / "no-soma.asc"} has an invalid or no soma'):
        sanitize(DATA / 'no-soma.asc', Path(tmpdir, 'output.asc'))

    with pytest.raises(CorruptedMorphology) as e:
        sanitize(DATA / 'neurite-with-multiple-types.swc', Path(tmpdir, 'output.asc'))
    assert e.value.args[0] == (
        f'{DATA / "neurite-with-multiple-types.swc"} has a neurite whose type changes'
        ' along the way\nChild section (id: 5) has a different type (SectionType.'
        'basal_dendrite) than its parent (id: 3) (type: SectionType.axon)')

    out_path = Path(tmpdir, 'output.asc')
    sanitize(DATA / 'negative-diameters.asc', out_path)
    assert_array_equal(next(Morphology(out_path).iter()).diameters, [2, 2, 0, 2])


def test_sanitize_all(tmpdir):
    tmpdir = Path(tmpdir)
    sanitize_all(DATA / 'input-sanitize-all', tmpdir)

    assert_array_equal(list(sorted(tmpdir.rglob('*.asc'))),
                       [tmpdir / 'a.asc',
                        tmpdir / 'sub-folder/sub-sub-folder/c.asc'])


def test_sanitize_all_np2(tmpdir):
    tmpdir = Path(tmpdir)
    sanitize_all(DATA / 'input-sanitize-all', tmpdir, nprocesses=2)

    assert_array_equal(list(sorted(tmpdir.rglob('*.asc'))),
                       [tmpdir / 'a.asc',
                        tmpdir / 'sub-folder/sub-sub-folder/c.asc'])


def test_error_annotation():
    annotation, summary, markers = annotate_neurolucida(
        Path(DATA, 'test-error-detection/error-morph.asc')
    )
    assert summary == {'fat end': 1,
                           'zjump': 1,
                           'narrow start': 1,
                           'dangling': 1,
                           'Multifurcation': 1}
    assert_equal(markers, [{'name': 'fat end', 'label': 'Circle3', 'color': 'Blue',
                            'data': [(7, np.array([[-5., -4.,  0., 20.]], dtype=np.float32))]},
                           {'name': 'zjump', 'label': 'Circle2', 'color': 'Green',
                            'data': [(2, [np.array([0., 5., 0., 1.], dtype=np.float32),
                                          np.array([0.,  5., 40.,  1.], dtype=np.float32)])]},
                           {'name': 'narrow start', 'label': 'Circle1', 'color': 'Blue',
                            'data': [(0, np.array([[0., 5., 0., 1.]], dtype=np.float32))]},
                           {'name': 'dangling', 'label': 'Circle6', 'color': 'Magenta',
                            'data': [(5, [np.array([10., -20.,  -4., 1.], dtype=np.float32)])]},
                           {'name': 'Multifurcation', 'label': 'Circle8', 'color': 'Yellow',
                            'data': [(0, np.array([[0., 5., 0., 1.]], dtype=np.float32))]}])


def test_error_annotation_all():

    input_dir = Path(DATA, 'test-error-detection')
    # this ensure morphs are ordered as expected
    morph_paths = sorted([str(morph) for morph in iter_morphology_files(input_dir)])
    annotations, summaries, markers = annotate_neurolucida_all(morph_paths)
    assert summaries == {str(morph_paths[0]): {'fat end': 1,
                                                   'zjump': 1,
                                                   'narrow start': 1,
                                                   'dangling': 1,
                                                   'Multifurcation': 1},
                             str(morph_paths[1]): {}}
    assert_equal(markers, {
        str(morph_paths[0]): [{'name': 'fat end', 'label': 'Circle3', 'color': 'Blue',
                               'data': [(7, np.array([[-5., -4.,  0., 20.]], dtype=np.float32))]},
                              {'name': 'zjump', 'label': 'Circle2', 'color': 'Green',
                               'data': [(2, [np.array([0., 5., 0., 1.], dtype=np.float32),
                                             np.array([0.,  5., 40.,  1.], dtype=np.float32)])]},
                              {'name': 'narrow start', 'label': 'Circle1', 'color': 'Blue',
                               'data': [(0, np.array([[0., 5., 0., 1.]], dtype=np.float32))]},
                              {'name': 'dangling', 'label': 'Circle6', 'color': 'Magenta',
                               'data': [(5, [np.array([10., -20.,  -4., 1.], dtype=np.float32)])]},
                              {'name': 'Multifurcation', 'label': 'Circle8', 'color': 'Yellow',
                               'data': [(0, np.array([[0., 5., 0., 1.]], dtype=np.float32))]}],
        str(morph_paths[1]): []})


def test_fix_points_in_soma():
    neuron = load_neuron(DATA / "simple_inside_soma.asc")

    fix_points_in_soma(neuron)

    # In the given morph:
    #     * in the first dendrite, the function should just remove the first point, because the
    #       one that should be added is too close to the second point.
    #     * in the second dendrite, the function should do nothing, because all points are outside
    #       the soma.
    #     * in the third dendrite, the function should replace the first point with a new one.
    #     * in the forth dendrite, the function should remove the six first points, because the
    #       one that should be added is too close to the second point.
    expected = np.array(
        [
            [1., 0., 0., 1.],
            [2., 0., 0., 1.],
            [2., 0., 0., 1.],
            [3., 1., 0., 1.],
            [2., 0., 0., 1.],
            [3., -1., 0., 1.],
            [0., 1., 0., 1.],
            [0., 2., 0., 1.],
            [0., 2., 0., 1.],
            [1., 3., 0., 1.],
            [0., 2., 0., 1.],
            [-1., 3., 0., 1.],
            [0.57735026, 0.57735026, 0.57735026, 1.],
            [1., 1., 1., 1.],
            [2., 2., 2., 1.],
            [2., 2., 2., 1.],
            [3., 3., 3., 1.],
            [2., 2., 2., 1.],
            [3., 2., 3., 1.],
            [0., 0., 1.00001 , 1.],
            [0., 0., 2., 1.],
            [0., 0., 2., 1.],
            [0., 1., 3., 1.],
            [0., 0., 2., 1.],
            [0., -1., 3., 1.],
        ],
        dtype=neuron.points.dtype
    )

    assert_array_almost_equal(
        neuron.points,
        expected
    )

    # Test that it fails when an entire section is located inside the soma
    neuron.soma.radius = 10
    with pytest.raises(CorruptedMorphology, match="An entire section is located inside the soma"):
        fix_points_in_soma(neuron)
