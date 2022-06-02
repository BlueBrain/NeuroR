import tempfile
import contextlib
from pathlib import Path

import numpy as np
import pytest
from morphio import Morphology
from numpy.testing import assert_array_equal, assert_equal, assert_array_almost_equal

from morph_tool.utils import iter_morphology_files
from neurom import load_morphology
from neuror import sanitize as tested
from neuror.sanitize import annotate_neurolucida, annotate_neurolucida_all
from neuror.sanitize import fix_points_in_soma

DATA = Path(__file__).parent / 'data'


@contextlib.contextmanager
def _tmp_file(content, extension):
    with tempfile.NamedTemporaryFile(suffix=f".{extension}") as tmp_file:
        filepath = tmp_file.name
        if content:
            with open(filepath, "w") as f:
                f.write(content)
        yield filepath


def test_fix_non_zero_segments__check_downstream_tree_is_not_removed():

    content = (
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
    with _tmp_file(content, extension="asc") as filepath:

        morph = tested.fix_non_zero_segments(filepath)

        # no surprizes here
        assert len(morph.root_sections) == 1

        # 1 section is removed, 4 are left
        assert len(morph.sections) == 4

        # Removing the zero length section will introduce a trifurcation
        assert len(morph.root_sections[0].children) == 3


def test_fix_non_zero_segments__raises_if_zero_length_root_section():

    content = (
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
      (0.0 -4.0 0.0 2.0)
      (0.0 -4.0 0.0 2.0)
      (
        (0.0 -4.0 0.0 4.0)
        (0.0 -4.0 0.0 4.0)
        (0.0 -4.0 0.0 4.0)
      |
        ( 0.0 -4.0 0.0 4.0)
        (-5.0 -4.0 0.0 4.0)
      )
    )
    """
    )
    with _tmp_file(content, extension="asc") as filepath:

        with pytest.raises(
            tested.ZeroLengthRootSection,
            match="Morphology has root sections at the soma with zero length."
        ):
            tested.fix_non_zero_segments(filepath)


def test_sanitize__raises_invalid_soma():

    content = (
    """
    ( (Color DarkCyan)
      (Axon)
      (   16.67    -0.44   -20.29     0.55)  ; Root
      (   16.67     2.89   -21.19     0.83)  ; 1, R
      (   16.11     7.04   -21.05     0.83)  ; 2
      (   14.17    12.03   -21.05     0.83)  ; 3
      (   13.34    16.74   -15.40     0.83)  ; 4
      (   11.96    20.62   -11.20     0.83)  ; 5
      (   16.64    38.10    -9.14     0.83)  ; 9
      ;  End of split
    )  ;  End of tree
    """
    )
    with _tmp_file(content, extension="asc") as filepath:

        with pytest.raises(
            tested.CorruptedMorphology,
            match=f'{filepath} has an invalid or no soma.',
        ):
            tested.sanitize(filepath, None)


def test_sanitize__raises_heterogeneous_neurite():
    import re
    content = (
    """
     1 1  0  0 0 1. -1
     2 3  0  0 0 1.  1
     3 3  0  5 0 1.  2
     4 3 -5  5 0 1.5 3
     5 3  6  5 0 1.5 3
     6 2  0  0 0 1.  1
     7 2  0 -4 0 1.  6
     8 2  6 -4 0 2.  7
     9 3 -5 -4 0 2.  7  # Type went from 2 to 3
    """
    )
    with _tmp_file(content, extension="swc") as filepath:
        with pytest.raises(
            tested.CorruptedMorphology,
            match=(
                f'{filepath} has a neurite whose type changes along the way\n'
                r'Child section \(id: 5\) has a different type \(SectionType.basal_dendrite\) '
                r'than its parent \(id: 3\) \(type: SectionType.axon\)'
            ),
        ):
            tested.sanitize(filepath, None)


def test_sanitize__negative_diameters():

    content = (
    """
    ("CellBody"
    (Color Red)
    (CellBody)
    (0 0 0 2)
    )

    ((Dendrite)
      (0 0 0 1)
      (1 1 0 2)
      (2 0 0 -3)
      (3 0 0 4)
      (
        (3 0 0 4)
        (4 1 0 5)
        |
        (3 0 0 4)
        (6 4 2 5)
      )
    )
    """
    )
    with _tmp_file(content, extension="asc") as input_filepath, \
         _tmp_file("", extension="asc") as output_filepath:

        tested.sanitize(input_filepath, output_filepath)

        assert_array_equal(
            Morphology(output_filepath).diameters,
            [1., 2., 0., 4., 4., 5., 4., 5.],
        )



def test_sanitize_all(tmpdir):
    tmpdir = Path(tmpdir)
    tested.sanitize_all(DATA / 'input-sanitize-all', tmpdir)

    assert_array_equal(list(sorted(tmpdir.rglob('*.asc'))),
                       [tmpdir / 'a.asc',
                        tmpdir / 'sub-folder/sub-sub-folder/c.asc'])


def test_sanitize_all_np2(tmpdir):
    tmpdir = Path(tmpdir)
    tested.sanitize_all(DATA / 'input-sanitize-all', tmpdir, nprocesses=2)

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
    neuron = load_morphology(DATA / "simple_inside_soma.asc")

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
    with pytest.raises(tested.CorruptedMorphology, match="An entire section is located inside the soma"):
        fix_points_in_soma(neuron)
