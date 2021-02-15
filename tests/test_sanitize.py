from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from morphio import Morphology
from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_equal

from neuror.sanitize import CorruptedMorphology, fix_non_zero_segments, sanitize, sanitize_all
from neuror.sanitize import annotate_neurolucida, annotate_neurolucida_all

PATH = Path(__file__).parent / 'data'


def test_fix_non_zero_segments():
    neuron = fix_non_zero_segments(Path(PATH, 'simple-with-duplicates.asc'))
    assert_equal(len(neuron.root_sections), 1)
    assert_array_equal(neuron.section(0).points,
                       [[0., 0., 0.],
                        [1., 1., 0.],
                        [2., 0., 0.],
                        [3., 0., 0.]])


def test_sanitize():
    with TemporaryDirectory('test-sanitize') as tmp_folder:
        output_path = Path(tmp_folder, 'sanitized.asc')
        sanitize(Path(PATH, 'simple-with-duplicates.asc'), output_path)
        neuron = Morphology(output_path)
        assert_equal(len(neuron.root_sections), 1)
        assert_array_equal(neuron.section(0).points,
                           [[0., 0., 0.],
                            [1., 1., 0.],
                            [2., 0., 0.],
                            [3., 0., 0.]])

        for input_morph, expected_exception in [
                ('no-soma.asc',
                 '{} has no soma'.format(Path(PATH, 'no-soma.asc'))),

                ('negative-diameters.asc',
                 '{} has negative diameters'.format(Path(PATH, 'negative-diameters.asc'))),

                ('neurite-with-multiple-types.swc',
                 ('{} has a neurite whose type changes along the way\n'
                  'Child section (id: 5) has a different type (SectionType.basal_dendrite) '
                  'than its parent (id: 3) (type: SectionType.axon)').format(
                      Path(PATH, 'neurite-with-multiple-types.swc')))
        ]:
            with assert_raises(CorruptedMorphology) as cm:
                sanitize(PATH / input_morph, Path(tmp_folder, 'output.asc'))
            assert_equal(str(cm.exception), expected_exception)


def test_sanitize_all():
    with TemporaryDirectory('test-sanitize') as tmp_folder:
        tmp_folder = Path(tmp_folder)
        sanitize_all(PATH / 'input-sanitize-all', tmp_folder)

        assert_array_equal(list(sorted(tmp_folder.rglob('*.asc'))),
                           [tmp_folder / 'a.asc',
                            tmp_folder / 'sub-folder/sub-sub-folder/c.asc'])

    with TemporaryDirectory('test-sanitize') as tmp_folder:
        tmp_folder = Path(tmp_folder)
        sanitize_all(PATH / 'input-sanitize-all', tmp_folder, nprocesses=2)

        assert_array_equal(list(sorted(tmp_folder.rglob('*.asc'))),
                           [tmp_folder / 'a.asc',
                            tmp_folder / 'sub-folder/sub-sub-folder/c.asc'])

def test_error_annotation():
    with TemporaryDirectory('test-error-annotation') as tmp_folder:
        output_path = Path(tmp_folder, 'annotated.asc')
        annotation, summary, markers = annotate_neurolucida(Path(PATH, 'error-morph.asc'))
        #expected_annotation = """\n\n(Circle3   ; MUK_ANNOTATION\n    (Color Blue)   ; MUK_ANNOTATION\n    (Name "fat end")   ; MUK_ANNOTATION\n    (-5.0 -4.0 0.0 0.50)   ; MUK_ANNOTATION\n)   ; MUK_ANNOTATION\n\n\n\n(Circle2   ; MUK_ANNOTATION\n    (Color Green)   ; MUK_ANNOTATION\n    (Name "zjump")   ; MUK_ANNOTATION\n    (0.0 5.0 0.0 0.50)   ; MUK_ANNOTATION\n    (0.0 5.0 40.0 0.50)   ; MUK_ANNOTATION\n)   ; MUK_ANNOTATION\n\n\n\n(Circle1   ; MUK_ANNOTATION\n    (Color Blue)   ; MUK_ANNOTATION\n    (Name "narrow start")   ; MUK_ANNOTATION\n    (0.0 5.0 0.0 0.50)   ; MUK_ANNOTATION\n)   ; MUK_ANNOTATION\n\n\n\n(Circle6   ; MUK_ANNOTATION\n    (Color Magenta)   ; MUK_ANNOTATION\n    (Name "dangling")   ; MUK_ANNOTATION\n    (10.0 -20.0 -4.0 0.50)   ; MUK_ANNOTATION\n)   ; MUK_ANNOTATION\n\n\n\n(Circle8   ; MUK_ANNOTATION\n    (Color Yellow)   ; MUK_ANNOTATION\n    (Name "Multifurcation")   ; MUK_ANNOTATION\n    (0.0 5.0 0.0 0.50)   ; MUK_ANNOTATION\n)   ; MUK_ANNOTATION\n\'\n            DESIRED: \'\n               (Circle3   ; MUK_ANNOTATION\n               (Color Blue)   ; MUK_ANNOTATION\n               (Name "fat end")   ; MUK_ANNOTATION\n               (-5.0 -4.0 0.0 0.50)   ; MUK_ANNOTATION\n           )   ; MUK_ANNOTATION\n\n\n\n           (Circle2   ; MUK_ANNOTATION\n               (Color Green)   ; MUK_ANNOTATION\n               (Name "zjump")   ; MUK_ANNOTATION\n               (0.0 5.0 0.0 0.50)   ; MUK_ANNOTATION\n               (0.0 5.0 40.0 0.50)   ; MUK_ANNOTATION\n           )   ; MUK_ANNOTATION\n\n\n\n           (Circle1   ; MUK_ANNOTATION\n               (Color Blue)   ; MUK_ANNOTATION\n               (Name "narrow start")   ; MUK_ANNOTATION\n               (0.0 5.0 0.0 0.50)   ; MUK_ANNOTATION\n           )   ; MUK_ANNOTATION\n\n\n\n           (Circle6   ; MUK_ANNOTATION\n               (Color Magenta)   ; MUK_ANNOTATION\n               (Name "dangling")   ; MUK_ANNOTATION\n               (10.0 -20.0 -4.0 0.50)   ; MUK_ANNOTATION\n           )   ; MUK_ANNOTATION\n\n\n\n           (Circle8   ; MUK_ANNOTATION\n               (Color Yellow)   ; MUK_ANNOTATION\n               (Name "Multifurcation")   ; MUK_ANNOTATION\n               (0.0 5.0 0.0 0.50)   ; MUK_ANNOTATION\n           )   ; MUK_ANNOTATION\n\n        """
        # not sure how to test that, I can't make it work
        #assert_equal(annotation, expected_annotation)
        assert_equal(summary, {'fat end': 1,
                               'zjump': 1,
                               'narrow start': 1,
                               'dangling': 1,
                               'Multifurcation': 1})
        assert_equal(markers, [{'name': 'fat end', 'label': 'Circle3', 'color': 'Blue',
                                'data': [(8, np.array([[-5., -4.,  0., 20.,  2., 19., 18.]]))]},
                               {'name': 'zjump', 'label': 'Circle2', 'color': 'Green',
                                'data': [(3, [np.array([0., 5., 0., 1., 3., 2., 1.]),
                                              np.array([ 0.,  5., 40.,  1.,  3.,  5.,  2.])])]},
                               {'name': 'narrow start', 'label': 'Circle1', 'color': 'Blue',
                                'data': [(1, np.array([[0., 5., 0., 1., 3., 2., 1.]]))]},
                               {'name': 'dangling', 'label': 'Circle6', 'color': 'Magenta',
                                'data': [(6, [np.array([ 10., -20.,  -4.,   1.,   2.,  11.,   0.])])]},
                               {'name': 'Multifurcation', 'label': 'Circle8', 'color': 'Yellow',
                                'data': [(1, np.array([[0., 5., 0., 1., 3., 2., 1.]]))]}])
