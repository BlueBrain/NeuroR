from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from morphio import Morphology
from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_equal

from morph_tool.utils import iter_morphology_files
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
                 '{} has an invalid or no soma'.format(Path(PATH, 'no-soma.asc'))),

                ('neurite-with-multiple-types.swc',
                 ('{} has a neurite whose type changes along the way\n'
                  'Child section (id: 5) has a different type (SectionType.basal_dendrite) '
                  'than its parent (id: 3) (type: SectionType.axon)').format(
                      Path(PATH, 'neurite-with-multiple-types.swc')))
        ]:
            with assert_raises(CorruptedMorphology) as cm:
                sanitize(PATH / input_morph, Path(tmp_folder, 'output.asc'))
            assert_equal(str(cm.exception), expected_exception)

        out_path = Path(tmp_folder, 'output.asc')
        sanitize(PATH / 'negative-diameters.asc', out_path)
        assert_equal(next(Morphology(out_path).iter()).diameters, [2, 2, 0, 2])


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
    annotation, summary, markers = annotate_neurolucida(
        Path(PATH, 'test-error-detection/error-morph.asc')
    )
    assert_equal(summary, {'fat end': 1,
                           'zjump': 1,
                           'narrow start': 1,
                           'dangling': 1,
                           'Multifurcation': 1})
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

    input_dir = Path(PATH, 'test-error-detection')
    # this ensure morphs are ordered as expected
    morph_paths = sorted([str(morph) for morph in iter_morphology_files(input_dir)])
    annotations, summaries, markers = annotate_neurolucida_all(morph_paths)
    assert_equal(summaries, {str(morph_paths[0]): {'fat end': 1,
                                                   'zjump': 1,
                                                   'narrow start': 1,
                                                   'dangling': 1,
                                                   'Multifurcation': 1},
                             str(morph_paths[1]): {}})
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
