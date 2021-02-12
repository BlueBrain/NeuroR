from pathlib import Path
from tempfile import TemporaryDirectory

from morphio import Morphology
from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_equal

from neuror.sanitize import CorruptedMorphology, fix_non_zero_segments, sanitize, sanitize_all

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
