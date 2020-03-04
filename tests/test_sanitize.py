from pathlib import Path
from neuror.sanitize import fix_non_zero_segments, sanitize, sanitize_all, CorruptedMorphology

from morphio import Morphology
from numpy.testing import assert_equal, assert_array_equal

from nose.tools import assert_raises
from tempfile import TemporaryDirectory

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

        assert_raises(CorruptedMorphology, sanitize,
                      Path(PATH, 'no-soma.asc'), Path(tmp_folder, 'no-soma.asc'))

        assert_raises(CorruptedMorphology, sanitize,
                      Path(PATH, 'negative-diameters.asc'), Path(tmp_folder, 'negative-diameter.asc'))


def test_sanitize_all():
    with TemporaryDirectory('test-sanitize') as tmp_folder:
        tmp_folder = Path(tmp_folder)
        sanitize_all(PATH / 'input-sanitize-all', tmp_folder)

        assert_array_equal(list(sorted(tmp_folder.rglob('*.asc'))),
                           [tmp_folder / 'a.asc',
                            tmp_folder / 'sub-folder/sub-sub-folder/c.asc'])
