'''Module for the sanitization of raw morphologies.'''
import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

import numpy as np
from morphio import MorphioError, SomaType, set_maximum_warnings
from morphio.mut import Morphology  # pylint: disable=import-error
from neurom.check import morphology_checks as mc
from neurom.check import CheckResult
from neurom.apps.annotate import annotate
from neurom import load_morphology

L = logging.getLogger('neuror')
_ZERO_LENGTH = 1e-4


class CorruptedMorphology(Exception):
    '''Exception for morphologies that should not be used'''


def iter_morphologies(folder):
    '''Recursively yield morphology files in folder and its sub-directories.'''
    return (path for path in folder.rglob('*') if path.suffix.lower() in {'.swc', '.h5', '.asc'})


def sanitize(input_neuron, output_path):
    '''Sanitize one morphology.

    - ensures it can be loaded with MorphIO
    - raises if the morphology has no soma or of invalid format
    - removes unifurcations
    - set negative diameters to zero
    - raises if the morphology has a neurite whose type changes along the way
    - removes segments with near zero lengths (shorter than 1e-4)

    Args:
        input_neuron (str|pathlib.Path|morphio.Morphology|morphio.mut.Morphology): input neuron
        output_path (str|pathlib.Path): output name
    '''
    neuron = Morphology(input_neuron)

    if neuron.soma.type == SomaType.SOMA_UNDEFINED:  # pylint: disable=no-member
        raise CorruptedMorphology('{} has an invalid or no soma'.format(input_neuron))

    neuron.remove_unifurcations()

    for section in neuron.iter():
        section.diameters = np.clip(section.diameters, 0, None)

    for root in neuron.root_sections:  # pylint: disable=not-an-iterable
        for section in root.iter():
            if section.type != root.type:
                raise CorruptedMorphology(f'{input_neuron} has a neurite whose type changes along '
                                          'the way\n'
                                          f'Child section (id: {section.id}) has a different type '
                                          f'({section.type}) than its parent (id: '
                                          f'{section.parent.id}) (type: {section.parent.type})')

    fix_non_zero_segments(neuron).write(str(output_path))


def _sanitize_one(path, input_folder, output_folder):
    '''Function to be called by sanitize_all to catch all exceptions
    and return path if in error

    Since Pool.imap_unordered only supports one argument, the argument
    is a tuple: (path, input_folder, output_folder).
    '''
    relative_path = path.relative_to(input_folder)
    output_dir = output_folder / relative_path.parent
    if not output_dir.exists():
        # exist_ok=True since there can be race conditions because of Pool
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sanitize(path, output_dir / path.name)
    except (MorphioError, CorruptedMorphology):
        return str(path)
    else:
        return None


def sanitize_all(input_folder, output_folder, nprocesses=1):
    '''Sanitize all morphologies in input_folder and its sub-directories.

    See :func:`~neuror.sanitize.sanitize` for more information on the sanitization process.

    Args:
        input_folder (str|pathlib.Path): input neuron
        output_folder (str|pathlib.Path): output name

    .. note:: the sub-directory structure is maintained.
    '''
    set_maximum_warnings(0)

    morphologies = list(iter_morphologies(Path(input_folder)))
    func = partial(_sanitize_one, input_folder=input_folder, output_folder=output_folder)
    if nprocesses == 1:
        results = map(func, morphologies)
    else:
        with Pool(nprocesses) as pool:
            results = list(pool.imap_unordered(func, morphologies, chunksize=100))
    errored_paths = list(filter(None, tqdm(results, total=len(morphologies))))
    if errored_paths:
        L.info('Files in error:')
        for path in errored_paths:
            L.info(path)


def fix_non_zero_segments(neuron, zero_length=_ZERO_LENGTH):
    '''Return a neuron with zero length segments removed

    Sections composed of a single zero length segment are deleted, where zero is parametrized
    by zero_length

    Args:
        neuron (str|pathlib.Path|morphio.Morphology|morphio.mut.Morphology): input neuron
        zero_length (float): smallest length of a segment

    Returns:
        a fixed morphio.mut.Morphology
    '''
    neuron = Morphology(neuron)
    to_be_deleted = []
    for section in neuron.iter():
        points = section.points
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distances[distances < zero_length] = 0
        indices = np.append(0, np.nonzero(distances)[0] + 1)
        if len(indices) != len(points):
            section.points = section.points[indices]
            section.diameters = section.diameters[indices]
        if len(indices) < 2:
            to_be_deleted.append(section)

    for section in to_be_deleted:
        neuron.delete_section(section)
    return neuron


def annotate_neurolucida(morph_path, checkers=None):
    """Annotate errors on a morphology in neurolucida format.

    Args:
        morph_path (str): absolute path to an ascii morphology
        checkers (dict): dict of checker functons from neurom with function as keys
            and marker data in a dict as values, if None, default checkers are used

    Default checkers include:
        - fat ends
        - z-jumps
        - narrow start
        - dangling branch
        - multifurcation

    Returns:
        annotations to append to .asc file
        dict of error summary
        dict of error markers
    """
    if checkers is None:
        checkers = {
            mc.has_no_fat_ends: {"name": "fat end", "label": "Circle3", "color": "Blue"},
            partial(mc.has_no_jumps, axis="z"): {
                "name": "zjump",
                "label": "Circle2",
                "color": "Green",
            },
            mc.has_no_narrow_start: {"name": "narrow start", "label": "Circle1", "color": "Blue"},
            mc.has_no_dangling_branch: {"name": "dangling", "label": "Circle6", "color": "Magenta"},
            mc.has_multifurcation: {
                "name": "Multifurcation",
                "label": "Circle8",
                "color": "Yellow",
            },
        }

    def _try(checker, neuron):
        """Try to apply a checker, returns True if exception raised, so the checker is bypassed."""
        try:
            return checker(neuron)
        except Exception as e:  # pylint: disable=broad-except
            L.exception("%s failed on %s", checker, morph_path)
            L.exception(e, exc_info=True)
            return CheckResult(True)

    neuron = load_morphology(morph_path)
    results = [_try(checker, neuron) for checker in checkers]
    markers = [
        dict(setting, data=result.info)
        for result, setting in zip(results, checkers.values())
        if not result.status
    ]
    summary = {
        setting["name"]: len(result.info)
        for result, setting in zip(results, checkers.values())
        if result.info
    }
    return annotate(results, checkers.values()), summary, markers


def annotate_neurolucida_all(morph_paths, nprocesses=1):
    """Annotate errors on a list of morphologies in neurolicida format.

    Args:
        morph_paths (list): list of str of paths to morphologies.

    Returns:
        dict annotations to append to .asc file (morph_path as keys)
        dict of dict of error summary (morph_path as keys)
        dict of dict of markers (morph_path as keys)
    """
    summaries, annotations, markers = {}, {}, {}
    with Pool(nprocesses) as pool:
        for morph_path, result in zip(
            morph_paths, pool.map(annotate_neurolucida, morph_paths)
        ):
            morph_path = str(morph_path)
            annotations[morph_path], summaries[morph_path], markers[morph_path] = result
    return annotations, summaries, markers
