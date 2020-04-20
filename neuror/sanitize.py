'''Module for the sanitization of raw morphologies.'''
import logging
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
from tqdm import tqdm

import morphio
from morphio import MorphioError, set_maximum_warnings, SomaType
from morphio.mut import Morphology  # pylint: disable=import-error

L = logging.getLogger('neuror')


class CorruptedMorphology(Exception):
    '''Exception for morphologies that should not be used'''


def iter_morphologies(folder):
    '''Recursively yield morphology files in folder and its sub-directories.'''
    return (path for path in folder.rglob('*') if path.suffix.lower() in {'.swc', '.h5', '.asc'})


def sanitize(input_neuron, output_path):
    '''Sanitize one morphology.

    - fixes non zero segments
    - raises if the morphology has no soma
    - raises if the morphology has negative diameters

    Args:
        input_neuron (str|pathlib.Path|morphio.Morphology|morphio.mut.Morphology): input neuron
        output_path (str|pathlib.Path): output name
    '''
    neuron = morphio.Morphology(input_neuron)
    if neuron.soma.type == SomaType.SOMA_UNDEFINED:  # pylint: disable=no-member
        raise CorruptedMorphology('{} has no soma'.format(input_neuron))
    if np.any(neuron.diameters < 0):
        raise CorruptedMorphology('{} negative diameters'.format(input_neuron))

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

    Note: the sub-directory structure is maintained.

    - fixes non zero segments
    - raises if the morphology has no soma
    - raises if the morphology has negative diameters

    Args:
        input_folder (str|pathlib.Path): input neuron
        output_folder (str|pathlib.Path): output name
    '''
    set_maximum_warnings(0)

    morphologies = list(iter_morphologies(Path(input_folder)))
    func = partial(_sanitize_one, input_folder=input_folder, output_folder=output_folder)
    if nprocesses == 1:
        results = map(func, morphologies)
    else:
        results = Pool(nprocesses).imap_unordered(func, morphologies, chunksize=100)
    errored_paths = list(filter(None, tqdm(results, total=len(morphologies))))
    if errored_paths:
        L.info('Files in error:')
        for path in errored_paths:
            L.info(path)


def fix_non_zero_segments(neuron):
    '''Return a neuron with zero length segments removed

    Sections composed of a single zero length segment are deleted
    Args:
        neuron (str|pathlib.Path|morphio.Morphology|morphio.mut.Morphology): input neuron

    Returns:
        a fixed morphio.mut.Morphology
    '''
    neuron = Morphology(neuron)
    to_be_deleted = list()
    for section in neuron.iter():
        points = section.points
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        indices = np.append(0, np.nonzero(distances)[0] + 1)
        if len(indices) != len(points):
            section.points = section.points[indices]
            section.diameters = section.diameters[indices]
        if len(indices) < 2:
            to_be_deleted.append(section)

    for section in to_be_deleted:
        neuron.delete_section(section)
    return neuron
