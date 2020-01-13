'''Module for the sanitization of raw morphologies.'''
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np

from morphio import MorphioError, set_maximum_warnings
from morphio.mut import Morphology  # pylint: disable=import-error

L = logging.getLogger('morph-repair')


def iter_morphologies(folder):
    '''Recursively yield morphology files in folder and its sub-directories.'''
    return (path for path in folder.rglob('*') if path.suffix.lower() in {'.swc', '.h5', '.asc'})


def sanitize(input_neuron, output_path):
    '''Sanitize one morphology.

    Note: it currently only fixes non zero segments but may do more in the future

    Args:
        input_neuron (str|pathlib.Path|morphio.Morphology|morphio.mut.Morphology): input neuron
        output_path (str|pathlib.Path): output name
    '''
    fix_non_zero_segments(input_neuron).write(str(output_path))


def sanitize_all(input_folder, output_folder):
    '''Sanitize all morphologies in input_folder and its sub-directories.

    Args:
        input_folder (str|pathlib.Path): input neuron
        output_path (str|pathlib.Path): output name
    '''
    set_maximum_warnings(0)

    in_errors = list()
    for path in tqdm(list(iter_morphologies(Path(input_folder)))):
        try:
            sanitize(path, Path(output_folder, path.name))
        except MorphioError:
            in_errors.append(path)
    L.info('Files in error:')
    L.info(in_errors)


def fix_non_zero_segments(filename):
    '''Return a neuron with zero length segments removed

    Sections composed of a single zero length segment are deleted
    Args:
        input_folder (str|pathlib.Path): input neuron

    Returns:
        a fixed morphio.mut.Morphology
    '''
    neuron = Morphology(filename)
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
