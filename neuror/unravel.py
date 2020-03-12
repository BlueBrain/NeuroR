'''The unravel module

Unravelling is the action of "stretching" the cell that
has been shrunk because of the dehydratation caused by the slicing'''
import json
import logging
import os
from pathlib import Path

import morphio
import numpy as np
import pandas as pd
from morph_tool.utils import iter_morphology_files
from scipy.spatial.ckdtree import cKDTree

from neuror.cut_plane import CutPlane
from neuror.utils import RepairJSON

L = logging.getLogger('neuror')

DEFAULT_WINDOW_HALF_LENGTH = 5


def _get_principal_direction(points):
    '''Return the principal direction of a point cloud
    It is the eigen vector of the covariance matrix with the highest eigen value'''

    X = np.copy(np.asarray(points))
    X -= np.mean(X, axis=0)
    C = np.dot(X.T, X)
    w, v = np.linalg.eig(C)
    return v[:, w.argmax()]


def _unravel_section(sec, new_section, window_half_length):
    '''Unravel a section'''
    points = sec.points
    point_count = len(points)
    if new_section.is_root:
        unravelled_points = [new_section.points[0]]
    else:
        unravelled_points = [new_section.parent.points[-1]]

    for window_center in range(1, point_count):
        window_start = max(0, window_center - window_half_length - 1)
        window_end = min(point_count, window_center + window_half_length + 1)

        direction = _get_principal_direction(points[window_start:window_end])

        segment = points[window_center] - points[window_center - 1]

        # make it span length the same as the original segment within the window
        direction *= np.linalg.norm(segment) / np.linalg.norm(direction)

        # point it in the same direction as before
        direction *= np.sign(np.dot(segment, direction))

        unravelled_points.append(direction + unravelled_points[window_center - 1])

    new_section.points = unravelled_points


def unravel(filename, window_half_length=DEFAULT_WINDOW_HALF_LENGTH):
    '''Return an unravelled neuron

    Segment are unravelled iteratively
    Each segment direction is replaced by the averaged direction in a sliding window
    around this segment. And the original segment length is preserved.
    The start position of the new segment is the end of the latest unravelled segment

    Args:
        filename (str): the neuron to unravel
        window_half_length (int): the number of segments that defines half of the sliding window

    Returns:
        a tuple (morphio.mut.Morphology, dict) where first item is the unravelled
            morphology and the second one is the mapping of each point coordinate
            before and after unravelling
    '''
    morph = morphio.Morphology(filename)
    new_morph = morphio.mut.Morphology(morph)  # pylint: disable=no-member

    coord_before = np.empty([0, 3])
    coord_after = np.empty([0, 3])

    for sec, new_section in zip(morph.iter(), new_morph.iter()):
        _unravel_section(sec, new_section, window_half_length)

        coord_before = np.append(coord_before, sec.points, axis=0)
        coord_after = np.append(coord_after, new_section.points, axis=0)

    mapping = pd.DataFrame({
        'x0': coord_before[:, 0],
        'y0': coord_before[:, 1],
        'z0': coord_before[:, 2],
        'x1': coord_after[:, 0],
        'y1': coord_after[:, 1],
        'z1': coord_after[:, 2],
    })

    L.info('Unravel successful for file: %s', filename)
    return new_morph, mapping


def unravel_plane(input_plane, mapping):
    '''Return a new CutPlane object where the cut-leaves
    position has been updated after unravelling'''
    plane = CutPlane.from_json(input_plane)
    leaves = plane.cut_leaves_coordinates
    t = cKDTree(mapping[['x0', 'y0', 'z0']])
    distances, indices = t.query(leaves)
    not_matching_leaves = np.where(distances > 1e-5)[0]
    if not_matching_leaves.size:
        raise Exception('Cannot find the following leaves in the mapping:\n{}'.format(
                        leaves[not_matching_leaves]))
    plane.cut_leaves_coordinates = mapping.iloc[indices][['x1', 'y1', 'z1']].values
    return plane


def unravel_all(raw_dir, unravelled_dir,
                raw_planes_dir,
                unravelled_planes_dir,
                window_half_length=DEFAULT_WINDOW_HALF_LENGTH):
    '''Repair all morphologies in input folder
    '''
    if not os.path.exists(raw_planes_dir):
        raise Exception('{} does not exist'.format(raw_planes_dir))

    if not os.path.exists(unravelled_planes_dir):
        os.mkdir(unravelled_planes_dir)

    for inputfilename in iter_morphology_files(raw_dir):
        L.info('Unravelling: %s', inputfilename)
        outfilename = Path(unravelled_dir, inputfilename.name)
        raw_plane = str(Path(raw_planes_dir, inputfilename.name).with_suffix('.json'))
        unravelled_plane = Path(unravelled_planes_dir, inputfilename.name).with_suffix('.json')

        try:
            neuron, mapping = unravel(str(inputfilename), window_half_length)
            neuron.write(str(outfilename))
            with open(str(unravelled_plane), 'w') as f:
                json.dump(unravel_plane(raw_plane, mapping).to_json(), f, cls=RepairJSON)

        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning('Unravelling %s failed', f)
            L.warning(e, exc_info=True)
