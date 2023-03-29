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
from neurom.morphmath import interval_lengths
from scipy.spatial import cKDTree

from neuror.cut_plane import CutPlane
from neuror.exceptions import NeuroRError
from neuror.utils import RepairJSON

L = logging.getLogger('neuror')

DEFAULT_WINDOW_HALF_LENGTH = 5
DEFAULT_WINDOW_HALF_LENGTH_PATH = 8


def _get_principal_direction(points):
    '''Return the principal direction of a point cloud
    It is the eigen vector of the covariance matrix with the highest eigen value'''

    X = np.copy(np.asarray(points))
    X -= np.mean(X, axis=0)
    C = np.dot(X.T, X)
    w, v = np.linalg.eigh(C)
    return v[:, w.argmax()]


def _unravel_section(section, window_half_length, soma, legacy_behavior, use_path_length):
    '''Unravel a section using number of adjacent points as window_half_length'''
    # pylint: disable=too-many-locals
    points = section.points
    if legacy_behavior and section.is_root and len(soma.points) > 1:
        points = np.vstack((soma.points[0], points))
    point_count = len(points)
    if section.is_root:
        if legacy_behavior and len(soma.points) > 1:
            unravelled_points = [soma.points[0]]
        else:
            unravelled_points = [section.points[0]]
    else:
        unravelled_points = [section.parent.points[-1]]

    for window_center in range(1, point_count):
        if use_path_length:
            intervals = interval_lengths(points)
            path_lengths = np.insert(np.cumsum(intervals), 0, 0)
            path_lengths -= path_lengths[window_center]

            # if no points in the window, we skip unravel
            if len(path_lengths[abs(path_lengths) < window_half_length]) == 1:
                unravelled_points.append(points[window_center])
                continue

            window_start = np.argmin(abs(path_lengths + window_half_length))
            window_end = np.argmin(abs(path_lengths - window_half_length)) + 1
        else:
            window_start = int(max(0, window_center - window_half_length - 1))
            window_end = int(min(point_count, window_center + window_half_length + 1))

        direction = _get_principal_direction(points[window_start:window_end])

        segment = points[window_center] - points[window_center - 1]
        window_direction = points[window_end - 1] - points[window_start]

        # make it span length the same as the original window_direction within the window
        direction *= np.linalg.norm(segment)

        scalar_product = np.dot(window_direction, direction)
        # point it in the same direction as the window
        direction *= np.sign(scalar_product or 1.)

        point = direction + unravelled_points[-1]
        unravelled_points.append(point)

    section.points = unravelled_points
    if legacy_behavior and section.is_root and len(soma.points) > 1:
        section.diameters = np.hstack((soma.diameters[0], section.diameters))


def unravel(filename, window_half_length=None,
            legacy_behavior=False, use_path_length=True):
    '''Return an unravelled neuron

    Segment are unravelled iteratively
    Each segment direction is replaced by the averaged direction in a sliding window
    around this segment, preserving the original segment length.
    The start position of the new segment is the end of the latest unravelled segment

    Based initially on:
    DOI: 10.7551/mitpress/9780262013277.001.0001
    Section: 9.2 Repair of Neuronal Dendrites

    Args:
        filename (str): the neuron to unravel
        window_half_length (int): path length that defines half of the sliding window
        legacy_behavior (bool): if yes, when the soma has more than one point, the first point of
            the soma is appended to the start of each neurite.
        use_path_length (bool): if False, the argument window_half_length will be recasted to an int
            and correspond to number of points on each side of the window.

    Returns:
        a tuple (morphio.mut.Morphology, dict) where first item is the unravelled
            morphology and the second one is the mapping of each point coordinate
            before and after unravelling
    '''
    morph = morphio.Morphology(filename, options=morphio.Option.nrn_order)
    new_morph = morphio.mut.Morphology(morph, options=morphio.Option.nrn_order)  # noqa, pylint: disable=no-member

    if window_half_length is None:
        window_half_length = DEFAULT_WINDOW_HALF_LENGTH_PATH \
            if use_path_length else DEFAULT_WINDOW_HALF_LENGTH

    coord_before = np.empty([0, 3])
    coord_after = np.empty([0, 3])
    for sec, new_section in zip(morph.iter(), new_morph.iter()):
        _unravel_section(
            new_section, window_half_length, morph.soma, legacy_behavior, use_path_length
        )
        coord_before = np.append(coord_before, sec.points, axis=0)
        coord_after = np.append(coord_after, new_section.points, axis=0)

        if legacy_behavior and len(morph.soma.points) > 1:  # pylint: disable=no-member
            coord_before = np.vstack((morph.soma.points[0], coord_before))  # noqa, pylint: disable=no-member
        mapping = pd.DataFrame({
            'x0': coord_before[:, 0],
            'y0': coord_before[:, 1],
            'z0': coord_before[:, 2],
            'x1': coord_after[:, 0],
            'y1': coord_after[:, 1],
            'z1': coord_after[:, 2],
        })

    L.debug('Unravel successful for file: %s', filename)
    return new_morph, mapping


def unravel_plane(plane, mapping):
    '''Return a new CutPlane object where the cut-leaves
    position has been updated after unravelling'''
    leaves = plane.cut_leaves_coordinates

    if not np.any(leaves):
        return plane

    t = cKDTree(mapping[['x0', 'y0', 'z0']])
    distances, indices = t.query(leaves)
    not_matching_leaves = np.where(distances > 1e-5)[0]
    if not_matching_leaves.size:
        raise NeuroRError('Cannot find the following leaves in the mapping:\n' +
                          str(leaves[not_matching_leaves]))
    plane.cut_leaves_coordinates = mapping.iloc[indices][['x1', 'y1', 'z1']].values
    return plane


def unravel_all(raw_dir, unravelled_dir,
                raw_planes_dir,
                unravelled_planes_dir,
                window_half_length=None):
    '''Repair all morphologies in input folder
    '''
    if not os.path.exists(raw_planes_dir):
        raise NeuroRError(f'{raw_planes_dir} does not exist')

    if not os.path.exists(unravelled_planes_dir):
        os.mkdir(unravelled_planes_dir)

    for inputfilename in iter_morphology_files(raw_dir):
        L.debug('Unravelling: %s', inputfilename)
        outfilename = Path(unravelled_dir, inputfilename.name)
        raw_plane = CutPlane.from_json(
            Path(raw_planes_dir, inputfilename.name).with_suffix('.json'))
        unravelled_plane = Path(unravelled_planes_dir, inputfilename.name).with_suffix('.json')

        try:
            neuron, mapping = unravel(str(inputfilename), window_half_length)
            neuron.write(str(outfilename))
            with open(str(unravelled_plane), 'w') as f:
                json.dump(unravel_plane(raw_plane, mapping).to_json(), f, cls=RepairJSON)

        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning('Unravelling %s failed', f)
            L.warning(e, exc_info=True)
