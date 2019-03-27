'''The unravel module

Unravelling is the action of "stretching" the cell that
has been shrunk because of the dehydratation caused by the slicing'''
import numpy as np

import morphio


def _get_principal_direction(points):
    '''Return the principal direction of a point cloud
    It is the eigen vector of the covariance matrix with the highest eigen value'''

    X = np.copy(np.asarray(points))
    X -= np.mean(X, axis=0)
    C = np.dot(X.T, X)
    w, v = np.linalg.eig(C)
    return v[:, w.argmax()]


def unravel(filename, window_half_length=5):
    '''Return an unravelled neuron

    Segment are unravelled iteratively
    Each segment direction is replaced by the averaged direction in a sliding window
    around this segment. And the original segment length is preserved.
    The start position of the new segment is the end of the latest unravelled segment

    Args:
        filename (str): the neuron to unravel
        window_half_length (int): the number of segments that defines half of the sliding window

    Returns:
        morphio.mut.Morphology
    '''
    morph = morphio.Morphology(filename)
    new_morph = morphio.mut.Morphology(morph)

    for sec, new_section in zip(morph.iter(), new_morph.iter()):
        points = sec.points
        point_count = len(points)
        unravelled_points = [sec.points[0]]

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
    return new_morph
