'''Utils module'''
import logging
import json

import numpy as np

from neurom import NeuriteType

L = logging.getLogger('neuror')


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def rotation_matrix(axis, theta):  # pylint: disable=too-many-locals
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class RepairJSON(json.JSONEncoder):
    '''JSON encoder that handles numpy types

    In python3, numpy.dtypes don't serialize to correctly, so a custom converter
    is needed.
    '''

    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, NeuriteType):
            return int(o)
        return json.JSONEncoder.default(self, o)


def direction(section):
    '''Return the direction vector of a section

    Args:
        section (morphio.mut.Section): section
    '''
    return np.diff(section.points[[0, -1]], axis=0)[0]


def section_length(section):
    '''Section length

    Args:
        section (morphio.mut.Section): section
    '''
    return np.linalg.norm(direction(section))
