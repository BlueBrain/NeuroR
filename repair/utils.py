'''Utils module'''
import logging
import json

import h5py
import numpy as np

from neurom import NeuriteType
from neurom.features.sectionfunc import branch_order

L = logging.getLogger('repair')


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


def read_apical_points(filename, neuron_old, neuron_new):
    '''Read apical section ID and point ID from file

    Returns a 4-tuple with:
    - neurom v2 section id (ie. shifted by -1)
    - apical section branch order
    - point ID in new morphology
    - point ID in old morphology
    '''
    with h5py.File(filename) as f:
        group = f['neuron1']
        try:
            section_id = group.attrs['apical'][0, 0]
            # Neurom v2 does not have a soma section so section ids are shifted!
            neurom_v2_section_id = section_id - 1
            apical_point_new = len(neuron_new.section(neurom_v2_section_id).points) - 1
            apical_point_old = len(neuron_old.section(neurom_v2_section_id).points) - 1
            apical_point_order = branch_order(neuron_new.sections[section_id])
            return neurom_v2_section_id, apical_point_order, apical_point_new, apical_point_old

        except (KeyError, IndexError):
            L.warning("Could not load apical point from file: %s", filename)
            return -1, 0, [0, 0, 0], [0, 0, 0]
