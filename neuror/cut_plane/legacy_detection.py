'''Module for the legacy cut plane detection.

As implemented in:
https://bbpgitlab.epfl.ch/nse/morphologyrepair/BlueRepairSDK/-/blob/main/BlueRepairSDK/src/repair.cpp#L263
'''
import logging
from collections import defaultdict

import numpy as np
from morph_tool import apical_point_section_segment
from neurom import iter_sections
from neurom.core import Section
from neurom.core.dataformat import COLS

from neuror.exceptions import NeuroRError
from neuror.utils import RepairType, repair_type_map

L = logging.getLogger(__name__)


def children_ids(section):
    '''
    https://bbpgitlab.epfl.ch/nse/morphologyrepair/BlueRepairSDK/-/blob/main/BlueRepairSDK/src/helper_dendrite.cpp#L111

    The original code returns the ids of the descendant sections
    but this implementation return the Section objects instead.
    '''
    return list(section.ipreorder())


def cut_detect(neuron, cut, offset, axis):
    '''Detect the cut leaves the old way

    The cut leaves are simply the leaves that live
    on the half-space (split along the 'axis' coordinate)
    with the biggest number of leaves
    '''
    count_plus = count_minus = sum_plus = sum_minus = 0

    for leaf in iter_sections(neuron, iterator_type=Section.ileaf):
        coord = leaf.points[-1, axis]
        if coord > offset:
            count_plus += 1
            sum_plus += coord
        else:
            count_minus += 1
            sum_minus += coord

    if count_plus == 0 or count_minus == 0:
        # pylint: disable=broad-exception-raised
        raise NeuroRError(
            "cut detection warning:one of the sides is empty. can't decide on cut side"
        )

    if -sum_minus / count_minus > sum_plus / count_plus:
        sign = 1
    else:
        sign = -1

    for leaf in iter_sections(neuron, iterator_type=Section.ileaf):
        if leaf.points[-1, axis] * sign > offset:
            cut[leaf] = True

    return sign


def internal_cut_detection(neuron, axis):
    '''As in:

    https://bbpgitlab.epfl.ch/nse/morphologyrepair/BlueRepairSDK/-/blob/main/BlueRepairSDK/src/repair.cpp#L263

    Use cut_detect to get the side of the half space the points live in.
    Then mark points which are children of the apical section.

'''
    axis = {'x': COLS.X, 'y': COLS.Y, 'z': COLS.Z}[axis.lower()]

    cut = defaultdict(lambda key: False)
    side = cut_detect(neuron, cut, 0, axis)

    # reclassify cut points in tuft,based on apical point position
    apical_section_id, point_id = apical_point_section_segment(neuron)
    if apical_section_id is not None:
        apical_section = neuron.sections[apical_section_id]
        apical_offset = apical_section.points[point_id, axis]
        cut_mark(children_ids(apical_section), cut, apical_offset, side, axis)
    else:
        apical_section = None

    extended_types = repair_type_map(neuron, apical_section)
    oblique_roots = get_obliques(neuron, extended_types)

    # reclassify points in obliques. based on the position of their root.
    for root in oblique_roots:
        offset = root.points[0]

        # FIXME: z is hard coded in the original code as well. It's probably a bug.
        cut_mark(root.children, cut, offset[COLS.Z], side, axis)

    cut_leaves = np.array([sec.points[-1, COLS.XYZ] for sec, is_cut in cut.items() if is_cut])

    return cut_leaves, side


def get_obliques(neuron, extended_types):
    '''
    Returns the oblique roots.

    https://bbpgitlab.epfl.ch/nse/morphologyrepair/BlueRepairSDK/-/blob/main/BlueRepairSDK/src/helper_dendrite.cpp#L212
    '''
    return [section for section in iter_sections(neuron)
            if (extended_types[section] == RepairType.oblique and
                (section.parent is None or extended_types[section.parent] == RepairType.trunk))]


def cut_mark(sections, cut, offset, side, axis):
    '''
    https://bbpgitlab.epfl.ch/nse/morphologyrepair/BlueRepairSDK/-/blob/main/BlueRepairSDK/src/helper_dendrite.cpp#L654
    '''
    for sec in sections:
        if sec.children:
            cut[sec] = False
            continue

        r = sec.points[-1, axis]

        growing_back = False
        mysec = sec

        while mysec.parent:
            for point in mysec.points:
                mr = point[axis]
                growing_back |= (mr - (r + (float(side) * 15.0))) * side > 0
            if growing_back:
                break
            mysec = mysec.parent

        cut[sec] = (r - offset) * side > 0 and not growing_back
    return cut
