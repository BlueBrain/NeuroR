'''The axon repair module'''
import logging

import morphio
import numpy as np
from morph_tool.transform import align, translate
from neurom import COLS
from neurom.features.section import branch_order, strahler_order

from neuror.utils import section_length

L = logging.getLogger('neuror')


def _tree_distance(sec1, sec2):
    '''
    Returns the number of sections between the 2 sections

    Reimplementation of:
    https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_axon.cpp#n35

    raises: if both sections are not part of the same neurite

    Note:
    I think the implementation of tree distance is true to the original
    but I would expect the tree distance of 2 children with the same parent to be 2 and not 1
    Because in the current case, (root, child1) and (child1, child2) have the
    same tree distance and it should probably not be the case
    '''
    original_sections = (sec1, sec2)
    dist = 0
    while True:
        diff = branch_order(sec1) - branch_order(sec2)
        if diff == 0:
            break

        if diff > 0:
            sec1 = sec1.parent
            dist += 1
        else:
            sec2 = sec2.parent
            dist += 1

    if sec1.id == sec2.id:
        return dist

    dist -= 1
    while sec1.id != sec2.id:
        sec1 = sec1.parent
        sec2 = sec2.parent
        dist += 2
        if None in {sec1, sec2}:
            raise Exception('Sections {} and {} are not part of the same neurite'.format(
                original_sections[0], original_sections[1]))

    return dist


def _downstream_pathlength(section):
    '''The sum of this section and its descendents's pathlengths

    Reimplementation of the C++ function "children_length":

    https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/morphstats.cpp#n112
    '''
    ret = section_length(section)
    for child in section.children:
        ret += _downstream_pathlength(child)
    return ret


def _similar_section(intact_axons, section):
    '''Use the "mirror" technique of BlueRepairSDK to find out the similar section

    https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_axon.cpp#n83

    Note:
    I have *absolutely* no clue why sorting by this metric
    '''
    dists = []
    for root in intact_axons:
        origin = root.points[0]
        origin_cut = section.points[0]
        diff = origin_cut - origin
        diff[COLS.X] = origin_cut[COLS.X] + origin[COLS.X]
        dists.append(np.linalg.norm(diff))

    return intact_axons[np.argmin(dists)]


def _sort_intact_sections_by_score(section, similar_section, axon_branches):
    '''Returns an array of sections sorted by their score'''
    reference = _downstream_pathlength(similar_section) - section_length(section)

    def score(branch):
        '''The score. The interpretation is something like the absolute difference in
        remaining children length'''
        return -abs(reference - _downstream_pathlength(branch))
    return sorted(axon_branches, key=score)


def repair(morphology, section, intact_sections, axon_branches, used_axon_branches, y_extent):
    '''Axonal repair

    Reimplementation of:
    https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/repair.cpp#n727

    1) Find the most similar section in INTACT_SECTIONS list to SECTION
    2) Sort AXON_BRANCHES according to a similarity score to the section found at step 1
    3) Loop through the sorted AXON_BRANCHES to find a section with same strahler orders
       and that, when appended, does not extend further than Y_EXTENT
    4) Append the first section that meets the conditions of step 3)
    5) Mark this section as used and do not re-use it

    Args:
        morphology (neurom.Neuron): the morphology to repair
        section (neurom.core._neuron.Section): the section to repair
        intact_sections (List[Section]): a list of all sections from this morphology
            that are part of an intact subtree. Note: these section won't be grafted
        axon_branches (List[Section]): a list a intact sections coming from donor morphologies
            These are the sections that will be appended

    ..note::
       The original code used to have more parameters. In the context of the bbp-morphology-workflow
       it seems that some of the parameters were always used with the same value. This
       reimplementation assumes the following BlueRepairSDK options:

        - --overlap=true
        - --incremental=false
        - --restrict=true
        - --distmethod=mirror'
    '''

    if not intact_sections:
        L.debug("No intact axon found. Not repairing!")
        return

    similar = _similar_section(intact_sections, section)
    branch_pool = _sort_intact_sections_by_score(section, similar, axon_branches)

    strahler_orders = {intact_section: strahler_order(intact_section)
                       for intact_section in intact_sections + branch_pool}

    L.debug('Branch pool count: %s', len(branch_pool))
    for branch in branch_pool:
        if (branch in used_axon_branches or strahler_orders[similar] != strahler_orders[branch]):
            continue

        L.debug("Pasting axon branch with ID %s", branch.id)

        end_point = section.points[-1, COLS.XYZ]
        appended = section.append_section(branch)
        translation = section.points[-1, COLS.XYZ] - appended.points[0, COLS.XYZ]
        align(appended, translation)
        translate(appended, translation)

        # Make sure the child section first point is exactly the end point of the parent section
        appended_points = np.copy(appended.points)
        appended_points[0] = end_point
        appended.points = appended_points

        if any(np.any(section.points[:, COLS.Y] > y_extent) for section in appended.iter()):
            L.debug("Discarded, exceeds y-limit")
            morphology.delete_section(appended)
        else:
            L.debug('Section appended')
            used_axon_branches.add(branch)
            return

    morphio.set_ignored_warning(morphio.Warning.wrong_duplicate, False)
