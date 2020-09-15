'''Module for the legacy cut plane detection.

As implemented in:
https://bbpcode.epfl.ch/source/xref/platform/BlueRepairSDK/BlueRepairSDK/src/repair.cpp#263
'''
import logging

from neurom import iter_sections
from neurom.core import Tree
from neurom.core.dataformat import COLS

L = logging.getLogger(__name__)


def cut_detect(neuron, axis):
    '''Detect the cut leaves the old way

    The cut leaves are simply the leaves that live
    on the half-space (split along the 'axis' coordinate)
    with the biggest number of leaves
    '''
    # In the original code, offset was a function argument
    # but it was always used with offset = 0
    offset = 0

    count_plus = count_minus = sum_plus = sum_minus = 0

    axis = {'x': COLS.X, 'y': COLS.Y, 'z': COLS.Z}[axis.lower()]

    for leaf in iter_sections(neuron, iterator_type=Tree.ileaf):
        coord = leaf.points[-1, axis]
        if coord > offset:
            count_plus += 1
            sum_plus += coord
        else:
            count_minus += 1
            sum_minus += coord

    if count_plus == 0 or count_minus == 0:
        raise Exception("cut detection warning:one of the sides is empty. can't decide on cut side")

    if -sum_minus / count_minus > sum_plus / count_plus:
        sign = 1
    else:
        sign = -1

    cut_leaves = list()
    for leaf in iter_sections(neuron, iterator_type=Tree.ileaf):
        coord = leaf.points[-1, axis]

        if coord * sign > offset:
            cut_leaves.append(coord)

    return cut_leaves, sign
