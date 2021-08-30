'''Fix zero diameters

Re-implementation of: https://bbpcode.epfl.ch/source/xref/sim/MUK/apps/Fix_Zero_Diameter.cpp
with sections recursion instead of point recursion
'''
from collections import namedtuple
import numpy as np

SMALL = 0.0001

Point = namedtuple('Point',
                   ['section',
                    'point_id'])  # index of the point within the given section


def _next_point_upstream(point):
    '''Yield upstream points until reaching the root'''
    section, point_id = point
    while not (section.is_root and point_id == 0):
        if point_id > 0:
            point_id -= 1
        else:
            section = section.parent
            point_id = len(section.points) - 1
        yield Point(section, point_id)


def _get_point_diameter(point):
    return point.section.diameters[point.point_id]


def _set_point_diameter(point, new_diameter):
    '''Set a given diameter

    Unfortunately morphio does not support single value assignment
    so one has to update the diameters for the whole sections
    '''
    diameters = point.section.diameters
    diameters[point.point_id] = new_diameter
    point.section.diameters = diameters


def _connect_average_legacy(from_point):
    '''Apply a ramp diameter between the two points

    Re-implementation of https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#232
    '''
    count = 0
    next_point = None
    for next_point in _next_point_upstream(from_point):
        count += 1
        if _get_point_diameter(next_point) > SMALL:
            break
    if next_point is None or count <= 1:
        return

    from_diam = _get_point_diameter(from_point)
    to_diam = _get_point_diameter(next_point)
    step_diam = (to_diam - from_diam) / count
    for step_num, point in enumerate(_next_point_upstream(from_point), 1):
        if _get_point_diameter(point) <= SMALL:
            _set_point_diameter(point, from_diam + step_diam * step_num)


def _connect_average(from_point):
    '''Apply a ramp diameter between the two points

    Re-implementation of https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#232
    Contrary to the previous implementation the diameter the ramp is computed in term of
    pathlength and no longer in term of point number
    '''
    prev_point = from_point
    pathlengths = []
    next_point = None
    for next_point in _next_point_upstream(from_point):
        pathlengths.append(np.linalg.norm(prev_point.section.points[prev_point.point_id] -
                                          next_point.section.points[next_point.point_id]))
        if _get_point_diameter(next_point) > SMALL:
            break
        prev_point = next_point
    if next_point is None or len(pathlengths) <= 1:
        return

    from_diam = _get_point_diameter(from_point)
    to_diam = _get_point_diameter(next_point)
    cumulative_pathlengths = np.cumsum(pathlengths)
    pathlength_fractions = cumulative_pathlengths / cumulative_pathlengths[-1]
    for point, fraction in zip(_next_point_upstream(from_point),
                               pathlength_fractions[:-1]):
        _set_point_diameter(point, from_diam + (to_diam - from_diam) * fraction)


def _fix_downstream(section):
    '''Re-implementation of recursePullFix() available at
    https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#66

    If the current diameter is below the threshold, change its value to the biggest value
    among the 1-degree children downstream diameters

    This is recursive only until a diameter above threshold is found so this
    fixes zero diameters that are located at the beginning of the neurite

    Fixes this:
    Soma--0--0--0--1--1--1
    But not this:
    Soma--1--1--1--0--0--0--1--1

    Args:
        section(morphio.Section):
    '''
    nonzero_indices = np.asarray(section.diameters > SMALL).nonzero()[0]
    if len(nonzero_indices) > 0:
        idx = nonzero_indices[0]
        diameter = section.diameters[idx]
        section.diameters = np.concatenate((np.repeat(diameter, idx), section.diameters[idx:]))
        return diameter
    else:
        child_diameters = [_fix_downstream(child) for child in section.children]
        max_child_diameter = max(child_diameters) if child_diameters else 0
        if max_child_diameter > SMALL:
            section.diameters = np.repeat(max_child_diameter, len(section.diameters))
        return max_child_diameter


def _fix_in_between(section, stack, legacy):
    '''Re-implementation of
    https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#162

    Fix diameters between two points with valid diameters by applying a ramp

    Args:
        point: the current points
        stack: the stack of upstream points
        legacy: whether to use legacy algorithm that didn't account for path lengths
    '''
    _connect = _connect_average
    if legacy:
        _connect = _connect_average_legacy

    nonzero_indices = np.asarray(section.diameters > SMALL).nonzero()[0]
    if len(nonzero_indices) > 0:
        if len(stack) > 0:
            _connect(Point(section, nonzero_indices[0]))
        # account for zero diameters inside `section`
        for i in range(1, len(nonzero_indices)):
            if nonzero_indices[i] - nonzero_indices[i - 1] > 1:
                _connect(Point(section, nonzero_indices[i]))
        stack.append(Point(section, nonzero_indices[-1]))
    for child in section.children:
        _fix_in_between(child, stack, legacy)
    if len(nonzero_indices) > 0:
        stack.pop()


def _fix_upstream(section, upstream_good_diameter):
    '''Re-implementation of recursePushFix() available at
    https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#94

    Reset the diameter to upstream_good_diameter if the current value and all child values
    are below threshold

    Args:
        section: the current section
        upstream_good_diameter: the diameter value coming from upstream that will be used if
            the current diameter is not suitable

    Returns:
        The current diameter if above threshold, else the smallest child value above threshold
        else (if no child value is above threshold) returns 0
    '''
    diameters = section.diameters
    nonzero_indices = np.asarray(diameters > SMALL).nonzero()[0]
    last_nonzero_idx = nonzero_indices[-1] if len(nonzero_indices) > 0 else -1
    if last_nonzero_idx >= 0:
        upstream_good_diameter = diameters[last_nonzero_idx]
    child_diameters = [_fix_upstream(child, upstream_good_diameter) for child in section.children]
    max_child_diameter = max(child_diameters) if child_diameters else 0
    if max_child_diameter < SMALL:
        section.diameters = np.concatenate((
            diameters[:last_nonzero_idx + 1],
            np.repeat(upstream_good_diameter, len(diameters) - (last_nonzero_idx + 1))
        ))
    return max_child_diameter


def fix_neurite(root_section, legacy=False):
    '''Apply all fixes to a neurite'''
    _fix_downstream(root_section)
    _fix_in_between(root_section, [], legacy)
    _fix_upstream(root_section, 0)


def fix_zero_diameters(neuron, legacy=False):
    '''Fix zero diameters'''
    for root in neuron.root_sections:
        fix_neurite(root, legacy)
