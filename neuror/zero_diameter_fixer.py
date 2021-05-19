'''Fix zero diameters

Re-implementation of: https://bbpcode.epfl.ch/source/xref/sim/MUK/apps/Fix_Zero_Diameter.cpp
'''
import sys
from collections import namedtuple

SMALL = 0.0001

Point = namedtuple('Point',
                   ['section',
                    'point_id'])  # index of the point within the given section


def fix_neurite(root_section):
    '''Apply all fixes to a neurite'''
    point = Point(root_section, 0)
    fix_from_downstream(point)
    fix_in_between(point, stack=list())
    fix_from_upstream(point, 0)


def next_section_points(point):
    '''Yields next point along the section or the starting points
    of the child sections

    This function mimics the traversal of the linked list returned by recurseSection() at
    https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#40

    It is what enables this reimplementation to use the same algorithm as the original code
    '''
    size = len(point.section.points)
    if point.point_id < size - 1:
        yield Point(point.section, point.point_id + 1)
    else:
        for child in point.section.children:
            yield Point(child, 0)


def next_section_points_upstream(point):
    '''Yield upstream points until reaching the root'''
    section, point_id = point
    while not (section.is_root and point_id == 0):
        if point_id > 0:
            point_id -= 1
        else:
            section = section.parent
            assert len(section.points)
            point_id = len(section.points) - 1
        yield Point(section, point_id)


def set_diameter(point, new_diameter):
    '''Set a given diameter

    Unfortunately morphio does not support single value assignment
    so one has to update the diameters for the whole sections
    '''
    diameters = point.section.diameters
    diameters[point.point_id] = new_diameter
    point.section.diameters = diameters


def fix_from_downstream(point):
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
    '''
    diameter = point.section.diameters[point.point_id]

    is_leaf = not list(next_section_points(point))
    if diameter > SMALL or is_leaf:
        return diameter

    biggest_diameter_downstream = max(fix_from_downstream(next_point)
                                      for next_point in next_section_points(point))

    set_diameter(point, biggest_diameter_downstream)
    return biggest_diameter_downstream


def fix_from_upstream(point, upstream_good_diameter):
    '''Re-implementation of recursePushFix() available at
    https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#94

    Reset the diameter to upstream_good_diameter if the current value and all child values
    are below threshold

    Args:
        point: the current point
        upstream_good_diameter: the diameter value coming from upstream that will be used if
            the current diameter is not suitable

    Returns:
        The current diameter if above threshold, else the smallest child value above threshold
        else (if no child value is above threshold) returns 0
    '''
    section, point_idx = point
    diameter = section.diameters[point_idx]
    if diameter > SMALL:
        for next_point in next_section_points(point):
            fix_from_upstream(next_point, diameter)
        return diameter

    next_diameters = [fix_from_upstream(next_point, upstream_good_diameter)
                      for next_point in next_section_points(point)]

    smallest = min(next_diameters) if next_diameters else 0

    if smallest == 0:
        set_diameter(point, upstream_good_diameter)
    return smallest


def fix_in_between(point, stack):
    '''Re-implementation of
    https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#162

    Fix diameters between two points with valid diameters by applying a ramp

    Args:
        point: the current points
        stack: the stack of upstream points
    '''
    stack.append(point)
    # pylint: disable=chained-comparison
    # Condition: current point is good and previous point has a zero diameter
    if(len(stack) > 1 and
       stack[-2].section.diameters[stack[-2].point_id] <= SMALL and
       stack[-1].section.diameters[stack[-1].point_id] > SMALL):
        count = 0
        for next_point in next_section_points_upstream(point):
            count += 1
            if next_point.section.diameters[next_point.point_id] > SMALL:
                connect_average(point, next_point, count)
                break

    for next_point in next_section_points(point):
        fix_in_between(next_point, stack)
    stack.pop()


def connect_average(start_point, stop_point, count):
    '''Apply a ramp diameter between the two points

    Re-implementation of https://bbpcode.epfl.ch/source/xref/sim/MUK/muk/Zero_Diameter_Fixer.cpp#232
    '''
    start_sec, start_idx = start_point
    stop_sec, stop_idx = stop_point
    start_diam = start_sec.diameters[start_idx]
    stop_diam = stop_sec.diameters[stop_idx]

    step_diam = (stop_diam - start_diam) / count
    for step_num, point in enumerate(next_section_points_upstream(start_point), 1):
        if point == stop_point:
            break
        set_diameter(point, start_diam + step_diam * step_num)


def fix_zero_diameters(neuron):
    '''Fix zero diameters'''
    old_limit = sys.getrecursionlimit()
    point_counts = sum(len(section.points) for section in neuron.iter())
    sys.setrecursionlimit(max(old_limit, point_counts))
    try:
        for root in neuron.root_sections:
            fix_neurite(root)
    finally:
        sys.setrecursionlimit(old_limit)
