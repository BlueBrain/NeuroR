'''Dendritic repair module

It is based on the BlueRepairSDK's implementation
'''
import logging
import os
from collections import defaultdict, Counter
from enum import Enum
from itertools import chain, tee

import numpy as np
from scipy.spatial.distance import cdist

import morphio
from cut_plane.detection import find_cut_plane
import neurom as nm
from neurom import (NeuriteType, iter_neurites, iter_sections, iter_segments,
                    load_neuron)
from neurom.core.dataformat import COLS
from neurom.features.sectionfunc import branch_order, section_path_length
from repair.utils import angle_between, rotation_matrix

L = logging.getLogger('repair')
SEG_LENGTH = 5.0
SHOLL_LAYER_SIZE = 10
BIFURCATION_BOOSTER = 0
NOISE_CONTINUATION = 0.7
SOMA_REPULSION = 0.7
BIFURCATION_ANGLE = 0

EPSILON = 1e-8


class Action(Enum):
    '''To bifurcate or not to bifurcate ?'''
    BIFURCATION = 1
    CONTINUATION = 2
    TERMINATION = 3


def is_cut_section(section, cut_points):
    '''Return true if the section is close from the cut plane'''
    return np.min(cdist(section.points[:, COLS.XYZ], cut_points)) < EPSILON


def is_neurite_intact(neurite, cut_points):
    '''Does the neurite have leaves belonging to the cut plane ?'''
    return all(not is_cut_section(section, cut_points)
               for section in iter_sections(neurite))


def find_intact_sub_trees(morph, cut_points):
    '''Returns intact neurites'''
    return [neurite for neurite in iter_neurites(morph)
            if is_neurite_intact(neurite, cut_points)]


def load_cut_plane_data(inputfile):
    '''Return cut plane'''
    width = 15
    return find_cut_plane(load_neuron(inputfile), width)


def direction(section):
    '''Return the direction vector of a section'''
    return np.diff(section.points[[0, -1]][:, COLS.XYZ], axis=0)[0]


def _section_length(section):
    '''Section length'''
    return np.linalg.norm(direction(section))


def branching_angles(section):
    '''Return a list of 2-tuples. The first element is the branching order and the second one is
    the angles between the direction of the section and its children's ones'''
    if _section_length(section) < EPSILON:
        return []
    res = []

    branching_order = branch_order(section)
    for child in section.children:
        if _section_length(child) < EPSILON:
            continue

        theta = np.math.acos(np.dot(direction(section), direction(child)) /
                             (_section_length(section) * _section_length(child)))
        res.append((branching_order, theta))
    return res


def intact_branching_angles(neurites):
    '''Returns lists of branching angles stored in a nested dict
    1st key: section type, 2nd key: branching order

    Args:
        neurites (List[Neurite])

    Returns:
        Dict[SectionType, Dict[int, List[int]]]: Branching angles'''
    res = defaultdict(lambda: defaultdict(list))
    for neurite in neurites:
        for order, angle in branching_angles(neurite.root_node):
            res[neurite.type][order].append(angle)
    return res


def best_case_angle_data(info, section_type, branching_order):
    '''Get the distribution of branching angles for this section type and
    branching order

    If no data are available, fallback on aggregate data'''
    angles = info['intact_branching_angles'][section_type]
    accurate_data = angles[branching_order]
    if accurate_data:
        return accurate_data

    return list(chain.from_iterable(angles.values()))


def compute_sholl_data(neurites, origin):
    '''Compute the number of termination, bifurcation and continuation section for each
    neurite type, sholl layer and shell order

    data[neurite_type][layer][order][action_type] = counts
    '''
    data = defaultdict(lambda: defaultdict(dict))
    sections = (section for neurite in neurites for section in iter_sections(neurite))
    for section in sections:
        order = branch_order(section)
        first_layer, last_layer = (np.linalg.norm(
            section.points[[0, -1], COLS.XYZ] - origin, axis=1) // SHOLL_LAYER_SIZE).astype(int)
        for layer in range(min(first_layer, last_layer), max(first_layer, last_layer) + 1):
            if order not in data[section.type][layer]:
                data[section.type][layer][order] = {Action.TERMINATION: 0,
                                                    Action.CONTINUATION: 0,
                                                    Action.BIFURCATION: 0}
            data[section.type][layer][order][Action.CONTINUATION] += 1

        data[section.type][last_layer][order][Action.BIFURCATION if section.children else
                                              Action.TERMINATION] += 1
    return data


def get_sholl_proba(sholl_data, section_type, sholl_layer, pseudo_order):
    '''Return the probabilities of bifurcation, termination and bifurcation
    in a dictionnary for the given sholl layer and branch order.

    If no data are available for this branch order, the action_counts
    are averaged on all branch orders for this sholl layer

    Args:
        sholl_data: nested dict that stores the number of section per SectionType, sholl order
            sholl layer and Action type
            sholl_data[neurite_type][layer][order][action_type] = counts
        section_type (SectionType): section type
        sholl_layer (int): sholl layer
        pseudo_order (int): pseudo order

    Returns:
        Dict[Action, float]: probability of each action
    '''

    section_type_data = sholl_data[section_type]
    try:
        data_layer = section_type_data[sholl_layer]
    except KeyError:
        return {Action.BIFURCATION: 0,
                Action.CONTINUATION: 0,
                Action.TERMINATION: 1}

    try:
        action_counts = data_layer[pseudo_order]
    except KeyError:
        action_counts = Counter()
        for data in data_layer.values():
            for action, count in data.items():
                action_counts[action] += count

    total_counts = sum(action_counts.values())
    if total_counts == 0:
        return {Action.BIFURCATION: 0,
                Action.CONTINUATION: 0,
                Action.TERMINATION: 1}

    boost_bifurcation = int(total_counts * BIFURCATION_BOOSTER)
    action_counts[Action.BIFURCATION] += boost_bifurcation
    total_counts += boost_bifurcation
    res = {action: count / float(total_counts) for action, count in action_counts.items()}
    return res


def get_sholl_layer(section, origin):
    '''Returns this section sholl layer'''
    return int(np.linalg.norm(section.points[-1, COLS.XYZ] - origin) / SHOLL_LAYER_SIZE)


def last_segment_vector(section, normalized=False):
    '''Returns the vector formed by the last 2 points of the section'''
    vec = section.points[-1, COLS.XYZ] - section.points[-2, COLS.XYZ]
    if normalized:
        return vec / np.linalg.norm(vec)
    return vec


def continuation(sec, origin):
    '''Continue growing the section'''
    # The following lines is from BlueRepairSDK's code but I'm not
    # convinced by its relevance
    # if first_sec:
    #     direction_ = sec.points[-1, COLS.XYZ] - sec.points[0, COLS.XYZ]
    # else:
    section_direction = last_segment_vector(sec)
    section_direction /= np.linalg.norm(section_direction)

    radial_direction = sec.points[0, COLS.XYZ] - origin
    length = np.linalg.norm(radial_direction)
    if length > EPSILON:
        radial_direction /= length
    else:
        radial_direction = sec.points[1, COLS.XYZ] - sec.points[0, COLS.XYZ]
        radial_direction /= np.linalg.norm(radial_direction)

    noise_direction = (2 * np.random.random(size=3) - 1)

    direction_ = section_direction + \
        SOMA_REPULSION * radial_direction + \
        NOISE_CONTINUATION * noise_direction

    direction_ /= np.linalg.norm(direction_)

    coord = sec.points[-1, COLS.XYZ] + direction_ * SEG_LENGTH
    radius = np.mean(sec.points[:, COLS.R])
    new_point = np.append(coord, radius)
    sec.points = np.vstack((sec.points, new_point))


def grow_until_sholl_sphere(section, origin, sholl_layer):
    '''Grow until reaching next sholl layer'''
    backwards_sections = 0
    while get_sholl_layer(section, origin) == sholl_layer and backwards_sections < 4:
        continuation(section, origin)

        # make sure we don't grow back the origin
        is_last_segment_toward_origin = np.dot(
            section.points[-1, COLS.XYZ] - origin,
            last_segment_vector(section)) < 0

        if is_last_segment_toward_origin:
            backwards_sections += 1
    return backwards_sections


def get_similar_radius(sections, radius):
    '''Return a non-leaf section with a similar radius'''
    threshold = 0.1

    for section in sections:
        if section.children and abs(section.points[-1, COLS.R] - radius) < threshold:
            return section
    L.warning("Oh no, no similar diameter found!")
    return None


def bifurcation(section, order_offset, info):
    '''Create 2 children at the end of the current section'''
    current_radius = section.points[-1, COLS.R]
    similar_section = get_similar_radius(info['dendritic_sections'], current_radius)

    if similar_section:
        child_radii = [2 * similar_section.children[0].points[0, COLS.R],
                       2 * similar_section.children[1].points[0, COLS.R]]
    else:
        child_radii = [section.points[-1, COLS.R] * 2,
                       section.points[-1, COLS.R] * 2]

    median_angle = np.median(
        best_case_angle_data(info, section.type, branch_order(section) - order_offset))

    last_segment_vec = last_segment_vector(section)
    orthogonal = np.cross(last_segment_vec, section.points[-1, COLS.XYZ])

    def shuffle_direction(direction_):
        '''Return the direction to which to grow a child section'''
        theta = median_angle + np.radians(np.random.random() * BIFURCATION_ANGLE)
        first_rotation = np.dot(rotation_matrix(orthogonal, theta), direction_)
        new_dir = np.dot(rotation_matrix(direction_, np.random.random() * np.pi * 2),
                         first_rotation)
        return new_dir / np.linalg.norm(new_dir)

    for child_radius in child_radii:
        child_start = section.points[-1, COLS.XYZ]
        points = [child_start.tolist(),
                  (child_start + shuffle_direction(last_segment_vec) * SEG_LENGTH).tolist()]
        prop = morphio.PointLevel(points, [current_radius, child_radius])

        child = section.append_section(prop)
        L.debug('section appended: %s', child.id)


def grow(section, info, first_sec, order_offset, origin):
    '''Grow main method

    Will either:
        - continue growing the section
        - create a bifurcation
        - terminate the growth
    '''
    sholl_layer = get_sholl_layer(section, origin)
    pseudo_order = branch_order(section) - order_offset
    if first_sec:
        continuation(section, origin)

    proba = get_sholl_proba(info['sholl'], section.type, sholl_layer, pseudo_order)
    action = np.random.choice(list(proba.keys()), p=list(proba.values()))

    if action == Action.CONTINUATION:
        L.debug('Continuing')
        backwards_sections = grow_until_sholl_sphere(section, origin, sholl_layer)

        if backwards_sections == 0:
            grow(section, info, False, order_offset, origin)

    elif action == Action.BIFURCATION:
        L.debug('Bifurcating')
        backwards_sections = grow_until_sholl_sphere(section, origin, sholl_layer)
        if backwards_sections == 0:
            bifurcation(section, order_offset, info)
            for child in section.children:
                grow(child, info, False, order_offset, origin)
    else:
        L.debug('Terminating')


def angle_between_segments(neuron):
    '''Returns a list of angles between successive segments'''
    angles = list()
    for section in iter_sections(neuron):
        iter_seg1, iter_seg2 = tee(iter_segments(section))
        next(iter_seg2)
        for (p0, p1), (_, p2) in zip(iter_seg1, iter_seg2):
            vec1 = (p1 - p0)[COLS.XYZ]
            vec2 = (p2 - p1)[COLS.XYZ]
            angles.append(angle_between(vec1, vec2))
    return angles


def compute_statistics(neuron, cut_plane, only_intact_neurites=True):
    '''Compute statistics'''
    if only_intact_neurites:
        neurites = find_intact_sub_trees(neuron, cut_plane)
    else:
        neurites = neuron.neurites

    dendritic_sections = list(iter_sections(neuron, neurite_filter=lambda n: n.type in {
                              nm.APICAL_DENDRITE, nm.BASAL_DENDRITE, }))
    data = {'intact_branching_angles': intact_branching_angles(neuron.neurites),
            'sholl': compute_sholl_data(neurites, neuron.soma.center),
            'angle_between_segments': angle_between_segments(neuron),
            'dendritic_sections': dendritic_sections}
    return data


def make_intact(filename, cut_points, outfilename):
    '''Clone neuron and remove non-intact neurites'''
    neuron = load_neuron(filename)
    for neurite in neuron.neurites:
        if not is_neurite_intact(neurite, cut_points):
            neuron.delete_section(neurite.morphio_root_node, recursive=True)
    neuron.write(outfilename)


def repair(inputfile, outputfile, seed=0):
    '''Repair the input morphology'''
    np.random.seed(seed)
    cut_plane_data = load_cut_plane_data(inputfile)
    if cut_plane_data['status'] != 'ok':
        return
    cut_plane = np.array(cut_plane_data['cut-leaves'])
    neuron = load_neuron(inputfile)
    info = compute_statistics(neuron, cut_plane)

    # BlueRepairSDK used to have a bounding cylinder filter but
    # I don't know what is it good at so I have commented
    # the only relevant line
    # bounding_cylinder_radius = 10000
    cut_sections_in_bounding_cylinder = [
        section for section in iter_sections(neuron)
        if (is_cut_section(section, cut_points=cut_plane)
            # and np.linalg.norm(section.points[-1, COLS.XZ]) < bounding_cylinder_radius
            )
    ]

    try:
        for section in sorted(cut_sections_in_bounding_cylinder, key=section_path_length):
            if section.type == NeuriteType.axon:
                continue
            if section.type != NeuriteType.basal_dendrite:
                raise NotImplementedError(
                    'Section type: {} is not handled yet'.format(section.type))
            grow(section, info, first_sec=True, order_offset=0,  # FIXME
                 origin=neuron.soma.center)
    except StopIteration:
        pass
    neuron.write(outputfile)


def repair_all(input_dir, output_dir, seed=0):
    '''Repair all morphologies in input folder'''
    def good_ext(filename):
        s = filename.split('.')
        if len(s) < 2:
            return False
        return s[-1].lower() in {'asc', 'h5', 'swc'}

    files = list(filter(good_ext, os.listdir(input_dir)))

    for f in files:
        L.info(f)
        inputfilename = os.path.join(input_dir, f)
        outfilename = os.path.join(output_dir, os.path.basename(f))
        try:
            repair(inputfilename, outfilename, seed=seed)
        except:  # noqa, pylint: disable=bare-except
            L.warning('%s failed', f)
