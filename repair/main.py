'''Dendritic repair module

It is based on the BlueRepairSDK's implementation
'''
import logging
import os
from collections import defaultdict, Counter
from enum import Enum
from itertools import chain, tee
from pprint import pformat

from pathlib2 import Path
import numpy as np
from scipy.spatial.distance import cdist

from morphio import PointLevel, SectionType
from cut_plane import CutPlane
from cut_plane.utils import iter_morphology_files
import neurom as nm
from neurom import (NeuriteType, iter_neurites, iter_segments,
                    load_neuron, iter_sections)
from neurom.core.dataformat import COLS
from neurom.features.sectionfunc import branch_order, section_path_length
from repair.utils import angle_between, rotation_matrix, read_apical_points
from repair.view import plot_repaired_neuron, view_all
from repair.unravel import unravel_all

L = logging.getLogger('repair')
SEG_LENGTH = 5.0
SHOLL_LAYER_SIZE = 10
BIFURCATION_BOOSTER = 0
NOISE_CONTINUATION = 0.7
SOMA_REPULSION = 0.7
BIFURCATION_ANGLE = 0

# Epsilon needs not to be to small otherwise leaves stored in json files
# are not found in the NeuroM neuron
EPSILON = 1e-6


class Action(Enum):
    '''To bifurcate or not to bifurcate ?'''
    BIFURCATION = 1
    CONTINUATION = 2
    TERMINATION = 3


class RepairType(Enum):
    '''The types used for the repair

    based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.h#n22  # noqa, pylint: disable=line-too-long
    '''
    trunk = 0
    tuft = 1
    oblique = 2
    basal = 3
    axon = 4


def is_cut_section(section, cut_points):
    '''Return true if the section is close from the cut plane'''
    return np.min(cdist(section.points[:, COLS.XYZ], cut_points)) < EPSILON


def is_branch_intact(branch, cut_points):
    '''Does the branch have leaves belonging to the cut plane ?'''
    return all(not is_cut_section(section, cut_points)
               for section in branch.ipreorder())


def find_intact_obliques(neuron, cut_leaves, repair_type_map):
    '''
    Find root sections of all intact obliques

    Root obliques are obliques with a section parent of type 'trunk'

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n193  # noqa, pylint: disable=line-too-long
    '''
    root_obliques = (section for section in iter_sections(neuron)
                     if (repair_type_map[section] == RepairType.oblique and
                         not section.is_root and
                         repair_type_map[section.parent] == RepairType.trunk))
    intacts = [oblique for oblique in root_obliques if is_branch_intact(oblique, cut_leaves)]
    return intacts


def find_intact_sub_trees(morph, cut_leaves, repair_type_map):
    '''Returns intact neurites'''
    basals = [neurite.root_node for neurite in iter_neurites(morph)
              if (neurite.type == NeuriteType.basal_dendrite and
                  is_branch_intact(neurite.root_node, cut_leaves))]
    if not basals:
        raise Exception('No intact basal dendrites !')

    obliques = find_intact_obliques(morph, cut_leaves, repair_type_map)

    return basals + obliques


def direction(section):
    '''Return the direction vector of a section'''
    return np.diff(section.points[[0, -1]][:, COLS.XYZ], axis=0)[0]


def _section_length(section):
    '''Section length'''
    return np.linalg.norm(direction(section))


def branching_angles(section, order_offset=0):
    '''Return a list of 2-tuples. The first element is the branching order and the second one is
    the angles between the direction of the section and its children's ones

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/morphstats.cpp#n194  # noqa, pylint: disable=line-too-long
    '''
    if _section_length(section) < EPSILON:
        return []
    res = []

    branching_order = branch_order(section) - order_offset
    for child in section.children:
        if _section_length(child) < EPSILON:
            continue

        theta = np.math.acos(np.dot(direction(section), direction(child)) /
                             (_section_length(section) * _section_length(child)))
        res.append((branching_order, theta))
    return res


def intact_branching_angles(branches, repair_type_map):
    '''
    Returns lists of branching angles stored in a nested dict
    1st key: section type, 2nd key: branching order

    Args:
        neurites (List[Neurite])

    Returns:
        Dict[SectionType, Dict[int, List[int]]]: Branching angles
    '''
    res = defaultdict(lambda: defaultdict(list))
    for branch in branches:
        order_offset = get_order_offset(branch, repair_type_map)
        for section in branch.ipreorder():
            for order, angle in branching_angles(section, order_offset):
                res[repair_type_map[section]][order].append(angle)
    return dict(res)


def best_case_angle_data(info, section_type, branching_order):
    '''Get the distribution of branching angles for this section type and
    branching order

    If no data are available, fallback on aggregate data

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n329  # noqa, pylint: disable=line-too-long
    '''
    angles = info['intact_branching_angles'][section_type]
    accurate_data = angles[branching_order]
    if accurate_data:
        return accurate_data

    return list(chain.from_iterable(angles.values()))


def get_origin(branch, soma_center, repair_type_map):
    '''Return what should be considered as the origin for this branch'''
    if repair_type_map[branch] == RepairType.oblique:
        return branch.points[0, COLS.XYZ]
    return soma_center


def get_order_offset(branch, repair_type_map):
    r'''
    Return what should be considered as the branch order offset for this branch

    For obliques, the branch order is computed with respect to
    the branch order of the first oblique section so we have to remove the offset.

                3
             2  |
 oblique ---->\ |
               \| 1
                |
                |
                | 0

    oblique order is 2 - 1 = 1
    '''
    if repair_type_map[branch] == RepairType.oblique:
        return branch_order(branch)
    return 0


def compute_sholl_data(branches, soma_center, repair_type_map):
    '''Compute the number of termination, bifurcation and continuation section for each
    neurite type, sholl layer and shell order

    data[neurite_type][layer][order][action_type] = counts

    Args:
        branches: a collection of Neurite or Section that will be traversed
        origin: The origin of the Sholl sphere. If none, the origin is set as the
            first point of each branch

    Note: This is based on
    https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/morphstats.cpp#n93
    '''
    data = defaultdict(lambda: defaultdict(dict))

    for branch in branches:
        origin = get_origin(branch, soma_center, repair_type_map)
        order_offset = get_order_offset(branch, repair_type_map)

        for section in branch.ipreorder():
            repair_type = repair_type_map[section]
            assert repair_type == repair_type_map[branch], \
                'RepairType should not change along the branch way'
            order = branch_order(section) - order_offset
            first_layer, last_layer = (np.linalg.norm(
                section.points[[0, -1], COLS.XYZ] - origin, axis=1) // SHOLL_LAYER_SIZE).astype(int)
            per_type = data[repair_type]
            for layer in range(min(first_layer, last_layer), max(first_layer, last_layer) + 1):
                if order not in per_type[layer]:
                    per_type[layer][order] = {Action.TERMINATION: 0,
                                              Action.CONTINUATION: 0,
                                              Action.BIFURCATION: 0}
                per_type[layer][order][Action.CONTINUATION] += 1

            per_type[last_layer][order][Action.BIFURCATION if section.children else
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

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n398  # noqa, pylint: disable=line-too-long
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
        # No data for this order. Average on all orders for this layer
        # As done in https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n426  # noqa, pylint: disable=line-too-long

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
    '''Continue growing the section

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n241  # noqa, pylint: disable=line-too-long
    '''
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
    '''Grow until reaching next sholl layer

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n363  # noqa, pylint: disable=line-too-long
    '''
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


def get_similar_child_diameters(sections, original_section):
    '''Find an existing section with a similar diameter and returns the diameters
    of its children

    If no similar section is found, returns twice the diameter of the last point

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n76  # noqa, pylint: disable=line-too-long
    '''
    threshold = 0.2
    diameter = original_section.points[-1, COLS.R] * 2

    for section in sections:
        if section.children and abs(section.points[-1, COLS.R] * 2 - diameter) < threshold:
            return [2 * section.children[0].points[0, COLS.R],
                    2 * section.children[1].points[0, COLS.R]]

    L.warning("Oh no, no similar diameter found!")
    return [original_section.points[-1, COLS.R] * 2,
            original_section.points[-1, COLS.R] * 2]


def bifurcation(section, order_offset, info, repair_type_map):
    '''Create 2 children at the end of the current section

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n287  # noqa, pylint: disable=line-too-long
    '''
    current_diameter = section.points[-1, COLS.R] * 2
    child_diameters = get_similar_child_diameters(info['dendritic_sections'], section)

    median_angle = np.median(
        best_case_angle_data(info, repair_type_map[section], branch_order(section) - order_offset))

    last_segment_vec = last_segment_vector(section)
    orthogonal = np.cross(last_segment_vec, section.points[-1, COLS.XYZ])

    def shuffle_direction(direction_):
        '''Return the direction to which to grow a child section'''
        theta = median_angle + np.radians(np.random.random() * BIFURCATION_ANGLE)
        first_rotation = np.dot(rotation_matrix(orthogonal, theta), direction_)
        new_dir = np.dot(rotation_matrix(direction_, np.random.random() * np.pi * 2),
                         first_rotation)
        return new_dir / np.linalg.norm(new_dir)

    for child_diameter in child_diameters:
        child_start = section.points[-1, COLS.XYZ]
        points = [child_start.tolist(),
                  (child_start + shuffle_direction(last_segment_vec) * SEG_LENGTH).tolist()]
        prop = PointLevel(points, [current_diameter, child_diameter])

        child = section.append_section(prop)
        L.debug('section appended: %s', child.id)


def grow(section, info, order_offset, origin, repair_type_map):
    '''Grow main method

    Will either:
        - continue growing the section
        - create a bifurcation
        - terminate the growth

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n387  # noqa, pylint: disable=line-too-long
    '''
    sholl_layer = get_sholl_layer(section, origin)
    pseudo_order = branch_order(section) - order_offset
    L.debug('In Grow. Layer: %s, order: %s', sholl_layer, pseudo_order)

    proba = get_sholl_proba(info['sholl'], repair_type_map[section], sholl_layer, pseudo_order)
    L.debug('action proba[%s][%s][%s]: %s', section.type, sholl_layer, pseudo_order, proba)
    action = np.random.choice(list(proba.keys()), p=list(proba.values()))

    if action == Action.CONTINUATION:
        L.info('Continuing')
        backwards_sections = grow_until_sholl_sphere(section, origin, sholl_layer)

        if backwards_sections == 0:
            grow(section, info, order_offset, origin, repair_type_map)
        L.debug(section.points[-1])

    elif action == Action.BIFURCATION:
        L.info('Bifurcating')
        backwards_sections = grow_until_sholl_sphere(section, origin, sholl_layer)
        if backwards_sections == 0:
            bifurcation(section, order_offset, info, repair_type_map)
            for child in section.children:
                repair_type_map[child] = repair_type_map[section]
                grow(child, info, order_offset, origin, repair_type_map)
    else:
        L.info('Terminating')


def angle_between_segments(neuron):
    '''Returns a list of angles between successive segments

    This is not used in the growth code but can be useful for analyses
    '''
    angles = list()
    for section in iter_sections(neuron):
        iter_seg1, iter_seg2 = tee(iter_segments(section))
        next(iter_seg2)
        for (p0, p1), (_, p2) in zip(iter_seg1, iter_seg2):
            vec1 = (p1 - p0)[COLS.XYZ]
            vec2 = (p2 - p1)[COLS.XYZ]
            angles.append(angle_between(vec1, vec2))
    return angles


def compute_statistics_for_intact_subtrees(neuron, cut_plane, repair_type_map):
    '''Compute statistics'''
    branches = find_intact_sub_trees(neuron, cut_plane, repair_type_map)

    return {
        'intact_branching_angles': intact_branching_angles(branches, repair_type_map),
        'dendritic_sections': [section for section in iter_sections(neuron)
                               if section.type in {nm.APICAL_DENDRITE, nm.BASAL_DENDRITE}],
        'sholl': compute_sholl_data(branches, neuron.soma.center, repair_type_map),
    }


def repair(inputfile, outputfile, seed=0, plane=None, plot_file=None):
    '''Repair the input morphology

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/repair.cpp#n469  # noqa, pylint: disable=line-too-long
    '''
    np.random.seed(seed)
    if plane is None:
        cut_plane = CutPlane.find(inputfile, bin_width=15)
    else:
        cut_plane = CutPlane.from_json(plane)

    cut_leaves = cut_plane.cut_leaves_coordinates

    neuron = load_neuron(inputfile)
    if cut_leaves.size == 0:
        L.warning('No cut leaves. Nothing to repair for morphology %s', inputfile)
        neuron.write(outputfile)
        return

    apical_section = read_apical_points(inputfile, neuron)
    repair_type_map = subtree_classification(neuron, apical_section)

    info = compute_statistics_for_intact_subtrees(neuron, cut_leaves, repair_type_map)
    L.debug(pformat(info))

    # BlueRepairSDK used to have a bounding cylinder filter but
    # I don't know what is it good at so I have commented
    # the only relevant line
    # bounding_cylinder_radius = 10000
    cut_sections_in_bounding_cylinder = [
        section for section in iter_sections(neuron)
        if (is_cut_section(section, cut_points=cut_leaves)
            # and np.linalg.norm(section.points[-1, COLS.XZ]) < bounding_cylinder_radius
            )
    ]

    cut_leaves_ids = {section: len(section.points)
                      for section in cut_sections_in_bounding_cylinder}

    try:
        for section in sorted(cut_sections_in_bounding_cylinder, key=section_path_length):
            if repair_type_map[section] == RepairType.basal:
                origin = get_origin(section, neuron.soma.center, repair_type_map)
                if section.type == NeuriteType.basal_dendrite:
                    continuation(section, origin)
                grow(section, info, get_order_offset(section, repair_type_map), origin,
                     repair_type_map)
    except StopIteration:
        pass

    if plot_file is not None:
        plot_repaired_neuron(neuron, cut_leaves_ids, plot_file)
    neuron.write(outputfile)
    L.info('Repair successful for %s', inputfile)


def repair_all(input_dir, output_dir, seed=0, planes_dir=None, plot_dir=None):
    '''Repair all morphologies in input folder'''
    for f in iter_morphology_files(input_dir):
        L.info(f)
        inputfilename = Path(input_dir, f)
        outfilename = Path(output_dir, os.path.basename(f))
        if planes_dir:
            plane = str(Path(planes_dir, inputfilename.name).with_suffix('.json'))
        else:
            plane = None

        if plot_dir is not None:
            name = 'neuron_{}.html'.format(Path(inputfilename).stem.replace(' ', '_'))
            plot_file = str(Path(plot_dir, name))
        else:
            plot_file = None

        try:
            repair(str(inputfilename), str(outfilename),
                   seed=seed, plane=plane, plot_file=plot_file)
        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning('%s failed', f)
            L.warning(e, exc_info=True)


def subtree_classification(neuron, apical_section):
    '''Assign a repair section type to each section

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/repair.cpp#n242  # noqa, pylint: disable=line-too-long
    '''
    repair_type_map = dict()
    for section in iter_sections(neuron):
        if section.type == SectionType.apical_dendrite:
            repair_type_map[section] = RepairType.oblique
        elif section.type == SectionType.basal_dendrite:
            repair_type_map[section] = RepairType.basal
        elif section.type == SectionType.axon:
            repair_type_map[section] = RepairType.axon

    if apical_section:
        for section in apical_section.ipreorder():
            repair_type_map[section] = RepairType.tuft

        # The value for the apical section gets overriden to 'trunk'
        for section in apical_section.iupstream():
            repair_type_map[section] = RepairType.trunk

    return repair_type_map


def full(root_dir, seed=0, window_half_length=5):
    '''
    Perform the unravelling and repair in ROOT_DIR:

    1) perform the unravelling of the neuron
    2) update the cut points position after unravelling and writes it
       in the unravelled/planes folder
    3) repair the morphology

    The ROOT_DIR is expected to contain the following folders:
    - raw/ with all raw morphologies to repair
    - raw/planes with all cut planes
    - unravelled/ where unravelled morphologies will be written
    - unravelled/planes where unravelled planes will be written
    - repaired/ where repaired morphologies will be written
    '''

    raw_dir = os.path.join(root_dir, 'raw')
    unravelled_dir = os.path.join(root_dir, 'unravelled')
    repaired_dir = os.path.join(root_dir, 'repaired')
    plots_dir = os.path.join(root_dir, 'plots')
    unravelled_planes_dir = str(Path(unravelled_dir, 'planes'))
    if not os.path.exists(raw_dir):
        raise Exception('%s does not exists' % raw_dir)

    for folder in [unravelled_dir, unravelled_planes_dir, repaired_dir, plots_dir]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    unravel_all(raw_dir, unravelled_dir, window_half_length)
    repair_all(unravelled_dir,
               repaired_dir,
               seed=seed,
               planes_dir=unravelled_planes_dir,
               plot_dir=plots_dir)
    view_all([raw_dir, unravelled_dir, repaired_dir],
             titles=['raw', 'unravelled', 'repaired'],
             output_pdf=os.path.join(root_dir, 'plots.pdf'))
