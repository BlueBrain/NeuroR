'''Dendritic repair module

It is based on the BlueRepairSDK's implementation
'''
import logging
from collections import Counter, OrderedDict, defaultdict
from enum import Enum
from itertools import chain

import neurom as nm
import numpy as np
from morph_tool import apical_point_section_segment
from morphio import PointLevel, SectionType
from neurom import NeuriteType, iter_neurites, iter_sections, load_neuron
from neurom.core.dataformat import COLS
from neurom.features.sectionfunc import branch_order, section_path_length
from scipy.spatial.distance import cdist

from neuror import axon
from neuror.cut_plane import CutPlane
from neuror.utils import direction, rotation_matrix, section_length

SEG_LENGTH = 5.0
SHOLL_LAYER_SIZE = 10
BIFURCATION_BOOSTER = 0
NOISE_CONTINUATION = 0.7
SOMA_REPULSION = 0.7
BIFURCATION_ANGLE = 0

# Epsilon needs not to be to small otherwise leaves stored in json files
# are not found in the NeuroM neuron
EPSILON = 1e-6

L = logging.getLogger('neuror')


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
    if cut_points.size == 0 or section.points.size == 0:
        return False
    return np.min(cdist(section.points[:, COLS.XYZ], cut_points)) < EPSILON


def is_branch_intact(branch, cut_points):
    '''Does the branch have leaves belonging to the cut plane ?'''
    return all(not is_cut_section(section, cut_points)
               for section in branch.ipreorder())


def _get_sholl_layer(section, origin):
    '''Returns this section sholl layer'''
    return int(np.linalg.norm(section.points[-1, COLS.XYZ] - origin) / SHOLL_LAYER_SIZE)


def _get_sholl_proba(sholl_data, section_type, sholl_layer, pseudo_order):
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
    Note2: OrderedDict ensures the reproducibility of np.random.choice outcome
    '''

    section_type_data = sholl_data[section_type]
    try:
        data_layer = section_type_data[sholl_layer]
    except KeyError:
        return OrderedDict([(Action.BIFURCATION, 0),
                            (Action.CONTINUATION, 0),
                            (Action.TERMINATION, 1)])

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
        return OrderedDict([(Action.BIFURCATION, 0),
                            (Action.CONTINUATION, 0),
                            (Action.TERMINATION, 1)])

    boost_bifurcation = int(total_counts * BIFURCATION_BOOSTER)
    action_counts[Action.BIFURCATION] += boost_bifurcation
    total_counts += boost_bifurcation
    res = OrderedDict([tuple((action, action_counts[action] / float(total_counts))) for action in
                       sorted(action_counts.keys(), key=lambda t: t.value)])
    return res


def _grow_until_sholl_sphere(section, origin, sholl_layer):
    '''Grow until reaching next sholl layer

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n363  # noqa, pylint: disable=line-too-long
    '''
    backwards_sections = 0
    while _get_sholl_layer(section, origin) == sholl_layer and backwards_sections < 4:
        _continuation(section, origin)

        # make sure we don't grow back the origin
        is_last_segment_toward_origin = np.dot(
            section.points[-1, COLS.XYZ] - origin,
            _last_segment_vector(section)) < 0

        if is_last_segment_toward_origin:
            backwards_sections += 1
    return backwards_sections


def _last_segment_vector(section, normalized=False):
    '''Returns the vector formed by the last 2 points of the section'''
    vec = section.points[-1, COLS.XYZ] - section.points[-2, COLS.XYZ]
    if normalized:
        return vec / np.linalg.norm(vec)
    return vec


def _get_similar_child_diameters(sections, original_section):
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


def _branching_angles(section, order_offset=0):
    '''Return a list of 2-tuples. The first element is the branching order and the second one is
    the angles between the direction of the section and its children's ones

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/morphstats.cpp#n194  # noqa, pylint: disable=line-too-long
    '''
    if section_length(section) < EPSILON:
        return []
    res = []

    branching_order = branch_order(section) - order_offset
    for child in section.children:
        if section_length(child) < EPSILON:
            continue

        theta = np.math.acos(np.dot(direction(section), direction(child)) /
                             (section_length(section) * section_length(child)))
        res.append((branching_order, theta))
    return res


def _continuation(sec, origin):
    '''Continue growing the section

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n241  # noqa, pylint: disable=line-too-long
    '''
    # The following lines is from BlueRepairSDK's code but I'm not
    # convinced by its relevance
    # if first_sec:
    #     direction_ = sec.points[-1, COLS.XYZ] - sec.points[0, COLS.XYZ]
    # else:
    section_direction = _last_segment_vector(sec)
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


def _y_cylindrical_extent(section):
    '''Returns the distance from the last section point to the origin in the XZ plane'''
    xz_last_point = section.points[-1, [0, 2]]
    return np.linalg.norm(xz_last_point)


def _max_y_dendritic_cylindrical_extent(neuron):
    '''Return the maximum distance of dendritic section ends and the origin in the XZ plane'''
    return max(_y_cylindrical_extent(section) for section in neuron.iter()
               if section.type in {SectionType.basal_dendrite, SectionType.apical_dendrite})


class Repair(object):
    '''The repair class'''

    def __init__(self, inputfile, axons=None, seed=0, plane=None):
        '''Repair the input morphology

        Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/repair.cpp#n469  # noqa, pylint: disable=line-too-long
        '''
        np.random.seed(seed)
        if plane is None:
            L.info('No cut plane specified. Calling CutPlane.find...')
            cut_plane = CutPlane.find(inputfile, bin_width=15)
            L.info('Found cut plane: %s', cut_plane)
        else:
            if isinstance(plane, CutPlane):
                cut_plane = plane
            else:
                cut_plane = CutPlane.from_json(plane)

        self.inputfile = inputfile
        self.axon_donors = axons or list()
        self.donated_intact_axon_sections = list()
        self.cut_leaves = cut_plane.cut_leaves_coordinates
        self.neuron = load_neuron(inputfile)
        self.repair_type_map = dict()
        self.max_y_cylindrical_extent = _max_y_dendritic_cylindrical_extent(self.neuron)
        self.max_y_extent = max(np.max(section.points[:, COLS.Y])
                                for section in self.neuron.iter())

        self.info = dict()
        apical_section_id, _ = apical_point_section_segment(self.neuron)
        if apical_section_id:
            self.apical_section = self.neuron.sections[apical_section_id]
        else:
            self.apical_section = None

    def run(self, outputfile, plot_file=None):
        '''Run'''
        if self.cut_leaves.size == 0:
            L.warning('No cut leaves. Nothing to repair for morphology %s', self.inputfile)
            self.neuron.write(outputfile)
            return

        # The only purpose of the 'planes' variable is to keep
        # each plane.morphology alive. Otherwise sections become unusable
        # https://github.com/BlueBrain/MorphIO/issues/29
        planes = []
        for axon_donor in self.axon_donors:
            plane = CutPlane.find(axon_donor)
            planes.append(plane)
            no_cut_plane = (plane.minus_log_prob < 50)
            self.donated_intact_axon_sections.extend(
                [section for section in iter_sections(plane.morphology)
                 if section.type == SectionType.axon and
                 (no_cut_plane or is_branch_intact(section, plane.cut_leaves_coordinates))])

        self._fill_repair_type_map()
        self._fill_statistics_for_intact_subtrees()
        intact_axonal_sections = [section for section in iter_sections(self.neuron)
                                  if section.type == SectionType.axon and
                                  is_branch_intact(section, self.cut_leaves)]

        # BlueRepairSDK used to have a bounding cylinder filter but
        # I don't know what is it good at so I have commented
        # the only relevant line
        # bounding_cylinder_radius = 10000
        cut_sections_in_bounding_cylinder = [
            section for section in iter_sections(self.neuron)
            if (is_cut_section(section, cut_points=self.cut_leaves)
                # and np.linalg.norm(section.points[-1, COLS.XZ]) < bounding_cylinder_radius
                )
        ]

        cut_leaves_ids = {section: len(section.points)
                          for section in cut_sections_in_bounding_cylinder}

        for section in sorted(cut_sections_in_bounding_cylinder, key=section_path_length):
            type_ = self.repair_type_map[section]
            L.info('Repairing: %s, %s', type_, section.id)
            if type_ in {RepairType.basal, RepairType.oblique, RepairType.tuft}:
                origin = self._get_origin(section)
                if section.type == NeuriteType.basal_dendrite:
                    _continuation(section, origin)
                self._grow(section, self._get_order_offset(section), origin)
            elif type_ == RepairType.axon:
                axon.repair(self.neuron, section, intact_axonal_sections,
                            self.donated_intact_axon_sections, self.max_y_extent)
            elif type_ == RepairType.trunk:
                L.info('Trunk repair is not (nor has ever been) implemented')
            else:
                raise Exception('Unknown type: {}'.format(type_))

        if plot_file is not None:
            try:
                from neuror.view import plot_repaired_neuron
                plot_repaired_neuron(self.neuron, cut_leaves_ids, plot_file)
            except ImportError:
                L.warning('Skipping writing plots as [plotly] extra is not installed')

        self.neuron.write(outputfile)
        L.info('Repair successful for %s', self.inputfile)

    def _find_intact_obliques(self):
        '''
        Find root sections of all intact obliques

        Root obliques are obliques with a section parent of type 'trunk'

        Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n193  # noqa, pylint: disable=line-too-long
        '''
        root_obliques = (section for section in iter_sections(self.neuron)
                         if (self.repair_type_map[section] == RepairType.oblique and
                             not section.is_root() and
                             self.repair_type_map[section.parent] == RepairType.trunk))
        intacts = [oblique for oblique in root_obliques
                   if is_branch_intact(oblique, self.cut_leaves)]
        return intacts

    def _find_intact_sub_trees(self):
        '''Returns intact neurites'''
        basals = [neurite.root_node for neurite in iter_neurites(self.neuron)
                  if (neurite.type == NeuriteType.basal_dendrite and
                      is_branch_intact(neurite.root_node, self.cut_leaves))]

        if not basals:
            raise Exception('No intact basal dendrites !')

        axons = [neurite.root_node for neurite in iter_neurites(self.neuron)
                 if (neurite.type == NeuriteType.axon and
                     is_branch_intact(neurite.root_node, self.cut_leaves))]
        obliques = self._find_intact_obliques()

        tufts = [section for section in iter_sections(self.neuron)
                 if (self.repair_type_map[section] == RepairType.tuft and
                     not is_cut_section(section, self.cut_leaves))]

        return basals + obliques + axons + tufts

    def _intact_branching_angles(self, branches):
        '''
        Returns lists of branching angles stored in a nested dict
        1st key: section type, 2nd key: branching order

        Args:
            branches (List[Neurite])

        Returns:
            Dict[SectionType, Dict[int, List[int]]]: Branching angles
        '''
        res = defaultdict(lambda: defaultdict(list))
        for branch in branches:
            order_offset = self._get_order_offset(branch)
            for section in branch.ipreorder():
                for order, angle in _branching_angles(section, order_offset):
                    res[self.repair_type_map[section]][order].append(angle)
        return dict(res)

    def _best_case_angle_data(self, section_type, branching_order):
        '''Get the distribution of branching angles for this section type and
        branching order

        If no data are available, fallback on aggregate data

        Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n329  # noqa, pylint: disable=line-too-long
        '''
        angles = self.info['intact_branching_angles'][section_type]
        accurate_data = angles[branching_order]
        if accurate_data:
            return accurate_data

        return list(chain.from_iterable(angles.values()))

    def _get_origin(self, branch):
        '''Return what should be considered as the origin for this branch'''
        if self.repair_type_map[branch] == RepairType.oblique:
            return branch.points[0, COLS.XYZ]
        if self.repair_type_map[branch] == RepairType.tuft:
            return self.apical_section.points[-1, COLS.XYZ]
        return self.neuron.soma.center

    def _get_order_offset(self, branch):
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
        if self.repair_type_map[branch] == RepairType.oblique:
            return branch_order(branch)
        if self.repair_type_map[branch] == RepairType.tuft:
            return branch_order(self.apical_section)
        return 0

    def _compute_sholl_data(self, branches):
        '''Compute the number of termination, bifurcation and continuation section for each
        neurite type, sholl layer and shell order

        data[neurite_type][layer][order][action_type] = counts

        Args:
            branches: a collection of Neurite or Section that will be traversed

        Note: This is based on
        https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/morphstats.cpp#n93
        '''
        data = defaultdict(lambda: defaultdict(dict))

        for branch in branches:
            origin = self._get_origin(branch)
            order_offset = self._get_order_offset(branch)

            for section in branch.ipreorder():
                repair_type = self.repair_type_map[section]
                assert repair_type == self.repair_type_map[branch], \
                    'RepairType should not change along the branch way'
                order = branch_order(section) - order_offset
                first_layer, last_layer = (np.linalg.norm(
                    section.points[[0, -1], COLS.XYZ] - origin, axis=1) // SHOLL_LAYER_SIZE).astype(
                        int)
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

    def _bifurcation(self, section, order_offset):
        '''Create 2 children at the end of the current section

        Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n287  # noqa, pylint: disable=line-too-long
        '''
        current_diameter = section.points[-1, COLS.R] * 2
        child_diameters = _get_similar_child_diameters(self.info['dendritic_sections'], section)

        median_angle = np.median(
            self._best_case_angle_data(self.repair_type_map[section],
                                       branch_order(section) - order_offset))

        last_segment_vec = _last_segment_vector(section)
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

    def _grow(self, section, order_offset, origin):
        '''grow main method

        Will either:
            - continue growing the section
            - create a bifurcation
            - terminate the growth

        Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n387  # noqa, pylint: disable=line-too-long
        '''
        if (self.repair_type_map[section] == RepairType.tuft and
                _y_cylindrical_extent(section) > self.max_y_cylindrical_extent):
            return

        sholl_layer = _get_sholl_layer(section, origin)
        pseudo_order = branch_order(section) - order_offset
        L.debug('In _grow. Layer: %s, order: %s', sholl_layer, pseudo_order)

        proba = _get_sholl_proba(self.info['sholl'],
                                 self.repair_type_map[section],
                                 sholl_layer,
                                 pseudo_order)

        L.debug('action proba[%s][%s][%s]: %s', section.type, sholl_layer, pseudo_order, proba)
        action = np.random.choice(list(proba.keys()), p=list(proba.values()))

        if action == Action.CONTINUATION:
            L.info('Continuing')
            backwards_sections = _grow_until_sholl_sphere(section, origin, sholl_layer)

            if backwards_sections == 0:
                self._grow(section, order_offset, origin)
            L.debug(section.points[-1])

        elif action == Action.BIFURCATION:
            L.info('Bifurcating')
            backwards_sections = _grow_until_sholl_sphere(section, origin, sholl_layer)
            if backwards_sections == 0:
                self._bifurcation(section, order_offset)
                for child in section.children:
                    self.repair_type_map[child] = self.repair_type_map[section]
                    self._grow(child, order_offset, origin)

    def _fill_statistics_for_intact_subtrees(self):
        '''Compute statistics'''
        branches = self._find_intact_sub_trees()

        self.info = dict(
            intact_branching_angles=self._intact_branching_angles(branches),
            dendritic_sections=[section for section in iter_sections(self.neuron)
                                if section.type in {nm.APICAL_DENDRITE, nm.BASAL_DENDRITE}],
            sholl=self._compute_sholl_data(branches),
        )

    def _fill_repair_type_map(self):
        '''Assign a repair section type to each section

        Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/repair.cpp#n242  # noqa, pylint: disable=line-too-long
        '''
        for section in iter_sections(self.neuron):
            if section.type == SectionType.apical_dendrite:
                self.repair_type_map[section] = RepairType.oblique
            elif section.type == SectionType.basal_dendrite:
                self.repair_type_map[section] = RepairType.basal
            elif section.type == SectionType.axon:
                self.repair_type_map[section] = RepairType.axon

        if self.apical_section is not None:
            for section in self.apical_section.ipreorder():
                self.repair_type_map[section] = RepairType.tuft

            # The value for the apical section must be overriden to 'trunk'
            for section in self.apical_section.iupstream():
                self.repair_type_map[section] = RepairType.trunk


def repair(inputfile, outputfile, axons=None, seed=0, plane=None, plot_file=None):
    '''The repair function'''
    if axons is None:
        axons = list()
    obj = Repair(inputfile, axons=axons, seed=seed, plane=plane)
    obj.run(outputfile, plot_file=plot_file)
