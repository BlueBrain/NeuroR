'''Dendritic repair module

It is based on the BlueRepairSDK's implementation
'''
import logging
from collections import Counter, OrderedDict, defaultdict
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional

import morphio
import neurom as nm
import numpy as np
from morph_tool import apical_point_section_segment
from morph_tool.spatial import point_to_section_segment
from morphio import PointLevel, SectionType
from neurom import NeuriteType, iter_neurites, iter_sections, load_morphology
from neurom.core.dataformat import COLS
from neurom.core import Section
from neurom.features.section import branch_order, section_path_length
from nptyping import NDArray
from scipy.spatial.distance import cdist

from neuror import axon, cut_plane
from neuror.utils import RepairType, direction, repair_type_map, rotation_matrix, section_length

_PARAMS = {
    'seg_length': 5.0,  # lenghts of new segments
    'sholl_layer_size': 10,  # resoluion of the shll profile
    'noise_continuation': 0.5,  # together with seg_length, this controls the tortuosity
    'soma_repulsion': 0.2,  # if 0, previous section direction, if 1, radial direction
    'bifurcation_angle': 20,  # noise amplitude in degree around mean bif angle on the cell
    'path_length_ratio': 0.5,  # a smaller value will make a strornger taper rate
    'children_diameter_ratio': 0.8,  # 1: child diam = parent diam, 0: child diam = tip diam
    'tip_percentile': 25,  # percentile of tip radius distributions to use as tip radius
}

# Epsilon can not be to small otherwise leaves stored in json files
# are not found in the NeuroM neuron
EPSILON = 1e-6

L = logging.getLogger('neuror')


class Action(Enum):
    '''To bifurcate or not to bifurcate ?'''
    BIFURCATION = 1
    CONTINUATION = 2
    TERMINATION = 3


def is_cut_section(section, cut_points):
    '''Return true if the section is close from the cut plane'''
    if cut_points.size == 0 or section.points.size == 0:
        return False
    return np.min(cdist(section.points[:, COLS.XYZ], cut_points)) < EPSILON


def is_branch_intact(branch, cut_points):
    '''Does the branch have leaves belonging to the cut plane ?'''
    return all(not is_cut_section(section, cut_points) for section in branch.ipreorder())


def _get_sholl_layer(section, origin, sholl_layer_size):
    '''Returns this section sholl layer'''
    return int(np.linalg.norm(section.points[-1, COLS.XYZ] - origin) / sholl_layer_size)


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

    res = OrderedDict([tuple((action, action_counts[action] / float(total_counts))) for action in
                       sorted(action_counts.keys(), key=lambda t: t.value)])
    return res


def _grow_until_sholl_sphere(
    section, origin, sholl_layer, params, taper, tip_radius, current_trunk_radius
):
    '''Grow until reaching next sholl layer

    Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n363  # noqa, pylint: disable=line-too-long
    '''
    backwards_sections = 0
    while _get_sholl_layer(
        section, origin, params['sholl_layer_size']
    ) == sholl_layer and backwards_sections < 4:
        _continuation(section, origin, params, taper(current_trunk_radius), tip_radius)

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


def _continuation(sec, origin, params, taper, tip_radius):
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

    # better to use last section point, to get direction at the tip from where we grow
    radial_direction = sec.points[-1, COLS.XYZ] - origin
    length = np.linalg.norm(radial_direction)
    if length > EPSILON:
        radial_direction /= length
    else:
        radial_direction = sec.points[1, COLS.XYZ] - sec.points[0, COLS.XYZ]
        radial_direction /= np.linalg.norm(radial_direction)

    # NOTE: This is not an equiprobability uniform generator
    # It is not drawing a point inside a sphere but a cube.
    # so the probability of drawing in direction of the corners is higher
    noise_direction = (2 * np.random.random(size=3) - 1)
    noise_direction /= np.linalg.norm(noise_direction)

    direction_ = (
        section_direction
        + params["soma_repulsion"] * radial_direction
        + params["noise_continuation"] * noise_direction
    )
    direction_ /= np.linalg.norm(direction_)

    coord = sec.points[-1, COLS.XYZ] + direction_ * params["seg_length"]
    # we apply tapering function + minimum diameter
    radius = max(tip_radius, sec.points[-1, COLS.R] - taper)
    new_point = np.append(coord, radius)
    sec.points = np.vstack((sec.points, new_point))


def _y_cylindrical_extent(section):
    '''Returns the distance from the last section point to the origin in the XZ plane'''
    xz_last_point = section.points[-1, [0, 2]]
    return np.linalg.norm(xz_last_point)


def _max_y_dendritic_cylindrical_extent(neuron):
    '''Return the maximum distance of dendritic section ends and the origin in the XZ plane'''
    return max((_y_cylindrical_extent(section) for section in neuron.iter()
                if section.type in {SectionType.basal_dendrite, SectionType.apical_dendrite}),
               default=0)


class Repair(object):
    '''The repair class'''

    def __init__(self,  # pylint: disable=too-many-arguments
                 inputfile: Path,
                 axons: Optional[Path] = None,
                 seed: Optional[int] = 0,
                 cut_leaves_coordinates: Optional[NDArray[(3, Any)]] = None,
                 legacy_detection: bool = False,
                 repair_flags: Optional[Dict[RepairType, bool]] = None,
                 apical_point: NDArray[3, float] = None,
                 params: Dict = None):
        '''Repair the input morphology

        The repair algorithm uses sholl analysis of intact branches to grow new branches from cut
        leaves. The algorithm is fairly complex, but can be controled via a few parameters in the
        params dictionary. By default, they are:
        _PARAMS = {
            'seg_length': 5.0,  # lenghts of new segments
            'sholl_layer_size': 10,  # resoluion of the shll profile
            'noise_continuation': 0.5,  # together with seg_length, this controls the tortuosity
            'soma_repulsion': 0.2,  # if 0, previous section direction, if 1, radial direction
            'bifurcation_angle': 20,  # noise amplitude in degree around mean bif angle on the cell
            'path_length_ratio': 0.5,  # a smaller value will make a strornger taper rate
            'children_diameter_ratio': 0.8,  # 1: child diam = parent diam, 0: child diam = tip diam
            'tip_percentile': 25,  # percentile of tip radius distributions to use as tip radius
        }

        Args:
            inputfile: the input neuron to repair
            axons: donor axons whose section will be used to repair this axon
            seed: the numpy seed
            cut_leaves_coordinates: List of 3D coordinates from which to start the repair
            legacy_detection: if True, use the legacy cut plane detection
                (see neuror.legacy_detection)
            repair_flags: a dict of flags where key is a RepairType and value is whether
                it should be repaired or not. If not provided, all types will be repaired.
            apical_point: 3d vector for apical point, else, the automatic apical detection is used
                if apical_point == -1, no automatic detection will be tried
            params: repair internal parameters (see comments in code for details)

        Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/repair.cpp#n469  # noqa, pylint: disable=line-too-long
        '''
        np.random.seed(seed)
        self.legacy_detection = legacy_detection
        self.inputfile = inputfile
        self.axon_donors = axons or []
        self.donated_intact_axon_sections = []
        self.repair_flags = repair_flags or {}
        self.params = params if params is not None else _PARAMS

        CutPlane = cut_plane.CutPlane
        if legacy_detection:
            self.cut_leaves = CutPlane.find_legacy(inputfile, 'z').cut_leaves_coordinates
        elif cut_leaves_coordinates is None:
            self.cut_leaves = CutPlane.find(inputfile, bin_width=15).cut_leaves_coordinates
        else:
            self.cut_leaves = np.asarray(cut_leaves_coordinates)

        self.neuron = load_morphology(inputfile)
        self.repair_type_map = {}
        self.max_y_cylindrical_extent = _max_y_dendritic_cylindrical_extent(self.neuron)
        self.max_y_extent = max(np.max(section.points[:, COLS.Y])
                                for section in self.neuron.iter())

        self.info = {}
        apical_section_id = None
        if apical_point != -1:
            if apical_point:
                apical_section_id = point_to_section_segment(self.neuron, apical_point)[0]
            else:
                apical_section_id = apical_point_section_segment(self.neuron)[0]

        if apical_section_id:
            self.apical_section = self.neuron.sections[apical_section_id]
        else:
            self.apical_section = None

        # record the tip radius as a lower bound for diameters with taper, excluding axons
        # because they are treated separately, with thinner diameters
        _diameters = [
            np.mean(leaf.points[:, COLS.R])
            for neurite in self.neuron.neurites
            if neurite.type is not NeuriteType.axon
            for leaf in iter_sections(neurite, iterator_type=Section.ileaf)
        ]
        self.tip_radius = (
            np.percentile(_diameters, self.params["tip_percentile"]) if _diameters else None
        )
        self.current_trunk_radius = None

        # estimate a global tapering rate of the morphology as a function of trunk radius,
        # such that the tip radius is attained on average af max_path_length, defined as a fraction
        # of the maximal path length via the parameter path_lengt_ratio. The smaller this parameter
        # the faster the radii will convert to tip_radius.
        # TODO: maybe we would need this per neurite_type
        max_path_length = self.params["path_length_ratio"] * np.max(
            nm.get("terminal_path_lengths", self.neuron)
        )
        self.taper = (
            lambda trunk_radius: (trunk_radius - self.tip_radius)
            * self.params["seg_length"]
            / max_path_length
        )

    # pylint: disable=too-many-locals, too-many-branches
    def run(self,
            outputfile: Path,
            plot_file: Optional[Path] = None):
        '''Run'''
        if self.cut_leaves.size == 0:
            L.warning('No cut leaves. Nothing to repair for morphology %s', self.inputfile)
            self.neuron.write(outputfile)
            return

        # See https://github.com/BlueBrain/MorphIO/issues/161
        keep_axons_alive = []

        for axon_donor in self.axon_donors:
            if self.legacy_detection:
                plane = cut_plane.CutPlane.find_legacy(axon_donor, 'z')
            else:
                plane = cut_plane.CutPlane.find(axon_donor)
            keep_axons_alive.append(plane)
            self.donated_intact_axon_sections.extend(
                [section for section in iter_sections(plane.morphology)
                 if section.type == SectionType.axon and
                 is_branch_intact(section, plane.cut_leaves_coordinates)])

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

        used_axon_branches = set()

        cut_leaves_ids = {section: len(section.points)
                          for section in cut_sections_in_bounding_cylinder}

        for section in sorted(cut_sections_in_bounding_cylinder, key=section_path_length):
            type_ = self.repair_type_map[section]
            if not self.repair_flags.get(type_, True):
                L.debug('Repair flag set to False for section type %s : --> Skipping repair.',
                        type_)
                continue
            L.debug('Repairing: %s, section id: %s', type_, section.id)
            if type_ in {RepairType.basal, RepairType.oblique, RepairType.tuft}:
                self.current_trunk_radius = [
                    sec.points[0, COLS.R] for sec in section.iupstream()
                ][-1]
                origin = self._get_origin(section)
                if section.type == NeuriteType.basal_dendrite:
                    _continuation(section, origin, self.params,
                                  self.taper(self.current_trunk_radius), self.tip_radius)
                self._grow(section, self._get_order_offset(section), origin)
            elif type_ == RepairType.axon:
                axon.repair(self.neuron, section, intact_axonal_sections,
                            self.donated_intact_axon_sections, used_axon_branches,
                            self.max_y_extent)
            elif type_ == RepairType.trunk:
                L.debug('Trunk repair is not (nor has ever been) implemented')
            else:
                raise Exception('Unknown type: {}'.format(type_))

        if plot_file is not None:
            try:
                from neuror.view import plot_repaired_neuron
                plot_repaired_neuron(self.neuron, cut_leaves_ids, plot_file)
            except ImportError:
                L.warning('Skipping writing plots as [plotly] extra is not installed')

        self.neuron.write(outputfile)
        L.debug('Repair successful for %s', self.inputfile)

    def _find_intact_obliques(self):
        '''
        Find root sections of all intact obliques

        Root obliques are obliques with a section parent of type 'trunk'

        Note: based on
        https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n193
        '''
        root_obliques = (section for section in iter_sections(self.neuron)
                         if (self.repair_type_map[section] == RepairType.oblique and
                             not section.is_root() and
                             self.repair_type_map[section.parent] == RepairType.trunk))
        intacts = [oblique for oblique in root_obliques
                   if is_branch_intact(oblique, self.cut_leaves)]
        return intacts

    def _find_intact_sub_trees(self):
        '''Returns intact neurites

        There is a fallback mechanism in case there are no intact basals:
        https://bbpcode.epfl.ch/source/xref/platform/BlueRepairSDK/BlueRepairSDK/src/repair.cpp#658
        '''
        basals = [neurite.root_node for neurite in iter_neurites(self.neuron)
                  if (neurite.type == NeuriteType.basal_dendrite and
                      is_branch_intact(neurite.root_node, self.cut_leaves))]

        if not basals:
            L.debug("No intact basals found. Falling back on less strict selection.")
            basals = [section for section in iter_sections(self.neuron)
                      if (section.type == NeuriteType.basal_dendrite and
                          not is_cut_section(section, self.cut_leaves))]

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

        ..note:: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n329  # noqa, pylint: disable=line-too-long
        '''
        angles = self.info['intact_branching_angles'][section_type]
        return angles[branching_order] or list(chain.from_iterable(angles.values()))

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
                first_layer, last_layer = (
                    np.linalg.norm(section.points[[0, -1], COLS.XYZ] - origin, axis=1)
                    // self.params['sholl_layer_size']
                ).astype(int)
                per_type = data[repair_type]

                starting_layer = min(first_layer, last_layer)

                # TODO: why starting_layer + 1 and not starting_layer ?
                # bcoste The continuation from the starting layer should be taken into account
                # But that's how it is done in:
                # https://bbpcode.epfl.ch/source/xref/platform/BlueRepairSDK/BlueRepairSDK/src/morphstats.cpp#88
                for layer in range(starting_layer + 1, max(first_layer, last_layer)):
                    if order not in per_type[layer]:
                        per_type[layer][order] = {Action.TERMINATION: 0,
                                                  Action.CONTINUATION: 0,
                                                  Action.BIFURCATION: 0}
                    per_type[layer][order][Action.CONTINUATION] += 1

                if order not in per_type[last_layer]:
                    per_type[last_layer][order] = {Action.TERMINATION: 0,
                                                   Action.CONTINUATION: 0,
                                                   Action.BIFURCATION: 0}

                per_type[last_layer][order][Action.BIFURCATION if section.children else
                                            Action.TERMINATION] += 1
        return data

    def _bifurcation(self, section, order_offset):
        '''Create 2 children at the end of the current section

        Note: based on https://bbpcode.epfl.ch/browse/code/platform/BlueRepairSDK/tree/BlueRepairSDK/src/helper_dendrite.cpp#n287  # noqa, pylint: disable=line-too-long
        '''
        current_diameter = section.points[-1, COLS.R] * 2

        child_diameters = 2 * [self.params["children_diameter_ratio"] * current_diameter]

        median_angle = np.median(
            self._best_case_angle_data(
                self.repair_type_map[section], branch_order(section) - order_offset
            )
        )

        last_segment_vec = _last_segment_vector(section)
        orthogonal = np.cross(last_segment_vec, section.points[-1, COLS.XYZ])

        def shuffle_direction(direction_):
            '''Return the direction to which to grow a child section'''
            theta = median_angle + np.radians(np.random.random() * self.params['bifurcation_angle'])
            first_rotation = np.dot(rotation_matrix(orthogonal, theta), direction_)
            new_dir = np.dot(rotation_matrix(direction_, np.random.random() * np.pi * 2),
                             first_rotation)
            return new_dir / np.linalg.norm(new_dir)

        for child_diameter in child_diameters:
            child_start = section.points[-1, COLS.XYZ]
            points = [child_start.tolist(),
                      (child_start + shuffle_direction(last_segment_vec) *
                       self.params['seg_length']).tolist()]
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

        sholl_layer = _get_sholl_layer(section, origin, self.params['sholl_layer_size'])
        pseudo_order = branch_order(section) - order_offset
        L.debug('In _grow. Layer: %s, order: %s', sholl_layer, pseudo_order)

        proba = _get_sholl_proba(self.info['sholl'],
                                 self.repair_type_map[section],
                                 sholl_layer,
                                 pseudo_order)

        L.debug('action proba[%s][%s][%s]: %s', section.type, sholl_layer, pseudo_order, proba)
        action = np.random.choice(list(proba.keys()), p=list(proba.values()))

        if action == Action.CONTINUATION:
            L.debug('Continuing')
            backwards_sections = _grow_until_sholl_sphere(section, origin, sholl_layer,
                                                          self.params, self.taper,
                                                          self.tip_radius,
                                                          self.current_trunk_radius)

            if backwards_sections == 0:
                self._grow(section, order_offset, origin)
            L.debug(section.points[-1])

        elif action == Action.BIFURCATION:
            L.debug('Bifurcating')
            backwards_sections = _grow_until_sholl_sphere(section, origin, sholl_layer,
                                                          self.params, self.taper,
                                                          self.tip_radius,
                                                          self.current_trunk_radius)
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
        self.repair_type_map = repair_type_map(self.neuron, self.apical_section)


def repair(inputfile: Path,  # pylint: disable=too-many-arguments
           outputfile: Path,
           axons: Optional[List[Path]] = None,
           seed: int = 0,
           cut_leaves_coordinates: Optional[NDArray[(3, Any)]] = None,
           legacy_detection: bool = False,
           plot_file: Optional[Path] = None,
           repair_flags: Optional[Dict[RepairType, bool]] = None,
           apical_point: List = None,
           params: Dict = None):
    '''The repair function

    Args:
        inputfile: the input morph
        outputfile: the output morph
        axons: the axons
        seed: the numpy seed
        cut_leaves_coordinates: List of 3D coordinates from which to start the repair
        plot_file: the filename of the plot
        repair_flags: a dict of flags where key is a RepairType and value is whether
            it should be repaired or not. If not provided, all types will be repaired.
        apical_point: 3d vector for apical point, else, the automatic apical detection is used
        params: repair internal parameters, None will use defaults
    '''
    ignored_warnings = (
        # We append the section at the wrong place and then we reposition them
        # afterwards so we can ignore the warning
        morphio.Warning.wrong_duplicate,

        # the repair process creates new sections that may not have siblings
        morphio.Warning.only_child,

        # We are appending empty section and filling them later on
        morphio.Warning.appending_empty_section,
    )

    for warning in ignored_warnings:
        morphio.set_ignored_warning(warning, True)

    if axons is None:
        axons = []

    if params is None:
        params = _PARAMS

    obj = Repair(inputfile, axons=axons, seed=seed, cut_leaves_coordinates=cut_leaves_coordinates,
                 legacy_detection=legacy_detection, repair_flags=repair_flags,
                 apical_point=apical_point, params=params)
    obj.run(outputfile, plot_file=plot_file)

    for warning in ignored_warnings:
        morphio.set_ignored_warning(warning, False)
