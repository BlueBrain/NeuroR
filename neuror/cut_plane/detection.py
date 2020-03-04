'''Module for the detection of the cut plane'''
import json
import logging
from itertools import product
from operator import attrgetter
from pathlib import Path
from typing import List, Union

import numpy as np
import neurom as nm
from neurom import geom, iter_sections, load_neuron
from neurom.core import Tree
from neurom.core._neuron import Neuron
from neurom.core.dataformat import COLS
from scipy.optimize import minimize
from scipy.special import factorial

from neuror.cut_plane.planes import HalfSpace, PlaneEquation

L = logging.getLogger(__name__)


class CutPlane(HalfSpace):
    '''The cut plane class

    It is composed of a HalfSpace and a morphology
    The morphology is part of the HalfSpace, the cut space is
    the complementary HalfSpace
    '''

    def __init__(self, coefs: List[float],
                 upward: bool,
                 morphology: Union[str, Path, Neuron],
                 bin_width: float):
        '''Cut plane ctor.

        Args:
            coefs: the [abcd] coefficients of a plane equation: a X + b Y + c Z + d = 0
            upward: if true, the morphology points satisfy: a X + b Y + c Z + d > 0
                    else, they satisfy: a X + b Y + c Z + d < 0
            morphology: the morphology
            bin_width: the bin width
        '''
        super(CutPlane, self).__init__(coefs[0], coefs[1], coefs[2], coefs[3], upward)

        if isinstance(morphology, Neuron):
            self.morphology = morphology
        elif morphology:
            self.morphology = load_neuron(morphology)

        self.bin_width = bin_width
        self.cut_leaves_coordinates = None
        self.status = None
        self.minus_log_prob = None
        if morphology is not None:
            self._compute_cut_leaves_coordinates()
            self._compute_probabilities()
            self._compute_status()

    @classmethod
    def from_json(cls, cut_plane_obj, morphology=None):
        '''Factory constructor from a JSON file

        cut_plane_obj (dict|str|pathlib.Path): a cut plane
            It can be a python dictionary or a path to a json file that contains one
        '''
        assert isinstance(cut_plane_obj, (str, dict, Path))

        if not isinstance(cut_plane_obj, dict):
            with open(cut_plane_obj) as f:
                cut_plane_obj = json.load(f)

        data = cut_plane_obj['cut-plane']
        obj = CutPlane([data['a'], data['b'], data['c'], data['d']],
                       data['upward'],
                       morphology,
                       cut_plane_obj['details']['bin-width'])

        obj.cut_leaves_coordinates = np.array(cut_plane_obj['cut-leaves'])
        obj.status = cut_plane_obj['status']
        obj.minus_log_prob = cut_plane_obj['details']['-LogP']
        return obj

    # pylint: disable=arguments-differ
    @classmethod
    def from_rotations_translations(cls, transformations, morphology, bin_width):
        plane = PlaneEquation.from_rotations_translations(transformations)
        return cls(plane.coefs, True, morphology, bin_width)

    @classmethod
    def find(cls, neuron, bin_width=3,
             searched_axes=('X', 'Y', 'Z'),
             searched_half_spaces=(-1, 1),
             fix_position=None):
        """
        Find and return the cut plane that is oriented along X, Y or Z.
        6 potential positions are considered: for each axis, they correspond
        to the coordinate of the first and last point of the neuron.

        Description of the algorithm:
        1) The distribution of all points along X, Y and Z is computed
           and put into 3 histograms.

        2) For each histogram we look at the first and last empty bins
           (ie. the last bin before the histogram starts rising,
           and the first after it reaches zero again). Under the assumption
           that there is no cut plane, the posteriori probability
           of observing this empty bin given the value of the not-empty
           neighbour bin is then computed.
        3) The lowest probability of the 6 probabilities (2 for each axes)
           corresponds to the cut plane

        Args:
            neuron (Neuron|str|pathlib.Path): a Neuron object or path
            bin_width: The size of the binning

            display: where or not to display the control plots
                     Note: It is the user responsability to call matplotlib.pyplot.show()

            searched_axes: x, y or z. Specify the planes for which to search the cut plane

            searched_half_spaces: A negative value means the morphology lives
                on the negative side of the plane, and a positive one the opposite.

            fix_position: If not None, this is the position
                for which to search the cut plane. Only the orientation will be searched for.
                This can be useful to find the orientation of a plane whose position is known.

        Returns:
            A cut plane object
        """
        if not isinstance(neuron, Neuron):
            neuron = load_neuron(neuron)

        # pylint: disable=invalid-unary-operand-type
        coef_d = -fix_position if fix_position is not None else 0

        planes = [CutPlane([int(axis.upper() == 'X'), int(axis.upper() == 'Y'),
                            int(axis.upper() == 'Z'), coef_d],
                           upward=(side > 0),
                           morphology=neuron,
                           bin_width=bin_width)
                  for axis, side in product(searched_axes, searched_half_spaces)]

        best_plane = max(planes, key=attrgetter('minus_log_prob'))

        if fix_position is None:
            _, bins = best_plane.histogram()
            # The orientation of the plane is defined such as the morphology lives
            # on the positive side once coordinates are projected
            # (see HalfSpace.project_on_directed_normal)
            # So the first bin is where we look for the cut plane
            best_plane.coefs[3] = bins[0]
        else:
            best_plane.coefs[3] = coef_d
        return CutPlane(best_plane.coefs, best_plane.upward, neuron, best_plane.bin_width)

    @property
    def cut_sections(self):
        '''Returns sections that ends within the cut plane'''
        leaves = np.array([leaf
                           for neurite in self.morphology.neurites
                           for leaf in iter_sections(neurite, iterator_type=Tree.ileaf)])
        leaves_coord = [leaf.points[-1, COLS.XYZ] for leaf in leaves]
        return leaves[self.distance(leaves_coord) < self.bin_width]

    def _compute_cut_leaves_coordinates(self):
        '''Returns cut leaves coordinates'''
        leaves = [leaf.points[-1, COLS.XYZ] for leaf in self.cut_sections]
        self.cut_leaves_coordinates = np.vstack(leaves) if leaves else np.array([])

    def to_json(self):
        '''Return a dictionary with the following items:
            status: 'ok' if everything went right, else an informative string
            cut_plane: a tuple (plane, position) where 'plane' is 'X', 'Y' or 'Z'
                       and 'position' is the position
            cut_leaves: an np.array of all termination points in the cut plane
            figures: if 'display' option was used, a dict where values are tuples (fig, ax)
                     for each figure
            details: A dict currently only containing -LogP of the bin where the cut plane was found
        '''
        return {'cut-leaves': self.cut_leaves_coordinates,
                'status': self.status,
                'details': {'-LogP': self.minus_log_prob, 'bin-width': self.bin_width},
                'cut-plane': super(CutPlane, self).to_json()}

    def histogram(self):
        '''Get the point distribution projected along the normal to the plane

        Returns:
            a numpy.histogram
        '''
        points = _get_points(self.morphology)
        projected_points = self.project_on_directed_normal(points)
        min_, max_ = np.min(projected_points, axis=0), np.max(projected_points, axis=0)
        binning = np.arange(min_, max_ + self.bin_width, self.bin_width)
        return np.histogram(projected_points, bins=binning)

    def _compute_probabilities(self):
        '''Returns -log(p) where p is the a posteriori probabilities of the observed values
        in the bins X min, X max, Y min, Y max, Z min, Z max

        Parameters:
            hist: a dict of the X, Y and Z 1D histograms

        Returns: a dict of -log(p) values'''

        hist = self.histogram()
        if not hist[0].size:
            self.minus_log_prob = np.nan
        else:
            self.minus_log_prob = get_minus_log_p(0, hist[0][0])
        self._compute_status()
        return self.minus_log_prob

    def _compute_status(self):
        '''Returns ok if the probability that there is a cut plane is high enough'''
        _THRESHOLD = 50
        if np.isnan(self.minus_log_prob):
            self.status = 'The proba is NaN, something went wrong'
        elif self.minus_log_prob < _THRESHOLD:
            self.status = ('The probability that there is in fact NO '
                           'cut plane is high: -log(p) = {0} !').format(self.minus_log_prob)
        else:
            self.status = 'ok'


def _success_function(params, points, bin_width):
    '''The success function is low (=good) when the difference of points
    on the left side and right side of the plane is high'''
    plane = PlaneEquation.from_rotations_translations(params)
    n_left, n_right = plane.count_near_plane(points, bin_width)
    res = -abs(n_left - n_right)
    return res


def _minimize(x0, points, bin_width):
    '''Returns a tuple of the optimized values of
    (rot_x, rot_y, rot_z, transl_x, transl_y, transl_z)'''
    delta_angle = 10  # in degrees
    delta_transl = 10
    delta = np.array([delta_angle, delta_angle, delta_angle,
                      delta_transl, delta_transl, delta_transl])
    bounds_min = x0 - delta
    bounds_max = x0 + delta
    bounds = list(zip(bounds_min, bounds_max))
    result = minimize(_success_function,
                      x0=x0,
                      args=(points, bin_width),
                      bounds=bounds,
                      method='Nelder-Mead')

    if result.status:
        raise Exception(result.message)
    return result.x


def get_minus_log_p(k, mu):
    '''Compute -Log(p|k) where p is the a posteriori probability to observe k counts
    in bin given than the mean value was "mu":
    demo: p(k|mu) = exp(-mu) * mu**k / k!
    '''
    return mu - k * np.log(mu) + np.log(factorial(k))


def _get_points(neuron):
    return np.array([point
                     for neurite in (neuron.neurites or [])
                     for section in nm.iter_sections(neurite)
                     for point in section.points])


def plot(neuron, result):
    '''Plot the neuron, the cut plane and the cut leaves'''
    try:
        from plotly_helper.neuron_viewer import NeuronBuilder
        from plotly_helper.object_creator import scatter
        from plotly_helper.shapes import line
    except ImportError:
        raise ImportError(
            'neuror[plotly] is not installed.'
            ' Please install it by doing: pip install neuror[plotly]')

    bbox = geom.bounding_box(neuron)

    plane = result['cut-plane']

    for display_plane, idx in [('xz', 0), ('yz', 1), ('3d', None)]:
        builder = NeuronBuilder(neuron, display_plane, line_width=4, title='{}'.format(neuron.name))
        if idx is not None:
            if plane['a'] == 0 and plane['b'] == 0:
                builder.helper.add_shapes([
                    line(bbox[0][idx], -plane['d'], bbox[1][idx], -plane['d'], width=4)
                ])
            builder.helper.add_data({'a': scatter(result['cut-leaves'][:, [idx, 2]],
                                                  showlegend=False, width=5)})
        else:
            builder.helper.add_data({'a': scatter(result['cut-leaves'], width=2)})
        builder.plot()
