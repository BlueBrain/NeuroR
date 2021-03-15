"""Detect cut leaves with new algo."""
from itertools import product
import numpy as np
from neurom.core.dataformat import COLS
from neuror.cut_plane.planes import HalfSpace


def _get_cut_leaves(plane, morphology, bin_width, percentile_threshold):
    """Compute the cut leaves from a plane."""
    # get the cut leaves
    leaves = np.array([section for section in morphology.iter() if len(section.children) == 0])
    leaves_coord = np.array([leaf.points[-1, COLS.XYZ] for leaf in leaves])
    cut_filter = plane.distance(leaves_coord) < bin_width
    cut_leaves = leaves[cut_filter]

    # compute the min cut leave given the percentile
    projected_uncut_leaves = plane.project_on_directed_normal(leaves_coord[~cut_filter])
    _min, _max = min(projected_uncut_leaves), max(projected_uncut_leaves)
    bins = np.arange(_min, _max, bin_width)
    _dig = np.digitize(projected_uncut_leaves, bins)
    leaves_threshold = np.percentile(np.unique(_dig, return_counts=True)[1], percentile_threshold)

    quality = len(cut_leaves) - leaves_threshold
    if quality > 0:
        return leaves_coord[cut_filter], quality
    else:
        return None, None


def find_cut_leaves(
    morph,
    bin_width=3,
    percentile_threshold=70.0,
    searched_axes=("Z",),
    searched_half_spaces=(-1, 1),
):
    """Find all cut leaves for cuts with strong signal for real cut.

    The algorithm works as follow. Given the searched_axes and searched_half_spaces,
    a list of candidate cuts is created, consisting of a slice with bin_width adjusted
    to the most extreme points of the morphology in the direction.
    Each cut contains a set of leaves, which are considered as cut leaves if there quality
    is positive. The quality of a cut is defined as a cut for which the number of leaves
    in the cut, minus the 'percentile_threshold' percentile of the distribution of the number
    of leaves in all other slices of bin_width size of the morphology.
    More explicitely, if a cut has more leaves than most of other possible cuts of the same size,
    it is likely to be a real cut from an invitro slice.

    Note that all cuts can be valid, thus cut leaves can be on both sides.

    Args:
        morph (morphio.Morphology): morphology
        bin_width: the bin width
        percentile_threshold: the minimum percentile of leaves counts in bins
        searched_axes: x, y or z. Specify the planes for which to search the cut plane
        searched_half_spaces: A negative value means the morphology lives
                on the negative side of the plane, and a positive one the opposite.
    Returns:
        ndarray: cut leaves
        list: list of qualities in dicts with axis an side for each
    """
    # create planes
    planes = [
        HalfSpace(
            int(axis.upper() == "X"),
            int(axis.upper() == "Y"),
            int(axis.upper() == "Z"),
            0,
            upward=(side > 0),
        )
        for axis, side in product(searched_axes, searched_half_spaces)
    ]

    # set the plane coef_d as furthest morphology point
    for plane, (axis, side) in zip(planes, product(searched_axes, searched_half_spaces)):
        points = np.vstack([section.points for section in morph.iter()])
        projected_points = plane.project_on_directed_normal(points)
        plane.coefs[3] = -side * np.min(projected_points, axis=0)

    # find the leaves
    cuts = [_get_cut_leaves(plane, morph, bin_width, percentile_threshold) for plane in planes]

    # return only leaves of planes with valid cut
    leaves = [leave for leave, _ in cuts if leave is not None]
    qualities = [
        {"axis": axis, "side": side, "quality": np.around(quality, 3)}
        for (_, quality), (axis, side) in zip(cuts, product(searched_axes, searched_half_spaces))
        if quality is not None
    ]
    return np.vstack(leaves) if leaves else [], qualities
