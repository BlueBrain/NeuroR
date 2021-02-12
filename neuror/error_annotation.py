"""Annotation errors on morphologies."""
from functools import partial
import logging
from neurom.check import neuron_checks as nc
from neurom.apps.annotate import annotate
from neurom import load_neuron

L = logging.getLogger("neuror")

CHECKERS = {
    nc.has_no_fat_ends: {"name": "fat end", "label": "Circle3", "color": "Blue"},
    partial(nc.has_no_jumps, axis="z"): {"name": "zjump", "label": "Circle2", "color": "Green"},
    nc.has_no_narrow_start: {"name": "narrow start", "label": "Circle1", "color": "Blue"},
    nc.has_no_dangling_branch: {"name": "dangling", "label": "Circle6", "color": "Magenta"},
    nc.has_multifurcation: {"name": "Multifurcation", "label": "Circle8", "color": "Yellow"},
}


def annotate_single_morphology(morph_path, checkers=None):
    """Annotate errors on a morphology.

    Args:
        morph_path (str): absolute path to an ascii morphology
        checkers (dict): dict of checker functons from neurom with function as keys
            and marker data in a dict as values, if None, default checkers are used

    Default checkers include:
        - fat ends
        - z-jumps
        - narrow start
        - dangling branch
        - multifurcation

    Returns:
        annotations to append to .asc file
        dict of error summary
        dict of error markers
    """
    if checkers is None:
        checkers = CHECKERS

    neuron = load_neuron(morph_path)
    results = [checker(neuron) for checker in checkers]
    markers = [
        dict(setting, data=result.info)
        for result, setting in zip(results, checkers.values())
        if not result.status
    ]
    summary = {
        setting["name"]: len(result.info)
        for result, setting in zip(results, checkers.values())
        if result.info
    }
    return annotate(results, checkers.values()), summary, markers


def annotate_morphologies(morph_paths):
    """Annotate errors on a list of morphologies.

    Args:
        morph_paths (list): list of str of paths to morphologies.

    Returns:
        dict annotations to append to .asc file (morph_path as keys)
        dict of dict of error summary (morph_path as keys)
        dict of dict of markers (morph_path as keys)
    """
    summaries, annotations, markers = {}, {}, {}
    for morph_path in morph_paths:
        try:
            (
                annotations[str(morph_path)],
                summaries[str(morph_path)],
                markers[str(morph_path)],
            ) = annotate_single_morphology(morph_path)
        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning("%s failed", morph_path)
            L.warning(e, exc_info=True)
    return annotations, summaries, markers
