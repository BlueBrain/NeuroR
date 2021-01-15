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


def annotate_single_morphology(morph_path):
    """Annotate errors on a morphology.

    Args:
        morph_path (str): absolute path to an ascii morphology

    Returns:
        annotations to append to .asc file
        dict of error summary
    """
    neuron = load_neuron(morph_path)
    results = [checker(neuron) for checker in CHECKERS]
    summary = {
        setting["name"]: len(result.info)
        for result, setting in zip(results, CHECKERS.values())
        if result.info
    }
    return annotate(results, CHECKERS.values()), summary


def annotate_morphologies(morph_paths):
    """Annotate errors on a list of morphologies.

    Args:
        morph_paths (list): list of str of paths to morphologies.

    Returns:
        dict annotations to append to .asc file (morph_path as keys)
        dict of dict of error summary (morph_path as keys)
    """
    summaries = {}
    annotations = {}
    for morph_path in morph_paths:
        try:
            annotations[str(morph_path)], summaries[str(morph_path)] = annotate_single_morphology(
                morph_path
            )
        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning("%s failed", morph_path)
            L.warning(e, exc_info=True)
    return annotations, summaries