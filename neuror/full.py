'''
The module to run the full repair
'''
import logging
from pathlib import Path

import pandas as pd
from morph_tool.utils import iter_morphology_files

from neuror.main import repair

L = logging.getLogger('neuror')


def repair_all(input_dir, output_dir, seed=0, axons=None, cut_points_dir=None, plots_dir=None):
    '''Repair all morphologies in input folder'''
    for inputfilename in iter_morphology_files(input_dir):
        outfilename = Path(output_dir, inputfilename.name)
        if cut_points_dir:
            cut_points = pd.read_csv(Path(cut_points_dir, inputfilename.name).with_suffix('.csv'))
        else:
            cut_points = None

        if plots_dir is not None:
            name = 'neuron_{}.html'.format(Path(inputfilename).stem.replace(' ', '_'))
            plot_file = str(Path(plots_dir, name))
        else:
            plot_file = None

        try:
            repair(inputfilename, outfilename,
                   seed=seed, axons=axons, cut_leaves_coordinates=cut_points, plot_file=plot_file)
        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning('%s failed', inputfilename)
            L.warning(e, exc_info=True)
