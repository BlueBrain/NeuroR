'''
The module to run the full repair
'''
import logging
import os
from pathlib import Path

from morph_tool.utils import iter_morphology_files

from neuror.main import repair
from neuror.unravel import unravel_all

L = logging.getLogger('neuror')


def _get_folders(root_dir,
                 raw_dir=None,
                 raw_planes_dir=None,
                 unravelled_dir=None,
                 unravelled_planes_dir=None,
                 repaired_dir=None,
                 plots_dir=None):
    '''Get folder paths with sensible defaults'''
    folders = dict()
    folders['raw'] = raw_dir or str(Path(root_dir, 'raw'))
    folders['raw_planes'] = raw_planes_dir or str(Path(folders['raw'], 'planes'))
    folders['unravelled'] = unravelled_dir or str(Path(root_dir, 'unravelled'))
    folders['unravelled_planes'] = unravelled_planes_dir or str(
        Path(folders['unravelled'], 'planes'))
    folders['repaired'] = repaired_dir or str(Path(root_dir, 'repaired'))
    folders['plots'] = plots_dir or str(Path(root_dir, 'plots'))
    return folders


def repair_all(input_dir, output_dir, seed=0, axons=None, planes_dir=None, plots_dir=None):
    '''Repair all morphologies in input folder'''
    for f in iter_morphology_files(input_dir):
        L.info(f)
        inputfilename = Path(input_dir, f)
        outfilename = Path(output_dir, os.path.basename(f))
        if planes_dir:
            plane = str(Path(planes_dir, inputfilename.name).with_suffix('.json'))
        else:
            plane = None

        if plots_dir is not None:
            name = 'neuron_{}.html'.format(Path(inputfilename).stem.replace(' ', '_'))
            plot_file = str(Path(plots_dir, name))
        else:
            plot_file = None

        try:
            repair(str(inputfilename), str(outfilename),
                   seed=seed, axons=axons, plane=plane, plot_file=plot_file)
        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning('%s failed', f)
            L.warning(e, exc_info=True)


# pylint: disable=too-many-arguments
def full(root_dir,
         seed=0,
         window_half_length=5,
         raw_dir=None,
         raw_planes_dir=None,
         unravelled_dir=None,
         unravelled_planes_dir=None,
         repaired_dir=None,
         plots_dir=None):
    '''
    Perform the unravelling and repair in ROOT_DIR:

    1) perform the unravelling of the neuron
    2) update the cut points position after unravelling and writes it
       in the unravelled/planes folder
    3) repair the morphology

    All output directories can be overriden with the corresponding arguments.
    Here is the default structure:

    - raw_dir: ROOT_DIR/raw/ with all raw morphologies to repair
    - raw_planes_dir: RAW_DIR/planes with all cut planes
    - unravelled_dir: ROOT_DIR/unravelled/ where unravelled morphologies will be written
    - unravelled_planes_dir: UNRAVELLED_DIR/planes where unravelled planes will be written
    - repaired_dir: ROOT_DIR/repaired/ where repaired morphologies will be written
    - plots_dir: ROOT_DIR/plots where the plots will be put
    '''

    folders = _get_folders(root_dir,
                           raw_dir,
                           raw_planes_dir,
                           unravelled_dir,
                           unravelled_planes_dir,
                           repaired_dir,
                           plots_dir)

    for folder in ['raw', 'raw_planes']:
        if not os.path.exists(folders[folder]):
            raise Exception('%s does not exists' % folders[folder])

    for folder in ['unravelled', 'unravelled_planes', 'repaired', 'plots']:
        if not os.path.exists(folders[folder]):
            os.mkdir(folders[folder])

    unravel_all(folders['raw'],
                folders['unravelled'],
                folders['raw_planes'],
                folders['unravelled_planes'],
                window_half_length=window_half_length)
    repair_all(folders['unravelled'],
               folders['repaired'],
               seed=seed,
               planes_dir=folders['unravelled_planes'],
               plots_dir=folders['plots'])

    try:
        from neuror.view import view_all
        view_all([folders['raw'], folders['unravelled'], folders['repaired']],
                 titles=['raw', 'unravelled', 'repaired'],
                 output_pdf=str(Path(folders['plots'], 'report.pdf')))
    except ImportError:
        L.warning('Skipping writing plots as [plotly] extra is not installed')
