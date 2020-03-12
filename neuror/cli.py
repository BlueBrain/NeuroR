'''The morph-tool command line launcher'''
import json
import logging
import os
from pprint import pprint

import click
from morph_tool.utils import iter_morphology_files
from morphio.mut import Morphology  # pylint: disable=import-error
from neurom import load_neuron
from neurom.utils import NeuromJSON

from neuror.cut_plane.detection import CutPlane
from neuror.unravel import DEFAULT_WINDOW_HALF_LENGTH

logging.basicConfig()
L = logging.getLogger('neuror')


@click.group()
@click.option('-v', '--verbose', count=True, default=0,
              help='-v for INFO, -vv for DEBUG')
def cli(verbose):
    '''The CLI entry point'''
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    L.setLevel(level)


@cli.group()
def unravel():
    '''CLI utilities related to unravelling'''


@cli.group()
def cut_plane():
    '''CLI utilities related to cut-plane repair'''


@cli.group()
def sanitize():
    '''CLI utilities related to sanitizing raw morphologies.

    It currently only deals with removing duplicate points but it may do
    more in the future.
    '''


@cut_plane.group()
def compute():
    '''CLI utilities to detect cut planes'''


@cut_plane.group()
def repair():
    '''CLI utilities to repair cut planes'''

# pylint: disable=function-redefined
@repair.command(short_help='Repair one morphology')
@click.argument('input_file', type=click.Path(exists=True, file_okay=True))
@click.argument('output_file')
@click.option('--plot_file', type=click.Path(file_okay=True), default=None,
              help='Where to save the plot')
@click.option('-a', '--axon-donor', multiple=True,
              help='A morphology that provides a reference axon')
@click.option('--plane',
              type=click.Path(exists=True, file_okay=True), default=None,
              help=('A custom cut plane to use. '
                    'Cut planes are created by the code hosted at '
                    'bbpcode.epfl.ch/nse/cut-plane using the CLI command "cut-plane compute" '))
def file(input_file, output_file, plot_file, axon_donor, plane):
    '''Repair dendrites of a cut neuron'''
    from neuror.main import repair  # pylint: disable=redefined-outer-name
    repair(input_file, output_file, axons=axon_donor, plane=plane, plot_file=plot_file)


# pylint: disable=function-redefined
@repair.command(short_help='Repair all morphologies in a folder')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('--plot_dir', default=None, type=click.Path(exists=True, file_okay=False,
                                                          writable=True))
@click.option('-a', '--axon-donor', multiple=True,
              help='A morphology that provides a reference axon')
@click.option('--planes',
              type=click.Path(exists=True, file_okay=True), default=None,
              help=('A folder containing cut planes for each morphology. '
                    'The filename must be the morphology filename followed by ".csv". '
                    'Cut planes are created by the code hosted at '
                    'bbpcode.epfl.ch/nse/cut-plane using the CLI command "cut-plane compute" '))
def folder(input_dir, output_dir, plot_dir, axon_donor, planes):
    '''Repair dendrites of all neurons in a directory'''
    from neuror.full import repair_all
    repair_all(input_dir, output_dir, axons=axon_donor, planes_dir=planes, plots_dir=plot_dir)


# pylint: disable=too-many-arguments
@repair.command(short_help='Unravel and repair one morphology')
@click.argument('root_dir', type=click.Path(exists=True, file_okay=True))
@click.option('--window-half-length', default=DEFAULT_WINDOW_HALF_LENGTH)
@click.option('--raw-dir', default=None, help='Folder of input raw morphologies')
@click.option('--raw-planes-dir', default=None, help='Folder of input raw cut planes')
@click.option('--unravelled-dir', default=None, help='Folder of unravelled morphologies')
@click.option('--unravelled-planes-dir', default=None, help='Folder of unravelled cut planes')
@click.option('--repaired-dir', default=None, help='Folder of repaired morphologies')
@click.option('--plots-dir', default=None, help='Folder of plots')
@click.option('--seed', default=0, help='The numpy.random seed')
def full(root_dir, window_half_length, raw_dir, raw_planes_dir, unravelled_dir,
         unravelled_planes_dir, repaired_dir, plots_dir, seed):
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
    from neuror.full import full  # pylint: disable=redefined-outer-name
    full(root_dir,
         seed=seed,
         window_half_length=window_half_length,
         raw_dir=raw_dir,
         raw_planes_dir=raw_planes_dir,
         unravelled_dir=unravelled_dir,
         unravelled_planes_dir=unravelled_planes_dir,
         repaired_dir=repaired_dir,
         plots_dir=plots_dir)


@unravel.command(short_help='Unravel one morphology')
@click.argument('input_file', type=click.Path(exists=True, file_okay=True))
@click.argument('output_file')
@click.option('--mapping-file', type=click.Path(file_okay=True), default=None,
              help=('Path to the file that contains the coordinate mapping before '
                    'and after unravelling'))
@click.option('--window-half-length', default=DEFAULT_WINDOW_HALF_LENGTH)
def file(input_file, output_file, mapping_file, window_half_length):
    '''Unravel a cell'''
    from neuror.unravel import unravel  # pylint: disable=redefined-outer-name
    neuron, mapping = unravel(input_file, window_half_length=window_half_length)
    neuron.write(output_file)
    if mapping_file is not None:
        if not mapping_file.lower().endswith('csv'):
            raise Exception('the mapping file must end with .csv')
        mapping.to_csv(mapping_file)


@unravel.command(short_help='Unravel all morphologies in a folder')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('--raw-plane-dir', type=click.Path(exists=True, file_okay=False), default=None,
              help='The path to raw cut planes (if None, defaults to INPUT_DIR/planes)')
@click.option('--unravelled-plane-dir', type=click.Path(exists=True, file_okay=False), default=None,
              help='The path to unravelled cut planes (if None, defaults to OUTPUT_DIR/planes)')
@click.option('--window-half-length', default=DEFAULT_WINDOW_HALF_LENGTH)
def folder(input_dir, output_dir, raw_planes_dir, unravelled_planes_dir, window_half_length):
    '''Unravel all cells in a folder'''
    from neuror.unravel import unravel_all
    unravel_all(input_dir, output_dir, raw_planes_dir, unravelled_planes_dir,
                window_half_length=window_half_length)


@cli.command(short_help='Generate PDF with morphology plots')
@click.argument('folders', nargs=-1)
@click.option('--title', '-t', multiple=True)
def report(folders, title):
    '''Generate a PDF with plots of pre and post repair neurons'''
    from neuror.view import view_all
    if not folders:
        print('Need to pass at least one folder')
        return

    if title:
        assert len(title) == len(folders)
    else:
        title = ['Plot {}'.format(i) for i in range(1, len(folders) + 1)]
    view_all(folders, title)


@cli.command(short_help='Fix zero diameters')
@click.argument('input_file')
@click.argument('output_file')
def zero_diameters(input_file, output_file):
    '''Output a morphology where the zero diameters have been removed'''
    from neuror.zero_diameter_fixer import fix_zero_diameters
    neuron = Morphology(input_file)
    fix_zero_diameters(neuron)
    neuron.write(output_file)


# pylint: disable=function-redefined
@sanitize.command(short_help='Sanitize a morphology')
@click.argument('input_file')
@click.argument('output_file')
def file(input_file, output_file):
    '''Sanitize a raw morphology.'''
    from neuror.sanitize import sanitize  # pylint: disable=redefined-outer-name
    sanitize(input_file, output_file)


# pylint: disable=function-redefined
@sanitize.command(short_help='Sanitize all morphologies in a folder')
@click.argument('input_folder')
@click.argument('output_folder')
def folder(input_folder, output_folder):
    '''Sanitize all morphologies in the folder.'''
    from neuror.sanitize import sanitize_all
    sanitize_all(input_folder, output_folder)


@cut_plane.group()
def compute():
    '''CLI utilities to compute cut planes'''


def _check_results(result):
    '''Check the result status'''
    if not result:
        L.error('Empty results')
        return -1

    status = result.get('status')
    if status.lower() != 'ok':
        L.warning('Incorrect status: %s', status)
        return 1

    return 0


@compute.command(short_help='Find a 3D cut plane by providing a manual hint')
@click.argument('filename', type=str, required=True)
def hint(filename):
    """Launch the app to manually search for the cut plane. After running the command,
    either click the link in the console or open your browser and go to the address
    shown in the console.

    Example::

       cut-plane compute hint ./tests/data/Neuron_slice.h5
"""
    from neuror.cut_plane.viewer import app, set_neuron
    set_neuron(filename)
    app.run_server(debug=True)


def _export_cut_plane(filename, output, width, display, searched_axes, fix_position):
    '''Find the position of the cut plane (it assumes the plane is aligned along X, Y or Z) for
    morphology FILENAME.

    It returns the cut plane and the positions of all cut terminations.
'''
    if os.path.isdir(filename):
        raise Exception('filename ({}) should not be a directory'.format(filename))

    result = CutPlane.find(filename,
                           width,
                           searched_axes=searched_axes,
                           fix_position=fix_position).to_json()

    if not output:
        pprint(result)
    else:
        with open(output, 'w') as output_file:
            json.dump(result, output_file, cls=NeuromJSON)

    _check_results(result)

    if display:
        from neuror.cut_plane.detection import plot
        plot(load_neuron(filename), result)


@compute.command(short_help='Compute a cut plane for morphology FILENAME')
@click.argument('filename', type=str, required=True)
@click.option('-o', '--output',
              help=('Output name for the JSON file (default=STDOUT)'))
@click.option('-w', '--width', type=float, default=3,
              help='The bin width (in um) of the 1D distributions')
@click.option('-d', '--display', is_flag=True, default=False,
              help='Flag to enable the display control plots')
@click.option('-p', '--plane', type=click.Choice(['x', 'y', 'z']), default=None,
              help='Force the detection along the given plane')
@click.option('--position', type=float, default=None,
              help='Force the position. Requires --plane to be set as well')
def file(filename, output, width, display, plane, position):
    '''Find the position of the cut plane (it assumes the plane is aligned along X, Y or Z) for
    morphology FILENAME.

    It returns the cut plane and the positions of all cut terminations.
    Compute a cut plane and outputs it either as a STDOUT stream or in a file
    if ``-o`` option is passed.
    The control plots can be displayed by passing the ``-d`` option.
    The bin width can be changed with the ``-w`` option (see below)

    Description of the algorithm:

    #. The distribution of all points along X, Y and Z is computed
       and put into 3 histograms.
    #. For each histogram we look at the first and last empty bins
       (that is, the last bin before the histogram starts rising,
       and the first after it reaches zero again). Under the assumption
       that there is no cut plane, the posteriori probability
       of observing this empty bin given the value of the non-empty
       neighbour bin is then computed.
    #. The lowest probability of the 6 probabilities (2 for each axes)
       corresponds to the cut plane.

    .. image:: /_images/distrib_1d.png
       :align: center

    Returns:
       A dictionary with the following items:

       :status: 'ok' if everything went right, else an informative string
       :cut_plane: a tuple (plane, position) where 'plane' is 'X', 'Y' or 'Z'
           and 'position' is the position
       :cut_leaves: an np.array of all termination points in the cut plane
       :figures: if 'display' option was used, a dict where values are tuples (fig, ax)
           for each figure
       :details: A dict currently only containing -LogP of the bin where the cut plane was found

    Example::

       cut-plane compute file -d tests/data/Neuron_slice.h5  -o my-plane.json -w 10
    '''
    _export_cut_plane(filename, output, width, display, plane or ('x', 'y', 'z'), position)


@compute.command(short_help='Compute cut planes for morphologies located in INPUT_DIR')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('-w', '--width', type=float, default=3,
              help='The bin width (in um) of the 1D distributions')
@click.option('-d', '--display', is_flag=True, default=False,
              help='Flag to enable the display control plots')
@click.option('-p', '--plane', type=click.Choice(['x', 'y', 'z']), default=None,
              help='Force the detection along the given plane')
def folder(input_dir, output_dir, width, display, plane):
    '''Compute cut planes for all morphology in INPUT_DIR and save them into OUTPUT_DIR

    See "cut-plane compute --help" for more information'''
    for f in iter_morphology_files(input_dir):
        L.info('Seaching cut plane for file: %s', f)
        inputfilename = os.path.join(input_dir, f)
        outfilename = os.path.join(output_dir, os.path.basename(f) + '.json')
        try:
            _export_cut_plane(inputfilename, outfilename, width, display=display,
                              searched_axes=(plane or ('X', 'Y', 'Z')),
                              fix_position=None)
        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning('Cut plane computation for %s failed', f)
            L.warning(e, exc_info=True)


@cut_plane.command()
@click.argument('out_filename', nargs=1)
@click.argument('plane_paths', nargs=-1)
def join(out_filename, plane_paths):
    '''Merge cut-planes from json files located at PLANE_PATHS into one.
    The output is writen at OUT_FILENAME

    Example::

       cut-plane join result.json plane1.json plane2.json plane3.json
    '''
    data = []
    for plane in plane_paths:
        with open(plane) as in_f:
            data += json.load(in_f)

    with open(out_filename, 'w') as out_f:
        json.dump(data, out_f)
