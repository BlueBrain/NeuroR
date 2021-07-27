'''The morph-tool command line launcher'''
import json
import shutil
import logging
import os
from pathlib import Path
from pprint import pprint

import click
from morph_tool.utils import iter_morphology_files
from morphio.mut import Morphology  # pylint: disable=import-error
from neurom import load_morphology
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


@cli.group()
def error_annotation():
    '''CLI utilities related to error annotations'''


@error_annotation.command(short_help='Annotate errors on a morphology')
@click.argument('input_file', type=click.Path(exists=True, file_okay=True))
@click.argument('output_file')
@click.option('--error_summary_file', type=click.Path(file_okay=True), default='error_summary.json',
              help='Path to json file to save error summary')
@click.option('--marker_file', type=click.Path(file_okay=True), default='markers.json',
              help='Path to json file to save markers')
def file(input_file, output_file, error_summary_file, marker_file):
    '''Annotate errors on a morphology.'''
    from neuror.sanitize import annotate_neurolucida

    if Path(input_file).suffix not in ['.asc', '.ASC']:
        raise Exception('Only .asc/.ASC files are allowed, please convert with morph-tool.')

    annotations, summary, markers = annotate_neurolucida(input_file)
    shutil.copy(input_file, output_file)
    with open(output_file, 'a') as morph_file:
        morph_file.write(annotations)
    with open(error_summary_file, 'w') as summary_file:
        json.dump(summary, summary_file, cls=NeuromJSON)
    with open(marker_file, 'w') as m_file:
        json.dump(markers, m_file, cls=NeuromJSON)


@error_annotation.command(short_help='Annotate errors on morphologies')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('--error_summary_file', type=click.Path(file_okay=True), default='error_summary.json',
              help='Path to json file to save error summary')
@click.option('--marker_file', type=click.Path(file_okay=True), default='markers.json',
              help='Path to json file to save markers')
def folder(input_dir, output_dir, error_summary_file, marker_file):
    '''Annotate errors on a morphologies in a folder.'''
    from neuror.sanitize import annotate_neurolucida_all

    output_dir = Path(output_dir)
    morph_paths = list(iter_morphology_files(input_dir))
    annotations, summaries, markers = annotate_neurolucida_all(morph_paths)
    for morph_path, annotation in annotations.items():
        output_file = output_dir / Path(morph_path).name
        shutil.copy(morph_path, output_file)
        with open(output_file, 'a') as morph_file:
            morph_file.write(annotation)
    with open(error_summary_file, 'w') as summary_file:
        json.dump(summaries, summary_file, indent=4, cls=NeuromJSON)
    with open(marker_file, 'w') as m_file:
        json.dump(markers, m_file, cls=NeuromJSON)


# pylint: disable=function-redefined
@repair.command(short_help='Repair one morphology')
@click.argument('input_file', type=click.Path(exists=True, file_okay=True))
@click.argument('output_file')
@click.option('--plot_file', type=click.Path(file_okay=True), default=None,
              help='Where to save the plot')
@click.option('-a', '--axon-donor', multiple=True,
              help='A morphology that provides a reference axon')
@click.option('--cut-file',
              type=click.Path(exists=True, file_okay=True), default=None,
              help=('Path to a CSV whose columns represents the X, Y and Z '
                    'coordinates of points from which to start the repair'))
def file(input_file, output_file, plot_file, axon_donor, cut_file):
    '''Repair dendrites of a cut neuron'''
    import pandas

    from neuror.main import repair  # pylint: disable=redefined-outer-name

    if cut_file:
        cut_points = pandas.read_csv(Path(cut_file).with_suffix('.csv')).values
    else:
        cut_points = None
    repair(input_file, output_file, axons=axon_donor, cut_leaves_coordinates=cut_points,
           plot_file=plot_file)


# pylint: disable=function-redefined
@repair.command(short_help='Repair all morphologies in a folder')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('--plot_dir', default=None, type=click.Path(exists=True, file_okay=False,
                                                          writable=True))
@click.option('-a', '--axon-donor', multiple=True,
              help='A morphology that provides a reference axon')
@click.option('--cut-file-dir',
              type=click.Path(exists=True, file_okay=True), default=None,
              help=('A dir with the cut points CSV file for each morphology. '
                    'See also "neuror cut-plane repair file --help".'))
def folder(input_dir, output_dir, plot_dir, axon_donor, cut_file_dir):
    '''Repair dendrites of all neurons in a directory'''
    from neuror.full import repair_all
    repair_all(input_dir, output_dir, axons=axon_donor, cut_points_dir=cut_file_dir,
               plots_dir=plot_dir)


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
@click.option('--nprocesses', default=1, help='The number of processes to spawn')
def folder(input_folder, output_folder, nprocesses):
    '''Sanitize all morphologies in the folder.'''
    from neuror.sanitize import sanitize_all
    sanitize_all(input_folder, output_folder, nprocesses=nprocesses)


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
        plot(load_morphology(filename), result)


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
    for inputfilename in iter_morphology_files(input_dir):
        L.info('Seaching cut plane for file: %s', inputfilename)
        outfilename = os.path.join(output_dir, inputfilename.with_suffix('.json'))
        try:
            _export_cut_plane(inputfilename, outfilename, width, display=display,
                              searched_axes=(plane or ('X', 'Y', 'Z')),
                              fix_position=None)
        except Exception as e:  # noqa, pylint: disable=broad-except
            L.warning('Cut plane computation for %s failed', inputfilename)
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
