'''The morph-tool command line launcher'''
import logging

import click

logging.basicConfig()
logger = logging.getLogger('morph-repair')


@click.group()
@click.option('-v', '--verbose', count=True, default=0,
              help='-v for INFO, -vv for DEBUG')
def cli(verbose):
    '''The CLI entry point'''
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logger.setLevel(level)


@cli.group()
def unravel():
    '''CLI utilities related to unravelling'''


# pylint: disable=too-many-arguments
@cli.command(short_help='Unravel and repair one morphology')
@click.argument('root_dir', type=click.Path(exists=True, file_okay=True))
@click.option('--window-half-length', default=5)
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
    from morph_repair.full import full  # pylint: disable=redefined-outer-name
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
@click.option('--window-half-length', default=5)
def file(input_file, output_file, mapping_file, window_half_length):
    '''Unravel a cell'''
    from morph_repair.unravel import unravel  # pylint: disable=redefined-outer-name
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
@click.option('--window-half-length', default=5)
def folder(input_dir, output_dir, raw_planes_dir, unravelled_planes_dir, window_half_length):
    '''Unravel all cells in a folder'''
    from morph_repair.unravel import unravel_all
    unravel_all(input_dir, output_dir, window_half_length, raw_planes_dir, unravelled_planes_dir)


# pylint: disable=function-redefined
@cli.command(short_help='morph-repair')
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
    from morph_repair.main import repair
    repair(input_file, output_file, axons=axon_donor, plane=plane, plot_file=plot_file)


# pylint: disable=function-redefined
@cli.command(short_help='repair all')
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
    from morph_repair.full import repair_all
    repair_all(input_dir, output_dir, axons=axon_donor, planes_dir=planes, plots_dir=plot_dir)


@cli.command(short_help='Generate PDF')
@click.argument('folders', nargs=-1)
@click.option('--title', '-t', multiple=True)
def report(folders, title):
    '''Generate a PDF with plots of pre and post repair neurons'''
    from morph_repair.view import view_all
    if not folders:
        print('Need to pass at least one folder')
        return

    if title:
        assert len(title) == len(folders)
    else:
        title = ['Plot {}'.format(i) for i in range(1, len(folders) + 1)]
    view_all(folders, title)
