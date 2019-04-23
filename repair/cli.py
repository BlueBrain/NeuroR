'''The morph-tool command line launcher'''
import logging

import click

logging.basicConfig()
logger = logging.getLogger('repair')
logger.setLevel(logging.DEBUG)


@click.group()
def cli():
    '''The CLI entry point'''


@cli.command(short_help='Unravel one morphology')
@click.argument('input_file', type=click.Path(exists=True, file_okay=True))
@click.argument('output_file')
@click.option('--window-half-length', default=5)
@click.option('--quiet/--no-quiet', default=False)
def unravel_cell(input_file, output_file, window_half_length, quiet):
    '''Unravel a cell'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.unravel import unravel  # pylint: disable=redefined-outer-name

    unravel(input_file, window_half_length=window_half_length).write(output_file)


@cli.command(short_help='Unravel all morphologies in a folder')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('--window-half-length', default=5)
@click.option('--quiet/--no-quiet', default=False)
def unravel_folder(input_dir, output_dir, window_half_length, quiet):
    '''Unravel all cells in a folder'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.unravel import unravel_all

    unravel_all(input_dir, output_dir, window_half_length)


@cli.command(short_help='repair')
@click.argument('input_file', type=click.Path(exists=True, file_okay=True))
@click.argument('output_file')
@click.option('--plane',
              type=click.Path(exists=True, file_okay=True), default=None,
              help=('A custom cut plane to use. '
                    'Cut planes are created by the code hosted at '
                    'bbpcode.epfl.ch/nse/cut-plane using the CLI command "cut-plane compute" '))
@click.option('--quiet/--no-quiet', default=False)
def cell(input_file, output_file, plane, quiet):
    '''Repair dendrites of a cut neuron'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.main import repair

    repair(input_file, output_file, plane=plane)


@cli.command(short_help='repair all')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('--planes',
              type=click.Path(exists=True, file_okay=True), default=None,
              help=('A folder containing cut planes for each morphology. '
                    'The filename must be the morphology filename followed by ".json". '
                    'Cut planes are created by the code hosted at '
                    'bbpcode.epfl.ch/nse/cut-plane using the CLI command "cut-plane compute" '))
@click.option('--quiet/--no-quiet', default=False)
def folder(input_dir, output_dir, planes, quiet):
    '''Repair dendrites of all neurons in a directory'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.main import repair_all
    repair_all(input_dir, output_dir, planes_dir=planes)


@cli.command(short_help='Generate PDF')
@click.argument('folders', nargs=-1)
@click.option('--title', '-t', multiple=True)
@click.option('--quiet/--no-quiet', default=False)
def report(folders, title, quiet):
    '''Generate a PDF with plots of pre and post repair neurons'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.view import view_all
    if not folders:
        print('Need to pass at least one folder')
        return

    if title:
        assert len(title) == len(folders)
    else:
        title = ['Plot {}'.format(i) for i in range(1, len(folders) + 1)]
    view_all(folders, title)
