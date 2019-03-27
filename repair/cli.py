'''The morph-tool command line launcher'''
import logging

import click

logging.basicConfig()
logger = logging.getLogger('repair')
logger.setLevel(logging.DEBUG)


@click.group()
def cli():
    '''The CLI entry point'''


@cli.command(short_help='unravel')
@click.argument('input_file', type=click.Path(exists=True, file_okay=True))
@click.argument('output_file')
@click.option('--window-half-length', default=5)
@click.option('--quiet/--no-quiet', default=False)
def unravel(input_file, output_file, window_half_length, quiet):
    '''Unravel a cell'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.unravel import unravel  # pylint: disable=redefined-outer-name

    unravel(input_file, window_half_length=window_half_length).write(output_file)


@cli.command(short_help='repair')
@click.argument('input_file', type=click.Path(exists=True, file_okay=True))
@click.argument('output_file')
@click.option('--quiet/--no-quiet', default=False)
def cell(input_file, output_file, quiet):
    '''Repair dendrites of a cut neuron'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.main import repair

    repair(input_file, output_file)


@cli.command(short_help='repair all')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('--quiet/--no-quiet', default=False)
def folder(input_dir, output_dir, quiet):
    '''Repair dendrites of all neurons in a directory'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.main import repair_all
    repair_all(input_dir, output_dir)


@cli.command(short_help='Generate PDF')
@click.argument('input_dir')
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.argument('old_repair_dir', default=None)
@click.option('--quiet/--no-quiet', default=False)
def report(input_dir, output_dir, old_repair_dir, quiet):
    '''Generate a PDF with plots of pre and post repair neurons'''
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG)
    from repair.view import view_all
    view_all(input_dir, output_dir, old_repair_dir)
