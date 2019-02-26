'''Generate output plots'''
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt

from neurom import load_neuron
from neurom.view.plotly import PlotBuilder
from neurom.view.view import plot_neuron

L = logging.getLogger('repair')


def view_all(before_repair_dir, after_repair_dir, old_repair_dir, output_pdf=None):
    '''Generate PDF report'''
    if not output_pdf:
        path = './plots'
        output_pdf = os.path.join(path, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(output_pdf)
    limits = {'x': [-150, 150], 'y': [-150, 150]}

    def plot(filename, subplot, title):
        ax = plt.subplot(subplot, facecolor='w', aspect='equal')
        ax.set_xlim(*limits['x'])
        ax.set_ylim(*limits['y'])
        neuron = load_neuron(filename)
        plot_neuron(ax, neuron, **kwargs)
        ax.set_title(title)
        ax.set_aspect('equal', 'datalim')

    files = os.listdir(after_repair_dir)
    for f in files:
        L.info(f)
        try:
            kwargs = {'plane': 'xz'}
            fig = plt.figure()
            plot(os.path.join(before_repair_dir, f), 131, 'raw')
            plot(os.path.join(old_repair_dir, f), 132, 'old repair')
            plot(os.path.join(after_repair_dir, f), 133, 'new repair')
            fig.suptitle(f)
            pp.savefig()
        except Exception as e:  # pylint: disable=broad-except
            L.info("e: %s", e)
            L.info('failu: %s', f)
    pp.close()


def plot_repaired_neuron(neuron, cut_points, plot_out_path='.'):
    ''' Draw a neuron using plotly

    Repaired section are displayed with a different colors'''
    for mode in ['3d', 'xz']:
        name = os.path.join(plot_out_path, 'neuron_{}'.format(neuron.name)).replace(' ', '_')
        builder = PlotBuilder(neuron, mode, name, False)
        for section_id, offset in cut_points.items():
            section = neuron.sections[section_id]
            builder.color_section(section, 'green', recursive=True, index_offset=offset)
        builder.plot(show_link=False, auto_open=True)
