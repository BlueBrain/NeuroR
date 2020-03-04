'''Generate output plots'''
import logging
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt  # noqa, pylint: disable=ungrouped-imports,wrong-import-order,wrong-import-position
import numpy as np
from matplotlib.backends.backend_pdf import \
    PdfPages  # noqa, pylint: disable=ungrouped-imports,wrong-import-order,wrong-import-position
from neurom import geom, load_neuron
from neurom.view.view import plot_neuron

L = logging.getLogger('neuror')

try:
    from plotly_helper.neuron_viewer import NeuronBuilder
except ImportError:
    raise ImportError(
        'neuror[plotly] is not installed.'
        ' Please install it by doing: pip install neuror[plotly]')

matplotlib.use('Agg')


def get_common_bounding_box(neurons):
    '''Returns the bounding box that wraps all neurons'''
    common_bbox = geom.bounding_box(neurons[0])
    for neuron in neurons[1:]:
        bbox = geom.bounding_box(neuron)
        common_bbox[0] = np.min(np.vstack([common_bbox[0], bbox[0]]), axis=0)
        common_bbox[1] = np.max(np.vstack([common_bbox[1], bbox[1]]), axis=0)

    return common_bbox


def plot(neuron, bbox, subplot, title, **kwargs):
    '''2D neuron plot'''
    ax = plt.subplot(subplot, facecolor='w', aspect='equal')
    xlim = (bbox[0][0], bbox[1][0])
    ylim = (bbox[0][2], bbox[1][2])

    plot_neuron(ax, neuron, **kwargs)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _neuron_subplot(folders, f, pp, subplot, titles):
    kwargs = {'plane': 'xz'}
    fig = plt.figure()
    neurons = [load_neuron(os.path.join(folder, f)) for folder in folders]

    common_bbox = get_common_bounding_box(neurons)

    for i, (neuron, title) in enumerate(zip(neurons, titles)):
        plot(neuron, common_bbox, subplot + 1 + i, title, **kwargs)
    fig.suptitle(f)
    pp.savefig()


def view_all(folders, titles, output_pdf=None):
    '''Generate PDF report'''
    if not output_pdf:
        path = './plots'
        output_pdf = os.path.join(path, datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.pdf')
        if not os.path.exists(path):
            os.mkdir(path)

    pp = PdfPages(output_pdf)

    files = os.listdir(folders[0])

    subplot = 100 + 10 * len(folders)
    for f in files:
        L.info(f)

        try:
            _neuron_subplot(folders, f, pp, subplot, titles)
        except Exception as e:  # pylint: disable=broad-except
            L.info("e: %s", e)
            L.info('failu: %s', f)
    pp.close()
    L.info('Done writing %s', output_pdf)


def plot_repaired_neuron(neuron, cut_points, plot_file=None):
    ''' Draw a neuron using plotly

    Repaired section are displayed with a different colors'''

    for mode in ['3d', 'xz']:
        builder = NeuronBuilder(neuron, mode, neuron.name, False)
        for section, offset in cut_points.items():
            builder.color_section(section, 'green', recursive=True, start_point=offset)

        if plot_file is not None:
            root, ext = os.path.splitext(plot_file)
            plot_file = '{}_{}{}'.format(root, mode, ext)

        builder.plot(show_link=False, auto_open=False, filename=plot_file)
