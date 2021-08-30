'''App to find cut planes with arbitrary cut planes orientations
with the help of a manual hint

Related to https://bbpteam.epfl.ch/project/issues/browse/NGV-85
'''
import json

import neurom as nm
import numpy as np
from neurom import load_morphology
from neurom.geom import bounding_box
from pyquaternion import Quaternion

from neuror.cut_plane.detection import CutPlane, _minimize
from neuror.cut_plane.planes import _get_displaced_pos

try:
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output, State
    from plotly_helper.neuron_viewer import NeuronBuilder
except ImportError as e:
    raise ImportError(
        'neuror[plotly] is not installed.'
        ' Please install it by doing: pip install neuror[plotly]') from e


# Copy of: https://codepen.io/chriddyp/pen/bWLwgP.css
external_stylesheets = ['dash.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

BIN_WIDTH = 10


class NumpyEncoder(json.JSONEncoder):
    '''JSON encoder that handles numpy types

    In python3, numpy.dtypes don't serialize to correctly, so a custom converter
    is needed.
    '''

    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def create_plane(pos, quat):
    """ Create a 3d plane using a center position and a quaternion for orientation

    Args :
        pos: x,y,z position of the plane's center (array([x,y,z]))
        quat: quaternion representing the orientations (Quaternion)
        size_multiplier: plane size in space coordinates (float)
        opacity: set the opacity value (float)

    Returns :
        A square surface to the plotly format
    """

    length = np.linalg.norm(BBOX[1] - BBOX[0]) / 2.
    positif_x = _get_displaced_pos(pos, quat, length, (1, 0, 0))
    positif_y = _get_displaced_pos(pos, quat, length, (0, 1, 0))
    negatif_x = _get_displaced_pos(pos, quat, length, (-1, 0, 0))
    negatif_y = _get_displaced_pos(pos, quat, length, (0, -1, 0))

    x = [[positif_x[0], positif_y[0]], [negatif_y[0], negatif_x[0]]]
    y = [[positif_x[1], positif_y[1]], [negatif_y[1], negatif_x[1]]]
    z = [[positif_x[2], positif_y[2]], [negatif_y[2], negatif_x[2]]]

    return dict(
        z=z,
        x=x,
        y=y,
        showscale=False,
        type='surface',
        surfacecolor=['green', 'green'], opacity=1
    )


ROT_X, ROT_Y, ROT_Z = 4, 45, -21
TRANSL_X, TRANSL_Y, TRANSL_Z = 0, 0, 61

app.layout = html.Div(children=[
    html.Pre(id='click-data'),
    html.Pre(id='neuron'),

    html.Pre(id='optimized'),

    html.Div(
        [
            "Rotate the plane until it is aligned with the cut plane, then click optimize, "
            "then click export. "
            'On the histogram, all points should fall on only one side of the red line',
            html.Div(
                [
                    dcc.Graph(id='graph'),
                ],
                style={'width': '70%', 'display': 'inline-block'},
            ),

            html.Div(
                [
                    dcc.Graph(id='bar'),
                    html.Div([
                        html.Button('Optimize', id='button'),
                        dcc.Checklist(id='hide-plane',
                                      options=[
                                          {'label': 'Hide plane', 'value': 'hidden'}
                                      ],
                                      value=[]),

                        "Rotations are with respect to the frame attached to the plane.",
                        html.Div(
                            [html.Div(['Rotate X:'], id='output-x-rotate'),
                             dcc.Input(id='rotate-x-slider', type='number', value=ROT_X), ]
                        ),

                        html.Div(
                            [html.Div(['Rotate Y:'], id='output-y-rotate'),
                             dcc.Input(id='rotate-y-slider', type='number', value=ROT_Y), ],
                        ),

                        html.Div(
                            [html.Div(['Rotate Z:'], id='output-z-rotate'),
                             dcc.Input(id='rotate-z-slider', type='number', value=ROT_Z), ]
                        ),

                        html.Div(
                            [html.Div(['Translate X:'], id='output-x-translate'),
                             dcc.Input(id='translate-x-slider', type='number', value=TRANSL_X)]
                        ),

                        html.Div(
                            [html.Div(['Translate Y:'], id='output-y-translate'),
                             dcc.Input(id='translate-y-slider', type='number', value=TRANSL_Y), ],
                        ),

                        html.Div(
                            [html.Div(['Translate Z:'], id='output-z-translate'),
                             dcc.Input(id='translate-z-slider',
                                       type='number', value=TRANSL_Z, min=-10000), ]
                        ),

                        dcc.Input(id='export-path-input', type='text', value='/tmp/cut-plane.json'),
                        html.Button('export', id='export')], style={'margin-left': '100px'})
                ],
                style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}
            ),
        ],
        style={'width': '100%', 'display': 'inline'})
]
)

NEURON = FIGURE = BBOX = None


def set_neuron(filename):
    '''Globally loads the neuron'''
    global NEURON, FIGURE, BBOX  # pylint: disable=global-statement
    NEURON = load_morphology(filename)
    FIGURE = NeuronBuilder(NEURON, '3d').get_figure()
    BBOX = bounding_box(NEURON)


@app.callback(
    Output('graph', 'figure'),
    [
        Input('rotate-x-slider', 'value'),
        Input('rotate-y-slider', 'value'),
        Input('rotate-z-slider', 'value'),
        Input('translate-x-slider', 'value'),
        Input('translate-y-slider', 'value'),
        Input('translate-z-slider', 'value'),
        Input('hide-plane', 'value'),
    ],
    [State('graph', 'relayoutData')]

)
def display_click_data(rot_x, rot_y, rot_z, transl_x, transl_y, transl_z, hide, layout):
    '''callback that redraw everything when sliders are changed'''
    qx = Quaternion(axis=[1, 0, 0], angle=rot_x / 180. * np.pi)
    qy = Quaternion(axis=[0, 1, 0], angle=rot_y / 180. * np.pi)
    qz = Quaternion(axis=[0, 0, 1], angle=rot_z / 180. * np.pi)
    plane = create_plane([transl_x, transl_y, transl_z], qx * qy * qz)

    FIGURE['data'] = [x for x in FIGURE['data'] if not isinstance(x, dict)]
    if not hide:
        FIGURE['data'].append(plane)
        FIGURE['layout']['autosize'] = False
        FIGURE['layout']['height'] = 1500

    if layout and layout.get('scene.camera'):
        FIGURE['layout']['scene']['camera'] = layout['scene.camera']

    return FIGURE


@app.callback(
    dash.dependencies.Output('bar', 'figure'),
    [
        Input('rotate-x-slider', 'value'),
        Input('rotate-y-slider', 'value'),
        Input('rotate-z-slider', 'value'),
        Input('translate-x-slider', 'value'),
        Input('translate-y-slider', 'value'),
        Input('translate-z-slider', 'value'),
    ])
def update_output(rot_x, rot_y, rot_z, transl_x, transl_y, transl_z):
    '''Update histo when sliders are changed'''
    transformations = [rot_x, rot_y, rot_z, transl_x, transl_y, transl_z]
    bin_width = 10
    cut_plane = CutPlane.from_rotations_translations(transformations, NEURON, bin_width)
    hist, binning = cut_plane.histogram()

    binning += bin_width / 2.

    return {
        'data': [
            {'x': binning, 'y': hist, 'type': 'bar', 'name': 'SF'},
        ],
        'layout': {
            'title': 'Distance to plane distribution',
            'xaxis': {
                'title': 'Distance to plane (um)'
            },
            'yaxis': {
                'title': 'Counts'
            },
            'shapes': [
                # Line Vertical
                {
                    'type': 'line',
                    'x0': 0,
                    'y0': 0,
                    'x1': 0,
                    'y1': np.max(hist),
                    'line': {
                        'color': 'red',
                        'width': 3,
                    },
                }
            ]
        }
    }


@app.callback(
    dash.dependencies.Output('optimized', 'data-*'),
    [Input('button', 'n_clicks')],
    [
        State('rotate-x-slider', 'value'),
        State('rotate-y-slider', 'value'),
        State('rotate-z-slider', 'value'),
        State('translate-x-slider', 'value'),
        State('translate-y-slider', 'value'),
        State('translate-z-slider', 'value'),
    ]
)
def optimize(n_clicks, rot_x, rot_y, rot_z, transl_x, transl_y, transl_z):
    '''Optimize cut plane parameters'''
    if not n_clicks:
        return rot_x, rot_y, rot_z, transl_x, transl_y, transl_z
    points = np.array([point
                       for neurite in (NEURON.neurites or [])
                       for section in nm.iter_sections(neurite)
                       for point in section.points])
    params = rot_x, rot_y, rot_z, transl_x, transl_y, transl_z
    result = _minimize(params, points, bin_width=BIN_WIDTH)
    return result


@app.callback(
    Output('rotate-x-slider', 'value'),
    [Input('optimized', 'data-*')])
def update_post_optim_x_rotate(params):
    '''callback'''
    return params[0]


@app.callback(
    Output('rotate-y-slider', 'value'),
    [Input('optimized', 'data-*')])
def update_post_optim_y_rotate(params):
    '''callback'''
    return params[1]


@app.callback(
    Output('rotate-z-slider', 'value'),
    [Input('optimized', 'data-*')])
def update_post_optim_z_rotate(params):
    '''callback'''
    return params[2]


@app.callback(
    Output('translate-x-slider', 'value'),
    [Input('optimized', 'data-*')])
def update_post_optim_x_translate(params):
    '''callback'''
    return params[3]


@app.callback(
    Output('translate-y-slider', 'value'),
    [Input('optimized', 'data-*')])
def update_post_optim_y_translate(params):
    '''callback'''
    return params[4]


@app.callback(
    Output('translate-z-slider', 'value'),
    [Input('optimized', 'data-*')])
def update_post_optim_z_translate(params):
    '''callback'''
    return params[5]


@app.callback(
    dash.dependencies.Output('click-data', 'children'),
    [Input('export', 'n_clicks')],
    [State('rotate-x-slider', 'value'),
     State('rotate-y-slider', 'value'),
     State('rotate-z-slider', 'value'),
     State('translate-x-slider', 'value'),
     State('translate-y-slider', 'value'),
     State('translate-z-slider', 'value'),
     State('export-path-input', 'value')])
def export(n_clicks, rot_x, rot_y, rot_z, transl_x, transl_y, transl_z, output_path):
    '''Write the final file cut-plane.json to disk'''
    if not n_clicks:
        return
    plane = CutPlane.from_rotations_translations(
        [rot_x, rot_y, rot_z, transl_x, transl_y, transl_z],
        NEURON, BIN_WIDTH)
    payload = [plane.to_json()]
    with open(output_path, 'w') as f:
        json.dump(payload, f, cls=NumpyEncoder)
