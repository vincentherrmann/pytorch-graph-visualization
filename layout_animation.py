from network_graph import *

import numpy as np
import qtpy
from openGLviz.net_visualizer import Visualizer
from vispy import gloo, app
from threading import Thread
import time
import random
import datashader as ds
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import hex2color


def hexes2colors(h):
    colors = [list(hex2color(c)) for c in h]
    colors = np.float32(colors)
    colors = np.concatenate([colors, np.ones([len(h), 1], dtype=np.float32)], axis=1)
    return colors


net = Network()
net.add_layer('input_layer', [3, 9, 5])
net.add_layer('hidden_layer_1', [8, 7, 3])
net.add_layer('hidden_layer_2', [16, 5, 1])
net.add_layer('output_layer', [32, 3, 1])

net.add_conv2d_connections('input_layer', 'hidden_layer_1', kernel_size=(3, 3), padding=(0, 0, 0, 0))
net.add_conv2d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=(3, 3), padding=(0, 0, 0, 0))
net.add_conv2d_connections('hidden_layer_2', 'output_layer', kernel_size=(3, 1), padding=(0, 0, 0, 0))

for i in range(4):
    net = net.collapse_layers(factor=2, dimension=0)

for i in range(2):
    net = net.collapse_layers(factor=2, dimension=1)

for i in range(2):
    net = net.collapse_layers(factor=2, dimension=2)

layout = None

canvas = ds.Canvas(plot_width=800, plot_height=800,
                   x_range=(0,1), y_range=(0,1),
                   x_axis_type='linear', y_axis_type='linear')

texture = np.random.rand(1, 800, 800).astype(np.float32) * 0.


positions = np.random.rand(1, 100, 2).astype(np.float32)
viz = Visualizer(node_positions=positions, animate=False, edge_textures=texture)
viz.animate = False
viz.node_alpha_factor = 2.
viz.min_node_radius = 0.01
viz.edges_colors = hexes2colors(['#000000', '#3f34a0', '#334f9a', '#337294', '#338e8c'])
viz.node_colors = hexes2colors(['#005cff', '#a8ae3a',
                                '#005cff', '#a8ae3a',
                                '#005cff', '#a8ae3a',
                                '#005cff', '#a8ae3a'])

window = qtpy.QtWidgets.QMainWindow()
window.setCentralWidget(viz.native)

global_step = 0
level_step_counter = 10000
current_net = net

def update_plot():
    global current_net
    global global_step
    global level_step_counter
    global layout
    while True:
        if level_step_counter > 300:
            level_step_counter = 0
            if global_step > 0:
                current_net = current_net.give_positions_to_parent(perturbation=0.01)
                if current_net is None:
                    break
            layout = NetworkForceLayout(current_net,
                                        spring_optimal_distance=1.,
                                        attraction_normalization=0.,
                                        repulsion=1.,
                                        step_size=0.05,
                                        step_discount_factor=0.95,
                                        centering=0.,
                                        drag=0.5,
                                        noise=0.,
                                        mac=0.5,
                                        num_dim=2,
                                        force_limit=1.,
                                        distance_exponent=2.)

        layout.simulation_step()
        positions = layout.x.cpu().numpy()[np.newaxis, :].copy()
        positions -= positions.min()
        positions /= positions.max()

        edges = np.zeros((current_net.num_connections * 3, 3), dtype=np.float32)
        edges[0::3, :2] = positions[0, current_net.connections[:, 0], :]
        edges[1::3, :2] = positions[0, current_net.connections[:, 1], :]
        edges[2::3, :] = float('nan')
        edges[0::3, 2] = 1.
        edges[1::3, 2] = 1.
        edges = pd.DataFrame(data=edges)
        edges.columns = ['x', 'y', 'val']
        edges_lines = canvas.line(edges, 'x', 'y', agg=ds.sum('val')).values.astype(np.float32)
        edges_lines[edges_lines != edges_lines] = 0.
        edges_lines = pow(edges_lines / edges_lines.max(), 0.25)
        edges_lines = gaussian_filter(edges_lines, sigma=0.8)

        viz.set_new_node_positions(positions, new_weights=current_net.weights[None, :].numpy())
        viz.edge_textures = edges_lines[np.newaxis, :, :]
        viz.update()

        level_step_counter += 1
        global_step += 1
        #time.sleep(0.1)


update_thread = Thread(target=update_plot, daemon=True)
update_thread.start()

window.show()
app.run()
