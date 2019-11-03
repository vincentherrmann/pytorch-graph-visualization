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

net.add_layer('scalogram', [2, 292])

net.add_layer('scalogram_block_0_main_conv_1', [32, 228])
net.add_layer('scalogram_block_0_main_conv_2', [32, 114])

net.add_layer('scalogram_block_1_main_conv_1', [32, 114])
net.add_layer('scalogram_block_1_main_conv_2', [32, 114])

net.add_layer('scalogram_block_2_main_conv_1', [64, 82])
net.add_layer('scalogram_block_2_main_conv_2', [64, 41])

net.add_layer('scalogram_block_3_main_conv_1', [64, 41])
net.add_layer('scalogram_block_3_main_conv_2', [64, 41])

net.add_layer('scalogram_block_4_main_conv_1', [128, 26])
net.add_layer('scalogram_block_4_main_conv_2', [128, 13])

net.add_layer('scalogram_block_5_main_conv_1', [128, 13])
net.add_layer('scalogram_block_5_main_conv_2', [128, 13])

net.add_layer('scalogram_block_6_main_conv_1', [256, 5])
net.add_layer('scalogram_block_6_main_conv_2', [256, 5])

net.add_layer('scalogram_block_7_main_conv_1', [512, 3])
net.add_layer('scalogram_block_7_main_conv_2', [512, 1])

net.add_layer('ar_block_0', [512, 1])
net.add_layer('ar_block_1', [512, 1])
net.add_layer('ar_block_2', [512, 1])
net.add_layer('ar_block_3', [512, 1])
net.add_layer('ar_block_4', [256, 1])
net.add_layer('ar_block_5', [256, 1])
net.add_layer('ar_block_6', [256, 1])
net.add_layer('ar_block_7', [256, 1])
net.add_layer('ar_block_8', [256, 1])

# Encoder
# BLOCK 0
net.add_conv1d_connections('scalogram', 'scalogram_block_0_main_conv_1',
                           kernel_size=65)
net.add_conv1d_connections('scalogram_block_0_main_conv_1', 'scalogram_block_0_main_conv_2',
                           kernel_size=3, stride=2, padding=(1, 1))
net.add_conv1d_connections('scalogram', 'scalogram_block_0_main_conv_2',
                           kernel_size=1, stride=2)

# BLOCK 1
net.add_conv1d_connections('scalogram_block_0_main_conv_2', 'scalogram_block_1_main_conv_1',
                           kernel_size=3, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_1_main_conv_1', 'scalogram_block_1_main_conv_2',
                           kernel_size=3, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_0_main_conv_2', 'scalogram_block_1_main_conv_2',
                           kernel_size=1)


# BLOCK 2
net.add_conv1d_connections('scalogram_block_1_main_conv_2', 'scalogram_block_2_main_conv_1',
                           kernel_size=33)
net.add_conv1d_connections('scalogram_block_2_main_conv_1', 'scalogram_block_2_main_conv_2',
                           kernel_size=3, stride=2, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_1_main_conv_2', 'scalogram_block_2_main_conv_2',
                           kernel_size=1, stride=2)

# BLOCK 3
net.add_conv1d_connections('scalogram_block_2_main_conv_2', 'scalogram_block_3_main_conv_1',
                           kernel_size=3, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_3_main_conv_1', 'scalogram_block_3_main_conv_2',
                           kernel_size=3, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_2_main_conv_2', 'scalogram_block_3_main_conv_2',
                           kernel_size=1)


# BLOCK 4
net.add_conv1d_connections('scalogram_block_3_main_conv_2', 'scalogram_block_4_main_conv_1',
                           kernel_size=16)
net.add_conv1d_connections('scalogram_block_4_main_conv_1', 'scalogram_block_4_main_conv_2',
                           kernel_size=3, stride=2, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_3_main_conv_2', 'scalogram_block_4_main_conv_2',
                           kernel_size=1, stride=2)

# BLOCK 5
net.add_conv1d_connections('scalogram_block_4_main_conv_2', 'scalogram_block_5_main_conv_1',
                           kernel_size=3, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_5_main_conv_1', 'scalogram_block_5_main_conv_2',
                           kernel_size=3, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_4_main_conv_2', 'scalogram_block_5_main_conv_2',
                           kernel_size=1)


# BLOCK 6
net.add_conv1d_connections('scalogram_block_5_main_conv_2', 'scalogram_block_6_main_conv_1',
                           kernel_size=9)
net.add_conv1d_connections('scalogram_block_6_main_conv_1', 'scalogram_block_6_main_conv_2',
                           kernel_size=3, stride=1, padding=(1, 1))
net.add_conv1d_connections('scalogram_block_5_main_conv_2', 'scalogram_block_6_main_conv_2',
                           kernel_size=1, stride=2)

# BLOCK 7
net.add_conv1d_connections('scalogram_block_6_main_conv_2', 'scalogram_block_7_main_conv_1',
                           kernel_size=3)
net.add_conv1d_connections('scalogram_block_7_main_conv_1', 'scalogram_block_7_main_conv_2',
                           kernel_size=3)
net.add_conv1d_connections('scalogram_block_6_main_conv_2', 'scalogram_block_7_main_conv_2',
                           kernel_size=1)

# Autoregressive model
# BLOCK 0
net.add_conv1d_connections('scalogram_block_7_main_conv_2', 'ar_block_0',
                           kernel_size=1)

# BLOCK 1
net.add_conv1d_connections('ar_block_0', 'ar_block_1',
                           kernel_size=1)

# BLOCK 2
net.add_conv1d_connections('ar_block_1', 'ar_block_2',
                           kernel_size=1)

# BLOCK 3
net.add_conv1d_connections('ar_block_2', 'ar_block_3',
                           kernel_size=1)

# BLOCK 4
net.add_conv1d_connections('ar_block_3', 'ar_block_4',
                           kernel_size=1)

# BLOCK 5
net.add_conv1d_connections('ar_block_4', 'ar_block_5',
                           kernel_size=1)

# BLOCK 3
net.add_conv1d_connections('ar_block_5', 'ar_block_6',
                           kernel_size=1)

# BLOCK 4
net.add_conv1d_connections('ar_block_6', 'ar_block_7',
                           kernel_size=1)

# BLOCK 5
net.add_conv1d_connections('ar_block_7', 'ar_block_8',
                           kernel_size=1)

# scoring
net.add_conv1d_connections('ar_block_8', 'scalogram_block_7_main_conv_2',
                           kernel_size=1)


for i in range(9):
    net = net.collapse_layers(factor=2, dimension=0)

for i in range(7):
    net = net.collapse_layers(factor=2, dimension=1)


layout = None

canvas = ds.Canvas(plot_width=800, plot_height=800,
                   x_range=(0,1), y_range=(0,1),
                   x_axis_type='linear', y_axis_type='linear')

texture = np.random.rand(1, 800, 800).astype(np.float32) * 0.


positions = np.random.rand(1, 0, 2).astype(np.float32)
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
last_positions = net.positions
current_net = net
pos_max = 1.
pos_min = -1.
range_gamma = 0.1

def update_plot():
    global current_net
    global global_step
    global level_step_counter
    global layout
    global last_positions
    global pos_max
    global pos_min
    while True:
        position_change = torch.mean(torch.norm(current_net.positions - last_positions, 2, dim=1))
        last_positions = current_net.positions.clone()
        if position_change < 0.001 and level_step_counter > 200:
            level_step_counter = 0
            if global_step > 0:
                current_net = current_net.give_positions_to_parent(perturbation=0.1)
                if current_net is None:
                    break
                last_positions = current_net.positions.clone()
            layout = NetworkForceLayout(current_net,
                                        spring_optimal_distance=1.,
                                        attraction_normalization=0.,
                                        repulsion=1.,
                                        step_size=0.05,
                                        step_discount_factor=0.95,
                                        centering=25.,
                                        drag=0.1,
                                        noise=0.1,
                                        mac=0.5,
                                        num_dim=2,
                                        force_limit=1.,
                                        distance_exponent=2.)

        layout.simulation_step()
        positions = layout.x.cpu().numpy()[np.newaxis, :].copy()
        n_pos_max = np.max(positions)
        n_pos_min = np.min(positions)
        pos_max = range_gamma * n_pos_max + (1 - range_gamma) * pos_max
        pos_min = range_gamma * n_pos_min + (1 - range_gamma) * pos_min
        positions -= pos_min
        positions /= (pos_max - pos_min)

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
        #edges_lines = gaussian_filter(edges_lines, sigma=0.8)

        viz.set_new_node_positions(positions, new_weights=current_net.weights[None, :].numpy())
        viz.edge_textures = edges_lines[np.newaxis, :, :]
        viz.update()
        img = gloo.read_pixels(alpha=False)

        level_step_counter += 1
        global_step += 1
        #time.sleep(0.1)


update_thread = Thread(target=update_plot, daemon=True)
update_thread.start()

window.show()
app.run()
