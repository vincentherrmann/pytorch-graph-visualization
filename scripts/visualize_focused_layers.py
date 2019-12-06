import numpy as np
import qtpy
import time
import torch
import pandas as pd
import datashader as ds
from collections import OrderedDict

from scipy.interpolate import interp1d, griddata
from layout_calculation import hexes2colors
from openGLviz.net_visualizer import Visualizer
from threading import Thread
from vispy import app, gloo
import imageio
from immersions.input_optimization.activation_utilities import ModelActivations, activation_selection_dict
from scripts.create_networks import scalogram_resnet_network_smaller

name = 'immersions_scalogram_resnet_house_smaller'

activation_shapes_path = 'C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_house_smaller/immersions_scalogram_resnet_house_smaller_activation_shapes.p'
activations = ModelActivations(activation_shapes_path,
                               ignore_time_dimension=True,
                               remove_results=True)

global current_layer
current_layer = 'no_layer'

def write_img():
    global current_layer
    img = gloo.read_pixels(alpha=False)
    imageio.imwrite(name + '_' + current_layer + '.png', img)

viz = Visualizer(node_positions=np.random.rand(1, 0, 2).astype(np.float32),
                 animate=False,
                 edge_textures=np.zeros((1, 800, 800)).astype(np.float32),
                 size=(1200, 1200),
                 draw_callback=write_img)
viz.min_node_radius = 0.002
viz.node_radius_factor = 0.002
viz.animate = False
viz.node_alpha_factor = 2.
#viz.edges_colors = hexes2colors(['#000000', '#3f34a0', '#334f9a', '#337294', '#338e8c'])
viz.edges_colors = hexes2colors(['#000000', '#ffffff'])
viz.node_colors = hexes2colors(['#ffffff'])

window = qtpy.QtWidgets.QMainWindow()
window.setFixedSize(1500, 1500)
window.setCentralWidget(viz.native)

positions = np.load(
    'C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_house_smaller/immersions_scalogram_resnet_house_smaller_layout_positions.npy')
position_min = positions.min()
position_max = positions.max()
positions = (positions - position_min) / (position_max - position_min)

net = scalogram_resnet_network_smaller()

canvas = ds.Canvas(plot_width=800, plot_height=800,
                   x_range=(0,1), y_range=(0,1),
                   x_axis_type='linear', y_axis_type='linear')

layers = ['scalogram',
          'scalogram_block_0_main_conv_1',
          'scalogram_block_0_main_conv_2',
          'scalogram_block_1_main_conv_1',
          'scalogram_block_1_main_conv_2',
          'scalogram_block_2_main_conv_1',
          'scalogram_block_2_main_conv_2',
          'scalogram_block_3_main_conv_1',
          'scalogram_block_3_main_conv_2',
          'scalogram_block_4_main_conv_1',
          'scalogram_block_4_main_conv_2',
          'scalogram_block_5_main_conv_1',
          'scalogram_block_5_main_conv_2',
          'scalogram_block_6_main_conv_1',
          'scalogram_block_6_main_conv_2',
          'scalogram_block_7_main_conv_1',
          'scalogram_block_7_main_conv_2',
          'ar_block_0',
          'ar_block_1',
          'ar_block_2',
          'ar_block_3',
          'ar_block_4',
          'ar_block_5',
          'ar_block_6',
          'ar_block_7',
          'ar_block_8',
          'prediction']

def visualize_layers():
    global current_layer

    current_layer = 'connections'

    print("calc connections")
    edges = torch.FloatTensor(net.num_connections*3, 3)
    edges[0::3, :2] = torch.from_numpy(positions[net.connections[:, 0], :])
    edges[1::3, :2] = torch.from_numpy(positions[net.connections[:, 1], :])
    edges[2::3, :] = float('nan')
    edges[0::3, 2] = 1. #current_weights[net.connections[:, 0]]
    edges[1::3, 2] = 1. #current_weights[net.connections[:, 1]]
    edges = pd.DataFrame(data=edges.numpy())
    edges.columns = ['x', 'y', 'val']
    edges_lines = canvas.line(edges, 'x', 'y', agg=ds.sum('val')).values.astype(np.float32)
    edges_lines[edges_lines != edges_lines] = 0.
    edges_lines = pow(edges_lines / edges_lines.max(), 0.25)
    viz.edge_textures = edges_lines[np.newaxis, :, :]
    viz.update()

    viz.edge_textures = edges_lines[np.newaxis, :, :] * 0.

    for layer in layers:
        time.sleep(1)
        current_layer = layer
        activation_selection_dict = {
                    'layer': current_layer,
                    'channel': 0.,
                    'channel_region': 1.,
                    'pitch': 0.,
                    'pitch_region': 1.,
                    'time': 0.,
                    'time_region': 1.,
                    'keep_selection': 0.
                }
        activations.select_activations(activation_selection_dict)

        layer_positions = positions[activations.focus]
        viz.set_new_node_positions(layer_positions[None, :])
        viz.update()


viz_thread = Thread(target=visualize_layers, daemon=True)
viz_thread.start()


window.show()
app.run()