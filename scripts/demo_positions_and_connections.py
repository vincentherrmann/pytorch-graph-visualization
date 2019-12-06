import numpy as np
import qtpy
import time

from scipy.interpolate import interp1d, griddata
from layout_calculation import hexes2colors
from openGLviz.net_visualizer import Visualizer
from threading import Thread
from vispy import app

interpolated_positions = np.load('C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_house_smaller/immersions_scalogram_resnet_house_smaller_layout_240_positions.npy').astype(np.float32)
connections = np.load('C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_house_smaller/immersions_scalogram_resnet_house_smaller_connections_240.npy')
connections[connections != connections] = 0.
connections /= connections.max()
connections = np.power(connections, 0.25)

viz = Visualizer(node_positions=np.random.rand(1, 0, 2).astype(np.float32),
                 animate=False,
                 edge_textures=np.zeros((1, 800, 800)).astype(np.float32),
                 size=(800, 800))
viz.min_node_radius = 0.002
viz.node_radius_factor = 0.001
viz.animate = False
viz.node_alpha_factor = 2.
#viz.edges_colors = hexes2colors(['#000000', '#3f34a0', '#334f9a', '#337294', '#338e8c'])
viz.edges_colors = hexes2colors(['#000000', '#24248f', '#24598f', '#248f8f', '#248f59'])
viz.node_colors = hexes2colors(['#005cff', '#a8ae3a',
                                '#005cff', '#a8ae3a',
                                '#005cff', '#a8ae3a',
                                '#005cff', '#a8ae3a'])

window = qtpy.QtWidgets.QMainWindow()
window.setCentralWidget(viz.native)


def visualize():
    current_frame = 0
    while True:
        pos = interpolated_positions[current_frame]
        viz.set_new_node_positions(pos[None, :])
        viz.edge_textures = connections[current_frame, :, :][None, :]
        viz.update()
        current_frame = (current_frame + 1) % interpolated_positions.shape[0]
        time.sleep(1/60.)

window.show()
viz_thread = Thread(target=visualize, daemon=True)
viz_thread.start()
app.run()