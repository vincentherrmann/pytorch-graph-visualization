import numpy as np
import qtpy
import time

from scipy.interpolate import interp1d, griddata
from layout_calculation import hexes2colors
from openGLviz.net_visualizer import Visualizer
from threading import Thread
from vispy import app

positions = np.load(
    '/Users/vincentherrmann/Documents/Projekte/Immersions/models/immersions_scalogram_resnet_maestro_smaller/immersions_scalogram_resnet_maestro_smaller_layout_timestep_positions.npy')
num_steps = positions.shape[0]

# random motion
random_size = 5


def lowpass(array, level=1):
    for i in range(level):
        array = (array + np.roll(array, 1, axis=0)) * 0.5
    return array


random_x = lowpass(np.random.randn(num_steps, random_size, random_size), level=4)
random_y = lowpass(np.random.randn(num_steps, random_size, random_size), level=4)

min_x = positions[:, :, 0].min() - 0.1
max_x = positions[:, :, 0].max() + 0.1
min_y = positions[:, :, 1].min() - 0.1
max_y = positions[:, :, 1].max() + 0.1

grid_coords = np.mgrid[0:random_size, 0:random_size].reshape(2, -1).transpose().astype(np.float)
grid_coords[:, 0] = grid_coords[:, 0] / (random_size - 1) * (max_x - min_x) + min_x
grid_coords[:, 1] = grid_coords[:, 1] / (random_size - 1) * (max_y - min_y) + min_y

perturbed_positions = positions.copy()
pertubation = 0.3
#pertubation = 0.

for step in range(num_steps):
    offsets_x = griddata(grid_coords, random_x[step].reshape(-1), positions[step], method='cubic')
    offsets_y = griddata(grid_coords, random_y[step].reshape(-1), positions[step], method='cubic')
    perturbed_positions[step, :, 0] += offsets_x * pertubation
    perturbed_positions[step, :, 1] += offsets_y * pertubation

interp_steps = 240
interp_linspace = np.linspace(0, 1, num=interp_steps, endpoint=False)

perturbed_positions = np.concatenate([perturbed_positions, perturbed_positions[:2]], axis=0)
orig_linspace = np.linspace(0, perturbed_positions.shape[0] / num_steps, perturbed_positions.shape[0], endpoint=False)

interp_p = interp1d(orig_linspace, perturbed_positions, kind='cubic', axis=0)
interpolated_positions = interp_p(interp_linspace)

np.save('/Users/vincentherrmann/Documents/Projekte/Immersions/models/immersions_scalogram_resnet_maestro_smaller/immersions_scalogram_resnet_maestro_smaller_layout_240_positions.npy', interpolated_positions)

# normalize
position_min = interpolated_positions.min()
position_max = interpolated_positions.max()
interpolated_positions = (interpolated_positions - position_min) / (position_max - position_min)
interpolated_positions = interpolated_positions.astype(np.float32)

viz = Visualizer(node_positions=np.random.rand(1, 0, 2).astype(np.float32),
                 animate=False,
                 edge_textures=np.zeros((1, 800, 800)).astype(np.float32),
                 size=(800, 800))
viz.min_node_radius = 0.002
viz.node_radius_factor = 0.001
viz.animate = False
viz.node_alpha_factor = 2.
viz.edges_colors = hexes2colors(['#000000', '#3f34a0', '#334f9a', '#337294', '#338e8c'])
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
        viz.update()
        current_frame = (current_frame + 1) % interpolated_positions.shape[0]
        time.sleep(1/60.)

window.show()
viz_thread = Thread(target=visualize, daemon=True)
viz_thread.start()
app.run()

