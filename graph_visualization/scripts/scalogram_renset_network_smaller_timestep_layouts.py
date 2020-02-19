from graph_visualization.network_graph import *
from graph_visualization.scripts.create_networks import scalogram_resnet_network_smaller
from graph_visualization.layout_calculation import LayoutCalculation
import imageio
import pickle
import torch

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'

net = scalogram_resnet_network_smaller()
net.to(dev)

time_positions = [0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

with open('C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_maestro_smaller/immersions_scalogram_resnet_maestro_smaller_data_statistics.p', 'rb') as handle:
    noise_statistics = pickle.load(handle)['element_std']

reference_positions = np.load('C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_maestro_smaller/immersions_scalogram_resnet_maestro_smaller_layout_positions.npy')

del noise_statistics['c_code']
del noise_statistics['z_code']
del noise_statistics['prediction']

writer = imageio.get_writer('immersions_scalogram_resnet_maestro_smaller_timesteps.mp4', fps=30)
# writer = None

layout_calculation = LayoutCalculation(net=net, video_writer=writer, device=dev, size=(1200, 1200))
layout_calculation.centering = 20.
layout_calculation.max_centering = 20.
layout_calculation.step_size = 0.1
layout_calculation.plot_connections = False
layout_calculation.range_gamma = 0.5
layout_calculation.viz.scale_factor = 100.
layout_calculation.viz.focus = np.zeros(layout_calculation.viz.node_positions.shape[1]) > 0.

time_positions_weights = []
for i, pos in enumerate(time_positions):
    time_positions_weights.append(layout_calculation.interpolate_statistics(noise_statistics, position=pos,
                                                        window_size=2 / len(time_positions)))
time_positions_weights = np.stack(time_positions_weights, axis=0)
time_positions_weights = torch.from_numpy(time_positions_weights)

new_positions = None

timestep_layouts = []

for i, pos in enumerate(time_positions):
    print("timestep", i)
    #weights = layout_calculation.interpolate_statistics(noise_statistics, position=pos, window_size=1/len(time_positions))
    weights = time_positions_weights[i]
    net.positions = torch.from_numpy(reference_positions).to(dev)
    net.weights = (weights.to(dev)*1.) + 0.

    layout_calculation.net = net
    layout_calculation.pos_min = torch.min(net.positions).item()
    layout_calculation.pos_max = torch.max(net.positions).item()
    layout_positions = layout_calculation.start_simulation()
    timestep_layouts.append(layout_positions)

np.save('immersions_scalogram_resnet_maestro_smaller_layout_timestep_positions', np.stack(timestep_layouts, axis=0))
writer.close()