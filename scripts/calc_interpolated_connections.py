import datashader as ds
import torch
import pandas as pd
import numpy as np

from scripts.create_networks import scalogram_resnet_network_smaller

interpolated_positions = np.load('C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_house_smaller/immersions_scalogram_resnet_house_smaller_layout_240_positions.npy').astype(np.float32)
position_min = interpolated_positions.min()
position_max = interpolated_positions.max()
interpolated_positions = (interpolated_positions - position_min) / (position_max - position_min)
np.save('C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_house_smaller/immersions_scalogram_resnet_house_smaller_layout_240_positions.npy', interpolated_positions)

net = scalogram_resnet_network_smaller()

canvas = ds.Canvas(plot_width=800, plot_height=800,
                   x_range=(0,1), y_range=(0,1),
                   x_axis_type='linear', y_axis_type='linear')

connections = []

for time_frame in range(240):
    print("time frame", time_frame)
    positions = interpolated_positions[time_frame]
    #current_weights = weights[time_frame]
    edges = torch.FloatTensor(net.num_connections*3, 3)
    edges[0::3, :2] = torch.from_numpy(positions[net.connections[:, 0], :])
    edges[1::3, :2] = torch.from_numpy(positions[net.connections[:, 1], :])
    edges[2::3, :] = float('nan')
    edges[0::3, 2] = 1. #current_weights[net.connections[:, 0]]
    edges[1::3, 2] = 1. #current_weights[net.connections[:, 1]]
    edges = pd.DataFrame(data=edges.numpy())
    edges.columns = ['x', 'y', 'val']
    edges_lines = canvas.line(edges, 'x', 'y', agg=ds.sum('val'))
    connections.append(edges_lines)

connection_array = np.stack(connections, axis=0).astype(np.float32)
np.save('C:/Users/HEV7RNG/Documents/Immersions/models/immersions_scalogram_resnet_house_smaller/immersions_scalogram_resnet_house_smaller_connections_240.npy', connection_array)