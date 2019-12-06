from network_graph import *
from scripts.create_networks import scalogram_resnet_network
from layout_calculation import LayoutCalculation
import imageio
import pickle
import torch


if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'

net = scalogram_resnet_network()

# for i in range(4):
#     net = net.collapse_layers(factor=2, dimension=0)


writer = imageio.get_writer('immersions_scalogram_resnet_maestro.mp4', fps=30)
# writer = None

layout_calculation = LayoutCalculation(net=net, video_writer=writer, device=dev, size=(1000, 1000))

with open('C:/Users/HEV7RNG/Downloads/immersions_scalogram_resnet_maestro_data_statistics.p', 'rb') as handle:
    noise_statistics = pickle.load(handle)['element_std']

del noise_statistics['c_code']
del noise_statistics['z_code']
del noise_statistics['prediction']

weights = layout_calculation.interpolate_statistics(noise_statistics, position=0.5, window_size=None)

net.weights = (weights*1.) + 0.
net.to(dev)

for i in range(8):
    net = net.collapse_layers(factor=2, dimension=0)
    net.to(dev)
for i in range(8):
    net = net.collapse_layers(factor=2, dimension=1)
    net.to(dev)


layout_calculation.net = net

layout_positions = layout_calculation.start_simulation()
np.save('layout_positions', layout_positions)
writer.close()