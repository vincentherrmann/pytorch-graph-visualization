from graph_visualization.network_graph import *
from graph_visualization.scripts.create_networks import resnet18_1d_network
from graph_visualization.layout_calculation import LayoutCalculation
import imageio
import torch


if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'

#dev = 'cpu'
#net = vgg16_1d_network()
net = resnet18_1d_network()

# for i in range(4):
#     net = net.collapse_layers(factor=2, dimension=0)


writer = imageio.get_writer('resnet18_1d_layout_2.mp4', fps=30)
#writer = None

layout_calculation = LayoutCalculation(net=net, video_writer=writer, device=dev, size=(1200, 1200))
layout_calculation.range_gamma = 0.5
layout_calculation.additional_centering_per_level = 1.
layout_calculation.max_centering = 20.
layout_calculation.centering = 0.
layout_calculation.viz.scale_factor = 100.
layout_calculation.viz.focus = np.zeros(layout_calculation.viz.node_positions.shape[1]) > 0.

for i in range(8):
    net = net.collapse_layers(factor=2, dimension=0)
    net.to(dev)
for i in range(8):
    net = net.collapse_layers(factor=2, dimension=1)
    net.to(dev)


layout_calculation.net = net
layout_calculation.plot_connections = True

layout_positions = layout_calculation.start_simulation()
np.save('resnet18_1d_layout_positions_2', layout_positions)
writer.close()