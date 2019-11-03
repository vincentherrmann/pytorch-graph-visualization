from network_graph import *
from layout_calculation import LayoutCalculation
import imageio
import pickle

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


# for i in range(4):
#     net = net.collapse_layers(factor=2, dimension=0)


writer = imageio.get_writer('immersions_scalogram_resnet_house.mp4', fps=60)
# writer = None

layout_calculation = LayoutCalculation(net=net, video_writer=writer)

with open('/Users/vincentherrmann/Documents/Projekte/Immersions/immersions/immersions/misc/immersions_scalogram_resnet_house_data_statistics.p', 'rb') as handle:
    noise_statistics = pickle.load(handle)['element_std']

del noise_statistics['c_code']
del noise_statistics['z_code']
del noise_statistics['prediction']

weights = layout_calculation.interpolate_statistics(noise_statistics, position=0.5, window_size=None)

net.weights = (weights*0.) + 1.

for i in range(9):
    net = net.collapse_layers(factor=2, dimension=0)

for i in range(7):
    net = net.collapse_layers(factor=2, dimension=1)

layout_calculation.net = net

layout_calculation.start_simulation()