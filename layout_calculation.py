from network_graph import *

import numpy as np
import qtpy
from openGLviz.net_visualizer import Visualizer
from vispy import gloo, app
from threading import Thread
import time
import random
import imageio
import datashader as ds
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import hex2color


def hexes2colors(h):
    colors = [list(hex2color(c)) for c in h]
    colors = np.float32(colors)
    colors = np.concatenate([colors, np.ones([len(h), 1], dtype=np.float32)], axis=1)
    return colors


class LayoutCalculation:
    def __init__(self, net, video_writer=None, size=(800, 600), steps_per_frame=5, device='cpu'):
        self.variance_offset = 0.1
        self.steps_per_frame = steps_per_frame
        self.device = device
        self.net = net
        self.video_writer = video_writer
        self.viz = Visualizer(node_positions=np.random.rand(1, 0, 2).astype(np.float32),
                              animate=False,
                              edge_textures=np.zeros((1, 800, 800)).astype(np.float32),
                              draw_callback=self.write_frame,
                              size=size)
        self.viz.min_node_radius = 0.0005
        self.viz.node_radius_factor = 0.001
        self.viz.animate = False
        self.viz.node_alpha_factor = 2.
        self.viz.edges_colors = hexes2colors(['#000000', '#3f34a0', '#334f9a', '#337294', '#338e8c'])
        self.viz.node_colors = hexes2colors(['#005cff', '#a8ae3a',
                                        '#005cff', '#a8ae3a',
                                        '#005cff', '#a8ae3a',
                                        '#005cff', '#a8ae3a'])

        self.ds_canvas = ds.Canvas(plot_width=800, plot_height=800,
                                   x_range=(0, 1), y_range=(0, 1),
                                   x_axis_type='linear', y_axis_type='linear')

        self.window = qtpy.QtWidgets.QMainWindow()
        self.window.setCentralWidget(self.viz.native)

        self.simulation_thread = Thread(target=self.run_simulation, daemon=True)

        self.range_gamma = 0.1

    def run_simulation(self):
        global_step = 0
        level_step_counter = 1e10
        last_positions = self.net.positions
        current_net = self.net
        pos_max = 2.
        pos_min = -2.
        max_centering = 25.
        additional_centering_per_level = 2.
        centering = 2.

        while True:
            position_change = torch.mean(torch.norm(current_net.positions - last_positions, 2, dim=1))
            last_positions = current_net.positions.clone()
            if position_change < 0.001 and level_step_counter > 200:
                print("move to new level")
                level_step_counter = 0
                if global_step > 0:
                    current_net = current_net.give_positions_to_parent(perturbation=0.1)
                    if current_net is None:
                        break
                    last_positions = current_net.positions.clone()
                centering = min(max_centering, centering+additional_centering_per_level)
                print("positions:", last_positions.shape, "centering:", centering)
                layout = NetworkForceLayout(current_net,
                                            spring_optimal_distance=1.,
                                            attraction_normalization=0.,
                                            repulsion=1.,
                                            step_size=0.5,
                                            step_discount_factor=0.98,
                                            centering=centering,
                                            drag=0.2,
                                            noise=0.,
                                            mac=0.5,
                                            num_dim=2,
                                            force_limit=1.,
                                            distance_exponent=2.5,
                                            device=self.device)
            for i in range(self.steps_per_frame):
                layout.simulation_step()
                level_step_counter += 1
                global_step += 1
            positions = layout.x.cpu().numpy()[np.newaxis, :].copy()
            n_pos_max = np.max(positions)
            n_pos_min = np.min(positions)
            pos_max = self.range_gamma * n_pos_max + (1 - self.range_gamma) * pos_max
            pos_min = self.range_gamma * n_pos_min + (1 - self.range_gamma) * pos_min
            positions -= pos_min
            positions /= (pos_max - pos_min)

            edges = np.zeros((current_net.num_connections * 3, 3), dtype=np.float32)
            edges[0::3, :2] = positions[0, current_net.connections[:, 0].cpu(), :]
            edges[1::3, :2] = positions[0, current_net.connections[:, 1].cpu(), :]
            edges[2::3, :] = float('nan')
            edges[0::3, 2] = 1.
            edges[1::3, 2] = 1.
            edges = pd.DataFrame(data=edges)
            edges.columns = ['x', 'y', 'val']
            edges_lines = self.ds_canvas.line(edges, 'x', 'y', agg=ds.sum('val')).values.astype(np.float32)
            edges_lines[edges_lines != edges_lines] = 0.
            edges_lines = pow(edges_lines / edges_lines.max(), 0.25)
            # edges_lines = gaussian_filter(edges_lines, sigma=0.8)

            self.viz.set_new_node_positions(positions, new_weights=current_net.weights[None, :].cpu().numpy())
            self.viz.edge_textures = edges_lines[np.newaxis, :, :]
            self.viz.update()


            # time.sleep(0.1)

        if self.video_writer is not None:
            print("close video writer")
            self.video_writer.close()

        np.save('layout_positions', layout.x.cpu().numpy())

    def write_frame(self):
        if self.video_writer is not None:
            img = gloo.read_pixels(alpha=False)
            self.video_writer.append_data(img)

    def start_simulation(self):
        self.window.show()
        self.simulation_thread.start()

        app.run()

    @staticmethod
    def window_func(x):
        w = 0.5 * (1 - torch.cos(2 * x * np.pi))
        w[x < 0] = 0
        w[x > 1] = 0
        return w

    def interpolate_position(self, x, pos, dim=None, window_size=0.1):
        if dim is None:
            dim = len(x.shape) - 1

        length = x.shape[dim]
        if length == 1:
            return x.squeeze(dim)

        if window_size is None:
            return torch.mean(x, dim=dim)

        if length < 1 / window_size:
            pos = min(max(0., pos), 1.)
            if pos == 1:
                pos = 0.
            idx = int(pos * length)
            idx2 = idx + 1 if idx < length - 1 else 0
            interp = pos * length - idx
            print(interp)
            a = torch.index_select(x, dim=dim, index=torch.LongTensor([idx])).squeeze(dim)
            b = torch.index_select(x, dim=dim, index=torch.LongTensor([idx2])).squeeze(dim)
            return (1 - interp) * a + interp * b
        else:
            support = (torch.linspace(0.5, 1 / window_size + 0.5, length) - pos / window_size) % (1 / window_size)
            w = self.window_func(support)
            w /= w.sum()
            shape = list(x.shape)
            for i in range(len(shape)):
                if i != dim:
                    shape[i] = 1
            x = x * w.view(shape) / torch.sum(w)
            x = torch.sum(x, dim=dim)
            return x

    def interpolate_statistics(self, stats, position, window_size=0.125):
        weights = []
        for key in stats.keys():
            variances = stats[key]
            var_max = torch.max(variances)
            print(key)
            print("var max:", var_max)
            variances /= var_max
            variances += self.variance_offset
            variances /= variances.mean()
            print("max var:", variances.max())
            print("min var:", variances.min())
            print("mean:", variances.mean())
            print("shape:", variances.shape)
            print()

            weights.append(self.interpolate_position(variances, pos=position, window_size=window_size).view(-1))
        weights = torch.cat(weights)
        return weights

if __name__ == '__main__':
    net = Network()
    net.add_layer('input_layer', [3, 9, 5])
    net.add_layer('hidden_layer_1', [32, 7, 3])
    net.add_layer('hidden_layer_2', [64, 5, 1])
    net.add_layer('output_layer', [128, 3, 1])

    net.add_conv2d_connections('input_layer', 'hidden_layer_1', kernel_size=(3, 3), padding=(0, 0, 0, 0))
    net.add_conv2d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=(3, 3), padding=(0, 0, 0, 0))
    net.add_conv2d_connections('hidden_layer_2', 'output_layer', kernel_size=(3, 1), padding=(0, 0, 0, 0))

    for i in range(6):
        net = net.collapse_layers(factor=2, dimension=0)

    for i in range(2):
        net = net.collapse_layers(factor=2, dimension=1)

    for i in range(2):
        net = net.collapse_layers(factor=2, dimension=2)

    writer = imageio.get_writer('layout_2d_conv.mp4', fps=60)
    #writer = None

    layout_calculation = LayoutCalculation(net=net, video_writer=writer, size=(600, 600))
    layout_calculation.start_simulation()
