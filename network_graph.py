import numpy as np
import torch
import matplotlib
import threading
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import matplotlib.animation
import time


class Network:
    def __init__(self):
        self.layers = {}
        self.connections = []
        self._num_units = 0

    @property
    def num_units(self):
        return self._num_units

    def add_layer(self, name, shape):
        try:
            len(shape)
        except:
            shape = [shape]
        layer_units = np.prod(shape)
        indices = torch.arange(self.num_units, self.num_units + layer_units, dtype=torch.long)
        indices = indices.view(shape)
        self._num_units += layer_units
        self.layers[name] = indices

    def add_full_connections(self, input_layer, output_layer):
        in_indices = self.layers[input_layer].flatten()
        out_indices = self.layers[output_layer].flatten()
        sources = in_indices.repeat([len(out_indices), 1])
        self.connections.append((out_indices, sources))

    def add_conv1d_connections(self, input_layer, output_layer, kernel_size, stride=1):
        in_indices = self.layers[input_layer]
        out_indices = self.layers[output_layer]
        in_channels = in_indices.shape[0]

        connection_indices = torch.zeros([out_indices.shape[0],  # channels
                                          out_indices.shape[1],  # position
                                          in_channels,
                                          kernel_size,
                                          2], dtype=torch.long)
        for c in range(in_channels):
            connection_indices[:, :, c, :, 0] = c
        for o in range(out_indices.shape[1]):
            for k in range(kernel_size):
                connection_indices[:, o, :, k, 1] = k + stride * o

        connection_indices = connection_indices.view(-1, 2)
        connections = torch.zeros(connection_indices.shape[0], dtype=torch.long)
        for i in range(connections.shape[0]):
            connections[i] = in_indices[connection_indices[i, 0], connection_indices[i, 1]]

        self.connections.append((out_indices.flatten(), connections.view(-1, in_channels*kernel_size)))

    def connection_count_per_unit(self):
        connection_counts = torch.zeros(self.num_units, dtype=torch.long)
        for origins, targets in self.connections:
            connection_counts[origins] += targets.shape[1]
            for i in range(targets.shape[0]):
                connection_counts[targets[i, :]] += 1
        return connection_counts

    def to(self, device):
        for i, (origins, targets) in enumerate(self.connections):
            self.connections[i] = (origins.to(device), targets.to(device))
        for key, value in self.layers.items():
            self.layers[key] = value.to(device)

class NetworkForceLayout:
    def __init__(self, network, gravity=-0.005, attraction=0.01, centering=0.1, friction=1., normalize_attraction=False, step_size=0.1, device='cpu'):
        self.network = network
        self.device = device
        self.network.to(device)
        self.x = torch.randn([self.network.num_units, 2], device=self.device)
        self.v = torch.zeros_like(self.x)
        self.a = torch.zeros_like(self.x)
        self.movable = torch.ones(self.network.num_units, device=self.device)
        self.colors = torch.ones([self.network.num_units, 4], device=self.device)
        self.set_default_colors()
        self.connection_counts = self.network.connection_count_per_unit().float().to(self.device)

        self.gravity = gravity
        self.attraction = attraction
        self.centering = centering
        self.friction = friction
        self.normalize_attraction = normalize_attraction
        self.step_size = step_size

        #plt.ion()
        self.fig, self.ax = plt.subplots(1, 1)
        self.scatter = None
        self.lines = None

        self.plotting_thread = None
        #plt.show()

    def set_position(self, indices, pos, fix=False):
        i = indices.flatten()
        self.x[i, :] = pos
        if fix:
            self.movable[i] = 0.

    def simulation_step(self):
        diff = self.x.unsqueeze(1) - self.x.unsqueeze(0)

        # gravity
        f = self.gravity * torch.sum(diff / ((torch.norm(diff, 2, dim=2, keepdim=True)**3) + 1e-5), dim=0)

        # attraction
        a_f = torch.zeros_like(f)
        for origins, targets in self.network.connections:
            attraction_force = self.attraction * (self.x[targets, :] - self.x[origins, :].unsqueeze(1))
            a_f[origins, :] += torch.sum(attraction_force, dim=1)
            a_f[targets, :] -= attraction_force
        if self.normalize_attraction:
            a_f *= 1 / self.connection_counts.unsqueeze(1)
        f += a_f

        # centering
        f -= self.centering * self.x

        # friction
        f -= self.friction * self.v
        f = torch.clamp(f, -0.1, 0.1)

        a = f  # since we use mass = 1 for all nodes

        # velocity verlet integration
        self.x += self.movable.unsqueeze(1) * (self.v * self.step_size + 0.5 * self.a * self.step_size**2)
        self.v += 0.5 * (self.a + a) * self.step_size
        self.a = a

    def simulate(self, num_steps=100, plot_interval=None):
        tic = time.time()
        for step in range(num_steps):
            self.simulation_step()
            if plot_interval is not None:
                if step % plot_interval == 0:
                    print("step", step, 'time per step:', (time.time() - tic) / plot_interval)
                    tic = time.time()
                    #self.plotting_thread = threading.Thread(target=self.plot)
                    #self.plotting_thread.daemon = True
                    #self.plotting_thread.start()
                    #self.plot()

    # PLOTTING #########
    def line_data(self):
        lines = []
        for origins, targets in self.network.connections:
            origin_positions = self.x[origins, :].unsqueeze(1).repeat([1, targets.shape[1], 1])
            target_positions = self.x[targets, :]
            lines.append(torch.stack([origin_positions.view(-1, 2), target_positions.view(-1, 2)], dim=1))
        lines = torch.cat(lines, dim=0)
        return lines

    def plot(self, plot_connections=True):
        #fig, ax = plt.subplots(1, 1)
        try:
            self.lines.remove()
            self.scatter.remove()
        except:
            pass
        if plot_connections:
            self.lines = mc.LineCollection(self.line_data(), lw=0.5, alpha=0.2)
            self.ax.add_collection(self.lines)
        self.scatter = self.ax.scatter(self.x[:, 0], self.x[:, 1], linewidths=1, c=self.colors)
        self.ax.autoscale()
        self.fig.canvas.draw()
        plt.show()

    def set_default_colors(self, colormap='Set1'):
        cmap = matplotlib.cm.get_cmap(colormap)
        for l, (name, indices) in enumerate(self.network.layers.items()):
            color = cmap(l)
            i = indices.flatten()
            self.colors[i, 0] = color[0]
            self.colors[i, 1] = color[1]
            self.colors[i, 2] = color[2]
            self.colors[i, 3] = color[3]


def animation_step(i, simulation, plot_connections=True):
    for _ in range(200):
        simulation.simulation_step()
    try:
        simulation.lines.remove()
        simulation.scatter.remove()
    except:
        pass
    if plot_connections:
        simulation.lines = mc.LineCollection(simulation.line_data(), lw=0.5, alpha=0.2)
        simulation.ax.add_collection(simulation.lines)
    simulation.scatter = simulation.ax.scatter(simulation.x[:, 0], simulation.x[:, 1], linewidths=1, c=simulation.colors)
    simulation.ax.autoscale()
    simulation.fig.canvas.draw()
    # plt.draw()
    #plt.show()


def animate_simulation(simulation):
    ani = matplotlib.animation.FuncAnimation(simulation.fig, animation_step, frames=50, fargs=(simulation, True))
    return ani





