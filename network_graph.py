import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import threading
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import time
import copy
import math
from BarnesHutTree import *


class Network(object):
    def __init__(self, num_dim=2):
        super().__init__()
        self.num_dim = num_dim
        self.layers = {}
        self.connections = []
        self.layer_connections = {}
        self._num_units = 0
        self.parent_graph = None
        self.expand_lookup = None
        self.positions = None
        self.weights = None
        self.colors = None

        self.fig, self.ax = plt.subplots(1, 1)
        self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax = self.fig.add_subplot(111)
        self.scatter = None
        self.lines = None

    @property
    def num_units(self):
        return self._num_units

    @property
    def num_connections(self):
        n = 0
        for origins, targets, weights in self.connections:
            n += np.prod(targets.shape)
        return n

    def add_layer(self, name, shape, positions=None, weights=None, colors=None):
        try:
            len(shape)
        except:
            shape = [shape]
        layer_units = np.prod(shape)
        indices = torch.arange(self.num_units, self.num_units + layer_units, dtype=torch.long)
        indices = indices.view(shape)
        self._num_units += layer_units
        self.layers[name] = indices
        self.layer_connections[name] = []

        if positions is None:
            positions = torch.randn([layer_units, self.num_dim])
        if weights is None:
            weights = torch.ones(layer_units)
        if colors is None:
            colors = torch.ones([layer_units, 4])

        if self.positions is None:
            self.positions = positions
        else:
            self.positions = torch.cat([self.positions, positions], dim=0)
        if self.weights is None:
            self.weights = weights
        else:
            self.weights = torch.cat([self.weights, weights], dim=0)
        if self.colors is None:
            self.colors = colors
        else:
            self.colors = torch.cat([self.colors, colors], dim=0)

    def add_full_connections(self, input_layer, output_layer):
        in_indices = self.layers[input_layer].flatten()
        out_indices = self.layers[output_layer].flatten()
        connections = in_indices.repeat([len(out_indices), 1])
        weights = torch.ones_like(connections, dtype=torch.float)
        self.connections.append((out_indices, connections, weights))
        self.layer_connections[input_layer].append(output_layer)

    def add_conv1d_connections(self, input_layer, output_layer, kernel_size, stride=1, padding=(0, 0)):
        in_indices = self.layers[input_layer]
        out_indices = self.layers[output_layer]
        in_channels = in_indices.shape[0]

        connection_to_channels = torch.arange(in_channels).repeat(kernel_size).view(1, 1, -1)
        connection_to_channels = connection_to_channels.repeat(out_indices.shape[0], out_indices.shape[1], 1)

        kernel_offset = torch.arange(kernel_size).view(-1, 1).repeat(1, in_channels).view(1, -1)
        connection_to_time = (torch.arange(out_indices.shape[1]) * stride).unsqueeze(1) + kernel_offset - padding[0]
        connection_to_time[connection_to_time < 0] = -1
        connection_to_time[connection_to_time >= in_indices.shape[1]] = -1
        connection_to_time = connection_to_time.unsqueeze(0).repeat(out_indices.shape[0], 1, 1)

        connections = in_indices[connection_to_channels, connection_to_time]
        connections[connection_to_time < 0] = -1
        connections = connections.view(-1, in_channels*kernel_size)
        weights = torch.ones_like(connections, dtype=torch.float)

        self.connections.append((out_indices.flatten(), connections, weights))
        self.layer_connections[input_layer].append(output_layer)

    def collapse_layers(self, factor=2, dimension=0):
        collapsed_graph = Network()
        collapsed_graph.parent_graph = self
        expand_lookup = torch.zeros(self.num_units, dtype=torch.long)
        for name, indices in self.layers.items():
            if dimension >= len(indices.shape) or indices.shape[dimension] < factor:
                collapse_factor = 1
            else:
                collapse_factor = factor

            padding = [0] * (len(indices.shape) * 2)
            padding[len(padding) - 1 - dimension*2] = math.ceil(indices.shape[dimension] / collapse_factor) * collapse_factor - indices.shape[dimension]
            padded_indices = F.pad(indices, padding, value=-1)
            view_shape = list(padded_indices.shape)
            view_shape[dimension] = padded_indices.shape[dimension] // collapse_factor
            view_shape.insert(dimension + 1, collapse_factor)
            collapse_lookup = padded_indices.view(view_shape)
            permutation = list(range(len(view_shape)))
            permutation.pop(dimension+1)
            permutation.append(dimension+1)
            collapse_lookup = collapse_lookup.permute(permutation)
            collapsed_layer_shape = list(collapse_lookup.shape)
            collapsed_layer_shape.pop(-1)
            collapse_lookup = collapse_lookup.contiguous().view(-1, collapse_factor)

            collapsed_graph.add_layer(name, shape=collapsed_layer_shape)

            new_indices = collapsed_graph.layers[name].view(-1, 1).repeat(1, collapse_factor).flatten()
            allow = collapse_lookup.flatten() >= 0
            expand_lookup[collapse_lookup.flatten()[allow]] = new_indices[allow]
        collapsed_graph.expand_lookup = expand_lookup

        collapse_num = torch.zeros_like(collapsed_graph.weights)
        collapse_num.scatter_add_(0, expand_lookup, torch.ones_like(expand_lookup, dtype=torch.float))
        collapsed_graph.positions *= 0
        collapsed_graph.positions.scatter_add_(0, expand_lookup.unsqueeze(1).repeat([1, self.num_dim]), self.positions)
        collapsed_graph.positions /= collapse_num.unsqueeze(1)
        collapsed_graph.weights *= 0
        collapsed_graph.weights.scatter_add_(0, expand_lookup, self.weights)

        for origins, targets, weights in self.connections:
            new_origins = expand_lookup[origins]
            new_targets = torch.zeros_like(targets) - 1
            allow = targets > 0
            new_targets[allow] = expand_lookup[targets][allow]

            #unique_origins, unique_inverse = torch.unique(new_origins, return_inverse=True)
            #new_weights = torch.zeros([unique_origins.shape[0], weights.shape[1]], device=weights.device)
            #new_weights.scatter_add_(0, , weights)
            collapsed_graph.connections.append((new_origins, new_targets, weights))
        collapsed_graph.expand_lookup = expand_lookup
        return collapsed_graph

    def give_positions_to_parent(self, perturbation=1e-2):
        self.parent_graph.positions = self.positions[self.expand_lookup, :]
        self.parent_graph.positions += (torch.rand_like(self.parent_graph.positions) * 2. - 1.) * perturbation
        return self.parent_graph

    def connection_count_per_unit(self):
        connection_counts = torch.zeros(self.num_units, dtype=torch.long)
        for origins, targets, weights in self.connections:
            connection_counts[origins] += targets.shape[1]
            for i in range(targets.shape[0]):
                connection_counts[targets[i, :]] += 1
        return connection_counts

    def to(self, device):
        for i, (origins, targets, weights) in enumerate(self.connections):
            self.connections[i] = (origins.to(device), targets.to(device), weights.to(device))
        for key, value in self.layers.items():
            self.layers[key] = value.to(device)

    # PLOTTING #########
    def line_data(self):
        lines = []
        for origins, targets, weights in self.connections:
            origin_positions = self.positions[origins, :2].unsqueeze(1).repeat([1, targets.shape[1], 1])
            target_positions = self.positions[targets, :2]
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
        self.scatter = self.ax.scatter(self.positions[:, 0], self.positions[:, 1], linewidths=1, c=self.colors)
        self.ax.autoscale()
        self.fig.canvas.draw()
        plt.show()

    def set_default_colors(self, colormap='Set1'):
        cmap = matplotlib.cm.get_cmap(colormap)
        layer_wise_coloring = False
        try:
            layer_wise_coloring = True
            num_colors = len(cmap.colors)
        except:
            layer_wise_coloring = False
        for l, (name, indices) in enumerate(self.layers.items()):
            if layer_wise_coloring:
                color = torch.FloatTensor(cmap(l % num_colors))
                i = indices.flatten()
                self.colors[i, :] = color
            else:
                i = indices.flatten()
                colors = torch.from_numpy(cmap(i.cpu().float() / self.num_units))
                self.colors[i, :] = colors.float()


class NetworkForceLayout:
    def __init__(self,
                 network,
                 num_dim=2,
                 repulsion=-0.005,
                 spring_optimal_distance=0.01,
                 centering=0.,
                 drag=1.,
                 noise=0.,
                 attraction_normalization=0.,
                 step_size=0.1,
                 step_discount_factor=0.9,
                 device='cpu',
                 mac=0.7,
                 force_limit=0.1):
        self.network = network
        self.device = device
        self.mac = mac
        self.network.to(device)
        self.num_dim = num_dim
        self.x = torch.randn([self.network.num_units, self.num_dim], device=self.device)
        #self.x *= self.network.num_units**0.5
        self.v = torch.zeros_like(self.x)
        self.a = torch.zeros_like(self.x)
        self.movable = torch.ones(self.network.num_units, device=self.device)
        self.colors = torch.ones([self.network.num_units, 4], device=self.device)
        self.set_default_colors()
        self.connection_counts = self.network.connection_count_per_unit().float().to(self.device)
        self.avg_connection_count = torch.mean(self.connection_counts)
        self.force_limit = force_limit

        self.repulsion = repulsion
        self.spring_optimal_distance = spring_optimal_distance
        self.centering = centering
        self.drag = drag
        self.noise = noise
        self.attraction_normalization = attraction_normalization
        self.step_size = step_size
        self.step_discount_factor = step_discount_factor
        self.max_levels = 16
        self.energy = float("inf")
        self.enegry_progress = 0

        #plt.ion()
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig = plt.figure()
        #self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax = self.fig.add_subplot(111)

        #self.scatter = None
        #self.lines = None

        #self.plotting_thread = None
        #plt.show()

    def set_position(self, indices, pos, fix=False):
        i = indices.flatten()
        self.x[i, :] = pos
        if fix:
            self.movable[i] = 0.

    def simulation_step(self):
        #self.x += self.movable.unsqueeze(1) * (self.v * self.step_size + 0.5 * self.a * self.step_size ** 2)

        f = torch.zeros_like(self.x)
        force_noise = torch.randn_like(self.x) * self.noise
        f += force_noise

        # electrostatic repulsion
        electrical_force = torch.zeros_like(f)
        if self.repulsion != 0.0:
            if self.mac > 0:
                mass = torch.ones_like(self.x[:, 0])
                qt = BarnesHutTree(self.x, mass, device=self.device, max_levels=self.max_levels)
                electrical_force = qt.traverse(self.x, mass, mac=self.mac, force_function=electrostatic_function)
            else:
                diff = self.x.unsqueeze(1) - self.x.unsqueeze(0)
                electrical_force = torch.sum(diff / ((torch.norm(diff, 2, dim=2, keepdim=True) ** 2) + 1e-5), dim=0)
            electrical_force *= self.repulsion * self.spring_optimal_distance ** 2
        f += electrical_force

        #f = torch.zeros_like(self.x)

        #for i in range(f.shape[0]):
        #    print("index", i, "- f:", f[i, :], "- f_g:", f_g[i, :])
        #f_d = torch.sum((f-f_g)**2)

        # attraction
        attraction_force = torch.zeros_like(f)
        if self.spring_optimal_distance != 0.0:
            attraction_force = torch.zeros_like(f)
            for origins, targets, weights in self.network.connections:
                diff = self.x[targets, :] - self.x[origins, :].unsqueeze(1)
                dist = torch.norm(diff, 2, dim=2, keepdim=True)
                a_f = (diff * dist) / self.spring_optimal_distance
                a_f[targets < 0] = 0.
                attraction_force[origins, :] += torch.sum(a_f, dim=1)
                attraction_force[targets, :] -= a_f
            if self.attraction_normalization > 0.:
                attraction_force *= self.avg_connection_count / (1 + self.attraction_normalization*(self.connection_counts.unsqueeze(1)-1))
        f += attraction_force

        # centering
        dist = torch.norm(self.x, 2, dim=1, keepdim=True)
        centering_force = -self.centering * (self.x / dist) * dist ** 2
        f += centering_force

        # drag
        #v_norm = torch.norm(self.v, 2, dim=1)
        #drag_force = -self.drag * (self.v / (v_norm.unsqueeze(1) + 1e-9)) * v_norm.unsqueeze(1) ** 2
        #f += drag_force

        f_norm = torch.norm(f, 2, dim=1)
        ratio_noise = torch.mean(torch.norm(force_noise, 2, dim=1) / f_norm)
        ratio_gravity = torch.mean(torch.norm(electrical_force, 2, dim=1) / f_norm)
        ratio_attraction = torch.mean(torch.norm(attraction_force, 2, dim=1) / f_norm)
        ratio_centering = torch.mean(torch.norm(centering_force, 2, dim=1) / f_norm)
        #ratio_drag = torch.mean(torch.norm(drag_force, 2, dim=1) / f_norm)
        energy = torch.sum(f ** 2)

        #out_of_bound = f_norm > self.force_limit
        #f[out_of_bound] = self.force_limit * f[out_of_bound] / f_norm[out_of_bound].unsqueeze(1)

        a = f  # since we use mass = 1 for all nodes

        x_update = f / f_norm.unsqueeze(1)

        self.x += self.step_size * x_update

        # velocity verlet integration
        #self.v /= (1 + self.drag)
        #self.v += 0.5 * (self.a + a) * self.step_size
        #self.a = a

        self.update_step_size(energy)
        self.energy = energy

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

    def update_step_size(self, new_energy):
        if new_energy < self.energy:
            self.enegry_progress += 1
            if self.enegry_progress >= 5:
                self.enegry_progress = 0
                self.step_size /= self.step_discount_factor
        else:
            self.enegry_progress = 0
            self.step_size *= self.step_discount_factor


    # PLOTTING #########
    def line_data(self):
        lines = []
        for origins, targets, weights in self.network.connections:
            origin_positions = self.x[origins, :2].unsqueeze(1).repeat([1, targets.shape[1], 1])
            target_positions = self.x[targets, :2]
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
        layer_wise_coloring = False
        try:
            layer_wise_coloring = True
            num_colors = len(cmap.colors)
        except:
            layer_wise_coloring = False
        for l, (name, indices) in enumerate(self.network.layers.items()):
            if layer_wise_coloring:
                color = torch.FloatTensor(cmap(l % num_colors))
                color = color.to(self.device)
                i = indices.flatten()
                self.colors[i, :] = color
            else:
                i = indices.flatten()
                colors = torch.from_numpy(cmap(i.cpu().float() / self.network.num_units))
                colors = colors.to(self.device)
                self.colors[i, :] = colors.float()


class NetworkGradientLayout:
    def __init__(self,
                 network,
                 gravity=-0.005,
                 attraction=0.01,
                 centering=0.1,
                 noise=0.,
                 attraction_normalization=0.,
                 lr=0.1,
                 device='cpu',
                 mac=0.7,
                 connection_target=0.):
        self.network = network
        self.device = device
        self.mac = mac
        self.network.to(device)
        self.x = torch.randn([self.network.num_units, 2], device=self.device)
        self.x *= self.network.num_units**0.5 * 0.1
        self.x.requires_grad = True
        self.movable = torch.ones(self.network.num_units, device=self.device)
        self.colors = torch.ones([self.network.num_units, 4], device=self.device)
        self.set_default_colors()
        self.connection_counts = self.network.connection_count_per_unit().float().to(self.device)
        self.avg_connection_count = torch.mean(self.connection_counts)
        self.connection_target = connection_target

        self.gravity = gravity
        self.attraction = attraction
        self.centering = centering
        self.noise = noise
        self.attraction_normalization = attraction_normalization
        self.lr = lr
        self.max_levels = 16

        self.optimizer = torch.optim.Adam([self.x], lr=self.lr)

        self.fig, self.ax = plt.subplots(1, 1)

    def set_position(self, indices, pos, fix=False):
        i = indices.flatten()
        self.x[i, :] = pos
        if fix:
            self.movable[i] = 0.

    def loss(self):
        gravity_loss = torch.zeros(size=[], device=self.device)
        if self.gravity != 0.0:
            if self.mac > 0:
                mass = torch.ones_like(self.x[:, 0])
                qt = BarnesHutTree(self.x, mass, device=self.device, max_levels=self.max_levels)
                bh_force =self.gravity * qt.traverse(self.x, mass, mac=self.mac, force_function=energy_function)
                gravity_loss = torch.sum(bh_force[:, 0])
            else:
                diff = self.x.unsqueeze(1) - self.x.unsqueeze(0)
                dist = torch.norm(diff, 2, dim=2)
                bf_force = self.gravity * torch.sum(1 / ((dist**2) + 1e-5))
                gravity_loss = torch.sum(torch.norm(bf_force, 2, dim=1))

        attraction_loss = torch.zeros_like(gravity_loss)
        if self.attraction != 0.0:
            per_point_attraction = torch.zeros_like(self.x[:, 0])
            for origins, targets, weights in self.network.connections:
                diff = self.x[targets, :] - self.x[origins, :].unsqueeze(1)
                dist = torch.norm(diff, 2, dim=2)
                #dist[targets < 0] = 0.

                per_point_attraction[origins] += self.attraction * torch.sum(dist, dim=1)
                per_point_attraction[targets] += self.attraction * dist
            if self.attraction_normalization > 0.:
                per_point_attraction *= self.avg_connection_count / (1 + self.attraction_normalization*(self.connection_counts-1))
            attraction_loss = torch.sum(per_point_attraction)

        dist = torch.norm(self.x, 2, dim=1)
        centering_loss = self.centering * torch.sum(dist)

        loss = gravity_loss + attraction_loss + centering_loss
        return loss

    def simulation_step(self):
        loss = self.loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.x.grad += torch.randn_like(self.x) * self.noise
        grad_norm = torch.norm(self.x.grad, 2, dim=1)
        out_of_bound = grad_norm > 100.
        self.x.grad[out_of_bound] = 100. * self.x.grad[out_of_bound] / grad_norm[out_of_bound].unsqueeze(1)
        self.optimizer.step()

    def set_default_colors(self, colormap='Set1'):
        cmap = matplotlib.cm.get_cmap(colormap)
        layer_wise_coloring = False
        try:
            layer_wise_coloring = True
            num_colors = len(cmap.colors)
        except:
            layer_wise_coloring = False
        for l, (name, indices) in enumerate(self.network.layers.items()):
            if layer_wise_coloring:
                color = torch.FloatTensor(cmap(l % num_colors))
                color = color.to(self.device)
                i = indices.flatten()
                self.colors[i, :] = color
            else:
                i = indices.flatten()
                colors = torch.from_numpy(cmap(i.cpu().float() / self.network.num_units))
                colors = colors.to(self.device)
                self.colors[i, :] = colors.float()

    def line_data(self):
        lines = []
        for origins, targets, weights in self.network.connections:
            origin_positions = self.x[origins, :].unsqueeze(1).repeat([1, targets.shape[1], 1])
            target_positions = self.x[targets, :]
            lines.append(torch.stack([origin_positions.view(-1, 2), target_positions.view(-1, 2)], dim=1))
        lines = torch.cat(lines, dim=0)
        return lines.detach()


def animation_step(i, simulation, plot_connections=True):
    for _ in range(1):
        simulation.simulation_step()
    try:
        simulation.scatter.remove()
        simulation.lines.remove()
    except:
        pass
    if plot_connections:
        simulation.lines = mc.LineCollection(simulation.line_data(), lw=0.5, alpha=0.2)
        simulation.ax.add_collection(simulation.lines)
    print("energy:", simulation.energy)
    print("step size:", simulation.step_size)
    pos = simulation.x.detach()
    simulation.scatter = simulation.ax.scatter(pos[:, 0], pos[:, 1], c=simulation.colors, s=8.)
    simulation.ax.autoscale()
    simulation.fig.canvas.draw()
    # plt.draw()
    #plt.show()


def animate_simulation(simulation):
    ani = matplotlib.animation.FuncAnimation(simulation.fig, animation_step, frames=50, fargs=(simulation, True))
    return ani






