import numpy as np
import torch
import matplotlib
import threading
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import time
from BarnesHutTree import *


class Network:
    def __init__(self):
        self.layers = {}
        self.connections = []
        self.layer_connections = {}
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
        self.layer_connections[name] = []

    def add_full_connections(self, input_layer, output_layer):
        in_indices = self.layers[input_layer].flatten()
        out_indices = self.layers[output_layer].flatten()
        sources = in_indices.repeat([len(out_indices), 1])
        self.connections.append((out_indices, sources))
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

        self.connections.append((out_indices.flatten(), connections.view(-1, in_channels*kernel_size)))
        self.layer_connections[input_layer].append(output_layer)

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
    def __init__(self,
                 network,
                 num_dim=2,
                 gravity=-0.005,
                 attraction=0.01,
                 centering=0.1,
                 drag=1.,
                 noise=0.,
                 attraction_normalization=0.,
                 step_size=0.1,
                 device='cpu',
                 mac=0.7,
                 force_limit=0.1,
                 connection_target=0.):
        self.network = network
        self.device = device
        self.mac = mac
        self.network.to(device)
        self.num_dim = num_dim
        self.x = torch.randn([self.network.num_units, self.num_dim], device=self.device)
        self.x *= self.network.num_units**0.5
        self.v = torch.zeros_like(self.x)
        self.a = torch.zeros_like(self.x)
        self.movable = torch.ones(self.network.num_units, device=self.device)
        self.colors = torch.ones([self.network.num_units, 4], device=self.device)
        self.set_default_colors()
        self.connection_counts = self.network.connection_count_per_unit().float().to(self.device)
        self.avg_connection_count = torch.mean(self.connection_counts)
        self.force_limit = force_limit
        self.connection_target = connection_target

        self.gravity = gravity
        self.attraction = attraction
        self.centering = centering
        self.drag = drag
        self.noise = noise
        self.attraction_normalization = attraction_normalization
        self.step_size = step_size
        self.max_levels = 16

        #plt.ion()
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

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
        self.x += self.movable.unsqueeze(1) * (self.v * self.step_size + 0.5 * self.a * self.step_size ** 2)

        f = torch.zeros_like(self.x)
        force_noise = torch.randn_like(self.x) * self.noise
        f += force_noise

        # gravity
        gravitational_force = torch.zeros_like(f)
        if self.gravity != 0.0:
            if self.mac > 0:
                mass = torch.ones_like(self.x[:, 0])
                qt = BarnesHutTree(self.x, mass, device=self.device, max_levels=self.max_levels)
                bh_force = self.gravity * qt.traverse(self.x, mass, mac=self.mac)
                gravitational_force = bh_force
            else:
                diff = self.x.unsqueeze(1) - self.x.unsqueeze(0)
                bf_force = self.gravity * torch.sum(diff / ((torch.norm(diff, 2, dim=2, keepdim=True)**3) + 1e-5), dim=0)
                gravitational_force = bf_force
        f += gravitational_force

        #f = torch.zeros_like(self.x)

        #for i in range(f.shape[0]):
        #    print("index", i, "- f:", f[i, :], "- f_g:", f_g[i, :])
        #f_d = torch.sum((f-f_g)**2)

        # attraction
        attraction_force = torch.zeros_like(f)
        if self.attraction != 0.0:
            attraction_force = torch.zeros_like(f)
            for origins, targets in self.network.connections:
                #diff = self.x[targets, :] - self.x[origins, :].unsqueeze(1)
                #dist = torch.norm(diff, 2, dim=2, keepdim=True)
                #attraction_force = self.attraction * (diff / dist) * (dist - self.connection_target)**2
                a_f = self.attraction * (self.x[targets, :] - self.x[origins, :].unsqueeze(1))
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
        ratio_gravity = torch.mean(torch.norm(gravitational_force, 2, dim=1) / f_norm)
        ratio_attraction = torch.mean(torch.norm(attraction_force, 2, dim=1) / f_norm)
        ratio_centering = torch.mean(torch.norm(centering_force, 2, dim=1) / f_norm)
        #ratio_drag = torch.mean(torch.norm(drag_force, 2, dim=1) / f_norm)

        out_of_bound = f_norm > self.force_limit
        f[out_of_bound] = self.force_limit * f[out_of_bound] / f_norm[out_of_bound].unsqueeze(1)

        a = f  # since we use mass = 1 for all nodes

        # velocity verlet integration
        self.v /= (1 + self.drag)
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
            for origins, targets in self.network.connections:
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
        for origins, targets in self.network.connections:
            origin_positions = self.x[origins, :].unsqueeze(1).repeat([1, targets.shape[1], 1])
            target_positions = self.x[targets, :]
            lines.append(torch.stack([origin_positions.view(-1, 2), target_positions.view(-1, 2)], dim=1))
        lines = torch.cat(lines, dim=0)
        return lines.detach()


def animation_step(i, simulation, plot_connections=True):
    for _ in range(20):
        simulation.simulation_step()
    try:
        simulation.scatter.remove()
        simulation.lines.remove()
    except:
        pass
    if plot_connections:
        simulation.lines = mc.LineCollection(simulation.line_data(), lw=0.5, alpha=0.2)
        simulation.ax.add_collection(simulation.lines)
    pos = simulation.x.detach()
    simulation.scatter = simulation.ax.scatter(pos[:, 0], pos[:, 1], pos[:, 1], c=simulation.colors)
    simulation.ax.autoscale()
    simulation.fig.canvas.draw()
    # plt.draw()
    #plt.show()


def animate_simulation(simulation):
    ani = matplotlib.animation.FuncAnimation(simulation.fig, animation_step, frames=50, fargs=(simulation, False))
    return ani






