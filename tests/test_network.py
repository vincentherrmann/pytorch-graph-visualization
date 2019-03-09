from unittest import TestCase
from network_graph import *

class TestNetwork(TestCase):
    def test_network(self):
        net = Network()
        net.add_layer('input_layer', [5, 12])
        net.add_layer('hidden_layer', [7, 12])
        net.add_layer('output_layer', [1, 16])

        net.add_conv1d_connections('input_layer', 'hidden_layer', kernel_size=3, padding=(1, 1))
        net.add_full_connections('hidden_layer', 'output_layer')
        assert net.num_units == 80

        connection_counts = net.connection_count_per_unit()
        pass

    def test_collapsing(self):
        net = Network()
        net.add_layer('input_layer', [2, 12])
        net.add_layer('hidden_layer_1', [8, 12])
        net.add_layer('hidden_layer_2', [8, 8])
        net.add_layer('output_layer', [1, 10])

        net.add_conv1d_connections('input_layer', 'hidden_layer_1', kernel_size=3, padding=(1, 1))
        # net.add_full_connections('hidden_layer', 'output_layer')
        net.add_conv1d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=3)
        net.add_full_connections('hidden_layer_2', 'output_layer')

        collapsed_graph = net.collapse_layers(factor=3, dimension=1)
        num_units = collapsed_graph.num_units
        num_connections = collapsed_graph.num_connections
        pass


class TestNetworkForceLayout(TestCase):
    def test_simulation(self):
        net = Network()
        net.add_layer('input_layer', [2, 12])
        net.add_layer('hidden_layer_1', [8, 12])
        net.add_layer('hidden_layer_2', [8, 8])
        net.add_layer('output_layer', [1, 10])

        net.add_conv1d_connections('input_layer', 'hidden_layer_1', kernel_size=3, padding=(1, 1))
        # net.add_full_connections('hidden_layer', 'output_layer')
        net.add_conv1d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=3)
        net.add_full_connections('hidden_layer_2', 'output_layer')

        #net.layer_connections = {'input_layer': ['hidden_layer_1', 'hidden_layer_2', 'output_layer'],
        #                         'hidden_layer_1': ['hidden_layer_2', 'output_layer'],
        #                         'hidden_layer_2': ['output_layer']}

        #net = net.collapse_layers(factor=3, dimension=0)
        #net = net.collapse_layers(factor=2, dimension=1)

        layout = NetworkForceLayout(net,
                                    spring_optimal_distance=1.,
                                    attraction_normalization=0.,
                                    repulsion=1.,
                                    step_size=0.2,
                                    step_discount_factor=0.9,
                                    centering=0.,
                                    drag=0.,
                                    noise=0.,
                                    mac=0.7,
                                    num_dim=2)
        layout.set_default_colors('jet')
        #input_positions = torch.linspace(-1.5, 1.5, 12)
        #input_positions = input_positions.repeat([2, 1])
        #input_positions = torch.stack([input_positions, torch.zeros([2, 12])-2.], dim=2)
        #input_positions[1, :, 1] -= 0.2
        #layout.set_position(net.layers['input_layer'], input_positions.view(-1, 2), fix=False)

        #output_positions = torch.linspace(-2, 2, 10).unsqueeze(0)
        #output_positions = torch.stack([output_positions, torch.zeros([1, 10]) + 2.], dim=2)
        #layout.set_position(net.layers['output_layer'], output_positions.view(-1, 2), fix=False)
        layout.simulation_step()
        line_data = layout.line_data()

        #layout.plot()
        #layout.simulate(num_steps=2000, attraction=0.2, centering=0.2, normalize_attraction=True,
        #                plot_interval=1000)
        #layout.plot()
        ani = animate_simulation(layout)
        plt.show()

class TestNetworkGradientLayout(TestCase):
    def test_simulation(self):
        net = Network()
        net.add_layer('input_layer', [2, 12])
        net.add_layer('hidden_layer_1', [8, 12])
        net.add_layer('hidden_layer_2', [8, 8])
        net.add_layer('output_layer', [1, 10])

        net.add_conv1d_connections('input_layer', 'hidden_layer_1', kernel_size=3, padding=(1, 1))
        #net.add_full_connections('hidden_layer', 'output_layer')
        net.add_conv1d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=3)
        net.add_full_connections('hidden_layer_2', 'output_layer')

        #net.layer_connections = {'input_layer': ['hidden_layer_1', 'hidden_layer_2', 'output_layer'],
        #                         'hidden_layer_1': ['hidden_layer_2', 'output_layer'],
        #                         'hidden_layer_2': ['output_layer']}

        layout = NetworkGradientLayout(net,
                                       attraction=0.5,
                                       attraction_normalization=0.1,
                                       gravity=-0.02,
                                       centering=0.01,
                                       noise=0.1,
                                       mac=0.5,
                                       lr=0.1)

        #loss = layout.loss()
        layout.set_default_colors('gist_ncar')
        #input_positions = torch.linspace(-1.5, 1.5, 12)
        #input_positions = input_positions.repeat([2, 1])
        #input_positions = torch.stack([input_positions, torch.zeros([2, 12])-2.], dim=2)
        #input_positions[1, :, 1] -= 0.2
        #layout.set_position(net.layers['input_layer'], input_positions.view(-1, 2), fix=False)

        #output_positions = torch.linspace(-2, 2, 10).unsqueeze(0)
        #output_positions = torch.stack([output_positions, torch.zeros([1, 10]) + 2.], dim=2)
        #layout.set_position(net.layers['output_layer'], output_positions.view(-1, 2), fix=False)
        layout.simulation_step()
        line_data = layout.line_data()

        #layout.plot()
        #layout.simulate(num_steps=2000, attraction=0.2, centering=0.2, normalize_attraction=True,
        #                plot_interval=1000)
        #layout.plot()
        ani = animate_simulation(layout)
        plt.show()


class TestBarnesHutSimulation(TestCase):
    def test_barnes_hut_simulation(self):
        net = Network()
        net.add_layer('input_layer', [1, 111])
        layout = NetworkForceLayout(net,
                                    spring_optimal_distance=0.0,
                                    attraction_normalization=True,
                                    repulsion=-0.005,
                                    step_size=0.1,
                                    centering=0.0,
                                    drag=1.,
                                    noise=0.0,
                                    mac=0.7)
        for step in range(10):
            layout.simulation_step()




