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

        connection_counts = net.in_connection_weight_per_unit()
        pass

    def test_collapsing(self):
        net = Network()
        net.add_layer('input_layer', [2, 12])
        net.add_layer('hidden_layer_1', [512, 12])
        net.add_layer('hidden_layer_2', [512, 8])
        net.add_layer('output_layer', [1, 10])

        net.add_conv1d_connections('input_layer', 'hidden_layer_1', kernel_size=3, padding=(1, 1))
        # net.add_full_connections('hidden_layer', 'output_layer')
        net.add_conv1d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=3)
        net.add_full_connections('hidden_layer_2', 'output_layer')

        collapsed_graph = net.collapse_layers(factor=3, dimension=1)
        num_units = collapsed_graph.num_units
        num_connections = collapsed_graph.num_connections
        pass

    def test_collapsing_2(self):
        net = Network()
        net.add_layer('input_layer', [2, 256])
        net.add_layer('hidden_1', [8, 126])
        net.add_layer('hidden_2', [8, 126])
        # net.add_layer('input_layer', [2, 256])
        # net.add_layer('hidden_1', [32, 126])
        # net.add_layer('hidden_2', [32, 126])
        # net.add_layer('hidden_3', [64, 61])
        # net.add_layer('hidden_4', [128, 30])
        # net.add_layer('hidden_5', [256, 26])
        # net.add_layer('output_layer', [512, 1])

        net.add_conv1d_connections('input_layer', 'hidden_1', kernel_size=5, stride=2)
        net.add_conv1d_connections('hidden_1', 'hidden_2', kernel_size=64, padding=(0, 63))
        # net.add_conv1d_connections('hidden_2', 'hidden_3', kernel_size=5, stride=2)
        # net.add_conv1d_connections('hidden_3', 'hidden_4', kernel_size=32)
        # net.add_conv1d_connections('hidden_4', 'hidden_5', kernel_size=5)
        # net.add_conv1d_connections('hidden_5', 'output_layer', kernel_size=26)

        for i in range(4):
            print("start collapsing")
            net = net.collapse_layers(factor=2, dimension=0)
            print("collapsing finished")
            net.set_default_colors('jet')
            print("unit count:", net.num_units)

    def test_inheritance(self):
        net = Network()
        net.add_layer('input_layer', [2, 12])
        net.add_layer('hidden_layer_1', [8, 12])
        net.add_layer('hidden_layer_2', [8, 8])
        net.add_layer('output_layer', [1, 10])

        net.add_conv1d_connections('input_layer', 'hidden_layer_1', kernel_size=3, padding=(1, 1))
        net.add_conv1d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=3)
        net.add_full_connections('hidden_layer_2', 'output_layer')

        collapse_1 = net.collapse_layers(factor=2, dimension=0)
        collapse_2 = collapse_1.collapse_layers(factor=3, dimension=1)

        reconstruction_1 = collapse_2.give_positions_to_parent(perturbation=0.01)
        reconstruction_net = reconstruction_1.give_positions_to_parent(perturbation=0.01)

        collapse_2.set_default_colors('jet')
        reconstruction_1.set_default_colors('jet')
        reconstruction_net.set_default_colors('jet')

        plot_1 = NetworkPlot()
        collapse_2.plot_on(plot_1)

        plot_2 = NetworkPlot()
        reconstruction_1.plot_on(plot_2)

        plot_3 = NetworkPlot()
        reconstruction_net.plot_on(plot_3)
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
                                    spring_optimal_distance=0.5,
                                    attraction_normalization=0.,
                                    repulsion=2.,
                                    step_size=0.5,
                                    step_discount_factor=0.9,
                                    centering=0.,
                                    drag=0.,
                                    noise=0.,
                                    mac=0.7,
                                    num_dim=2)
        net.set_default_colors('jet')
        #layout.set_default_colors('jet')
        #input_positions = torch.linspace(-1.5, 1.5, 12)
        #input_positions = input_positions.repeat([2, 1])
        #input_positions = torch.stack([input_positions, torch.zeros([2, 12])-2.], dim=2)
        #input_positions[1, :, 1] -= 0.2
        #layout.set_position(net.layers['input_layer'], input_positions.view(-1, 2), fix=False)

        #output_positions = torch.linspace(-2, 2, 10).unsqueeze(0)
        #output_positions = torch.stack([output_positions, torch.zeros([1, 10]) + 2.], dim=2)
        #layout.set_position(net.layers['output_layer'], output_positions.view(-1, 2), fix=False)
        #layout.simulation_step()
        #line_data = layout.line_data()

        #layout.plot()
        #layout.simulate(num_steps=2000, attraction=0.2, centering=0.2, normalize_attraction=True,
        #                plot_interval=1000)
        #layout.plot()
        plot = NetworkPlot()
        ani = animate_simulation(layout, plot)
        plt.show()

    def test_multilevel_simulation(self):
        plot = NetworkPlot()

        net = Network()
        net.add_layer('input_layer', [2, 12])
        net.add_layer('hidden_layer_1', [8, 12])
        net.add_layer('hidden_layer_2', [8, 8])
        net.add_layer('output_layer', [1, 10])

        net.add_conv1d_connections('input_layer', 'hidden_layer_1', kernel_size=3, padding=(1, 1))
        net.add_conv1d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=3)
        net.add_full_connections('hidden_layer_2', 'output_layer')
        #net.add_conv1d_connections('input_layer', 'hidden_layer_2', kernel_size=3)
        net.add_full_connections('hidden_layer_1', 'output_layer')

        net.set_default_colors('jet')

        net = net.collapse_layers(factor=2, dimension=0)
        net.set_default_colors('jet')
        net = net.collapse_layers(factor=2, dimension=0)
        net.set_default_colors('jet')
        net = net.collapse_layers(factor=2, dimension=1)
        net.set_default_colors('jet')
        net = net.collapse_layers(factor=2, dimension=1)
        net.set_default_colors('jet')
        net = net.collapse_layers(factor=2, dimension=1)
        net.set_default_colors('jet')
        net = net.collapse_layers(factor=2, dimension=1)
        net.set_default_colors('jet')

        net.to('cpu')

        global layout
        global current_net
        global last_positions
        layout = None
        current_net = net
        last_positions = net.positions.clone()

        def multilevel_animation_step(i):
            global layout
            global current_net
            global last_positions
            position_change = torch.mean(torch.norm(current_net.positions - last_positions, 2, dim=1))
            last_positions = current_net.positions.clone()
            print("position change:", position_change)
            if position_change < 0.001:
                if i > 0:
                    current_net = current_net.give_positions_to_parent(perturbation=0.1)
                    last_positions = current_net.positions.clone()
                layout = NetworkForceLayout(current_net,
                                            spring_optimal_distance=1.,
                                            attraction_normalization=1.,
                                            repulsion=1.,
                                            step_size=0.1,
                                            step_discount_factor=0.9,
                                            centering=0.,
                                            drag=0.5,
                                            noise=0.5,
                                            mac=0.5,
                                            num_dim=2,
                                            force_limit=1.)
            animation_step(i, layout, plot, True)

        ani = matplotlib.animation.FuncAnimation(plot.fig,
                                                 multilevel_animation_step,
                                                 frames=1000,
                                                 interval=50,
                                                 repeat=False)
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




