from unittest import TestCase
from network_graph import *

class TestNetwork(TestCase):
    def test_network(self):
        net = Network()
        net.add_layer('input_layer', [2, 12])
        net.add_layer('hidden_layer', [4, 10])
        net.add_layer('output_layer', [1, 16])

        net.add_conv1d_connections('input_layer', 'hidden_layer', kernel_size=3)
        net.add_full_connections('hidden_layer', 'output_layer')
        assert net.num_units == 80

        connection_counts = net.connection_count_per_unit()
        pass


class TestNetworkForceLayout(TestCase):
    def test_simulation(self):
        net = Network()
        net.add_layer('input_layer', [2, 12])
        net.add_layer('hidden_layer_1', [8, 10])
        net.add_layer('hidden_layer_2', [8, 8])
        net.add_layer('output_layer', [1, 16])

        net.add_conv1d_connections('input_layer', 'hidden_layer_1', kernel_size=3)
        #net.add_full_connections('hidden_layer', 'output_layer')
        net.add_conv1d_connections('hidden_layer_1', 'hidden_layer_2', kernel_size=3)
        net.add_full_connections('hidden_layer_2', 'output_layer')

        #net.layer_connections = {'input_layer': ['hidden_layer_1', 'hidden_layer_2', 'output_layer'],
        #                         'hidden_layer_1': ['hidden_layer_2', 'output_layer'],
        #                         'hidden_layer_2': ['output_layer']}

        layout = NetworkForceLayout(net, attraction=0.01, normalize_attraction=True, gravity=-0.005)
        input_positions = torch.linspace(-1.5, 1.5, 12)
        input_positions = input_positions.repeat([2, 1])
        input_positions = torch.stack([input_positions, torch.zeros([2, 12])-2.], dim=2)
        input_positions[1, :, 1] -= 0.2
        layout.set_position(net.layers['input_layer'], input_positions.view(-1, 2), fix=True)

        output_positions = torch.linspace(-2, 2, 16).unsqueeze(0)
        output_positions = torch.stack([output_positions, torch.zeros([1, 16]) + 2.], dim=2)
        layout.set_position(net.layers['output_layer'], output_positions.view(-1, 2), fix=False)
        line_data = layout.line_data()

        #layout.plot()
        #layout.simulate(num_steps=2000, attraction=0.2, centering=0.2, normalize_attraction=True,
        #                plot_interval=1000)
        #layout.plot()
        ani = animate_simulation(layout)
        plt.show()


