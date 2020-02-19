from unittest import TestCase
from graph_visualization.layout_calculation import LayoutCalculation
import torch


class TestLayoutCalculation(TestCase):
    def test_interpolate_position(self):
        lc = LayoutCalculation(net=None)
        x = torch.ones(100)
        interp = lc.interpolate_position(x, pos=0., window_size=0.125)
        interp = lc.interpolate_position(x, pos=0.5, window_size=0.125)
        interp = lc.interpolate_position(x, pos=0.98, window_size=0.125)
        pass
