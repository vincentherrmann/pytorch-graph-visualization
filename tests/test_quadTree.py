from unittest import TestCase
from barnes_hut_tree import *
import time


class TestQuadTree(TestCase):
    def test_quad_tree(self):
        print("\n --- 100 points: ---\n ")
        positions = torch.rand(100, 2)
        mass = torch.ones(100)
        tic = time.time()
        qt = BarnesHutTree(positions, mass)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        tic = time.time()
        qt = BarnesHutTree(positions, mass, max_levels=4)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        print("\n --- 1000 points: ---\n ")
        positions = torch.rand(1000, 2)
        mass = torch.ones(1000)
        tic = time.time()
        qt = BarnesHutTree(positions, mass)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        tic = time.time()
        qt = BarnesHutTree(positions, mass, max_levels=4)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        print("\n --- 10000 points: ---\n ")
        positions = torch.rand(10000, 2)
        mass = torch.ones(10000)
        tic = time.time()
        qt = BarnesHutTree(positions, mass)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        tic = time.time()
        qt = BarnesHutTree(positions, mass, max_levels=4)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        tic = time.time()
        qt = BarnesHutTree(positions, mass, max_levels=8)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        tic = time.time()
        qt = BarnesHutTree(positions, mass, max_levels=10)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        print("\n --- 30000 points: ---\n ")
        positions = torch.rand(30000, 2)
        mass = torch.ones(30000)
        tic = time.time()
        qt = BarnesHutTree(positions, mass)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        tic = time.time()
        qt = BarnesHutTree(positions, mass, max_levels=4)
        qt.traverse(positions, mass)
        print("levels:", qt.num_levels)
        print("duration:", time.time() - tic)
        print("")

        #big_tensor = torch.rand(30000, 2)
        #tic = time.time()
        #diff = big_tensor.unsqueeze(0) - big_tensor.unsqueeze(1)
        #dist = torch.norm(diff, 2, dim=2)
        #print("brute force time 30000:", time.time()-tic)


        #duration: 0.007807016372680664
        #duration: 0.03351593017578125
        #duration: 0.30025196075439453
        #duration: 2.884951114654541

    def test_approximation_quality(self):
        #positions = torch.FloatTensor([[0.1, 0.], [1., 0.], [0.5, 0.49], [0.49, 0.48], [0.48, 0.47]]) #torch.rand(5, 2)
        #mass = torch.FloatTensor([1., 2., 3., 4., 5.])

        positions = torch.rand([100, 2])
        mass = torch.rand(100)

        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = torch.norm(diff, 2, dim=2)
        m = (mass.unsqueeze(0) * mass.unsqueeze(1)).unsqueeze(2)
        electrical_force = torch.sum(m * (diff / (dist.unsqueeze(2)**2 + 1e-9)), dim=0)

        qt = BarnesHutTree(positions, mass)

        #approx_1 = qt.traverse(positions, mass, force_function=electrostatic_function, mac=2.)
        #mse_1 = torch.mean(torch.norm(electrical_force - approx_1, 2, dim=1) ** 2)
        #approx_2 = qt.traverse(positions, mass, force_function=electrostatic_function, mac=1.)
        #mse_2 = torch.mean(torch.norm(electrical_force - approx_2, 2, dim=1) ** 2)
        #approx_3 = qt.traverse(positions, mass, force_function=electrostatic_function, mac=0.5)
        #mse_3 = torch.mean(torch.norm(electrical_force - approx_3, 2, dim=1) ** 2)
        #approx_4 = qt.traverse(positions, mass, force_function=electrostatic_function, mac=0.2)
        #mse_4 = torch.mean(torch.norm(electrical_force - approx_4, 2, dim=1) ** 2)
        approx_5 = qt.traverse(positions, mass, force_function=electrostatic_function, mac=0.01)
        mse_5 = torch.mean(torch.norm(electrical_force - approx_5, 2, dim=1) ** 2)

        assert mse_1 > mse_2
        assert mse_2 > mse_3
        assert mse_3 > mse_4
        assert mse_4 > mse_5

    # electrical_force:
    # tensor([[  -14.5011,   -15.1268],
    #         [   27.2456,   -23.3940],
    #         [  971.8731,   984.6697],
    #         [  395.7605,   412.8437],
    #         [-1380.3782, -1358.9927]])

