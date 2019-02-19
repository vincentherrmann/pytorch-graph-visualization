from unittest import TestCase
from QuadTree import *
import time


class TestQuadTree(TestCase):
    def test_quad_tree(self):
        positions = torch.rand(100, 2)
        mass = torch.ones(100)
        tic = time.time()
        qt = QuadTree(positions, mass)
        qt.traverse(positions, mass)
        print("levels:", qt.levels)
        print("duration:", time.time() - tic)
        print("")

        positions = torch.rand(1000, 2)
        mass = torch.ones(1000)
        tic = time.time()
        qt = QuadTree(positions, mass)
        qt.traverse(positions, mass)
        print("levels:", qt.levels)
        print("duration:", time.time() - tic)
        print("")

        positions = torch.rand(10000, 2)
        mass = torch.ones(10000)
        tic = time.time()
        qt = QuadTree(positions, mass)
        qt.traverse(positions, mass)
        print("levels:", qt.levels)
        print("duration:", time.time() - tic)
        print("")

        positions = torch.rand(30000, 2)
        mass = torch.ones(30000)
        tic = time.time()
        qt = QuadTree(positions, mass)
        qt.traverse(positions, mass)
        print("levels:", qt.levels)
        print("duration:", time.time() - tic)
        print("")

        big_tensor = torch.rand(30000, 2)
        tic = time.time()
        diff = big_tensor.unsqueeze(0) - big_tensor.unsqueeze(1)
        dist = torch.norm(diff, 2, dim=2)
        print("brute force time 30000:", time.time()-tic)


        #duration: 0.007807016372680664
        #duration: 0.03351593017578125
        #duration: 0.30025196075439453
        #duration: 2.884951114654541
