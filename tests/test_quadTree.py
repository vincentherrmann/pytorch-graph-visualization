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
        print("duration:", time.time() - tic)

        positions = torch.rand(1000, 2)
        mass = torch.ones(1000)
        tic = time.time()
        qt = QuadTree(positions, mass)
        qt.traverse(positions, mass)
        print("duration:", time.time() - tic)

        positions = torch.rand(10000, 2)
        mass = torch.ones(10000)
        tic = time.time()
        qt = QuadTree(positions, mass)
        qt.traverse(positions, mass)
        print("duration:", time.time() - tic)

        positions = torch.rand(100000, 2)
        mass = torch.ones(100000)
        tic = time.time()
        qt = QuadTree(positions, mass)
        qt.traverse(positions, mass)
        print("duration:", time.time() - tic)

        #duration: 0.007807016372680664
        #duration: 0.03351593017578125
        #duration: 0.30025196075439453
        #duration: 2.884951114654541
