import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from main.ReLuLayer.ReLu.ReLu import ReLuLayer


class TestReLuLayer(unittest.TestCase):
    def setUp(self):
        self.relu = ReLuLayer()

    def test_forward_scalar_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        output = self.relu.forward(x)
        assert_array_almost_equal(output, expected, decimal=6)

    def test_forward_matrix(self):
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        output = self.relu.forward(x)
        assert_array_almost_equal(output, expected, decimal=6)

    def test_backward_scalar_values(self):

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.relu.forward(x)


        dout = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        expected = np.array([0.0, 0.0, 0.0, 0.4, 0.5])
        grad = self.relu.backward(dout)
        assert_array_almost_equal(grad, expected, decimal=6)

    def test_backward_matrix(self):
        # First do forward pass to set up the cache
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        self.relu.forward(x)


        dout = np.array([[0.1, 0.2], [0.3, 0.4]])
        expected = np.array([[0.0, 0.2], [0.3, 0.0]])
        grad = self.relu.backward(dout)
        assert_array_almost_equal(grad, expected, decimal=6)