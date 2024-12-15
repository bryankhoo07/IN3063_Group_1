import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from main.Sigmoid.Sigmoid.Sigmoid import SigmoidLayer


class TestSigmoidLayer(unittest.TestCase):
    def setUp(self):
        self.sigmoid = SigmoidLayer()

    def test_forward_scalar_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.119203, 0.268941, 0.5, 0.731059, 0.880797])
        output = self.sigmoid.forward(x)
        assert_array_almost_equal(output, expected, decimal=4)

    def test_forward_matrix(self):
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        expected = np.array([[0.268941, 0.880797], [0.952574, 0.017986]])
        output = self.sigmoid.forward(x)
        assert_array_almost_equal(output, expected, decimal=4)

    def test_backward_scalar_values(self):

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.sigmoid.forward(x)


        dout = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        expected = np.array([0.010499, 0.039322, 0.075000, 0.078645, 0.052497])
        grad = self.sigmoid.backward(dout)
        assert_array_almost_equal(grad, expected, decimal=6)

    def test_backward_matrix(self):

        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        self.sigmoid.forward(x)


        dout = np.array([[0.1, 0.2], [0.3, 0.4]])
        expected = np.array([[0.019661, 0.020990], [0.013549, 0.007065]])
        grad = self.sigmoid.backward(dout)
        assert_array_almost_equal(grad, expected, decimal=4)