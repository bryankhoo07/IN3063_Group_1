import unittest
import numpy as np
from main.softmaxLayer.softmax import SoftmaxLayer

class TestSoftmaxLayer(unittest.TestCase):
    def test_multiple_vectors(self):
        test_cases = [
            (np.array([[1.0, 2.0, 3.0]]), np.array([[0.09003057, 0.24472847, 0.66524096]])),
            (np.array([[2.0, 1.0, 0.1]]), np.array([[0.65900114, 0.24243297, 0.09856589]])),
            (np.array([[0.0, 0.0, 0.0]]), np.array([[0.33333333, 0.33333333, 0.33333333]])),
            (np.array([[5.0, 5.0, 5.0]]), np.array([[0.33333333, 0.33333333, 0.33333333]])),
            (np.array([[6543.2, 8954.3, 6785.487]]), np.array([[0.00000000, 1.00000000, 0.00000000]])),
        ]
        softmax = SoftmaxLayer()
        for input_vector, expected_output in test_cases:
            output = softmax.forward_pass(input_vector)
            np.testing.assert_almost_equal(output, expected_output, decimal=6)

if __name__ == "__main__":
    unittest.main()
