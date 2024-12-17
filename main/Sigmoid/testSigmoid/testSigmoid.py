import unittest
import numpy as np
import tensorflow as tf
import time
from main.Sigmoid.Sigmoid.Sigmoid import SigmoidLayer


class TestSigmoidLayer(unittest.TestCase):
    def setUp(self):
        self.sigmoid = SigmoidLayer()
        # Load and preprocess CIFAR-10
        (self.x_train, _), (self.x_test, _) = tf.keras.datasets.cifar10.load_data()
        self.x_test = self.x_test.astype('float32') / 255.0

    def test_forward_passes(self):
        # Get a single CIFAR-10 image and flatten it
        image = self.x_test[0].reshape(-1)  # 3072 dimensions

        print("\nTesting Sigmoid on CIFAR-10 image:")

        # Test Sigmoid forward pass
        start_time = time.time()
        sigmoid_output = self.sigmoid.forward(image)
        sigmoid_time = time.time() - start_time

        # Analyze Sigmoid results
        near_zero = np.sum(sigmoid_output < 0.1)
        near_one = np.sum(sigmoid_output > 0.9)
        middle_range = np.sum((sigmoid_output >= 0.1) & (sigmoid_output <= 0.9))

        print(f"\nSigmoid Results:")
        print(f"Forward pass time: {sigmoid_time:.4f} seconds")
        print(f"Near zero (<0.1): {near_zero} ({near_zero / len(image):.1%})")
        print(f"Middle range: {middle_range} ({middle_range / len(image):.1%})")
        print(f"Near one (>0.9): {near_one} ({near_one / len(image):.1%})")
        print(f"Value range: [{sigmoid_output.min():.3f}, {sigmoid_output.max():.3f}]")

        # Test with different batch sizes
        batch_sizes = [32, 64, 128, 256]
        print("\nTesting Sigmoid forward pass with different batch sizes:")
        for batch_size in batch_sizes:
            batch = self.x_test[:batch_size].reshape(batch_size, -1)
            start_time = time.time()
            _ = self.sigmoid.forward(batch)
            batch_time = time.time() - start_time
            print(f"Batch size {batch_size}: {batch_time:.4f} seconds")

        # Test key Sigmoid properties
        self.assertTrue(np.all(sigmoid_output > 0), "Sigmoid output should be greater than 0")
        self.assertTrue(np.all(sigmoid_output < 1), "Sigmoid output should be less than 1")

        # Test backward pass
        dout = np.random.randn(*image.shape)
        start_time = time.time()
        gradient = self.sigmoid.backward(dout)
        backward_time = time.time() - start_time
        print(f"\nBackward pass time: {backward_time:.4f} seconds")

        # Performance assertions
        self.assertLess(sigmoid_time, 0.01, "Sigmoid forward pass too slow")
        self.assertLess(backward_time, 0.01, "Sigmoid backward pass too slow")


if __name__ == "__main__":
    unittest.main()