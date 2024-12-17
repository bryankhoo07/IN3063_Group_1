import unittest
import numpy as np
import tensorflow as tf
import time
from main.ReLuLayer.ReLu.ReLu import ReLuLayer


class TestActivationFunctions(unittest.TestCase):
    def setUp(self):
        self.relu = ReLuLayer()
        # Load and preprocess CIFAR-10
        (self.x_train, _), (self.x_test, _) = tf.keras.datasets.cifar10.load_data()
        self.x_test = self.x_test.astype('float32') / 255.0

    def test_forward_passes(self):
        # Get a single CIFAR-10 image and flatten it
        image = self.x_test[0].reshape(-1)  # 3072 dimensions

        print("\nTesting activations on CIFAR-10 image:")

        # Test ReLU forward pass
        start_time = time.time()
        relu_output = self.relu.forward(image)
        relu_time = time.time() - start_time

        # Analyze ReLU results
        relu_zeros = np.sum(relu_output == 0)
        relu_active = np.sum(relu_output > 0)
        print(f"\nReLU Results:")
        print(f"Forward pass time: {relu_time:.4f} seconds")
        print(f"Zero activations: {relu_zeros} ({relu_zeros / len(image):.1%})")
        print(f"Active neurons: {relu_active} ({relu_active / len(image):.1%})")
        print(f"Value range: [{relu_output.min():.3f}, {relu_output.max():.3f}]")

        # Test with different batch sizes
        batch_sizes = [32, 64, 128, 256]
        print("\nTesting ReLU forward pass with different batch sizes:")
        for batch_size in batch_sizes:
            batch = self.x_test[:batch_size].reshape(batch_size, -1)
            start_time = time.time()
            _ = self.relu.forward(batch)
            batch_time = time.time() - start_time
            print(f"Batch size {batch_size}: {batch_time:.4f} seconds")

        # Test some key properties
        self.assertTrue(np.all(relu_output >= 0), "ReLU output contains negative values")
        self.assertTrue(np.any(relu_output > 0), "ReLU output contains no positive values")
        self.assertTrue(relu_zeros + relu_active == len(image), "ReLU activation count mismatch")

        # Basic performance assertions
        self.assertLess(relu_time, 0.01, "ReLU forward pass too slow")


if __name__ == "__main__":
    unittest.main()