import unittest
import numpy as np
from dropout import Dropout

#---Basic Testing---

# Create inputs and dropout instance
inputs = np.array([[1.0, 2.0, 3.0], 
                   [4.0, 5.0, 6.0]])
dropout = Dropout(rate=0.5)

# Training mode
dropout.forward(inputs, training=True)
print("Training mode output:\n", dropout.output)

# Inference mode
dropout.forward(inputs, training=False)
print("Inference mode output (should match inputs):\n", dropout.output)

# Assert inference mode output matches the inputs
assert np.allclose(dropout.output, inputs), "Inference mode failed!"

#---Statistical Validation---

# Generate random inputs
inputs = np.random.randn(1000, 1000)

# Apply dropout during training
dropout.forward(inputs, training=True)

# Calculate mean of inputs and outputs
input_mean = np.mean(inputs)
output_mean = np.mean(dropout.output)

# Validate scaling
print(f"Input mean: {input_mean}, Output mean: {output_mean}")
assert np.isclose(output_mean, input_mean, atol=0.1), "Scaling failed!"

#---binary Mask Test---
# Training mode
dropout.forward(inputs, training=True)

# Proportion of active neurons
active_neurons = np.sum(dropout.binary_mask) / np.prod(dropout.binary_mask.shape)
expected_rate = dropout.rate

print(f"Active neurons: {active_neurons}, Expected: {expected_rate}")
assert np.isclose(active_neurons, expected_rate, atol=0.05), "Binary mask proportion incorrect!"


#---backward pass Test---
# Gradient from the next layer
dvalues = np.ones_like(inputs)

# Forward and backward passes
dropout.forward(inputs, training=True)
dropout.backward(dvalues)

# Check gradients
print("Gradient flow test:")
print("Binary mask:\n", dropout.binary_mask)
print("Backpropagated gradients:\n", dropout.dinputs)

# Ensure gradients are scaled properly
assert np.allclose(dropout.dinputs, dvalues * dropout.binary_mask), "Gradient scaling failed!"

#---Edge case testing---

# Rate = 0 (no dropout)
dropout = Dropout(rate=0.0)
dropout.forward(inputs, training=True)
assert np.allclose(dropout.output, inputs), "Rate=0 test failed!"

# Rate = 1 (all neurons dropped)
dropout = Dropout(rate=1.0)
dropout.forward(inputs, training=True)
assert np.allclose(dropout.output, 0), "Rate=1 test failed!"

# Empty input
dropout = Dropout(rate=0.5)
empty_inputs = np.array([])
dropout.forward(empty_inputs, training=True)
assert dropout.output.size == 0, "Empty input test failed!"

class TestDropout(unittest.TestCase):
    def test_inference(self):
        inputs = np.random.randn(10, 10)
        dropout = Dropout(rate=0.5)
        dropout.forward(inputs, training=False)
        self.assertTrue(np.allclose(dropout.output, inputs), "Inference mode failed!")
    
    def test_scaling(self):
        inputs = np.random.randn(1000, 1000)
        dropout = Dropout(rate=0.5)
        dropout.forward(inputs, training=True)
        self.assertAlmostEqual(np.mean(inputs), np.mean(dropout.output), delta=0.1, msg="Scaling failed!")

    def test_binary_mask(self):
        inputs = np.random.randn(100, 100)
        dropout = Dropout(rate=0.5)
        dropout.forward(inputs, training=True)
        active_neurons = np.sum(dropout.binary_mask) / np.prod(dropout.binary_mask.shape)
        self.assertAlmostEqual(active_neurons, dropout.rate, delta=0.05, msg="Binary mask proportion incorrect!")

if __name__ == "__main__":
    unittest.main()