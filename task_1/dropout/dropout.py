import numpy as np

class Dropout:
    
    def __init__(self,rate):
        """
        Initialize the Dropout layer.
        Parameters:
        - rate (float): Dropout rate (fraction of units to drop). Must be between 0 and 1.
        
        """
        
        if not (0 <= rate <= 1):
            raise ValueError("Dropout rate must be between 0 and 1.")
        self.rate = 1 - rate # Keep probability
        
    def forward (self, inputs, training=True):
        """
        Forward pass for the dropout layer.
        Parameters:
        - inputs (np.array): Input data.
        - training (bool): Whether the layer is in training mode.
        """
        
        self.inputs = inputs
        if training:
            # Generate binary mask and scale it
            self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape)
            if self.rate == 0:
                self.output = 0
            else:
                self.output = inputs * self.binary_mask / self.rate
        else:
            # No dropout during inference
            self.output = inputs
        
    def backward (self, dvalues):
        """
        Backward pass for the dropout layer.
        Parameters:
        - dvalues (np.array): Gradient of the loss with respect to the output.
        """
        
        # Gradient is passed only for active neurons
        self.dinputs = dvalues * self.binary_mask
        
        


