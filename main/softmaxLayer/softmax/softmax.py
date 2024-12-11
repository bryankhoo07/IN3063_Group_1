import numpy as np

class softmaxLayer:
    def __init__(self):
        pass

    def forwardPass(self):
        exp_logits = np.exp(logits - np.max(logits, axis = 1, keepdims=True))

    def backwardPass(self):
        return None