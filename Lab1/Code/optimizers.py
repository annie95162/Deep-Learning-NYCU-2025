import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr 

    def update(self, weig, grad):
        if weig is None or grad is None:
            raise ValueError("Error: weights (weig) or gradients (grad) cannot be None.")

        for key in weig.keys():
            if key not in grad:
                continue

            weig[key] -= self.lr * grad[key]
