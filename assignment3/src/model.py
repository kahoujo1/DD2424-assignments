"""
Implementation of a 2-layer FNN model for CIFAR-10 classification.
"""
import numpy as np

from nodes import Dropout, LinearLayer, ReLU, Patchify

class Model:
    def __init__(self, f: int, nf: int, d_hidden: int, K: int, p : float = 0.0):
        np.random.seed(42) # for reproducibility
        self.layers = [Patchify(f, nf),
                       ReLU(),
                       LinearLayer((32//f)**2 * nf, d_hidden),
                       ReLU(),
                       Dropout(p=p),
                       LinearLayer(d_hidden, K)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the model.
        
        Args:
            X (numpy array): Input data of shape (D, N) where N is the batch size and D is the dimensionality.

        Returns:
            numpy array: Output logits of shape (K, N) where K is the number of classes.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad: np.ndarray) -> None:
        """
        Performs the backward pass of the model.

        Args:
            grad (numpy array): Gradient of the loss with respect to the output logits.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self, lr: np.float64) -> None:
        """
        Updates the model parameters.
        
        Args:
            lr (np.float64): The learning rate.
        """
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                layer.update_params(lr)
            if isinstance(layer, Patchify):
                layer.update_params(lr)

    def set_train_mode(self, mode: bool) -> None:
        """
        Sets the model to training or evaluation mode.
        
        Args:
            mode (bool): If True, set to training mode. If False, set to evaluation mode.
        """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.set_training(mode)