"""
Implementation of the 1-layer FNN model for CIFAR-10 classification.
"""
import numpy as np

from nodes import LinearLayer

class Model:
    def __init__(self, d_in: int, K: int):
        self.layers = [LinearLayer(d_in, K)]

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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class label probabilities for the input data.
        
        Args:
            X (numpy array): Input data of shape (D, N) where N is the batch size and D is the dimensionality.

        Returns:
            numpy array: Predicted class probabilities of shape (K, N) where K is the number of classes.
        """
        logits = self.forward(X)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=0, keepdims=True)
        return probs
    
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