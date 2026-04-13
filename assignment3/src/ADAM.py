"""
Implementation of the ADAM optimizer.
"""
import numpy as np
from model import Model
from nodes import LinearLayer
from optimizer import Optimizer
from typing import override

class ADAM(Optimizer):
    def __init__(self, model, loss_fn, lr, reg, vertical_flip_prob=0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(model, loss_fn, lr, reg, vertical_flip_prob)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # momentum and variance terms for each layer, initialized to zero
        self.m_W = [np.zeros_like(layer.W) for layer in self.model.layers if isinstance(layer, LinearLayer)]
        self.v_W = [np.zeros_like(layer.W) for layer in self.model.layers if isinstance(layer, LinearLayer)]
        self.m_b = [np.zeros_like(layer.b) for layer in self.model.layers if isinstance(layer, LinearLayer)]
        self.v_b = [np.zeros_like(layer.b) for layer in self.model.layers if isinstance(layer, LinearLayer)]
        self.t = 0

    @override
    def step(self, X: np.ndarray, Y: np.ndarray):
        """
        Applies one optimization step using ADAM optimizer.

        Args:
            X (numpy array): Input batch of shape (D, N) where N is the batch size and D is the dimensionality.
            Y (numpy array): True labels of shape (K, N) where K is the number of classes and N is the batch size.
        """      
        self.set_train_mode()
        # forward pass
        loss = self.compute_loss(X, Y)

        # backward pass
        grad_loss = self.loss_fn.backward()
        self.model.backward(grad_loss)

        # add L2 regularization gradient
        for layer in self.model.layers:
            if isinstance(layer, LinearLayer):
                layer.grad_W += 2 * self.reg * layer.W

        # update momentum and variance terms
        self.t += 1 
        i = 0
        for layer in self.model.layers:
            if isinstance(layer, LinearLayer):
                self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
                self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.grad_W ** 2)
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_b ** 2)
                # counter zero bias
                m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                # update parameters manually since we need to use the corrected momentum and variance terms
                layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                layer.grad_W = 0
                layer.grad_b = 0
                layer.X = None
                i += 1

        