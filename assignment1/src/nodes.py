import numpy as np

class Node:
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class LinearLayer(Node):
    def __init__(self, d_in, d_out):
        np.random.seed(42) # for reproducibility
        self.W = np.random.randn(d_out, d_in) * 0.01
        self.b = np.zeros(d_out)
        self.X = None # save input for backward pass
        self.grad_W = 0
        self.grad_b = 0
    
    def forward(self, X):
        """
        Calculates the forward pass for a linear layer.

        Args:
            X (numpy array): Input data of shape (d_in, N) where N is the batch size.

        Returns:
            numpy array: Output of the linear layer W@X + b of shape (d_out, N).
        """
        self.X = X
        return self.W@X + self.b
    
    def backward(self, grad):
        """
        Calculates the vector jacobian product and returns column vector gradient according to the chain rule.
        """
        self.grad_W += grad@self.X.T *(1/grad.shape[1])
        self.grad_b += np.sum(grad, axis=1, keepdims=True) *(1/grad.shape[1])
        return self.W.T @ grad

    def update_params(self, lr: np.float64):
        """
        Updates the layer parameters and resets gradient.

        Args:
            lr (np.float64): The learning rate.
        """
        assert self.X is not None, "The input has to be saved in cache"
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b
        self.grad_W = 0
        self.grad_b = 0 
        self.X = None


class CrossEntropyLoss(Node):
    def __init__(self):
        self.P = None # save softmax probabilities for backward pass

    def forward(self, S: np.array, Y: np.array):
        """
        Calculates the forward pass for the cross-entropy loss with softmax.

        Args:
            S (numpy array): Input logits of shape (K, N) where K is the number of classes and N is the batch size.
            Y (numpy array): True probability distribution of the classes with size (K, N) (most often one hot encoding).
        """
        self.P = np.exp(S)
        reg = np.sum(self.P,axis = 0, keepdims=True)
        self.P /= reg
        loss = -np.sum(Y * np.log(self.P)) / S.shape[1]
        return loss
    
    def backward(self, Y: np.array):
        """
        Calculates the gradient with respect to the input logits S.

        Args:
            Y (numpy array): True probability distribution of the classes with size (K, N) (most often one hot encoding).
        """
        return self.P - Y
