import numpy as np

class Node:
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class LinearLayer(Node):
    def __init__(self, d_in, d_out):
        np.random.seed(42) # for reproducibility
        self.W = np.random.randn(d_out, d_in) * np.sqrt(2/d_in) # He initialization
        self.b = np.zeros((d_out, 1))
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

        Args:
            grad (numpy array): Upper gradient.

        returns:
            numpy array: Gradient with respect to the input of the layer.
        """
        self.grad_W += grad@self.X.T
        self.grad_b += np.sum(grad, axis=1, keepdims=True)
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

class ReLU(Node):
    def __init__(self):
        self.X = None 

    def forward(self, X):
        """
        Calculates the forward pass for a ReLU activation function, i.e. max(0, x).
        
        Args:
            X (numpy array): Input for the layer.
        
        Returns:
            numpy array: Output of the ReLU activation function.
        """
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, grad):
        """
        Calculates the vector jacobian product and returns column vector gradient according to the chain rule.

        Args:
            grad (numpy array): Upper (column vector) gradient.

        Returns:
            numpy array: Gradient with respect to the input of the layer.
        """
        assert self.X is not None, "The input has to be saved in cache"
        grad_copy = grad.copy()
        grad_copy[self.X <= 0] = 0 # mask out the negative values
        return grad_copy

class CrossEntropyLoss(Node):
    def __init__(self):
        self.P = None # save softmax probabilities for backward pass
        self.Y = None # save true labels for backward pass

    def forward(self, logits: np.array, Y: np.array) -> np.float64:
        """
        Calculates the forward pass for the cross-entropy loss with softmax.

        Args:
            logits (numpy array): Input logits of shape (K, N) where K is the number of classes and N is the batch size.
            Y (numpy array): True probability distribution of the classes with size (K, N) (most often one hot encoding).

        Returns:
            numpy.float64: The average cross-entropy loss over the batch.
        """
        self.Y = Y
        self.P = np.exp(logits)
        reg = np.sum(self.P,axis = 0, keepdims=True)
        self.P /= reg
        loss = -np.sum(Y * np.log(self.P)) / logits.shape[1]
        return loss
    
    def backward(self) -> np.array:
        """
        Calculates the gradient with respect to the input logits.

        Returns:
            numpy.array: Gradient of the loss with respect to the input logits, of shape (K, N).
        """
        return (self.P - self.Y) / self.P.shape[1]

class KBinaryCELoss(Node):
    def __init__(self):
        self.P = None # save sigmoid probabilityes for backward pass
        self.Y = None # save true labels for backward pass

    def forward(self, logits: np.array, Y: np.array) -> np.float64:
        """
        Calculates the forward pass for K-binary cross-entropy loss with sigmoid.
        
        Args:
            logits (numpy array): Input logits of shape (K, N) where K is the number of classes and N is the batch size.
            Y (numpy array): True probability distribution of the classes with size (K, N), where each column is a one-hot encoded vector.

        Returns:
            numpy.float64: The average binary cross-entropy loss over the batch.
        """
        self.Y = Y
        self.P = 1 / (1 + np.exp(-logits))
        # average over batch
        eps = 1e-12
        P_clipped = np.clip(self.P, eps, 1 - eps)   
        loss = -np.sum(Y*np.log(P_clipped) + (1-Y)*np.log(1-P_clipped)) / (logits.shape[1] * logits.shape[0])
        return loss
    
    def backward(self) -> np.array:
        """
        Calculates the gradient with respect to the input logits.

        Returns:
            numpy.array: Gradient of the loss with respect to the input logits, of shape (K, N).
        """
        return (self.P - self.Y) / (self.P.shape[1] * self.P.shape[0])