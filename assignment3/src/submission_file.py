import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import time

def load_batch(batchname: str):
    """
    Loads data from the CIFAR-10 dataset.

    Args: 
        batchname (string): the name of the batch to load, e.g. 'data_batch_1'

    Returns:
        Array of [X, Y, y] where
        - X contains the flattened image pixel data of shape (32x32x3, N)
        - Y is the true probability class distribution of shape (K, N) (one hot encoding)'
        - y is a vector of true labels (N,)
    """
    # Load a batch of training data
    cifar_dir = '../../assignment1/data/cifar-10-batches-py/'
    with open(cifar_dir + batchname, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    # Extract the image data and cast to float from the dict dictionary
    X = dict[b'data'].astype(np.float64) / 255.0
    X = X.transpose()
    nn = X.shape[1]
    # extract the labels
    y = dict[b'labels']
    y = np.array(y)
    # Create one hot-encoding of the labels
    K = 10
    Y = np.zeros((K, nn))
    Y[y, np.arange(nn)] = 1
    return X, Y, y

def load_training_batches():
    """
    Loads all the training batches and concatenates them.

    Returns:
        Array of [X, Y, y] where
        - X contains the flattened image pixel data of shape (32x32x3, N)
        - Y is the true probability class distribution of shape (K, N) (one hot encoding)'
        - y is a vector of true labels (N,)
    """
    X, Y, y = load_batch("data_batch_1")
    for i in range(2,6):
        X_temp, Y_temp, y_temp = load_batch(f"data_batch_{i}")
        X = np.concatenate((X, X_temp), axis=1)
        Y = np.concatenate((Y, Y_temp), axis=1)
        y = np.concatenate((y, y_temp))
    return X, Y, y

def calculate_mean_grad_difference(grad1: np.ndarray, grad2: np.ndarray) -> np.float64:
    """
    Calculates the mean relative error between two gradients.
    Args:
        grad1 (numpy array): The first gradient to compare.
        grad2 (numpy array): The second gradient to compare.
    Returns:
        numpy.float64: The mean relative error between the two gradients.
    """
    denum = max(1e-8, np.mean(np.abs(grad1) + np.abs(grad2)))
    return np.mean(np.abs(grad1 - grad2) / denum)

def precompute_Mx(X: np.ndarray, f: int) -> np.ndarray:
    """
    Precomputes the matrix Mx for the initial Patchify layer.
    
    Args:
        X (numpy array): Input image data of shape (32x32x3, N).
        f (int): The size of the filter (patch) to be extracted from the input image.
    
    Returns:
        numpy array: The precomputed Mx matrix of shape (Np, 3f^2, N), where Np is the number of patches and N is the batch size.
    """
    N = X.shape[1]
    X_ims = np.transpose(X.reshape((32, 32, 3, N), order='F'), (1, 0, 2, 3))
    Np = (32//f)**2
    Mx = np.zeros((Np, f*f*3, N), dtype=np.float32)
    for n in range(N):
        region = 0
        for i in range(32//f):
            for j in range(32//f):
                patch = X_ims[i*f:(i+1)*f, j*f:(j+1)*f, :, n]
                Mx[region, :, n] = patch.reshape((1, f*f*3), order='C')
                region += 1
    return Mx

class Node:
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class LinearLayer(Node):
    def __init__(self, d_in, d_out):
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
        return np.fmax(0, X)
    
    def backward(self, grad):
        """
        Calculates the vector jacobian product and returns column vector gradient according to the chain rule.

        Args:
            grad (numpy array): Upper (column vector) gradient.

        Returns:
            numpy array: Gradient with respect to the input of the layer.
        """
        assert self.X is not None, "The input has to be saved in cache"
        return grad * (self.X > 0)

class Dropout(Node):
    def __init__(self, p: np.float64):
        """
        Initializes the Dropout layer.

        Args:
            p (numpy.float64): The dropout probability (probability of dropping a unit).
        """
        self.p = p
        self.mask = None
        self.training = False
    
    def set_training(self, training: bool):
        """
        Sets the mode of the Dropout layer.

        Args:
            training (bool): If True, the layer is in training mode; if False, it is in evaluation mode.
        """
        self.training = training

    def forward(self, X) -> np.array:
        """
        Calculates the forward pass for a Dropout layer.
        
        Args:
            X (numpy array): Input for the layer (assuming activation output).
        
        Returns:
            numpy array: Output of the Dropout layer.
        """
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p) # inverted dropout
            return X * self.mask
        else:
            return X
    
    def backward(self, grad) -> np.array:
        if self.training:
            assert self.mask is not None, "The mask has to be saved in cache"
            return grad * self.mask
        else:
            return grad

class Patchify(Node):
    """
    The Patchify layer extracts non-overlapping patches from the input image using convolution filters.
    """
    def __init__(self, f: int, nf: int):
        """
        Initializes the Patchify layer.
        
        Args:
            f (int): The size of the filter (patch) to be extracted from the input image.
            nf (int): The number of filters.
        """
        self.f = f
        self.nf = nf
        self.F = np.random.randn(f*f*3, nf) * np.sqrt(2/(f*f*3)) # He initialization for convolution filters, shape (f*f*3, nf)
        self.b = np.zeros((1, self.nf, 1)) # bias for each filter (1 for broadcasting)
        self.grad_F = np.zeros_like(self.F)
        self.grad_b = np.zeros_like(self.b)
        self.Mx = None # save input for backward pass

    def forward(self, Mx: np.array) -> np.array:
        """
        Calculates the forward pass for the Patchify layer.
        
        Args:
            Mx (numpy array): Input matix Mx of shape (Np, f*f*3, N), where Np is the number of patches, f is the filter size, and N is the batch size.
        
        Returns:
            numpy array: Output of the Patchify layer of shape (Np * Nf, N).
        """
        self.Mx = Mx
        res = np.einsum('ijn, jl -> iln', Mx, self.F, optimize=True) + self.b
        # res += self.b
        # return np.fmax(res.reshape((Mx.shape[0]*self.nf, Mx.shape[2]), order='C'), 0)
        return res.reshape((Mx.shape[0]*self.nf, Mx.shape[2]), order='C')

    def backward(self, grad):
        """
        Calculates the gradient w.r.t. to the filters and bias.

        Args:
            grad (numpy array): Upper gradient of shape (Np * Nf, N).

        Returns:
            None (as this layer is the first layer, there is no gradient to return).
        """
        grad_reshaped = grad.reshape((self.Mx.shape[0], self.nf, grad.shape[1]), order='C')
        self.grad_F += np.einsum('jin, jln -> il', self.Mx, grad_reshaped, optimize=True)
        self.grad_b += np.sum(grad_reshaped, axis=(0, 2), keepdims=True)
        return None

    def update_params(self, lr: np.float64):
        """
        Updates the layer parameters and resets gradient.

        Args:
            lr (np.float64): The learning rate.
        """
        self.F -= lr * self.grad_F
        self.b -= lr * self.grad_b
        self.grad_F = np.zeros_like(self.F)
        self.grad_b = np.zeros_like(self.b)
        self.Mx = None # clear cache
    

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
        shifted_logits = logits - np.max(logits, axis=0, keepdims=True) # for numerical stability
        self.P = np.exp(shifted_logits)
        reg = np.sum(self.P,axis = 0, keepdims=True)
        self.P /= reg
        # add small constant to prevent log(0)
        eps = 1e-15
        loss = -np.sum(Y * np.log(self.P + eps)) / logits.shape[1]
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


class Model:
    def __init__(self, f: int, nf: int, d_hidden: int, K: int, p : float = 0.0):
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
    

class Optimizer:
    def __init__(self, model: Model, loss_fn: CrossEntropyLoss | KBinaryCELoss, lr: np.float64, reg: np.float64, label_smoothing: float = 0.0):
        """
        Initializes the optimizer.

        Args:
            model (Model): The model to optimize.
            loss_fn (CrossEntropyLoss | KBinaryCELoss): The loss function to optimize.
            lr (np.float64): The learning rate.
            reg (np.float64): The regularization parameter (lambda).
            label_smoothing (float): The probability of label smoothing to distribute over the non-true classes (only applicable for CrossEntropyLoss). Defaults to 0.0 (no label smoothing).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.reg = reg
        self.label_smoothing = label_smoothing
        # variables for tracking training progress
        self.train_cost_history = []
        self.val_cost_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.plot_update_value = [] # to properly plot the history when using cyclical learning rate
        self.lr_history = []

    def predict(self, Mx: np.ndarray) -> np.ndarray:
        """
        Predicts the class label probabilities for the input data.
        
        Args:
            Mx (numpy array): Precomputed Mx matrix for the input batch of shape (Np, 3f^2, N), where N is the batch size.

        Returns:
            numpy.array: Predicted class probabilities of shape (K, N) where K is the number of classes.
        """
        logits = self.model.forward(Mx)
        probs = None
        if isinstance(self.loss_fn, CrossEntropyLoss):
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=0, keepdims=True)
        elif isinstance(self.loss_fn, KBinaryCELoss):
            probs = 1 / (1 + np.exp(-logits))
        return probs
    
    def compute_loss(self, Mx: np.ndarray, Y: np.ndarray) -> np.float64:
        """
        Computes the loss for the given input.

        Args: 
            Mx (numpy array): Precomputed Mx matrix for the input batch of shape (Np, 3f^2, N), where N is the batch size.
            Y (numpy array): True labels of shape (K, N) where K is the number of classes and N is the batch size.

        Returns:
            numpy.float64: the computed loss
        """
        logits = self.model.forward(Mx)
        loss = self.loss_fn.forward(logits, Y)
        # add L2 regularization:
        for layer in self.model.layers:
            if isinstance(layer, LinearLayer):
                loss += self.reg * np.sum(layer.W ** 2)
            if isinstance(layer, Patchify):
                loss += self.reg * np.sum(layer.F ** 2)
        return loss
    
    def compute_accuracy(self, Mx: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Computes the accuracy for the given input.

        Args: 
            Mx (numpy array): Precomputed Mx matrix for the input batch of shape (Np, 3f^2, N), where N is the batch size.
            y (numpy array): True labels of shape (N,) where N is the batch size.

        Returns:
            numpy.float64: the computed accuracy
        """
        preds = np.argmax(self.predict(Mx), axis=0)
        return np.mean(preds == y)

    def step(self, Mx: np.ndarray, Y: np.ndarray) -> None:
        """
        Applies one optimization step.

        Applies one step of the mini-batch gradient descent algorithm, i.e. compute the forward pass
        to obtain the loss, compute the backward pass to obtain the gradients, and update the model parameters accordingly.

        Args:
            Mx (numpy array): Precomputed Mx matrix for the input batch of shape (Np, 3f^2, N), where N is the batch size.
            Y (numpy array): True labels of shape (K, N) where K is the number of classes and N is the batch size.
        """      
        self.set_train_mode()
        # forward pass
        loss = self.compute_loss(Mx, Y)

        # backward pass
        grad_loss = self.loss_fn.backward()
        self.model.backward(grad_loss)

        # add L2 regularization gradient
        for layer in self.model.layers:
            if isinstance(layer, LinearLayer):
                layer.grad_W += 2 * self.reg * layer.W
            if isinstance(layer, Patchify):
                layer.grad_F += 2 * self.reg * layer.F
        # update parameters
        self.model.update_params(self.lr)

    def train_with_cyclical_lr(self, Mx_train: np.ndarray, y_train: np.ndarray, Mx_val: np.ndarray, y_val: np.ndarray, lr_min: float, lr_max: float, step_size: int, n_cycles: int, batch_size: int = 100, print_every: int = 0) -> None:
        """
        Trains the model using a cyclical learning rate scheduler.
        
        Args:
            Mx_train (numpy array): Precomputed Mx matrix for the training data of shape (Np, 3f^2, N_train).
            y_train (numpy array): Training labels of shape (N_train,), where N_train is the number of training samples.
            Mx_val (numpy array): Precomputed Mx matrix for the validation data of shape (Np, 3f^2, N_val).
            y_val (numpy array): Validation labels of shape (N_val,), where N_val is the number of validation samples.
            lr_min (float): The minimum learning rate for the cyclical schedule.
            lr_max (float): The maximum learning rate for the cyclical schedule.
            step_size (int): The number of update steps in half a cycle (i.e. the number of steps to take from lr_min to lr_max and vice versa).
            n_cycles (int): The number of cycles in the training process.
            batch_size (int, optional): The size of each mini-batch. Defaults to 100.
            print_every (int, optional): If greater than 0, prints training progress every print_every update steps. Defaults to 0 (no printing).
        """
        # transform y to one hot encoding
        N_train = y_train.shape[0]
        N_val = y_val.shape[0]
        K = 10
        Y_train = np.zeros((K, N_train))
        if self.label_smoothing > 0:
            assert isinstance(self.loss_fn, CrossEntropyLoss), "Label smoothing is only applicable for CrossEntropyLoss."
            Y_train += self.label_smoothing / (K - 1)
        Y_train[y_train, np.arange(N_train)] = 1 - self.label_smoothing
        Y_val = np.zeros((K, N_val))
        Y_val[y_val, np.arange(N_val)] = 1
        N_batches = N_train // batch_size # number of batches per epoch
        steps_taken = 0
        for cycle in range(n_cycles):
            for step in range(2 * step_size):
                # compute learning rate
                cycle_progress = (step % (2 * step_size)) / (2 * step_size)
                if cycle_progress < 0.5:
                    lr = lr_min + 2 * cycle_progress * (lr_max - lr_min)
                else:
                    lr = lr_max - 2 * (cycle_progress - 0.5) * (lr_max - lr_min)
                self.lr = lr
                self.lr_history.append(lr)
                # split to batches after end of epoch
                idx = steps_taken % N_batches
                if idx == 0:
                    perm = np.random.permutation(N_train)
                    Mx_train_shuffled = Mx_train[:, :, perm]
                    Y_train_shuffled = Y_train[:, perm]
                Mx_batch = Mx_train_shuffled[:, :, idx*batch_size:idx*batch_size+batch_size]
                Y_batch = Y_train_shuffled[:, idx*batch_size:idx*batch_size+batch_size]
                self.step(Mx_batch, Y_batch)
                # compute training and validation loss and accuracy for tracking
                if ((steps_taken + 1) % (step_size) == 0 or steps_taken == 0):
                    self.set_eval_mode()
                    train_loss = self.compute_loss(Mx_train, Y_train)
                    acc_loss = self.compute_loss(Mx_val, Y_val)
                    reg_cost = self.reg * sum(np.sum(layer.W ** 2) for layer in self.model.layers if isinstance(layer, LinearLayer))
                    reg_cost += self.reg * sum(np.sum(layer.F ** 2) for layer in self.model.layers if isinstance(layer, Patchify))
                    self.train_cost_history.append(train_loss)
                    self.val_cost_history.append(acc_loss)
                    self.train_loss_history.append(train_loss - reg_cost)
                    self.val_loss_history.append(acc_loss - reg_cost)
                    self.train_acc_history.append(self.compute_accuracy(Mx_train, y_train))
                    self.val_acc_history.append(self.compute_accuracy(Mx_val, y_val))
                    self.plot_update_value.append(steps_taken + 1)
                # print training progress
                if print_every > 0 and (steps_taken + 1) % print_every == 0:
                    print(f'Update step {steps_taken + 1} - Train Loss: {self.train_loss_history[-1]:.4f}, Val Loss: {self.val_loss_history[-1]:.4f}, Train Acc: {self.train_acc_history[-1]:.4f}, Val Acc: {self.val_acc_history[-1]:.4f}, LR: {self.lr:.6f}')
                steps_taken += 1
            # increase step size for next cycle
            step_size *= 2

    def plot_cyclical_lr_training_progress(self) -> None:
        """
        Plots the training and validation loss and accuracy curves for cyclical learning rate training.
        """
        steps = np.array(self.plot_update_value)
        plt.figure(figsize=(6, 5))
        plt.plot(steps, self.train_loss_history, label='Train Loss')
        plt.plot(steps, self.val_loss_history, label='Val Loss')
        plt.xlabel('Update step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(6, 5))
        plt.plot(steps, self.train_cost_history, label='Train Cost (including regularization)')
        plt.plot(steps, self.val_cost_history, label='Val Cost (including regularization)')
        plt.xlabel('Update step')
        plt.ylabel('Cost')
        plt.title('Training and Validation Cost')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 5))
        plt.plot(steps, self.train_acc_history, label='Train Accuracy')
        plt.plot(steps, self.val_acc_history, label='Val Accuracy')
        plt.xlabel('Update step')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.grid()
        plt.ylim(0,1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_learning_rate_history(self) -> None:
        """
        Plots the learning rate history (useful for cyclical learning rate).
        """
        steps = np.arange(1, len(self.lr_history) + 1)
        plt.figure(figsize=(6, 5))
        plt.plot(steps, self.lr_history)
        plt.xlabel('Update Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate History')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def flip_vertically(self, X: np.ndarray) -> np.ndarray:
        """
        Flips the input images vertically.

        Args: 
            X (numpy array): Input batch of shape (D, N), where N is the number of samples, and D is the flattened image dimensionalitu.
        
        Returns:
            numpy array: Vertically flipped images of shape (D, N).
        """
        # Reshape X to (32, 32, 3, N) to represent the images in their original shape
        N = X.shape[1]
        X_reshaped = X.reshape((32, 32, 3, N), order='F')
        
        # Flip the images vertically by reversing the order of the rows
        X_flipped = X_reshaped[::-1, :, :, :]
        
        # Reshape back to (D, N)
        X_flipped = X_flipped.reshape((32*32*3, N), order='F')
        
        return X_flipped
    
    def set_train_mode(self):
        """
        Sets the model to training mode.
        """
        self.model.set_train_mode(True)

    def set_eval_mode(self):
        """
        Sets the model to evaluation mode.
        """
        self.model.set_train_mode(False)


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: np.array) -> None:
        """
        Calculates the mean and stardard deviation from the supplied dataset.

        Args:
            X (np.array): dataset matrix of shape (D, N) where N is the number of samples, and D is the dimensionality.
        """
        self.mean = np.mean(X, axis = 1, keepdims=True)
        self.std = np.std(X, axis = 1, keepdims=True)

    def transform(self, X: np.array) -> np.array:
        """
        Transforms supplied dataset by centering and scaling using precomputed mean and standard deviation.

        Args:
            X (np.array): dataset matrix of shape (D, N) where N is the number of samples, and D is the dimensionality.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet!")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.array) -> np.array:
        """
        Fits the scaler to the supplied dataset and then transforms it.

        Args:
            X (np.array): dataset matrix of shape (D, N) where N is the number of samples, and D is the dimensionality.
        """
        self.fit(X)
        return self.transform(X)

def excercise1():
    debug_file = 'debug_info.npz'
    load_data = np.load(debug_file)
    X = load_data['X']
    print("X: ", X.shape)
    X_ims = np.transpose(X.reshape((32, 32, 3, 5), order='F'), (1, 0, 2, 3))
    print("X_ims: ", X_ims.shape)
    Fs = load_data['Fs']
    print("Fs: ", Fs.shape)

    # the basic approach -> for loops
    N = X_ims.shape[3]
    f = Fs.shape[0]
    Nf = Fs.shape[3]
    my_output = np.zeros((32//f, 32//f, Nf, N))
    print("my_output: ", my_output.shape)
    for n in range(N): # for each image
        for j in range(32//f):
            for i in range(32//f):
                patch = X_ims[i*f:(i+1)*f, j*f:(j+1)*f, :, n]
                for k in range(Nf): # for each filter
                    my_output[i, j, k, n] = np.sum(patch * Fs[:, :, :, k])
    outputs = load_data['conv_outputs']
    print("Difference in outputs: ", np.mean(np.abs(outputs - my_output)))
    # creating Mx
    Np = (32//f)**2
    Mx = np.zeros((Np, f*f*3, N))
    for n in range(N):
        region = 0
        for i in range(32//f):
            for j in range(32//f):
                patch = X_ims[i*f:(i+1)*f, j*f:(j+1)*f, :, n]
                Mx[region, :, n] = patch.reshape((1, f*f*3), order='C')
                region += 1
    Mx_gt = load_data['MX']
    print("Difference in MX: ", np.mean(np.abs(Mx - Mx_gt)))
    Fs_flat = Fs.reshape((f*f*3, Nf), order='C')
    # compute the convolutions
    outputs_mat = np.einsum('ijn, jl -> iln', Mx, Fs_flat, optimize=True)
    outputs_flat = outputs.reshape((Np, Nf, N), order='C')
    print("Difference in outputs (matrix): ", np.mean(np.abs(outputs_flat - outputs_mat)))


def excercise2():
    # number of hidden layer is 10
    model = Model(4, 2, 10, 10)
    debug_file = 'debug_info.npz'
    load_data = np.load(debug_file)
    for key in load_data.files:
        print(f"{key}: {load_data[key].shape}")
    # initialize the layers for testing:
    model.layers[0].F = load_data['Fs'].reshape((4*4*3, 2), order = 'C')
    model.layers[2].W = load_data['W1']
    model.layers[2].b = load_data['b1']
    model.layers[5].W = load_data['W2']
    model.layers[5].b = load_data['b2']
    # forward pass
    X = load_data['X']
    X_ims = np.transpose(X.reshape((32, 32, 3, 5), order='F'), (1, 0, 2, 3))
    N = X_ims.shape[3]
    f = 4
    Nf = 2
    Np = (32//f)**2
    Mx = np.zeros((Np, f*f*3, N))
    for n in range(N):
        region = 0
        for i in range(32//f):
            for j in range(32//f):
                patch = X_ims[i*f:(i+1)*f, j*f:(j+1)*f, :, n]
                Mx[region, :, n] = patch.reshape((1, f*f*3), order='C')
                region += 1
    output = model.forward(Mx)
    relu = ReLU()
    my_conv_flat = model.layers[2].X
    conv_outputs_mat = load_data['conv_outputs_mat'].reshape((Np*Nf, N), order='C')
    conv_flat = np.fmax(conv_outputs_mat, 0)
    print("Difference in conv outputs: ", np.mean(np.abs(my_conv_flat - conv_flat)))
    my_X1 = model.layers[5].X
    gt_x1 = load_data['X1']
    print("Difference in X1: ", np.mean(np.abs(my_X1 - gt_x1)))
    # check softmax output
    ce_loss = CrossEntropyLoss()
    loss = ce_loss.forward(output, load_data['Y'])
    my_softmax = ce_loss.P
    gt_softmax = load_data['P']
    print("Difference in softmax output: ", np.mean(np.abs(my_softmax - gt_softmax)))
    # check backward pass
    grad = ce_loss.backward()
    model.backward(grad)
    my_grad_F = model.layers[0].grad_F
    gt_grad_F = load_data['grad_Fs_flat']
    print("Difference in grad_F: ", np.mean(np.abs(my_grad_F - gt_grad_F)))
    my_grad_W1 = model.layers[2].grad_W
    gt_grad_W1 = load_data['grad_W1']
    print("Difference in grad_W1: ", np.mean(np.abs(my_grad_W1 - gt_grad_W1)))
    my_grad_b1 = model.layers[2].grad_b
    gt_grad_b1 = load_data['grad_b1']
    print("Difference in grad_b1: ", np.mean(np.abs(my_grad_b1 - gt_grad_b1)))
    my_grad_W2 = model.layers[5].grad_W
    gt_grad_W2 = load_data['grad_W2']
    print("Difference in grad_W2: ", np.mean(np.abs(my_grad_W2 - gt_grad_W2)))
    my_grad_b2 = model.layers[5].grad_b
    gt_grad_b2 = load_data['grad_b2']
    print("Difference in grad_b2: ", np.mean(np.abs(my_grad_b2 - gt_grad_b2)))

def excercise2_precompute_Mx():
    X, _, _ = load_batch('data_batch_1')
    f = 4
    Mx = precompute_Mx(X, f)
    print("Mx shape: ", Mx.shape)

def test_grads_with_torch():
    # create the model
    model = Model(4, 2, 10, 10)
    # load the data
    X, Y, y = load_batch('data_batch_1')
    X = X[:, :5] # use only 5 samples for testing 
    Y = Y[:, :5]
    y = y[:5]
    # calculate the gradients with torch:
    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)
    # patchify the input
    f = 4
    N = X.shape[1]
    Np = (32//f)**2
    Mx = precompute_Mx(X, f)
    Mx = np.asarray(Mx, dtype=np.float32)
    Mx_torch = torch.from_numpy(Mx)
    F = torch.tensor(model.layers[0].F,dtype=torch.float32, requires_grad=True)
    b = torch.tensor(model.layers[0].b, dtype=torch.float32, requires_grad=True)
    W1 = torch.tensor(model.layers[2].W, dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor(model.layers[2].b, dtype=torch.float32, requires_grad=True)
    W2 = torch.tensor(model.layers[5].W, dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor(model.layers[5].b, dtype=torch.float32, requires_grad=True)

    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)
    # create the forward pass with torch
    tmp = torch.zeros((Np, 2, N))
    for n in range(N):
        tmp[:, :, n] = torch.matmul(Mx_torch[:,:,n], F) + b[:,:,0]
    # reshape
    tmp_flat = tmp.view(Np*2, N)
    # apply relu
    tmp_flat = apply_relu(tmp_flat)
    # apply first linear layer
    tmp_linear1 = apply_relu(torch.matmul(W1, tmp_flat) + b1)
    # apply second linear layer
    logits = torch.matmul(W2, tmp_linear1) + b2
    # apply softmax
    P = apply_softmax(logits)
    # compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    
    print(f"Loss computed with PyTorch: {loss.item():.12f}")
    # compute the backward pass
    loss.backward()
    # compute my gradients
    logits = model.forward(Mx)
    ce_loss = CrossEntropyLoss()
    loss = ce_loss.forward(logits, Y)
    print(f"Loss computed with my implementation: {loss:.12f}")
    grad = ce_loss.backward()
    model.backward(grad)
    # compare the gradients
    # w2
    torch_grad_W2 = W2.grad.numpy()
    my_grad_W2 = model.layers[5].grad_W
    print("Difference in grad_W2: ", calculate_mean_grad_difference(torch_grad_W2, my_grad_W2))
    # b2
    torch_grad_b2 = b2.grad.numpy()
    my_grad_b2 = model.layers[5].grad_b
    print("Difference in grad_b2: ", calculate_mean_grad_difference(torch_grad_b2, my_grad_b2))
    # w1
    torch_grad_W1 = W1.grad.numpy()
    my_grad_W1 = model.layers[2].grad_W
    print("Difference in grad_W1: ", calculate_mean_grad_difference(torch_grad_W1, my_grad_W1))
    # b1
    torch_grad_b1 = b1.grad.numpy()
    my_grad_b1 = model.layers[2].grad_b
    print("Difference in grad_b1: ", calculate_mean_grad_difference(torch_grad_b1, my_grad_b1))
    # F
    torch_grad_F = F.grad.numpy()
    my_grad_F = model.layers[0].grad_F
    print("Difference in grad_F: ", calculate_mean_grad_difference(torch_grad_F, my_grad_F))
    # patchify bias
    torch_grad_b = b.grad.numpy()
    my_grad_b = model.layers[0].grad_b
    print("Difference in grad_b: ", calculate_mean_grad_difference(torch_grad_b, my_grad_b))

def excercise3():
    f = 4
    nf = 10
    nh = 50
    lam = 0.003
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=lam)
    scaler = Scaler()
    X, Y, y = load_training_batches()
    # split into training and validation set
    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:,49000:], y[49000:]
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    # transform X into Mx for the initial Patchify layer
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    # train the model
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=100)
    X_test, Y_test, y_test = load_batch('test_batch')
    X_test = scaler.transform(X_test)
    Mx_test = precompute_Mx(X_test, f)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()

def excercise3_architectures():
    # data
    # split into training and validation set
    scaler = Scaler()
    X, Y, y = load_training_batches()
    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:,49000:], y[49000:]
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test, Y_test, y_test = load_batch('test_batch')
    X_test = scaler.transform(X_test)
    # for plotting
    test_accuracies = []
    val_accuracies = []
    runtime = []
    # architecture 1
    f = 2
    nf = 3
    nh = 50
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.003)
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    time.sleep(1) # sleep for a second to avoid any issues with time measurement
    start_time = time.time()
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=0)
    end_time = time.time()
    runtime.append(end_time - start_time)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    test_accuracies.append(test_acc)
    val_accuracies.append(val_acc)
    print(f"Architecture 1 - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    # architecture 2
    f = 4
    nf = 10
    nh = 50
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.003)
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    start_time = time.time()
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=0)
    end_time = time.time()
    runtime.append(end_time - start_time)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    test_accuracies.append(test_acc)
    val_accuracies.append(val_acc)
    print(f"Architecture 2 - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    # architecture 3
    f = 8
    nf = 40
    nh = 50
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.003)
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    start_time = time.time()
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=0)
    end_time = time.time()
    runtime.append(end_time - start_time)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    test_accuracies.append(test_acc)
    val_accuracies.append(val_acc)
    print(f"Architecture 3 - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    # architecture 4
    f = 16
    nf = 160
    nh = 50
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.003)
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    start_time = time.time()
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=0)
    end_time = time.time()
    runtime.append(end_time - start_time)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    test_accuracies.append(test_acc)
    val_accuracies.append(val_acc)
    print(f"Architecture 4 - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    # create a bar chart of the accuracies
    architectures = ['f=2, nf=3', 'f=4, nf=10', 'f=8, nf=40', 'f=16, nf=160']
    x = np.arange(len(architectures))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, val_accuracies, width, label='Validation Accuracy')
    rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.legend()
    plt.ylim(0, 1)
    plt.show()
    # create a bar chart of the runtimes
    fig, ax = plt.subplots()
    rects = ax.bar(architectures, runtime)
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime by architecture')
    plt.show()

def excercise3_train_for_longer():
    scaler = Scaler()
    X, Y, y = load_training_batches()
    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:,49000:], y[49000:]
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test, Y_test, y_test = load_batch('test_batch')
    X_test = scaler.transform(X_test)
    # architecture 2
    f = 4
    nf = 10
    nh = 50
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.003)
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=0)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    print(f"Architecture 2 (longer training) - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()
    # architecture 3
    f = 8
    nf = 40
    nh = 50
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.003)
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=0)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    print(f"Architecture 3 (longer training) - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()
    # architecture 2 with nf = 40
    f = 4
    nf = 40
    nh = 50
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.003)
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=0)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    print(f"Architecture 2 (longer training) - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()

def excercise4():
    # data
    scaler = Scaler()
    X, Y, y = load_training_batches()
    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:,49000:], y[49000:]
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test, Y_test, y_test = load_batch('test_batch')
    X_test = scaler.transform(X_test)
    # architecture 5 params
    f = 4
    nf = 40
    nh = 300
    lam = 0.0025
    # precompute Mx for all sets
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    # without label smoothing
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=lam)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=4, batch_size=100, print_every=0)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    print(f"Architecture 5 (no label smoothing) - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()
    # with label smoothing
    model = Model(f, nf, nh, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=lam, label_smoothing=0.2)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=4, batch_size=100, print_every=0)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    print(f"Architecture 5 (with label smoothing) - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()

def main():
    pass

if __name__ == "__main__":
    # main()
    # excercise1()
    # excercise2()
    # excercise2_precompute_Mx()
    # test_grads_with_torch()
    # excercise3()
    # excercise3_architectures()
    # excercise3_train_for_longer()
    excercise4()
