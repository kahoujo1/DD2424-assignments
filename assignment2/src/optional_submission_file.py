import numpy as np
import copy
import matplotlib.pyplot as plt
from torch_gradient_computations import ComputeGradsWithTorch
import pickle
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
    
class Dropout(Node):
    def __init__(self, p: np.float64):
        """
        Initializes the Dropout layer.

        Args:
            p (numpy.float64): The dropout probability.
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
        assert self.mask is not None, "The mask has to be saved in cache"
        if self.training:
            return grad * self.mask
        else:
            return grad

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

class Model:
    def __init__(self, d_in: int, d_hidden: int, K: int, p : float = 0.0):
        self.layers = [LinearLayer(d_in, d_hidden),
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

    def set_train_mode(self, mode: bool) -> None:
        """
        Sets the model to training or evaluation mode.
        
        Args:
            mode (bool): If True, set to training mode. If False, set to evaluation mode.
        """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.set_training(mode)
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
class Optimizer:
    def __init__(self, model: Model, loss_fn: CrossEntropyLoss | KBinaryCELoss, lr: np.float64, reg: np.float64, vertical_flip_prob: float = 0, do_batch_translation: bool = False):
        """
        Initializes the optimizer.

        Args:
            model (Model): The model to optimize.
            loss_fn (CrossEntropyLoss | KBinaryCELoss): The loss function to optimize.
            lr (np.float64): The learning rate.
            reg (np.float64): The regularization parameter (lambda).
            vertical_flip_prob (float, optional): The probability of applying vertical flipping as a data augmentation technique. Defaults to 0 (no flipping).
            do_batch_translation (bool, optional): Whether to apply random translation to the batch as a data augmentation technique. Defaults to False (no translation).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.reg = reg
        self.vertical_flip_prob = vertical_flip_prob
        self.do_batch_translation = do_batch_translation
        # variables for tracking training progress
        self.train_cost_history = []
        self.val_cost_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.plot_update_value = [] # to properly plot the history when using cyclical learning rate
        self.lr_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class label probabilities for the input data.
        
        Args:
            X (numpy array): Input data of shape (D, N) where N is the batch size and D is the dimensionality.

        Returns:
            numpy.array: Predicted class probabilities of shape (K, N) where K is the number of classes.
        """
        logits = self.model.forward(X)
        probs = None
        if isinstance(self.loss_fn, CrossEntropyLoss):
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=0, keepdims=True)
        elif isinstance(self.loss_fn, KBinaryCELoss):
            probs = 1 / (1 + np.exp(-logits))
        return probs
    
    def compute_loss(self, X: np.ndarray, Y: np.ndarray) -> np.float64:
        """
        Computes the loss for the given input.

        Args: 
            X (numpy array): Input batch of shape (D, N) where N is the batch size and D is the dimensionality.
            Y (numpy array): True labels of shape (K, N) where K is the number of classes and N is the batch size.

        Returns:
            numpy.float64: the computed loss
        """
        logits = self.model.forward(X)
        loss = self.loss_fn.forward(logits, Y)
        # add L2 regularization:
        for layer in self.model.layers:
            if isinstance(layer, LinearLayer):
                loss += self.reg * np.sum(layer.W ** 2)
        return loss
    
    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Computes the accuracy for the given input.

        Args: 
            X (numpy array): Input batch of shape (D, N) where N is the batch size and D is the input dimensionality.
            y (numpy array): True labels of shape (N,) where N is the batch size.

        Returns:
            numpy.float64: the computed accuracy
        """
        preds = np.argmax(self.predict(X), axis=0)
        return np.mean(preds == y)

    def step(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Applies one optimization step.

        Applies one step of the mini-batch gradient descent algorithm, i.e. compute the forward pass
        to obtain the loss, compute the backward pass to obtain the gradients, and update the model parameters accordingly.

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

        # update parameters
        self.model.update_params(self.lr)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, num_epochs: int, batch_size: int = 100, print_every: int = 0) -> None:
        """
        Trains the model.

        Args:
            X_train (numpy array): Training data of shape (D, N_train), where N_train is the number of training samples and D is the input dimensionality.
            y_train (numpy array): Training labels of shape (N_train,), where N_train is the number of training samples.
            X_val (numpy array): Validation data of shape (D, N_val), where N_val is the number of validation samples and D is the input dimensionality.
            y_val (numpy array): Validation labels of shape (N_val,), where N_val is the number of validation samples.
            num_epochs (int): The number of epochs to train for.
            batch_size (int, optional): The size of each mini-batch. Defaults to 100.
            print_every (int, optional): If greater than 0, prints training progress every print_every epochs. Defaults to 0 (no printing).
        """
        N_train = X_train.shape[1]
        N_val = X_val.shape[1]
        K = 10
        Y_train = np.zeros((K, N_train))
        Y_train[y_train, np.arange(N_train)] = 1
        Y_val = np.zeros((K, N_val))
        Y_val[y_val, np.arange(N_val)] = 1

        for epoch in range(num_epochs):
            # shuffle training data
            perm = np.random.permutation(N_train)
            X_train_shuffled = X_train[:, perm]
            Y_train_shuffled = Y_train[:, perm]
            # flip each image in the batch with the specified probability
            if self.vertical_flip_prob > 0:
                flip_mask = np.random.rand(N_train) < self.vertical_flip_prob
                X_train_shuffled[:, flip_mask] = self.flip_vertically(X_train_shuffled[:, flip_mask])
            if self.do_batch_translation:
                X_train_shuffled = self.translate_batch(X_train_shuffled)
            # mini-batch training
            for i in range(0, N_train, batch_size):
                X_batch = X_train_shuffled[:, i:i+batch_size]
                Y_batch = Y_train_shuffled[:, i:i+batch_size]
                self.step(X_batch, Y_batch)
            # compute training and validation loss and accuracy for tracking
            self.set_eval_mode()
            self.train_cost_history.append(self.compute_loss(X_train, Y_train))
            self.val_cost_history.append(self.compute_loss(X_val, Y_val))
            self.train_loss_history.append(self.compute_loss(X_train, Y_train) - self.reg * sum(np.sum(layer.W ** 2) for layer in self.model.layers if isinstance(layer, LinearLayer)))
            self.val_loss_history.append(self.compute_loss(X_val, Y_val) - self.reg * sum(np.sum(layer.W ** 2) for layer in self.model.layers if isinstance(layer, LinearLayer)))
            self.train_acc_history.append(self.compute_accuracy(X_train, y_train))
            self.val_acc_history.append(self.compute_accuracy(X_val, y_val))
            # print training progress
            if print_every > 0 and (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {self.train_loss_history[-1]:.4f}, Val Loss: {self.val_loss_history[-1]:.4f}, Train Acc: {self.train_acc_history[-1]:.4f}, Val Acc: {self.val_acc_history[-1]:.4f}')

    def train_with_cyclical_lr(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, lr_min: float, lr_max: float, step_size: int, n_cycles: int, batch_size: int = 100, print_every: int = 0) -> None:
        """
        Trains the model using a cyclical learning rate scheduler.
        
        Args:
            X_train (numpy array): Training data of shape (D, N_train), where N_train is the number of training samples and D is the input dimensionality.
            y_train (numpy array): Training labels of shape (N_train,), where N_train is the number of training samples.
            X_val (numpy array): Validation data of shape (D, N_val), where N_val is the number of validation samples and D is the input dimensionality.
            y_val (numpy array): Validation labels of shape (N_val,), where N_val is the number of validation samples.
            lr_min (float): The minimum learning rate for the cyclical schedule.
            lr_max (float): The maximum learning rate for the cyclical schedule.
            step_size (int): The number of update steps in half a cycle (i.e. the number of steps to take from lr_min to lr_max and vice versa).
            n_cycles (int): The number of cycles in the training process.
            batch_size (int, optional): The size of each mini-batch. Defaults to 100.
            print_every (int, optional): If greater than 0, prints training progress every print_every update steps. Defaults to 0 (no printing).
        """
        # transform y to one hot encoding
        N_train = X_train.shape[1]
        N_val = X_val.shape[1]
        K = 10
        Y_train = np.zeros((K, N_train))
        Y_train[y_train, np.arange(N_train)] = 1
        Y_val = np.zeros((K, N_val))
        Y_val[y_val, np.arange(N_val)] = 1
        N_batches = N_train // batch_size # number of batches per epoch
        for step in range(n_cycles * 2 * step_size):
            # compute learning rate
            cycle_progress = (step % (2 * step_size)) / (2 * step_size)
            if cycle_progress < 0.5:
                lr = lr_min + 2 * cycle_progress * (lr_max - lr_min)
            else:
                lr = lr_max - 2 * (cycle_progress - 0.5) * (lr_max - lr_min)
            self.lr = lr
            self.lr_history.append(lr)
            # split to batches after end of epoch
            idx = step % N_batches
            if idx == 0:
                perm = np.random.permutation(N_train)
                X_train_shuffled = X_train[:, perm]
                Y_train_shuffled = Y_train[:, perm]
                # flip each image in the batch with the specified probability
                if self.vertical_flip_prob > 0:
                    flip_mask = np.random.rand(N_train) < self.vertical_flip_prob
                    X_train_shuffled[:, flip_mask] = self.flip_vertically(X_train_shuffled[:, flip_mask])
                if self.do_batch_translation:
                    X_train_shuffled = self.translate_batch(X_train_shuffled)
            X_batch = X_train_shuffled[:, idx*batch_size:idx*batch_size+batch_size]
            Y_batch = Y_train_shuffled[:, idx*batch_size:idx*batch_size+batch_size]
            self.step(X_batch, Y_batch)
            # compute training and validation loss and accuracy for tracking
            if ((step + 1) % 100 == 0 or step == 0):
                self.set_eval_mode()
                self.train_cost_history.append(self.compute_loss(X_train, Y_train))
                self.val_cost_history.append(self.compute_loss(X_val, Y_val))
                self.train_loss_history.append(self.compute_loss(X_train, Y_train) - self.reg * sum(np.sum(layer.W ** 2) for layer in self.model.layers if isinstance(layer, LinearLayer)))
                self.val_loss_history.append(self.compute_loss(X_val, Y_val) - self.reg * sum(np.sum(layer.W ** 2) for layer in self.model.layers if isinstance(layer, LinearLayer)))
                self.train_acc_history.append(self.compute_accuracy(X_train, y_train))
                self.val_acc_history.append(self.compute_accuracy(X_val, y_val))
                self.plot_update_value.append(step) 
            # print training progress
            if print_every > 0 and (step + 1) % print_every == 0:
                print(f'Update step {step + 1} - Train Loss: {self.train_loss_history[-1]:.4f}, Val Loss: {self.val_loss_history[-1]:.4f}, Train Acc: {self.train_acc_history[-1]:.4f}, Val Acc: {self.val_acc_history[-1]:.4f}, LR: {self.lr:.6f}')

                

    def plot_training_progress(self) -> None:
        """
        Plots the training and validation loss and accuracy curves.
        """
        epochs = np.arange(1, len(self.train_loss_history) + 1)
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, self.train_loss_history, label='Train Loss')
        plt.plot(epochs, self.val_loss_history, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, self.train_cost_history, label='Train Cost (including regularization)')
        plt.plot(epochs, self.val_cost_history, label='Val Cost (including regularization)')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Training and Validation Cost')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 5))
        plt.plot(epochs, self.train_acc_history, label='Train Accuracy')
        plt.plot(epochs, self.val_acc_history, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

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
    
    def translate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Translates the input images by a random amount in the range [-3, 3] pixels in both x and y directions.

        Args:
            X (numpy array): Input batch of shape (D, N), where N is the number of samples and D is the dimensionality.

        Returns:
            numpy array: Translated images of shape (D, N).
        """
        N = X.shape[1]
        X_reshaped = X.reshape((32, 32, 3, N), order='F')
        X_translated = np.zeros_like(X_reshaped)
        for i in range(N):
            tx = np.random.randint(-3, 4)
            ty = np.random.randint(-3, 4)
            X_translated[:, :, :, i] = np.roll(X_reshaped[:, :, :, i], shift=(tx, ty), axis=(0, 1))
            # mask out the rolled in pixels with zeros
            if tx > 0:
                X_translated[:tx, :, :, i] = 0
            elif tx < 0:
                X_translated[tx:, :, :, i] = 0
            if ty > 0:
                X_translated[:, :ty, :, i] = 0  
            elif ty < 0:
                X_translated[:, ty:, :, i] = 0
        X_translated = X_translated.reshape((32*32*3, N), order='F')
        return X_translated
    
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, lr_values: np.ndarray, reg_values: np.ndarray, batch_values:np.ndarray) -> None:
        """
        Performs a grid search over the specified parameters, to find the best combination.
        
        Args:
            X_train (numpy array): Training input data.
            y_train (numpy array): Training labels.
            X_val (numpy array): Validation input data.
            y_val (numpy array): Validation labels.
            lr_values (numpy array): Array of learning rate values to search over.
            reg_values (numpy array): Array of regularization parameter values to search over.
            batch_values (numpy array): Array of batch size values to search over.
        """
        best_val_acc = 0.0
        best_params = None
        for lr in lr_values:
            for reg in reg_values:
                for batch_size in batch_values:
                    print(f'Testing lr={lr}, reg={reg}, batch_size={batch_size}')
                    model = copy.deepcopy(self.model)  # create a fresh model for each run
                    optimizer = Optimizer(model, self.loss_fn, lr=lr, reg=reg, vertical_flip_prob=self.vertical_flip_prob)
                    optimizer.train(X_train, y_train, X_val, y_val, num_epochs=150, batch_size=batch_size, print_every=0)
                    val_acc = np.max(optimizer.val_acc_history)
                    epoch = np.argmax(optimizer.val_acc_history)
                    print(f'Max validation accuracy {val_acc:.4f} at epoch {epoch}')
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = (epoch, lr, reg, batch_size)
        print(f"Final best validation accuracy: {best_val_acc}")
        print(f"Best parameters: epoch={best_params[0]}, lr={best_params[1]}, reg={best_params[2]}, batch_size={best_params[3]}")
    
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


def main():
    # Load all data batches
    X_test, Y_test, y_test = load_batch("test_batch")

    X, Y, y = load_batch("data_batch_1")
    for i in range(2,6):
        X_temp, Y_temp, y_temp = load_batch(f"data_batch_{i}")
        X = np.concatenate((X, X_temp), axis=1)
        Y = np.concatenate((Y, Y_temp), axis=1)
        y = np.concatenate((y, y_temp))
    # split into training and validation sets
    print("Shape for combined data:")
    print(X.shape, Y.shape, y.shape)
    X_train = X[:, :49000]
    y_train = y[:49000]
    X_val = X[:, 49000:]
    y_val = y[49000:]
    # scale the data
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    n_batch = 100
    N = X_train.shape[1]
    n_s = int(2 * np.floor(N / n_batch))
    model = Model(32*32*3, 50, 10)
    loss = CrossEntropyLoss()
    optimizer = Optimizer(model, loss, lr=0.1, reg=0.0012915)
    optimizer.train_with_cyclical_lr(X_train, y_train, X_val, y_val, lr_min = 1e-5, lr_max = 1e-1, step_size=n_s, n_cycles=3, batch_size=n_batch, print_every=0)
    print("validation accuracy: ", optimizer.compute_accuracy(X_val, y_val))
    print("test accuracy: ", optimizer.compute_accuracy(X_test, y_test))
    optimizer.plot_cyclical_lr_training_progress()

if __name__ == "__main__":
    main()