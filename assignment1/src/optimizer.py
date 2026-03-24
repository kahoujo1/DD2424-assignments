"""
Optimizer class for training.
"""

import numpy as np
import matplotlib.pyplot as plt
from model import Model
from nodes import CrossEntropyLoss, LinearLayer

class Optimizer:
    def __init__(self, model: Model, loss_fn: CrossEntropyLoss, lr: np.float64, reg: np.float64, vertical_flip_prob: float = 0):
        """
        Initializes the optimizer.

        Args:
            model (Model): The model to optimize.
            loss_fn (CrossEntropyLoss): The loss function to optimize.
            lr (np.float64): The learning rate.
            reg (np.float64): The regularization parameter (lambda).
            vertical_flip_prob (float, optional): The probability of applying vertical flipping as a data augmentation technique. Defaults to 0 (no flipping).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.reg = reg
        self.vertical_flip_prob = vertical_flip_prob
        # variables for tracking training progress
        self.train_cost_history = []
        self.val_cost_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

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
        preds = np.argmax(self.model.predict(X), axis=0)
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
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, num_epochs: int, batch_size: int = 16, print_every: int = 0) -> None:
        """
        Trains the model.

        Args:
            X_train (numpy array): Training data of shape (D, N_train), where N_train is the number of training samples and D is the input dimensionality.
            y_train (numpy array): Training labels of shape (N_train,), where N_train is the number of training samples.
            X_val (numpy array): Validation data of shape (D, N_val), where N_val is the number of validation samples and D is the input dimensionality.
            y_val (numpy array): Validation labels of shape (N_val,), where N_val is the number of validation samples.
            num_epochs (int): The number of epochs to train for.
            batch_size (int, optional): The size of each mini-batch. Defaults to 16.
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
            # mini-batch training
            for i in range(0, N_train, batch_size):
                X_batch = X_train_shuffled[:, i:i+batch_size]
                Y_batch = Y_train_shuffled[:, i:i+batch_size]
                self.step(X_batch, Y_batch)
            # compute training and validation loss and accuracy for tracking
            self.train_cost_history.append(self.compute_loss(X_train, Y_train))
            self.val_cost_history.append(self.compute_loss(X_val, Y_val))
            self.train_loss_history.append(self.compute_loss(X_train, Y_train) - self.reg * sum(np.sum(layer.W ** 2) for layer in self.model.layers if isinstance(layer, LinearLayer)))
            self.val_loss_history.append(self.compute_loss(X_val, Y_val) - self.reg * sum(np.sum(layer.W ** 2) for layer in self.model.layers if isinstance(layer, LinearLayer)))
            self.train_acc_history.append(self.compute_accuracy(X_train, y_train))
            self.val_acc_history.append(self.compute_accuracy(X_val, y_val))
            # print training progress
            if print_every > 0 and (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {self.train_loss_history[-1]:.4f}, Val Loss: {self.val_loss_history[-1]:.4f}, Train Acc: {self.train_acc_history[-1]:.4f}, Val Acc: {self.val_acc_history[-1]:.4f}')

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