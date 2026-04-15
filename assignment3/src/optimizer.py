"""
Optimizer class for training.
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from model import Model
from nodes import CrossEntropyLoss, LinearLayer, KBinaryCELoss, Patchify

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
        Y_train[y_train, np.arange(N_train)] = 1
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
                if ((steps_taken + 1) % (step_size/2) == 0 or steps_taken == 0):
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