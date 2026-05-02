"""
This file serves as the implementation of RNN model for assignment 4.
"""
import numpy as np
from torch_gradient_computations_row_wise import ComputeGradsWithTorch
import matplotlib.pyplot as plt

class Converter:
    def __init__(self, chars):
        self.chars = chars
        self.char_to_index = {char: idx for idx, char in enumerate(chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(chars)}
    
    def char2onehot(self, text):
        """
        Converts a string of characters into a one-hot encoded numpy array.

        Args:
            text (str): The input string to convert.
        
        Returns:
            np.ndarray: A 2D array of shape (len(text), len(chars)) where each row represents the one-hot encoding of the each character.
        """
        onehot = np.zeros((len(text), len(self.chars)), dtype=np.float32)
        for i, char in enumerate(text):
            onehot[i, self.char_to_index[char]] = 1.0
        return onehot
    
    def onehot2char(self, onehot: np.ndarray) -> str:
        """
        Converts a one-hot encoded numpy array back into a string of characters.

        Args:
            onehot (np.ndarray): A 2D array of shape (N, len(chars)) where each row is a one-hot encoding of a character.

        Returns:
            str: The resulting string of characters.
        """
        chars = []
        for row in onehot:
            idx = np.argmax(row)
            chars.append(self.index_to_char[idx])
        return ''.join(chars)

class RNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size # K in the equations
        self.hidden_size = hidden_size # m in the equations
        # note that the inputs will be saved as rows
        self.h = np.zeros((1, self.hidden_size), dtype=np.float32) # initial hidden state
        # Initialize weights and biases
        self.U = (1/np.sqrt(2*self.input_size)) * np.random.randn(self.input_size, self.hidden_size) # Kxm
        self.W = (1/np.sqrt(2*self.hidden_size)) * np.random.randn(self.hidden_size, self.hidden_size) # mxm
        self.b = np.zeros((1, self.hidden_size)) # 1xm
        self.V = (1/np.sqrt(self.hidden_size)) * np.random.randn(self.hidden_size, self.input_size) # mxK
        self.c = np.zeros((1, self.input_size)) # 1xK
        # gradients
        self.grad_U = np.zeros_like(self.U)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.grad_V = np.zeros_like(self.V)
        self.grad_c = np.zeros_like(self.c)
        # attributes needed for backpropagation
        self.hidden_states = None
        self.outputs = None
        self.inputs = None


    def predict_next_n(self, x0 : np.ndarray, n: int) -> np.ndarray:
        """
        Predicts the next n characters given an input x0.
        
        Args:
            x0 (np.ndarray): A one-hot encoded vector of shape (1, input_size) representing the initial character.
            n (int): The number of characters to predict.
        
        Returns:
            np.ndarray: A one-hot encoded array of shape (n, input_size) representing the predicted characters.
        """
        Y = np.zeros((n, self.input_size), dtype=np.float32)
        x = x0
        h = self.h.copy() # use the current hidden state as the starting point
        for t in range(n):
            h = np.tanh(x @ self.U + h @ self.W + self.b)
            o = h @ self.V + self.c # compute output logits
            o = np.exp(o - np.max(o)) # for numerical stability
            o /= np.sum(o) # softmax to get probabilities
            # sample the next character from the output distribution
            y = np.random.choice(self.input_size, p=o.ravel())
            Y[t][y] = 1.0
            x = np.zeros_like(x)
            x[0, y] = 1.0 # set the next input to the predicted character
        return Y
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the RNN for a given input sequence X.

        Args:
            X (np.ndarray): A 2D array of shape (T, input_size) where T is the sequence length and input_size is the size of the one-hot encoding.
        
        Returns:
            np.ndarray: A 2D array of shape (T, input_size) representing the output logits (to utilize softmax backpropagation trick) for each time step.
        """
        T = X.shape[0]
        Ux = X @ self.U # precompute input to hidden transformations
        H = np.zeros((T+1, self.hidden_size), dtype=np.float32) # to store hidden states for backpropagation
        H[-1, :] = self.h # save to the end for backpropagation
        Y = np.zeros((T, self.input_size), dtype=np.float32)
        for t in range(T):
            self.h = np.tanh(Ux[t, :] + self.h @ self.W + self.b)
            H[t, :] = self.h
            o = self.h @ self.V + self.c # compute output logits
            Y[t, :] = o
        # TODO: save the needed values for backpropagation
        self.hidden_states = H
        self.outputs = Y
        self.inputs = X
        return Y
    
    def backward(self, grad_Y: np.ndarray):
        """
        Performs backpropagation through time to compute gradients of the loss with respect to the model parameters.

        Args:
            grad_Y (np.ndarray): A 2D array of shape (T, input_size) representing the gradient of the loss with respect to the output logits for each time step.
        
        Updates:
            self.grad_U, self.grad_W, self.grad_b, self.grad_V, self.grad_c: The computed gradients for the model parameters.
        """
        T = grad_Y.shape[0]
        grad_h_next = np.zeros((1, self.hidden_size), dtype=np.float32) # gradient of the loss with respect to the next hidden state
        for t in reversed(range(T)):
            grad_o = grad_Y[t, :]
            self.grad_V += np.outer(self.hidden_states[t, :], grad_o) # gradient of the loss with respect to V
            self.grad_c += grad_o
            grad_h = grad_o @ self.V.transpose() + grad_h_next # gradient of the loss with respect to the hidden state at time t
            grad_h_raw = (1 - self.hidden_states[t, :] ** 2) * grad_h # backprop through tanh
            self.grad_b += grad_h_raw
            self.grad_W += np.outer(self.hidden_states[t-1, :], grad_h_raw)
            tmp_grad_U = np.zeros_like(self.U)
            tmp_grad_U[self.inputs[t, :] == 1] = grad_h_raw
            self.grad_U += tmp_grad_U
            grad_h_next = grad_h_raw @ self.W.transpose() # update the gradient for the next hidden state
    
    def update_parameters(self, learning_rate):
        """
        Updates the model parameters using the computed gradients and a given learning rate.

        Args:
            learning_rate (float): The learning rate to use for the parameter update.
        """
        self.U -= learning_rate * self.grad_U
        self.W -= learning_rate * self.grad_W
        self.b -= learning_rate * self.grad_b
        self.V -= learning_rate * self.grad_V
        self.c -= learning_rate * self.grad_c
        # reset gradients after update
        self.grad_U = np.zeros_like(self.U)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.grad_V = np.zeros_like(self.V)
        self.grad_c = np.zeros_like(self.c)

class CrossEntropyLoss():
    def __init__(self):
        self.P = None # save softmax probabilities for backward pass
        self.Y = None # save true labels for backward pass

    def forward(self, logits: np.array, Y: np.array) -> np.float64:
        """
        Calculates the forward pass for the cross-entropy loss with softmax.

        Args:
            logits (numpy array): Input logits of shape (N, K) where K is the number of classes and N is the batch size.
            Y (numpy array): True probability distribution of the classes with size (N, K) (most often one hot encoding).

        Returns:
            numpy.float64: The average cross-entropy loss over the batch.
        """
        self.Y = Y
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True) # for numerical stability
        self.P = np.exp(shifted_logits)
        reg = np.sum(self.P,axis = 1, keepdims=True)
        self.P /= reg
        loss = -np.sum(Y * np.log(self.P + 1e-8)) / logits.shape[0]
        return loss
    
    def backward(self) -> np.array:
        """
        Calculates the gradient with respect to the input logits.

        Returns:
            numpy.array: Gradient of the loss with respect to the input logits, of shape (N, K).
        """
        return (self.P - self.Y) / self.P.shape[0]
    
class AdamOptimizer:
    def __init__(self, rnn: RNN, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.rnn = rnn
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Initialize moment estimates for each parameter
        self.m_U = np.zeros_like(rnn.U)
        self.v_U = np.zeros_like(rnn.U)
        self.m_W = np.zeros_like(rnn.W)
        self.v_W = np.zeros_like(rnn.W)
        self.m_b = np.zeros_like(rnn.b)
        self.v_b = np.zeros_like(rnn.b)
        self.m_V = np.zeros_like(rnn.V)
        self.v_V = np.zeros_like(rnn.V)
        self.m_c = np.zeros_like(rnn.c)
        self.v_c = np.zeros_like(rnn.c)
        self.timestep = 0
        # structures for saving progress
        self.best_model = None
        self.best_loss = float('inf')
        self.loss_history = []
    
    def step(self):
        self.timestep += 1
        # Update moment estimates for U
        self.m_U = self.beta1 * self.m_U + (1 - self.beta1) * self.rnn.grad_U
        self.v_U = self.beta2 * self.v_U + (1 - self.beta2) * (self.rnn.grad_U ** 2)
        m_U_hat = self.m_U / (1 - self.beta1 ** self.timestep)
        v_U_hat = self.v_U / (1 - self.beta2 ** self.timestep)
        # Update moment estimates for W
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * self.rnn.grad_W
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (self.rnn.grad_W ** 2)
        m_W_hat = self.m_W / (1 - self.beta1 ** self.timestep)
        v_W_hat = self.v_W / (1 - self.beta2 ** self.timestep)
        # Update moment estimates for b
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * self.rnn.grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (self.rnn.grad_b ** 2)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.timestep)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.timestep)
        # Update moment estimates for V
        self.m_V = self.beta1 * self.m_V + (1 - self.beta1) * self.rnn.grad_V
        self.v_V = self.beta2 * self.v_V + (1 - self.beta2) * (self.rnn.grad_V ** 2)
        m_V_hat = self.m_V / (1 - self.beta1 ** self.timestep)
        v_V_hat = self.v_V / (1 - self.beta2 ** self.timestep)
        # Update moment estimates for c
        self.m_c = self.beta1 * self.m_c + (1 - self.beta1) * self.rnn.grad_c
        self.v_c = self.beta2 * self.v_c + (1 - self.beta2) * (self.rnn.grad_c ** 2)
        m_c_hat = self.m_c / (1 - self.beta1 ** self.timestep)
        v_c_hat = self.v_c / (1 - self.beta2 ** self.timestep)
        # Update parameters
        self.rnn.U -= self.learning_rate * m_U_hat / (np.sqrt(v_U_hat) + self.epsilon)
        self.rnn.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        self.rnn.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        self.rnn.V -= self.learning_rate * m_V_hat / (np.sqrt(v_V_hat) + self.epsilon)
        self.rnn.c -= self.learning_rate * m_c_hat / (np.sqrt(v_c_hat) + self.epsilon)
        # Reset gradients after update
        self.rnn.grad_U = np.zeros_like(self.rnn.grad_U)
        self.rnn.grad_W = np.zeros_like(self.rnn.grad_W)
        self.rnn.grad_b = np.zeros_like(self.rnn.grad_b)
        self.rnn.grad_V = np.zeros_like(self.rnn.grad_V)
        self.rnn.grad_c = np.zeros_like(self.rnn.grad_c)

    def train(self, book_data: str, seq_length: int, num_epochs: int, converter: Converter):
        epoch = 1
        book_start = 0
        loss_fn = CrossEntropyLoss()
        smooth_loss = None
        while epoch <= num_epochs:
            X_chars = book_data[book_start:book_start+seq_length]
            Y_chars = book_data[book_start+1:book_start+seq_length+1]
            X = converter.char2onehot(X_chars)
            Y = converter.char2onehot(Y_chars)
            logits = self.rnn.forward(X)
            loss_value = loss_fn.forward(logits, Y)
            smooth_loss = loss_value if smooth_loss is None else 0.999 * smooth_loss + 0.001 * loss_value
            self.loss_history.append(smooth_loss)
            self.rnn.backward(loss_fn.backward())
            self.step()
            if smooth_loss < self.best_loss:
                self.best_loss = smooth_loss
                self.best_model = (self.rnn.U.copy(), self.rnn.W.copy(), self.rnn.b.copy(), self.rnn.V.copy(), self.rnn.c.copy())
            book_start += seq_length
            if book_start + seq_length + 1 >= len(book_data):
                book_start = 0
                epoch += 1
                self.rnn.h = np.zeros((1, self.rnn.hidden_size), dtype=np.float32) # reset hidden state at the end of each epoch
            if self.timestep % 5000 == 0:
                print(f"Epoch {epoch}, Update Step {self.timestep}, Loss: {smooth_loss:.4f}")
            if self.timestep % 10000 == 0:
                sample = self.rnn.predict_next_n(X[0:1, :], 200)
                print("------------")
                print("Sampled text:  \n", converter.onehot2char(sample))
                print("------------")
        print(f"Training completed. Best Loss: {self.best_loss:.4f}")

    def save_best_model(self, filename):
        np.savez(filename, U=self.best_model[0], W=self.best_model[1], b=self.best_model[2], V=self.best_model[3], c=self.best_model[4])

    def load_model(self, filename):
        data = np.load(filename)
        print(f"Loaded model from {filename}.")
        self.rnn.U = data['U']
        self.rnn.W = data['W']
        self.rnn.b = data['b']
        self.rnn.V = data['V']
        self.rnn.c = data['c']

    def plot_loss_history(self):
        plt.plot(self.loss_history)
        plt.xlabel('Update Steps')
        plt.ylabel('Smoothed Loss')
        plt.title('Training Loss History')
        plt.show()

def gradient_test():
    seq_length = 25
    hidden_size = 10
    book_fname = "../data/goblet_book.txt"
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()
    unique_chars = list(set(book_data))
    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1:seq_length+1]
    converter = Converter(unique_chars)
    X = converter.char2onehot(X_chars)
    Y = converter.char2onehot(Y_chars)
    rnn = RNN(input_size=len(unique_chars), hidden_size=hidden_size)
    logits = rnn.forward(X)
    loss = CrossEntropyLoss()
    loss_value = loss.forward(logits, Y)
    print("Loss:", loss_value)
    y = np.argmax(Y, axis=1).squeeze()
    print(y.shape)
    gt_grads = ComputeGradsWithTorch(X, y, np.zeros((1, hidden_size)), rnn.W, rnn.U, rnn.V, rnn.b, rnn.c)
    rnn.backward(loss.backward())
    print("Gradient check for W:", np.mean(np.abs(rnn.grad_W - gt_grads['W'])))   
    print("Gradient check for U:", np.mean(np.abs(rnn.grad_U - gt_grads['U'])))
    print("Gradient check for V:", np.mean(np.abs(rnn.grad_V - gt_grads['V'])))
    print("Gradient check for b:", np.mean(np.abs(rnn.grad_b - gt_grads['b'])))
    print("Gradient check for c:", np.mean(np.abs(rnn.grad_c - gt_grads['c'])))

def train_rnn():
    seq_length = 25
    hidden_size = 100
    num_epochs = 4
    book_fname = "../data/goblet_book.txt"
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()
    rnn = RNN(input_size=len(set(book_data)), hidden_size=hidden_size)
    optimizer = AdamOptimizer(rnn)
    converter = Converter(sorted(list(set(book_data))))
    X = converter.char2onehot(book_data[0:1])
    sample = rnn.predict_next_n(X, 1000)
    print("------------")
    print("Sampled text:  \n", converter.onehot2char(sample))
    print("------------")
    optimizer.train(book_data, seq_length, num_epochs, converter)
    optimizer.plot_loss_history()
    optimizer.save_best_model("rnn_goblet_book.npz")
    sample = rnn.predict_next_n(X, 1000)
    print("------------")
    print("Sampled text after training:  \n", converter.onehot2char(sample))
    print("------------")
    print("Best model saved to rnn_goblet_book.npz")

def task_4():
    # Load the best model and generate text
    seq_length = 25
    hidden_size = 100
    book_fname = "../data/goblet_book.txt"
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()
    rnn = RNN(input_size=len(set(book_data)), hidden_size=hidden_size)
    optimizer = AdamOptimizer(rnn)
    optimizer.load_model("rnn_goblet_book.npz")
    converter = Converter(sorted(list(set(book_data))))
    X = converter.char2onehot(book_data[0:1])
    sample = rnn.predict_next_n(X, 1000)
    print("------------")
    print(converter.onehot2char(sample))
    print("------------")

if __name__ == "__main__":
    # gradient_test()
    train_rnn()
    # task_4()