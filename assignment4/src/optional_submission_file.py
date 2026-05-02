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
    def __init__(self, input_size, hidden_size, batch_size=1):
        self.input_size = input_size # K in the equations
        self.hidden_size = hidden_size # m in the equations
        self.batch_size = batch_size
        # note that the inputs will be saved as rows
        self.h = np.zeros((batch_size, self.hidden_size), dtype=np.float32) # initial hidden state
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
        # use the hidden state from the first batch
        h = self.h[0:1, :]
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
            X (np.ndarray): A 3D array of shape (T, B, input_size) where T is the sequence length, B is the batch size, and input_size is the size of the one-hot encoding.
        
        Returns:
            np.ndarray: A 3D array of shape (T, B, input_size) representing the output logits (to utilize softmax backpropagation trick) for each time step.
        """
        T = X.shape[0]
        B = X.shape[1]
        Ux = X @ self.U # precompute input to hidden transformations
        H = np.zeros((T+1, B, self.hidden_size), dtype=np.float32) # to store hidden states for backpropagation
        H[-1, :, :] = self.h # save to the end for backpropagation
        Y = np.zeros((T, B, self.input_size), dtype=np.float32)
        for t in range(T):
            self.h = np.tanh(Ux[t, :, :] + self.h @ self.W + self.b)
            H[t, :, :] = self.h
            o = self.h @ self.V + self.c # compute output logits
            Y[t, :, :] = o
        # TODO: save the needed values for backpropagation
        self.hidden_states = H
        self.outputs = Y
        self.inputs = X
        return Y
    
    def backward(self, grad_Y: np.ndarray):
        """
        Performs backpropagation through time to compute gradients of the loss with respect to the model parameters.

        Args:
            grad_Y (np.ndarray): A 3D array of shape (T, B, input_size) representing the gradient of the loss with respect to the output logits for each time step.
        
        Updates:
            self.grad_U, self.grad_W, self.grad_b, self.grad_V, self.grad_c: The computed gradients for the model parameters.
        """
        T = grad_Y.shape[0]
        B = grad_Y.shape[1]
        grad_h_next = np.zeros((B, self.hidden_size), dtype=np.float32) # gradient of the loss with respect to the next hidden state
        for t in reversed(range(T)):
            grad_o = grad_Y[t, :, :] # (B, K)
            h_t = self.hidden_states[t, :, :] # (B, m)
            h_t_prev = self.hidden_states[t-1, :, :] # (B, m)
            self.grad_V += h_t.T @ grad_o 
            self.grad_c += grad_o.sum(axis=0, keepdims=True) 

            grad_h = grad_o @ self.V.transpose() + grad_h_next # gradient of the loss with respect to the hidden state at time t
            grad_h_raw = (1 - h_t ** 2) * grad_h # backprop through tanh
            self.grad_b += grad_h_raw.sum(axis=0, keepdims=True)

            self.grad_W += h_t_prev.T @ grad_h_raw
            self.grad_U += self.inputs[t, :, :].T @ grad_h_raw
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
        self.num_of_batches = None

    def forward(self, logits: np.array, Y: np.array) -> np.float64:
        """
        Calculates the forward pass for the cross-entropy loss with softmax.

        Args:
            logits (numpy array): Input logits of shape (T, B, K) where B is the batch number, K is the number of classes and T is the sequence length.
            Y (numpy array): True probability distribution of the classes with size (T, B, K) (most often one hot encoding).

        Returns:
            numpy.float64: The average cross-entropy loss over the batches.
        """
        self.num_of_batches = logits.shape[1] # B
        logits = logits.reshape(-1, logits.shape[-1]) # reshape to (B*T, K)
        Y = Y.reshape(-1, Y.shape[-1]) # reshape to (B*T, K)
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
            numpy.array: Gradient of the loss with respect to the input logits, of shape (B, T, K).
        """
        grad = (self.P - self.Y) / self.P.shape[0] 
        grad = grad.reshape(-1, self.num_of_batches, self.P.shape[-1]) # reshape back to (T, B, K)
        return grad
    
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
        self.validation_loss_history = []
    
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
        batch_size = self.rnn.batch_size
        loss_fn = CrossEntropyLoss()
        # split into B+1 chunks
        batch_matrix = converter.char2onehot(book_data)
        chunk_size = len(book_data) // (batch_size + 1)
        batch_matrix = batch_matrix[:chunk_size * (batch_size + 1), :]  # trim to a multiple of batch_size
        batch_matrix = batch_matrix.reshape(batch_size + 1, chunk_size, self.rnn.input_size).transpose(1, 0, 2)  # (seq_len, batch_size, input_size)
        training_matrix = batch_matrix[:, :batch_size, :] # (seq_len, batch_size, input_size)
        validation_matrix = batch_matrix[:, batch_size:, :] # (seq_len, 1, input_size)
        smooth_loss = None
        smooth_val_loss = None
        val_h = np.zeros((1, self.rnn.hidden_size), dtype=np.float32) # hidden state for validation
        while epoch <= num_epochs:
            X = training_matrix[book_start:book_start+seq_length, :, :]
            Y = training_matrix[book_start+1:book_start+seq_length+1, :, :]
            logits = self.rnn.forward(X)
            loss_value = loss_fn.forward(logits, Y)
            smooth_loss = loss_value if smooth_loss is None else 0.999 * smooth_loss + 0.001 * loss_value
            self.loss_history.append(smooth_loss)
            self.rnn.backward(loss_fn.backward())
            self.step()
            # calculate validation loss and save best model
            rnn_h = self.rnn.h.copy() # save current hidden state
            self.rnn.h = val_h
            val_logits = self.rnn.forward(validation_matrix[book_start:book_start+seq_length, :, :])
            val_loss = loss_fn.forward(val_logits, validation_matrix[book_start+1:book_start+seq_length+1, :, :])
            smooth_val_loss = val_loss if len(self.validation_loss_history) == 0 else 0.999 * self.validation_loss_history[-1] + 0.001 * val_loss
            self.validation_loss_history.append(smooth_val_loss)
            self.rnn.h = rnn_h # restore hidden state after validation
            if smooth_loss < self.best_loss:
                self.best_loss = smooth_loss
                self.best_model = (self.rnn.U.copy(), self.rnn.W.copy(), self.rnn.b.copy(), self.rnn.V.copy(), self.rnn.c.copy())
            book_start += seq_length
            if book_start + seq_length + 1 >= len(training_matrix):
                book_start = 0
                epoch += 1
                self.rnn.h = np.zeros((self.rnn.batch_size, self.rnn.hidden_size), dtype=np.float32) # reset hidden state at the end of each epoch
                val_h = np.zeros((1, self.rnn.hidden_size), dtype=np.float32)
            if self.timestep % 5000 == 0:
                print(f"Epoch {epoch}, Update Step {self.timestep}, Loss: {smooth_loss:.4f}")
            if self.timestep % 10000 == 0:
                sample = self.rnn.predict_next_n(X[0:1, 0, :], 200)
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
        plt.plot(self.loss_history, label='Training Loss')
        plt.plot(self.validation_loss_history, label='Validation Loss')
        plt.legend()
        plt.xlabel('Update Steps')
        plt.ylabel('Smoothed Loss')
        plt.title('Training Loss History')
        plt.show()

def train_rnn():
    seq_length = 25
    hidden_size = 100
    batch_size = 16
    num_epochs = 10
    book_fname = "../data/goblet_book.txt"
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()
    rnn = RNN(input_size=len(set(book_data)), hidden_size=hidden_size, batch_size=batch_size)
    optimizer = AdamOptimizer(rnn)
    converter = Converter(sorted(list(set(book_data))))
    X = converter.char2onehot(book_data[0:1])
    sample = rnn.predict_next_n(X, 1000)
    print("------------")
    print("Sampled text:  \n", converter.onehot2char(sample))
    print("------------")
    optimizer.train(book_data, seq_length, num_epochs, converter)
    optimizer.plot_loss_history()
    optimizer.save_best_model("optional_model.npz")
    sample = rnn.predict_next_n(X, 1000)
    print("------------")
    print("Sampled text after training:  \n", converter.onehot2char(sample))
    print("------------")
    print("Best model saved to optional_model.npz")

def _updates_per_epoch(text_len, batch_size, seq_length):
    # replicate logic from AdamOptimizer.train for chunk sizing
    chunk_size = text_len // (batch_size + 1)
    if chunk_size <= 0:
        return 1
    # number of updates when stepping by seq_length until end
    return max(1, (chunk_size - seq_length - 1) // seq_length + 1)

def run_batch_experiments(book_fname="../data/goblet_book.txt", seq_length=25, hidden_size=100, batch_sizes=(1,4,8,16), base_batch=1, base_epochs=5):
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()
    text_len = len(book_data)
    # compute target total updates from base configuration
    base_updates_per_epoch = _updates_per_epoch(text_len, base_batch, seq_length)
    target_updates = base_updates_per_epoch * base_epochs
    all_runs = {}
    for b in batch_sizes:
        updates_per_epoch = _updates_per_epoch(text_len, b, seq_length)
        epochs = max(1, int(round(target_updates / updates_per_epoch)))
        rnn = RNN(input_size=len(set(book_data)), hidden_size=hidden_size, batch_size=b)
        optimizer = AdamOptimizer(rnn)
        converter = Converter(sorted(list(set(book_data))))
        print(f"Training batch_size={b}, epochs={epochs}, target_updates~{target_updates}")
        optimizer.train(book_data, seq_length, epochs, converter)
        # save histories as numpy arrays
        loss_arr = np.array(optimizer.loss_history)
        val_arr = np.array(optimizer.validation_loss_history)
        np.savez(f"loss_hist_bs{b}.npz", loss=loss_arr, val=val_arr)
        all_runs[b] = (loss_arr, val_arr)
    # also save combined
    np.savez("all_loss_histories.npz", **{f"bs{b}_loss": all_runs[b][0] for b in all_runs}, **{f"bs{b}_val": all_runs[b][1] for b in all_runs})

def plot_loss_histories(batch_sizes=(1,4,8,16)):
    # plot the training losses together for comparison
    plt.figure(figsize=(12, 6))
    for b in batch_sizes:
        data = np.load(f"loss_hist_bs{b}.npz")
        plt.plot(data['loss'], label=f'Batch Size {b}')
    plt.legend()
    plt.xlabel('Update Steps')
    plt.ylabel('Smoothed Training Loss')
    plt.title('Training Loss History for Different Batch Sizes')
    plt.show()
    # validation losses together
    plt.figure(figsize=(12, 6))
    for b in batch_sizes:
        data = np.load(f"loss_hist_bs{b}.npz")
        plt.plot(data['val'], label=f'Batch Size {b}')
    plt.legend()
    plt.xlabel('Update Steps')
    plt.ylabel('Smoothed Validation Loss')
    plt.title('Validation Loss History for Different Batch Sizes')
    plt.show()

if __name__ == "__main__":
    # gradient_test()
    # train_rnn()

    run_batch_experiments()
    # task_4()