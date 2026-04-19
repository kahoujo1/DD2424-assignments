import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from nodes import *
from model import Model
from scaler import Scaler
from utils import *
from optimizer import Optimizer

def label_smoothing():
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
    nf = 60
    nh = 400
    lam = 0.002
    # precompute Mx for all sets
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    # without label smoothing
    model = Model(f, nf, nh, 10, p = 0.2)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=lam, label_smoothing=0.1, lr_decay=0.9)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=4, batch_size=100, print_every=0)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    print(f"Architecture 5 (no label smoothing) - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()
    optimizer.plot_learning_rate_history()

def architecture_search():
    """
    Performs a search over different architectures and hyperparameters, and saves the results to a file for later analysis.
    """
    scaler = Scaler()
    X, Y, y = load_training_batches()
    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:,49000:], y[49000:]
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test, Y_test, y_test = load_batch('test_batch')
    X_test = scaler.transform(X_test)
    f = 4 # we can keep f fixed to 4 since it performed best in the previous assignment
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    nf_values = [40, 60, 100, 200]
    nh_values = [200, 400, 800]
    lam_values = [0.001, 0.005, 0.01]
    smoothing_values = [0.0, 0.1, 0.2]
    decay_values = [0.8, 0.9, 1.0]
    results = []
    for nf in nf_values:
        for nh in nh_values:
            for lam in lam_values:
                for smoothing in smoothing_values:
                    for decay in decay_values:
                        print(f"Training model with nf={nf}, nh={nh}, lam={lam}, smoothing={smoothing}, decay={decay}...")
                        model = Model(f, nf, nh, 10)
                        optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=lam, label_smoothing=smoothing, lr_decay=decay)
                        optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=4, batch_size=100, print_every=0)
                        test_acc = optimizer.compute_accuracy(Mx_test, y_test)
                        val_acc = optimizer.compute_accuracy(Mx_val, y_val)
                        results.append((f, nf, nh, lam, smoothing, decay, test_acc, val_acc))
                        print(f"f={f}, nf={nf}, nh={nh}, lam={lam}, smoothing={smoothing}, decay={decay} - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    with open('architecture_search_results.pkl', 'wb') as f:
        pickle.dump(results, f)
# Maximum from the search (note that it was stopped after some time, so there may be better combinations that were not explored):
#f=4, nf=40, nh=400, lam=0.005, smoothing=0.0, decay=0.9 - Test accuracy: 0.6605, Validation accuracy: 0.6640

def test_flipping():
    scaler = Scaler()
    X, Y, y = load_training_batches()
    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:,49000:], y[49000:]
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test, Y_test, y_test = load_batch('test_batch')
    X_test = scaler.transform(X_test)
    f = 4
    nf = 10
    Mx_train = precompute_Mx(X_train, f)
    Mx_train_flipped = precompute_Mx(flip_vertically(X_train),f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    # without flipping:
    print("Testing with and without flipping...")
    model = Model(f, nf, 100, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.005, label_smoothing=0.1, lr_decay=0.9)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=1)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    print(f"Without flipping - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()
    # with flipping:
    model = Model(f, nf, 100, 10)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.005, label_smoothing=0.1, lr_decay=0.9)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=400, flip_prob=0.5, Mx_train_flip=Mx_train_flipped)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val)
    print(f"With flipping - Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()

def main():
    scaler = Scaler()
    X, Y, y = load_training_batches()
    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:,49000:], y[49000:]
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test, Y_test, y_test = load_batch('test_batch')
    X_test = scaler.transform(X_test)
    f = 4
    nf = 80
    Mx_train = precompute_Mx(X_train, f)
    Mx_train_flipped = precompute_Mx(flip_vertically(X_train),f)
    Mx_val = precompute_Mx(X_val, f)
    Mx_test = precompute_Mx(X_test, f)
    model = Model(f, nf, 600, 10, p = 0.3)
    optimizer = Optimizer(model, CrossEntropyLoss(), lr=0.001, reg=0.002, label_smoothing=0.1, lr_decay=0.9)
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=5, batch_size=100, print_every=800, flip_prob=0.5, Mx_train_flip=Mx_train_flipped)
    optimizer.set_eval_mode()
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    val_acc = optimizer.compute_accuracy(Mx_val, y_val) 
    print(f"Test accuracy: {test_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()
    optimizer.plot_learning_rate_history()

if __name__ == "__main__":
    main()
    # architecture_search()
    # test_flipping()