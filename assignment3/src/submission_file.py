import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch

from nodes import *
from model import Model
from ADAM import ADAM
from optimizer import Optimizer
from utils import *

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
    X, Y, y = load_training_batches()
    # split into training and validation set
    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:,49000:], y[49000:]
    # transform X into Mx for the initial Patchify layer
    Mx_train = precompute_Mx(X_train, f)
    Mx_val = precompute_Mx(X_val, f)
    # train the model
    optimizer.train_with_cyclical_lr(Mx_train, y_train, Mx_val, y_val, lr_min=1e-5, lr_max=1e-1, step_size=800, n_cycles=3, batch_size=100, print_every=100)
    X_test, Y_test, y_test = load_batch('test_batch')
    Mx_test = precompute_Mx(X_test, f)
    test_acc = optimizer.compute_accuracy(Mx_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    optimizer.plot_cyclical_lr_training_progress()
def main():
    pass

if __name__ == "__main__":
    # main()
    # excercise1()
    # excercise2()
    # excercise2_precompute_Mx()
    # test_grads_with_torch()
    excercise3()