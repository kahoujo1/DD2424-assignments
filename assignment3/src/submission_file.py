import numpy as np
import matplotlib.pyplot as plt
import pickle

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

def main():
    pass

if __name__ == "__main__":
    # main()
    # excercise1()
    excercise2()