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

def main():
    pass

if __name__ == "__main__":
    # main()
    excercise1()