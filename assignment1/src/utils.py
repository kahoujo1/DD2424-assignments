import pickle
import numpy as np

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
    cifar_dir = '../data/cifar-10-batches-py/'
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