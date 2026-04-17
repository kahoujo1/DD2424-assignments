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

def load_training_batches():
    """
    Loads all the training batches and concatenates them.

    Returns:
        Array of [X, Y, y] where
        - X contains the flattened image pixel data of shape (32x32x3, N)
        - Y is the true probability class distribution of shape (K, N) (one hot encoding)'
        - y is a vector of true labels (N,)
    """
    X, Y, y = load_batch("data_batch_1")
    for i in range(2,6):
        X_temp, Y_temp, y_temp = load_batch(f"data_batch_{i}")
        X = np.concatenate((X, X_temp), axis=1)
        Y = np.concatenate((Y, Y_temp), axis=1)
        y = np.concatenate((y, y_temp))
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

def precompute_Mx(X: np.ndarray, f: int) -> np.ndarray:
    """
    Precomputes the matrix Mx for the initial Patchify layer.
    
    Args:
        X (numpy array): Input image data of shape (32x32x3, N).
        f (int): The size of the filter (patch) to be extracted from the input image.
    
    Returns:
        numpy array: The precomputed Mx matrix of shape (Np, 3f^2, N), where Np is the number of patches and N is the batch size.
    """
    N = X.shape[1]
    X_ims = np.transpose(X.reshape((32, 32, 3, N), order='F'), (1, 0, 2, 3))
    Np = (32//f)**2
    Mx = np.zeros((Np, f*f*3, N), dtype=np.float32)
    for n in range(N):
        region = 0
        for i in range(32//f):
            for j in range(32//f):
                patch = X_ims[i*f:(i+1)*f, j*f:(j+1)*f, :, n]
                Mx[region, :, n] = patch.reshape((1, f*f*3), order='C')
                region += 1
    return Mx

def flip_vertically(X: np.ndarray) -> np.ndarray:
    """
    Flips the input images vertically.

    Args: 
        X (numpy array): Input batch of shape (D, N), where N is the number of samples, and D is the flattened image dimensionality.
    
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