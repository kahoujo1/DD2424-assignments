import numpy as np

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: np.array) -> None:
        """
        Calculates the mean and stardard deviation from the supplied dataset.

        Args:
            X (np.array): dataset matrix of shape (D, N) where N is the number of samples, and D is the dimensionality.
        """
        self.mean = np.mean(X, axis = 1, keepdims=True)
        self.std = np.std(X, axis = 1, keepdims=True)

    def transform(self, X: np.array) -> np.array:
        """
        Transforms supplied dataset by centering and scaling using precomputed mean and standard deviation.

        Args:
            X (np.array): dataset matrix of shape (D, N) where N is the number of samples, and D is the dimensionality.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet!")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.array) -> np.array:
        """
        Fits the scaler to the supplied dataset and then transforms it.

        Args:
            X (np.array): dataset matrix of shape (D, N) where N is the number of samples, and D is the dimensionality.
        """
        self.fit(X)
        return self.transform(X)