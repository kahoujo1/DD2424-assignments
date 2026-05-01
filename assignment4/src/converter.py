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
