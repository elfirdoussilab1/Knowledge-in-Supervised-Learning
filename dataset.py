import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

type_to_path = {
    'book': './data/amazon_review/books.mat',
    'dvd': './data/amazon_review/dvd.mat',
    'elec': './data/amazon_review/elec.mat',
    'kitchen': './data/amazon_review/kitchen.mat',
}

# Amazon review dataset
class Amazon_data:
    def __init__(self, type = 'book', n = 1600):
        # n is the number of training samples
        # Load the dataset
        data = loadmat(type_to_path[type])
        self.X = data['fts'] # shape (N, p)

        # Transformation on labels
        self.y = data['labels'].reshape((len(self.X), )).astype(int) # shape (n, ), features are sorted by reversed order (ones then zeros)
        self.y = 1 - 2 * self.y
        # Preprocessing
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        self.vmu_1 = np.mean(self.X[self.y < 0], axis = 0)
        self.vmu_2 = np.mean(self.X[self.y > 0], axis = 0)

        # Train and test splits
        # Generate a shuffled index
        indices = np.random.permutation(len(self.X))

        # Split the indices
        train_idx = indices[:n]
        test_idx = indices[n:]
        self.X_train, self.y_train = self.X[train_idx], self.y[train_idx]
        self.X_test, self.y_test = self.X[test_idx], self.y[test_idx]
        