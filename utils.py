# This file contains all the util functions used in almost every experiment reported in the papaer.
import numpy as np
import random
import numbers
from scipy.stats import norm

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Data generation
def gaussian_mixture(vmu, n, pi=0.5):
    p = len(vmu)
    y = np.ones(n)
    y[:int(n * pi)] = -1
    Z = np.random.randn(p, n) # (z_1, ..., z_n)
    X = np.outer(vmu, y) + Z # np.outer = vmu.T @ y
    return X, y # X of shape (p, n)

def generate_data(n, vmu, vk, mode, pi = 0.5):  
    """
    Function to generate synthetic data
    params:
        n (int): total number of data vectors
        vmu (np.array): Mean vector of the data
        vk: knowledge k in the theory: either a p dimensional vetor when mode = 'features', or scalar when mode = 'labels
        model (str): either 'features' or 'labels'
        pi (int): n*pi is the proportion of negative labels (doesn't have impact here on classifier)
    """
    p = len(vmu)
    X_train, y_train = gaussian_mixture(vmu, n, pi)
    X_test, y_test = gaussian_mixture(vmu, 2*n)
    if mode == 'labels':
        assert isinstance(vk, numbers.Number), f"k must be a scalar number when using mode {mode}"
        # Multiply labels by (1 - k)
        y_train = y_train * (1 - vk)
    elif mode == 'features':
        assert len(vk) == p
        # Add y_i * vk to each sample x_i
        X_train += np.outer(vk, y_train)
    else:
        raise NotImplementedError("mode should be either 'features' or 'labels'.")
    return (X_train, y_train), (X_test, y_test)

# Binary accuracy function
def accuracy(y, y_pred):
    acc = np.mean(y == y_pred)
    return max(acc, 1 - acc)

# Decision functions:
def classifier_vector(X, y, gamma):
    return np.linalg.inv(X @ X.T / X.shape[1] + gamma * np.eye(X.shape[0])) @ X @ y / X.shape[1]  


# g(x) = <w, x>
g = lambda w, X: X.T @ w

# Labelling
decision = lambda w, X: 2 * (g(w, X) >= 0) - 1

# Losses
def L2_loss(w, X, y):
    # X of shape (p, n)
    return np.mean((X.T @ w - y)**2)

def empirical_accuracy(batch, n, vmu, vk, gamma, pi, mode, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
           (X_train, y_train), (X_test, y_test) = generate_data(n, vmu, vk, mode, pi)
        
        else: 
            raise NotImplementedError("Not implmeented yet.")
        # elif 'amazon' in data_type:
        #     type = data_type.split('_')[1]
        #     data = dataset.Amazon(epsp, epsm, type, pi, n)
        #     X_train, y_train_noisy = data.X_train.T, data.y_train_noisy
        #     X_test, y_test = data.X_test.T, data.y_test

        w = classifier_vector(X_train, y_train, gamma)
        res += accuracy(y_test, decision(w, X_test))
    return res / batch

def empirical_mean(batch, n, vmu, vk, gamma, pi, mode, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_train, y_train), (X_test, y_test) = generate_data(n, vmu, vk, mode, pi)
        
        else:
            raise NotImplementedError("Not implmeented yet.")

        w = classifier_vector(X_train, y_train, gamma)
        res += np.mean(y_test * (X_test.T @ w))
    return res / batch

def empirical_mean_2(batch, n, vmu, vk, gamma, pi, mode, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_train, y_train), (X_test, y_test) = generate_data(n, vmu, vk, mode, pi)
        else:
            raise NotImplementedError("Not implmeented yet.")

        w = classifier_vector(X_train, y_train, gamma)
        res += np.mean((X_test.T @ w)**2)
    return res / batch

def empirical_risk(batch, n, vmu, vk, gamma, pi, mode, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        # generate new data
        if 'synthetic' in data_type:
            (X_train, y_train), (X_test, y_test) = generate_data(n, vmu, vk, mode, pi)
        else:
            raise NotImplementedError("Not implmeented yet.")

        w = classifier_vector(X_train, y_train, gamma)
        res += L2_loss(w, X_test, y_test)
    return res / batch
    

def information_vectors(k_norms, mode, p = 1):
    if mode == 'labels':
        return k_norms
    else: # features
        k = len(k_norms)
        vks = np.zeros((k, p))
        vks[:,0] = k_norms
        return vks
    
# Gaussian density function
def gaussian(x, mean, std):
    return np.exp(- (x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

def bayes_accuracy(vmu):
    """
    Compute Bayes accuracy when the means are symmetric: N(-mu, I) vs N(mu, I)
    
    Parameters:
    - mu: Mean vector (numpy array)
    
    Returns:
    - Bayes accuracy (float)
    """
    return norm.cdf(np.linalg.norm(vmu))
