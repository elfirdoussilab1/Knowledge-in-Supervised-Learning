# This file contains the implementation of the theoretical results of the report
import numpy as np
from utils import *
import scipy.integrate as integrate
import utils

def Delta(eta, gamma):
    return (eta - gamma - 1 + np.sqrt((eta - gamma - 1)**2 + 4*eta*gamma)) / (2 * gamma)

def test_expectation_w_1(n, p, vmu, vk, gamma):
    eta = p/n
    delta = Delta(eta, gamma)
    num = np.dot(vmu + vk, vmu)
    den = np.sum((vmu + vk)**2)
    return num / (den + 1 + gamma * (1 + delta))

def test_expectation_w_2(n, p, vmu, k, gamma):
    eta = p/n
    delta = Delta(eta, gamma)
    mu_2 = np.sum(vmu**2)
    return (1 - k) * mu_2 / (mu_2 + 1 + gamma * (1 + delta))

def test_expectation(classifier, n, p, vmu, vk, gamma):
    if classifier == 'w_1':
        return test_expectation_w_1(n, p, vmu, vk, gamma)
    elif classifier == 'w_2':
        return test_expectation_w_2(n, p, vmu, vk, gamma)
    else:
        raise NotImplementedError("classifier should be either 'w_1' or 'w_2'")

# Comuting the h
def H(n, p, gamma):
    eta = p /n
    delta = Delta(eta, gamma)
    return 1 - eta / (1 + gamma * (1 + delta))**2

# Second order moments
def test_expectation_2_w_1(n, p, vmu, vk, gamma):
    # Useful quantities
    eta = p/n
    delta = Delta(eta, gamma)
    h = H(n, p, gamma)
    # <vmu + vk, vmu>
    r_1 = np.dot(vmu + vk, vmu)
    # || vmu + vk ||^2
    r_2 = np.sum((vmu + vk)**2)

    # Remaining terms
    denom = r_2 + 1 + gamma * (1 + delta)
    s = (r_1**2 + r_2**2 * (1 - h) / h + r_2 / h ) / denom - 2 * (1 - h) * r_2 / h
    s = s / denom
    return s + (1 - h) / h

def test_expectation_2_w_2(n, p, vmu, k, gamma):
    # Useful quantities
    eta = p/n
    delta = Delta(eta, gamma)
    h = H(n, p, gamma)
    mu_2 = np.sum(vmu**2)
    denom = mu_2 + 1 + gamma*(1 + delta)

    # Computing final term
    s = mu_2 * ((mu_2 + 1) / denom - 2 * (1 - h)) / (h * denom) + (1 - h) / h

    return s * (1 - k)**2

def test_expectation_2(classifier, n, p, vmu, vk, gamma):
    if classifier == 'w_1':
        return test_expectation_2_w_1(n, p, vmu, vk, gamma)
    elif classifier == 'w_2':
        return test_expectation_2_w_2(n, p, vmu, vk, gamma)
    else:
        raise NotImplementedError("classifier should be either 'w_1' or 'w_2'")
    
# Computing Test accuracy
def test_accuracy(classifier, n, p, vmu, vk, gamma):
    # E[g] and E[g^2]
    mean = test_expectation(classifier, n, p, vmu, vk, gamma)
    expec_2 = test_expectation_2(classifier, n, p, vmu, vk, gamma)
    std = np.sqrt(expec_2 - mean**2)
    return 1 - integrate.quad(lambda x: utils.gaussian(x, 0, 1), abs(mean)/std, np.inf)[0]

# Computing Test Risk
def test_risk(classifier, n, p, vmu, vk, gamma):

    # E[g] and E(g^2)
    mean = test_expectation(classifier, n, p, vmu, vk, gamma)
    expec_2 = test_expectation_2(classifier, n, p, vmu, vk, gamma)
    return expec_2 + 1 - 2 * mean
