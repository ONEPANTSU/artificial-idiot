import numpy as np


def cross_entropy(y, p):
    """Cross Entropy"""
    y_arr = np.zeros(len(p))
    y_arr[y] = 1
    return p - y_arr


def softmax(x):
    """Softmax"""
    return np.exp(x) / np.exp(x).sum()


def relu(x):
    """ReLU"""
    return max(0.0, x)


def relu_derivative(x):
    """Производная ReLU"""
    return 1 if x >= 0 else 0


def sigmoid(x):
    """Сигмоида"""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """Производная сигмоиды"""
    return sigmoid(x) * (1 - sigmoid(x))
