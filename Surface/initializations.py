import numpy as np


def glorot_uniform(shape):
    limit = np.sqrt(6 / np.sum(shape))
    return np.random.uniform(-limit, limit, size=shape)


def glorot_normal(shape):
    stddev = np.sqrt(2 / np.sum(shape))
    return np.random.normal(0, stddev, size=shape)


def he_uniform(shape):
    limit = np.sqrt(6 / shape[0])
    return np.random.uniform(-limit, limit, size=shape)


def he_normal(shape):
    stddev = np.sqrt(2 / shape[0])
    return np.random.normal(0, stddev, size=shape)
