import numpy as np
from scipy.stats import truncnorm


def calculate_fan_in_fan_out(shape):
    if len(shape) == 2:  # Linear layer
        fan_in, fan_out = shape[1], shape[0]
    else:  # Conv1D, Conv2D, Conv3D
        receptive_field_size = np.prod(shape[2:])  # Product of kernel dimensions
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    return fan_in, fan_out


def glorot_uniform(shape):
    fan_in, fan_out = calculate_fan_in_fan_out(shape)
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def glorot_normal(shape):
    fan_in, fan_out = calculate_fan_in_fan_out(shape)
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, stddev, size=shape)


def he_uniform(shape):
    fan_in, _ = calculate_fan_in_fan_out(shape)
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


def he_normal(shape):
    fan_in, _ = calculate_fan_in_fan_out(shape)
    stddev = np.sqrt(2 / fan_in)
    return np.random.normal(0, stddev, size=shape)


def trunc_norm(shape, mean=0.0, stddev=None, lower=-2.0, upper=2.0):
    fan_in, _ = calculate_fan_in_fan_out(shape)
    if stddev is None:
        stddev = np.sqrt(1. / fan_in)  # He initialization strategy
    lower_bound = (lower - mean) / stddev
    upper_bound = (upper - mean) / stddev
    distribution = truncnorm(a=lower_bound, b=upper_bound, loc=mean, scale=stddev)
    return distribution.rvs(size=shape)
