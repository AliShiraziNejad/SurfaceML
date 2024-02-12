import numpy as np


def softmax(x, axis=None):
    """
    Compute the softmax of each slice of an N-dimensional array x along the specified axis.
    If axis is None, softmax will be applied over the entire array.

    Args:
    - x (np.ndarray): Input array.
    - axis (int, optional): Axis along which softmax is to be computed. Default is None.

    Returns:
    - np.ndarray: Array with the same shape as x, with softmax applied along the specified axis.
    """

    # Shift the input x by subtracting its max value along the specified axis
    # for numerical stability (prevents overflow/underflow issues with exp).
    x_max = np.max(x, axis=axis, keepdims=True)

    x_exp = np.exp(x - x_max)

    sum_x_exp = np.sum(x_exp, axis=axis, keepdims=True)

    softmax_x = x_exp / sum_x_exp

    return softmax_x


def log_softmax(x, axis=None):
    """
    Compute the log softmax of each slice of an N-dimensional array x along the specified axis.
    If axis is None, log softmax will be applied over the entire array.

    Args:
    - x (np.ndarray): Input array.
    - axis (int, optional): Axis along which log softmax is to be computed. Default is None.

    Returns:
    - np.ndarray: Array with the same shape as x, with log softmax applied along the specified axis.
    """

    # Shift the input x by subtracting its max value along the specified axis
    # for numerical stability (prevents overflow/underflow issues with exp).
    x_max = np.max(x, axis=axis, keepdims=True)

    x_shifted = x - x_max

    log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))

    log_softmax_x = x_shifted - log_sum_exp

    return log_softmax_x


