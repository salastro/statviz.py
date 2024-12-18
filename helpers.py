import numpy as np


def calc_bin_w(Xuq: np.ndarray) -> float:
    """
    Calculate bin width using Scott's Rule.
    Xuq: unique values
    Returns: bin width
    """
    n = len(Xuq)
    std = np.std(Xuq)
    if std == 0:
        bin_width = 1.0
    else:
        bin_width = 3.5 * std / (n ** (1 / 3))
    return bin_width


def calc_stats(X: np.ndarray, P: np.ndarray) -> tuple[float, float, float]:
    """
    Calculates and displays mean, variance, and third moment.
    X: data
    P: probability
    Returns: mean, variance, and third moment
    """
    Xuq = np.unique(X)
    mean_X = np.sum(Xuq * P)
    var_X = np.sum((Xuq - mean_X) ** 2 * P)
    third_moment = np.sum((Xuq - mean_X) ** 3 * P)
    return mean_X, var_X, third_moment
