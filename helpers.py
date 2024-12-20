import numpy as np
from scipy.io import loadmat


def read_file_joint(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads sample pairs (X, Y) from a MATLAB .mat file.
    Returns arrays with float64 dtype.
    """
    try:
        data = loadmat(filename).get("XY")
        # Convert to float64 during extraction
        X = data[0, :].astype(np.float64)
        Y = data[1, :].astype(np.float64)

        if X.size == 0 or Y.size == 0:
            raise ValueError("The .mat file must contain 'X' and 'Y' variables.")
        if len(X) != len(Y):
            raise ValueError("The number of samples in 'X' and 'Y' do not match.")
        return X, Y
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def calc_joint_prob(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculates joint probability distribution using float64 precision.
    """
    # Convert inputs to float64 if they aren't already
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    Xuq, X_inv = np.unique(X, return_inverse=True)
    Yuq, Y_inv = np.unique(Y, return_inverse=True)

    Nxy = np.zeros((len(Xuq), len(Yuq)), dtype=np.float64)
    np.add.at(Nxy, (X_inv, Y_inv), 1)

    Pxy = Nxy / len(X)
    return Pxy


def calc_marg_prob(Pxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal distributions in float64 precision.
    """
    Pxy = Pxy.astype(np.float64)
    Px = np.sum(Pxy, axis=1, dtype=np.float64)
    Py = np.sum(Pxy, axis=0, dtype=np.float64)
    return Px, Py


def calc_cov(X: np.ndarray, Y: np.ndarray, Pxy: np.ndarray) -> float:
    """
    Calculate covariance using float64 precision.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Pxy = Pxy.astype(np.float64)

    Xuq = np.unique(X)
    Yuq = np.unique(Y)

    Ex = np.sum(Xuq * np.sum(Pxy, axis=1), dtype=np.float64)
    Ey = np.sum(Yuq * np.sum(Pxy, axis=0), dtype=np.float64)

    Exy = np.sum(np.outer(Xuq, Yuq) * Pxy, dtype=np.float64)
    covariance = Exy - (Ex * Ey)
    return covariance


def calc_cor(X: np.ndarray, Y: np.ndarray, Pxy: np.ndarray) -> float:
    """
    Calculate correlation coefficient using float64 precision.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Pxy = Pxy.astype(np.float64)

    Xuq = np.unique(X)
    Yuq = np.unique(Y)

    Ex = np.sum(Xuq * np.sum(Pxy, axis=1), dtype=np.float64)
    Ey = np.sum(Yuq * np.sum(Pxy, axis=0), dtype=np.float64)

    Ex2 = np.sum((Xuq**2) * np.sum(Pxy, axis=1), dtype=np.float64)
    Ey2 = np.sum((Yuq**2) * np.sum(Pxy, axis=0), dtype=np.float64)

    Var_X = Ex2 - Ex**2
    Var_Y = Ey2 - Ey**2

    covariance = calc_cov(X, Y, Pxy)
    correlation = covariance / np.sqrt(Var_X * Var_Y)
    return correlation


def calc_bin_width(Xuq: np.ndarray) -> float:
    """
    Calculate bin width using Scott's Rule.
    Xuq: unique values
    Returns: bin width
    """
    n = len(Xuq)
    std = np.std(Xuq)
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
