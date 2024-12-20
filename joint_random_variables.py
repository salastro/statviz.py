import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

from helpers import *
from utils import *


def read_file(filename: str) -> tuple[np.ndarray, np.ndarray]:
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
    if len(X) == 0 or len(Y) == 0:
        return np.array([], dtype=np.float64)

    # Convert inputs to float64 if they aren't already
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    Xuq, X_inv = np.unique(X, return_inverse=True)
    Yuq, Y_inv = np.unique(Y, return_inverse=True)

    Nxy = np.zeros((len(Xuq), len(Yuq)), dtype=np.float64)
    np.add.at(Nxy, (X_inv, Y_inv), 1)

    Pxy = Nxy / len(X)
    return Pxy


def calc_marg(Pxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def plot_marg_prob(
    X: np.ndarray, Y: np.ndarray, Px: np.ndarray, Py: np.ndarray
) -> None:
    """
    Plot marginal distributions using float64 arrays.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Px = Px.astype(np.float64)
    Py = Py.astype(np.float64)

    Xuq = np.unique(X)
    Yuq = np.unique(Y)

    _, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

    _, bins_X = np.histogram(X, bins="scott", density=True)
    _, bins_Y = np.histogram(Y, bins="scott", density=True)

    ax2.hist(
        Xuq,
        bins=bins_X,
        weights=Px,
        edgecolor="black",
        align="mid",
        rwidth=0.9,
        color="blue",
        alpha=0.7,
    )
    ax2.set_title("Marginal Distribution P(X)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Probability")
    ax2.grid(True)

    ax3.hist(
        Yuq,
        bins=bins_Y,
        weights=Py,
        edgecolor="black",
        align="mid",
        rwidth=0.9,
        color="orange",
        alpha=0.7,
    )
    ax3.set_title("Marginal Distribution P(Y)")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Probability")
    ax3.grid(True)

    plt.tight_layout()
    plt.show(block=False)


def plot_joint_prob(X: np.ndarray, Y: np.ndarray, Pxy: np.ndarray) -> None:
    """
    Plot joint probability distribution using float64 arrays.
    Bars are sized according to the bin widths for better visualization.
    """
    _, bins_X = np.histogram(X, bins="scott", density=True)
    _, bins_Y = np.histogram(Y, bins="scott", density=True)

    H, xedges, yedges = np.histogram2d(X, Y, density=True, bins=(bins_X, bins_Y))

    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")

    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    dx = np.diff(xedges)[0]
    dy = np.diff(yedges)[0]
    dz = H.ravel()

    # Filter out zero probabilities
    mask = dz > 0
    xpos = xpos[mask]
    ypos = ypos[mask]
    zpos = zpos[mask]
    dz = dz[mask]

    # Normalize the probabilities to ensure they sum to 1
    dz_sum = np.sum(dz)
    dz /= dz_sum

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Probability")
    ax.set_title("Joint Probability Distribution P(X, Y)")

    plt.tight_layout()
    plt.show(block=False)


def handle_args():
    parser = argparse.ArgumentParser(
        description="Analyze a random variable from a file."
    )
    parser.add_argument("filename", type=str, help="Name of the input file.")
    return parser.parse_args()


# Main program
def main():

    args = handle_args()
    filename = args.filename

    X, Y = read_file(filename)
    if X is None or Y is None:
        return

    # Compute distributions
    Pxy = calc_joint_prob(X, Y)
    Px, Py = calc_marg(Pxy)

    # Plot 3D Joint and Marginal Distributions
    plot_marg_prob(X, Y, Px, Py)
    plot_joint_prob(X, Y, Pxy)

    # Calculate covariance and correlation
    cov = calc_cov(X, Y, Pxy)
    cor = calc_cor(X, Y, Pxy)
    mean_X, var_X, third_moment_X = calc_stats(np.unique(X), Px)
    mean_Y, var_Y, third_moment_Y = calc_stats(np.unique(Y), Py)

    # Display results
    print("\n=== Results ===")
    print(f"Covariance: {cov:.4f}")
    print(f"Correlation Coefficient: {cor:.4f}")
    print("\nStatistical Measures for X:")
    print(f"Mean = {mean_X:.4f}")
    print(f"Variance = {var_X:.4f}")
    print(f"Third Moment = {third_moment_X:.4f}")
    print("\nStatistical Measures for Y:")
    print(f"Mean = {mean_Y:.4f}")
    print(f"Variance = {var_Y:.4f}")
    print(f"Third Moment = {third_moment_Y:.4f}")

    # Show plots
    plt.show(block=True)


if __name__ == "__main__":
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        exit(0)
