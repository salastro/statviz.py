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

    joint_counts = np.zeros((len(Xuq), len(Yuq)), dtype=np.float64)
    np.add.at(joint_counts, (X_inv, Y_inv), 1)

    joint_prob = joint_counts / len(X)
    return joint_prob


# Function to compute marginal distributions
def calc_marg(joint_prob):
    Px = np.sum(joint_prob, axis=1)
    Py = np.sum(joint_prob, axis=0)
    return Px, Py


# Function to calculate covariance
def calc_cov(X, Y, joint_prob):
    Xuq, Yuq = np.unique(X), np.unique(Y)
    Ex = np.sum(Xuq * np.sum(joint_prob, axis=1))
    Ey = np.sum(Yuq * np.sum(joint_prob, axis=0))
    Exy = 0

    for i, x in enumerate(Xuq):
        for j, y in enumerate(Yuq):
            Exy += x * y * joint_prob[i, j]

    covariance = Exy - (Ex * Ey)
    return covariance


# Function to calculate correlation coefficient
def calc_cor(X, Y, joint_prob):
    Xuq, Yuq = np.unique(X), np.unique(Y)
    Ex = np.sum(Xuq * np.sum(joint_prob, axis=1))
    Ey = np.sum(Yuq * np.sum(joint_prob, axis=0))
    Ex2 = np.sum((Xuq**2) * np.sum(joint_prob, axis=1))
    Ey2 = np.sum((Yuq**2) * np.sum(joint_prob, axis=0))

    Var_X = Ex2 - Ex**2
    Var_Y = Ey2 - Ey**2

    covariance = calc_cov(X, Y, joint_prob)
    correlation = covariance / np.sqrt(Var_X * Var_Y)
    return correlation


# Function to plot joint and marginal distributions
def plot_marg_prob(X, Y, Px, Py):
    """
    Plots the 3D joint probability distribution in one figure
    and the marginal distributions in a separate figure.

    Args:
        X (array): Values of random variable X.
        Y (array): Values of random variable Y.
        joint_prob (2D array): Joint probability distribution.
        Px (array): Marginal probability distribution for X.
        Py (array): Marginal probability distribution for Y.
    """
    Xuq, Yuq = np.unique(X), np.unique(Y)

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

    # Marginal Distribution for X
    bin_width_X = calc_bin_w(X)
    bin_edges_X = np.arange(
        Xuq[0] - bin_width_X / 2, Xuq[-1] + bin_width_X, bin_width_X
    )
    ax2.hist(
        Xuq,
        bins=bin_edges_X,
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

    # Marginal Distribution for Y
    bin_width_Y = calc_bin_w(Y)
    bin_edges_Y = np.arange(
        Yuq[0] - bin_width_Y / 2, Yuq[-1] + bin_width_Y, bin_width_Y
    )
    ax3.hist(
        Yuq,
        bins=bin_edges_Y,
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


def plot_joint_prob(Z: np.ndarray, W: np.ndarray, P: np.ndarray) -> None:
    """
    Plots the joint probability distribution of Z and W with automatic bin widths for the bars
    and adds probabilities within each bin.
    """
    # Calculate automatic width for Z (Scott's Rule)
    Zuq, Wuq = np.unique(Z), np.unique(W)

    # Calculate bin widths
    width_z = calc_bin_w(Zuq)
    width_w = calc_bin_w(Wuq)

    # Calculate bin edges
    z_min = np.min(Zuq)
    z_max = np.max(Zuq)
    w_min = np.min(Wuq)
    w_max = np.max(Wuq)
    z_edges = np.arange(z_min - width_z / 2, z_max + width_z / 2, width_z)
    w_edges = np.arange(w_min - width_w / 2, w_max + width_w / 2, width_w)

    # Use scipy.stats.binned_statistic_2d for binning and averaging probabilities
    joint_probabilities, z_edges, w_edges, binnumbers = binned_statistic_2d(
        Zuq, Wuq, P, statistic="mean", bins=[z_edges, w_edges]
    )

    # Get centers of the bins
    z_bin_centers = (z_edges[:-1] + z_edges[1:]) / 2
    w_bin_centers = (w_edges[:-1] + w_edges[1:]) / 2

    # Generate Z and W for plotting
    z_mesh, w_mesh = np.meshgrid(z_bin_centers, w_bin_centers)
    z_values = z_mesh.flatten()
    w_values = w_mesh.flatten()
    joint_probabilities = joint_probabilities.flatten()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.bar3d(
        z_values,
        w_values,
        np.zeros_like(joint_probabilities),
        width_z,
        width_w,
        joint_probabilities,
        shade=True,
    )
    ax.set_xlabel("Z")
    ax.set_ylabel("W")
    ax.set_zlabel("Probability")
    ax.set_title("Joint Probability Distribution of Z and W")
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
    joint_prob = calc_joint_prob(X, Y)
    Px, Py = calc_marg(joint_prob)

    # Plot 3D Joint and Marginal Distributions
    plot_marg_prob(X, Y, Px, Py)
    plot_joint_prob(X, Y, joint_prob)

    # Calculate covariance and correlation
    cov = calc_cov(X, Y, joint_prob)
    cor = calc_cor(X, Y, joint_prob)
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
