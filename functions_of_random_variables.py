import argparse
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from helpers import *
from utils import *


def plot_marg_prob(
    Z: np.ndarray, W: np.ndarray, Pz: np.ndarray, Pw: np.ndarray
) -> None:
    """
    Plot marginal distributions using float64 arrays.
    """
    Z = Z.astype(np.float64)
    W = W.astype(np.float64)
    Pz = Pz.astype(np.float64)
    Pw = Pw.astype(np.float64)

    Zuq = np.unique(Z)
    Wuq = np.unique(W)

    _, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

    _, bins_Z = np.histogram(Z, bins="scott", density=True)
    _, bins_W = np.histogram(W, bins="scott", density=True)

    ax2.hist(
        Zuq,
        bins=bins_Z,
        weights=Pz,
        edgecolor="black",
        align="mid",
        rwidth=0.9,
        color="blue",
        alpha=0.7,
    )
    ax2.set_title("Marginal Distribution P(Z)")
    ax2.set_xlabel("Z")
    ax2.set_ylabel("Probability")
    ax2.grid(True)

    ax3.hist(
        Wuq,
        bins=bins_W,
        weights=Pw,
        edgecolor="black",
        align="mid",
        rwidth=0.9,
        color="orange",
        alpha=0.7,
    )
    ax3.set_title("Marginal Distribution P(W)")
    ax3.set_xlabel("W")
    ax3.set_ylabel("Probability")
    ax3.grid(True)

    plt.tight_layout()
    plt.show(block=False)


def plot_joint_prob(Z: np.ndarray, W: np.ndarray, Pxy: np.ndarray) -> None:
    """
    Plot joint probability distribution using float64 arrays.
    Bars are sized according to the bin widths for better visualization.
    """
    _, bins_Z = np.histogram(Z, bins="scott", density=True)
    _, bins_W = np.histogram(W, bins="scott", density=True)

    hist, wedges, zedges = np.histogram2d(Z, W, density=True, bins=(bins_Z, bins_W))

    wpos, zpos = np.meshgrid(wedges[:-1], zedges[:-1], indexing="ij")

    wpos = wpos.ravel()
    zpos = zpos.ravel()
    hpos = np.zeros_like(wpos)

    dz = np.diff(wedges)[0]
    dw = np.diff(zedges)[0]
    dh = hist.ravel()

    # Filter out zero probabilities
    mask = dh > 0
    wpos = wpos[mask]
    zpos = zpos[mask]
    hpos = hpos[mask]
    dh = dh[mask]

    # Normalize the probabilities to ensure they sum to 1
    dz_sum = np.sum(dh)
    dh /= dz_sum

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(wpos, zpos, hpos, dz, dw, dh)
    ax.set_xlabel("Z")
    ax.set_ylabel("W")
    ax.set_zlabel("Probability")
    ax.set_title("Joint Probability Distribution P(Z, W)")

    plt.tight_layout()
    plt.show(block=False)


def calc_func_of_rv(
    X: np.ndarray,
    f: Callable[[float], float],
) -> np.ndarray:
    F = np.array([f(x) for x in X])
    return F


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Name of the input file.")
    args = parser.parse_args()
    return args


def main():
    args = handle_args()
    filename = args.filename
    X, Y = read_file_joint(filename)

    # Define the functions for Z and W
    def z_func(x: float) -> float:
        return 2 * x - 1

    def w_func(y: float) -> float:
        return 2 - 3 * y

    # Calculate Z and W
    Z = calc_func_of_rv(X, z_func)
    W = calc_func_of_rv(Y, w_func)

    # Compute joint and marginal distributions of Z and W
    Pzw = calc_joint_prob(Z, W)
    Pz, Pw = calc_marg_prob(Pzw)

    # Plot marginal distributions of Z and W
    plot_marg_prob(Z, W, Pz, Pw)

    # Compute statistics
    mean_Z, var_Z, third_moment_Z = calc_stats(Z, Pz)
    mean_W, var_W, third_moment_W = calc_stats(W, Pw)

    # Compute covariance and correlation coefficient
    cov = calc_cov(Z, W, Pzw)
    cor = calc_cor(Z, W, Pzw)

    # Plot joint probability distribution of Z and W
    plot_joint_prob(Z, W, Pzw)

    # Display results
    print("\n=== Results ===")
    print(f"Covariance: {cov:.4f}")
    print(f"Correlation Coefficient: {cor:.4f}")
    print("\nStatistical Measures for Z:")
    print(f"Mean = {mean_Z:.4f}")
    print(f"Variance = {var_Z:.4f}")
    print(f"Third Moment = {third_moment_Z:.4f}")
    print("\nStatistical Measures for W:")
    print(f"Mean = {mean_W:.4f}")
    print(f"Variance = {var_W:.4f}")
    print(f"Third Moment = {third_moment_W:.4f}")


    plt.show(block=True)


if __name__ == "__main__":
    main()
