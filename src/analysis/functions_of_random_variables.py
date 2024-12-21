import argparse
from sys import stdout
from typing import Callable, TextIO

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, sympify
from sympy.utilities.lambdify import lambdify

from .helpers import *
from .utils import *


def matheval(expr: str, variables: list) -> Callable:
    """
    Safely evaluate a mathematical expression provided as a string using sympy.
    expr: String representation of the expression.
    valid_symbols: List of valid symbols.
    """
    try:
        locals = {name: symbols(name) for name in variables}
        expr = sympify(expr, locals=locals)
        func = lambdify(variables, expr, modules="numpy")
        return func
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {e}")


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


def handle_args():
    parser = argparse.ArgumentParser(
        description="Analyzes functions of random variables and computes statistics from a file."
    )
    parser.add_argument("-f", "--filename", type=str, help="Name of the input file.")
    parser.add_argument(
        "-Z", "--Z_func", nargs="?", type=str, default="2*x-1", help="Function for Z."
    )
    parser.add_argument(
        "-W", "--W_func", nargs="?", type=str, default="2-3*y", help="Function for W."
    )
    args = parser.parse_args()
    return args


def main(stream: TextIO = stdout):
    args = handle_args()
    filename = args.filename
    X, Y = read_file_joint(filename)

    # Define functions for Z and W
    z_func = matheval(args.Z_func, ["x", "y"])
    w_func = matheval(args.W_func, ["x", "y"])

    # Calculate Z and W
    Z = z_func(X, Y)
    W = w_func(X, Y)

    # Compute joint and marginal distributions of Z and W
    Pzw = calc_joint_prob(Z, W)
    Pz, Pw = calc_marg_prob(Pzw)

    # Compute covariance and correlation coefficient
    cov = calc_cov(Z, W, Pzw)
    cor = calc_cor(Z, W, Pzw)

    stream.write("\n=== Results ===\n")
    stream.write(f"Covariance: {cov:.4f}\n")
    stream.write(f"Correlation Coefficient: {cor:.4f}\n")
    # Compute statistics
    mean_Z, var_Z, third_moment_Z = calc_stats(Z, Pz)
    mean_W, var_W, third_moment_W = calc_stats(W, Pw)

    # Display results
    stream.write("\nStatistical Measures for Z:\n")
    stream.write(f"Mean = {mean_Z:.4f}\n")
    stream.write(f"Variance = {var_Z:.4f}\n")
    stream.write(f"Third Moment = {third_moment_Z:.4f}\n")
    stream.write("\nStatistical Measures for W:\n")
    stream.write(f"Mean = {mean_W:.4f}\n")
    stream.write(f"Variance = {var_W:.4f}\n")
    stream.write(f"Third Moment = {third_moment_W:.4f}\n")
    stream.write("\n")
    stream.flush()

    # Plot marginal distributions of Z and W
    plot_marg_prob(Z, W, Pz, Pw)

    # Plot joint probability distribution of Z and W
    plot_joint_prob(Z, W, Pzw)

    plt.show(block=True)


if __name__ == "__main__":
    main()
