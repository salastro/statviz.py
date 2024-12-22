import argparse
from sys import stdout
from typing import TextIO

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from statviz.analysis.helpers import *
from statviz.analysis.utils import *

N=100


def read_file_single(filename: str) -> np.ndarray:
    """
    Reads a MATLAB .mat file and returns sample space X and probabilities P.
    File format:
        The .mat file should contain two variables:
            - 'X': array of sample space values
            - 'P': array of corresponding probabilities
    """
    try:
        data = loadmat(filename)
        X = np.array(
            data.get("X", []), dtype=np.float64
        )  # Get 'X', default to empty array if not found

        if X.size == 0:
            raise ValueError("The .mat file must contain 'X' variable.")
        return X
    except Exception as e:
        print(f"Error reading the file: {e}")
        exit(1)


def calc_prob(X: np.ndarray) -> np.ndarray:
    """
    Calculates and normalizes probabilities of unique elements in the input array X.

    Parameters:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized probabilities corresponding to unique elements in X.
    """
    if len(X) == 0:
        return np.array([])  # Return an empty array for empty input

    _, counts = np.unique(X, return_counts=True)
    P = np.array(
        counts / counts.sum(), dtype=np.float64
    )  # Explicit normalization step
    return P


def plot_prob_cdf(X: np.ndarray, P: np.ndarray) -> None:
    """
    Plots the probability distribution (as a histogram) and cumulative distribution function (CDF)
    with automatic bin width calculation.
    """
    Xuq = np.unique(X)
    CDF = np.cumsum(P)

    # Calculate automatic bin width using Scott's Rule
    _, bins = np.histogram(X, bins="scott", density=True)

    plt.figure(figsize=(10, 6))

    # Plot the histogram
    plt.subplot(2, 1, 1)
    plt.hist(Xuq, bins=bins, weights=P, edgecolor="black", align="mid", rwidth=0.9)
    plt.title("Probability Distribution")
    plt.xlabel("Sample Space")
    plt.ylabel("Probability")
    plt.grid(True)

    # Plot the CDF
    plt.subplot(2, 1, 2)
    plt.step(Xuq, CDF, where="post", linewidth=2)
    plt.title("Cumulative Distribution Function (CDF)")
    plt.xlabel("Sample Space")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)


def calc_mgf_deriv(
    X: np.ndarray, P: np.ndarray, t_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the Moment Generating Function (MGF) and its derivatives."""
    Xuq = np.unique(X)
    t_values = np.linspace(0, t_max, N)
    MGF = np.array([np.sum(np.exp(t * Xuq) * P) for t in t_values])
    MGF_prime = np.array([np.sum(Xuq * np.exp(t * Xuq) * P) for t in t_values])
    MGF_double_prime = np.array([np.sum(Xuq**2 * np.exp(t * Xuq) * P) for t in t_values])
    return MGF, MGF_prime, MGF_double_prime


def plot_mgf_deriv(
    MGF: np.ndarray,
    MGF_prime: np.ndarray,
    MGF_double_prime: np.ndarray,
    t_max: float,
) -> None:
    """Plots the Moment Generating Function (MGF) and its derivatives."""
    t_values = np.linspace(0, t_max, N)
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t_values, MGF, linewidth=2)
    plt.title("Moment Generating Function (MGF)")
    plt.xlabel("t")
    plt.ylabel("M(t)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t_values, MGF_prime, linewidth=2)
    plt.title("First Derivative of MGF (M'(t))")
    plt.xlabel("t")
    plt.ylabel("M'(t)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t_values, MGF_double_prime, linewidth=2)
    plt.title("Second Derivative of MGF (M''(t))")
    plt.xlabel("t")
    plt.ylabel("M''(t)")
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)


def handle_args():
    parser = argparse.ArgumentParser(
        description="Analyze a single random variable and compute statistics from a file."
    )
    parser.add_argument("-f", "--filename", type=str, help="Name of the input file.")
    parser.add_argument("-t", "--t_max", type=float, help="Maximum value of t.")
    return parser.parse_args()


def main(stream: TextIO = stdout):
    # Input file
    args = handle_args()
    filename = args.filename
    t_max = args.t_max
    X = read_file_single(filename)
    P = calc_prob(X)

    # Step 1: Plot Probability Distribution and CDF
    plot_prob_cdf(X, P)

    # Step 2: Calculate and Display Statistical Measures
    mean_X, var_X, third_moment = calc_stats(X, P)
    stream.write("\n=== Results ===\n")
    stream.write("\nStatistical Measures:\n")
    stream.write(f"Mean = {mean_X:.4f}\n")
    stream.write(f"Variance = {var_X:.4f}\n")
    stream.write(f"Third Moment = {third_moment:.4f}\n")

    # Step 3: Plot MGF and Derivatives
    MGF, MGF_prime, MGF_double_prime = calc_mgf_deriv(X, P, t_max)
    MGF_0, MGF_prime_0, MGF_double_prime_0 = MGF[0], MGF_prime[0], MGF_double_prime[0]
    stream.write("\nValues at t = 0:\n")
    stream.write(f"M(0) = {MGF_0:.4f}\n")
    stream.write(f"M'(0) = {MGF_prime_0:.4f} (Mean)\n")
    stream.write(f"M''(0) = {MGF_double_prime_0:.4f}\n")
    plot_mgf_deriv(MGF, MGF_prime, MGF_double_prime, t_max)
    plt.show(block=True)  # Keep the program alive until plots closed


if __name__ == "__main__":
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        exit(0)
