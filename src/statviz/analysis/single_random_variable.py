import argparse
from sys import stdout
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import TextIO

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from statviz.analysis.helpers import *
from statviz.analysis.utils import *


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
            data.get("X", []), dtype=np.float128
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
        counts / counts.sum(), dtype=np.float128
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


def _calc_mgf_chunk(
    t_chunk: np.ndarray, Xuq: np.ndarray, P: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MGF and derivatives for a chunk of t values."""
    T, X_mesh = np.meshgrid(t_chunk, Xuq)
    P_mesh = P[:, np.newaxis]

    exp_tx = np.exp(T * X_mesh)

    MGF_chunk = np.sum(exp_tx * P_mesh, axis=0)
    MGF_prime_chunk = np.sum(X_mesh * exp_tx * P_mesh, axis=0)
    MGF_double_prime_chunk = np.sum(X_mesh**2 * exp_tx * P_mesh, axis=0)

    return MGF_chunk, MGF_prime_chunk, MGF_double_prime_chunk


def calc_mgf_deriv(
    X: np.ndarray, P: np.ndarray, t_max: float, n_chunks: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the Moment Generating Function (MGF) and its derivatives using parallel processing.

    Args:
        X: Input values array
        P: Probability/weights array
        t_max: Maximum t value for calculation
        n_chunks: Number of chunks to split the calculation into. Defaults to number of CPU cores.

    Returns:
        Tuple of (MGF, MGF_prime, MGF_double_prime) arrays
    """
    Xuq = np.unique(X)
    n = 100
    t_values = np.linspace(0, t_max, n)

    # Determine number of chunks based on CPU cores if not specified
    if n_chunks is None:
        n_chunks = cpu_count()

    # Split t_values into chunks
    t_chunks = np.array_split(t_values, n_chunks)

    # Create partial function with fixed arguments
    worker_func = partial(_calc_mgf_chunk, Xuq=Xuq, P=P)

    # Process chunks in parallel
    with Pool(processes=n_chunks) as pool:
        results = pool.map(worker_func, t_chunks)

    # Combine results from all chunks
    MGF = np.concatenate([r[0] for r in results])
    MGF_prime = np.concatenate([r[1] for r in results])
    MGF_double_prime = np.concatenate([r[2] for r in results])

    return MGF, MGF_prime, MGF_double_prime


def plot_mgf_deriv(
    MGF: np.ndarray,
    MGF_prime: np.ndarray,
    MGF_double_prime: np.ndarray,
    t_max: float,
) -> None:
    """Plots the Moment Generating Function (MGF) and its derivatives."""
    n = 100
    t_values = np.linspace(0, t_max, n)
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
