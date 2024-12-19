import argparse

import matplotlib.pyplot as plt
import numpy as np

from helpers import *
from utils import *


def read_file(filename: str) -> np.ndarray:
    """
    Reads the input file and returns sample space X and probabilities P.
    File format:
        First line: number of points (n)
        Next n lines: two columns (sample space values and probabilities)
    """
    try:
        with open(filename, "r") as file:
            num_points = int(file.readline().strip())
            data = np.loadtxt(file)
            X = data[:]  # Sample space

        if len(data) != num_points:
            raise ValueError("Number of data points does not match header.")

        return X
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def calc_prop(X: np.ndarray) -> np.ndarray:
    """Calculates the probability distribution."""
    Xuq = np.unique(X)
    P = np.zeros_like(Xuq, dtype=float)
    for x in X:
        x_idx = np.where(Xuq == x)[0]
        P[x_idx] += 1
    P /= len(X)
    return P


def plot_prob_cdf(X: np.ndarray, P: np.ndarray) -> None:
    """
    Plots the probability distribution (as a histogram) and cumulative distribution function (CDF)
    with automatic bin width calculation.
    """
    Xuq = np.unique(X)
    CDF = np.cumsum(P)

    # Calculate automatic bin width using Scott's Rule
    bin_width = calc_bin_w(Xuq)

    # Generate histogram bin edges based on X and automatic bin_width
    bin_edges = np.arange(Xuq[0] - bin_width / 2, Xuq[-1] + bin_width, bin_width)

    plt.figure(figsize=(10, 6))

    # Plot the histogram
    plt.subplot(2, 1, 1)
    plt.hist(Xuq, bins=bin_edges, weights=P, edgecolor="black", align="mid", rwidth=0.9)
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
    t_values = np.linspace(0, t_max, 1000)
    MGF = np.array([np.sum(np.exp(t * Xuq) * P) for t in t_values])
    MGF_prime = np.array([np.sum(Xuq * np.exp(t * Xuq) * P) for t in t_values])
    MGF_double_prime = np.array(
        [np.sum(Xuq**2 * np.exp(t * Xuq) * P) for t in t_values]
    )
    return MGF, MGF_prime, MGF_double_prime


def plot_mgf_deriv(
    MGF: np.ndarray,
    MGF_prime: np.ndarray,
    MGF_double_prime: np.ndarray,
    t_max: float,
) -> None:
    """Plots the Moment Generating Function (MGF) and its derivatives."""
    t_values = np.linspace(0, t_max, 1000)
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
        description="Analyze a random variable from a file."
    )
    parser.add_argument("filename", type=str, help="Name of the input file.")
    return parser.parse_args()


def main():
    # Input file
    args = handle_args()
    filename = args.filename
    X = read_file(filename)
    P = calc_prop(X)

    # Step 1: Plot Probability Distribution and CDF
    plot_prob_cdf(X, P)

    # Step 2: Calculate and Display Statistical Measures
    mean_X, var_X, third_moment = calc_stats(X, P)
    print("\n=== Results ===")
    print("\nStatistical Measures:")
    print(f"Mean = {mean_X:.4f}")
    print(f"Variance = {var_X:.4f}")
    print(f"Third Moment = {third_moment:.4f}")

    # Step 3: Plot MGF and Derivatives
    t_max = get_positive_float("Enter the maximum value of t: ")
    MGF, MGF_prime, MGF_double_prime = calc_mgf_deriv(X, P, t_max)
    MGF_0, MGF_prime_0, MGF_double_prime_0 = MGF[0], MGF_prime[0], MGF_double_prime[0]
    print("\nValues at t = 0:")
    print(f"M(0) = {MGF_0:.4f}")
    print(f"M'(0) = {MGF_prime_0:.4f} (Mean)")
    print(f"M''(0) = {MGF_double_prime_0:.4f}")
    plot_mgf_deriv(MGF, MGF_prime, MGF_double_prime, t_max)
    plt.show(block=True)  # Keep the program alive until plots closed


if __name__ == "__main__":
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        exit(0)
