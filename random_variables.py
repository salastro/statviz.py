import argparse

import matplotlib.pyplot as plt
import numpy as np


def read_input(filename: str) -> np.ndarray:
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
    unique_X = np.unique(X)
    P = np.zeros_like(unique_X, dtype=float)
    for x in X:
        x_idx = np.where(unique_X == x)[0]
        P[x_idx] += 1
    P /= len(X)
    return P


def plot_prob_cdf(X: np.ndarray, P: np.ndarray) -> None:
    """
    Plots the probability distribution (as a histogram) and cumulative distribution function (CDF)
    with automatic bin width calculation.
    """
    CDF = np.cumsum(P)

    # Calculate automatic bin width using Scott's Rule
    n = len(X)
    std_x = np.std(X)
    if std_x == 0:
        bin_width = 1.0
    else:
        bin_width = 3.5 * std_x / (n ** (1 / 3))

    # Generate histogram bin edges based on X and automatic bin_width
    bin_edges = np.arange(X[0] - bin_width / 2, X[-1] + bin_width, bin_width)

    plt.figure(figsize=(10, 6))

    # Plot the histogram
    plt.subplot(2, 1, 1)
    plt.hist(X, bins=bin_edges, weights=P, edgecolor="black", align="mid", rwidth=0.9)
    plt.title("Probability Distribution")
    plt.xlabel("Sample Space")
    plt.ylabel("Probability")
    plt.grid(True)

    # Plot the CDF
    plt.subplot(2, 1, 2)
    plt.step(X, CDF, where="post", linewidth=2)
    plt.title("Cumulative Distribution Function (CDF)")
    plt.xlabel("Sample Space")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)


def calc_stats(X: np.ndarray, P: np.ndarray) -> tuple[float, float, float]:
    """Calculates and displays mean, variance, and third moment."""
    mean_X = np.sum(X * P)
    var_X = np.sum((X - mean_X) ** 2 * P)
    third_moment = np.sum((X - mean_X) ** 3 * P)
    return mean_X, var_X, third_moment


def calc_mgf_deriv(
    X: np.ndarray, P: np.ndarray, t_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the Moment Generating Function (MGF) and its derivatives."""
    t_values = np.linspace(0, t_max, 1000)
    MGF = np.array([np.sum(np.exp(t * X) * P) for t in t_values])
    MGF_prime = np.array([np.sum(X * np.exp(t * X) * P) for t in t_values])
    MGF_double_prime = np.array([np.sum(X**2 * np.exp(t * X) * P) for t in t_values])
    return MGF, MGF_prime, MGF_double_prime, t_values


def plot_mgf_deriv(
    MGF: np.ndarray,
    MGF_prime: np.ndarray,
    MGF_double_prime: np.ndarray,
    t_values: np.ndarray,
) -> None:
    """Plots the Moment Generating Function (MGF) and its derivatives."""
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
    X = read_input(filename)
    unique_X = np.unique(X)
    P = calc_prop(X)

    # Step 1: Plot Probability Distribution and CDF
    plot_prob_cdf(unique_X, P)

    # Step 2: Calculate and Display Statistical Measures
    mean_X, var_X, third_moment = calc_stats(unique_X, P)
    print("\nStatistical Measures:")
    print(f"Mean = {mean_X:.4f}")
    print(f"Variance = {var_X:.4f}")
    print(f"Third Moment = {third_moment:.4f}")

    # Step 3: Plot MGF and Derivatives
    t_max = 5  # Define the range of t values
    MGF, MGF_prime, MGF_double_prime, t_values = calc_mgf_deriv(unique_X, P, t_max)
    MGF_0, MGF_prime_0, MGF_double_prime_0 = MGF[0], MGF_prime[0], MGF_double_prime[0]
    print("\nValues at t = 0:")
    print(f"M(0) = {MGF_0:.4f}")
    print(f"M'(0) = {MGF_prime_0:.4f} (Mean)")
    print(f"M''(0) = {MGF_double_prime_0:.4f}")
    plot_mgf_deriv(MGF, MGF_prime, MGF_double_prime, t_values)
    plt.show(block=True)  # Keep the program alive until plots closed


if __name__ == "__main__":
    main()
