import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import argparse


# Function to generate a sample file
def generate_sample_file(filename: str, num_samples: int) -> None:
    """
    Generates a sample file with random pairs of (X, Y).

    Args:
        filename (str): Name of the output file.
        num_samples (int): Number of sample pairs to generate.
    """
    X = np.random.randint(1, 10, size=num_samples)
    Y = np.random.randint(1, 10, size=num_samples)

    with open(filename, "w") as file:
        file.write(f"{num_samples}\n")
        for x, y in zip(X, Y):
            file.write(f"{x} {y}\n")

    print(f"Sample file '{filename}' generated successfully with {num_samples} pairs.")


# Function to read data from a file
def read_file(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads sample pairs (X, Y) from a file.

    Args:
        filename (str): Input file name.

    Returns:
        X: List of X values.
        Y: List of Y values.
    """
    try:
        with open(filename, "r") as file:
            num_samples = int(file.readline().strip())
            data = np.loadtxt(file)
            X = data[:, 0]
            Y = data[:, 1]

        if len(data) != num_samples:
            raise ValueError("Number of data points does not match header.")

        if data.shape[1] != 2:
            raise ValueError("Invalid data format. Expected 2 columns.")

        return X, Y
    except Exception as e:
        raise e


# Function to compute joint probability distribution
def compute_joint_distribution(X, Y):
    unique_X, unique_Y = np.unique(X), np.unique(Y)
    joint_prob = np.zeros((len(unique_X), len(unique_Y)))

    for x, y in zip(X, Y):
        x_idx = np.where(unique_X == x)[0][0]
        y_idx = np.where(unique_Y == y)[0][0]
        joint_prob[x_idx, y_idx] += 1

    joint_prob /= len(X)
    return unique_X, unique_Y, joint_prob


# Function to compute marginal distributions
def compute_marginals(joint_prob):
    Px = np.sum(joint_prob, axis=1)
    Py = np.sum(joint_prob, axis=0)
    return Px, Py


# Function to calculate covariance
def calculate_covariance(X, Y, joint_prob, unique_X, unique_Y):
    Ex = np.sum(unique_X * np.sum(joint_prob, axis=1))
    Ey = np.sum(unique_Y * np.sum(joint_prob, axis=0))
    Exy = 0

    for i, x in enumerate(unique_X):
        for j, y in enumerate(unique_Y):
            Exy += x * y * joint_prob[i, j]

    covariance = Exy - (Ex * Ey)
    return covariance


# Function to calculate correlation coefficient
def calculate_correlation(X, Y, joint_prob, unique_X, unique_Y):
    Ex = np.sum(unique_X * np.sum(joint_prob, axis=1))
    Ey = np.sum(unique_Y * np.sum(joint_prob, axis=0))
    Ex2 = np.sum((unique_X**2) * np.sum(joint_prob, axis=1))
    Ey2 = np.sum((unique_Y**2) * np.sum(joint_prob, axis=0))

    Var_X = Ex2 - Ex**2
    Var_Y = Ey2 - Ey**2

    covariance = calculate_covariance(X, Y, joint_prob, unique_X, unique_Y)
    correlation = covariance / np.sqrt(Var_X * Var_Y)
    return correlation


# Function to plot joint and marginal distributions
def plot_marginal_distributions(unique_X, unique_Y, joint_prob, Px, Py, bin_width_X, bin_width_Y):
    """
    Plots the 3D joint probability distribution in one figure
    and the marginal distributions in a separate figure.

    Args:
        unique_X (array): Unique values of random variable X.
        unique_Y (array): Unique values of random variable Y.
        joint_prob (2D array): Joint probability distribution.
        Px (array): Marginal probability distribution for X.
        Py (array): Marginal probability distribution for Y.
    """
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

    # Marginal Distribution for X
    bin_edges_X = np.arange(
        unique_X[0] - bin_width_X / 2, unique_X[-1] + bin_width_X, bin_width_X
    )
    ax2.hist(
        unique_X,
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
    bin_edges_Y = np.arange(
        unique_Y[0] - bin_width_Y / 2, unique_Y[-1] + bin_width_Y, bin_width_Y
    )
    ax3.hist(
        unique_Y,
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


def handle_args():
    parser = argparse.ArgumentParser(description="Analyze a random variable from a file.")
    parser.add_argument("bin_width_x", type=float, help="Width of bins for the probability distribution for X.")
    parser.add_argument("bin_width_y", type=float, help="Width of bins for the probability distribution for Y.")
    parser.add_argument("filename", type=str, help="Name of the input file.")
    return parser.parse_args()


# Main program
def main():

    args = handle_args()
    filename = args.filename
    bin_width_X = args.bin_width_x
    bin_width_Y = args.bin_width_y

    X, Y = read_file(filename)
    if X is None or Y is None:
        return

    # Compute distributions
    unique_X, unique_Y, joint_prob = compute_joint_distribution(X, Y)
    Px, Py = compute_marginals(joint_prob)

    # Plot 3D Joint and Marginal Distributions
    plot_marginal_distributions(unique_X, unique_Y, joint_prob, Px, Py, bin_width_X, bin_width_Y)

    # Calculate covariance and correlation
    covariance = calculate_covariance(X, Y, joint_prob, unique_X, unique_Y)
    correlation = calculate_correlation(X, Y, joint_prob, unique_X, unique_Y)

    # Display results
    print("\n=== Results ===")
    print(f"Covariance: {covariance:.4f}")
    print(f"Correlation Coefficient: {correlation:.4f}")

    # Show plots
    plt.show(block=True)


if __name__ == "__main__":
    main()
