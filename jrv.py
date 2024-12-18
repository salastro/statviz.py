import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


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
def plot_distributions_3d(unique_X, unique_Y, joint_prob, Px, Py):
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
    # --- Figure 1: 3D Joint Probability Distribution ---
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")

    # Create mesh grid for 3D bar plot
    X_mesh, Y_mesh = np.meshgrid(unique_Y, unique_X)
    xpos = X_mesh.ravel()
    ypos = Y_mesh.ravel()
    zpos = np.zeros_like(xpos)  # Starting z position

    dx = dy = 1  # Width of bars
    dz = joint_prob.ravel()  # Heights (probability values)

    # Create the 3D bar plot
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
    ax1.set_title("3D Joint Probability Distribution")
    ax1.set_xlabel("Y")
    ax1.set_ylabel("X")
    ax1.set_zlabel("P(X, Y)")

    plt.tight_layout()
    plt.show(block=False)

    # --- Figure 2: Marginal Distributions ---
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

    # Marginal Distribution for X
    ax2.bar(unique_X, Px, color="blue", alpha=0.7)
    ax2.set_title("Marginal Distribution P(X)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Probability")
    ax2.grid(True)

    # Marginal Distribution for Y
    ax3.bar(unique_Y, Py, color="orange", alpha=0.7)
    ax3.set_title("Marginal Distribution P(Y)")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Probability")
    ax3.grid(True)

    plt.tight_layout()
    plt.show(block=False)


# Main program
def main():
    print("=== Random Variable Analysis ===")
    print("1. Generate a sample file with random data")
    print("2. Analyze data from an input file")
    choice = input("Enter your choice (1/2): ").strip()

    filename = "samples.txt"
    if choice == "1":
        num_samples = int(
            input("Enter the number of sample pairs to generate: ").strip()
        )
        generate_sample_file(filename, num_samples)

    elif choice == "2":
        pass

    else:
        print("Invalid choice. Exiting.")

    X, Y = read_file(filename)
    if X is None or Y is None:
        return

    # Compute distributions
    unique_X, unique_Y, joint_prob = compute_joint_distribution(X, Y)
    Px, Py = compute_marginals(joint_prob)

    # Plot 3D Joint and Marginal Distributions
    plot_distributions_3d(unique_X, unique_Y, joint_prob, Px, Py)

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
