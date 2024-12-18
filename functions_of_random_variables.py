import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from mpl_toolkits.mplot3d import Axes3D


def read_input(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads the input file and returns sample space X and Y.
    File format:
        First line: number of points (n)
        Next n lines: two columns (sample space values for X and Y)
    """
    try:
        with open(filename, "r") as file:
            num_points = int(file.readline().strip())
            data = np.loadtxt(file)
            X = data[:, 0]  # Sample space X
            Y = data[:, 1]  # Sample space Y

        if len(X) != num_points:
            raise ValueError("Number of data points does not match header.")
        if data.shape[1] != 2:
            raise ValueError("Invalid data format. Expected 2 columns.")
        return X, Y
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def compute_joint_distribution(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the joint probability distribution of X and Y.
    """
    unique_X = np.unique(X)
    unique_Y = np.unique(Y)
    joint_prob = np.zeros((len(unique_X), len(unique_Y)))
    for x, y in zip(X, Y):
      x_idx = np.where(unique_X == x)[0][0]
      y_idx = np.where(unique_Y == y)[0][0]
      joint_prob[x_idx, y_idx] += 1
    joint_prob /= len(X)
    return unique_X, unique_Y, joint_prob

def compute_marginals(joint_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """
    Computes the marginal probability distributions from the joint probability distribution.
  """
  Px = np.sum(joint_prob, axis=1)
  Py = np.sum(joint_prob, axis=0)
  return Px, Py


def calculate_z_w(
    X: np.ndarray, Y: np.ndarray, z_func: Callable[[float], float], w_func: Callable[[float], float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates Z and W as functions of X and Y.
    """
    Z = np.array([z_func(x) for x in X])
    W = np.array([w_func(y) for y in Y])
    return Z, W


def plot_distribution(values: np.ndarray, probabilities: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    """Plots the probability distribution of a random variable."""
    unique_vals = np.unique(values)
    prob_for_unique = np.zeros(len(unique_vals))
    for i, val in enumerate(unique_vals):
      prob_for_unique[i] = np.sum(probabilities[values == val])

    plt.figure(figsize=(8, 6))
    plt.stem(unique_vals, prob_for_unique, basefmt=" ")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def plot_joint_distribution(Z: np.ndarray, W: np.ndarray, P: np.ndarray) -> None:
  """
  Plots the joint probability distribution of Z and W.
  """
  unique_pairs = np.unique(list(zip(Z,W)), axis=0)
  joint_probabilities = np.zeros(len(unique_pairs))

  for i, pair in enumerate(unique_pairs):
    z_val, w_val = pair
    joint_probabilities[i] = np.sum(P[(Z == z_val) & (W == w_val)])
  
  z_values = unique_pairs[:,0]
  w_values = unique_pairs[:,1]

  fig = plt.figure(figsize=(10,8))
  ax = fig.add_subplot(projection='3d')
  ax.bar3d(z_values, w_values, np.zeros_like(joint_probabilities), 1, 1, joint_probabilities, shade=True)
  ax.set_xlabel("Z")
  ax.set_ylabel("W")
  ax.set_zlabel("Probability")
  ax.set_title("Joint Probability Distribution of Z and W")
  plt.show(block=False)


def plot_distributions_3d(unique_X, unique_Y, joint_prob, Px, Py):
    """
    Plots the 3D joint probability distribution in one figure
    and the marginal distributions in a separate figure.
    """
    # --- Figure 1: 3D Joint Probability Distribution ---
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")

    # Create mesh grid for 3D bar plot
    X_mesh, Y_mesh = np.meshgrid(unique_Y, unique_X)
    xpos = X_mesh.ravel()
    ypos = Y_mesh.ravel()
    zpos = np.zeros_like(xpos)  # Starting z position

    dx = dy = 0.4  # Width of bars
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


def main():
    # Input file
    filename = "samples.txt"  # Default file name
    X, Y = read_input(filename)

    # Compute joint and marginal distributions
    unique_X, unique_Y, joint_prob = compute_joint_distribution(X, Y)
    Px, Py = compute_marginals(joint_prob)

    # Plot 3D Joint and Marginal Distributions
    plot_distributions_3d(unique_X, unique_Y, joint_prob, Px, Py)

     # Calculate covariance and correlation
    covariance = calculate_covariance(X, Y, joint_prob, unique_X, unique_Y)
    correlation = calculate_correlation(X, Y, joint_prob, unique_X, unique_Y)

    # Define the functions for Z and W
    def z_func(x: float) -> float:
        return 2 * x - 1

    def w_func(y: float) -> float:
        return 2 - 3 * y

    # Calculate Z and W
    Z, W = calculate_z_w(X, Y, z_func, w_func)
    
    P = np.ones(len(X))/len(X)

    # Plot probability distribution of Z
    plot_distribution(Z, P, "Probability Distribution of Z", "Z", "Probability")

    # Plot probability distribution of W
    plot_distribution(W, P, "Probability Distribution of W", "W", "Probability")

    # Plot joint probability distribution of Z and W
    plot_joint_distribution(Z, W, P)
    print("\n=== Results ===")
    print(f"Covariance: {covariance:.4f}")
    print(f"Correlation Coefficient: {correlation:.4f}")
    plt.show(block=True)


if __name__ == "__main__":
    main()
