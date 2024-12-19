import argparse

import numpy as np
from scipy.io import savemat

from helpers import *
from utils import *


def uniform_distribution(samples, a, b):
    return np.random.uniform(a, b, samples)


def normal_distribution(samples, mean, std_dev):
    return np.random.normal(mean, std_dev, samples)


def binomial_distribution(samples, n, p):
    return np.random.binomial(n, p, samples)


def poisson_distribution(samples, lam):
    return np.random.poisson(lam, samples)


def exponential_distribution(samples, scale):
    return np.random.exponential(scale, samples)


def gamma_distribution(samples, shape, scale):
    return np.random.gamma(shape, scale, samples)


def save_file(data, filename="generated_data.mat"):
    """
    Saves a 1D array to a MATLAB .mat file.
    The data will be stored in a variable named 'data'.
    """
    try:
        savemat(filename, {"X": np.array(data)})
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving the file: {e}")


def save_file_joint(X, Y, filename="generated_data.mat"):
    """
    Saves two 1D arrays (X and Y) to a MATLAB .mat file.
    The data will be stored in variable named 'XY'.
    """
    try:
        savemat(filename, {"XY": np.array([X, Y])})
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving the file: {e}")


def generate_joint_rv(samples, filename="generated_data.txt"):
    print("1. X ~ Uniform Distribution (U(a, b))")
    print("2. X ~ Normal Distribution (N(mean, std_dev))")
    print("3. X ~ Binomial Distribution (Bin(n, p))")
    print("4. X ~ Poisson Distribution (Poisson(lambda))")
    print("5. X ~ Exponential Distribution (Exp(scale))")
    print("6. X ~ Gamma Distribution (Gamma(shape, scale))")

    X_choice = input("Enter your choice (1-6): ").strip()

    if (X_choice > "6") or (X_choice < "1"):
        print("Invalid choice.")
        exit(1)

    match X_choice:
        case "1":
            a = float(input("Enter lower bound (a): ").strip())
            b = float(input("Enter upper bound (b): ").strip())
            X = uniform_distribution(samples, a, b)

        case "2":
            mean = float(input("Enter mean: ").strip())
            std_dev = get_positive_float("Enter standard deviation: ")
            X = normal_distribution(samples, mean, std_dev)

        case "3":
            n = get_positive_integer("Enter number of trials (n): ")
            p = get_probability("Enter probability of success (p): ")
            X = binomial_distribution(samples, n, p)

        case "4":
            lam = get_positive_float("Enter lambda (rate parameter): ")
            X = poisson_distribution(samples, lam)

        case "5":
            scale = get_positive_float("Enter scale parameter: ")
            X = exponential_distribution(samples, scale)

        case "6":
            shape = get_positive_float("Enter shape parameter: ")
            scale = get_positive_float("Enter scale parameter: ")
            X = gamma_distribution(samples, shape, scale)
        case _:
            print("Invalid custom option.")
            exit(1)

    print("1. Y ~ Uniform Distribution (U(a, b))")
    print("2. Y ~ Normal Distribution (N(mean, std_dev))")
    print("3. Y ~ Binomial Distribution (Bin(n, p))")
    print("4. Y ~ Poisson Distribution (Poisson(lambda))")
    print("5. Y ~ Exponential Distribution (Exp(scale))")
    print("6. Y ~ Gamma Distribution (Gamma(shape, scale))")
    print("7. Custom Y = f(X)")

    if (X_choice > "7") or (X_choice < "1"):
        print("Invalid choice.")
        exit(1)

    Y_choice = input("Enter your choice (1-7): ").strip()

    match Y_choice:
        case "1":
            a = float(input("Enter lower bound (a): ").strip())
            b = float(input("Enter upper bound (b): ").strip())
            Y = uniform_distribution(samples, a, b)

        case "2":
            mean = float(input("Enter mean: ").strip())
            std_dev = get_positive_float("Enter standard deviation: ")
            Y = normal_distribution(samples, mean, std_dev)

        case "3":
            n = get_positive_integer("Enter number of trials (n): ")
            p = get_probability("Enter probability of success (p): ")
            Y = binomial_distribution(samples, n, p)

        case "4":
            lam = get_positive_float("Enter lambda (rate parameter): ")
            Y = poisson_distribution(samples, lam)

        case "5":
            scale = get_positive_float("Enter scale parameter: ")
            Y = exponential_distribution(samples, scale)

        case "6":
            shape = get_positive_float("Enter shape parameter: ")
            scale = get_positive_float("Enter scale parameter: ")
            Y = gamma_distribution(samples, shape, scale)
        case "7":
            f = input("Enter the function f(X) for Y: ").strip()
            try:
                Y = eval(f)
            except Exception as e:
                print(f"Error: {e}")
        case _:
            print("Invalid custom option.")
            exit(1)

    print("Generated X:", X)
    print("Generated Y:", Y)
    save_file_joint(X, Y, filename)


def handle_args():
    parser = argparse.ArgumentParser(
        description="Generate a test case for random variables."
    )
    parser.add_argument(
        "filename",
        type=str,
        nargs="?",
        default="generated_data.txt",
        help="Name of the input file (default: generated_data.txt).",
    )
    return parser.parse_args()


def generate_test_case():
    args = handle_args()
    filename = args.filename

    print("\n=== Test Case Generator ===")
    print("1. Uniform Distribution (U(a, b))")
    print("2. Normal Distribution (N(mean, std_dev))")
    print("3. Binomial Distribution (Bin(n, p))")
    print("4. Poisson Distribution (Poisson(lambda))")
    print("5. Exponential Distribution (Exp(scale))")
    print("6. Gamma Distribution (Gamma(shape, scale))")
    print("7. Joint Random Variable with Operations")

    choice = input("Enter your choice (1-7): ").strip()

    if (choice > "7") or (choice < "1"):
        print("Invalid choice.")
        exit(1)

    samples = get_positive_integer("Enter the number of samples to generate: ")

    match choice:
        case "1":
            a = float(input("Enter lower bound (a): ").strip())
            b = float(input("Enter upper bound (b): ").strip())
            data = uniform_distribution(samples, a, b)

        case "2":
            mean = float(input("Enter mean: ").strip())
            std_dev = get_positive_float("Enter standard deviation: ")
            data = normal_distribution(samples, mean, std_dev)

        case "3":
            n = get_positive_integer("Enter number of trials (n): ")
            p = get_probability("Enter probability of success (p): ")
            data = binomial_distribution(samples, n, p)

        case "4":
            lam = get_positive_float("Enter lambda (rate parameter): ")
            data = poisson_distribution(samples, lam)

        case "5":
            scale = get_positive_float("Enter scale parameter: ")
            data = exponential_distribution(samples, scale)

        case "6":
            shape = get_positive_float("Enter shape parameter: ")
            scale = get_positive_float("Enter scale parameter: ")
            data = gamma_distribution(samples, shape, scale)

        case "7":
            generate_joint_rv(samples, filename)
            return

        case _:
            print("Invalid choice.")
            exit(1)

    print("Generated Data:")
    print(data)
    save_file(data, filename)


if __name__ == "__main__":
    generate_test_case()
