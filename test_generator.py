import sys

import numpy as np


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


def get_positive_integer(prompt):
    try:
        value = int(input(prompt).strip())
        if value <= 0:
            raise ValueError("Value must be a positive integer.")
        return value
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def get_positive_float(prompt):
    try:
        value = float(input(prompt).strip())
        if value <= 0:
            raise ValueError("Value must be positive.")
        return value
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def get_probability(prompt):
    try:
        value = float(input(prompt).strip())
        if not (0 <= value <= 1):
            raise ValueError("Probability must be between 0 and 1.")
        return value
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def save_file(data, filename="generated_data.txt"):
    with open(filename, "w") as file:
        file.write(f"{len(data)}\n")
        for value in data:
            file.write(f"{value}\n")
    print(f"Data saved to {filename}")

def save_file_joint(X, Y, filename="generated_data.txt"):
    with open(filename, "w") as file:
        file.write(f"{len(X)}\n")
        for x, y in zip(X, Y):
            file.write(f"{x} {y}\n")
    print(f"Data saved to {filename}")



def generate_joint_rv(samples):
    print("1. X ~ Uniform Distribution (U(a, b))")
    print("2. X ~ Normal Distribution (N(mean, std_dev))")
    print("3. X ~ Binomial Distribution (Bin(n, p))")
    print("4. X ~ Poisson Distribution (Poisson(lambda))")
    print("5. X ~ Exponential Distribution (Exp(scale))")
    print("6. X ~ Gamma Distribution (Gamma(shape, scale))")

    X_choice = input("Enter your choice (1-6): ").strip()

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
            sys.exit(1)

    print("1. Y ~ Uniform Distribution (U(a, b))")
    print("2. Y ~ Normal Distribution (N(mean, std_dev))")
    print("3. Y ~ Binomial Distribution (Bin(n, p))")
    print("4. Y ~ Poisson Distribution (Poisson(lambda))")
    print("5. Y ~ Exponential Distribution (Exp(scale))")
    print("6. Y ~ Gamma Distribution (Gamma(shape, scale))")
    print("7. Custom Y = f(X)")

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
            sys.exit(1)

    print("Generated X:", X)
    print("Generated Y:", Y)
    save_file_joint(X, Y)


def generate_test_case():
    print("\n=== Test Case Generator ===")
    print("1. Uniform Distribution (U(a, b))")
    print("2. Normal Distribution (N(mean, std_dev))")
    print("3. Binomial Distribution (Bin(n, p))")
    print("4. Poisson Distribution (Poisson(lambda))")
    print("5. Exponential Distribution (Exp(scale))")
    print("6. Gamma Distribution (Gamma(shape, scale))")
    print("7. Joint Random Variable with Operations")

    choice = input("Enter your choice (1-7): ").strip()
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
            generate_joint_rv(samples)
            return

        case _:
            print("Invalid choice.")
            sys.exit(1)

    print("Generated Data:")
    print(data)
    save_file(data)


if __name__ == "__main__":
    generate_test_case()
