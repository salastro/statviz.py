import numpy as np
import sys

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

def generate_test_case():
    print("\n=== Test Case Generator ===")
    print("1. Uniform Distribution (U(a, b))")
    print("2. Normal Distribution (N(mean, std_dev))")
    print("3. Binomial Distribution (Bin(n, p))")
    print("4. Poisson Distribution (Poisson(lambda))")
    print("5. Exponential Distribution (Exp(scale))")
    print("6. Gamma Distribution (Gamma(shape, scale))")
    print("7. Custom Random Variable with Operations")

    choice = input("Enter your choice (1-7): ").strip()

    try:
        samples = int(input("Enter the number of samples to generate: ").strip())
        if samples <= 0:
            raise ValueError("Number of samples must be positive integer.")

        if choice == "1":
            a = float(input("Enter lower bound (a): ").strip())
            b = float(input("Enter upper bound (b): ").strip())
            data = uniform_distribution(samples, a, b)

        elif choice == "2":
            mean = float(input("Enter mean: ").strip())
            std_dev = float(input("Enter standard deviation: ").strip())
            data = normal_distribution(samples, mean, std_dev)

        elif choice == "3":
            n = int(input("Enter number of trials (n): ").strip())
            p = float(input("Enter probability of success (p): ").strip())
            if not (0 <= p <= 1):
                raise ValueError("Probability must be between 0 and 1.")
            data = binomial_distribution(samples, n, p)

        elif choice == "4":
            lam = float(input("Enter lambda (rate parameter): ").strip())
            if lam <= 0:
                raise ValueError("Lambda must be positive.")
            data = poisson_distribution(samples, lam)

        elif choice == "5":
            scale = float(input("Enter scale parameter: ").strip())
            if scale <= 0:
                raise ValueError("Scale parameter must be positive.")
            data = exponential_distribution(samples, scale)

        elif choice == "6":
            shape = float(input("Enter shape parameter: ").strip())
            scale = float(input("Enter scale parameter: ").strip())
            if shape <= 0 or scale <= 0:
                raise ValueError("Shape and scale parameters must be positive.")
            data = gamma_distribution(samples, shape, scale)

        elif choice == "7":
            print("Custom RV: X and Y with Operations")
            print("1. X \u223c Exp(scale), Y = 3X + 2")
            print("2. X \u223c {-1, 1} Uniform, Y = X + n where n \u223c N(0, 0.5)")
            custom_choice = input("Choose an option (1/2): ").strip()

            if custom_choice == "1":
                scale = float(input("Enter scale for X (Exp(scale)): ").strip())
                X = exponential_distribution(samples, scale)
                Y = 3 * X + 2

            elif custom_choice == "2":
                X = np.random.choice([-1, 1], size=samples)
                noise = np.random.normal(0, 0.5, samples)
                Y = X + noise

            else:
                raise ValueError("Invalid custom option.")

            print("Generated X:", X)
            print("Generated Y:", Y)
            save_file(X, "X_values_custom.txt")
            save_file(Y, "Y_values_custom.txt")
            return
        else:
            print("Invalid choice.")
            return

        print("Generated Data:")
        print(data)
        save_file(data)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

def save_file(data, filename="generated_data.txt"):
    with open(filename, "w") as file:
        file.write(f"{len(data)}\n")
        for value in data:
            file.write(f"{value}\n")
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    generate_test_case()

