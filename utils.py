def get_positive_integer(prompt):
    try:
        value = int(input(prompt).strip())
        if value <= 0:
            raise ValueError("Value must be a positive integer.")
        return value
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


def get_positive_float(prompt):
    try:
        value = float(input(prompt).strip())
        if value <= 0:
            raise ValueError("Value must be positive.")
        return value
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


def get_probability(prompt):
    try:
        value = float(input(prompt).strip())
        if not (0 <= value <= 1):
            raise ValueError("Probability must be between 0 and 1.")
        return value
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
