def factorial(n):
    """
    Calculate the factorial of a non-negative integer n.
    :param n: Non-negative integer
    :return: Factorial of n
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

# Example usage
if __name__ == '__main__':
    number = 5  # Change this value to test with other numbers
    print(f'The factorial of {number} is {factorial(number)}')