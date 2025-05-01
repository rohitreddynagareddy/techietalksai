def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0:
        return 1
    else:
        return n * factorial(n - 1)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python fact.py <number>")
        sys.exit(1)
    number = int(sys.argv[1])
    print(f"Factorial of {number} is: {factorial(number)}")