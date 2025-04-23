
https://x.com/elder_plinius/status/1914018292890485102

Write factorial python code and save as /app/app/results/fact.py 

def factorial(n):
    if n < 0:
        return "Factorial not defined for negative numbers"
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Example usage:
if __name__ == "__main__":
    num = 5  # Change this number to test
    print(f"Factorial of {num} is {factorial(num)}")