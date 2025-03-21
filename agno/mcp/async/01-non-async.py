import time

def greet():
    print("Hello")
    time.sleep(3)  # Blocking delay
    print("World")

def count():
    for i in range(1, 6):
        print(f"Counting: {i}")
        time.sleep(1)  # Blocking delay

def main():
    # Manually interleave execution to simulate concurrency
    # print("Hello")
    greet()
    # for i in range(1, 9):  # Run count for 4 seconds
    #     print(f"Counting: {i}")
    #     time.sleep(1)

    count()
    # print("World")  # Now execute the final step after 5s
    print("Done")

main()
