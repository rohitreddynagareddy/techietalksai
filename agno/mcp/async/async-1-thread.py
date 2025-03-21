import time
import threading

def greet():
    print("Hello")
    time.sleep(5)  # Blocking delay
    print("World")

def count():
    for i in range(1, 9):
        print(f"Counting: {i}")
        time.sleep(1)  # Blocking delay

def info(msg):
    print(f"Greet 2 {msg}")



# Create and start threads
greet_thread = threading.Thread(target=greet)  # Create a thread for greet()
count_thread = threading.Thread(target=count)  # Create a thread for count()

greet_thread.start()  # Start greet() in a separate thread
count_thread.start()  # Start count() in another separate thread

info("Main thread reached line 26")

# Wait for both threads to complete
greet_thread.join()  # Main thread waits for greet_thread to finish
count_thread.join()  # Main thread waits for count_thread to finish

info("Main thread reached line 32")
