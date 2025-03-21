import asyncio
import time  # Import time module for timestamps

async def greet():
    start_time = time.time()
    print(f"[{start_time:.2f}] Hello")
    await asyncio.sleep(5)  # Simulating a delay
    end_time = time.time()
    print(f"[{end_time:.2f}] World (Duration: {end_time - start_time:.2f} sec)")

async def count():
    start_time = time.time()
    for i in range(1, 9):
        print(f"[{time.time():.2f}] Counting: {i}")
        await asyncio.sleep(1)  # Simulating work every second
    end_time = time.time()
    print(f"[{end_time:.2f}] Counting completed (Duration: {end_time - start_time:.2f} sec)")

async def main():
    print(f"[{time.time():.2f}] SEQUENTIAL START")
    await greet()
    await count()

    print(f"[{time.time():.2f}] CONCURRENT START")
    await asyncio.gather(greet(), count())

asyncio.run(main())
