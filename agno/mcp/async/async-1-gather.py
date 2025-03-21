import asyncio
import time

def sync_function(n):
    """Synchronous function with blocking sleep."""
    print(f"[{time.time():.2f}] sync_function {n} started")
    time.sleep(3)  # Blocking delay
    print(f"[{time.time():.2f}] sync_function {n} completed")

async def async_function(n):
    """Asynchronous function with non-blocking sleep."""
    print(f"[{time.time():.2f}] async_function {n} started")
    await asyncio.sleep(3)  # Non-blocking delay
    print(f"[{time.time():.2f}] async_function {n} completed")

# ---------------- SYNC VERSION ----------------
print("\n🔹 Running Synchronous (Blocking) Version:")
start_time = time.time()

for i in range(1, 4):
    sync_function(i)

end_time = time.time()
print(f"⏳ Synchronous Execution Time: {end_time - start_time:.2f} seconds")

# ---------------- ASYNC VERSION ----------------
print("\n🔹 Running Asynchronous (Non-Blocking) Version:")
async def main():
    start_time = time.time()
    
    # Run multiple async tasks in parallel
    await asyncio.gather(async_function(1), async_function(2), async_function(3))
    
    end_time = time.time()
    print(f"⚡ Asynchronous Execution Time: {end_time - start_time:.2f} seconds")

asyncio.run(main())
