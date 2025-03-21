import asyncio
import threading

async def async_task():
    print(f"Async Task Running in: {threading.current_thread().name}")
    await asyncio.sleep(2)
    print("Async Task Completed!")

async def main():
    print(f"Main Function in: {threading.current_thread().name}")
    await async_task()

# ✅ Runs main() inside the main thread
asyncio.run(main())

#Main Function in: MainThread
#Async Task Running in: MainThread
#Async Task Completed!
