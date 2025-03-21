import asyncio
import threading

async def async_task():
    print(f"Async Task in: {threading.current_thread().name}")
    await asyncio.sleep(2)
    print("Async Task Completed!")

def run_main_in_main_thread():
    loop = asyncio.new_event_loop()  # ✅ Create a new event loop in the main thread
    asyncio.set_event_loop(loop)  # ✅ Ensure this loop is used in the main thread
    loop.run_until_complete(async_task())  # ✅ Run the async function synchronously

# ✅ Run the main function in the main thread
run_main_in_main_thread()

#Async Task in: MainThread
#Async Task Completed!

