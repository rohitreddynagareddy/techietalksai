import asyncio
import time
async def task():
    print("Task started!")
    await asyncio.sleep(5)
    print("Task completed!")

print("Going to wait for 5s")
# Get the current event loop
loop = asyncio.get_event_loop()
time.sleep(5)
print("Wait over")
# Schedule a task to run
loop.run_until_complete(task())


print("Going to wait for 5s")
asyncio.run(task())
time.sleep(5)
print("Wait over")