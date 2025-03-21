import asyncio
import time  # Import time module for timestamps

async def task1():
    start_time = time.time()
    print(f"[{start_time:.2f}] Task 1 started")
    await asyncio.sleep(2)
    end_time = time.time()
    print(f"[{end_time:.2f}] Task 1 completed (Duration: {end_time - start_time:.2f} sec)")
    return "Task 1 done"

async def task2():
    start_time = time.time()
    print(f"[{start_time:.2f}] Task 2 started")
    await asyncio.sleep(1)
    end_time = time.time()
    print(f"[{end_time:.2f}] Task 2 completed (Duration: {end_time - start_time:.2f} sec)")
    return "Task 2 done"

async def independent_task():
    """Runs in the background."""
    start_time = time.time()
    print(f"[{start_time:.2f}] Independent task started")
    await asyncio.sleep(3)
    end_time = time.time()
    print(f"[{end_time:.2f}] Independent task completed (Duration: {end_time - start_time:.2f} sec)")

async def main():
    start_time = time.time()
    print(f"[{start_time:.2f}] Main parallel execution started")

    # ✅ Schedule an independent task (runs in the background)
    background_task = asyncio.create_task(independent_task())

    # ✅ Run task1 and task2 concurrently
    results = await asyncio.gather(task1(), task2())

    # ✅ Ensure background_task completes before exiting
    await background_task

    end_time = time.time()
    print(f"[{end_time:.2f}] Main completed (Duration: {end_time - start_time:.2f} sec)")
    print(results)

    # Sequential Execution (Runs One by One)
    print("\n🔹 Sequential Execution Starts")
    start_seq_time = time.time()
    
    await independent_task()
    await task1()
    await task2()

    end_seq_time = time.time()
    print(f"[{end_seq_time:.2f}] Sequential execution completed (Duration: {end_seq_time - start_seq_time:.2f} sec)")

# `asyncio.run()` is used to start the event loop
asyncio.run(main())
