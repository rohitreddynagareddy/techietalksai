import asyncio

async def greet():
    print("Hello")
    await asyncio.sleep(5)  # Simulating a delay
    print("World")

async def count():
    for i in range(1, 9):
        print(f"Counting: {i}")
        await asyncio.sleep(1)  # Simulating work every second

async def main():
    print("SEQUENTIAL START")
    await  greet()
    await count()

    print("CONCURRENT START")
    await asyncio.gather(greet(), count())

asyncio.run(main())