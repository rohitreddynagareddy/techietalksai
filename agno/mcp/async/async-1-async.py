import asyncio

async def greet():
    print("Hello")
    await asyncio.sleep(5)
    print("World")


asyncio.run(greet())

