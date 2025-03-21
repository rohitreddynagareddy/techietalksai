import asyncio

class AsyncIterable:
    def __init__(self):
        print("CALLED INIT")
        self.count = 0

    def __aiter__(self):
        print("CALLED AITER")
        return self

    async def __anext__(self):
        print("CALLED ANEXT")
        if self.count >= 3:
            raise StopAsyncIteration
        await asyncio.sleep(1)
        self.count += 1
        return self.count

async def main():
    async for number in AsyncIterable():
        print(number)

asyncio.run(main())

