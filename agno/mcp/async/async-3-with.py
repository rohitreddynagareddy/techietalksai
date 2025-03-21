import asyncio

class AsyncContextManager:
    async def __aenter__(self):
        await asyncio.sleep(1)
        print('Entering context')
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await asyncio.sleep(1)
        print('Exiting context')

    async def do_something(self):
        print('Starting something')
        await asyncio.sleep(5)
        print('Done something')

    async def do_something2(self):
        print('Starting something2')
        await asyncio.sleep(2)
        print('Done something2')

async def main():
    async with AsyncContextManager() as manager:
        await manager.do_something()
        await manager.do_something2()

async def main2():
    async with AsyncContextManager() as manager:
        # ✅ Run both methods concurrently using asyncio.gather()
        await asyncio.gather(
            manager.do_something(),
            manager.do_something2()
        )

print("SEQUENTIAL BUT ASYNC")
asyncio.run(main())
print("CONCURRENT(USING GATHER) AND ASYNC")
asyncio.run(main2())

