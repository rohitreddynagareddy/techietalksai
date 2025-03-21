import asyncio

async def say(message, delay):
    print(f"Start {message} {delay}s")
    await asyncio.sleep(delay)
    # print(message)
    print(f"Done {message} {delay}s")

async def main():
    await asyncio.gather(
    	say("Third", 3),
        say("First", 2),
        say("Second", 1),
        # say("Third", 3)
    )

asyncio.run(main())

