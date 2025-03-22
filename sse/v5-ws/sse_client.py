import aiohttp
import asyncio

async def sse_client():
    url = "http://sse-server:8000/sse"  # Docker container hostname
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            async for line in response.content:
                if line:
                    print(f"📩 Received from Server: {line.decode().strip()}")

asyncio.run(sse_client())
