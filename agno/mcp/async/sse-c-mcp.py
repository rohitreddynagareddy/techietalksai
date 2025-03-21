import aiohttp
import asyncio
import sys
import threading

sys.stdout.reconfigure(line_buffering=True)

async def listen_to_sse():
    url = "http://localhost:8000/stream"
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    async for line in response.content:
                        message = line.decode().strip()
                        if message:
                            formatted_message = message.replace("data: ", "")
                            print(f"\n📩 Received from Server: {formatted_message}\n➡ Enter message to send: ", end="", flush=True)
        except aiohttp.ClientError:
            print("\n⚠️ Connection lost. Reconnecting in 5 seconds...\n")
            await asyncio.sleep(5)

def send_messages_sync():
    url = "http://localhost:8000/rpc"
    while True:
        message = input("Enter message to send to server: ")
        asyncio.run(send_message(url, message))

async def send_message(url, message):
    async with aiohttp.ClientSession() as session:
        payload = {
            "jsonrpc": "2.0",
            "method": "send_message",
            "params": {"message": message},
            "id": 1
        }
        async with session.post(url, json=payload) as response:
            response_data = await response.json()
            print(f"Server response: {response_data}")

async def main():
    loop = asyncio.get_event_loop()
    thread = threading.Thread(target=send_messages_sync, daemon=True)
    thread.start()
    await listen_to_sse()

asyncio.run(main())