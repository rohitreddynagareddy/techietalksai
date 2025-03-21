import aiohttp
import asyncio
import sys
import threading

# Disable buffering for proper output
sys.stdout.reconfigure(line_buffering=True)

async def listen_to_sse():
    """Keep trying to reconnect if the server goes down."""
    url = "http://sse-server:8000/sse"
    
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
    """Send user input messages to the server (runs in a separate thread)."""
    url = "http://sse-server:8000/messages"
    while True:
        message = input("Enter message to send to server: ")  # Blocking input
        asyncio.run(send_message(url, message))

async def send_message(url, message):
    """Send message to the server asynchronously."""
    async with aiohttp.ClientSession() as session:
        payload = {"message": message}
        async with session.post(url, json=payload) as response:
            response_data = await response.json()
            print(f"Server response: {response_data}")

async def main():
    """Run SSE listener and message sender in parallel."""
    loop = asyncio.get_event_loop()
    thread = threading.Thread(target=send_messages_sync, daemon=True)
    thread.start()  # Start input handling in a separate thread
    await listen_to_sse()  # Continue listening to SSE

asyncio.run(main())
