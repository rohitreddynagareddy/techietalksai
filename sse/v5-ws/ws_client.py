import asyncio
import websockets

async def listen_to_server():
    uri = "ws://ws-server:8000/ws"  # ✅ Correct WebSocket URL
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(f"📩 Received from Server: {message}")

asyncio.run(listen_to_server())
