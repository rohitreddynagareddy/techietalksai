import asyncio
import websockets

async def send_messages(websocket):
    """Sends a message to the server every 5 seconds."""
    counter = 0
    while True:
        counter += 1
        message = f"Client message {counter}"
        await websocket.send(message)
        print(f"➡️ Sent to Server: {message}")
        await asyncio.sleep(5)  # ✅ Sends every 5 seconds

async def receive_messages(websocket):
    """Receives messages from the WebSocket server."""
    while True:
        message = await websocket.recv()
        print(f"📩 Received from Server: {message}")

async def websocket_client():
    """Creates a WebSocket connection and runs send & receive tasks concurrently."""
    uri = "ws://ws-server:8000/ws"
    async with websockets.connect(uri) as websocket:
        await asyncio.gather(
            send_messages(websocket),
            receive_messages(websocket),
        )

# ✅ Start the WebSocket client
asyncio.run(websocket_client())
