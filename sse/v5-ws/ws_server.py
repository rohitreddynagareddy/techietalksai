# import asyncio
# import websockets
import datetime

# connected_clients = set()

# async def handler(websocket, path):
#     """Handles WebSocket connections."""
#     connected_clients.add(websocket)
#     try:
#         async for message in websocket:
#             # ✅ Append timestamp and acknowledgment
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             response = f"✅ Received: '{message}' at {timestamp}"
            
#             # ✅ Send the response back to the sender
#             await websocket.send(response)
            
#             # ✅ Broadcast the message to all clients
#             for client in connected_clients:
#                 if client != websocket:  # Don't send back to the sender
#                     await client.send(f"🔄 Broadcast: {message} from another client")
#     except websockets.exceptions.ConnectionClosed:
#         pass
#     finally:
#         connected_clients.remove(websocket)

# # ✅ Start WebSocket server
# start_server = websockets.serve(handler, "0.0.0.0", 8000)

# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

app = FastAPI()
connected_clients = []  # List of connected WebSocket clients

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections."""
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()  # Receive message from client
            # ✅ Append timestamp and acknowledgment
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = f"✅ Received: '{data}' at {timestamp}"
            print(f"🔹 Received: {response}")
            for client in connected_clients:
                await client.send_text(f"📩 {response}")  # Broadcast message to all clients
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print("❌ Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, WebSocket
# import uvicorn

# app = FastAPI()

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         print(f"📩 Received from Client: {data}")
#         await websocket.send_text(f"👋 Acknowledged: {data}")  # Send response

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
