from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

message_queue = asyncio.Queue()  # Stores messages for SSE clients
waiting_count = 0  # Counter for waiting messages

async def event_stream():
    """Continuously streams messages, including keep-alive waiting messages."""
    global waiting_count
    while True:
        try:
            # ✅ Try getting a message with a timeout of 5 seconds
            message = await asyncio.wait_for(message_queue.get(), timeout=5)
            waiting_count = 0  # Reset waiting count when a real message arrives
            yield f"data: {message}\n\n"
        except asyncio.TimeoutError:
            # ✅ Send "Waiting for messages..." every 5 seconds if no messages arrive
            waiting_count += 1
            yield f"data: Waiting for messages... (Count: {waiting_count})\n\n"

@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for real-time streaming."""
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/messages")
async def post_message(request: Request):
    """Receives messages from UI and adds them to the SSE queue."""
    data = await request.json()
    message = data.get("message", "No message provided")
    response = {"status": "Received", "message": message}

    # ✅ Notify SSE clients
    await message_queue.put(json.dumps(response))  

    return response  # ✅ REST immediately returns a response

@app.post("/send_rest")
async def send_rest_message(request: Request):
    """Receives REST messages, responds immediately, and notifies SSE clients."""
    data = await request.json()
    message = data.get("message", "No REST message provided")
    
    response = {"status": "Received via REST", "message": message}

    # ✅ REST returns response immediately
    immediate_response = response.copy()
    
    # ✅ Push message to SSE queue so real-time clients get notified
    await message_queue.put(json.dumps(response))  

    return immediate_response  # ✅ REST returns response instantly

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
