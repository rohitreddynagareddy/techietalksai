from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()
message_queue = asyncio.Queue()  # Stores messages from clients
waiting_count = 0  # Counter for waiting messages

async def event_stream():
    """Continuously sends events, including user messages and keep-alive messages."""
    global waiting_count
    while True:
        try:
            # Try getting a message with a timeout (prevents blocking forever)
            message = await asyncio.wait_for(message_queue.get(), timeout=5)
            yield f"data: Acknowledgment - Received: {message}\n\n"
        except asyncio.TimeoutError:
            # Send a keep-alive message every 5 seconds to prevent client disconnects
            waiting_count += 1
            yield f"data: Waiting for messages... (Count: {waiting_count})\n\n"

@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for real-time streaming."""
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/messages")
async def post_message(request: Request):
    """Receives JSON messages and adds them to the queue."""
    data = await request.json()
    message = data.get("message", "No message provided")
    await message_queue.put(json.dumps({"status": "Received", "message": message}))
    return {"status": "Message received", "message": message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
