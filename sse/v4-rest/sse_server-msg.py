from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

message_queue = asyncio.Queue()  # Stores messages from clients

async def event_stream():
    """Sends events, including user acknowledgments."""
    while True:
        message = await message_queue.get()  # Wait for a new message
        yield f"data: Acknowledgment - Received: {message}\n\n"  # SSE format

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
