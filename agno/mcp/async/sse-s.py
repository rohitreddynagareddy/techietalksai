from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()
message_queue = asyncio.Queue()  # Stores messages from clients


async def event_stream():
    """Continuously send messages from queue to clients."""
    while True:
        message = await message_queue.get()  # Wait for a new message
        yield f"data: {message}\n\n"


# async def event_stream():
#     """Generator function that continuously yields SSE events."""
#     counter = 0
#     while True:
#         counter += 1
#         yield f"data: Event {counter}\n\n"  # SSE format
#         await asyncio.sleep(2)  # Simulate real-time event every 2 seconds

@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint that streams real-time events."""
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/messages")
async def post_message(request: Request):
    """Receive messages from clients and add them to the queue."""
    data = await request.json()
    await message_queue.put(data.get("message", "No message"))
    return {"status": "Message received"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

