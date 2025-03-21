from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import asyncio
from typing import AsyncGenerator

app = FastAPI(debug=True)

# Simplified SSE transport implementation
class SseServerTransport:
    def __init__(self):
        self.connections = set()

    async def event_generator(self, client_id: str) -> AsyncGenerator[str, None]:
        self.connections.add(client_id)
        try:
            while True:
                # Implement your actual event generation logic here
                # Example: Send keepalive every 15 seconds
                yield "data: keepalive\n\n"
                await asyncio.sleep(15)
        finally:
            self.connections.remove(client_id)

    async def handle_post_message(self, data: dict):
        # Implement message handling logic
        print("Received message:", data)
        return {"status": "received"}

sse_transport = SseServerTransport()

@app.get("/sse")
async def sse_endpoint(request: Request):
    async def event_stream() -> AsyncGenerator[str, None]:
        client_id = request.client.host  # Or use proper client identification
        async for event in sse_transport.event_generator(client_id):
            yield event
            
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.post("/messages")
async def messages_endpoint(request: Request):
    data = await request.json()
    return await sse_transport.handle_post_message(data)

@app.on_event("shutdown")
async def shutdown_event():
    print("Cleaning up SSE connections")
    sse_transport.connections.clear()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)