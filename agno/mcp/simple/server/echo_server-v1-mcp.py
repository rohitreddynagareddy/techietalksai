from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
import uvicorn

# Initialize FastAPI app
app = FastAPI(debug=True)

# Initialize FastMCP and SseServerTransport
mcp_app = FastMCP("example-server")
sse_transport = SseServerTransport("/sse")

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )


# SSE endpoint
@app.get("/sse")
async def sse_endpoint(request: Request):
    async def event_generator():
        async with sse_transport.connect_sse(request.scope, request.receive, request.send) as streams:
            await mcp_app.run(streams[0], streams[1], mcp_app.create_initialization_options())
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Endpoint to handle incoming messages
# @app.post("/messages")
# async def messages_endpoint(request: Request):
#     await sse_transport.handle_post_message(request.scope, request.receive, request.send)
@app.post("/messages")
async def messages_endpoint(scope, receive, send):
    await sse_transport.handle_post_message(scope, receive, send)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
