from sse_starlette.sse import EventSourceResponse
import asyncio
import json
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Route

# Create an MCP server
mcp = FastMCP("WordCountServer")

# Define the word-counting tool
@mcp.tool("countWords", description="Counts the number of words in the provided text.")
async def count_words(text: str):
    word_count = len(text.split())
    return {"content": [{"type": "text", "text": f"Word count: {word_count}"}]}

@mcp.tool()
async def mcp_endpoint():
    queue = asyncio.Queue()

    async def event_generator():
        while True:
            message = await queue.get()
            yield {"data": json.dumps(message)}

    # Connect the server to the queue
    async def handle_message(message):
        await queue.put(message)

    await mcp.connect(lambda msg: handle_message(msg))
    return EventSourceResponse(event_generator())

# Mount the SSE server to the existing ASGI server
app = Starlette(
    routes=[
        Route('/sse', endpoint=mcp.sse_app()),  # Corrected here
    ]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
