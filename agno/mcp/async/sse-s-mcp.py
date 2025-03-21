from fastmcp import FastMCP
import asyncio

mcp = FastMCP("Chat Server")
message_queue = asyncio.Queue()
all_messages = []

@mcp.tool()
async def send_message(message: str) -> str:
    all_messages.append(message)
    await message_queue.put(message)
    return "Message sent successfully"

@mcp.resource("/messages")
async def get_messages() -> str:
    return "\n".join(all_messages)

@mcp.stream()
async def stream_messages():
    while True:
        message = await message_queue.get()
        yield message

if __name__ == "__main__":
    mcp.run()