import asyncio
from mcp import ClientSession
import aiohttp
import json

async def main():
    # Connect to the SSE endpoint
    async with aiohttp.ClientSession() as http_session:
        async with http_session.get("http://mcp-server:8000/mcp") as response:
            # Create MCP client session
            async def send_message(message):
                print(f"Would send: {message}")  # In a real app, implement a send channel if needed

            session = ClientSession(
                receive=response.content,
                send=send_message,
                content_type="text/event-stream"
            )
            await session.initialize()

            # Call the countWords tool
            result = await session.call_tool("countWords", {"text": "Hello world this is a test"})
            print("Tool result:", result.toolResult.content[0]["text"])

            # Keep the session alive for demo purposes
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
