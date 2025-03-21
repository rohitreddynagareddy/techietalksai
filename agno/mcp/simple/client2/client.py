import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    server_url = "http://server:8000/sse"
    headers = {"Accept": "text/event-stream"}

    async with sse_client(server_url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            tools = tools_response.tools
            print(f"Available tools: {[tool.name for tool in tools]}")

            echo_tool = next((tool for tool in tools if tool.name == "echo_tool"), None)
            if echo_tool:
                message = "Hello, MCP!"
                result = await session.call_tool(echo_tool.name, {"message": message})
                print(f"Echo tool response: {result.content}")
            else:
                print("Echo tool not found on the server.")

if __name__ == "__main__":
    asyncio.run(main())
