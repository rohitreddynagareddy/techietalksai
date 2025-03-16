import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["echo_server.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
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

