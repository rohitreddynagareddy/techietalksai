import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="docker",
        args=["run", "-i", "--rm", "sree-greet"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            tools = tools_response.tools
            print(f"Available tools: {[tool.name for tool in tools]}")

            echo_tool = next((tool for tool in tools if tool.name == "hello-world2"), None)
            if echo_tool:
                message = "Hello"
                result = await session.call_tool(echo_tool.name, {"greeting": message})
                print(f"Tool response: {result.content}")
            else:
                print("Tool 'hello-world2' not found.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
