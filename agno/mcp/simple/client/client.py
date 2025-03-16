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

            prompt_response = await session.list_prompts()
            # prompt = prompt_response.tools
            # print(f"Available tools: {[tool.name for tool in tools]}")
            print(f"Available prompts: {prompt_response}")
            # Available prompts: meta=None nextCursor=None prompts=[Prompt(name='echo_prompt', description='', arguments=[PromptArgument(name='message', description=None, required=True)])]
            prompt_response = await session.get_prompt("echo_prompt", {"message": "ABCD"})
            print(f"Get Prompt response: {prompt_response}")

            echo_tool = next((tool for tool in tools if tool.name == "echo_tool"), None)
            if echo_tool:
                message = "World!"
                result = await session.call_tool(echo_tool.name, {"message": message})
                print(f"Echo tool response: {result.content}")
            else:
                print("Echo tool not found on the server.")

if __name__ == "__main__":
    asyncio.run(main())

