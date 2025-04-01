import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    async with stdio_client(
        StdioServerParameters(command="uv", args=["run", "mcp-simple-prompt"])
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available prompts
            prompts = await session.list_prompts()
            print("====PROMPTS=====")
            print(prompts)

            # Get the prompt with arguments
            prompt = await session.get_prompt(
                "simple",
                {
                    "context": "User is a software developer",
                    "topic": "Python async programming",
                },
            )
            print("====PROMPT=====")
            print(prompt)


asyncio.run(main())


# /Users/sree/Downloads/AI/techietalksai/agno/mcp/async/mcp-server/sree/python-sdk/examples/servers/simple-prompt
# (base) SreeMacMin16GB-1280:simple-prompt sree$ python client.py 
# ====PROMPTS=====
# nextCursor=None prompts=[Prompt(name='simple', description='A simple prompt that can take optional context and topic arguments', arguments=[PromptArgument(name='context', description='Additional context to consider', required=False), PromptArgument(name='topic', description='Specific topic to focus on', required=False)])]
# ====PROMPT=====
# description='A simple prompt with optional context and topic arguments' messages=[PromptMessage(role='user', content=TextContent(type='text', text='Here is some relevant context: User is a software developer')), PromptMessage(role='user', content=TextContent(type='text', text='Please help me with the following topic: Python async programming'))]
