import asyncio
from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def run_blender_example():
    # Load environment variables
    load_dotenv()

    # Create MCPClient with Blender MCP configuration
    config = {"mcpServers": {"blender": {"command": "uvx", "args": ["blender-mcp"]}}}
    client = MCPClient.from_dict(config)

    # Create LLM
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    # Create LLM
    #gpt-4.1-mini 
    #gpt-4.1-mini-2025-04-14
    #gpt-4.1-nano-2025-04-14
    # llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14") 
    llm = ChatOpenAI(model="gpt-4.1-mini")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    try:
        # Run the query
        result = await agent.run(
            "Create an inflatable cube with soft material and a plane as ground.",
            max_steps=30,
        )
        print(f"\nResult: {result}")
    finally:
        # Ensure we clean up resources properly
        if client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_blender_example())
