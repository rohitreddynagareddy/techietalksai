import asyncio
import os
from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


async def run_airbnb_example():
    # Load environment variables
    load_dotenv()

    # Create MCPClient with Airbnb configuration
    client = MCPClient.from_config_file(
        os.path.join(os.path.dirname(__file__), "airbnb_mcp.json")
    )

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
        # Run a query to search for accommodations
        result = await agent.run(
            "Find me a nice place to stay in Barcelona for 2 adults "
            "for a week in August. I prefer places with a pool and "
            "good reviews. Show me the top 3 options.",
            max_steps=30,
        )
        print(f"\nResult: {result}")
    finally:
        # Ensure we clean up resources properly
        if client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_airbnb_example())
