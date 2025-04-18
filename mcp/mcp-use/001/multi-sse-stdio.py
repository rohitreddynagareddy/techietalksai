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
    # client = MCPClient.from_config_file(
    #     os.path.join(os.path.dirname(__file__), "multi.json")
    # )
    # Create a configuration with multiple servers
    config = {
        "mcpServers": {
            "booking": {
                "url": "http://localhost:8888/sse"
            },
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "./results",
                ],
            },
        }
    }
    # Create MCPClient with the multi-server configuration
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
    agent = MCPAgent(
        llm=llm, 
        client=client, 
        max_steps=30,
        # use_server_manager=True  # Enable the Server Manager
        )

    try:
        # Run a query to search for accommodations
        # result = await agent.run(
        #     "Find me a nice place to stay in Barcelona for 2 adults "
        #     "for a week in August. I prefer places with a pool and "
        #     "good reviews. Show me the top 3 options.",
        #     max_steps=30,
        # )
        # print(f"\nResult: {result}")

        # Example: Manually selecting a server for a specific task
        # result = await agent.run(
        #     "Search for Airbnb listings in Barcelona",
        #     server_name="airbnb" # Explicitly use the airbnb server
        # )
        # print(f"\nResult: {result}")

        # result_google = await agent.run(
        #     "Find restaurants near the first result using Google Search",
        #     server_name="playwright" # Explicitly use the playwright server
        # )
        # print(f"\nResult: {result}")

        # Example 1: Using tools from different servers in a single query

        # TESTED
        # result = await agent.run(
        #     # "Search for a nice place to stay in Barcelona on Airbnb, "
        #     # "then use Google to find nearby restaurants and attractions."
        #     # "Write the result in the current directory in restarant.txt",
        #     "search for the latest AI news from Google news and save as AI-news.md",
        #     max_steps=30,
        # )
        # print(result)

         # TESTED
        result = await agent.run(
            # "Search for a nice place to stay in Barcelona on Airbnb, "
            # "then use Google to find nearby restaurants and attractions."
            # "Write the result in the current directory in restarant.txt",
            "Book for Sree and save it as booking.md",
            max_steps=30,
        )
        print(result)

        
    finally:
        # Ensure we clean up resources properly
        if client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_airbnb_example())
