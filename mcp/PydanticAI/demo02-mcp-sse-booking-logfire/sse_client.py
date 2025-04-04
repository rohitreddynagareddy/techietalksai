from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent
import asyncio

from dotenv import load_dotenv
import os
load_dotenv()
import logfire
logfire_api_key = os.getenv("LOGFIRE_API_KEY")
logfire.configure(token=logfire_api_key)
logfire.instrument_openai()

server = MCPServerHTTP(url='http://sse-server:8888/sse')   

agent = Agent(
      'openai:gpt-4o-mini', 
      mcp_servers=[server]
    )  

print("I AM SSE CLIENT")

async def main():
    async with agent.run_mcp_servers():  
        result = await agent.run('List tools')
        print(result.data)

        result = await agent.run('Book for Sree')
        print(result.data)

asyncio.run(main())
