from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent
import asyncio

from dotenv import load_dotenv
import os
load_dotenv()
# import logfire
# logfire_api_key = os.getenv("LOGFIRE_API_KEY")
# logfire.configure(token=logfire_api_key)
# # logfire.info('Hello, {name}!', name='world')
# logfire.info('I AM SSE BASED MCP CLIENT')
# # print("I AM SSE BASED MCP CLIENT")
# logfire.instrument_openai()
# logfire.instrument_mcp()

server = MCPServerHTTP(url='http://sse-server:8888/sse')   

agent = Agent(
      'openai:gpt-4o-mini', 
      mcp_servers=[server]
    )  

async def main():
    async with agent.run_mcp_servers():  
        # result = await agent.run('List tools')
        # print(result.data)

        # result = await agent.run('Book for Sree')
        # print(result.data)

        result = await agent.run('Get me the greeting resource from sse-server')
        print(result.data)

asyncio.run(main())
