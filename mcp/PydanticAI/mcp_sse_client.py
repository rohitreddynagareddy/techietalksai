from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP
import asyncio

  
# server = MCPServerHTTP(url='http://host.docker.internal:3001/sse') 
server = MCPServerHTTP(url='http://sse-server-py:3001/sse')   
agent = Agent(
			'openai:gpt-4o-mini', 
			mcp_servers=[server]
		)  

print("I AM SSE CLIENT")

async def main():
    async with agent.run_mcp_servers():  
        # result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
        # result = await agent.run('List tools')
        result = await agent.run('Fetch https://httpbin.org/anything')
    # print(result)
    print(result.data)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.

asyncio.run(main())