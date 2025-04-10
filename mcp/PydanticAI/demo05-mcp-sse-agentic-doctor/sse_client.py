from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai.mcp import MCPServerHTTP
import asyncio
import logfire
# configure logfire

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# logfire_api_key = os.getenv("LOGFIRE_API_KEY")
# logfire.configure(token=logfire_api_key)
# logfire.instrument_openai()
  
# server = MCPServerHTTP(url='http://host.docker.internal:3001/sse') 
# server1 = MCPServerHTTP(url='http://sse-server-py:3001/sse')   
server = MCPServerHTTP(url='http://sse-server:8888/sse')   
# server2 = MCPServerHTTP(url='http://sse-server-no-agent:8888/sse')   
# server3 = MCPServerHTTP(url='http://sse-server-agentic:8889/sse')   
# agent = Agent(
#       'openai:gpt-4o-mini', 
#        mcp_servers=[server]
#     )  

# model = OpenAIModel(
#     model_name='deepseek-chat',
#     base_url='https://api.deepseek.com/v1',
#     api_key=DEEPSEEK_API_KEY,
#     mcp_servers=[server]
# )

# model = OpenAIModel(
#     'deepseek-chat',
#     provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY),
# )
# agent = Agent(
#       model, 
#       mcp_servers=[server]
#     ) 

model = OpenAIModel(
    'grok-2-1212',
    provider=OpenAIProvider(base_url='https://api.x.ai/v1', api_key=XAI_API_KEY),
)
agent = Agent(
    model, 
    mcp_servers=[server],
    system_prompt="""Use the doctor tool to provide the anwser exactly as received from the tool."""
) 

print(f"I AM SSE CLIENT: {model.model_name}")

async def main():
    async with agent.run_mcp_servers():  
        # result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
        # result = await agent.run('List tools')
        # result = await agent.run('Fetch https://httpstat.us/200')
        # print(result.data)

        result = await agent.run('List tools')
        print(result.data)

        # result = await agent.run('Book for Sree')
        # print(result.data)

        # result = await agent.run('Write a report comparing NVDA to TSLA')
        # result = await agent.run('Ask family doctor for consultation on my ear ache')
        # result = await agent.run('I have ear ache')
        # print(result.data)

#    print(result.data)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.

asyncio.run(main())

# from mcp import ClientSession, StdioServerParameters, types
# from mcp.client.stdio import stdio_client

# # Create server parameters for stdio connection
# server_params = StdioServerParameters(
#     command="python",  # Executable
#     args=["example_server.py"],  # Optional command line arguments
#     env=None,  # Optional environment variables
# )


# # server = MCPServerHTTP(url='http://sse-server-py-2:3002/sse')   
# # agent = Agent(
# #             'openai:gpt-4o-mini', 
# #             mcp_servers=[server]
# #         )  


# # Optional: create a sampling callback
# async def handle_sampling_message(
#     message: types.CreateMessageRequestParams,
# ) -> types.CreateMessageResult:
#     return types.CreateMessageResult(
#         role="assistant",
#         content=types.TextContent(
#             type="text",
#             text="Hello, world! from model",
#         ),
#         model="gpt-3.5-turbo",
#         stopReason="endTurn",
#     )


# async def run():
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(
#             read, write, sampling_callback=handle_sampling_message
#         ) as session:
#             # Initialize the connection
#             await session.initialize()

#             # List available prompts
#             prompts = await session.list_prompts()

#             # Get a prompt
#             prompt = await session.get_prompt(
#                 "example-prompt", arguments={"arg1": "value"}
#             )

#             # List available resources
#             resources = await session.list_resources()

#             # List available tools
#             tools = await session.list_tools()

#             # Read a resource
#             content, mime_type = await session.read_resource("file://some/path")

#             # Call a tool
#             result = await session.call_tool("tool-name", arguments={"arg1": "value"})


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(run())