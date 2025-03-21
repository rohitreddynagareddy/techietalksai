from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio, MCPServerHTTP
import asyncio


# THIS WORKS! STDIO
server = MCPServerStdio('npx', ['-y', '@pydantic/mcp-run-python', 'stdio'])

# THIS TOO WORKS! STDIO
# server = MCPServerStdio('npx', ['-y', '@modelcontextprotocol/server-filesystem', '/app/app'])
# pydanticai-app-stdio-npx-1  | Secure MCP Filesystem Server running on stdio
# pydanticai-app-stdio-npx-1  | Allowed directories: [ '/app/app' ]
# pydanticai-app-stdio-npx-1  | Here are the files in the current folder:
# pydanticai-app-stdio-npx-1  | 
# pydanticai-app-stdio-npx-1  | - [FILE] .env
# pydanticai-app-stdio-npx-1  | - [FILE] .env.example
# pydanticai-app-stdio-npx-1  | - [FILE] Dockerfile
# pydanticai-app-stdio-npx-1  | - [FILE] docker-compose.yml
# pydanticai-app-stdio-npx-1  | - [FILE] mcp_client.py
# pydanticai-app-stdio-npx-1  | - [FILE] mcp_run_python.py
# pydanticai-app-stdio-npx-1  | - [FILE] mcp_server.py
# pydanticai-app-stdio-npx-1  | - [FILE] mcp_sse_client.py
# pydanticai-app-stdio-npx-1  | - [FILE] mcp_stdio_client.py
# pydanticai-app-stdio-npx-1  | - [FILE] requirements.txt
# pydanticai-app-stdio-npx-1 exited with code 0


# DEBUGGING
# /Users/sree/Downloads/AI/techietalksai/agno/mcp/async/mcp-server/sree/python-sdk/examples/servers/simple-tool
# uv run mcp-simple-tool --transport sse --port 3001

# docker run --rm curlimages/curl:8.12.1 -L -v http://host.docker.internal:3001/sse
#   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
#                                  Dload  Upload   Total   Spent    Left  Speed
#   0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0* Host host.docker.internal:3001 was resolved.
# * IPv6: (none)
# * IPv4: 192.168.65.2
# *   Trying 192.168.65.2:3001...
# * Connected to host.docker.internal (192.168.65.2) port 3001
# * using HTTP/1.x
# > GET /sse HTTP/1.1
# > Host: host.docker.internal:3001
# > User-Agent: curl/8.12.1
# > Accept: */*
# > 
# * Request completely sent off
# < HTTP/1.1 200 OK
# < date: Fri, 21 Mar 2025 13:08:31 GMT
# < server: uvicorn
# < cache-control: no-cache
# < connection: keep-alive
# < x-accel-buffering: no
# < content-type: text/event-stream; charset=utf-8
# < Transfer-Encoding: chunked
# < 
# { [87 bytes data]
# event: endpoint
# data: /messages/?session_id=297a456f65564dc9a3d6fcb6abd37f47


# WORKS!
# uv run mcp-simple-tool --transport sse --port 3001
# server = MCPServerHTTP(url='http://sse-server-py:3001/sse') 
# server = MCPServerHTTP(url='http://host.docker.internal:3001/sse') 
# prompt = 'list the tools'
# pydanticai-app-stdio-npx-1  | The available tool is:
# pydanticai-app-stdio-npx-1  | 
# pydanticai-app-stdio-npx-1  | 1. **functions.fetch**: Fetches a website and returns its content. It requires a URL as a parameter.

agent = Agent('openai:gpt-4o-mini', mcp_servers=[server])

async def main():
    async with agent.run_mcp_servers():
        # result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
        # result = await agent.run('List files in the current folder')
        result = await agent.run(prompt)
    print(result.data)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.

asyncio.run(main())
