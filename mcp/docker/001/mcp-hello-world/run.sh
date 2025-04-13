

docker build -t sree-greet . --no-cache

# docker run -i -v ./logs:/app/logs sree-greet
docker run -i --rm -v /Users/sree/demo:/app/logs sree-greet


Starting the MCP Server
{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2024-11-05","capabilities":{"experimental":{},"tools":{"listChanged":false}},"serverInfo":{"name":"mcp-hello-world","version":"1.6.0"}}}
Starting the MCP Server
{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2024-11-05","capabilities":{"experimental":{},"tools":{"listChanged":false}},"serverInfo":{"name":"mcp-hello-world","version":"1.6.0"}}}

The issue arises because the MCP server communicates via raw byte streams (not through Python's sys.stdout), so your Logger class won't capture this data. Here's how to fix it:

1. Modify server.py to Log Raw STDIO Streams
Use this updated server.py to intercept raw input/output: