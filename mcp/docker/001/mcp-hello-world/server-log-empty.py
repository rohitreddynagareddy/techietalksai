import sys
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

DEFAULT_USER_AGENT_AUTONOMOUS = "ModelContextProtocol/1.0 (Autonomous; +https://github.com/modelcontextprotocol/servers)"
DEFAULT_USER_AGENT_MANUAL = "ModelContextProtocol/1.0 (User-Specified; +https://github.com/modelcontextprotocol/servers)"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "a", buffering=1)  # Line buffering
        self.buffer = self.terminal.buffer  # To handle binary writes if needed

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

async def serve() -> None:
    """Run the hello world MCP server."""
    server = Server("mcp-hello-world")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="hello-world2",
                description="A simple tool that returns a greeting.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "greeting": {
                            "type": "string",
                            "description": "The greeting to return.",
                        }
                    },
                    "required": ["greeting"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name, arguments: dict) -> list[TextContent]:
        return [TextContent(type="text", text=f"{arguments['greeting']} World!")]

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)

async def main():
    sys.stdout = Logger("/app/logs/logs.txt")
    sys.stderr = sys.stdout  # Redirect stderr to the same logger
    print("Starting the MCP Server")
    await serve()

if __name__ == "__main__":
    asyncio.run(main())
