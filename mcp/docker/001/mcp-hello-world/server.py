from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)
import asyncio
import sys


# Add these lines at the top
import sys
import aiofiles

# Replace the log_data function with
async def log_data(data: bytes, direction: str) -> None:
    """Log to both file and stdout"""
    line = f"[{direction}] {data.decode().strip()}\n"
    async with aiofiles.open("/app/logs/logs.txt", "a") as f:
        await f.write(line)
    sys.stdout.write(line)  # Also show in container stdout
    sys.stdout.flush()

    
DEFAULT_USER_AGENT_AUTONOMOUS = "ModelContextProtocol/1.0 (Autonomous; +https://github.com/modelcontextprotocol/servers)"
DEFAULT_USER_AGENT_MANUAL = "ModelContextProtocol/1.0 (User-Specified; +https://github.com/modelcontextprotocol/servers)"

# class Logger:
#     def __init__(self, filename):
#         self.terminal = sys.__stdout__
#         self.log = open(filename, "w", buffering=1)
#         self.buffer = self.terminal.buffer  # Add this line

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()

async def main():
    # Open the log file in write mode
    log_file = open("/app/logs/logs.txt", "w")
    # Redirect stdout and stderr to the log file
    # sys.stdout = log_file
    # sys.stderr = log_file

    await serve()

async def serve() -> None:
    """Run the hello world MCP server.
    """
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


if __name__ == "__main__":
    # sys.stdout = Logger("/app/logs/logs.txt")
    # sys.stderr = sys.stdout  # Redirect stderr to the same logger
    print("Starting the MCP Server")
    asyncio.run(main())

