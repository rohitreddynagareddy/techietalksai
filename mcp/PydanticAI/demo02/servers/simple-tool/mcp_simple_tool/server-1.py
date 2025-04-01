import anyio
import click
import httpx
import logging
import mcp.types as types
from mcp.server.lowlevel import Server
from starlette.responses import JSONResponse
from starlette.applications import Starlette
from starlette.routing import Mount, Route
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Health check handler function
async def health_check(request):
    logger.debug("Health check endpoint called.")
    return JSONResponse({"status": "ok"})

async def fetch_website(
    url: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.debug(f"Fetching website content from URL: {url}")
    headers = {
        "User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"
    }
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        logger.debug(f"Received response with status code: {response.status_code}")
        return [types.TextContent(type="text", text=response.text)]

@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    logger.debug(f"Starting server with transport: {transport} on port: {port}")
    app = Server("mcp-website-fetcher")

    @app.call_tool()
    async def fetch_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        logger.debug(f"Tool called: {name} with arguments: {arguments}")
        if name != "fetch":
            logger.error(f"Unknown tool: {name}")
            raise ValueError(f"Unknown tool: {name}")
        if "url" not in arguments:
            logger.error("Missing required argument 'url'")
            raise ValueError("Missing required argument 'url'")
        return await fetch_website(arguments["url"])

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        logger.debug("Listing available tools.")
        return [
            types.Tool(
                name="fetch",
                description="Fetches a website and returns its content",
                inputSchema={
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch",
                        }
                    },
                },
            )
        ]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            logger.debug(f"Handling new SSE connection from: {request.client}")
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                logger.debug("SSE connection established.")
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            logger.debug("SSE connection closed.")

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
                Route("/health", endpoint=health_check),
            ],
        )

        logger.info(f"Starting Uvicorn server on port {port}")
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            logger.debug("Starting stdio server.")
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            logger.debug("Stdio server stopped.")

        anyio.run(arun)

    return 0
