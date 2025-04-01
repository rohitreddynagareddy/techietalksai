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
import random # <-- Import random
import string # <-- Import string

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Tool Implementation Functions ---

async def _fetch_website_impl(
    url: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Internal function to handle the actual fetching logic."""
    logger.debug(f"Fetching website content from URL: {url}")
    headers = {
        "User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"
    }
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        try:
            response = await client.get(url)
            response.raise_for_status() # Raises exception for 4xx or 5xx status codes
            logger.debug(f"Received response with status code: {response.status_code} from {url}")
            return [types.TextContent(type="text", text=response.text)]
        except httpx.RequestError as exc:
            logger.error(f"HTTP Request Error fetching {url}: {exc}")
            # Return error information as text content
            return [types.TextContent(type="text", text=f"Error fetching URL: {exc}")]
        except httpx.HTTPStatusError as exc:
             logger.error(f"HTTP Status Error fetching {url}: {exc.response.status_code} - {exc}")
             # Return error information including status code
             return [types.TextContent(type="text", text=f"Error fetching URL: Status {exc.response.status_code} - {exc.response.text[:200]}...")] # Truncate long error pages

async def _make_booking_impl(name: str) -> list[types.TextContent]:
    """Internal function to handle the booking logic."""
    logger.debug(f"Creating booking for name: {name}")
    if not name or not isinstance(name, str):
         logger.error("Invalid 'name' provided for booking.")
         # Return error information as text content
         return [types.TextContent(type="text", text="Error: Invalid or missing 'name' for booking.")]

    # Generate a random booking reference (e.g., BK- seguido de 6 letras/números maiúsculos)
    reference = "BK-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"Generated booking reference '{reference}' for '{name}'")
    result_text = f"Booking confirmed for {name}. Your reference number is: {reference}"
    return [types.TextContent(type="text", text=result_text)]

# --- Health Check ---
async def health_check(request):
    logger.debug("Health check endpoint called.")
    return JSONResponse({"status": "ok"})


# --- Main Server Setup ---
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
    # Give the server a descriptive name
    app = Server("mcp-tools-server")

    # --- Tool Dispatcher ---
    @app.call_tool()
    async def handle_tool_call(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handles incoming requests to call a specific tool."""
        logger.debug(f"Tool requested: '{name}' with arguments: {arguments}")

        if name == "fetch":
            if "url" not in arguments:
                logger.error(f"Missing required argument 'url' for tool '{name}'")
                # Return error message in the expected format
                return [types.TextContent(type="text", text="Error: Missing required argument 'url' for fetch tool.")]
            # Call the fetch implementation
            return await _fetch_website_impl(arguments["url"])

        elif name == "booking":
            if "name" not in arguments:
                logger.error(f"Missing required argument 'name' for tool '{name}'")
                 # Return error message in the expected format
                return [types.TextContent(type="text", text="Error: Missing required argument 'name' for booking tool.")]
            # Call the booking implementation
            return await _make_booking_impl(arguments["name"])

        else:
            logger.error(f"Unknown tool requested: {name}")
            # Return error message in the expected format
            return [types.TextContent(type="text", text=f"Error: Unknown tool '{name}'. Available tools: fetch, booking.")]
            # Or raise an exception if preferred, though returning an error message might be more informative for the client
            # raise ValueError(f"Unknown tool: {name}")

    # --- Tool Lister ---
    @app.list_tools()
    async def list_available_tools() -> list[types.Tool]:
        """Lists all tools provided by this server."""
        logger.debug("Listing available tools.")
        return [
            # Fetch Tool Definition
            types.Tool(
                name="fetch",
                description="Fetches a website URL and returns its HTML text content. Handles redirects.",
                inputSchema={
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The complete HTTP or HTTPS URL to fetch (e.g., https://example.com).",
                        }
                    },
                },
                # outputSchema is often omitted, relying on description, but could be added for more rigor
            ),
            # Booking Tool Definition
            types.Tool(
                name="booking",
                description="Creates a booking for the specified person and returns a unique booking reference number.",
                inputSchema={
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The full name of the person for whom the booking is being made.",
                        }
                    },
                },
                 # outputSchema could specify the format of the TextContent if needed
            ),
        ]

    # --- Transport Handling (SSE or Stdio) ---
    if transport == "sse":
        from mcp.server.sse import SseServerTransport

        # Note: Changed endpoint from /sse to / for simplicity if desired, or keep /sse
        # Using /messages/ for POST as per original code
        sse_transport = SseServerTransport("/messages/") # Path for POSTing messages *to* the server

        async def handle_sse_connection(request):
            """Handles the initial SSE connection request from a client."""
            logger.debug(f"Handling new SSE connection request from: {request.client}")
            # The sse_transport manages the actual SSE stream communication
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                # streams[0] is for reading from client, streams[1] is for writing to client
                logger.debug(f"SSE connection established for {request.client}. Running MCP app logic.")
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            logger.debug(f"SSE connection closed for {request.client}.")
            # Note: The response is handled internally by connect_sse

        starlette_app = Starlette(
            debug=True, # Set to False in production
            routes=[
                # Endpoint where clients connect to establish the SSE stream
                Route("/sse", endpoint=handle_sse_connection),
                # Endpoint where clients POST messages *to* the server (part of SSE protocol)
                Mount("/messages/", app=sse_transport.handle_post_message),
                # Standard health check endpoint
                Route("/health", endpoint=health_check),
            ],
        )

        logger.info(f"Starting Uvicorn server with SSE transport on http://0.0.0.0:{port}")
        uvicorn.run(starlette_app, host="0.0.0.0", port=port, log_level="info") # Match uvicorn log level if desired

    else: # stdio transport
        from mcp.server.stdio import stdio_server

        async def arun_stdio():
            logger.info("Starting MCP server with stdio transport.")
            async with stdio_server() as streams:
                # streams[0] is stdin, streams[1] is stdout
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            logger.info("Stdio server stopped.")

        # Run the stdio server using anyio
        try:
            anyio.run(arun_stdio)
        except KeyboardInterrupt:
            logger.info("Stdio server interrupted by user.")

    return 0 # Exit code for CLI

# Ensure the script runs main() when executed
# if __name__ == "__main__":
#     main()