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

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider

import os
from dotenv import load_dotenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")


import logfire
from dotenv import load_dotenv
import os
load_dotenv()
logfire_api_key = os.getenv("LOGFIRE_API_KEY")
logfire.configure(token=logfire_api_key)
logfire.instrument_openai()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logfire.info('MCP Server using SSE Transport')

async def _poem_impl(name: str) -> list[types.TextContent]:
    """Internal function to handle the booking logic."""
    logger.debug(f"Creating poem on name: {name}")

    MODEL_CHOICE = "DeepSeek"
    # Initialize selected model
    if MODEL_CHOICE == "OpenAI":
        model = OpenAIModel(
            # model_name='gpt-3.5-turbo',
            model_name='gpt-4o-mini',
            # base_url='https://api.openai.com/v1',
            # api_key=OPENAI_API_KEY,
        )
    elif MODEL_CHOICE == "DeepSeek":
        model = OpenAIModel(
            'deepseek-chat',
            provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY),
        )
    elif MODEL_CHOICE == "Gemini":
        model = GeminiModel(
            # model_name='gemini-2.0-flash-exp',
            model_name='gemini-1.5-flash',
            api_key=GEMINI_API_KEY,
        )

    server_agent = Agent(
        # 'anthropic:claude-3-5-haiku-latest', system_prompt='always reply in rhyme'
        model=model, system_prompt='always reply in rhyme'
    )
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {name}')
    # return r.data


    if not name or not isinstance(name, str):
         logger.error("Invalid 'name' provided for poem.")
         # Return error information as text content
         return [types.TextContent(type="text", text="Error: Invalid or missing 'name' for poem.")]

    # Generate a random booking reference (e.g., BK- seguido de 6 letras/números maiúsculos)
    # reference = "BK-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"Generated poem '{r.data}' for '{name}'")
    result_text = f"Poem for {name}. {r.data}"
    return [types.TextContent(type="text", text=r.data)]

async def _doctor_impl(name: str) -> list[types.TextContent]:
    """Internal function to handle the booking logic."""
    logger.debug(f"Doctor query on: {name}")

    # MODEL_CHOICE = "DeepSeek"
    MODEL_CHOICE = "XAI"
    # Initialize selected model
    if MODEL_CHOICE == "OpenAI":
        model = OpenAIModel(
            # model_name='gpt-3.5-turbo',
            model_name='gpt-4o-mini',
            # base_url='https://api.openai.com/v1',
            # api_key=OPENAI_API_KEY,
        )
    elif MODEL_CHOICE == "DeepSeek":
        model = OpenAIModel(
            'deepseek-chat',
            provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY),
        )
    elif MODEL_CHOICE == "XAI":
        model = OpenAIModel(
            'grok-2-1212',
            provider=OpenAIProvider(base_url='https://api.x.ai/v1', api_key=XAI_API_KEY),
        )
    elif MODEL_CHOICE == "Gemini":
        model = GeminiModel(
            # model_name='gemini-2.0-flash-exp',
            model_name='gemini-1.5-flash',
            api_key=GEMINI_API_KEY,
        )
        #         1. Request Key Information First:
        # - Symptom details (type, duration, severity)
        # - Medical history/current conditions
        # - Allergies/current medications
        # - Pregnancy/breastfeeding status
    system_prompt = """Act as a virtual family doctor. Your role is to:



        1. Analyze for Red Flags:
        [IF chest pain, breathing difficulty, neurological symptoms, severe trauma, or fever >39°C (102.2°F): 
            → "Urgently consult a healthcare provider or visit ER immediately"]

        2. For Approved Cases:
        a) OTC Recommendations Table:
        | Symptom        | Medication (Generic) | Dosage       | Warnings          |
        |----------------|----------------------|--------------|-------------------|
        | [Symptom 1]    | [Drug 1]             | [Dose 1]     | [Contraindication]|

        b) Home Remedies Section:
        • [Remedy 1] - [Application method]
        • [Remedy 2] - [Frequency]

        3. Always Include:
        - "Monitor for: [Warning signs]"
        - "Consult physician if: [Timeframe/conditions]"
        - "Confirm compatibility with existing medications with your pharmacist"

        4. Safety Protocols:
        - Never diagnose conditions
        - Avoid brand names unless specifying generic alternatives
        - State limitations: "This advice cannot replace clinical evaluation"
        - No data retention disclaimer"""
    server_agent = Agent(
        # 'anthropic:claude-3-5-haiku-latest', system_prompt='always reply in rhyme'
        model=model, system_prompt=system_prompt
    )
    """Poem generator"""
    r = await server_agent.run(f'Provide medical adivise for {name}')
    
    logger.info(f"=====================================")
    logger.info(f"I AM SSE SERVER TOOL doctor: {model.model_name}")
    logger.info(f"=====================================")

    if not name or not isinstance(name, str):
         logger.error("Invalid 'name' provided to the doctor.")
         # Return error information as text content
         return [types.TextContent(type="text", text="Error: Invalid or missing 'name' for doctor.")]

    # Generate a random booking reference (e.g., BK- seguido de 6 letras/números maiúsculos)
    # reference = "BK-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"Generated report '{r.data}' for '{name}'")
    result_text = f"Report for {name}. {r.data}"
    return [types.TextContent(type="text", text=r.data)]

# --- Main Server Setup ---
@click.command()
@click.option("--port", default=8888, help="Port to listen on for SSE")
@click.option("--transport", default="sse", help="Transport type" )
def main(port: int, transport: str) -> int:
    logger.debug(f"Starting server with transport: {transport} on port: {port}")
    # Give the server a descriptive name
    app = Server("sse-server")
    # --- Tool Dispatcher ---
    @app.call_tool()
    async def handle_tool_call(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handles incoming requests to call a specific tool."""
        logger.debug(f"Tool requested: '{name}' with arguments: {arguments}")

        if name == "doctor":
            if "name" not in arguments:
                logger.error(f"Missing required argument 'name' for tool '{name}'")
                 # Return error message in the expected format
                return [types.TextContent(type="text", text="Error: Missing required argument 'name' for booking tool.")]
            # Call the booking implementation
            return await _doctor_impl(arguments["name"])

        else:
            logger.error(f"Unknown tool requested: {name}")
            # Return error message in the expected format
            return [types.TextContent(type="text", text=f"Error: Unknown tool '{name}'. Available tools: fetch, booking.")]

    # --- Tool Lister ---
    @app.list_tools()
    async def list_available_tools() -> list[types.Tool]:
        """Lists all tools provided by this server."""
        logger.debug("Listing available tools.")
        return [
            # Booking Tool Definition
            types.Tool(
                # name="poem",
                # description="Creates a poem for the specified name as the theme.",
                # inputSchema={
                #     "type": "object",
                #     "required": ["name"],
                #     "properties": {
                #         "name": {
                #             "type": "string",
                #             "description": "The name of the theme for the poem.",
                #         }
                #     },
                # },
                 # outputSchema could specify the format of the TextContent if needed
                name="doctor",
                description="Familty doctor for consultaion.",
                inputSchema={
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "medical condition.",
                        }
                    },
                },
                 # outputSchema could specify the format of the TextContent if needed
            ),
        ]

    logger.debug(f"Here..")
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
        ],
    )

    logger.info(f"Starting Uvicorn server with SSE transport on http://0.0.0.0:{port}")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port, log_level="info") # Match uvicorn log level if desired

    return 0 # Exit code for CLI

# Ensure the script runs main() when executed
if __name__ == "__main__":
    main()