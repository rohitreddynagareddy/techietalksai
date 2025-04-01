# mcp_server/rhyme_tool.py
import os
import sys
from pydantic import BaseModel, Field
from pydantic_ai import Pipe, LLM, Input, Output
from mcp import tool # Import the tool decorator

# --- Environment Variable Check ---
# Ensure you have OPENAI_API_KEY set in your .env file or environment
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.", file=sys.stderr) # Log to stderr for stdio

# --- Pydantic Models ---
class Topic(BaseModel):
    topic: str = Field(..., description="The central theme for the rhyme.")

class Rhyme(BaseModel):
    rhyme: str = Field(..., description="A two-line rhyming couplet about the topic.")

# --- PydanticAI Pipeline ---
# Initialize the LLM and Pipe globally so it's reused by the tool
# Handle potential initialization errors
llm_instance = None
rhyme_pipe_instance = None
initialization_error = None

try:
    # Check for API key presence before initializing LLM might be safer
    if os.getenv("OPENAI_API_KEY"):
        llm_instance = LLM() # Defaults to OpenAI based on environment variables
        rhyme_pipe_instance = Pipe(
            input_model=Input(Topic),
            output_model=Output(Rhyme),
            task_description="Generate a creative two-line rhyming couplet based on the provided topic.",
            llm=llm_instance,
        )
        print("PydanticAI Pipe initialized successfully.", file=sys.stderr)
    else:
        initialization_error = "OPENAI_API_KEY not set. Cannot initialize LLM."
        print(f"Initialization Error: {initialization_error}", file=sys.stderr)

except Exception as e:
    initialization_error = f"Failed to initialize PydanticAI Pipe: {e}"
    print(f"Initialization Error: {initialization_error}", file=sys.stderr)

# --- MCP Tool Definition ---
# This function will be exposed by the 'mcp server run' command
@tool(name="rhyme_generator") # Give the tool an explicit name
async def generate_rhyme_tool(topic: str) -> str:
    """Generates a two-line rhyming couplet for the given topic via PydanticAI."""
    print(f"MCP Tool: Received request for topic: '{topic}'", file=sys.stderr)

    if initialization_error:
        print(f"MCP Tool: Returning initialization error: {initialization_error}", file=sys.stderr)
        return f"Sorry, the rhyming service is not configured correctly. Error: {initialization_error}"

    if rhyme_pipe_instance is None:
        # This case should ideally be covered by initialization_error, but as a safeguard:
        err_msg = "Rhyme generation model is not available."
        print(f"MCP Tool: {err_msg}", file=sys.stderr)
        return f"Sorry, the rhyming service isn't ready. {err_msg}"

    try:
        topic_input = Topic(topic=topic)
        # Use arun for async execution if available and needed, otherwise run sync
        # Let's assume sync pipe execution is acceptable within the async tool function for now
        # result = await rhyme_pipe_instance.arun(topic_input) # If pipe supports async
        result = rhyme_pipe_instance(topic_input) # Sync call

        print(f"MCP Tool: Generated rhyme: '{result.rhyme}'", file=sys.stderr)
        return result.rhyme
    except Exception as e:
        error_message = f"Error generating rhyme for '{topic}': {e}"
        print(f"MCP Tool: {error_message}", file=sys.stderr)
        # You might want more specific error handling here
        return f"An error occurred, I could not rhyme about '{topic}'. Please try again. (Details: {e})"

# Note: We don't need a __main__ block here if we use 'mcp server run <module>:<tool_func> --stdio'
