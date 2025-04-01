# streamlit_app/app.py
import streamlit as st
import asyncio
import os
import sys
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Configuration ---
# Define how to start the MCP server process IN THE CONTAINER
SERVER_MODULE_PATH = "mcp_server.rhyme_tool:generate_rhyme_tool" # Path to the @tool function

# Use sys.executable to ensure the same Python interpreter is used
# Run 'mcp' as a module (-m) to avoid PATH issues
# Ensure WORKDIR in Dockerfile is /app
# MCP_SERVER_PARAMS = StdioServerParameters(
#     command=sys.executable,
#     args=['-m', 'mcp', 'server', 'run', SERVER_MODULE_PATH, '--stdio'],
#     env=dict(os.environ), # Pass environment (crucial for API keys)
#     cwd="/app" # Run the command from the app's root directory in the container
# )
MCP_SERVER_PARAMS = StdioServerParameters(
    command='mcp', # <<< CORRECT: Call the 'mcp' executable directly
    args=['server', 'run', SERVER_MODULE_PATH, '--stdio'], # <<< CORRECT: Arguments for the 'mcp' command
    env=dict(os.environ), # Pass environment (crucial for API keys)
    cwd="/app" # Run the command from the app's root directory in the container
)

# Name of the tool defined in rhyme_tool.py with @tool(name=...)
MCP_TOOL_NAME = "rhyme_generator"

# --- MCP Client Logic ---

# This function encapsulates the entire process: start server, connect, call tool, shutdown server.
# It's INEFFICIENT as it restarts the server for every request, but avoids complex async state
# management within Streamlit's execution model.
async def get_rhyme_via_stdio(topic: str) -> str:
    """
    Starts the MCP stdio server, calls the rhyme tool, and returns the result.
    Handles startup, communication, and shutdown for each call.
    """
    print(f"Attempting MCP stdio call for topic: '{topic}'", file=sys.stderr)
    start_time = time.time()
    try:
        # stdio_client manages the subprocess lifetime
        # async with stdio_client(MCP_SERVER_PARAMS, process_startup_timeout=15) as (read, write):
        async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
            # ClientSession manages the MCP communication protocol
            # async with ClientSession(read, write, session_startup_timeout=10) as session:
            async with ClientSession(read, write) as session:
                print("MCP Client: stdio_client and ClientSession entered.", file=sys.stderr)
                await session.initialize() # Perform MCP handshake
                print(f"MCP Client: Session initialized. Calling tool '{MCP_TOOL_NAME}'...", file=sys.stderr)

                # Call the tool defined in rhyme_tool.py
                # The input must be a dictionary matching the tool function's parameters
                result = await session.call_tool(MCP_TOOL_NAME, {'topic': topic}, timeout=30) # Tool input payload

                print(f"MCP Client: Tool call successful.", file=sys.stderr) # Verbose logging

                # Process the result - Adjust based on your tool's return type
                # Assuming the tool returns a simple string directly
                if result and result.content and isinstance(result.content, list) and len(result.content) > 0:
                    # The content of the result is often a list of parts.
                    # If the tool returns a string, it might be in result.content[0].text
                    # or directly as result.content[0] if simple. Let's check common patterns.
                    content_item = result.content[0]
                    if hasattr(content_item, 'text') and isinstance(content_item.text, str):
                        rhyme = content_item.text
                    elif isinstance(content_item, str):
                         rhyme = content_item
                    else:
                         # Fallback if the structure is unexpected
                         rhyme = str(content_item)

                    print(f"MCP Client: Extracted rhyme: '{rhyme}'", file=sys.stderr)
                    return rhyme
                else:
                    print(f"MCP Client: Tool result format unexpected: {result}", file=sys.stderr)
                    return "Error: Received unexpected result format from rhyming tool."

    except asyncio.TimeoutError as e:
        print(f"MCP Client Error: Timeout during operation: {e}", file=sys.stderr)
        return f"Error: The rhyming process timed out ({e})."
    except FileNotFoundError as e:
         print(f"MCP Client Error: Command not found: {e}. Ensure '{MCP_SERVER_PARAMS.command}' and 'mcp' module are available.", file=sys.stderr)
         return f"Error: Cannot find command to start rhyming service '{MCP_SERVER_PARAMS.command} -m mcp ...'. Setup issue?"
    except ConnectionRefusedError as e: # Might happen if process fails immediately
         print(f"MCP Client Error: Connection refused: {e}. Server process likely failed to start.", file=sys.stderr)
         return f"Error: Failed to connect to rhyming service. It might have crashed on startup. Check logs. ({e})"
    except Exception as e:
        # Catch any other exception during the process
        print(f"MCP Client Error: An unexpected error occurred: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return f"An unexpected error occurred while getting the rhyme: {e}"
    finally:
        end_time = time.time()
        print(f"MCP stdio call duration: {end_time - start_time:.2f} seconds", file=sys.stderr)


# --- Streamlit App UI ---
st.set_page_config(page_title="Rhyme Bot (MCP stdio)", page_icon="🎤")
st.title("🎤 Rhyme Bot (MCP stdio)")
st.caption(f"Ask me for a rhyme! (Uses MCP stdio via `{SERVER_MODULE_PATH}`)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "What topic should I rhyme about today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter a topic..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display thinking message and call the MCP tool
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Hmm, let me think of a rhyme...")

        # Run the async MCP call function from Streamlit's sync context
        try:
            # Use asyncio.run() for simplicity. This blocks the Streamlit thread until completion.
            rhyme_result = asyncio.run(get_rhyme_via_stdio(prompt))
            message_placeholder.markdown(rhyme_result)
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": rhyme_result})
        except Exception as e:
            # Catch potential errors from asyncio.run itself or if get_rhyme_via_stdio raises one unexpectedly
            st.error(f"Failed to get rhyme: {e}")
            error_msg = f"Sorry, couldn't get a rhyme due to an error: {e}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
