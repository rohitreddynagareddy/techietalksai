# app/main.py
import streamlit as st
import os
import logging
from pydantic import ValidationError
from typing import Optional # Import Optional
# import nest_asyncio
import random
import asyncio
from pydantic_ai import Agent, RunContext
from openai import OpenAI, AuthenticationError, RateLimitError

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os
load_dotenv()

from pydantic_ai.mcp import MCPServerHTTP
server = MCPServerHTTP(url='http://sse-server:8888/sse') 
agent = Agent(
      'openai:gpt-4o-mini', 
       mcp_servers=[server]
    ) 
async def main(topic: str) -> str:
    async with agent.run_mcp_servers():  
        result = await agent.run(topic)
        return result.data

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📚 Remote Research SaaS") # Updated title
st.subheader("🛜 SSE Agentic MCP Server") # Updated title
st.markdown(f"""
Protect your intellectual property using Remote Agentic MCP Server.
""")

# Input Area
st.header("✍️ Research Topic")
default_text = "Example: Write a report comparing NVDA to TSLA"
input_text = st.text_input("Enter text for research:")

# Extraction Button
if st.button("✨ Do Research"): # Updated button text
    if not input_text:
        st.warning("Please enter some text to process.")
    else:
        st.info(f"Asking Agentic MCP Server to do the research...")
        try:
            with st.spinner("Agent is processing the request (may use MCP tools)..."):
                extracted_data = asyncio.run(main(input_text))

            # --- Display Results ---
            st.success(f"Agent successfully processed the request via Remote SSE Agentic MCP Server!")
            st.subheader("📊 Research Report:")
            st.success(extracted_data)

        except Exception as e:
            # Catch-all for other potential errors during Agent execution
            logger.error(f"An unexpected error occurred during Agent run: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {type(e).__name__}: {e}")

else:
    st.info("Enter the research topic: eg: Write a report comparing NVDA to TSLA")

st.markdown("---")
st.markdown("Powered by [Pydantic_AI](https://github.com/pydantic/pydantic-ai), [Streamlit](https://streamlit.io)")