# app/main.py
import streamlit as st
import os
import logging
from pydantic import ValidationError
from typing import Optional # Import Optional
# import nest_asyncio
import random
import asyncio

# Corrected import for the Agent class
from pydantic_ai import Agent, RunContext

from openai import OpenAI, AuthenticationError, RateLimitError

# Import your Pydantic models
from models import UserDetails, ProductInfo, ThoughtProcess # Use relative import within the package

# nest_asyncio.apply()

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LLM Client Setup ---
# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client (handle potential missing key)
# NOTE: Pydantic-AI handles client initialization internally based on model string/API keys
# We might still need the client for other potential direct calls, but Pydantic-AI manages its own.
llm_client = None # Keep for potential future direct use, but Pydantic-AI Agent doesn't strictly need it passed
if openai_api_key:
    try:
        # Basic check if key is present and looks okay (optional)
        # llm_client = OpenAI(api_key=openai_api_key)
        # llm_client.models.list()
        logger.info("OpenAI API Key found.")
    except AuthenticationError:
        logger.error("OpenAI Authentication Error detected with provided key.")
        # Error will be shown below if needed by Pydantic-AI
    except Exception as e:
        logger.error(f"Error during initial OpenAI client check (optional): {e}")
else:
    logger.warning("OPENAI_API_KEY environment variable not found.")
    # Warning will be shown below

# --- PydanticAI Setup ---
# Select the LLM model (use a cost-effective one for demos if possible)
LLM_MODEL_STRING = "gpt-4o-mini" # Example, uses OpenAI implicitly if key is set

# Check if the provider is potentially OpenAI and if the key is missing
# This is a heuristic; Pydantic-AI might support other ways to specify OpenAI models
is_likely_openai = LLM_MODEL_STRING.startswith("gpt-") or "openai" in LLM_MODEL_STRING
show_key_error = is_likely_openai and not openai_api_key

# Dictionary mapping model names to model classes for dynamic selection
AVAILABLE_MODELS = {
    "ThoughtProcess": ThoughtProcess,
    "Product Info": ProductInfo,
    "Product Info": ProductInfo,
}

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📄 Pydantic_AI Agent Demo with Tools") # Updated title
st.markdown(f"""
This app demonstrates using `pydantic_ai.Agent` with an LLM (configured as `{LLM_MODEL_STRING}`)
to extract structured data *and* potentially use tools based on the input text.
""")

if show_key_error:
    st.error("OpenAI API Key is missing in the `.env` file. Please add it and restart the application, as the selected model likely requires it.")
    st.stop() # Stop execution if key is missing for configured provider

# Input Area
st.header("Input Text")
default_text = "Example: John Smith is a 42 year old engineer living in London. He bought a fantastic new laptop for $1200! His email address is abcd@xyz.com"
input_text = st.text_area("Enter text for extraction and analysis:", height=150, value=default_text)

# Model Selection
st.header("Select Extraction Model")
selected_model_name = st.selectbox("Choose the Pydantic model (`result_type`) for extraction:", options=list(AVAILABLE_MODELS.keys()))
TargetModel = AVAILABLE_MODELS[selected_model_name]

# Display selected model schema for clarity
with st.expander("View Selected Model Schema (`result_type`)"):
    st.json(TargetModel.model_json_schema())

# Extraction Button
if st.button("✨ Extract Information & Analyze Sentiment"): # Updated button text
    if not input_text:
        st.warning("Please enter some text to process.")
    else:
        st.info(f"Asking Agent (LLM: {LLM_MODEL_STRING}) to extract '{selected_model_name}' and analyze sentiment...")
        try:
            # --- Core Pydantic_AI Agent Logic with Tool ---

            # 1. Instantiate the Agent *inside* the handler because TargetModel changes
            #    - Pass the LLM model string
            #    - Pass the desired Pydantic model class as result_type
            #    - Add a system prompt encouraging tool use
            agent = Agent(
                LLM_MODEL_STRING,
                result_type=TargetModel,
                system_prompt=(
                    "You are an expert data extractor. Extract information matching the "
                    f"'{TargetModel.__name__}' schema. Also, use the available tools "
                    "if they seem relevant to the task or input text, such as analyzing sentiment."
                )
            )
            logger.info(f"Agent initialized for model {LLM_MODEL_STRING} and result_type {TargetModel.__name__}")

            # 2. Define the tool function(s) and associate with this agent instance
            #    Needs to be defined here to be bound to the current 'agent' instance.
            # @agent.tool
            @agent.tool_plain
            async def analyze_sentiment(text_to_analyze: str) -> str:
            # def analyze_sentiment(ctx: RunContext[str], text_to_analyze: str) -> str:
                """
                Analyzes the sentiment of the provided text: text_to_analyze.
                Returns 'Positive', 'Negative', or 'Neutral'.
                """
                print("SREE")
                logger.info(f"SREE analyze_sentiment: {text_to_analyze}")
                # logger.info(f"SREE2 analyze_sentiment: {ctx.deps}")
                text_lower = text_to_analyze.lower()
                # Basic keyword-based sentiment analysis (replace with a real model if needed)
                positive_keywords = ["good", "great", "excellent", "fantastic", "happy", "love", "like", "best", "wonderful"]
                negative_keywords = ["bad", "poor", "terrible", "awful", "sad", "hate", "dislike", "worst"]

                pos_count = sum(keyword in text_lower for keyword in positive_keywords)
                neg_count = sum(keyword in text_lower for keyword in negative_keywords)

                return f"text_to_analyze: {text_to_analyze} pos:{pos_count} neg: {neg_count}"

                if pos_count > neg_count:
                    logger.info("Sentiment tool returning: Positive")
                    return "Positive"
                elif neg_count > pos_count:
                    logger.info("Sentiment tool returning: Negative")
                    return "Negative"
                else:
                    logger.info("Sentiment tool returning: Neutral")
                    return "Neutral"

            logger.info(f"Tool 'analyze_sentiment' registered with agent.")

            @agent.tool()
            async def my_random_number(
                # ctx: RunContext, num1: int
                ctx: RunContext #, num1: int
            ) -> int:
              """Creates a random number.

              Args:
                  ctx: The context (not utilized in this tool).
              """

              # Generate a random integer between 1 and 10 (inclusive)
              random_number = random.randint(1, 19)
              print(f"MY RANDOM {random_number}")
              return random_number

            # 3. Construct the prompt for the agent
            #    Clearly ask for both extraction and sentiment analysis
            prompt = (
                f"Please extract the relevant information according to the '{selected_model_name}' structure "
                f"AND analyze the overall sentiment of the following text: \n\n'{input_text}' using the tool"
            )

            logger.info(f"Tool 'my_random_number' registered with agent.")

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            @agent.tool()
            async def mcp_stdio_client(ctx: RunContext) -> str:
                """Creates a poem.
                """
                server_params = StdioServerParameters(
                    # command='uv', args=['run', 'mcp_server.py', 'server'], env=os.environ
                    command='python', args=['mcp_server.py', 'server'], env=os.environ
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return await session.call_tool('poet', {'theme': 'socks'})
            logger.info(f"MCP Tool 'mcp_stdio_server' registered with agent.")

            @agent.tool()
            async def mcp_sse_client2(ctx: RunContext, query: str) -> str:
                """
                booking: Makes a booking for the given name.        
                """
                from pydantic_ai.mcp import MCPServerHTTP
                # server = MCPServerHTTP(url='http://host.docker.internal:3001/sse') 
                server = MCPServerHTTP(url='http://sse-server-py-2:3002/sse')   
                agent = Agent(
                            'openai:gpt-4o-mini', 
                            mcp_servers=[server]
                        )  
                async with agent.run_mcp_servers():  
                    return await agent.run(query) #https://httpstat.us/200 https://httpbin.org/anything
            logger.info(f"MCP Tool 'mcp_sse_server' registered with agent.")
 
            @agent.tool()
            async def mcp_sse_client(ctx: RunContext, query: str) -> str:
                """
                fetch: Fetches and returns the contents of the website url.
                """
                from pydantic_ai.mcp import MCPServerHTTP
                # server = MCPServerHTTP(url='http://host.docker.internal:3001/sse') 
                server = MCPServerHTTP(url='http://sse-server-py:3001/sse')   
                agent = Agent(
                            'openai:gpt-4o-mini', 
                            mcp_servers=[server]
                        )  
                async with agent.run_mcp_servers():  
                    return await agent.run(query) #https://httpstat.us/200 https://httpbin.org/anything
            logger.info(f"MCP Tool 'mcp_sse_server2' registered with agent.")


            # 4. Run the agent synchronously with the combined prompt
            with st.spinner("Agent is processing the request (may use tools)..."):
                # The agent handles prompting, LLM call (potentially including tool calls/results), parsing, and validation
                result = agent.run_sync(prompt)
                # result object contains .data (the Pydantic model instance) and .usage()

            extracted_data = result.data # Access the Pydantic object

            # --- Display Results ---
            st.success(f"Agent successfully processed the request!")
            st.subheader("Result (Pydantic Object):")

            # Display as dictionary/JSON for readability
            if extracted_data:
                 st.json(extracted_data.model_dump_json(indent=2)) # Should include sentiment if tool was used
                 st.subheader("Raw Pydantic Object:")
                 st.write(extracted_data)
                 # Check if sentiment was populated (indication tool might have been used and incorporated)
                 if hasattr(extracted_data, 'sentiment') and extracted_data.sentiment:
                     st.info(f"Sentiment analysis result ('{extracted_data.sentiment}') was included in the output.")
                 else:
                     st.info("Sentiment field was not populated in the final output.")

                 st.subheader("LLM Usage Info:")
                 st.write(result.usage())
            else:
                 st.warning("The agent returned no structured data, although no error occurred.")


        except ValidationError as e:
            st.error(f"Data Validation Error: LLM output did not match the '{selected_model_name}' schema after processing.")
            st.json(e.errors())
            logger.error(f"Validation Error: {e}")
        except AuthenticationError as e: # Catch specific errors if needed
            logger.error(f"OpenAI Authentication Error during Agent run: {e}")
            st.error(f"OpenAI Authentication Error: {e}. Please check your API key.")
        except RateLimitError as e:
             logger.error(f"OpenAI Rate Limit Error during Agent run: {e}")
             st.error(f"OpenAI Rate Limit Error: {e}. Check usage limits or try again later.")
        except Exception as e:
            # Catch-all for other potential errors during Agent execution
            logger.error(f"An unexpected error occurred during Agent run: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {type(e).__name__}: {e}")
            if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                 st.error("This might be related to your API key configuration.")

else:
    st.info("Enter text and click the button to start extraction and sentiment analysis. e.g: Fetch website: https://httpstat.us/200. Can you create a poem about a cat. Book for user Sree")

st.markdown("---")
st.markdown("Powered by [Pydantic_AI](https://github.com/pydantic/pydantic-ai), [Streamlit](https://streamlit.io)")