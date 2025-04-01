# app/main.py
import streamlit as st
import os
import logging
from pydantic import ValidationError

# Corrected import for the Agent class
from pydantic_ai import Agent

from openai import OpenAI, AuthenticationError, RateLimitError

# Import your Pydantic models
from models import UserDetails, ProductInfo # Use relative import within the package

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LLM Client Setup ---
# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client (handle potential missing key)
llm_client = None
if openai_api_key:
    try:
        llm_client = OpenAI(api_key=openai_api_key)
        # Test connection with a lightweight call (optional but good practice)
        # llm_client.models.list()
        logger.info("OpenAI client initialized successfully.")
    except AuthenticationError:
        logger.error("OpenAI Authentication Error: Invalid API Key.")
        st.error("OpenAI Authentication Error: Invalid API Key provided in .env file.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        st.error(f"Failed to initialize OpenAI client: {e}")
else:
    logger.warning("OPENAI_API_KEY environment variable not found.")
    st.warning("OpenAI API Key not found. Please set it in the .env file and restart.")

# --- PydanticAI Setup ---
# Select the LLM model (use a cost-effective one for demos if possible)
LLM_MODEL_STRING = "gpt-4o-mini"

# Check if the provider is OpenAI and if the key is missing
is_openai_provider = LLM_MODEL_STRING.startswith("openai:")
show_key_error = is_openai_provider and not openai_api_key

# Dictionary mapping model names to model classes for dynamic selection
AVAILABLE_MODELS = {
    "User Details": UserDetails,
    "Product Info": ProductInfo,
}

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📄 Pydantic_AI Agent Demo")
st.markdown(f"""
This app demonstrates using `pydantic_ai.Agent` with an LLM (configured as `{LLM_MODEL_STRING}`)
to extract structured data from unstructured text based on Pydantic models.
""")

if show_key_error:
    st.error("OpenAI API Key is missing in the `.env` file. Please add it and restart the application.")
    st.stop() # Stop execution if key is missing for configured provider

# Input Area
st.header("Input Text")
input_text = st.text_area("Enter text from which to extract information:", height=150, value="Example: John Smith is a 42 year old engineer living in London. He bought a new laptop for $1200. His email addreess is abcd@xyz.com")

# Model Selection
st.header("Select Extraction Model")
selected_model_name = st.selectbox("Choose the Pydantic model (`result_type`) for extraction:", options=list(AVAILABLE_MODELS.keys()))
TargetModel = AVAILABLE_MODELS[selected_model_name]

# Display selected model schema for clarity
with st.expander("View Selected Model Schema (`result_type`)"):
    st.json(TargetModel.model_json_schema())

# Extraction Button
if st.button("✨ Extract Information"):
    if not input_text:
        st.warning("Please enter some text to extract information from.")
    else:
        st.info(f"Asking Agent (LLM: {LLM_MODEL_STRING}) to extract '{selected_model_name}'...")
        try:
            # --- Core Pydantic_AI Agent Logic ---
            # 1. Instantiate the Agent
            #    - Pass the LLM model string (e.g., "openai:gpt-3.5-turbo")
            #    - Pass the desired Pydantic model class as result_type
            agent = Agent(LLM_MODEL_STRING, result_type=TargetModel)
            logger.info(f"Agent initialized for model {LLM_MODEL_STRING} and result_type {TargetModel.__name__}")

            # 2. Run the agent synchronously with the input text
            with st.spinner("Agent is processing the request..."):
                # The agent handles prompting, LLM call, parsing, and validation
                result = agent.run_sync(input_text)
                # result object contains .data (the Pydantic model instance) and .usage()

            extracted_data = result.data # Access the Pydantic object

            # --- Display Results ---
            st.success(f"Agent successfully extracted data as '{selected_model_name}'!")
            st.subheader("Extracted Data (Pydantic Object):")

            # Display as dictionary/JSON for readability
            if extracted_data:
                 st.json(extracted_data.model_dump_json(indent=2))
                 st.subheader("Raw Pydantic Object:")
                 st.write(extracted_data)
                 st.subheader("LLM Usage Info:")
                 st.write(result.usage())
            else:
                 st.warning("The agent returned no structured data, although no error occurred.")


        except ValidationError as e:
            st.error(f"Data Validation Error: LLM output did not match the '{selected_model_name}' schema after processing.")
            # Pydantic_ai might provide more context in the exception, or just the Pydantic error
            st.json(e.errors())
            logger.error(f"Validation Error: {e}")
        # Catch potential errors from the underlying LLM library (like OpenAI's) if needed
        # except AuthenticationError:
        #     logger.error("OpenAI Authentication Error during Agent run.")
        #     st.error("OpenAI Authentication Error: Invalid API Key. Please check the .env file.")
        # except RateLimitError:
        #      logger.error("OpenAI Rate Limit Error during Agent run.")
        #      st.error("OpenAI Rate Limit Error: Check usage limits or try again later.")
        except Exception as e:
            # Catch-all for other potential errors during Agent execution
            logger.error(f"An unexpected error occurred during Agent run: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {type(e).__name__}: {e}")
            # If it's an OpenAI auth error, provide a specific hint
            if "AuthenticationError" in str(type(e)):
                 st.error("This might be an OpenAI Authentication Error. Please verify your API key in the .env file.")

else:
    st.info("Enter text and click the button to start extraction.")

st.markdown("---")
st.markdown("Powered by [Pydantic_AI](https://github.com/pydantic/pydantic-ai), [Streamlit](https://streamlit.io)")

# Add an empty __init__.py to make 'app' a package
# touch app/__init__.py