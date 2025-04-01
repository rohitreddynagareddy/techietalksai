import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import asyncio # Required for running async agent functions

# --- Pydantic AI ---
from pydantic_ai import Agent # Use Agent
# from pydantic_ai.langchain import tool
from langchain.tools import tool

from pydantic_ai.messages import UserPrompt, AssistantResponse, SystemPrompt, Message # Import message types

# --- LLM Provider ---
# pydantic-ai handles client creation internally when using model strings like 'openai:gpt-4o'
# Ensure OPENAI_API_KEY is in the environment for this to work seamlessly.

# --- Langchain (for Vector DB) ---
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- Utilities ---
import math
import logging
from typing import List, Dict, Any, AsyncGenerator

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
# Make sure OPENAI_API_KEY is available in your environment or .env file
load_dotenv()

# --- Constants ---
VECTOR_STORE_PATH = "faiss_vector_store"
DEFAULT_MODEL = "openai:gpt-4o" # Or choose another model like 'openai:gpt-3.5-turbo'

# === TOOL DEFINITIONS (Keep as before) ===

@tool()
def calculator(expression: str) -> str:
    """
    Calculates the result of a mathematical expression.
    Use this tool for any mathematical calculations like addition, subtraction, multiplication, division, exponentiation, etc.
    Only use standard Python mathematical operators and functions from the 'math' module.
    Example: '2 + 2', 'math.sqrt(16) * 5'
    """
    try:
        allowed_globals = {"math": math, "__builtins__": {}}
        allowed_locals = {}
        result = eval(expression, allowed_globals, allowed_locals)
        return f"The result of '{expression}' is {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}. Please ensure the expression is valid Python math."

@tool()
def query_knowledge_base(query: str) -> str:
    """
    Queries the vector knowledge base created from uploaded PDF documents.
    Use this tool ONLY when the user asks questions specifically about the content
    of the uploaded PDF files. Provide the user's query exactly as they typed it.
    Do not use this tool for general knowledge questions.
    """
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        return "Knowledge base is not initialized or empty. Please upload and process PDF documents first."
    try:
        logger.info(f"Querying knowledge base with: {query}")
        vector_store = st.session_state.vector_store
        results = vector_store.similarity_search(query, k=3) # Get top 3 relevant chunks
        relevant_docs = [doc.page_content for doc in results]

        if not relevant_docs:
            return f"I couldn't find relevant information for '{query}' in the uploaded documents."

        context = "\n---\n".join(relevant_docs)
        logger.info(f"Found context: {context[:500]}...")
        # Return context for the LLM to synthesize the final answer
        return f"Based on the documents, here's relevant information regarding '{query}':\n\n{context}"

    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        return f"An error occurred while querying the knowledge base: {e}"

# === Helper Functions ===

def get_api_key():
    """Gets the API key from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for deployed apps)
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        # Fallback to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
    return api_key

# Vector Store Processing (Keep as before, ensure API key is passed/available)
def process_pdf_files(pdf_files, api_key):
    """Loads, splits, embeds, and stores PDF documents in the vector store."""
    if not pdf_files:
        st.warning("No PDF files uploaded.")
        return None
    if not api_key:
        st.error("OpenAI API Key is required for embedding PDFs.")
        return None

    try:
        embeddings_model = OpenAIEmbeddings(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize embeddings model: {e}. Check your API key.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    all_docs = []
    temp_dir = tempfile.TemporaryDirectory()

    with st.spinner("Processing PDF documents..."):
        for pdf_file in pdf_files:
            try:
                temp_filepath = os.path.join(temp_dir.name, pdf_file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(pdf_file.getvalue())
                logger.info(f"Loading PDF: {pdf_file.name}")
                loader = PyPDFLoader(temp_filepath)
                documents = loader.load()
                split_docs = text_splitter.split_documents(documents)
                all_docs.extend(split_docs)
                logger.info(f"Processed {len(split_docs)} chunks from {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing {pdf_file.name}: {e}")
        temp_dir.cleanup()

    if not all_docs:
        st.warning("Could not extract any text from the uploaded PDF(s).")
        return None

    try:
        with st.spinner("Creating vector embeddings and storing..."):
            vector_store = FAISS.from_documents(all_docs, embeddings_model)
            # If you need persistence, save/load logic would go here
            # vector_store.save_local(VECTOR_STORE_PATH)
            st.session_state.vector_store = vector_store # Update session state
            logger.info(f"Created new vector store with {len(all_docs)} chunks.")
        st.success(f"Knowledge base created/updated with {len(all_docs)} text chunks from {len(pdf_files)} PDF(s).")
        return st.session_state.vector_store
    except Exception as e:
        st.error(f"Error creating/updating vector store: {e}")
        return None

def convert_streamlit_history_to_pydantic(history: List[Dict[str, str]]) -> List[Message]:
    """Converts Streamlit chat history dict to list of pydantic-ai Message objects."""
    messages: List[Message] = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(UserPrompt(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AssistantResponse(content=msg["content"]))
        # Add handling for 'system' role if you use it
        # elif msg["role"] == "system":
        #     messages.append(SystemPrompt(content=msg["content"]))
    return messages

async def stream_agent_response(agent: Agent, user_prompt: str, history_messages: List[Message]) -> AsyncGenerator[str, None]:
    """Runs the agent asynchronously and yields text chunks."""
    try:
        async with agent.run_stream(user_prompt, message_history=history_messages) as result:
            async for text_chunk in result.stream():
                yield text_chunk
            # After streaming, you could potentially access final structured info
            # from `result` if needed, e.g., tool calls details.
            # final_response = result.final_response # Hypothetical attribute
            # logger.info(f"Agent finished. Final response object: {final_response}")
    except Exception as e:
        logger.error(f"Error during agent streaming: {e}", exc_info=True)
        yield f"\n\nSorry, an error occurred: {e}" # Yield error message as part of the stream


# === Streamlit App ===

st.set_page_config(page_title="AI Chatbot Agent", layout="wide")
st.title("🤖 AI Chatbot Agent with Tools")
st.caption(f"Powered by Pydantic AI ({DEFAULT_MODEL}), Langchain & Streamlit")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores {"role": str, "content": str}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "agent_instance" not in st.session_state:
    st.session_state.agent_instance = None
if "api_key" not in st.session_state:
    # Try to get API key from secrets/env on first run
    st.session_state.api_key = get_api_key()


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # API Key Input - Allow overriding the env/secret key
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        key="api_key_input_widget", # Use a distinct key for the widget
        value=st.session_state.api_key or "", # Show current key (masked) or empty
        help="Leave blank to use key from environment/secrets, or enter a new key.",
    )

    # Update API key in session state and environment if changed
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        os.environ["OPENAI_API_KEY"] = api_key_input # Update env var for pydantic-ai
        st.session_state.agent_instance = None # Force re-initialization
        st.success("API Key updated.")
        # No need to explicitly validate here, Agent init will fail if invalid

    # Initialize Agent if not already done and API key is available
    if not st.session_state.agent_instance and st.session_state.api_key:
        try:
            # Set the API key in the environment if it's not already there
            if not os.getenv("OPENAI_API_KEY"):
                 os.environ["OPENAI_API_KEY"] = st.session_state.api_key

            st.session_state.agent_instance = Agent(
                DEFAULT_MODEL, # Use the model string
                tools=[calculator, query_knowledge_base],
                debug=True # Shows reasoning and tool calls in console logs
            )
            st.success(f"Agent initialized with {DEFAULT_MODEL}.")
        except Exception as e:
            st.error(f"Failed to initialize Agent: {e}. Check API Key and model name.")
            st.session_state.agent_instance = None # Ensure it's None on failure
            st.session_state.api_key = None # Clear potentially bad key
            os.environ.pop("OPENAI_API_KEY", None) # Remove bad key from env

    elif not st.session_state.api_key:
        st.warning("Please provide your OpenAI API Key.")


    st.divider()
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader",
        disabled=not st.session_state.api_key # Disable if no API key
    )

    if st.button("Process Uploaded PDFs", key="process_pdfs", disabled=not uploaded_files or not st.session_state.api_key):
        # process_pdf_files updates st.session_state.vector_store internally now
        process_pdf_files(uploaded_files, st.session_state.api_key)

    # Display Knowledge Base Status
    if st.session_state.vector_store is not None:
        st.success("✅ Knowledge Base Ready")
    else:
        st.info("ℹ️ Upload PDFs and click 'Process' to build the Knowledge Base.")

    st.divider()
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything...", disabled=not st.session_state.agent_instance):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare history for the agent
    history_pydantic = convert_streamlit_history_to_pydantic(st.session_state.messages[:-1]) # Exclude the last user prompt

    # Get and stream AI response
    with st.chat_message("assistant"):
        # Use st.write_stream which handles the async generator
        try:
            response_generator = stream_agent_response(
                st.session_state.agent_instance,
                prompt,
                history_pydantic
            )
            # st.write_stream returns the final concatenated string
            full_response = st.write_stream(response_generator)

            # Add the complete assistant response to history AFTER streaming
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred while getting the response: {e}")
            logger.error(f"Error in Streamlit chat loop: {e}", exc_info=True)
            # Add error message to chat history
            error_message = f"Sorry, I encountered an error processing your request: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            # We need to rerun to show the error message added to the state
            st.rerun()

# Add a placeholder if agent is not ready
if not st.session_state.agent_instance:
    st.info("Please provide a valid OpenAI API Key in the sidebar to initialize the chatbot.")