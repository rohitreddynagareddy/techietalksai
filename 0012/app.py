import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.messages import ModelMessage
from pydantic_ai import Agent, ModelRetry, RunContext
from rich import print
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from typing import List
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dataclasses import dataclass
import random
import string

# Apply nest_asyncio first to handle event loops
nest_asyncio.apply()

# At the top of your Streamlit app
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize RAG components
@st.cache_resource
def initialize_rag():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Create vector store directory if not exists
    os.makedirs("data/vector_store", exist_ok=True)
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/texts", exist_ok=True)
    
    return embeddings, text_splitter

def process_pdfs():
    embeddings, text_splitter = initialize_rag()
    pdf_folder = "data/pdfs"
    
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            pages = loader.load_and_split(text_splitter)
            documents.extend(pages)
    
    if documents:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="data/vector_store"
        )
    return bool(documents)

def process_text_files():
    embeddings, text_splitter = initialize_rag()
    text_folder = "data/texts"
    
    documents = []
    for text_file in os.listdir(text_folder):
        if text_file.endswith(".txt"):
            loader = TextLoader(os.path.join(text_folder, text_file))
            pages = loader.load_and_split(text_splitter)
            documents.extend(pages)
    
    if documents:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="data/vector_store"
        )
    return bool(documents)

def process_website(url: str):
    embeddings, text_splitter = initialize_rag()
    
    # Load website content
    loader = WebBaseLoader(url)
    documents = loader.load_and_split(text_splitter)
    
    if documents:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="data/vector_store"
        )
    return bool(documents)

# Initialize RAG system on app start
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = process_pdfs() or process_text_files()

# Add this after page config
st.markdown("""
<style>
    .stButton>button {
        background-color: #f0f2f6;
        color: #2c3e50;
        border: 1px solid #ced4da;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e2e6ea;
        border-color: #adb5bd;
    }
</style>
""", unsafe_allow_html=True)

# Model selection and new chat button
col1, col2 = st.columns([3, 1])
with col1:
    MODEL_CHOICE = st.selectbox("Choose AI Model", ["OpenAI", "DeepSeek", "Gemini"])

with col2:
    if st.button("🔄 New Chat", help="Start a new conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_history = []
        st.rerun()

# Add PDF, text file, and website upload section to sidebar
with st.sidebar:
    st.header("Knowledge Base Management")
    
    # PDF Upload
    uploaded_pdfs = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_pdfs:
        for uploaded_file in uploaded_pdfs:
            file_path = os.path.join("data/pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.rag_initialized = process_pdfs()
        st.success(f"Processed {len(uploaded_pdfs)} new PDF documents!")
    
    # Text File Upload
    uploaded_texts = st.file_uploader(
        "Upload Text documents", 
        type=["txt"],
        accept_multiple_files=True
    )
    
    if uploaded_texts:
        for uploaded_file in uploaded_texts:
            file_path = os.path.join("data/texts", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.rag_initialized = process_text_files()
        st.success(f"Processed {len(uploaded_texts)} new text documents!")
    
    # Website URL Input
    website_url = st.text_input("Add Website URL", placeholder="https://example.com")
    if website_url:
        if st.button("Process Website"):
            with st.spinner("Processing website content..."):
                success = process_website(website_url)
                if success:
                    st.success("Website content processed successfully!")
                else:
                    st.error("Failed to process website content.")

# Modify the Agent creation to include RAG context
def get_rag_context(query: str, k: int = 3) -> str:
    if not st.session_state.rag_initialized:
        return ""
    
    embeddings, _ = initialize_rag()
    db = Chroma(
        persist_directory="data/vector_store",
        embedding_function=embeddings
    )
    
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

# Load environment variables
load_dotenv()

# Validate API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in the environment or .env file.")
    st.stop()
if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY is not set in the environment or .env file.")
    st.stop()
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set in the environment or .env file.")
    st.stop()

# Initialize selected model
if MODEL_CHOICE == "OpenAI":
    model = OpenAIModel(
        model_name='gpt-4o-mini',
        base_url='https://api.openai.com/v1',
        api_key=OPENAI_API_KEY,
    )
elif MODEL_CHOICE == "DeepSeek":
    model = OpenAIModel(
        model_name='deepseek-chat',
        base_url='https://api.deepseek.com/v1',
        api_key=DEEPSEEK_API_KEY,
    )
elif MODEL_CHOICE == "Gemini":
    model = GeminiModel(
        model_name='gemini-1.5-flash',
        api_key=GEMINI_API_KEY,
    )

# Response structure
class AIResponse(BaseModel):
    content: str
    category: str = "general"

# Initialize session states
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Create agent
@dataclass
class Deps:
    test_api_key: str | None

agent = Agent(
    model=model,
    result_type=AIResponse,
    system_prompt="You're a helpful assistant. Respond conversationally and keep answers concise.",
    deps_type=Deps,
    retries=3,
)

@agent.tool
async def retriever(ctx: RunContext[Deps], query: str) -> float:
    """Retrieve documentation sections based on a search query."""
    observation_relevant_chunks = get_rag_context(query, 3)
    observation_text = f"Observation: {observation_relevant_chunks}"
    conversation = []
    conversation.append({"role": "user", "content": f"{query}"})
    conversation.append({"role": "assistant", "content": observation_text})
    return conversation

# @agent.tool
# async def booking(ctx: RunContext[Deps], name: str = "Guest") -> str:
#     """Generate a random booking reference number as confirmation."""
#     # Generate a random alphanumeric booking reference
#     booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
#     return f"Your booking is confirmed, {name}! Booking Reference: {booking_ref}"
@agent.tool
async def booking(ctx: RunContext[Deps], name: str = "Guest") -> str:
    """Generate a random booking reference number as confirmation."""
    if name == "Guest":
        # Prompt the user to provide their name
        return "Please provide your name to complete the booking."
    else:
        # Generate a random alphanumeric booking reference
        booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        return f"Your booking is confirmed, {name}! Booking Reference: {booking_ref}"

# Model-specific avatars
MODEL_AVATARS = {
    "OpenAI": "🦾",  # Robot arm emoji
    "DeepSeek": "🚀",  # Rocket emoji
    "Gemini": "🤖"   # Robot face emoji
}

# UI Setup
st.title("💬 Multi-Model Chat Assistant")
st.caption(f"Currently using: {MODEL_CHOICE} {MODEL_AVATARS[MODEL_CHOICE]}")

# Update the chat history display
for message in st.session_state.messages:
    avatar = MODEL_AVATARS[message.get("model")] if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Generating response..."):
            # Event loop management
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Execute async operation
            result = loop.run_until_complete(
                agent.run(
                    f"{prompt}", 
                    message_history=st.session_state.message_history,
                )
            )
            
            # Update message history
            new_messages = result.new_messages()
            if MODEL_CHOICE == "OpenAI":
                for msg in new_messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if len(tool_call.id) > 40:
                                tool_call.id = tool_call.id[:40]
            
            st.session_state.message_history.extend(new_messages)

            print("\n[bold]Message History:[/bold]")
            for i, msg in enumerate(st.session_state["message_history"]):
                print(f"\n[yellow]--- Message {i+1} ---[/yellow]")
                print(msg)

        # Display assistant response
        with st.chat_message("assistant", avatar=MODEL_AVATARS[MODEL_CHOICE]):
            st.markdown(result.data.content)        

        # Add to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.data.content,
            "model": MODEL_CHOICE  # Store model info with the message
        })
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}"
        })