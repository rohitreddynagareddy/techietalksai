from typing import Iterator, List
import streamlit as st
from textwrap import dedent
import os
from dotenv import load_dotenv
import random
import string
from agno.agent import Agent, RunResponse
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.vectordb.pgvector import PgVector
import nest_asyncio
from agno.models.groq import Groq
import time

from agno.document.chunking.recursive import RecursiveChunking
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.document import DocumentChunking
from agno.document.chunking.semantic import SemanticChunking

from agno.vectordb.lancedb import LanceDb, SearchType
from agno.vectordb.pineconedb import PineconeDb

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logger.info("===== App Start =====")
# logger.warning("This is a warning message printed in Docker logs")


# Apply nest_asyncio for async handling
nest_asyncio.apply()

# Set up page config
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize environment
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
# Validate API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY]):
    st.error("Missing required API keys in environment variables")
    st.stop()

# Model avatars
MODEL_AVATARS = {
    "OpenAI": "🦾",
    "DeepSeek": "🚀",
    "Gemini": "🤖"
}

def booking_tool(name: str = "Guest") -> str:
    """Use this function to Generate Thai Hotel booking or appointment confirmation, returns booking reference number

    Args:
        name (str): name of the user or Guest.

    Returns:
        str: string containing the booking reference number
    """
    if name == "Guest":
        return "Please provide your name to complete the booking."
    booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"Your booking is confirmed, {name}! Booking Reference: {booking_ref}"

# Initialize RAG knowledge base
# def init_knowledge_lancedb(urls: List[str] = None) -> PDFUrlKnowledgeBase:
#     return PDFUrlKnowledgeBase(
#         urls=urls or [],
#         vector_db=LanceDb(
#             uri="data/vector_store",
#             table_name="knowledge_base",
#             search_type=SearchType.hybrid,
#             chunking_strategy=chunking_strategy,
#             embedder=OpenAIEmbedder(
#                 id="text-embedding-3-small",
#                 api_key=OPENAI_API_KEY
#             ),
#         ),
#     )

if "search_type_selected" not in globals():
    search_type_selected = "Hybrid PostgreSQL"


def init_knowledge() -> PDFUrlKnowledgeBase:
    urls = st.session_state["knowledge_urls"]

    if chunking_strategy_selected == "Agentic":
        chunking_strategy = AgenticChunking()
    elif chunking_strategy_selected == "Fixed":
        chunking_strategy = FixedSizeChunking()
    elif chunking_strategy_selected == "Recursive":
        chunking_strategy = RecursiveChunking()
    elif chunking_strategy_selected == "Document":
        chunking_strategy = DocumentChunking()
    elif chunking_strategy_selected == "Semantic":
        chunking_strategy = SemanticChunking()
    else:
        chunking_strategy = FixedSizeChunking()


    # st.success(urls)
    if search_type_selected == "Hybrid PostgreSQL":
        db_url = "postgresql+psycopg://ai:ai@pgvector:5432/ai"
        logger.info("===== init_knowledge PG =====")
        return PDFUrlKnowledgeBase(
            # urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
            urls=urls or [],
            vector_db=PgVector(
                table_name="recipes", 
                db_url=db_url,
            ),
            chunking_strategy=chunking_strategy,
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=OPENAI_API_KEY
            ),
        )
    elif search_type_selected == "Hybrid LanceDB":
        logger.info("===== init_knowledge Lance =====")
        return PDFUrlKnowledgeBase(
            urls=urls or [],
            vector_db=LanceDb(
                uri="data/vector_store",
                table_name="knowledge_base",
                search_type=SearchType.hybrid,
                embedder=OpenAIEmbedder(
                    id="text-embedding-3-small",
                    api_key=OPENAI_API_KEY
                ),
            ),
            chunking_strategy=chunking_strategy,
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=OPENAI_API_KEY
            ),
        )
    elif search_type_selected == "Hybrid Pinecone":
        logger.info("===== init_knowledge Pine =====")
        return PDFUrlKnowledgeBase(
            urls=urls or [],
            vector_db = PineconeDb(
                name="thai-recipe",
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
                api_key=api_key,
                use_hybrid_search=True,
                hybrid_alpha=0.5,
            ),
            chunking_strategy=chunking_strategy,
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=OPENAI_API_KEY
            ),
        )
    logger.info("===== init_knowledge fails =====")

# def init_knowledge_pg(urls: List[str] = None) -> PDFUrlKnowledgeBase:
# def init_knowledge_pg() -> PDFUrlKnowledgeBase:
# # if 'knowledge_base' not in st.session_state:
#     # with st.spinner("📚 Loading Thai recipe database..."):
#     #     try:
#             db_url = "postgresql+psycopg://ai:ai@pgvector:5432/ai"
#             urls = st.session_state["knowledge_urls"]
#             return PDFUrlKnowledgeBase(
#                 # urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#                 urls=urls or [],
#                 vector_db=PgVector(
#                     table_name="recipes", 
#                     db_url=db_url,
#                 ),
#                 search_type=SearchType.hybrid,
#                 embedder=OpenAIEmbedder(
#                     id="text-embedding-3-small",
#                     api_key=OPENAI_API_KEY
#                 ),
#             )
#             # knowledge_base.load(recreate=False)
def load_knowledge(recreate_tf):
    logger.info("===== load_knowledge =====")
    if "knowledge_db" in st.session_state:
        with st.spinner("Initializing knowledge base..."):
            # time.sleep(10)
            logger.info("===== load_knowledge init =====")
            st.session_state["knowledge_db"] = init_knowledge()
            logger.info("===== load_knowledge load =====")
            st.session_state.knowledge_db.load(recreate=recreate_tf)
            logger.info("===== load_knowledge loaded =====")
            # st.success("Loaded Knowledge..")
        return
    logger.info("===== load_knowledge returns =====")
    # st.success("Not Loaded DB")

if "chunking_strategy_selected" not in globals():
    chunking_strategy_selected = "Fixed"


# Initialize agent with dynamic knowledge
def create_agent(model_choice: str):
    # Select model
    if model_choice == "OpenAI":
        model = OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY)
    elif model_choice == "DeepSeek":
        model = DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY)
    elif model_choice == "Gemini":
        model = Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY)

    if reason:
        return Agent(
            model=model,
            instructions=dedent("""
                You're a helpful assistant. Respond conversationally and keep answers concise.
                Follow these steps:
                1. For bookings use only the tools provided, like the bookingtool
                2. Check knowledge base for technical information as second priority
                3. Use web search for real-time data as third priority
                4. Cite sources when using external information

                When presenting mathematical expressions:
                - Use double dollar signs for block equations: $$\\frac{3}{5} = 0.6$$
                - Use escaped percentage signs: 60\\%
                - For inline equations: $x^2 + y^2 = z^2$

                When asked to book use the booking tool do not reason
            """),
            # knowledge=init_knowledge(knowledge_urls),
            knowledge=st.session_state.knowledge_db, #init_knowledge_pg(knowledge_urls),
            tools=[booking_tool, DuckDuckGoTools()],
            # tools=[booking_tool],
            # reasoning_model=DeepSeek(id="deepseek-reasoner"),
            # reasoning_model=OpenAIChat(id="o1-mini"),
            reasoning_model=Groq(
                id="deepseek-r1-distill-llama-70b", temperature=0.6, max_tokens=1024, top_p=0.95
            ),
            show_tool_calls=True,
            markdown=True,
            add_references=True,
        )
    else:
        return Agent(
            model=model,
            instructions=dedent("""
                You're a helpful assistant. Respond conversationally and keep answers concise.
                Follow these steps:
                1. For bookings use only the tools provided, like the bookingtool
                2. Check knowledge base for technical information as second priority
                3. Use web search for real-time data as third priority
                4. Cite sources when using external information

                When presenting mathematical expressions:
                - Use double dollar signs for block equations: $$\\frac{3}{5} = 0.6$$
                - Use escaped percentage signs: 60\\%
                - For inline equations: $x^2 + y^2 = z^2$

                When asked to book use the booking tool do not reason
            """),
            # knowledge=init_knowledge(knowledge_urls),
            knowledge=st.session_state.knowledge_db, #init_knowledge_pg(knowledge_urls),
            tools=[booking_tool, DuckDuckGoTools()],
            # tools=[booking_tool],
            # reasoning_model=DeepSeek(id="deepseek-reasoner"),
            # reasoning_model=OpenAIChat(id="o1-mini"),
            # reasoning_model=Groq(
            #     id="deepseek-r1-distill-llama-70b", temperature=0.6, max_tokens=1024, top_p=0.95
            # ),
            show_tool_calls=True,
            markdown=True,
            add_references=True,
        )


if "knowledge_urls" not in st.session_state:
    st.session_state["knowledge_urls"] = []
    # st.session_state["knowledge_urls"] = ["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"]
    # st.session_state["knowledge_urls"] = ['file:///app/data/uploads/ThaiRecipes.pdf']

if "knowledge_db" not in st.session_state:
    # urls = len(st.session_state.knowledge_urls)
    # st.success(f"Count {urls} URLs")
    st.session_state["knowledge_db"] = init_knowledge()

# UI Components
st.title("💬 Flexible VectorDB and Chunking Multi-Model Agno Chat Agent")
st.write(f"**Chunking Strategy:** {chunking_strategy_selected} | **VectorDB:** {search_type_selected}" )
# model_choice = st.selectbox("Choose AI Model", list(MODEL_AVATARS.keys()))
# st.caption(f"Currently using: {model_choice} {MODEL_AVATARS[model_choice]}")

# File management sidebar
with st.sidebar:
    st.title("💬 Chatbot Options")
    st.sidebar.subheader("Model Selection")
    model_choice = st.selectbox("Choose AI Model", list(MODEL_AVATARS.keys()))
    st.caption(f"Currently using: {model_choice} {MODEL_AVATARS[model_choice]}")

    st.sidebar.subheader("Reasoning Yes/No")
    reason = st.sidebar.checkbox("Reasoning")
    
    # Function to show loading when chunking strategy changes
    def chunking_strategy_change():
        with st.sidebar:
            with st.spinner("Loading new chunking strategy..."):
                load_knowledge(True)  # Reload knowledge with new strategy
                time.sleep(5)  # Simulate loading time

    def search_type_change():
        with st.sidebar:
            with st.spinner("Loading new search type..."):
                load_knowledge(True)  # Reload knowledge with new strategy
                time.sleep(3)  # Simulate loading time

    # Add chunking strategy selector with loading indicator
    st.sidebar.subheader("Chunking Strategy")
    chunking_strategy_selected = st.sidebar.radio(
        "Select Chunking Method",
        options=["Agentic", "Fixed", "Recursive", "Document", "Semantic"],
        index=1,
        key="chunking_strategy_radio",
        on_change=chunking_strategy_change
    )
    
    # Add search type selector
    st.sidebar.subheader("Search Type")
    search_type_selected = st.sidebar.radio(
        "Search Type",
        options=["Hybrid PostgreSQL", "Hybrid LanceDB", "Hybrid Pinecone"],
        index=0,
        key="search_type_radio",
        on_change=search_type_change
    )

    st.header("Knowledge Base Management")
    
    # PDF Upload
    # uploaded_files = st.file_uploader(
    #     "Upload Documents",
    #     type=["pdf", "txt"],
    #     accept_multiple_files=True
    # )
    
    # Website URL
    website_url = st.text_input("Add PDF URL", placeholder="https://example.com/abcd.pdf")
    
    # Process new content
    if st.button("Update Knowledge Base"):
        knowledge_urls = st.session_state.get("knowledge_urls", [])

        # # Process uploaded files
        # if uploaded_files:
        #     os.makedirs("data/uploads", exist_ok=True)
        #     for file in uploaded_files:
        #         file_path = f"data/uploads/{file.name}"
        #         with open(file_path, "wb") as f:
        #             f.write(file.getbuffer())
        #         knowledge_urls.append(f"file://{os.path.abspath(file_path)}")
        
        # Process website
        if website_url:
            knowledge_urls.append(website_url)
        
        st.session_state.knowledge_urls = knowledge_urls
        # st.success(f"Added {len(uploaded_files)} files and {1 if website_url else 0} websites")
        st.success(f"Added {1 if website_url else 0} websites")
        # if 'knowledge_base' in st.session_state:
        #     del st.session_state["knowledge_base"]
        # if "knowledge_urls" in st.session_state:
        # with st.spinner("Initializing knowledge base..."):
            # agent.knowledge.load(recreate=True)
            # st.success("BB")
        # st.success(knowledge_urls)
        load_knowledge(True)
            # st.session_state.knowledge="Yes"


# Initialize session state
if "knowledge_urls" not in st.session_state:
    st.session_state.knowledge_urls = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Create agent with current knowledge
agent = create_agent(model_choice)

# if 'knowledge_base' not in st.session_state:
# # if agent.knowledge and not agent.knowledge.exists():
#     # if "knowledge_urls" in st.session_state:
#     with st.spinner("Initializing knowledge base..."):
#         # agent.knowledge.load(recreate=True)
#         # st.session_state.knowledge_db.load(True)


# if 'knowledge_base_loaded' not in st.session_state:
#     load_knowledge(False)
#     st.session_state["knowledge_base_loaded"] = "KB Loaded"

# Chat interface
for message in st.session_state.messages:
    avatar = MODEL_AVATARS[model_choice] if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Handle input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        full_response = ""
        with st.spinner("Generating response..."):
            # Create a single container for the assistant's response
            with st.chat_message("assistant", avatar=MODEL_AVATARS[model_choice]):
                response_placeholder = st.empty()  # Create a placeholder

                response = agent.run(prompt, stream=True) # not for knowledge, show_full_reasoning=True)
                for _resp_chunk in response:
                    # Display tool calls if available
                    #if _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                    #    display_tool_calls(tool_calls_container, _resp_chunk.tools)

                    # Display response
                    if _resp_chunk.content is not None:
                        full_response += _resp_chunk.content
                        # response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response.replace("%", "\\%") + "▌", 
                                unsafe_allow_html=True)
                        # placeholder.markdown(response)
                        # Display response
                        # with st.chat_message("assistant", avatar=MODEL_AVATARS[model_choice]):
                        #     st.markdown(full_response)
                    # Remove the cursor and show final result
                    # response_placeholder.markdown(full_response)
                    response_placeholder.markdown(full_response.replace("%", "\\%"), 
                                unsafe_allow_html=True)
            # # Display response
            # with st.chat_message("assistant", avatar=MODEL_AVATARS[model_choice]):
            #     st.markdown(response.content)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "model": model_choice
            })
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Add copyright footer
# st.markdown("""
# ---
# © Schogini Systems Private Limited | [www.schogini.com](https://www.schogini.com)
# """, unsafe_allow_html=True)

