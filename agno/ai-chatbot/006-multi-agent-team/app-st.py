from typing import Iterator, List
import streamlit as st
from textwrap import dedent
import os
from dotenv import load_dotenv
import random
import string
from agno.agent import Agent, RunResponse, AgentMemory
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
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.memory.db.sqlite import SqliteMemoryDb

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

def restaurant_booking_tool(name: str = "Guest", party_size: int = 2, date: str = None, time: str = None) -> str:
    """Use this function to book a table at the restaurant

    Args:
        name (str): Name of the guest
        party_size (int): Number of people in the party
        date (str): Date of reservation (format: YYYY-MM-DD)
        time (str): Time of reservation (format: HH:MM)

    Returns:
        str: Booking confirmation with reference number
    """
    if name == "Guest":
        return "Please provide your name to complete the booking."
    if not date or not time:
        return "Please provide both date and time for your restaurant reservation."
    booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"Your table for {party_size} is confirmed, {name}! Date: {date}, Time: {time}. Booking Reference: {booking_ref}"

def hotel_booking_tool(name: str = "Guest", room_type: str = "Standard", check_in: str = None, check_out: str = None, guests: int = 1) -> str:
    """Use this function to book a hotel room

    Args:
        name (str): Name of the guest
        room_type (str): Type of room (Standard, Deluxe, Suite)
        check_in (str): Check-in date (format: YYYY-MM-DD)
        check_out (str): Check-out date (format: YYYY-MM-DD)
        guests (int): Number of guests

    Returns:
        str: Booking confirmation with reference number
    """
    if name == "Guest":
        return "Please provide your name to complete the booking."
    if not check_in or not check_out:
        return "Please provide both check-in and check-out dates for your hotel reservation."
    booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"Your {room_type} room for {guests} guest(s) is confirmed, {name}! Check-in: {check_in}, Check-out: {check_out}. Booking Reference: {booking_ref}"

def airport_pickup_tool(name: str = "Guest", flight_number: str = None, arrival_date: str = None, arrival_time: str = None, passengers: int = 1) -> str:
    """Use this function to book an airport pickup service

    Args:
        name (str): Name of the guest
        flight_number (str): Flight number
        arrival_date (str): Date of arrival (format: YYYY-MM-DD)
        arrival_time (str): Time of arrival (format: HH:MM)
        passengers (int): Number of passengers

    Returns:
        str: Booking confirmation with reference number
    """
    if name == "Guest":
        return "Please provide your name to complete the booking."
    if not flight_number or not arrival_date or not arrival_time:
        return "Please provide flight number, arrival date, and arrival time for your airport pickup."
    booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"Your airport pickup for {passengers} passenger(s) is confirmed, {name}! Flight: {flight_number}, Arrival: {arrival_date} at {arrival_time}. Booking Reference: {booking_ref}"


chunking_strategy = SemanticChunking()
search_method=SearchType.hybrid

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


# Select model for main agent
if "model_choice" not in st.session_state:
    model_choice = "OpenAI"

if model_choice == "OpenAI":
    main_model = OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY)
    child_model = OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY)
elif model_choice == "DeepSeek":
    main_model = DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY)
    child_model = DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY)
elif model_choice == "Gemini":
    main_model = Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY)
    child_model = Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY)

# Initialize session state
if "airport_knowledge_urls" not in st.session_state:
    st.session_state.airport_knowledge_urls = []
if "hotel_knowledge_urls" not in st.session_state:
    st.session_state.hotel_knowledge_urls = []
if "restaurant_knowledge_urls" not in st.session_state:
    st.session_state.restaurant_knowledge_urls = []

# Create specialized knowledge bases for each domain
def create_domain_specific_knowledge(domain: str) -> PDFUrlKnowledgeBase:
    """Create a domain-specific knowledge base for a specialized agent"""
    
    if domain == "restaurant":
        urls = st.session_state["restaurant_knowledge_urls"]
    elif domain == "hotel":
        urls = st.session_state["hotel_knowledge_urls"]
    elif domain == "airport":
        urls = st.session_state["airport_knowledge_urls"]
    else:
        urls = []
    # In a real app, you'd filter URLs based on domain or use different knowledge sources
    # For this example, we're using the same knowledge base but would filter appropriately
    db_url = "postgresql+psycopg://ai:ai@pgvector:5432/ai"
    table_name = f"{domain}_knowledge"
    logger.info(f"===== init_{domain}_knowledge PG =====")
    return PDFUrlKnowledgeBase(
        urls=urls or [],
        vector_db=PgVector(
            table_name=table_name, 
            db_url=db_url,
        ),
        chunking_strategy=chunking_strategy,
        search_type=search_method,
        embedder=OpenAIEmbedder(
            id="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        ),
    )

# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=PgVector(table_name="recipes_semantic_chunking", db_url=db_url),
#     chunking_strategy=SemanticChunking(),
# )
# knowledge_base.load(recreate=True)  # Comment out after first run

restaurant_agent_storage = SqliteAgentStorage(
    table_name="restaurant_agent_memories", db_file="tmp/agents.db"
)
restaurant_knowledge_db = create_domain_specific_knowledge("restaurant")
if "restaurant_knowledge_db" not in st.session_state or not st.session_state["restaurant_knowledge_db"]:
    with st.spinner("Initializing Restaurant knowledge base..."):
        restaurant_knowledge_db.load(recreate=True)
        st.session_state["restaurant_knowledge_db"] = True
restaurant_agent = Agent(
    model=main_model,
    instructions=dedent("""
        You're a restaurant table booking agent.
        Pass the entities collected like the name etc. too to the tools.
    """),
    knowledge=restaurant_knowledge_db, #init_knowledge_pg(knowledge_urls),
    # knowledge=st.session_state.knowledge_db, #init_knowledge_pg(knowledge_urls),
    tools=[restaurant_booking_tool],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
    # storage=restaurant_agent_storage,
    # add_history_to_messages=True,
    # num_history_responses=3,
)
hotel_agent_storage = SqliteAgentStorage(
    table_name="hotel_agent_memories", db_file="tmp/agents.db"
)
hotel_knowledge_db = create_domain_specific_knowledge("hotel")
if "hotel_knowledge_db" not in st.session_state or not st.session_state["hotel_knowledge_db"]:
    with st.spinner("Initializing Hotel knowledge base..."):
        hotel_knowledge_db.load(recreate=True)
        st.session_state["hotel_knowledge_db"] = True
hotel_agent = Agent(
    model=main_model,
    instructions=dedent("""
        You're a hotel room booking agent.
        Pass the entities collected like the name etc. too to the tools.
    """),
    knowledge=hotel_knowledge_db, #init_knowledge_pg(knowledge_urls),
    # knowledge=st.session_state.knowledge_db, #init_knowledge_pg(knowledge_urls),
    tools=[hotel_booking_tool],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
    # storage=hotel_agent_storage,
    # add_history_to_messages=True,
    # num_history_responses=3,
)
airport_agent_storage = SqliteAgentStorage(
    table_name="airport_agent_memories", db_file="tmp/agents.db"
)
airport_knowledge_db = create_domain_specific_knowledge("airport")
if "airport_knowledge_db" not in st.session_state or not st.session_state["airport_knowledge_db"]:
    with st.spinner("Initializing Airport knowledge base..."):
        airport_knowledge_db.load(recreate=True)
        st.session_state["airport_knowledge_db"] = True
airport_agent = Agent(
    model=main_model,
    instructions=dedent("""
        You're an airport pickup agent.
        Pass the entities collected like the name etc. too to the tools.
    """),
    knowledge=airport_knowledge_db, #init_knowledge_pg(knowledge_urls),
    # knowledge=st.session_state.knowledge_db, #init_knowledge_pg(knowledge_urls),
    tools=[airport_pickup_tool],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
    # storage=airport_agent_storage,
    # add_history_to_messages=True,
    # num_history_responses=3,
)

team_agent_storage = SqliteAgentStorage(
    table_name="team_agent_memories", db_file="tmp/agents.db"
)
team_of_agents = Agent(
    model=main_model,
    team=[restaurant_agent, hotel_agent, airport_agent],
    instructions=dedent("""
        You are hotel receptionist.
        Your role:
        1. Coordinate between the three agents restaurant_agent, hotel_agent, and airport_agent.
        2. Pass the entities collected like the name etc. too to the agents.
    """),
    # knowledge=st.session_state.knowledge_db, #init_knowledge_pg(knowledge_urls),
    # tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
    memory=AgentMemory(
        db=SqliteMemoryDb(
            table_name="agent_memory",
            db_file="tmp/agent_memory.db",
        ),
        create_user_memories=True,
        update_user_memories_after_run=True,
        create_session_summary=True,
        update_session_summary_after_run=True,
    ),
    storage=team_agent_storage,
    add_history_to_messages=True,
    num_history_responses=3,
)



if "knowledge_urls" not in st.session_state:
    st.session_state["knowledge_urls"] = []
    # st.session_state["knowledge_urls"] = ["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"]
    # st.session_state["knowledge_urls"] = ['file:///app/data/uploads/ThaiRecipes.pdf']

# if "knowledge_db" not in st.session_state:
#     # urls = len(st.session_state.knowledge_urls)
#     # st.success(f"Count {urls} URLs")
#     st.session_state["knowledge_db"] = init_knowledge()

# UI Components
st.title("💬 Flexible VectorDB and Chunking Multi-Model Agno Chat Agent")
# st.write(f"**Chunking Strategy:** {chunking_strategy_selected} | **VectorDB:** {search_type_selected}" )
# model_choice = st.selectbox("Choose AI Model", list(MODEL_AVATARS.keys()))
# st.caption(f"Currently using: {model_choice} {MODEL_AVATARS[model_choice]}")

# File management sidebar
with st.sidebar:
    st.title("💬 Chatbot Options")
    st.sidebar.subheader("Model Selection")
    model_choice = st.selectbox("Choose AI Model", list(MODEL_AVATARS.keys()))
    st.caption(f"Currently using: {model_choice} {MODEL_AVATARS[model_choice]}")

    # st.sidebar.subheader("Reasoning Yes/No")
    # reason = st.sidebar.checkbox("Reasoning")
    
    # Function to show loading when chunking strategy changes
    # def chunking_strategy_change():
    #     with st.sidebar:
    #         with st.spinner("Loading new chunking strategy..."):
    #             load_knowledge(True)  # Reload knowledge with new strategy
    #             time.sleep(5)  # Simulate loading time

    # def search_type_change():
    #     with st.sidebar:
    #         with st.spinner("Loading new search type..."):
    #             load_knowledge(True)  # Reload knowledge with new strategy
    #             time.sleep(3)  # Simulate loading time


    st.header("Knowledge Base Management")
    
    # PDF Upload
    # uploaded_files = st.file_uploader(
    #     "Upload Documents",
    #     type=["pdf", "txt"],
    #     accept_multiple_files=True
    # )
    st.subheader("Airport")
    airport_website_url = st.text_input("Add PDF URL", placeholder="https://example.com/abcd.pdf",key="a1")
    if st.button("Update Airport Pickup KB",key="a2"):
        airport_knowledge_urls = st.session_state.get("airport_knowledge_urls", [])
        if airport_website_url:
            airport_knowledge_urls.append(airport_website_url)
        
        st.session_state.airport_knowledge_urls = airport_knowledge_urls
        # st.success(f"Added {len(uploaded_files)} files and {1 if website_url else 0} websites")
        st.success(f"Added {1 if airport_website_url else 0} websites")
        st.session_state["airport_knowledge_db"] = False
        st.rerun()
    st.subheader("Hotel")
    hotel_website_url = st.text_input("Add PDF URL", placeholder="https://example.com/abcd.pdf",key="a3")
    if st.button("Update Room KB",key="a4"):
        hotel_knowledge_urls = st.session_state.get("hotel_knowledge_urls", [])
        if hotel_website_url:
            hotel_knowledge_urls.append(hotel_website_url)
        
        st.session_state.hotel_knowledge_urls = hotel_knowledge_urls
        # st.success(f"Added {len(uploaded_files)} files and {1 if website_url else 0} websites")
        st.success(f"Added {1 if hotel_website_url else 0} websites")
        st.session_state["hotel_knowledge_db"] = False
        st.rerun()
    st.subheader("Restaurant")
    restaurant_website_url = st.text_input("Add PDF URL", placeholder="https://example.com/abcd.pdf",key="a5")
    if st.button("Update Restaurant KB",key="a6"):
        restaurant_knowledge_urls = st.session_state.get("restaurant_knowledge_urls", [])
        if restaurant_website_url:
            restaurant_knowledge_urls.append(restaurant_website_url)
        
        st.session_state.restaurant_knowledge_urls = restaurant_knowledge_urls
        # st.success(f"Added {len(uploaded_files)} files and {1 if website_url else 0} websites")
        st.success(f"Added {1 if restaurant_website_url else 0} websites")
        st.session_state["restaurant_knowledge_db"] = False
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Create agent with current knowledge
# agent = create_agent(model_choice)

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

                response = team_of_agents.run(prompt, stream=True) # not for knowledge, show_full_reasoning=True)
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

