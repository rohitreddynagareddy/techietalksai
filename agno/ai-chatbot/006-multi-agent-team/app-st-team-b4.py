from typing import Iterator, List, Dict, Any
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

from agno.knowledge.base import KnowledgeBase

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

    search_method=SearchType.hybrid

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
            search_type=search_method,
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
                search_type=search_method,
                embedder=OpenAIEmbedder(
                    id="text-embedding-3-small",
                    api_key=OPENAI_API_KEY
                ),
            ),
            chunking_strategy=chunking_strategy,
            search_type=search_method,
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
    domains = ["restaurant", "hotel", "airport"]
    
    # Load main knowledge base
    if "knowledge_db" in st.session_state:
        with st.spinner("Initializing main knowledge base..."):
            logger.info("===== load_knowledge init =====")
            st.session_state["knowledge_db"] = init_knowledge()
            logger.info("===== load_knowledge load =====")
            st.session_state.knowledge_db.load(recreate=recreate_tf)
            logger.info("===== load_knowledge loaded =====")
    
    # Load domain-specific knowledge bases
    for domain in domains:
        domain_key = f"{domain}_knowledge_db"
        with st.spinner(f"Initializing {domain} knowledge base..."):
            logger.info(f"===== load_{domain}_knowledge init =====")
            st.session_state[domain_key] = create_domain_specific_knowledge(domain)
            if st.session_state[domain_key]:
                logger.info(f"===== load_{domain}_knowledge load =====")
                st.session_state[domain_key].load(recreate=recreate_tf)
                logger.info(f"===== load_{domain}_knowledge loaded =====")
    
    return
    
    logger.info("===== load_knowledge returns =====")
    # st.success("Not Loaded DB")

if "chunking_strategy_selected" not in globals():
    chunking_strategy_selected = "Fixed"


# Create specialized knowledge bases for each domain
def create_domain_specific_knowledge(domain: str) -> PDFUrlKnowledgeBase:
    """Create a domain-specific knowledge base for a specialized agent"""
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

    search_method = SearchType.hybrid
    
    # In a real app, you'd filter URLs based on domain or use different knowledge sources
    # For this example, we're using the same knowledge base but would filter appropriately
    if search_type_selected == "Hybrid PostgreSQL":
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
    elif search_type_selected == "Hybrid LanceDB":
        table_name = f"{domain}_knowledge"
        logger.info(f"===== init_{domain}_knowledge Lance =====")
        return PDFUrlKnowledgeBase(
            urls=urls or [],
            vector_db=LanceDb(
                uri="data/vector_store",
                table_name=table_name,
                search_type=search_method,
                embedder=OpenAIEmbedder(
                    id="text-embedding-3-small",
                    api_key=OPENAI_API_KEY
                ),
            ),
            chunking_strategy=chunking_strategy,
            search_type=search_method,
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=OPENAI_API_KEY
            ),
        )
    elif search_type_selected == "Hybrid Pinecone":
        logger.info(f"===== init_{domain}_knowledge Pine =====")
        return PDFUrlKnowledgeBase(
            urls=urls or [],
            vector_db = PineconeDb(
                name=f"{domain}-knowledge",
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
    logger.info(f"===== init_{domain}_knowledge fails =====")
    return None

# Initialize an agent with dynamic knowledge
def create_agent(model_choice: str):
    # Select model for main agent
    if model_choice == "OpenAI":
        main_model = OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY)
        child_model = OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY)
    elif model_choice == "DeepSeek":
        main_model = DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY)
        child_model = DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY)
    elif model_choice == "Gemini":
        main_model = Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY)
        child_model = Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY)
    
    # Create specialized restaurant agent
    restaurant_agent = Agent(
        model=child_model,
        instructions=dedent("""
            You are a specialized restaurant booking agent. Your expertise is exclusively in restaurant reservations.
            When asked about restaurant bookings:
            1. Use the restaurant_booking_tool to make reservations
            2. Be conversational but focused on restaurant bookings
            3. Provide helpful tips for dining experiences when relevant
        """),
        knowledge=st.session_state.get("restaurant_knowledge_db"),
        tools=[restaurant_booking_tool],
        show_tool_calls=True,
        markdown=True,
        name="Restaurant Agent"
    )
    
    # Create specialized hotel agent
    hotel_agent = Agent(
        model=child_model,
        instructions=dedent("""
            You are a specialized hotel booking agent. Your expertise is exclusively in hotel accommodations.
            When asked about hotel bookings:
            1. Use the hotel_booking_tool to make reservations
            2. Be conversational but focused on hotel bookings
            3. Provide helpful tips for comfortable stays when relevant
        """),
        knowledge=st.session_state.get("hotel_knowledge_db"),
        tools=[hotel_booking_tool],
        show_tool_calls=True,
        markdown=True, 
        name="Hotel Agent"
    )
    
    # Create specialized airport pickup agent
    airport_agent = Agent(
        model=child_model,
        instructions=dedent("""
            You are a specialized airport transportation agent. Your expertise is exclusively in airport pickups.
            When asked about airport transportation:
            1. Use the airport_pickup_tool to arrange transportation
            2. Be conversational but focused on airport transportation
            3. Provide helpful travel tips when relevant
        """),
        knowledge=st.session_state.get("airport_knowledge_db"),
        tools=[airport_pickup_tool],
        show_tool_calls=True,
        markdown=True,
        name="Airport Agent"
    )
    
    # Create agents dictionary for TeamAgent
    agent_team = {
        "restaurant": restaurant_agent,
        "hotel": hotel_agent,
        "airport": airport_agent
    }
    
    # Create the main team agent that coordinates the specialized agents
    if reason:
        return TeamAgent(
            model=main_model,
            instructions=dedent("""
                You are a travel concierge assistant that coordinates a team of specialized agents.
                Delegate tasks to the appropriate specialized agent:
                - Restaurant booking requests to the restaurant agent
                - Hotel booking requests to the hotel agent  
                - Airport pickup requests to the airport agent
                
                For general information:
                1. Check knowledge base for technical information
                2. Use web search for real-time data when needed
                3. Cite sources when using external information
                
                When presenting mathematical expressions:
                - Use double dollar signs for block equations: $$\\frac{3}{5} = 0.6$$
                - Use escaped percentage signs: 60\\%
                - For inline equations: $x^2 + y^2 = z^2$
                
                Coordinate between agents when a user has multiple related requests.
            """),
            knowledge=st.session_state.knowledge_db,
            agents=agent_team,
            tools=[DuckDuckGoTools()],
            reasoning_model=Groq(
                id="deepseek-r1-distill-llama-70b", temperature=0.6, max_tokens=1024, top_p=0.95
            ),
            show_tool_calls=True,
            markdown=True,
            add_references=True,
        )
    else:
        return TeamAgent(
            model=main_model,
            instructions=dedent("""
                You are a travel concierge assistant that coordinates a team of specialized agents.
                Delegate tasks to the appropriate specialized agent:
                - Restaurant booking requests to the restaurant agent
                - Hotel booking requests to the hotel agent  
                - Airport pickup requests to the airport agent
                
                For general information:
                1. Check knowledge base for technical information
                2. Use web search for real-time data when needed
                3. Cite sources when using external information
                
                When presenting mathematical expressions:
                - Use double dollar signs for block equations: $$\\frac{3}{5} = 0.6$$
                - Use escaped percentage signs: 60\\%
                - For inline equations: $x^2 + y^2 = z^2$
                
                Coordinate between agents when a user has multiple related requests.
            """),
            knowledge=st.session_state.knowledge_db,
            agents=agent_team,
            tools=[DuckDuckGoTools()],
            show_tool_calls=True,
            markdown=True,
            add_references=True,
        )


if "knowledge_urls" not in st.session_state:
    st.session_state["knowledge_urls"] = []
    # st.session_state["knowledge_urls"] = ["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"]
    # st.session_state["knowledge_urls"] = ['file:///app/data/uploads/ThaiRecipes.pdf']

if "knowledge_db" not in st.session_state:
    # Main knowledge base
    st.session_state["knowledge_db"] = init_knowledge()
    
    # Domain-specific knowledge bases
    domains = ["restaurant", "hotel", "airport"]
    for domain in domains:
        domain_key = f"{domain}_knowledge_db"
        if domain_key not in st.session_state:
            st.session_state[domain_key] = create_domain_specific_knowledge(domain)

# UI Components
st.title("💬 Multi-Agent Team with Specialized Knowledge")
st.write(f"**Chunking Strategy:** {chunking_strategy_selected} | **VectorDB:** {search_type_selected}" )
st.caption("This chatbot uses a team of specialized agents for restaurant bookings, hotel reservations, and airport pickups.")
# model_choice = st.selectbox("Choose AI Model", list(MODEL_AVATARS.keys()))
# st.caption(f"Currently using: {model_choice} {MODEL_AVATARS[model_choice]}")

# File management sidebar
with st.sidebar:
    st.title("💬 Multi-Agent Team Options")
    
    st.sidebar.subheader("Main Agent Model")
    model_choice = st.selectbox("Choose AI Model", list(MODEL_AVATARS.keys()))
    st.caption(f"Currently using: {model_choice} {MODEL_AVATARS[model_choice]}")

    # Display information about specialized agents
    st.sidebar.subheader("Team of Agents")
    st.markdown("""
    **Main Coordinator**: Travel Concierge Agent
    
    **Specialized Agents**:
    - 🍽️ Restaurant Agent: Handles restaurant bookings
    - 🏨 Hotel Agent: Manages hotel reservations
    - ✈️ Airport Agent: Arranges airport pickups
    """)

    st.sidebar.subheader("Reasoning Capabilities")
    reason = st.sidebar.checkbox("Enable reasoning with Groq")
    
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
    if message["role"] == "assistant":
        # Default to main model avatar
        avatar = MODEL_AVATARS[model_choice]
        
        # Customize avatar based on agent name if available
        if "agent_name" in message:
            if message["agent_name"] == "Restaurant Agent":
                avatar = "🍽️"
            elif message["agent_name"] == "Hotel Agent":
                avatar = "🏨"
            elif message["agent_name"] == "Airport Agent":
                avatar = "✈️"
        
        with st.chat_message(message["role"], avatar=avatar):
            if "agent_name" in message:
                st.caption(f"Response from: {message['agent_name']}")
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
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
            
            # Determine which agent responded based on content
            agent_name = "Travel Concierge"
            if "restaurant booking" in full_response.lower() or "dining" in full_response.lower():
                agent_name = "Restaurant Agent"
            elif "hotel" in full_response.lower() or "accommodation" in full_response.lower() or "room" in full_response.lower():
                agent_name = "Hotel Agent"
            elif "airport" in full_response.lower() or "flight" in full_response.lower() or "pickup" in full_response.lower():
                agent_name = "Airport Agent"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "model": model_choice,
                "agent_name": agent_name
            })
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Add copyright footer
# st.markdown("""
# ---
# © Schogini Systems Private Limited | [www.schogini.com](https://www.schogini.com)
# """, unsafe_allow_html=True)

