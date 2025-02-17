# app/main.py
from typing import Iterator, List, Optional
import streamlit as st
from textwrap import dedent
import os
from agno.agent import Agent, RunResponse
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType

# --------- LOAD API KEY ---------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --------------- INITIALIZE STORAGE -------------------
agent_storage = SqliteAgentStorage(table_name="recipe_agent", db_file="tmp/agents.db")

# --------------- TITLE AND INFO SECTION -------------------
st.title("🧑🍳 AI Thai Cooking Assistant with Memory")
st.write("Your personal Thai cuisine expert with conversation memory!")

# --------------- SESSION MANAGEMENT -------------------
def init_session():
    st.session_state.session_id = None
    st.session_state.user_id = "streamlit_user"
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    init_session()

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Session Management")
    
    # New session button
    if st.button("Start New Session"):
        init_session()
        st.rerun()
    
    # Session selector
    # existing_sessions = agent_storage.get_all_session_ids(st.session_state.user_id)
    existing_sessions = agent_storage.get_all_session_ids(st.session_state.user_id)
    selected_session = st.selectbox(
        "Continue Existing Session",
        options=existing_sessions,
        index=0 if not existing_sessions else None
    )
    
    if selected_session and selected_session != st.session_state.session_id:
        st.session_state.session_id = selected_session
        st.session_state.chat_history = agent_storage.get_all_sessions(
            user_id=st.session_state.user_id,
            # session_id=selected_session
        )
        st.rerun()

    st.markdown("---")
    st.subheader("Try These Queries:")
    st.markdown("""
    - How make authentic Pad Thai?
    - Difference between red/green curry?
    - What's galangal substitutes?
    - History of Tom Yum soup?
    - Essential Thai pantry items?
    """)

# --------------- AGENT INITIALIZATION -------------------
agent = Agent(
    user_id=st.session_state.user_id,
    session_id=st.session_state.session_id,
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    instructions=dedent("""\
        You are a passionate Thai cuisine expert! 🧑‍🍳
        Combine cooking instruction with food history expertise.

        Answer strategy:
        1. First check the recipe knowledge base
        2. Use web search for:
           - Ingredient substitutions
           - Historical context
           - Additional tips

        Response format:
        🌶️ Start with relevant emoji
        📖 Structured sections:
        - Context
        - Main content
        - Pro tips
        - Encouraging conclusion

        For recipes include:
        📝 Ingredients with subs
        🔢 Numbered steps
        💡 Success tips

        End with:
        - 'Happy cooking! ขอให้อร่อย!'\
    """),
    storage=agent_storage,
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipe_knowledge",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small", api_key=openai_api_key),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    read_chat_history=True,
    markdown=True,
)

# --------------- USER INPUT HANDLING -------------------
prompt = st.chat_input("Ask your Thai cooking question...")

if prompt:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("👩🍳 Cooking up your answer..."):
            response = agent.run(prompt, stream=False)
            
            # Update session ID if new session
            if st.session_state.session_id is None:
                st.session_state.session_id = agent.session_id
                
            # Store response in history
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})
            
            # Display response
            st.markdown(response.content)

# --------------- KNOWLEDGE MANAGEMENT -------------------
with st.sidebar:
    st.markdown("---")
    if st.button("Load/Reload Recipe Database"):
        with st.spinner("🧑🍳 Loading authentic Thai recipes..."):
            if agent.knowledge:
                agent.knowledge.load()
                st.success("Recipe database loaded!")

st.caption("Note: Maintains conversation history across sessions. May take 20-30 seconds for complex queries.")