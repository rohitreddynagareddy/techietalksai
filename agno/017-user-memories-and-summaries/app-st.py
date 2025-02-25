import os
import json
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, AgentMemory
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.storage.agent.sqlite import SqliteAgentStorage

from agno.models.openai import OpenAIChat
from agno.models.xai import xAI
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.models.groq import Groq
# --------- LOAD API KEY ---------
import os
# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
xai_api_key = os.getenv("XAI_API_KEY")
if not xai_api_key:
    st.error("xAI API key not found. Please set the XAI_API_KEY environment variable.")
    st.stop()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()


# --------------- SETUP ---------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------- TITLE & SIDEBAR -------------------
st.title("🧠 AI Assistant with Long-Term Memory")
st.write("Persistent memory across conversations with session summaries")

with st.sidebar:
    st.subheader("Session Management")
    user_id = st.text_input("User ID", value="user123")
    new_session = st.checkbox("Start New Session", True)
    
    if st.button("Initialize Agent"):
        # Create or load agent session
        agent_storage = SqliteAgentStorage(
            table_name="agent_memories", 
            db_file="tmp/agents.db"
        )
        
        session_id = None if new_session else agent_storage.get_all_session_ids(user_id)[0]
        
        st.session_state.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            # model=DeepSeek(id="deepseek-chat"),
            user_id=user_id,
            session_id=session_id,
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
            storage=agent_storage,
            add_history_to_messages=True,
            num_history_responses=3,

            show_tool_calls=True,
            debug_mode=True,

            description=dedent("""\
                You are a helpful AI assistant with long-term memory:
                - Remember user details and preferences
                - Maintain conversation context across sessions
                - Reference previous interactions naturally
                - Be truthful about memory limitations""")
        )
        
        if session_id is None:
            st.session_state.agent.session_id = f"session_{len(agent_storage.get_all_session_ids(user_id)) + 1}"
            st.success(f"New session started: {st.session_state.agent.session_id}")
        else:
            st.info(f"Continuing session: {session_id}")

# --------------- CHAT INTERFACE -------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    if not st.session_state.agent:
        st.error("Please initialize agent in sidebar first!")
        st.stop()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    response = st.session_state.agent.run(prompt)
    
    # Add agent response to history
    with st.chat_message("assistant"):
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})

# --------------- MEMORY VISUALIZATION -------------------
if st.session_state.agent:
    with st.expander("Memory Details", expanded=False):
        st.subheader("Chat History")
        st.json([
            m.to_dict() for m in st.session_state.agent.memory.messages
        ][-3:])  # Show last 3 messages

        # st.subheader("User Memories")
        # st.json([
        #     m.model_dump(include={"memory", "input"})
        #     for m in st.session_state.agent.memory.memories
        # ])
        # json.dumps(
        #             [
        #                 m.model_dump(include={"memory", "input"})
        #                 for m in agent.memory.memories
        #             ]
        #         )
        st.subheader("User Memories")
        if st.session_state.agent and hasattr(st.session_state.agent.memory, "memories") and st.session_state.agent.memory.memories is not None:
            # If memories is None, use an empty list so the loop doesn't break.
            # memories = st.session_state.agent.memory.memories or []
            memories_data = [
                # m.model_dump(include={"memory", "input"})
                # for m in memories
                st.session_state.agent.memory.memories
            ]
            st.json(st.session_state.agent.memory.memories)
        else:
            st.write("No memories available.")

        st.subheader("Session Summary")
        if st.session_state.agent and hasattr(st.session_state.agent.memory, "summary") and st.session_state.agent.memory.summary is not None:
            st.json(st.session_state.agent.memory.summary.model_dump())
        else:
            st.write("No summary available.")
# --------------- RESET BUTTON -------------------
if st.sidebar.button("Reset Session"):
    st.session_state.clear()
    st.rerun()