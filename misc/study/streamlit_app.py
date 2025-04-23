import streamlit as st
import os
from openai import OpenAI

# Constants
PROMPT_FILE = "system_prompt.txt"
PATH_FILE = "path.txt"

if os.path.exists(PATH_FILE):
    with open(PATH_FILE, "r") as f:
        existing_folder = f.read().strip()
else:
    existing_folder = ""

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio, MCPServerHTTP
import asyncio
server = MCPServerStdio('npx', ["-y", "@modelcontextprotocol/server-filesystem", existing_folder])

async def run_agent_query(agent, user_query):
    async with agent.run_mcp_servers():
        result = await agent.run(user_query)
    # return result.data
    return result.output

st.set_page_config(page_title="🛠️ Config Editor + Chat", layout="wide")

st.sidebar.title("⚙️ Settings")

# --- Sidebar Navigation ---
page = st.sidebar.radio(
    "Navigation",
    ["Chatbot", "System Prompt", "Choose Folder"],  # Default is Chatbot
)

# --- System Prompt Editing ---
if page == "System Prompt":
    st.header("📝 Edit System Prompt")

    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            prompt_text = f.read()
    else:
        prompt_text = ""

    new_prompt = st.text_area("Edit your system prompt below:", prompt_text, height=400)

    if st.button("💾 Save System Prompt"):
        with open(PROMPT_FILE, "w") as f:
            f.write(new_prompt)
        st.success("✅ System prompt saved successfully!")

# --- Folder Choosing ---
if page == "Choose Folder":
    st.header("📁 Choose a Folder")

    if os.path.exists(PATH_FILE):
        with open(PATH_FILE, "r") as f:
            existing_folder = f.read().strip()
    else:
        existing_folder = ""

    folder_path = st.text_input("Current folder path:", existing_folder)

    # if st.button("📂 Browse Folder"):
    #     uploaded_file = st.file_uploader("Pick any file inside your desired folder (temporary hack)", type=None)
    #     if uploaded_file is not None:
    #         folder = os.path.dirname(uploaded_file.name)
    #         folder_path = folder
    #         st.success(f"✅ Folder selected: {folder}")

    if st.button("💾 Save Folder Path"):
        if os.path.isdir(folder_path):
            with open(PATH_FILE, "w") as f:
                f.write(folder_path)
            st.success(f"✅ Path saved to {PATH_FILE}")
        else:
            st.error("❌ Invalid directory! Please check your path.")

# --- Chatbot Section ---
if page == "Chatbot":
    st.header("🤖 Agentic Vibe Coder with MCP Tools")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # AI Setup
    # Initialize OpenAI client
    # client = OpenAI()


    system_prompt = "You are a helpful assistant."
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            system_prompt = f.read()

    # OpenAI
    # agent=Agent(
    #     name="Assistant",
    #     instructions=system_prompt, #"Use the tools to achieve the task",
    #     #mcp_servers=[mcp_server_1, mcp_server_2]
    # )
    # PydanticAI
    agent = Agent(
        'openai:gpt-4o-mini', 
        mcp_servers=[server],
        # system_prompt=system_prompt,
        instructions=system_prompt,
        )

    user_input = st.chat_input("Ask something...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):


                # Prepare full conversation history
                full_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

                # response = client.chat.completions.create(
                #     model="gpt-4o-mini",  # or "gpt-3.5-turbo"
                #     messages=full_messages
                # )
                # reply = response.choices[0].message.content

                # Run agent inside event loop
                reply = asyncio.run(run_agent_query(agent, user_input))
                # reply = asyncio.run(run_agent_query(agent, full_messages))

                st.markdown(reply)

                # Save assistant reply to conversation
                st.session_state.messages.append({"role": "assistant", "content": reply})
