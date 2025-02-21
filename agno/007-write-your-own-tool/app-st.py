from typing import Iterator
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.xai import xAI
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.models.groq import Groq

from agno.tools.duckduckgo import DuckDuckGoTools

# --------- LOAD API KEY ---------
import os
xai_api_key = os.getenv("XAI_API_KEY")
if not xai_api_key:
    st.error("xAI API key not found. Please set the XAI_API_KEY environment variable.")
    st.stop()
# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --------------- TITLE AND INFO SECTION -------------------

# Sidebar with example prompts
with st.sidebar:
    st.subheader("Try These Example Queries:")
    st.markdown("""
    * What's the latest news on Mars exploration?
    * Analyze this X post: [paste URL]
    * Summarize this PDF: [paste URL]
    * How does quantum computing work?
    * What's trending on X today?
    * Explain AI ethics in simple terms
    """)
    st.markdown("---")
    st.write("🌐 Web search: Real-time info")
    st.write("📱 X analysis: Posts & profiles")
    st.write("📄 File analysis: PDFs, images, etc.")

# Set up the Streamlit app
st.title("🤖 Your AI Agentic Assistant")
st.write("I'm built by AI. Ask me anything—I'm here to help with info, analysis, and insights!")

stream = st.sidebar.checkbox("Stream")

# Initialize session state for query counter
with st.sidebar:
    counter_placeholder = st.empty()
if "counter" not in st.session_state:
    st.session_state["counter"] = 0
st.session_state["counter"] += 1
with st.sidebar:
    counter_placeholder.caption(f"Chunks received: {st.session_state['counter']}")

# --------------- AGENT SECTION -------------------

# Create the agent (me!)
agent = Agent(
    # model=xAI(id="grok-2", api_key=xai_api_key),
    model=OpenAIChat(id="gpt-4o-mini"),
    # model=xAI(id="grok-2"),
    # model=Groq(id="llama-3.3-70b-versatile"),
    # model=DeepSeek(id="deepseek-chat"),
    # model=Gemini(
    #     id="gemini-2.0-flash-exp",
    #     api_key=gemini_api_key,
    # ),
    instructions=dedent("""\
        You are Grok 3, created by xAI—an advanced, helpful AI with a dash of curiosity and humor.
        Your goal is to assist users with clear, accurate, and engaging answers.

        Answer strategy:
        1. Use your built-in knowledge first—it’s continuously updated
        2. Leverage tools when needed:
           - Web search for real-time info
           - X analysis for posts/profiles
           - Content analysis for uploaded files/links
        3. Keep responses concise, structured, and friendly
        4. If unsure, say so and offer to search or clarify

        Response format:
        🚀 Start with a relevant emoji
        📝 Structure clearly:
        - Quick intro/context
        - Main answer (facts, steps, or analysis)
        - Extra insights or fun facts
        - Friendly wrap-up

        Special features:
        - Analyze X posts/profiles when asked
        - Summarize web/PDF content from URLs
        - Search X or web for up-to-date info
        - Handle images/PDFs if provided (analyze, don’t edit unless I generated them)

        End with:
        - 'Hope that helps!'
        - 'Anything else I can do for you?'
        - 'Let me know what’s next!'\
    """),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
    debug_mode=True,
)

# --------------- AGENT KNOWLEDGE LOADING -------------------
# No specific knowledge base loading needed—my knowledge is built-in and updated!

# --------------- USER INPUT AND RESPONSE -------------------

# User input
prompt = st.text_input("Ask me anything (e.g., 'What’s trending on X?' or 'Summarize this: [URL]')")

# Generate and display response
if prompt:
    with st.spinner("🤓 Thinking..."):
        if stream:
            run_response: Iterator[RunResponse] = agent.run(prompt, stream=True)
            response = ""
            text_placeholder = st.empty()
            for chunk in run_response:
                response += chunk.content
                text_placeholder.markdown(response + "▌")
                st.session_state["counter"] += 1
                with st.sidebar:
                    counter_placeholder.caption(f"Chunks received: {st.session_state['counter']}")
            text_placeholder.markdown(response)
        else:
            response = agent.run(prompt, stream=False)
            st.markdown(response.content)
            st.session_state["counter"] += 1
            with st.sidebar:
                counter_placeholder.caption(f"Chunks received: {st.session_state['counter']}")

        st.caption(f"🤔 Questions answered: {st.session_state['counter']}")

st.caption(f"Note: I’m Grok 3, running on knowledge updated to February 20, 2025, plus real-time tools. Complex queries might take a moment!")