# app/main.py
import os
from typing import Iterator
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Set up the Streamlit app
st.title("🗽 NYC News Reporter Agent")
st.write("Welcome to your quirky NYC news reporter! Ask for the latest scoop from the Big Apple.")

counter_placeholder = st.empty()
if "counter" not in st.session_state:
    st.session_state["counter"] = 0
st.session_state["counter"] += 1
counter_placeholder.write(f"News queries processed: {st.session_state['counter']}")

# Create the agent with web search capabilities
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
    instructions=dedent("""\
        You are an enthusiastic news reporter with a flair for storytelling! 🗽
        Think of yourself as a mix between a witty comedian and a sharp journalist.

        Follow these guidelines for every report:
        1. Start with an attention-grabbing headline using relevant emoji
        2. Use the search tool to find current, accurate information
        3. Present news with authentic NYC enthusiasm and local flavor
        4. Structure your reports in clear sections:
        - Catchy headline
        - Brief summary of the news
        - Key details and quotes
        - Local impact or context
        5. Keep responses concise but informative (2-3 paragraphs max)
        6. Include NYC-style commentary and local references
        7. End with a signature sign-off phrase

        Sign-off examples:
        - 'Back to you in the studio, folks!'
        - 'Reporting live from the city that never sleeps!'
        - 'This is [Your Name], live from the heart of Manhattan!'

        Remember: Always verify facts through web searches and maintain that authentic NYC energy!\
    """),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# User input
prompt = st.text_input("Ask the reporter for a news story (e.g., 'What's happening in Times Square?')")

# Generate and display response
if prompt:
    with st.spinner("🚨 Chasing down the story... (this might take 30 seconds)"):
        stream = True  # Enable streaming for better user experience
        if stream:
            run_response: Iterator[RunResponse] = agent.run(prompt, stream=True)
            response = ""
            text_placeholder = st.empty()
            for chunk in run_response:
                response += chunk.content
                text_placeholder.markdown(response + "▌")
                st.session_state["counter"] += 1
                counter_placeholder.write(f"News queries processed: {st.session_state['counter']}")
            text_placeholder.markdown(response)  # Remove cursor
        else:
            response = agent.run(prompt, stream=False)
            st.markdown(response.content)
            st.session_state["counter"] += 1
            counter_placeholder.write(f"News queries processed: {st.session_state['counter']}")

# Sidebar with example prompts
with st.sidebar:
    st.subheader("Try These NYC News Prompts:")
    st.code("""
    What's the latest headline from Wall Street?
    Tell me about breaking news in Central Park
    What's happening at Yankees Stadium today?
    Updates on newest Broadway shows?
    Buzz about latest NYC restaurant opening?
    """)

st.caption("Note: Responses powered by real-time web searches and GPT-4o AI analysis")