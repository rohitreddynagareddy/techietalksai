# app/main.py
import os
from typing import Iterator
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat

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
counter_placeholder.write(st.session_state["counter"])

# Create the agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
    instructions=dedent("""\
        You are an enthusiastic news reporter with a flair for storytelling! 🗽
        Think of yourself as a mix between a witty comedian and a sharp journalist.

        Your style guide:
        - Start with an attention-grabbing headline using emoji
        - Share news with enthusiasm and NYC attitude
        - Keep your responses concise but entertaining
        - Throw in local references and NYC slang when appropriate
        - End with a catchy sign-off like 'Back to you in the studio!' or 'Reporting live from the Big Apple!'

        Remember to verify all facts while keeping that NYC energy high!\
    """),
    markdown=True,
)

# User input
prompt = st.text_input("Ask the reporter for a news story (e.g., 'What's happening in Times Square?')")

# Generate and display response
if prompt:
    with st.spinner("Getting the latest scoop..."):
        stream = False 
        if stream:
            run_response: Iterator[RunResponse] = agent.run(prompt, stream=True)
            response = ""
            text_placeholder = st.empty()
            for chunk in run_response:
                # st.write(chunk.content)
                response += chunk.content
                text_placeholder.markdown(response)
                st.session_state["counter"] += 1
                counter_placeholder.write(st.session_state["counter"])
        else:
            response = agent.run(prompt, stream=False)
            response = response.content
            st.write(response)