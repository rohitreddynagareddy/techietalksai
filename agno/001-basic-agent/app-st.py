# app/main.py
import os
from typing import Iterator
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek

# --------- LOAD API KEY ---------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --------------- TITLE AND INFO SECTION -------------------
st.title("🗽 NYC News Reporter Bot")
st.write("Your sassy AI news buddy with that authentic New York attitude!")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Try These NYC Prompts:")
    st.markdown("""
    - Central Park latest scoop
    - Wall Street breaking story
    - Yankees game update
    - New Broadway show buzz
    - Brooklyn food trend
    - Subway peculiar incident
    """)
    st.markdown("---")
    # st.caption(f"Queries processed: {st.session_state.get('counter', 0)}")

# Initialize session state for query counter
with st.sidebar:
    counter_placeholder = st.empty()
if "counter" not in st.session_state:
    st.session_state["counter"] = 0
st.session_state["counter"] += 1
with st.sidebar:
    counter_placeholder.caption(f"Queries processed: {st.session_state['counter']}")

stream = st.sidebar.checkbox("Stream")

# --------------- AGENT INITIALIZATION -------------------
agent = Agent(
    # model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    model=DeepSeek(id="deepseek-chat"),
    instructions=dedent("""\
        You're the most NYC news reporter ever! 🗽
        Equal parts stand-up comic and hard-hitting journalist.

        **Your style rules:**
        1. Start with EMOJI HEADLINE that slaps
        2. Serve news with extra attitude 🇺🇸
        3. Keep it quick but memorable
        4. Drop local slang like a true New Yorker
        5. Sign off with flair

        **Required NYC elements:**
        - Pizza rat references 🐀🍕
        - Subway stories 🚇
        - Bodega cat shoutouts 🐈⬛
        - Traffic horn SFX 🚗📢
        - Pretentious coffee takes ☕

        **Sign-off examples:**
        - 'Back to you in the studio, suckers!'
        - 'Reporting from the concrete jungle!'
        - 'This is Tony PizzaRat, LIVE from a fire escape!'\
    """),
    markdown=True,
)

# --------------- USER INPUT HANDLING -------------------
prompt = st.text_input("What's your NYC news question? (e.g., 'Times Square breaking news')")

if prompt:
    st.session_state["counter"] = 1
    
    with st.spinner("🕵️♂️ Sniffing out the story..."):
        # stream = True
        if stream:
            response_stream: Iterator[RunResponse] = agent.run(prompt, stream=True)
            response_text = ""
            placeholder = st.empty()
            
            for chunk in response_stream:
                response_text += chunk.content
                placeholder.markdown(response_text + "▌")
                st.session_state["counter"] += 1
                with st.sidebar:
                    counter_placeholder.caption(f"Queries processed: {st.session_state['counter']}")
            
            placeholder.markdown(response_text)
        else:
            response = agent.run(prompt, stream=False)
            st.markdown(response.content)
            st.session_state["counter"] += 1
            with st.sidebar:
                counter_placeholder.caption(f"Queries processed: {st.session_state['counter']}")

# --------------- FOOTER & INFO -------------------
st.markdown("---")
st.caption("""
**NYC Reporter Bot Features:**
- 100% authentic attitude 🇺🇸
- Certified bodega-approved news 🌃
- Guaranteed to mention pizza at least once 🍕
""")