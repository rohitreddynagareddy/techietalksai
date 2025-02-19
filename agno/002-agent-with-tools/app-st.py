
from typing import Iterator
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

# --------- LOAD API KEY ---------
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
if not gemini_api_key:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()


# --------------- TITLE AND INFO SECTION -------------------
st.title("🗽 NYC Web-Savvy News Reporter")
st.write("Your street-smart AI news buddy that hunts down fresh stories with authentic NYC attitude!")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Try These NYC Scoops:")
    st.markdown("""
    - Wall Street breaking story
    - Central Park latest incident
    - Yankees Stadium updates
    - New Broadway show buzz
    - Manhattan restaurant opening
    - Subway system developments
    """)
    st.markdown("---")
    
    # Query counter
    counter_placeholder = st.empty()
    if "counter" not in st.session_state:
        st.session_state["counter"] = 0
    counter_placeholder.caption(f"Stories investigated: {st.session_state['counter']}")

stream = st.sidebar.checkbox("Live Reporting Mode")

# --------------- AGENT INITIALIZATION -------------------
agent = Agent(
    # model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
    # model=DeepSeek(id="deepseek-chat"),
    model=Gemini(id="gemini-2.0-flash-exp", api_key=gemini_api_key,),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    instructions=dedent("""\
        You're the most connected news reporter in NYC! 🗽
        Equal parts digital sleuth and sidewalk poet. Always verify facts through web searches!

        **Reporting Protocol:**
        1. Start with EMOJI HEADLINE that slaps
        2. Hit the web for fresh intel 🔍
        3. Structure your scoop:
           - Breaking news summary
           - Key details from sources
           - Local impact analysis
        4. Season with NYC slang and attitude
        5. Sign off with reporter flair

        **Required NYC Elements:**
        - Pizza rat references 🐀🍕
        - Bodega cat sightings 🐈⬛
        - Subway drama updates 🚇
        - Traffic horn soundtrack 🚗📢
        - Brooklyn vs Manhattan rivalry

        **Sign-off Examples:**
        - 'Back to you in the studio, suckers!'
        - 'Reporting from the concrete jungle!'
        - 'This is Tony PizzaRat, LIVE from a fire escape!'

        **Fact-Check Rules:**
        - Always verify through web search
        - Cite latest updates
        - Flag unconfirmed rumors\
    """),
    markdown=True,
    debug_mode=True,
)

# --------------- USER INPUT HANDLING -------------------
prompt = st.text_input("What NYC story should we chase? (e.g., 'Times Square breaking news')")

if prompt:
    st.session_state["counter"] += 1
    
    with st.spinner("🕵️♂️ Sniffing out the story through back alleys and search engines..."):
        if stream:
            response_stream: Iterator[RunResponse] = agent.run(prompt, stream=True)
            response_text = ""
            placeholder = st.empty()
            
            for chunk in response_stream:
                response_text += chunk.content
                placeholder.markdown(response_text + "▌")
            
            placeholder.markdown(response_text)
        else:
            response = agent.run(prompt, stream=False)
            st.markdown(response.content)

    # Update counter display
    with st.sidebar:
        counter_placeholder.caption(f"Stories investigated: {st.session_state['counter']}")

# --------------- FOOTER & INFO -------------------
st.markdown("---")
st.caption("""
**NYC Reporter Features:**
- Web-powered news hunting 🔍
- 100% authentic NYC attitude 🇺🇸
- Bodega-certified fact checking 🥤
- Guaranteed subway references 🚇
""")

# Dependencies note (hidden but useful for documentation)
st.markdown("<!--- Run `pip install openai duckduckgo-search agno` for dependencies -->", unsafe_allow_html=True)

