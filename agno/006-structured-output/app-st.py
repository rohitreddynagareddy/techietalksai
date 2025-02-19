from textwrap import dedent
from typing import List
import os
import streamlit as st
from agno.agent import Agent, RunResponse
# from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field

# --------- NEWS REPORT MODEL ---------
class NewsReport(BaseModel):
    headline: str = Field(
        ...,
        description="Attention-grabbing headline starting with relevant emoji and containing NYC slang",
        example="🐀🍕 Pizza Rat Crisis: Bodega Cats Mobilize to Protect Their Turf!"
    )
    breaking_news: str = Field(
        ...,
        description="150-word max hard-hitting news piece with sassy New York attitude",
        example="Forget the subway delays, the real crisis is..."
    )
    locations: List[str] = Field(
        ...,
        description="3-5 NYC-specific locations mentioned in the story",
        example=["East Village Bodega", "Brooklyn Bridge", "Times Square"]
    )
    quotes: List[str] = Field(
        ...,
        description="2-3 spicy quotes from fictional NYC personalities",
        example=["'Yo, I seen rats bigger than my cab!' - Tony the Cabbie"]
    )
    hashtags: List[str] = Field(
        ...,
        description="5-7 trending hashtags combining NYC culture and news angles",
        example=["#BodegaShowdown", "#SubwayRatAlert"]
    )

# --------- AGENT CONFIGURATION ---------
nyc_agent = Agent(
    # model=DeepSeek(id="deepseek-chat"),
    model=OpenAIChat(id="gpt-4o-mini"),
    description=dedent("""\
        You're the most NYC news reporter ever! 🗽
        Equal parts stand-up comic and hard-hitting journalist.
        Specializes in turning urban incidents into viral stories with local flavor.\
    """),
    instructions=dedent("""\
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
    response_model=NewsReport,
    structured_outputs=True,
)

# --------- STREAMLIT UI ---------
st.title("🗽 NYC News Reporter Bot")
st.write("Your sassy AI news buddy with that authentic New York attitude!")

with st.sidebar:
    st.subheader("Try These NYC Prompts:")
    st.markdown("""
    - Central Park latest scoop
    - Wall Street breaking story
    - Yankees game drama
    - Broadway show scandal
    - Brooklyn gentrification fight
    """)
    st.markdown("---")
    # st.caption(f"Stories generated: {st.session_state.get('counter', 0)}")

# --------- NEWS GENERATION HANDLER ---------
prompt = st.text_input("What's your NYC news question? (e.g., 'Times Square rat uprising')")

if prompt:
    with st.spinner("🕵️♂️ Sniffing out the story..."):
        response: RunResponse = nyc_agent.run(prompt)
        report = response.content
        
        st.subheader(report.headline)
        st.write(report.breaking_news)
        
        with st.expander("📌 Story Details"):
            st.write("**Locations:**", ", ".join(report.locations))
            st.write("**Key Quotes:**")
            for quote in report.quotes:
                st.write(f"- {quote}")
            st.write("**Trending Tags:**", " ".join(report.hashtags))
        
        st.session_state["counter"] = st.session_state.get("counter", 0) + 1

# --------- EXAMPLE USAGE ---------
# To test in console:
# from rich.pretty import pprint
# response: RunResponse = nyc_agent.run("Subway rat beauty pageant scandal")
# pprint(response.content.dict())

# Example output structure:
"""
{
    'headline': '👑🐀 Subway Rat Pageant Crowned in Midnight Metro Drama',
    'breaking_news': 'Under the flickering lights of the 42nd Street station...',
    'locations': ['Times Square Station', 'Chinatown Express', 'LES Sewer Access'],
    'quotes': [
        "'Dis ain't no Disney parade!' - Maria Gonzalez, Pageant Judge",
        "'My Fluffy was robbed!' - Crazy Cat Lady of Bryant Park"
    ],
    'hashtags': ['#RatPageant2024', '#BodegaCatsVSSubwayRats', '#NYCDrama']
}
"""