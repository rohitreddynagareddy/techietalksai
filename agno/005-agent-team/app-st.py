# app/main.py
from typing import Iterator
import streamlit as st
from textwrap import dedent
import os
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# --------- LOAD API KEY ---------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --------------- TITLE AND INFO SECTION -------------------
st.title("📈 Financial Analyst Team")
st.write("Your AI-powered market research squad with real-time data and news analysis")

# --------------- AGENT TEAM INITIALIZATION -------------------
def create_agent_team():
    # Web Research Agent
    web_agent = Agent(
        name="Web Agent",
        role="Financial news researcher",
        model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        tools=[DuckDuckGoTools()],
        instructions=dedent("""\
            🔍 Financial News Analyst Protocol:
            1. Source latest market-moving news from reputable outlets
            2. Cross-verify information across 3+ sources
            3. Include timestamps and direct quotes
            4. Highlight regulatory changes and earnings reports
            5. Analyze industry sentiment trends\
        """),
        show_tool_calls=True,
        markdown=True,
    )

    # Financial Data Agent
    finance_agent = Agent(
        name="Finance Agent",
        role="Market data analyst",
        model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions=dedent("""\
            📊 Financial Analyst Protocol:
            1. Present real-time stock metrics (price, volume, P/E ratio)
            2. Analyze 52-week performance vs sector indices
            3. Compare analyst target prices to current values
            4. Highlight unusual trading activity
            5. Include institutional ownership trends\
        """),
        show_tool_calls=True,
        markdown=True,
    )

    # Lead Editor Agent
    return Agent(
        team=[web_agent, finance_agent],
        model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        instructions=dedent("""\
            📰 Chief Editor Protocol:
            1. Combine news and data into executive summary
            2. Structure report sections:
               - Market Snapshot
               - Key Developments
               - Financial Analysis
               - Risk Assessment
            3. Use tables/charts for financial metrics
            4. Include 5 key takeaways
            5. Conclude with 3-month outlook\
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
    )

agent_team = create_agent_team()

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Example Queries")
    st.code("""
    AAPL news + financials
    NVDA AI impact analysis
    EV sector: TSLA vs RIVN
    Semiconductor market outlook
    MSFT recent performance
    """)
    
    st.markdown("---")
    st.subheader("Agent Team Capabilities")
    st.write("🔍 Web Agent: Real-time news analysis")
    st.write("📊 Finance Agent: Market data & metrics")
    st.write("📰 Lead Editor: Integrated reports")

# --------------- USER INPUT & DISPLAY -------------------
query = st.text_input("Enter financial query (e.g., 'AAPL news and stock analysis')")

if query:
    with st.spinner("🔍 Assembling market intelligence..."):
        stream = True
        if stream:
            response_stream = agent_team.run(query, stream=True)
            response_text = ""
            report_placeholder = st.empty()
            
            for chunk in response_stream:
                response_text += chunk.content
                report_placeholder.markdown(response_text + "▌")
            
            report_placeholder.markdown(response_text)
        else:
            response = agent_team.run(query, stream=False)
            st.markdown(response.content)

# --------------- FOOTER & INFO -------------------
st.markdown("---")
st.caption("""
**Data Sources**: 
- Real-time market data from Yahoo Finance
- News analysis from web sources
- AI-powered insights from GPT-4o
""")
st.caption("Note: Response times vary based on query complexity (typically 15-45 seconds)")