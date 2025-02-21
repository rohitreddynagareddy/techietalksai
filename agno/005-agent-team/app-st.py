# app/main.py
from typing import Iterator
import streamlit as st
from textwrap import dedent
import os
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.xai import xAI
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.models.groq import Groq
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
        model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
        # model=xAI(id="grok-2"),
        # model=Groq(id="llama-3.3-70b-versatile"),
        # model=DeepSeek(id="deepseek-chat"),
        # model=Gemini(
        #     id="gemini-2.0-flash-exp",
        #     api_key=gemini_api_key,
        # ),
        tools=[DuckDuckGoTools()],
        # instructions=dedent("""\
        #     🔍 Financial News Analyst Protocol:
        #     1. Source latest market-moving news from reputable outlets
        #     2. Cross-verify information across 3+ sources
        #     3. Include timestamps and direct quotes
        #     4. Highlight regulatory changes and earnings reports
        #     5. Analyze industry sentiment trends\
        # """),        
        instructions=dedent("""\
            You are an experienced web researcher and news analyst! 🔍

            Follow these steps when searching for information:
            1. Start with the most recent and relevant sources
            2. Cross-reference information from multiple sources
            3. Prioritize reputable news outlets and official sources
            4. Always cite your sources with links
            5. Focus on market-moving news and significant developments

            Your style guide:
            - Present information in a clear, journalistic style
            - Use bullet points for key takeaways
            - Include relevant quotes when available
            - Specify the date and time for each piece of news
            - Highlight market sentiment and industry trends
            - End with a brief analysis of the overall narrative
            - Pay special attention to regulatory news, earnings reports, and strategic announcements\
        """),
        show_tool_calls=True,
        markdown=True,
        add_references=True,
        debug_mode=True,
    )

    # Financial Data Agent
    finance_agent = Agent(
        name="Finance Agent",
        role="Market data analyst",
        model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
        # model=xAI(id="grok-2"),
        # model=Groq(id="llama-3.3-70b-versatile"),
        # model=DeepSeek(id="deepseek-chat"),
        # model=Gemini(
        #     id="gemini-2.0-flash-exp",
        #     api_key=gemini_api_key,
        # ),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        # instructions=dedent("""\
        #     📊 Financial Analyst Protocol:
        #     1. Present real-time stock metrics (price, volume, P/E ratio)
        #     2. Analyze 52-week performance vs sector indices
        #     3. Compare analyst target prices to current values
        #     4. Highlight unusual trading activity
        #     5. Include institutional ownership trends\
        # """),        
        instructions=dedent("""\
            You are a skilled financial analyst with expertise in market data! 📊

            Follow these steps when analyzing financial data:
            1. Start with the latest stock price, trading volume, and daily range
            2. Present detailed analyst recommendations and consensus target prices
            3. Include key metrics: P/E ratio, market cap, 52-week range
            4. Analyze trading patterns and volume trends
            5. Compare performance against relevant sector indices

            Your style guide:
            - Use tables for structured data presentation
            - Include clear headers for each data section
            - Add brief explanations for technical terms
            - Highlight notable changes with emojis (📈 📉)
            - Use bullet points for quick insights
            - Compare current values with historical averages
            - End with a data-driven financial outlook\
        """),
        show_tool_calls=True,
        markdown=True,
        add_references=True,
        debug_mode=True,
    )

    # Lead Editor Agent
    return Agent(
        team=[web_agent, finance_agent],
        model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
        # model=xAI(id="grok-2"),
        # model=Groq(id="llama-3.3-70b-versatile"),
        # model=DeepSeek(id="deepseek-chat"),
        # model=Gemini(
        #     id="gemini-2.0-flash-exp",
        #     api_key=gemini_api_key,
        # ),
        # instructions=dedent("""\
        #     📰 Chief Editor Protocol:
        #     1. Combine news and data into executive summary
        #     2. Structure report sections:
        #        - Market Snapshot
        #        - Key Developments
        #        - Financial Analysis
        #        - Risk Assessment
        #     3. Use tables/charts for financial metrics
        #     4. Include 5 key takeaways
        #     5. Conclude with 3-month outlook\
        # """),
        instructions=dedent("""\
            You are the lead editor of a prestigious financial news desk! 📰

            Your role:
            1. Coordinate between the web researcher and financial analyst
            2. Combine their findings into a compelling narrative
            3. Ensure all information is properly sourced and verified
            4. Present a balanced view of both news and data
            5. Highlight key risks and opportunities

            Your style guide:
            - Start with an attention-grabbing headline
            - Begin with a powerful executive summary
            - Present financial data first, followed by news context
            - Use clear section breaks between different types of information
            - Include relevant charts or tables when available
            - Add 'Market Sentiment' section with current mood
            - Include a 'Key Takeaways' section at the end
            - End with 'Risk Factors' when appropriate
            - Sign off with 'Market Watch Team' and the current date\
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        add_references=True,
        debug_mode=True,
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

stream = st.sidebar.checkbox("Stream")

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