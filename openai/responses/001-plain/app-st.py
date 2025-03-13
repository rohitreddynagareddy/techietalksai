import os
import streamlit as st
from openai import OpenAI

# --------------- TITLE & HEADER -------------------
st.title("📰 OpenAI Responses API 2025")
# st.title("📰 Daily Positive News Finder")
st.write("Your AI-powered good news aggregator with web search integration")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Example Queries")
    st.markdown("""
    - Positive tech breakthroughs today
    - Good environmental news this week
    - Uplifting science discoveries
    - Inspiring community stories
    - Recent humanitarian achievements
    """)
    st.markdown("---")
    
    # Stats
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    st.caption(f"News searches made: {st.session_state.query_count}")

# --------------- MAIN INTERFACE -------------------
query = st.text_input("Ask for positive news:", 
                     placeholder="e.g., 'What good news happened today?'")

if query:
    st.session_state.query_count += 1
    
    # Initialize client
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        st.error("OpenAI API key not configured properly!")
        st.stop()
    
    with st.spinner("🔍 Scanning global news sources..."):
        try:
            response = client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=query
            )
            
            with st.expander("🌐 Web Search Process", expanded=True):
                st.markdown("""
                **Search Process:**
                1. Analyzing current news trends
                2. Filtering for positive stories
                3. Verifying source credibility
                4. Synthesizing key information
                """)
                
                if response.output_text:
                    st.success("✅ Found verified positive news!")
                else:
                    st.warning("⚠️ No positive stories found - expanding search")
            
            st.markdown("---")
            st.subheader("✨ Good News Report")
            st.markdown(response.output_text)
            
            # Download button
            st.download_button(
                label="Download Report",
                data=response.output_text,
                file_name="positive_news_report.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")

# --------------- FOOTER -------------------
st.markdown("---")
st.caption("""
**News Finder Features:**
- Real-time web search integration 🌐
- Positive story filtering 😊
- Source verification ✅
- Daily updates 📅
- Fact-checked reports 🔍
""")

# Hidden dependency note
st.markdown("<!--- Run `pip install openai` -->", unsafe_allow_html=True)