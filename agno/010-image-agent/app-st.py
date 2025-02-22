import os
import base64
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.media import Image
from agno.tools.duckduckgo import DuckDuckGoTools
from typing import Dict, Iterator, Optional

from agno.models.openai import OpenAIChat
from agno.models.xai import xAI
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.models.groq import Groq
# --------- LOAD API KEY ---------
import os
# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
xai_api_key = os.getenv("XAI_API_KEY")
if not xai_api_key:
    st.error("xAI API key not found. Please set the XAI_API_KEY environment variable.")
    st.stop()

# --------------- TITLE AND INFO SECTION -------------------
st.title("🎨 AI Image Reporter")
st.write("Your visual analysis & news companion with expert image storytelling!")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Try These Image Examples:")
    st.markdown("""
    **Famous Landmarks:**
    - Eiffel Tower
    - Taj Mahal
    - Golden Gate Bridge
    - Great Wall of China
    - Sydney Opera House
    
    **Prompt Ideas:**
    - "Latest news about this location"
    - "Historical significance"
    - "Cultural events here"
    - "Architectural analysis"
    - "Recent developments"
    """)
    st.markdown("---")
    
    # Stats
    if "analysis_count" not in st.session_state:
        st.session_state.analysis_count = 0
    st.caption(f"Images analyzed: {st.session_state.analysis_count}")

stream = st.sidebar.checkbox("Stream analysis process")

# --------------- AGENT INITIALIZATION -------------------
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
    # model=xAI(id="grok-2", api_key=xai_api_key),
    # model=OpenAIChat(id="gpt-4o-mini"),
    # model=xAI(id="grok-2"),
    # model=Groq(id="llama-3.3-70b-versatile"),
    # model=DeepSeek(id="deepseek-chat"),
    # model=Gemini(
    #     id="gemini-2.0-flash-exp",
    #     api_key=gemini_api_key,
    # ),
    tools=[DuckDuckGoTools()],
    instructions=dedent("""\
        You're a world-class visual journalist! 📸✨
        
        **Analysis Protocol:**
        1. Start with EMOJI HEADLINE
        2. Break down visual elements
        3. Research current events
        4. Connect historical context
        5. Verify facts through web search
        6. End with memorable sign-off
        
        **Reporting Style:**
        - Professional yet engaging
        - Vivid descriptive language
        - Cultural/historical references
        - Journalistic integrity
        - Human interest angles\
    """),
    markdown=True,
    show_tool_calls=True,
)

# --------------- USER INPUT HANDLING -------------------
col1, col2 = st.columns(2)
with col1:
    # Image input section
    uploaded_file = st.file_uploader("Drag & drop image", 
                                   type=["jpg", "jpeg", "png"],
                                   help="Upload an image or provide URL below")
    st.markdown("<p style='text-align: center; margin: 0.5rem;'>OR</p>", 
              unsafe_allow_html=True)
    image_url = st.text_input("Image URL:", 
                            placeholder="Paste image URL here",
                            help="Provide a direct image URL")
with col2:
    prompt = st.text_input("Analysis prompt:", 
                         placeholder="e.g., 'Latest news about this location'")

# log = st.expander("Workflow Actions", expanded=False)
# log = log.write("Workflow actions..")

if (uploaded_file or image_url) and prompt:
    st.session_state.analysis_count += 1
    
    # Handle image input
    if uploaded_file:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Convert to base64 for processing
        bytes_data = uploaded_file.getvalue()
        base64_data = base64.b64encode(bytes_data).decode()
        mime_type = uploaded_file.type
        data_url = f"data:{mime_type};base64,{base64_data}"
        images = [Image(url=data_url)]
    else:
        # Display URL image
        st.image(image_url, caption=f"Image URL: {image_url}", use_container_width=True)
        images = [Image(url=image_url)]
    
    st.markdown("---")
    
    with st.spinner("🔍 Analyzing image and researching context..."):
        # Add workflow actions expander
        # with st.expander("Workflow Actions", expanded=False):
        #     action_log = st.empty()
        
        # workflow_actions = []
        
        if stream:
            response_stream: Iterator[RunResponse] = agent.run(
                prompt,
                images=images,
                stream=True
            )
            
            response_text = ""
            placeholder = st.empty()
            
            for chunk in response_stream:
                response_text += chunk.content
                
                # Log tool calls
                # if hasattr(chunk, 'tool_name'):
                #     workflow_actions.append(f"🔧 {chunk.tool_name} tool executed")
                #     # action_log.info("\n\n".join(workflow_actions))
                #     log.write(workflow_actions)
                
                placeholder.markdown(response_text + "▌")
            
            placeholder.markdown(response_text)
        else:
            response = agent.run(prompt, images=images, stream=False)
            st.markdown(response.content)

# --------------- FOOTER & INFO -------------------
st.markdown("---")
st.caption("""
**Image Reporter Features:**
- Visual analysis + news integration 📰
- Historical context + current events ⏳
- Cultural insights + architectural review 🏛️
- Fact-checked reporting ✅
- Journalistic storytelling ✍️
""")

# Hidden dependency note
st.markdown("<!--- Run `pip install duckduckgo-search agno` -->", unsafe_allow_html=True)