import os
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.tools.dalle import DalleTools
from typing import Dict, Iterator, Optional

# --------- LOAD MODELS AND API KEYS ---------
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

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()


# --------------- TITLE AND INFO SECTION -------------------
st.title("🎨 AI Art Studio")
st.write("Create stunning AI-generated artwork with DALL-E")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Art Prompt Ideas:")
    st.markdown("""
    - Surreal floating city in clouds
    - Cyberpunk samurai portrait
    - Underwater bioluminescent city
    - Cozy cabin in snowy forest
    - Futuristic cityscape with flying cars
    """)
    st.markdown("---")
    
    # Stats
    if "art_count" not in st.session_state:
        st.session_state.art_count = 0
    st.caption(f"Artworks created: {st.session_state.art_count}")

stream = st.sidebar.checkbox("Stream creation process")

# --------------- AGENT INITIALIZATION -------------------
image_agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    tools=[DalleTools()],
    instructions=dedent("""\
        You're a professional AI artist! 🎨
        
        **Creation Protocol:**
        1. Analyze prompt for style/mood
        2. Enhance with artistic details
        3. Generate high-quality images
        4. Explain artistic choices
        5. Offer refinement suggestions
        
        **Artistic Styles:**
        - Photorealism
        - Surrealism
        - Cyberpunk
        - Fantasy
        - Abstract\
    """),
    markdown=True,
    show_tool_calls=True,
)

# --------------- USER INPUT HANDLING -------------------
prompt = st.text_input("Describe your artwork:", 
                      placeholder="e.g., 'Magical library with floating books'")

if prompt:
    st.session_state.art_count += 1
    
    with st.spinner("🎨 Creating your masterpiece..."):
        # Add workflow actions expander
        # with st.expander("Creation Process", expanded=False):
        #     action_log = st.empty()
        
        workflow_actions = []
        
        if stream:
            response_stream: Iterator[RunResponse] = image_agent.run(prompt, stream=True)
            response_text = ""
            placeholder = st.empty()
            
            for chunk in response_stream:
                response_text += chunk.content
                
                # Log tool calls
                if hasattr(chunk, 'tool_name'):
                    workflow_actions.append(f"🖌️ {chunk.tool_name.replace('_', ' ').title()}")
                    action_log.info("\n\n".join(workflow_actions))
                
                placeholder.markdown(response_text + "▌")
            
            placeholder.markdown(response_text)
        else:
            response = image_agent.run(prompt, stream=False)
            st.markdown(response.content)

    # Display generated images
    images = image_agent.get_images()
    if images:
        cols = st.columns(len(images))
        for idx, img in enumerate(images):
            with cols[idx]:
                st.image(img.url, caption=f"Artwork {idx+1}", use_container_width=True)
                st.download_button(
                    label=f"Download Artwork {idx+1}",
                    data=img.url,
                    file_name=f"artwork_{idx+1}.png",
                    mime="image/png"
                )
    else:
        st.warning("No images generated. Please try a different prompt.")

# --------------- FOOTER & INFO -------------------
st.markdown("---")
st.caption("""
**Art Studio Features:**
- Multiple artistic styles 🖼️
- High-resolution image generation 📸
- Artistic choice explanations 🎭
- Direct image downloads ⬇️
- Creative prompt enhancement ✨
""")

# Hidden dependency note
st.markdown("<!--- Run `pip install openai agno` -->", unsafe_allow_html=True)