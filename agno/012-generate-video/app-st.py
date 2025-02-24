import os
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.models_labs import ModelsLabTools

# --------- LOAD API KEY ---------
openai_api_key = os.getenv("OPENAI_API_KEY")
models_lab_key = os.getenv("MODELS_LAB_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Set OPENAI_API_KEY")
    st.stop()
if not models_lab_key:
    st.error("ModelsLab API key not found. Set MODELS_LAB_API_KEY")
    st.stop()

# --------------- TITLE AND INFO SECTION -------------------
st.title("🎥 AI Video Studio")
st.write("Create stunning AI-generated videos with ModelsLab")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Video Prompt Ideas:")
    st.markdown("""
    - Cosmic nebula journey
    - Autumn leaves in forest
    - Peaceful koi pond
    - Cozy fireplace flames
    - Magical portal opening
    """)
    st.markdown("---")
    
    # Stats
    if "video_count" not in st.session_state:
        st.session_state.video_count = 0
    st.caption(f"Videos created: {st.session_state.video_count}")

stream = st.sidebar.checkbox("Stream creation process")

# --------------- AGENT INITIALIZATION -------------------
video_agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    tools=[ModelsLabTools(api_key=models_lab_key)],
    instructions=dedent("""\
        You're a professional video director! 🎬
        
        **Creation Protocol:**
        1. Analyze prompt for style/mood
        2. Enhance with motion details
        3. Generate high-quality videos
        4. Explain creative choices
        5. Offer refinement suggestions
        
        **Video Styles:**
        - Cinematic
        - Animated
        - Nature
        - Abstract
        - Fantasy\
    """),
    markdown=True,
    show_tool_calls=True,
)

# --------------- USER INPUT HANDLING -------------------
prompt = st.text_input("Describe your video:", 
                      placeholder="e.g., 'Cosmic journey through a nebula'")

if prompt:
    st.session_state.video_count += 1
    
    with st.spinner("🎥 Directing your video..."):
        # Add workflow actions expander
        with st.expander("Creation Process", expanded=False):
            action_log = st.empty()
        
        workflow_actions = []
        
        if stream:
            response_stream: Iterator[RunResponse] = video_agent.run(prompt, stream=True)
            response_text = ""
            placeholder = st.empty()
            
            for chunk in response_stream:
                response_text += chunk.content
                
                # Log tool calls
                if hasattr(chunk, 'tool_name'):
                    workflow_actions.append(f"🎞️ {chunk.tool_name.replace('_', ' ').title()}")
                    action_log.info("\n\n".join(workflow_actions))
                
                placeholder.markdown(response_text + "▌")
            
            placeholder.markdown(response_text)
        else:
            response = video_agent.run(prompt, stream=False)
            st.markdown(response.content)

    # Display generated videos
    videos = video_agent.get_videos()
    if videos:
        cols = st.columns(len(videos))
        for idx, vid in enumerate(videos):
            with cols[idx]:
                st.video(vid.url)
                st.download_button(
                    label=f"Download Video {idx+1}",
                    data=vid.url,
                    file_name=f"video_{idx+1}.mp4",
                    mime="video/mp4"
                )
    else:
        st.warning("No videos generated. Please try a different prompt.")

# --------------- FOOTER & INFO -------------------
st.markdown("---")
st.caption("""
**Video Studio Features:**
- Multiple video styles 🎞️
- HD video generation 📽️
- Creative direction insights 🎨
- Direct video downloads ⬇️
- Dynamic motion control 🕹️
""")

# Hidden dependency note
st.markdown("<!--- Run `pip install openai agno` -->", unsafe_allow_html=True)