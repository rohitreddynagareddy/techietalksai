import os
import base64
import streamlit as st
from textwrap import dedent
import requests
from agno.agent import Agent, RunResponse
from agno.media import Audio
from agno.utils.audio import write_audio_to_file

# --------- LOAD API KEY ---------
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
st.title("🎤 AI Voice Analyst")
st.write("Advanced audio processing and natural voice interactions")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Try These Examples:")
    st.markdown("""
    - Conversation analysis
    - Emotion/tone detection
    - Speech pattern evaluation
    - Background noise check
    - Multilingual processing
    """)
    st.markdown("---")
    
    # Stats
    if "audio_count" not in st.session_state:
        st.session_state.audio_count = 0
    st.caption(f"Audio processed: {st.session_state.audio_count}")

voice = st.sidebar.selectbox("Response Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
# stream = st.sidebar.checkbox("Stream analysis process")

# --------------- AGENT INITIALIZATION -------------------
agent = Agent(
    model=OpenAIChat(
        # id="gpt-4o-audio-preview",
        # id="gpt-4o-mini-audio-preview",
        id="gpt-4o-mini-audio-preview-2024-12-17",
        api_key=openai_api_key,
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
    ),
    instructions=dedent("""\
        You're a voice interaction expert! 🎧
        
        **Analysis Protocol:**
        1. Analyze content & context
        2. Detect tone/emotion
        3. Evaluate speech clarity
        4. Identify background noise
        5. Provide concise summary
        
        **Response Guidelines:**
        - Natural conversational tone
        - Address main points clearly
        - Highlight key observations
        - Multilingual support
        - Technical audio analysis\
    """)
)

# --------------- USER INPUT HANDLING -------------------
col1, col2 = st.columns(2)
with col1:
    # Audio input options
    uploaded_file = st.file_uploader("Upload audio", 
                                   type=["wav", "mp3"],
                                   help="Upload audio file or provide URL below")
    st.markdown("<p style='text-align: center; margin: 0.5rem;'>OR</p>", 
              unsafe_allow_html=True)
    audio_url = st.text_input("Audio URL:", 
                            placeholder="Paste audio URL here",
                            help="Direct link to audio file")
with col2:
    prompt = st.text_input("Analysis prompt:", 
                         placeholder="e.g., 'Analyze tone and content'")

if (uploaded_file or audio_url) and prompt:
    st.session_state.audio_count += 1
    
    # Handle audio input
    audio_data = None
    if uploaded_file:
        st.audio(uploaded_file, format=uploaded_file.type)
        audio_data = uploaded_file.read()
    else:
        try:
            response = requests.get(audio_url)
            response.raise_for_status()
            st.audio(audio_url, format="audio/wav")
            audio_data = response.content
        except Exception as e:
            st.error(f"Error loading audio URL: {str(e)}")
            st.stop()
    
    # Process audio
    with st.spinner("🔍 Analyzing audio content..."):
        # Add workflow actions expander
        # with st.expander("Processing Steps", expanded=False):
        #     action_log = st.empty()
        
        workflow_actions = []
        
        # if stream:
        #     response_stream = agent.run(
        #         prompt,
        #         audio=[Audio(content=audio_data, format="wav")],
        #         stream=True
        #     )
            
        #     response_text = ""
        #     placeholder = st.empty()
            
        #     for chunk in response_stream:
        #         response_text += chunk.content
        #         placeholder.markdown(response_text + "▌")
            
        #     placeholder.markdown(response_text)
        # else:
        response = agent.run(
            prompt,
            audio=[Audio(content=audio_data, format="wav")],
            stream=False
        )
        # st.markdown(response.content)

    # Handle audio response
    if agent.run_response.response_audio is not None:
        # st.markdown("---")
        st.subheader("AI Voice Response")
        filename = "tmp/response.wav"
        write_audio_to_file(
            audio=agent.run_response.response_audio.content, filename=filename
        )        
        # Save and display audio
        # audio_bytes = agent.run_response.response_audio.content
        # st.audio(audio_bytes, format="audio/wav")
        st.audio(filename, format="audio/wav")
        
        # Download button
        # st.download_button(
        #     label="Download Response",
        #     data=agent.run_response.response_audio.content,
        #     file_name="ai_response.wav",
        #     mime="audio/wav"
        # )

# --------------- FOOTER & INFO -------------------
st.markdown("---")
st.caption("""
**Voice Analyst Features:**
- Speech content analysis 🗣️
- Emotional tone detection 😊🎻
- Background noise evaluation 🔊
- Multilingual processing 🌍
- Natural voice responses 🎧
""")

# Hidden dependency note
st.markdown("<!--- Run `pip install openai requests agno` -->", unsafe_allow_html=True)