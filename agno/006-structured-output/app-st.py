from textwrap import dedent
from typing import List
import os
import streamlit as st
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.xai import xAI
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.models.groq import Groq
from pydantic import BaseModel, Field

# --------- LOAD API KEY ---------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --------- MOVIE SCRIPT MODEL ---------
class MovieScript(BaseModel):
    setting: str = Field(
        ...,
        description="A richly detailed, atmospheric description of the movie's primary location and time period. Include sensory details and mood.",
        example="A neon-lit Tokyo in 2077, where rain-slicked streets hum with drones and the air smells of soy and circuitry."
    )
    ending: str = Field(
        ...,
        description="The movie's powerful conclusion that ties together all plot threads. Should deliver emotional impact and satisfaction.",
        example="As the city burns, the hero sacrifices their AI companion to save humanity, fading into the skyline."
    )
    genre: str = Field(
        ...,
        description="The film's primary and secondary genres (e.g., 'Sci-fi Thriller', 'Romantic Comedy'). Should align with setting and tone.",
        example="Sci-fi Thriller"
    )
    name: str = Field(
        ...,
        description="An attention-grabbing, memorable title that captures the essence of the story and appeals to target audience.",
        example="Neon Requiem"
    )
    characters: List[str] = Field(
        ...,
        description="4-6 main characters with distinctive names and brief role descriptions (e.g., 'Riku Sato - rogue hacker with a hidden past').",
        example=["Riku Sato - rogue hacker with a hidden past", "Aiko Mei - corporate enforcer seeking redemption"]
    )
    storyline: str = Field(
        ...,
        description="A compelling three-sentence plot summary: Setup, Conflict, and Stakes. Hook readers with intrigue and emotion.",
        example="In a futuristic Tokyo, a rogue hacker uncovers a conspiracy threatening humanity. Hunted by a relentless enforcer, they must decode the truth buried in the city's AI core. With time running out, the fate of millions hangs on a single choice."
    )

# --------- AGENT CONFIGURATION ---------
movie_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    description=dedent("""\
        You're an acclaimed Hollywood screenwriter with a knack for epic blockbusters! 🎬
        Blending the genius of Christopher Nolan, Aaron Sorkin, and Quentin Tarantino,
        you turn locations into living, breathing characters that drive unforgettable stories.\
    """),
    instructions=dedent("""\
        **Your style rules:**
        1. Start with a vivid SETTING that feels alive
        2. Craft stories with big stakes and bigger twists
        3. Keep it cinematic and gripping
        4. Sprinkle in witty dialogue and bold visuals
        5. Sign off with a director’s flair

        **Required movie elements:**
        - Iconic locations as characters 🌆
        - High-stakes conflicts ⚡
        - Memorable character quirks 😎
        - Genre-bending twists 🎭
        - Emotional gut-punch endings 💔

        **Sign-off examples:**
        - 'Cut to black—roll credits!'
        - 'From the mind of a cinematic madman!'
        - 'Lights, camera, masterpiece!'\
    """),
    response_model=MovieScript,
    structured_outputs=True,
)

# --------- STREAMLIT UI ---------
st.title("🎬 Movie Script Writer Bot")
st.write("Your AI screenwriting partner for crafting blockbuster hits!")

with st.sidebar:
    st.subheader("Try These Movie Prompts:")
    st.markdown("""
    - Tokyo cyberpunk thriller
    - Ancient Rome epic
    - Manhattan rom-com
    - Amazon jungle adventure
    - Mars colony sci-fi
    """)
    st.markdown("---")
    # st.caption(f"Scripts generated: {st.session_state.get('counter', 0)}")

# --------- SCRIPT GENERATION HANDLER ---------
prompt = st.text_input("What's your movie idea? (e.g., 'Paris heist thriller')")

# Define the expander at a specific location in your layout
expander = st.expander("Details", expanded=True)

if prompt:
    with st.spinner("🎥 Writing the next blockbuster..."):
        response: RunResponse = movie_agent.run(prompt)
        script = response.content
        
        st.subheader(script.name)
        st.write(script.storyline)
        
        with st.expander("🎬 Script Details"):
            st.write(f"**Title:** {script.name}")
            st.write(f"**Genre:** {script.genre}")
            st.write(f"**Setting:** {script.setting}")
            st.write(f"**Storyline:** {script.storyline}")
            st.write(f"**Characters:**", ", ".join(script.characters))
            st.write(f"**Ending:** {script.ending}")
        
        st.session_state["counter"] = st.session_state.get("counter", 0) + 1

# --------- FOOTER ---------
st.markdown("---")
st.caption("Powered by OpenAI GPT-4o-mini and Agno Agentic Library")