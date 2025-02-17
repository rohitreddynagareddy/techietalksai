from typing import Iterator
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType

# --------- LOAD API KEY ---------
# Load OpenAI API key from environment
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
# --------------- TITLE AND INFO SECTION -------------------

# Sidebar with example prompts
with st.sidebar:
    st.subheader("Try These Thai Cooking Queries:")
    st.markdown("""
    * How make authentic Pad Thai?
    * Difference between red/green curry?
    * What's galangal substitutes?
    * History of Tom Yum soup?
    * Essential Thai pantry items?
    * How make Pad Kra Pao?
    """)
    st.markdown("---")
    st.write("📚 Knowledge base: 50+ authentic recipes")
    st.write("🌐 Web search: Substitutions & history")


# Set up the Streamlit app
st.title("🧑🍳 AI Thai Cooking Assistant")
st.write("Welcome to your personal Thai cuisine expert! Ask about recipes, techniques, and food history.")


stream = st.sidebar.checkbox("Stream")


# Initialize session state for query counter
with st.sidebar:
    counter_placeholder = st.empty()
if "counter" not in st.session_state:
    st.session_state["counter"] = 0
st.session_state["counter"] += 1
with st.sidebar:
    counter_placeholder.caption(f"Chunks received: {st.session_state['counter']}")
# counter_placeholder.write(st.session_state["counter"])


# --------------- AGENT SECTION -------------------

# Create the agent with cooking knowledge
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
    instructions=dedent("""\
        You are a passionate and knowledgeable Thai cuisine expert! 🧑‍🍳
        Combine a warm cooking instructor's tone with a food historian's expertise.

        Answer strategy:
        1. First check the recipe knowledge base for authentic information
        2. Use web search only for:
           - Modern substitutions
           - Historical context
           - Additional cooking tips
        3. Prioritize knowledge base content for recipes
        4. Clearly cite sources when using web information

        Response format:
        🌶️ Start with relevant emoji
        📖 Structure clearly:
        - Introduction/context
        - Main content (recipe/steps/explanation)
        - Pro tips & cultural insights
        - Encouraging conclusion

        For recipes include:
        📝 Ingredients with substitutions
        🔢 Numbered steps
        💡 Success tips & common mistakes

        Special features:
        - Explain Thai ingredients & alternatives
        - Share cultural traditions
        - Adapt recipes for dietary needs
        - Suggest serving pairings

        End with:
        - 'Happy cooking! ขอให้อร่อย (Enjoy your meal)!'
        - 'May your Thai cooking adventure bring joy!'
        - 'Enjoy your homemade Thai feast!'\
    """),
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipe_knowledge",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small", api_key=openai_api_key),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
)


# --------------- AGENT KNOWLEDGE LOADING -------------------

# Load knowledge base once
# if agent.knowledge and agent.knowledge.exists() == False:
if agent.knowledge:
    # with st.spinner("🧑🍳 Loading authentic Thai recipes..."):
    agent.knowledge.load()

# --------------- RPINT AGENT KNOWLEDGE FOR DEBUGGING -------------------
# methods_info = []
# for method_name in dir(agent.knowledge):
#     if method_name.startswith('__'):
#         continue
#     method = getattr(agent.knowledge, method_name)
#     if callable(method):
#         try:
#             sig = inspect.signature(method)
#             methods_info.append(f"{method_name}{sig}")
#         except:
#             methods_info.append(method_name)
# st.markdown("**Agent Knowledge Methods:**")
# st.code('\n'.join(methods_info))
# st.write(agent.knowledge.exists())


# Add a button and check if it was clicked
if st.sidebar.button("Load Knowledge"):
    if agent.knowledge:
        with st.sidebar:
            with st.spinner("🧑🍳 Loading authentic Thai recipes..."):
                agent.knowledge.load()
                st.success("Recipe database loaded!")


# User input
prompt = st.text_input("Ask your Thai cooking question (e.g., 'How to make Pad Thai?')")

# Generate and display response
if prompt:
    with st.spinner("👩🍳 Cooking up your answer..."):
        # stream = True  # Enable streaming
        if stream:
            run_response: Iterator[RunResponse] = agent.run(prompt, stream=True)
            response = ""
            text_placeholder = st.empty()
            for chunk in run_response:
                response += chunk.content
                text_placeholder.markdown(response + "▌")
                st.session_state["counter"] += 1
                # counter_placeholder.write(st.session_state["counter"])
                with st.sidebar:
                    counter_placeholder.caption(f"Chunks received: {st.session_state['counter']}")
            text_placeholder.markdown(response)
        else:
            response = agent.run(prompt, stream=False)
            st.markdown(response.content)
            st.session_state["counter"] += 1
            # counter_placeholder.write(st.session_state["counter"])
            with st.sidebar:
                counter_placeholder.caption(f"Chunks received: {st.session_state['counter']}")

        st.caption(f"🍴 Cooking questions answered: {st.session_state['counter']}")


st.caption("Note: Combines curated recipes with web research. May take 20-30 seconds for complex queries.")