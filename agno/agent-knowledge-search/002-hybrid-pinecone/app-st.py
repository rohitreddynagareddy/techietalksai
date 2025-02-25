import os
import streamlit as st
from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pineconedb import PineconeDb
from agno.models.openai import OpenAIChat
import nltk  # type: ignore
import os
api_key = os.getenv("PINECONE_API_KEY")

# --------------- TITLE & HEADER -------------------
st.title("RAG - SearchType: Hybrid!")
st.write("Perform vector similarity and full-text search.")
st.title("🍜 Thai Recipe AI Agent")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Example Questions")
    st.markdown("""
    - How to make green curry?
    - What's in tom yum soup?
    - Vegetarian pad thai recipe
    - How to prepare sticky mango rice?
    - Traditional Thai dessert ideas
    """)
    st.markdown("---")
    
    # Query counter
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    st.caption(f"Recipes shared: {st.session_state.query_count}")
    
    # Database info
    st.markdown("---")
    st.markdown("**Database Status**")
    if 'knowledge_base' in st.session_state:
        st.success("Recipes loaded")
    else:
        st.warning("Loading recipes...")

# --------------- INITIALIZATION -------------------
if 'knowledge_base' not in st.session_state:
    with st.spinner("📚 Loading Thai recipe database..."):
        try:

            nltk.download("punkt")
            nltk.download("punkt_tab")

            # api_key = os.getenv("PINECONE_API_KEY")
            index_name = "thai-recipe-hybrid-search"

            vector_db = PineconeDb(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
                api_key=api_key,
                use_hybrid_search=True,
                hybrid_alpha=0.5,
            )
            
            st.session_state.knowledge_base = PDFUrlKnowledgeBase(
                urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
                vector_db=vector_db,
            )
            # st.session_state.knowledge_base.load(recreate=True)
            
            st.session_state.agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini"),
                knowledge=st.session_state.knowledge_base,
                search_knowledge=True,
                read_chat_history=True,
                show_tool_calls=True,
                markdown=True,
            )
            st.rerun()
        except Exception as e:
            st.error(f"Error initializing database: {str(e)}")
            st.stop()

# --------------- MAIN INTERFACE -------------------
question = st.text_input("Ask about Thai cooking:", 
                        placeholder="e.g., 'How to make authentic pad thai?'")

if question:
    st.session_state.query_count += 1
    
    with st.spinner("🔍 Searching recipes..."):
        try:
            response = st.session_state.agent.run(question)
            
            st.markdown("---")
            st.subheader("🥘 Recipe Guidance")
            st.markdown(response.content)
            
            with st.expander("📚 Source References", expanded=False):
                st.markdown("""
                Recipes sourced from:
                - [Thai Recipes PDF](https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf)
                """)
                
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

# --------------- FOOTER -------------------
st.markdown("---")
st.caption("""
**Recipe Assistant Features:**
- Authentic Thai recipes 🇹🇭
- Step-by-step instructions 👩🍳
- Ingredient breakdown 🧄
- Cooking tips & tricks 🔪
- Dietary adaptations 🌱
""")

# Hidden dependency note
st.markdown("<!--- Run `pip install agno pgvector` -->", unsafe_allow_html=True)