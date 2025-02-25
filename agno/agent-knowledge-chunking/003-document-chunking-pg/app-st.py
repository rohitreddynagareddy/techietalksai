import os
import streamlit as st
from agno.agent import Agent
from agno.document.chunking.document import DocumentChunking
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.models.openai import OpenAIChat

# --------------- TITLE & HEADER -------------------
st.title("RAG - DocumentChunking!")
st.write("A chunking strategy that splits text based on document structure like paragraphs and sections")
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
            db_url = "postgresql+psycopg://ai:ai@pgvector:5432/ai"
            
            st.session_state.knowledge_base = PDFUrlKnowledgeBase(
                urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
                vector_db=PgVector(table_name="recipes_agentic_chunking", db_url=db_url),
                chunking_strategy=DocumentChunking(),
            )
            st.session_state.knowledge_base.load(recreate=False)
            
            st.session_state.agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini"),
                knowledge=st.session_state.knowledge_base,
                search_knowledge=True,
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