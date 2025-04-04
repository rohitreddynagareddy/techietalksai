# Add these imports at the very top
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Initialize torch early
_ = nn.Linear(1,1)  # Force torch initialization

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from utils.file_processing import process_files
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
import asyncio

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add new imports at top
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
@st.cache_resource
def init_openai():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai

openai_client = init_openai()

# Add these new imports at the top
import umap
import plotly.express as px
import numpy as np

# Initialize components
@st.cache_resource
def init_chroma():
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "/data/chroma"))
    return client

@st.cache_resource
def init_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

client = init_chroma()
encoder = init_encoder()

# Create or get collection
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Streamlit UI
st.title("Document Search with ChromaDB")

# File upload section
st.header("Upload Documents")
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
docx_files = st.file_uploader("Upload Word files", type=["docx"], accept_multiple_files=True)

if st.button("Process Documents"):
    if pdf_files or docx_files:
        with st.spinner("Processing documents..."):
            # Process and split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )
            
            documents = process_files(pdf_files, docx_files, text_splitter)
            
            # Generate embeddings and upload to Chroma
            texts = [doc.page_content for doc in documents]
            embeddings = encoder.encode(texts, show_progress_bar=True)
            
            # Add to Chroma collection
            ids = [str(i) for i in range(len(texts))]
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist()
            )
        st.success(f"Processed {len(documents)} document chunks!")

# In the Vector Data Explorer section
if st.button("Generate Vector Space Visualization"):
    with st.spinner("Generating visualization..."):
        # Get all vectors from Chroma
        results = collection.get(include=["embeddings", "documents"])
        embeddings = np.array(results["embeddings"])
        documents = results["documents"]
        
        if len(embeddings) == 0:
            st.warning("No vectors found in the database!")
        else:
            # Reduce dimensionality
            reducer = umap.UMAP(random_state=42)
            reduced_vectors = reducer.fit_transform(embeddings)
            
            # Create DataFrame with all vectors
            vector_columns = [f"vec_{i}" for i in range(embeddings.shape[1])]
            df = pd.DataFrame(
                data=np.hstack((
                    reduced_vectors,
                    embeddings,
                    np.array([doc[:100] + "..." for doc in documents]).reshape(-1, 1)
                )),
                columns=["x", "y"] + vector_columns + ["text"]
            )
            
            # Create interactive plot
            fig = px.scatter(
                df,
                x="x",
                y="y",
                hover_data=["text"],
                title="Document Embeddings Visualization (UMAP)",
                labels={"x": "UMAP 1", "y": "UMAP 2"}
            )
            
            # Add cluster information
            fig.update_traces(
                marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
                selector=dict(mode='markers')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            st.subheader("Vector Space Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Vectors", len(embeddings))
            col2.metric("Dimensions (Original)", embeddings.shape[1])
            col3.metric("Dimensions (Visualized)", 2)
            
            # Add raw data preview
            with st.expander("Show Raw Vector Data"):
                # Format columns for better display
                pd.options.display.float_format = '{:.4f}'.format
                st.dataframe(
                    df.drop(columns=["x", "y"]).set_index("text"),
                    height=400,
                    column_config={
                        "text": "Document Snippet",
                        **{col: {"header": col.replace("vec_", "Dimension ")} 
                           for col in vector_columns}
                    }
                )
# Function to highlight only the first row and first column
def highlight_top_left(x):
    styles = pd.DataFrame('', index=x.index, columns=x.columns)
    styles.iloc[0, 0] = 'background-color: yellow'  # You can change the color
    return styles

# Search interface
st.header("Semantic Search")
search_query = st.text_input("Enter your search query:")
if search_query:
    with st.spinner("Searching..."):
        # Encode query
        query_embedding = encoder.encode(search_query).tolist()
        
        # Search Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        # Display results
        output = []
        for i, (text, score) in enumerate(zip(results['documents'][0], results['distances'][0])):
            output.append({
                "Score": 1 - score,  # Convert cosine distance to similarity
                "Text": text[:200] + "..."
            })
        

        df = pd.DataFrame(output)
        # Apply the style and show in Streamlit
        st.dataframe(df.style.apply(highlight_top_left, axis=None))


# Add new RAG section after semantic search
st.header("RAG-based LLM Query")
rag_query = st.text_input("Ask a question based on the documents:")
if rag_query:
    with st.spinner("Analyzing documents..."):
        # First perform semantic search
        query_embedding = encoder.encode(rag_query).tolist()
        hits = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        context = "\n\n".join([doc[:1000] for doc in hits['documents'][0]])
        
        if not context:
            st.error("No relevant documents found to answer this question")
        else:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"Answer using this context: {context}"},
                        {"role": "user", "content": rag_query}
                    ],
                    temperature=0.3
                )
                
                st.subheader("AI Answer")
                st.markdown(f"**{response.choices[0].message.content}**")
                
                with st.expander("View context documents used"):
                    for i, doc in enumerate(hits['documents'][0]):
                        st.markdown(f"**Document {i+1}:** {doc[:500]}...")
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")