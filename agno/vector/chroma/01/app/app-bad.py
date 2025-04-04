# app/app.py
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Initialize torch early to prevent Streamlit conflicts
_ = nn.Linear(1,1)  # Force torch initialization

import streamlit as st
import chromadb
import pandas as pd
import numpy as np
import umap
import plotly.express as px
import sys
import asyncio
import openai
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.file_processing import process_files

# Windows event loop fix
if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()

# Initialize components
@st.cache_resource
def init_chroma():
    return chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "/data/chroma"))

@st.cache_resource
def init_models():
    return {
        "sentence-transformers": SentenceTransformer('all-MiniLM-L6-v2'),
        "openai": openai
    }

client = init_chroma()
models = init_models()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sidebar controls
st.sidebar.header("Settings")
embedding_choice = st.sidebar.selectbox(
    "Embedding Model",
    ("sentence-transformers", "openai"),
    index=0,
    help="Choose between local SentenceTransformers or OpenAI embeddings"
)

# Update the recreate button handler
if st.sidebar.button("♻️ Recreate Vector Store"):
    try:
        try:
            client.delete_collection("documents")
        except chromadb.errors.ChromaError:
            pass
            
        # Create fresh collection
        collection = get_collection()
        st.sidebar.success("Vector store recreated! Please reprocess documents.")
        
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
# Update the exception imports
import chromadb.errors

# Update the get_collection function
def get_collection():
    dimensions = 384 if embedding_choice == "sentence-transformers" else 1536
    
    try:
        existing_collection = client.get_collection("documents")
        current_model = existing_collection.metadata.get("embedding_model", "")
        
        # Check if existing collection matches current settings
        if (current_model != embedding_choice or 
            int(existing_collection.metadata.get("dimension", 0)) != dimensions):
            
            client.delete_collection("documents")
            raise chromadb.errors.ChromaError("Metadata mismatch")
            
        return existing_collection
        
    except chromadb.errors.ChromaError:
        # Create new collection with current settings
        return client.create_collection(
            name="documents",
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_choice,
                "dimension": dimensions
            }
        )

collection = get_collection()

# Main UI
st.title("📄 Document Search with ChromaDB")

# File upload section
st.header("Upload Documents")
pdf_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
docx_files = st.file_uploader("Word files", type=["docx"], accept_multiple_files=True)

if st.button("Process Documents"):
    if pdf_files or docx_files:
        with st.spinner("Processing..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )
            
            documents = process_files(pdf_files, docx_files, text_splitter)
            texts = [doc.page_content for doc in documents]

            # Generate embeddings
            if embedding_choice == "sentence-transformers":
                encoder = models["sentence-transformers"]
                embeddings = encoder.encode(texts, show_progress_bar=True).tolist()
            else:
                response = openai.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                embeddings = [d.embedding for d in response.data]

            # Add to Chroma
            ids = [str(i) for i in range(len(texts))]
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings
            )
            
            st.success(f"✅ Processed {len(texts)} chunks with {embedding_choice}!")

# Visualization section
st.header("Vector Space Explorer")
if st.button("Generate Visualization"):
    with st.spinner("Creating visualization..."):
        results = collection.get(include=["embeddings", "documents"])
        if not results["embeddings"]:
            st.warning("No vectors found!")
            st.stop()
            
        embeddings = np.array(results["embeddings"])
        reducer = umap.UMAP(random_state=42)
        reduced = reducer.fit_transform(embeddings)
        
        df = pd.DataFrame({
            "x": reduced[:,0], 
            "y": reduced[:,1],
            "text": [doc[:100]+"..." for doc in results["documents"]]
        })
        
        fig = px.scatter(
            df, x="x", y="y", 
            hover_data=["text"],
            title="Document Embeddings Projection",
            labels={"x": "UMAP 1", "y": "UMAP 2"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Statistics")
        cols = st.columns(3)
        cols[0].metric("Total Vectors", len(embeddings))
        cols[1].metric("Dimensions", embeddings.shape[1])
        cols[2].metric("Visualized", 2)

# Search interface
st.header("🔍 Semantic Search")
search_query = st.text_input("Search query:")
if search_query:
    with st.spinner("Searching..."):
        # Encode query
        if embedding_choice == "sentence-transformers":
            encoder = models["sentence-transformers"]
            query_embedding = encoder.encode(search_query).tolist()
        else:
            response = openai.embeddings.create(
                input=[search_query],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        # Display results
        df = pd.DataFrame([{
            "Score": 1 - score,
            "Text": text[:200] + "..."
        } for text, score in zip(results['documents'][0], results['distances'][0])])
        
        st.dataframe(
            df.style.format({"Score": "{:.3f}"}),
            height=300
        )

# RAG Section
st.header("💬 RAG Chat")
rag_query = st.text_input("Ask a question:")
if rag_query:
    with st.spinner("Analyzing..."):
        # Semantic search first
        if embedding_choice == "sentence-transformers":
            encoder = models["sentence-transformers"]
            query_embedding = encoder.encode(rag_query).tolist()
        else:
            response = openai.embeddings.create(
                input=[rag_query],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
        
        hits = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        context = "\n\n".join([doc[:1000] for doc in hits['documents'][0]])
        if not context:
            st.error("No relevant documents found!")
            st.stop()
            
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Answer using this context:\n{context}"},
                    {"role": "user", "content": rag_query}
                ],
                temperature=0.3
            )
            
            st.markdown(f"**🤖 Answer:** {response.choices[0].message.content}")
            
            with st.expander("View sources"):
                for i, doc in enumerate(hits['documents'][0]):
                    st.markdown(f"**Source {i+1}:** {doc[:500]}...")
                    st.divider()
                    
        except Exception as e:
            st.error(f"API Error: {str(e)}")