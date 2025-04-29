import streamlit as st
import os
from agents.rag_agent import RAGAgent
from agents.booking_agent import BookingAgent
from tools.chroma_tool import ChromaTool  # Add this import

st.title("LangGraph Chat Application")

# Create data directories if they don't exist
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)

# Document upload section
with st.expander("Document Management (RAG Agent Only)"):
    uploaded_files = st.file_uploader(
        "Upload documents for RAG",
        type=["txt", "pdf", "docx", "md"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save files to documents directory
            save_path = os.path.join("data/documents", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved file: {uploaded_file.name} to {save_path}")

# Add this after the file upload section
if uploaded_files and st.button("Process Documents"):
    rag_tool = ChromaTool()
    result = rag_tool.ingest_documents()
    st.write(result)

# Chat interface
agent_type = st.selectbox("Choose Agent", ["RAG Agent", "Booking Agent"])
user_input = st.text_input("Enter your message:")

if st.button("Send"):
    if agent_type == "RAG Agent":
        agent = RAGAgent()
    else:
        agent = BookingAgent()
    response = agent.run(user_input)
    st.write(response)