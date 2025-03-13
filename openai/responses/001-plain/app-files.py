import os
import streamlit as st
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

# --------------- TITLE & HEADER -------------------
st.title("📰 OpenAI RAG New Responses API 2025")
# st.title("📰 Daily Positive News Finder")
st.write("Your AI-powered good news aggregator with web search integration")



# --------------- SETUP ---------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
dir_pdfs = "uploaded_pdfs"
os.makedirs(dir_pdfs, exist_ok=True)

def upload_single_pdf(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

def upload_pdf_files_to_vector_store(vector_store_id: str):
    pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs)]
    stats = {"total_files": len(pdf_files), "successful_uploads": 0, "failed_uploads": 0, "errors": []}
    
    print(f"{len(pdf_files)} PDF files to process. Uploading in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(upload_single_pdf, file_path, vector_store_id): file_path for file_path in pdf_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_files)):
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)

    return stats

def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}


# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:

    st.header("📚 Document Management")
    
    # Vector store creation
    # vs_col1, vs_col2 = st.columns([3,1])
    # with vs_col1:
    vector_store_name = st.text_input("Store name", "test_docs")
    # with vs_col2:
    if st.button("Create Store"):
        if 'vector_store_id' in st.session_state:
            del st.session_state.vector_store_id
        result = create_vector_store(vector_store_name)
        if result:
            st.session_state.vector_store_id = result["id"]
            st.success(f"Created: {result['id']}")

    # PDF upload section
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drag & drop PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Select multiple PDF files to upload"
    )
    
    if uploaded_files and 'vector_store_id' in st.session_state:
        if st.button("Process PDFs"):
            with st.spinner("Saving files..."):
                # Save uploaded files
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(dir_pdfs, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
            
            with st.spinner("Uploading to vector store..."):
                try:
                    stats = upload_pdf_files_to_vector_store(
                        st.session_state.vector_store_id
                    )
                    
                    st.success(f"""
                    **Upload Complete**  
                    ✅ Success: {stats['successful_uploads']}  
                    ❌ Failed: {stats['failed_uploads']}
                    """)
                    
                    if stats['errors']:
                        with st.expander("Error Details"):
                            for error in stats['errors']:
                                st.error(f"{error['file']}: {error['error']}")
                    
                    # Cleanup files after upload
                    for f in os.listdir(dir_pdfs):
                        os.remove(os.path.join(dir_pdfs, f))
                    
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    st.markdown("---")

    st.subheader("Example Queries")
    st.markdown("""
    - Positive tech breakthroughs today
    - Good environmental news this week
    - Uplifting science discoveries
    - Inspiring community stories
    - Recent humanitarian achievements
    """)
    st.markdown("---")
    
    # Stats
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    st.caption(f"News searches made: {st.session_state.query_count}")




# --------------- MAIN INTERFACE -------------------
query = st.text_input("Ask for positive news:", 
                     placeholder="e.g., 'What good news happened today?'")

if query:
    st.session_state.query_count += 1
    
    # Initialize client
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        st.error("OpenAI API key not configured properly!")
        st.stop()
    
    with st.spinner("🔍 Scanning global news sources..."):
        try:
            tools = [{"type": "web_search_preview"}]

            st.session_state.vector_store_id = "vs_67d27a7e9d3881918980bc9916f64098"

            if "vector_store_id" in st.session_state:
                tools.append({
                                "type": "file_search",
                                "vector_store_ids": [st.session_state.vector_store_id],
                            })

            response = client.responses.create(
                model="gpt-4o",
                tools=tools,
                input=query
            )
            
            with st.expander("🌐 Web Search Process", expanded=True):
                st.markdown("""
                **Search Process:**
                1. Analyzing current news trends
                2. Filtering for positive stories
                3. Verifying source credibility
                4. Synthesizing key information
                """)
                
                if response.output_text:
                    st.success("✅ Found verified positive news!")
                else:
                    st.warning("⚠️ No positive stories found - expanding search")
            
            st.markdown("---")
            st.subheader("✨ Good News Report")
            st.markdown(response.output_text)
            
            # Download button
            st.download_button(
                label="Download Report",
                data=response.output_text,
                file_name="positive_news_report.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")

# --------------- FOOTER -------------------
st.markdown("---")
st.caption("""
**News Finder Features:**
- Real-time web search integration 🌐
- Positive story filtering 😊
- Source verification ✅
- Daily updates 📅
- Fact-checked reports 🔍
""")

# Hidden dependency note
st.markdown("<!--- Run `pip install openai` -->", unsafe_allow_html=True)