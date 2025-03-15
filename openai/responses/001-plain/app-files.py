import os
import streamlit as st
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- TITLE & HEADER -------------------
st.title("📰 OpenAI File Search RAG")
# st.title("📰 Daily Positive News Finder")
st.subheader("OpenAI Responses API 2025 - Agentic Assistant + RAG + Web Search")



# --------------- SETUP ---------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
dir_pdfs = "uploaded_pdfs"
os.makedirs(dir_pdfs, exist_ok=True)

def get_vector_store_id(store_name: str) -> str:
    # Find vector store ID by name
    # store_name = "test_docs"  # Change this to your desired vector store name
    vector_store_id = None

    # List all vector stores
    vector_stores = client.vector_stores.list()

    # print(json.dumps(vector_stores.model_dump(), indent=4))

    for vs in vector_stores:
        if vs.name == store_name:
            # vector_store_id = vs.id
            return vs.id

            # if vector_store_id:
            #     # print(f"Vector Store ID: {vector_store_id}")
            #     vector_store = client.vector_stores.retrieve(
            #      vector_store_id=vector_store_id
            #     )
            #     print(json.dumps(vector_store.model_dump(), indent=4))

            # else:
            #     print("Vector store not found.")
    return None

def get_vector_store_files(vector_store_id: str) -> list:
    files = []
    if not vector_store_id:
        return files
    vector_store_files = client.vector_stores.files.list(
        vector_store_id=vector_store_id
    )
    for file in vector_store_files.data:
        file_id = file.id
        # print(f"File: {file_id}")
        file_info=client.files.retrieve(file_id)
        # print(json.dumps(file_info.model_dump(), indent=4))
        # {
        #     "id": "file-2RkkBxRQjMKZBTAaKBVLn3",
        #     "bytes": 653471,
        #     "created_at": 1741847224,
        #     "filename": "ThaiRecipes.pdf",
        #     "object": "file",
        #     "purpose": "assistants",
        #     "status": "processed",
        #     "expires_at": null,
        #     "status_details": null
        # }
        file_name = file_info.filename
        # print(f"Filename: {file_name}")
        # Filename: ThaiRecipes.pdf
        files.append(file_name)
    return files

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
        # print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

store_name = "Thai Recipes"
if "vector_store_id" not in st.session_state:
    vsid = get_vector_store_id(store_name)
    if vsid:
        st.session_state.vector_store_id = vsid
        logger.info(f"VECTOR STORE ID SAVED IN SESSION: {vsid}")
    else:
        logger.info(f"VECTOR STORE NOT FOUND IN OPENAI")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:

    st.header("📚 RAG Vector Store")
    
    # Vector store creation
    # vs_col1, vs_col2 = st.columns([3,1])
    # with vs_col1:
    vector_store_name = st.text_input("Vector Store name", store_name)
    # with vs_col2:
    if st.button("Create Store"):
        # if 'vector_store_id' in st.session_state:
        #     del st.session_state.vector_store_id
        if 'vector_store_id' not in st.session_state:
            result = create_vector_store(vector_store_name)
            if result:
                st.session_state.vector_store_id = result["id"]
                st.success(f"Created: {result['id']}")

    # PDF upload section
    st.subheader("Upload PDFs")
    files = get_vector_store_files(st.session_state.vector_store_id)
    with st.expander("🔽 Vector Store files"):
       st.markdown("\n".join(f"- {item}" for item in files))


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

    # st.subheader("Example Queries")
    # st.markdown("""
    # - Positive tech breakthroughs today
    # - Good environmental news this week
    # - Uplifting science discoveries
    # - Inspiring community stories
    # - Recent humanitarian achievements
    # """)
    # st.markdown("---")
    
    # Stats
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    st.caption(f"News searches made: {st.session_state.query_count}")


# --------------- MAIN INTERFACE -------------------
query = st.text_input("Ask me:", 
                     placeholder="e.g., 'Type your queries here'")

if query:
    st.session_state.query_count += 1
    
    # Initialize client
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        st.error("OpenAI API key not configured properly!")
        st.stop()
    
    with st.spinner("🔍 Generating response..."):
        try:
            tools = []
            tools.append({"type": "web_search_preview"})

            # st.session_state.vector_store_id = "vs_67d27a7e9d3881918980bc9916f64098"

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
            
            with st.expander("🌐 Agentic Process", expanded=True):
                st.markdown("""
                **Search Process:**
                1. Searching the local knowledge base
                2. Synthesizing key information
                """)
                
                if response.output_text:
                    st.success("✅ Found results!")
                else:
                    st.warning("⚠️ No results found")
            
            st.markdown("---")
            query=query.title()
            # st.subheader(f"✨ {query}")
            st.markdown(response.output_text)
            
            # Download button
            st.download_button(
                label="Download Report",
                data=response.output_text,
                file_name="report.md",
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