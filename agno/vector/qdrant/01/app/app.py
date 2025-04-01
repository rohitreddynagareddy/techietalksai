import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from utils.file_processing import process_files
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
# Replace the upload_points section with:
from qdrant_client.models import Batch


# Initialize components
@st.cache_resource
def init_qdrant():
    return QdrantClient(host="qdrant", port=6333)

@st.cache_resource
def init_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

client = init_qdrant()
encoder = init_encoder()

# Create collection if not exists
try:
    client.get_collection("documents")
except:
    client.create_collection(
        collection_name="documents",
        vectors_config={
            "size": encoder.get_sentence_embedding_dimension(),
            "distance": "Cosine"
        }
    )

# Add this after collection creation
client.update_collection(
    collection_name="documents",
    optimizer_config=models.OptimizersConfigDiff(
        indexing_threshold=0,
        memmap_threshold=2000,
    )
)



# Streamlit UI
st.title("Document Search with Qdrant")

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
            
            # Generate embeddings and upload to Qdrant
            # texts = [doc.page_content for doc in documents]
            # embeddings = encoder.encode(texts, show_progress_bar=True)
            
            # client.upload_points(
            #     collection_name="documents",
            #     points=[
            #         {
            #             "id": idx,
            #             "vector": vector.tolist(),
            #             "payload": {"text": text}
            #         }
            #         for idx, (text, vector) in enumerate(zip(texts, embeddings))
            #     ]
            # )
            # Generate embeddings and upload to Qdrant
            texts = [doc.page_content for doc in documents]
            embeddings = encoder.encode(texts, show_progress_bar=True)

            # client.upload_points(
            #     collection_name="documents",
            #     points=[
            #         PointStruct(
            #             id=idx,
            #             vector=vector.tolist(),
            #             payload={"text": text}
            #         )
            #         for idx, (text, vector) in enumerate(zip(texts, embeddings))
            #     ]
            # )

            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                
                client.upsert(
                    collection_name="documents",
                    points=Batch(
                        ids=list(range(i, i+len(batch_texts))),
                        vectors=[vec.tolist() for vec in batch_embeddings],
                        payloads=[{"text": text} for text in batch_texts]
                    )
                )

        st.success(f"Processed {len(documents)} document chunks!")

# Search interface
st.header("Semantic Search")
search_query = st.text_input("Enter your search query:")
if search_query:
    with st.spinner("Searching..."):
        # Encode query
        query_vector = encoder.encode(search_query).tolist()
        
        # Search Qdrant
        hits = client.search(
            collection_name="documents",
            query_vector=query_vector,
            limit=5
        )
        
        # Display results
        results = []
        for hit in hits:
            results.append({
                "Score": hit.score,
                "Text": hit.payload["text"][:200] + "..."
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df.style.highlight_max(axis=0))

# Dashboard link
st.markdown("### Qdrant Dashboard")
st.markdown("Access the Qdrant dashboard at [http://localhost:6334/dashboard](http://localhost:6334/dashboard)")
