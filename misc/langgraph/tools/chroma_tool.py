from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # Add this import
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChromaTool:
    def __init__(self):
        self.embedding = OpenAIEmbeddings()  # Now properly imported
        self.persist_dir = "/app/data/chroma_db"
        
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )
    
    def ingest_documents(self):
        """Process documents from the upload directory"""
        try:
            # Load documents
            loader = DirectoryLoader("/app/data/documents", glob="**/*.*")
            documents = loader.load()
            
            if not documents:
                return "No documents found in /app/data/documents"
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(splits)
            self.vector_store.persist()
            return f"Processed {len(splits)} document chunks"
            
        except Exception as e:
            return f"Ingestion failed: {str(e)}"

    def query(self, query_text: str) -> list:
        return self.vector_store.similarity_search(query_text, k=3)