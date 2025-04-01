import sys
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
from rich.console import Console
from rich.table import Table
import os

console = Console()

def main(query: str = "Tom Yum Soup"):
    # Initialize components
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "/data/chroma"))
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    collection = client.get_collection("documents")

    # Encode query
    query_embedding = encoder.encode(query).tolist()

    # Search Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # Display results
    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Rank", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Text Snippet", style="green")

    for i, (text, score) in enumerate(zip(results['documents'][0], results['distances'][0])):
        table.add_row(
            str(i+1), 
            f"{1 - score:.3f}", 
            text[:100] + "..."
        )

    console.print(table)

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Tom Yum Soup"
    main(query)
