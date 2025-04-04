import sys
import chromadb
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

def rag_query(query: str, context: str):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Answer using this context: {context}"},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

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

    # Display semantic results
    console.print("\n")
    table = Table(title=f"📄 Semantic Search Results for '{query}'")
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

    # RAG Query Section
    context = "\n\n".join([doc[:1000] for doc in results['documents'][0]])
    if context:
        console.print("\n🧠 Generating AI-powered answer...\n")
        answer = rag_query(query, context)
        console.print(f"[bold yellow]🤖 AI Answer:[/bold yellow] [white]{answer}[/white]")
        
        # Show context sources
        console.print("\n[bold]🔍 Context Sources:[/bold]")
        for i, doc in enumerate(results['documents'][0]):
            console.print(f"[cyan]{i+1}.[/cyan] {doc[:200]}...\n")
    else:
        console.print("[red]No documents found to generate answer![/red]")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Tom Yum Soup"
    main(query)