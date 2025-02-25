from agno.agent import Agent
from agno.document.chunking.recursive import RecursiveChunking
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.models.openai import OpenAIChat

# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db_url = "postgresql+psycopg://ai:ai@pgvector:5432/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes_agentic_chunking", db_url=db_url),
    chunking_strategy=RecursiveChunking(),
)
knowledge_base.load(recreate=True)  # Comment out after first run

agent = Agent(
	model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    search_knowledge=True,
)

agent.print_response("How to make Thai curry?", markdown=True)
