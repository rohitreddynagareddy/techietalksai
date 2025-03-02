from agno.agent import Agent
from agno.document.chunking.fixed import FixedSizeChunking
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.models.openai import OpenAIChat
from agno.models.ollama import OllamaTools

# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db_url = "postgresql+psycopg://ai:ai@pgvector:5432/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes_fixed_chunking", db_url=db_url),
    chunking_strategy=FixedSizeChunking(),
)
knowledge_base.load(recreate=False)  # Comment out after first run

agent = Agent(
	# model=OpenAIChat(id="gpt-4o-mini"),
	model=OllamaTools(
			id="deepseek-r1:latest",
			# id="phi4-mini",
			host="http://host.docker.internal:11434"
			),
    knowledge=knowledge_base,
    search_knowledge=True,
)

agent.print_response("How to make Thai curry?", markdown=True)
