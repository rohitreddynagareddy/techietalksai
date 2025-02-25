from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.models.openai import OpenAIChat
from agno.vectordb.pineconedb import PineconeDb
import nltk  # type: ignore

nltk.download("punkt")
nltk.download("punkt_tab")

import os
api_key = os.getenv("PINECONE_API_KEY")
index_name = "thai-recipe-hybrid-search"

vector_db = PineconeDb(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    api_key=api_key,
    use_hybrid_search=True,
    hybrid_alpha=0.5,
)

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=knowledge_base,
    search_knowledge=True,
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True,
)
agent.print_response(
    "How do I make chicken and galangal in coconut milk soup", stream=True
)
agent.print_response("What was my last question?", stream=True)