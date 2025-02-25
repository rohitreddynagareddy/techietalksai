from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb
from agno.models.openai import OpenAIChat
from agno.vectordb.search import SearchType

# LanceDB Vector DB
vector_db = LanceDb(
    table_name="recipes",
    uri="tmp/lancedb",
    search_type=SearchType.hybrid,
)
# Knowledge Base
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    user_id="Sree",
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    debug_mode=True,
)
agent.print_response(
    "How do I make chicken and galangal in coconut milk soup", stream=True
)
agent.print_response("What was my last question?", stream=True)