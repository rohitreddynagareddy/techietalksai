from langchain.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {input}
""")

class RAGAgent:
    def __init__(self):
        # ... existing initialization ...
        
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "prefix": "Always use the VectorSearch tool when answering questions. "
                          "If you don't know the answer, say so."
            }
        )