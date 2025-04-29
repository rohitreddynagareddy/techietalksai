from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
import os

def csv_agent_executor(file_path, question):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
    agent = create_csv_agent(
        llm,
        file_path,
        agent_type="openai-tools",
        verbose=True,
        allow_dangerous_code=True
    )
    return agent.invoke(question)
