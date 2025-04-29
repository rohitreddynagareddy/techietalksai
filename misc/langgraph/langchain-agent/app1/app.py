from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import os

def pandas_agent_executor(df, question):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )
    return agent.invoke(question)
