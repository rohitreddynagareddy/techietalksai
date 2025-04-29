from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_openai import OpenAI  # Updated import
from langchain_openai import ChatOpenAI
import pandas as pd
import os

# Initialize OpenAI
# llm = OpenAI(model="gpt-4.1-mini", openai_api_key=os.getenv('OPENAI_API_KEY'))
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))

# Create sample DataFrame
data = {
    'country': ['United States', 'China', 'Japan', 'Germany', 'India'],
    'gdp': [21400000, 14100000, 5070000, 3840000, 2970000],
    'happiness_index': [7.0, 5.5, 6.1, 7.0, 4.3]
}
df = pd.DataFrame(data)

# Create agent
# agent = create_pandas_dataframe_agent(llm, df, verbose=True)
# Previous imports remain the same
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True  # Add this security override
)

# Run agent
response = agent.invoke("Which country has the highest GDP?")
print(response)



from langchain_experimental.agents import create_csv_agent

agent_executor = create_csv_agent(
    llm,
    "titanic.csv",
    agent_type="openai-tools",
    verbose=True,
    allow_dangerous_code=True  # Add this security override
)
response = agent_executor.invoke("Number passengers survived?")
print(response)

